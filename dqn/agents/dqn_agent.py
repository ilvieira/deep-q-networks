import pickle
import os
import random as rnd
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import pandas as pd
from dqn.agents.networks.dqn import DQNetwork
from dqn.agents.agent import Agent

from dqn.memory.dqn_replay_memory_atari import DQNReplayMemoryAtari
from dqn.environments.atari_dqn_env import AtariDQNEnv
from dqn.policies.train_eval_policy import TrainEvalPolicy
from dqn.torch_extensions import clip_mse3
from dqn.policies.random_policy import RandomPolicy


class DQNAgent(Agent):
    """ Class that simulates the game and trains the DQN """

    def __init__(self, env, replay, n_actions, net_type, net_parameters,
                 minibatch_size=32,
                 optimizer=torch.optim.RMSprop,
                 C=10_000,
                 update_frequency=1,
                 gamma=0.99,
                 loss=clip_mse3,
                 policy=TrainEvalPolicy(),
                 populate_policy=None,
                 seed=0,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 optimizer_parameters=None):  # this parameter is only needed for the LinearDQN

        super().__init__(env, n_actions)
        self.device = torch.device(device)
        self.set_seed(seed)

        self.n_frames = 0
        self.update_frequency = update_frequency
        self.C = C

        # as of now there are only two types of nets supported: LinearDQN and DQNetwork. This needs to be updated if
        # more nets are implemented
        self.net_parameters = net_parameters
        # initialize Q, Q_target as a copy of Q ant the gradient descent algorithm (optimizer)
        self.Q = net_type(**self.net_parameters).to(self.device)
        self.update_target()
        self.setup_optimizer(optimizer, optimizer_parameters)

        # other parameters
        self.gamma = gamma
        self.minibatch_size = minibatch_size

        # initialize the replay memory
        self.replay_memory = replay
        self.policy = policy
        self.populate_policy = populate_policy if populate_policy is not None else RandomPolicy(self.n_actions)
        self.loss = loss

        # other auxiliary attributes:
        self._points_per_episode = []
        self._frames_per_episode = []

    # ================================================================================================================
    # Setup Methods
    # ================================================================================================================
    def set_seed(self, seed: int):
        self.seed = seed
        torch.manual_seed(seed)
        rnd.seed(seed)
        np.random.seed(seed)

    def setup_optimizer(self, optimizer, optimizer_parameters):
        self.optimizer_parameters = optimizer_parameters if optimizer_parameters is not None \
            else {"lr": 0.00025, "alpha": 0.95, "eps": 0.01}
        self.optimizer = optimizer(self.Q.parameters(), **self.optimizer_parameters)

    # ================================================================================================================
    # Agent Methods
    # ================================================================================================================

    def action(self, observation: np.ndarray):
        """Chooses an action given an observation"""
        phi = torch.tensor(self.expand_obs(observation)).float()

        # phi is added to the gpu so that Q can make a prediction on it
        with torch.no_grad():
            q_vals = self.Q(phi.to(self.device))
            action = self.policy.choose_action(q_vals.detach().cpu())
            return action

    def eval(self):
        super().eval()
        self.policy.eval()

    def train(self):
        super().train()
        self.policy.train()

    # ================================================================================================================
    # DQN Specific Methods
    # ================================================================================================================

    def populate_replay_memory(self, n_samples, verbose=True):
        """ Adds the first transitions to the replay memory by playing the game with random actions"""
        if verbose:
            print("Populating the replay memory...")
        start = time.time()

        transitions_added = 0
        while transitions_added < n_samples:
            # restart the environment and get the first observation
            prev_phi = self.expand_obs(self.env.reset())

            # done becomes True whenever a terminal state is reached
            done = False

            while (not done) and transitions_added < n_samples:
                at = self.populate_policy.choose_action()
                phi, rt, done, _ = self.env.step(at)
                phi = self.expand_obs(phi)
                transition = (prev_phi, at, rt, phi, done)
                self.replay_memory.append(transition)
                transitions_added += 1
                prev_phi = phi

        if verbose:
            print(f"Done - {time.time() - start}s")

    def learn(self, save_dir, save_replay=True, save_policy=True, verbose=True, max_steps=50_000_000, max_time=604_800,
              max_episodes=1_000_000_000, feedback_after_episodes=5, save_after_steps=1_000_000):
        """The algorithm as described in 'Human-level control through deep reinforcement learning'"""
        if verbose:
            print("Beginning the training stage...")
        start = time.time()
        save_params = {"save_dir": save_dir, "save_replay": save_replay, "save_policy": save_policy,
                       "feedback_after_episodes": feedback_after_episodes}
        self.save(**save_params)

        self.train()
        total_reward = 0
        max_reward = float('-inf')

        for episode in range(max_episodes):
            ep_reward = 0
            ep_frames = 0

            if verbose:
                if episode % feedback_after_episodes == 0 and not episode == 0:
                    print(f"... episode {episode} ... - {time.time() - start}")
                    print(f"  average reward: {total_reward / feedback_after_episodes}")
                    print(f"  maximum reward: {max_reward}")
                    print(f"  total frames: {self.n_frames}")
                    print(f"  transitions stored: {len(self.replay_memory)}")
                    total_reward = 0
                    max_reward = float('-inf')

            observation = self.env.reset()
            done = False

            while not done:
                observation, rt, done = self.play_and_store_transition(observation)
                self.n_frames += 1
                ep_frames += 1
                total_reward += rt
                ep_reward += rt

                # Backpropagation at each update_frequency frames
                if self.n_frames % self.update_frequency == 0:
                    self.optimize_model()

                # Update target network at each C frames
                if self.n_frames % self.C == 0:
                    self.update_target()

                # PERSISTENCE: Save after each save_after_steps_frame
                if self.n_frames > 0 and self.n_frames % save_after_steps == 0:
                    self.save(**save_params)
                    if verbose:
                        print(f"Agent saved ({self.n_frames} steps)")

                # STOP if the maximum number of steps or time are reached
                if self.n_frames > max_steps or time.time() - start > max_time:
                    if verbose:
                        print("Done")
                    return

            if ep_reward > max_reward:
                max_reward = ep_reward
            self._points_per_episode.append(ep_reward)
            self._frames_per_episode.append(ep_frames)

        if verbose:
            print("Done")

    def update_net(self, r, not_done, next_phi):
        with torch.no_grad():
            y = (r + self.gamma * not_done * self.Q_target(next_phi)
                 .max(axis=1, keepdim=True).values.view(self.minibatch_size))
        return y

    def optimize_model(self):
        r, prev_phi, next_phi, not_done, actions = self.replay_memory.sample(self.minibatch_size)
        y = self.update_net(r, not_done, next_phi)

        q_vals = torch.zeros(self.minibatch_size).to(self.device)
        self.optimizer.zero_grad()
        q_phi = self.Q(prev_phi)

        for i in range(self.minibatch_size):
            q_vals[i] = q_phi[i, actions[i]]

        loss = self.loss(q_vals, y.float())
        loss.backward()
        self.optimizer.step()

    def play_and_store_transition(self, observation):
        at = self.action(observation)
        next_observation, rt, done, _ = self.env.step(at)
        transition = (self.expand_obs(observation), at, rt, self.expand_obs(next_observation), done)
        self.replay_memory.append(transition)
        return next_observation, rt, done

    def update_target(self):
        self.Q_target = type(self.Q)(**self.net_parameters).to(self.device)
        self.Q_target.load_state_dict(self.Q.state_dict())

    # ================================================================================================================
    # Statistics and Plot Methods
    # ================================================================================================================

    def get_results(self, feedback_after_episodes, values, mode="avg"):
        """Mode should either be max or avg or simple"""
        if mode == "simple":
            return values, range(len(values)), "Points per episode"

        func = max if mode == "max" else lambda x: sum(x) / len(x) if mode == "avg" else None
        caption = f"Average points over the last {feedback_after_episodes} episodes" if mode == "avg" \
            else f"Maximum points over the last {feedback_after_episodes} episodes" if mode == "max" else None
        if func is None:
            raise (ValueError("mode attribute should either be 'simple', 'max' or 'avg'."))

        vals = []
        for i in range(feedback_after_episodes, len(values)):
            last_vals = values[i - feedback_after_episodes:i]
            vals.append(func(last_vals))
        return vals, range(feedback_after_episodes, len(values)), caption

    def plot_results(self, feedback_after_episodes, values, plot_dir, modes=None):
        colors = ['b', 'r', 'g']
        i = 0
        if modes is None:
            modes = ['avg', 'max']
        fig, ax = plt.subplots()
        for mode in modes:
            y, x, caption = self.get_results(feedback_after_episodes, values, mode)
            ax.plot(x, y, colors[i], label=caption)
            i += 1

        ax.set(xlabel="episodes", title="Performance of the Agent During the Learning Stage")
        ax.legend()
        fig.savefig(plot_dir + ".png")
        plt.close()

    def save_stats(self, stats_dir):
        if self.n_frames == 0:
            with open(stats_dir, "w") as stats_file:
                stats_file.write("Reward,Frames")
        with open(stats_dir, "a") as stats_file:
            for i in range(len(self._points_per_episode)):
                stats_file.write(f"\n{self._points_per_episode[i]},{self._frames_per_episode[i]}")
        self._frames_per_episode = []
        self._points_per_episode = []

    # ================================================================================================================
    # Persistence Methods
    # ================================================================================================================

    def get_checkpoint(self):
        parameters = dict()
        parameters["net_parameters"] = self.net_parameters
        parameters["n_actions"] = self.n_actions
        parameters["seed"] = self.seed
        parameters["n_frames"] = self.n_frames
        parameters["update_frequency"] = self.update_frequency
        parameters["C"] = self.C
        parameters["gamma"] = self.gamma
        parameters["minibatch_size"] = self.minibatch_size
        parameters["optimizer_class_name"] = self.optimizer.__class__.__name__
        parameters["optimizer_parameters"] = self.optimizer_parameters
        parameters["optimizer_state_dict"] = self.optimizer.state_dict()
        parameters["Q_class_name"] = self.Q.__class__.__name__
        parameters["Q_state_dict"] = self.Q.state_dict()
        parameters["Q_target_state_dict"] = self.Q_target.state_dict()
        parameters["loss"] = self.loss
        parameters["policy"] = self.policy
        return parameters

    def save(self, save_dir, save_replay=True, save_policy=True, feedback_after_episodes=1):
        agent_dir = save_dir if (save_dir[-1] == '/' or save_dir[-1] == '\\') else save_dir + '/'
        stats_dir = agent_dir + "stats.csv"
        if not os.path.exists(agent_dir):
            os.makedirs(agent_dir)

        # Save this agent
        checkpoint = self.get_checkpoint()
        if not save_policy:
            checkpoint.pop("policy")
        torch.save(checkpoint, agent_dir + "agent.tar")

        # Save the statistics of the learning stage
        self.save_stats(stats_dir)

        # Recover the recent statistics and the ones that were already stored in the stats.csv file. Use them to build a
        # plot of the progress during the learning stage
        stats = pd.read_csv(stats_dir)["Reward"]
        self.plot_results(feedback_after_episodes, stats, f"{agent_dir}{self.n_frames}_steps")

        # Save the replay memory
        if save_replay:
            with open(agent_dir + "replay.p", "wb") as replay_file:
                pickle.dump(self.replay_memory, replay_file)

    @classmethod
    def load(cls, env, agent_dir, net_type, import_replay=True, optimizer=torch.optim.RMSprop,
             device=("cuda" if torch.cuda.is_available() else "cpu"), replay=None, policy=None):
        agent_dir = agent_dir if (agent_dir[-1] == '/' or agent_dir[-1] == '\\') else agent_dir + '/'
        checkpoint = torch.load(agent_dir + "agent.tar", map_location=device)
        if import_replay:
            with open(agent_dir + "replay.p", "rb") as replay_file:
                replay = pickle.load(replay_file)
        else:
            replay = replay

        if net_type.__name__ != checkpoint["Q_class_name"]:
            raise ValueError(f"The networks for this agent are of the type '{checkpoint['Q_class_name']}', but there"
                             + f" was attempt to load it using a net of the type '{net_type}'.")
        if optimizer.__name__ != checkpoint["optimizer_class_name"]:
            raise ValueError(f"The optimizer for this agent is of the type '{checkpoint['optimizer_class_name']}', but"
                             + f" there was attempt to load it using an optimizer of the type '{optimizer}'.")
        return cls.get_agent_from_checkpoint(checkpoint, net_type, optimizer, env, replay, policy, device)

    @classmethod
    def get_agent_from_checkpoint(cls, checkpoint, net_type, optimizer, env, replay, policy, device):
        agent = cls(env, replay, checkpoint["n_actions"], net_type, checkpoint["net_parameters"],
                    minibatch_size=checkpoint["minibatch_size"],
                    optimizer=optimizer,
                    optimizer_parameters=checkpoint["optimizer_parameters"],
                    C=checkpoint["C"],
                    update_frequency=checkpoint["update_frequency"],
                    gamma=checkpoint["gamma"],
                    loss=checkpoint["loss"],
                    policy=policy,
                    seed=checkpoint["seed"],
                    device=device)

        if policy is None and "policy" in checkpoint:
            agent.policy = checkpoint["policy"]

        agent.Q.load_state_dict(checkpoint["Q_state_dict"])
        agent.Q_target.load_state_dict(checkpoint["Q_target_state_dict"])

        agent.setup_optimizer(optimizer, checkpoint["optimizer_parameters"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        agent.n_frames = checkpoint["n_frames"]
        return agent

    # ================================================================================================================
    # Other Auxiliary Methods
    # ================================================================================================================

    @staticmethod
    def expand_obs(screen: np.ndarray):
        return np.expand_dims(screen, axis=0)


class DQNAtariAgent(DQNAgent):
    def __init__(self, env, minibatch_size=32,
                 replay_memory_size=1_000_000,
                 C=10_000,
                 gamma=0.99,
                 loss=clip_mse3,
                 update_frequency=4,
                 seed=0,
                 device=("cuda" if torch.cuda.is_available() else "cpu"),
                 optimizer_parameters=None,
                 policy=TrainEvalPolicy()):
        replay = DQNReplayMemoryAtari(replay_memory_size, device=device)
        env = AtariDQNEnv(env)
        optimizer_parameters = optimizer_parameters if optimizer_parameters is not None \
            else {"lr": 0.00025, "alpha": 0.95, "eps": 0.01}
        super().__init__(env, replay, env.action_space.n, DQNetwork, {"number_of_actions":env.action_space.n},
                         minibatch_size=minibatch_size,
                         C=C,
                         gamma=gamma,
                         loss=loss,
                         policy=policy,
                         update_frequency=update_frequency,
                         seed=seed,
                         device=device,
                         optimizer=torch.optim.RMSprop,
                         optimizer_parameters=optimizer_parameters)

    def learn(self, save_dir, save_replay=True, store_stats=True, verbose=True, max_steps=50_000_000, max_time=604_800,
              max_episodes=1_000_000_000, feedback_after_episodes=5, save_after_steps=1_000_000, ):
        return super().learn(save_dir, save_replay=True, verbose=verbose, max_steps=max_steps, max_time=max_time,
                             max_episodes=max_episodes, feedback_after_episodes=feedback_after_episodes,
                             save_after_steps=save_after_steps)

    def eval(self):
        super().eval()
        self.env.eval()
        self.env.restart()

    def train(self):
        super().train()
        self.env.train()

    @classmethod
    def load(cls, env, agent_dir, net_type=DQNetwork, import_replay=True, optimizer=torch.optim.RMSprop,
             device=("cuda" if torch.cuda.is_available() else "cpu"), replay=None, policy=None):
        return super().load(env, agent_dir, net_type=DQNetwork, import_replay=import_replay, optimizer=optimizer,
                            device=device, replay=replay, policy=policy)

    @classmethod
    def get_agent_from_checkpoint(cls, checkpoint, net_type, optimizer, env, replay, policy, device):
        agent = cls(env, minibatch_size=checkpoint["minibatch_size"],
                    C=checkpoint["C"],
                    gamma=checkpoint["gamma"],
                    loss=checkpoint["loss"],
                    seed=checkpoint["seed"],
                    device=device,
                    optimizer_parameters=checkpoint["optimizer_parameters"],
                    policy=policy,
                    update_frequency=checkpoint["update_frequency"])

        if policy is None and "policy" in checkpoint:
            agent.policy = checkpoint["policy"]

        agent.Q.load_state_dict(checkpoint["Q_state_dict"])
        agent.Q_target.load_state_dict(checkpoint["Q_target_state_dict"])

        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.setup_optimizer(optimizer, checkpoint["optimizer_parameters"])

        agent.n_frames = checkpoint["n_frames"]
        agent.replay_memory = replay
        return agent


