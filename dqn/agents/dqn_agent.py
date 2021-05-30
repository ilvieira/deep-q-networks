# TODO: all that files that create a DQNAgent should define its own environment,
#  also: directory
#  also: populate replay

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

# TODO: allow the user to choose its own replay and add an AtariDQNAgent which automatically selects the atari replay
from dqn.memory.dqn_replay_memory_atari import DQNReplayMemoryAtari
from dqn.environments.atari_dqn_env import AtariDNQEnv
from dqn.policies.atari_dqn_policy import AtariDQNPolicy
from dqn.torch_extensions import clip_mse3
from dqn.policies.random_policy import RandomPolicy


class DQNAgent(Agent):
    # TODO: check which attributes are actually necessary
    # TODO: create a getter for the number of frames "trained"
    """ Class that simulates the game and trains the DQN """
    def __init__(self, env, name, replay, minibatch_size=32,
                 optimizer=torch.optim.RMSprop,
                 C=10_000,
                 update_frequency=1,
                 gamma=0.99,
                 loss=clip_mse3,
                 policy=AtariDQNPolicy(),
                 directory="Agents/", # TODO: agent_dir should only be needed in save and load
                 seed=0,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

        super().__init__(env)

        self.device = device
        self.set_seed(seed)
        self.agent_dir = directory + name + '/'

        self.n_steps = 0
        self.update_frequency = update_frequency
        self.C = C

        # initialize Q and Q_target as a copy of Q
        # TODO: allow the choice of the net by the user
        self.Q = DQNetwork(self.n_actions).to(self.device)
        self.update_target()

        # other parameters
        self.gamma = gamma
        self.minibatch_size = minibatch_size
        self.optimizer = optimizer(self.Q.parameters(), lr=0.0025, alpha=0.95, eps=0.01)
        self.clip_value = 1

        # initialize the replay memory
        self.replay_memory = replay

        self.policy = policy
        self.loss = loss

    # ================================================================================================================
    # Setup Methods
    # ================================================================================================================

    @staticmethod
    def set_seed(seed: int):
        torch.manual_seed(seed)
        rnd.seed(seed)
        np.random.seed(seed)

    # ================================================================================================================
    # Agent Methods
    # ================================================================================================================

    def eval(self):
        super().eval()
        self.env.eval()
        self.policy.eval()

    def train(self):
        super().train()
        self.env.train()
        self.policy.train()

    def action(self, observation: np.ndarray):
        """Chooses an action given an observation"""
        phi = torch.tensor(self.expand_obs(observation)).float()

        # phi is added to the gpu so that Q can make a prediction on it
        with torch.no_grad():
            q_vals = self.Q(phi.to(self.device))
            action = self.policy.choose_action(q_vals.detach().cpu())

            # remove prev_phi from the gpu to clear vram
            phi.cpu()
            return action

    # ================================================================================================================
    # DQN Specific Methods
    # ================================================================================================================

    def populate_replay_memory(self, n_samples, verbose=True):
        """ Adds the first transitions to the replay memory by playing the game with random actions"""
        random_policy = RandomPolicy(self.n_actions)

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
                at = random_policy.choose_action()
                phi, rt, done, _ = self.env.step(at)
                phi = self.expand_obs(phi)
                transition = (prev_phi, at, rt, phi, done)
                self.replay_memory.append(transition)
                transitions_added += 1
                prev_phi = phi

        if verbose:
            print(f"Done - {time.time()-start}s")

    def learn(self, store_stats=True, create_new=True, verbose=True,
              max_steps=50_000_000,
              max_time=604_800,
              max_episodes=1_000_000_000,
              feedback_after_episodes=5,
              save_after_steps=1_000_000,):
        """The algorithm as described in 'Human-level control through deep reinforcement learning'"""

        if store_stats:
            if not os.path.exists(self.agent_dir):
                os.makedirs(self.agent_dir)

        if verbose:
            print("Beginning the training stage...")
        start = time.time()

        self.train()

        total_reward = 0
        max_reward = float('-inf')
        points_per_episode = []
        frames_per_episode = []

        if store_stats:
            self.save_stats(points_per_episode, frames_per_episode, create_new=create_new)

        for episode in range(max_episodes):
            ep_reward = 0
            ep_frames = 0

            if verbose:
                if episode % feedback_after_episodes == 0 and not episode == 0:
                    print(f"... episode {episode} ... - {time.time()-start}")
                    print(f"  average reward: {total_reward / feedback_after_episodes}")
                    print(f"  maximum reward: {max_reward}")
                    print(f"  total frames: {self.n_steps}")
                    print(f"  transitions stored: {len(self.replay_memory)}")
                    total_reward = 0
                    max_reward = float('-inf')

            observation = self.env.reset()
            done = False

            while not done:
                observation, rt = self.play_and_store_transition(observation)
                self.n_steps += 1
                ep_frames += 1
                total_reward += rt
                ep_reward += rt

                # Backpropagation at each upadte_frequency frames
                if self.n_steps % self.update_frequency == 0:
                    self.optimize_model()

                # Update target network at each C frames
                if self.n_steps % self.C == 0:
                    self.update_target()

                # PERSISTENCE: Save after each save_after_steps_frame
                if self.n_steps > 0 and self.n_steps % save_after_steps == 0:
                    stats_dir = self.save_stats(points_per_episode, frames_per_episode) if store_stats else None
                    self.save(stats_dir=stats_dir, feedback_after_episodes=feedback_after_episodes)
                    points_per_episode = []
                    frames_per_episode = []
                    if verbose:
                        print(f"Agent saved ({self.n_steps} steps)")

                # STOP if the maximum number of steps or time are reached
                if self.n_steps > max_steps or time.time() - start > max_time:
                    if verbose:
                        print("Done")
                    stats_dir = self.save_stats(points_per_episode, frames_per_episode) if store_stats else None
                    self.save(stats_dir=stats_dir, feedback_after_episodes=feedback_after_episodes)
                    return

            if ep_reward > max_reward:
                max_reward = ep_reward
            points_per_episode.append(ep_reward)
            frames_per_episode.append(ep_frames)

        stats_dir = self.save_stats(points_per_episode, frames_per_episode) if store_stats else None
        self.save(stats_dir=stats_dir, feedback_after_episodes=feedback_after_episodes)
        if verbose:
            print("Done")

    def optimize_model(self):
        r, prev_phi, next_phi, not_done, actions = self.replay_memory.sample(self.minibatch_size)

        with torch.no_grad():
            y = (r + self.gamma * not_done * self.Q_target(next_phi)
                 .max(axis=1, keepdim=True).values.view(self.minibatch_size))

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
        return next_observation, rt

    def update_target(self):
        # TODO: allow the choice of net by the user. Probably a good idea to copy the class of Q
        self.Q_target = DQNetwork(self.n_actions).to(self.device)
        self.Q_target.load_state_dict(self.Q.state_dict())

    # ================================================================================================================
    # Statistics and Plot Methods
    # ================================================================================================================

    def get_results(self, feedback_after_episodes, values, mode="avg"):
        """Mode should either be max or avg or simple"""
        if mode == "simple":
            return values, range(len(values)), "Points per episode"

        func = max if mode =="max" else lambda x: sum(x)/len(x) if mode=="avg" else None
        caption = f"Average points over the last {feedback_after_episodes} episodes" if mode == "avg" \
            else f"Maximum points over the last {feedback_after_episodes} episodes" if mode == "max" else None
        if func is None:
            raise(ValueError("mode attribute should either be 'simple', 'max' or 'avg'."))

        vals = []
        for i in range(feedback_after_episodes, len(values)):
            last_vals = values[i-feedback_after_episodes:i]
            vals.append(func(last_vals))
        return vals, range(feedback_after_episodes, len(values)), caption

    def plot_results(self, feedback_after_episodes, values, plot_dir, modes=None):
        colors = ['b','r','g']
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

    def save_stats(self, points_per_episode, frames_per_episode, create_new=False):
        stats_dir = self.agent_dir + "stats.csv"
        if create_new:
            with open(stats_dir, "w") as stats_file:
                stats_file.write("Reward,Frames")
        with open(stats_dir, "a") as stats_file:
            for i in range(len(points_per_episode)):
                stats_file.write(f"\n{points_per_episode[i]},{frames_per_episode[i]}")
        return stats_dir

    # ================================================================================================================
    # Persistence Methods
    # ================================================================================================================

    def save(self, save_dir=None, save_replay=True, stats_dir=None, feedback_after_episodes=None):
        # TODO: while saving, store the parameters of this class
        if save_dir is None:
            save_dir = self.agent_dir
        agent_dir = save_dir if (save_dir[-1] == '/' or save_dir[-1] == '\\') else save_dir + '/'
        if not os.path.exists(agent_dir):
            os.makedirs(agent_dir)

        torch.save(self.Q.state_dict(), agent_dir+"q_dict.pt")
        torch.save(self.Q_target.state_dict(), agent_dir+"q_target_dict.pt")
        torch.save(self.optimizer.state_dict(), agent_dir+"optimizer.pt")
        if save_replay:
            with open(agent_dir+"replay.p", "wb") as replay_file:
                pickle.dump(self.replay_memory, replay_file)
        if stats_dir is not None:
            stats = pd.read_csv(stats_dir)["Reward"]
            self.plot_results(feedback_after_episodes, stats, f"{self.agent_dir}{self.n_steps}_steps")

    @classmethod
    def load(cls, env, name, directory="Agents/", import_replay=True, populate_replay=False,
             optimizer=torch.optim.RMSprop):
        # TODO: review loading as well
        agent = DQNAgent(env, name, None, directory=directory)
        agent_dir = os.getcwd() + '/' + directory + name + '/'

        if import_replay:
            with open(agent_dir + "replay.p", "rb") as replay_file:
                agent.replay_memory = pickle.load(replay_file)

        agent.Q.load_state_dict(torch.load(agent_dir + "q_dict.pt"))
        agent.Q_target.load_state_dict(torch.load(agent_dir + "q_target_dict.pt"))
        # TODO: is this necessary? Commented until can tell the answer. Need further testing to make a decision
        # agent.optimizer.load_state_dict(torch.load(agent_dir + "optimizer.pt"))

        agent.optimizer = optimizer(agent.Q.parameters(), lr=0.0025, alpha=0.95, eps=0.01)
        return agent

    # ================================================================================================================
    # Other Auxiliary Methods
    # ================================================================================================================

    @staticmethod
    def expand_obs(screen: np.ndarray):
        return np.expand_dims(screen, axis=0)


class DQNAtariAgent(DQNAgent):
    def __init__(self, env, name, minibatch_size=32,
                 replay_memory_size=1_000_000,
                 C=10_000,
                 gamma=0.99,
                 loss=clip_mse3,
                 directory="Agents/",
                 seed=0,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        replay = DQNReplayMemoryAtari(replay_memory_size, device=device)
        super().__init__(AtariDNQEnv(env), name, replay,
                         minibatch_size=minibatch_size,
                         C=C,
                         gamma=gamma,
                         loss=loss,
                         policy=AtariDQNPolicy(),
                         update_frequency=4,
                         directory=directory,
                         seed=seed,
                         device=device)

    def learn(self, store_stats=True, create_new=True, verbose=True,  max_steps=50_000_000, max_time=604_800,
              max_episodes=1_000_000_000, feedback_after_episodes=5, save_after_steps=1_000_000,):
        return super().learn(store_stats=store_stats, create_new=create_new, verbose=verbose, max_steps=max_steps,
                             max_time=max_time, max_episodes=max_episodes,
                             feed_back_after_episodes=feedback_after_episodes, save_after_steps=save_after_steps)

    def eval(self):
        super().eval()
        self.env.restart()

    def train(self):
        super().train()
        self.env.restart()



