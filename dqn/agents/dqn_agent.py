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
from dqn.memory.replay_memory import DQNReplayMemoryAtariV as ReplayMemory
from dqn.environments.atari_dqn_env import AtariDNQEnv
from dqn.policies.atari_dqn_policy import AtariDQNPolicy
from dqn.torch_extensions import clip_mse3
from dqn.policies.random_policy import RandomPolicy


class DQNAgent(Agent):
    # TODO: check which attributes are actually necessary
    """ Class that simulates the game and trains the DQN """
    def __init__(self, env, name, minibatch_size=32,
                 replay_memory_size=1_000_000,
                 replay_start_size=50_000,
                 C=10_000,
                 gamma=0.99,
                 hist_len=4,
                 atari=True,
                 loss=clip_mse3,
                 policy=AtariDQNPolicy(),
                 max_steps=50_000_000,
                 max_time=604_800,
                 max_episodes=1_000_000_000,
                 feed_back_after_episodes=5,
                 game_name="game",
                 save_after_steps=1_000_000,
                 directory="Agents/",
                 plot_name=None,
                 seed=0):

        super().__init__(AtariDNQEnv(env) if atari else env)

        # TODO: device should probably be decided in the __init__ input
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.set_seed(seed)

        # TODO: only the environment should be able to know if this is atari or not
        self._atari = atari

        # TODO: probably should be given as a parameter from the user
        self.agent_dir = os.getcwd() + '/' + directory + name + '/'

        # TODO: probably should only be defined in the plot creation method
        self._plot_name = plot_name if plot_name is not None else str(env) if atari else game_name

        # TODO: redefine how the feedback is displayed
        self.feedback_after_episodes = feed_back_after_episodes
        self.save_after_steps = save_after_steps
        self.n_steps = 0
        self.hist_len = hist_len

        self.C = C

        self.max_steps = max_steps
        self.max_time = max_time
        self.max_episodes = max_episodes
        self.frames_per_step = 4 if atari else 1

        # initialize Q and Q_target as a copy of Q
        # TODO: allow the choice of the net by the user
        self.Q = DQNetwork(self.n_actions).to(self.device)
        self.update_target()

        # other parameters
        self.gamma = gamma
        self.minibatch_size = minibatch_size
        # TODO: leave as an option to the user
        self.optimizer = torch.optim.RMSprop(self.Q.parameters(), lr=0.0025, alpha=0.95, eps=0.01)
        self._clip_value = 1

        # initialize the replay memory
        self.replay_memory = ReplayMemory(replay_memory_size)
        self.replay_start_size = replay_start_size

        # TODO: this should not be in setup.
        self.populate_replay_memory()

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

    # TODO: see what can be reused from Agent
    def action(self, observation: np.ndarray):
        """Chooses an action given an observation"""
        phi = torch.tensor(self.screen_to_torch(observation)).float()

        # phi is added to the gpu so that Q can make a prediction on it
        with torch.no_grad():
            q_vals = self.Q(phi.to(self.device))
            action = self.policy.choose_action(q_vals.detach().cpu())

            # remove prev_phi from the gpu to clear vram
            phi.cpu()
            return action

    # TODO: see what can be reused from Agent
    def play(self, render=True):
        self.eval()
        self.env.restart()
        observation = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            if render:
                self.env.render()
            at = self.action(observation)
            observation, rt, done, _ = self.env.step(at)
            total_reward += rt

        self.env.reset()
        return total_reward

    # ================================================================================================================
    # DQN Specific Methods
    # ================================================================================================================
    def populate_replay_memory(self, verbose=True):
        """ Adds the first transitions to the replay memory by playing the game with random actions"""
        random_policy = RandomPolicy(self.n_actions)

        if verbose:
            print("Populating the replay memory...")
            start = time.time()

        while len(self.replay_memory) < self.replay_start_size:
            # restart the environment and get the first observation
            prev_phi = self.screen_to_torch(self.env.reset())

            # done becomes True whenever a terminal state is reached
            done = False

            while (not done) and (len(self.replay_memory) < self.replay_start_size):
                at = random_policy.choose_action()
                phi, rt, done, _ = self.env.step(at)
                phi = self.screen_to_torch(phi)
                transition = (prev_phi, at, rt, phi, done)
                self.replay_memory.append(transition)
                prev_phi = phi
        if verbose:
            print(f"Done - {time.time()-start}s")

    def learn(self, store_stats=True, create_new=True):
        """The algorithm as described in 'Human-level control through deep reinforcement learning'

        NOTE: if self.feedback_after_episodes is an int n, the function will print feedback at each n episodes; if it is
              None, no feedback will be given.
        """
        if store_stats:
            if not os.path.exists(self.agent_dir):
                os.makedirs(self.agent_dir)

        if self.feedback_after_episodes is not None:
            print("Beginning the training stage...")
            start = time.time()

        self.train()

        total_reward = 0
        max_reward = float('-inf')
        points_per_episode = []
        frames_per_episode = []
        self.save_stats(points_per_episode, frames_per_episode, create_new=create_new)

        for episode in range(self.max_episodes):
            ep_reward = 0
            ep_frames = 0

            if self.feedback_after_episodes is not None:
                if episode % self.feedback_after_episodes == 0 and not episode == 0:
                    print(f"... episode {episode} ... - {time.time()-start}")
                    print(f"  average reward: {total_reward / self.feedback_after_episodes}")
                    print(f"  maximum reward: {max_reward}")
                    print(f"  total frames: {self.n_steps}")
                    print(f"  transitions stored: {len(self.replay_memory)}")
                    total_reward = 0
                    max_reward = float('-inf')

            observation = self.env.reset()
            done = False

            while not done:
                self.n_steps += 1
                at = self.action(observation)
                next_observation, rt, done, _ = self.env.step(at)
                transition = (self.screen_to_torch(observation), at, rt,
                              self.screen_to_torch(next_observation), done)
                observation = next_observation
                self.replay_memory.append(transition)

                total_reward += rt
                ep_reward += rt
                ep_frames += 1

                if self.n_steps % self.hist_len == 0:
                    self.optimize_model()

                if self.n_steps % self.C == 0:
                    self.update_target()

                if self.n_steps > 0 and self.n_steps % self.save_after_steps == 0:
                    stats_dir = self.save_stats(points_per_episode, frames_per_episode)
                    self.save(stats_dir=stats_dir)
                    points_per_episode = []
                    frames_per_episode = []
                    print(f"Agent saved ({self.n_steps} steps)")

                if self.n_steps > self.max_steps or time.time() - start > self.max_time:
                    if self.feedback_after_episodes is not None:
                        print("Done")
                    stats_dir = self.save_stats(points_per_episode, frames_per_episode)
                    self.save(stats_dir=stats_dir)
                    return

            if ep_reward > max_reward:
                max_reward = ep_reward
            points_per_episode.append(ep_reward)
            frames_per_episode.append(ep_frames)

        stats_dir = self.save_stats(points_per_episode, frames_per_episode)
        self.save(stats_dir=stats_dir)
        if self.feedback_after_episodes is not None:
            print("Done")

    def optimize_model(self):
        r, prev_phi, next_phi, not_done, actions = self.replay_memory.sample(self.minibatch_size, device=self.device)

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

        # clear space in the gpu by deleting these tensors which have no more use:
        del r, prev_phi, next_phi, not_done, y, q_vals, q_phi

    def update_target(self):
        # TODO: allow the choice of net by the user. Probably a good idea to copy the class of Q
        self.Q_target = DQNetwork(self.n_actions).to(self.device)
        self.Q_target.load_state_dict(self.Q.state_dict())

    def q_vals(self, obs):
        with torch.no_grad():
            return self.Q(obs).detach().cpu()

    # ================================================================================================================
    # Statistics and Plot Methods
    # ================================================================================================================

    def get_results(self, values, mode="avg"):
        """Mode should either be max or avg or simple"""
        if mode == "simple":
            return values, range(len(values)), "Points per episode"

        func = max if mode =="max" else lambda x: sum(x)/len(x) if mode=="avg" else None
        caption = f"Average points over the last {self.feedback_after_episodes} episodes" if mode == "avg" \
            else f"Maximum points over the last {self.feedback_after_episodes} episodes" if mode == "max" else None
        if func is None:
            raise(ValueError("mode attribute should either be 'simple', 'max' or 'avg'."))

        vals = []
        for i in range(self.feedback_after_episodes, len(values)):
            last_vals = values[i-self.feedback_after_episodes:i]
            vals.append(func(last_vals))
        return vals, range(self.feedback_after_episodes, len(values)), caption

    def plot_results(self, values, plot_dir, modes=None):
        colors = ['b','r','g']
        i = 0
        if modes is None:
            modes = ['avg', 'max']
        fig, ax = plt.subplots()
        for mode in modes:
            y, x, caption = self.get_results(values, mode)
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

    def save(self, save_dir=None, save_replay=True, stats_dir=None):
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
            self.plot_results(stats, f"{self.agent_dir}{self.n_steps}_steps")

    @classmethod
    def load(cls, env, name, directory="Agents/", import_replay=True, populate_replay=False):
        agent = DQNAgentOpt(env, name, directory=directory, replay_start_size=0) \
            if (import_replay or not populate_replay) else DQNAgentOpt(env, name, directory=directory)

        agent_dir = os.getcwd() + '/' + directory + name + '/'

        if import_replay:
            with open(agent_dir + "replay.p", "rb") as replay_file:
                agent.replay_memory = pickle.load(replay_file)

        agent.Q.load_state_dict(torch.load(agent_dir + "q_dict.pt"))
        agent.Q_target.load_state_dict(torch.load(agent_dir + "q_target_dict.pt"))
        # TODO: is this necessary?
        agent.optimizer.load_state_dict(torch.load(agent_dir + "optimizer.pt"))

        # TODO: allow user to choose optimizer
        agent.optimizer = torch.optim.RMSprop(agent.Q.parameters(), lr=0.0025, alpha=0.95, eps=0.01)
        return agent
    # ================================================================================================================
    # Other Auxiliary Methods
    # ================================================================================================================

    # TODO: this probably can be removed, check the atari wrapper used
    @staticmethod
    def screen_to_torch(screen: np.ndarray):
        return np.expand_dims(screen.transpose((2, 0, 1)), axis=0)


