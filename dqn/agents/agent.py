import gym
from abc import ABC, abstractmethod


class Agent(ABC):

    def __init__(self, env):
        # TODO: agent should be independent of gym, this should only depend on the environment
        self.env = gym.make(env) if type(env) == str else env

        # TODO: there is probably no need to save the number of actions here
        self.n_actions = self.env.action_space.n

        self._training = True

    def train(self):
        self._training = True

    def eval(self):
        self._training = False

    def play(self, render=True):
        self.eval()
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

    @abstractmethod
    def action(self, observation):
        raise NotImplementedError

