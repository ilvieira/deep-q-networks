import gym
from abc import ABC, abstractmethod


class Agent(ABC):

    def __init__(self, env, n_actions):
        self.env = env
        self.n_actions = n_actions
        self._training = True

    def train(self):
        self._training = True

    def eval(self):
        self._training = False

    def play(self, render=True, reset_at_the_end=True):
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
        self.env.render()
        if reset_at_the_end:
            self.env.reset()
        return total_reward

    @abstractmethod
    def action(self, observation):
        raise NotImplementedError

