import gym
from abc import ABC, abstractmethod
import numpy as np


class Agent(ABC):

    def __init__(self, env):
        self.env = gym.make(env) if type(env) == str else env
        self.n_actions = self.env.action_space.n
        self._training = True

    def train(self):
        self._training = True

    def eval(self):
        self._training = False

    def play(self):
        self.eval()
        observation = self.env.reset()
        done = False
        begin = True

        while not done:
            self.env.render()
            at = 1 if begin else self.action(observation)
            begin = False
            observation, rt, done, _ = self.env.step(at)

        self.env.reset()

    @abstractmethod
    def action(self, observation: np.ndarray):
        raise NotImplementedError

