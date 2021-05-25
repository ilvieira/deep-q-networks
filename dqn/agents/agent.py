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

    def play(self):
        self.eval()
        observation = self.env.reset()
        done = False
        begin = True

        while not done:
            self.env.render()
            # TODO: remove this line
            at = 1 if begin else self.action(observation)
            begin = False
            observation, rt, done, _ = self.env.step(at)
        self.env.reset()

    @abstractmethod
    def action(self, observation):
        raise NotImplementedError

