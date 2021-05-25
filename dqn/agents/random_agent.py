from .agent import Agent
import random as rnd


class RandomAgent(Agent):

    def __init__(self, env):
        super.__init__()

    def play(self):
        self.env.reset()

        done = False
        while not done:
            self.env.render()
            action = rnd.randrange(self.n_actions)
            _, _, done, _ = self.env.step(action)

        # reset env
        self.env.reset()

