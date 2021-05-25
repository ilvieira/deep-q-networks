from .agent import Agent
import random as rnd


class RandomAgent(Agent):

    def __init__(self, env):
        super.__init__()

    # TODO: implement action and see if there is a need to actually implement play here
    def play(self):
        self.env.reset()

        done = False
        while not done:
            self.env.render()
            action = rnd.randrange(self.n_actions)
            _, _, done, _ = self.env.step(action)

        # reset env
        self.env.reset()

