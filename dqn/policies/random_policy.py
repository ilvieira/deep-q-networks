from lib.policies.policy import Policy
import random as rnd


class RandomPolicy(Policy):
    def __init__(self, n_actions=None):
        self.n_actions = n_actions

    def choose_action(self, Q=None):
        # if the number of available actions is known, choose one randomly
        if self.n_actions is not None:
            return rnd.randrange(self.n_actions)

        # otherwise, get that information from the q-value vector
        Q = Q.numpy()
        N = Q.shape[1]
        return rnd.randrange(N)




