from .policy import Policy
from dqn.policies.random_policy import RandomPolicy
import random as rnd
import numpy as np


class EGreedy(Policy):
    """ Epsilon-greedy policy"""

    def __init__(self, epsilon, n_actions=None):
        self.epsilon = epsilon
        self.random_policy = RandomPolicy(n_actions)

    def choose_action(self, Q=None):
        """Q is an tensor with shape (N,), containing the Q values of the N possible actions"""
        p = rnd.uniform(0, 1)

        # with probability epsilon, choose a random action
        if p < self.epsilon:
            return self.random_policy.choose_action(Q=Q)

        # otherwise, choose one of the actions that maximizes the Q-value
        Q = Q.numpy()
        vmax = np.amax(Q)
        possible_indices = np.where(Q == vmax)
        return np.random.choice(possible_indices[1])

