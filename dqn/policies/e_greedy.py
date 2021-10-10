from .policy import Policy
from dqn.policies.random_policy import RandomPolicy
import random as rnd
import numpy as np
import torch


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
            print("random")

        # otherwise, choose one of the actions that maximizes the Q-value

        if torch.is_tensor(Q):
            Q = Q.numpy()
        elif not isinstance(Q, np.ndarray):
            raise TypeError("The Q-values given must either be an ndarray or a torch Tensor, but an instance of"
                            f"{type(Q)} was given instead.")

        vmax = np.amax(Q)
        possible_indices = np.where(Q == vmax)
        if len(Q.shape)>1:
            return np.random.choice(possible_indices[1])
        else:
            return np.random.choice(possible_indices[0])

        return np.random.choice(possible_indices[1])

