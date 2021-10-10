from dqn.policies.policy import Policy
import random as rnd
import torch
import numpy as np


class RandomPolicy(Policy):
    def __init__(self, n_actions=None):
        self.n_actions = n_actions

    def choose_action(self, Q=None):
        # if the number of available actions is known, choose one randomly
        if self.n_actions is not None:
            return rnd.randrange(self.n_actions)

        # otherwise, get that information from the q-value vector
        if torch.is_tensor(Q):
            Q = Q.numpy()
        elif not isinstance(Q, np.ndarray):
            raise TypeError("The Q-values given must either be an ndarray or a torch Tensor, but an instance of"
                            f"{type(Q)} was given instead.")

        if len(Q.shape)>1:
            N = Q.shape[1]
        else:
            N = Q.shape[0]
        return rnd.randrange(N)




