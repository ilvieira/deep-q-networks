from collections import deque
import numpy.random as rnd
from operator import itemgetter
import torch
import numpy as np

from dqn.memory.replay_memory import ReplayMemory
from dqn.memory import compress_frames, decompress_as_np


class DQNReplayMemoryAtariV(ReplayMemory):
    def __init__(self, size, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__(size)
        self._phi = deque(maxlen=size+4)
        # because every phi is the set of 4 frames, and phi_next contains on extra frame

        self._actions = deque(maxlen=size)
        self._rewards = deque(maxlen=size)
        self._not_done = deque(maxlen=size)
        self._device = device

    def append(self, transition):
        prev_phi = transition[0]
        action = transition[1]
        reward = transition[2]
        next_phi = transition[3]
        done = transition[4]

        # if the transition starts a new episode
        if len(self) == 0 or not self._not_done[-1]:
            for i in range(4):
                self._phi.append(compress_frames(prev_phi[None, :, i, :]))

        self._actions.append(action)
        self._rewards.append(reward)
        self._phi.append(compress_frames(next_phi[None, :, 3, :]))
        self._not_done.append(not done)

        super().append(transition)

    def _get_indices(self, sample_size):
        def sample_one(unused):
            # The argument 'unused' is just a dirty trick so that map can be used for this function

            # Do not check the first four frames, because there is not way of checking if any of them is terminal:
            # t[i] indicates if frame phi[i+4] is terminal
            # Do not check the last four, because 5 frames are needed for a full transition:
            # prev_phi <- (x0,x1,x2,x3), next_phi <- (x1,x2,x3,x4)
            i = rnd.randint(4, len(self)-4)
            while not all([self._not_done[j] for j in range(i-4, i)]):
                i = rnd.randint(4, len(self)-4)
            return i
        return list(map(sample_one, range(sample_size)))

    def __sample_phi(self, index):
        return torch.tensor(np.concatenate([decompress_as_np(self._phi[i]) for i in range(index, index+4)], axis=1))

    def __sample_next_phi(self, index):
        return torch.tensor(np.concatenate([decompress_as_np(self._phi[i]) for i in range(index+1, index+5)], axis=1))

    def sample(self, n_samples):
        indices = self._get_indices(n_samples)

        prev_phi = list(map(self.__sample_phi, indices))
        prev_phi = torch.cat(prev_phi, dim=0).to(self._device).float()

        next_phi = list(map(self.__sample_next_phi, indices))
        next_phi = torch.cat(list(next_phi), dim=0).to(self._device).float()

        actions = itemgetter(*indices)(self._actions)

        rewards = itemgetter(*indices)(self._rewards)
        rewards = torch.tensor(rewards).to(self._device)

        not_done = itemgetter(*indices)(self._not_done)
        not_done = torch.tensor(not_done).int().to(self._device)

        return rewards, prev_phi, next_phi, not_done, actions
