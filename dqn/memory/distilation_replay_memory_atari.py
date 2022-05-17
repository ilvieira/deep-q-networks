from collections import deque
import numpy.random as rnd
from operator import itemgetter
import torch
import numpy as np

from dqn.memory.replay_memory import ReplayMemory
from dqn.memory import compress_frames, decompress_as_np


class DistillationReplayMemoryAtari(ReplayMemory):
    """
    A transition for this type of replay will be a tuple (reward, prev_phi, next_phi, not_done, actions). Each phi is a
    tuple containing 4 consecutive frames.
    """

    def __init__(self, size, device=("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__(size)
        self._phi = deque(maxlen=size+3)
        self._q_vals = deque(maxlen=size)
        self._not_done = deque(maxlen=size)

    def append(self, transition):
        (prev_phi, q_vals, done) = transition

        # First transition
        if len(self) == 0:
            # Add the first 4 frames of the transition. This is only done once, because for future transitions, there
            # will only be a need to add a new frame for transition, because the remaining frames are repeated from
            # previous observations.
            for i in range(4):
                # Store all the frames from previous phi
                self._phi.append(compress_frames(prev_phi[None, :, i, :]))

        # When the next transition begins a new episode
        if len(self) > 0 and not self._not_done[-1]:
            for i in range(3):
                # Store all the frames from previous phi
                self._phi.append(compress_frames(prev_phi[None, :, i, :]))

                # Add None values to the other deques so that the indices still match.
                self._q_vals.append(None)
                self._not_done.append(None)

        # add only the last frame of the the observation
        self._phi.append(compress_frames(prev_phi[None, :, 3, :]))
        self._q_vals.append(q_vals)
        self._not_done.append(not done)

        super().append(transition)

    def _get_indices(self, sample_size):
        def sample_one(unused):
            # The argument 'unused' is just a dirty trick so that map can be used for this function

            # Do not check the first 3 frames, because there is not way of checking if any of them is terminal:
            # t[i] indicates if frame phi[i+4] is terminal
            # Do not check the last 3, because 4 frames are needed for a full transition:
            # prev_phi <- (x0,x1,x2,x3)
            i = rnd.randint(3, len(self)-3)

            # While the transition is not valid, look for a new one
            while self._not_done[i] is None:
                i = rnd.randint(3, len(self)-3)
            return i
        return list(map(sample_one, range(sample_size)))

    def __sample_phi(self, index):
        return torch.tensor(np.concatenate([decompress_as_np(self._phi[i]) for i in range(index, index+4)], axis=1))

    def sample(self, n_samples, device):
        indices = self._get_indices(n_samples)

        prev_phi = list(map(self.__sample_phi, indices))
        prev_phi = torch.cat(prev_phi, dim=0).to(device).float()

        q_vals = itemgetter(*indices)(self._q_vals)
        q_vals = torch.cat(q_vals, dim=0).float().to(device)

        return prev_phi, q_vals
