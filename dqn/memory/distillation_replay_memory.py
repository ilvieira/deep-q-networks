from collections import deque
import numpy.random as rnd
from operator import itemgetter
import torch

from dqn.memory.replay_memory import ReplayMemory
from dqn.memory import decompress_frames, compress_frames

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DistillationReplayMemory(ReplayMemory):

    def __init__(self, size):
        super().__init__(size)
        self._obs = deque(maxlen=size)
        self._q_vals = deque(maxlen=size)

    def append(self, transition):
        super().append(transition)
        self._obs.append(compress_frames(transition[0]))
        self._q_vals.append(transition[1])

    def sample(self, n_samples, device=default_device):
        indices = rnd.randint(0, len(self), n_samples)

        obs = map(decompress_frames, itemgetter(*indices)(self._obs))
        obs = torch.cat(list(obs), dim=0).float().to(device)

        q_vals = itemgetter(*indices)(self._q_vals)
        q_vals = torch.cat(q_vals, dim=0).float().to(device)

        return obs, q_vals

