from collections import deque
import numpy.random as rnd
from operator import itemgetter
import torch
from dqn.memory.replay_memory import ReplayMemory
from dqn.memory import decompress_frames, compress_frames


class DQNReplayMemory(ReplayMemory):
    def __init__(self, size, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__(size)
        self._prev_phi = deque(maxlen=size)
        self._next_phi = deque(maxlen=size)
        self._actions = deque(maxlen=size)
        self._rewards = deque(maxlen=size)
        self._not_done = deque(maxlen=size)
        self._device = device

    def append(self, transition):
        super().append(transition)
        # the transition comes as:
        # (self._screen_to_torch(observation), at, rt, self._screen_to_torch(next_observation), done)
        self._prev_phi.append(compress_frames(transition[0]))
        self._actions.append(transition[1])
        self._rewards.append(transition[2])
        self._next_phi.append(compress_frames(transition[3]))
        self._not_done.append(not transition[4])

    def sample(self, n_samples):
        indices = rnd.randint(0, len(self), n_samples)

        prev_phi = map(decompress_frames, itemgetter(*indices)(self._prev_phi))
        prev_phi = torch.cat(list(prev_phi), dim=0).float().to(self._device)

        next_phi = map(decompress_frames, itemgetter(*indices)(self._next_phi))
        next_phi = torch.cat(list(next_phi), dim=0).float().to(self._device)

        actions = itemgetter(*indices)(self._actions)

        rewards = itemgetter(*indices)(self._rewards)
        rewards = torch.tensor(rewards).to(self._device)

        not_done = itemgetter(*indices)(self._not_done)
        not_done = torch.tensor(not_done).int().to(self._device)

        return rewards, prev_phi, next_phi, not_done, actions
