from collections import deque
from abc import ABC
import numpy.random as rnd
from operator import itemgetter
import torch
import numpy as np

from lib.memory import decompress_frames, compress_frames, decompress_as_np

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemoryOpt(ABC):
    def __init__(self, size):
        self.__len = 0
        self.__max_len = size

    def append(self, transition):
        self.__len += 1
        self.__len = min(self.__len, self.__max_len)

    def __len__(self):
        return self.__len


class DQNReplayMemory(ReplayMemoryOpt):
    def __init__(self, size):
        super().__init__(size)
        self._prev_phi = deque(maxlen=size)
        self._next_phi = deque(maxlen=size)
        self._actions = deque(maxlen=size)
        self._rewards = deque(maxlen=size)
        self._not_done = deque(maxlen=size)

    def append(self, transition):
        super().append(transition)
        # the transition comes as:
        # (self._screen_to_torch(observation), at, rt, self._screen_to_torch(next_observation), done)
        self._prev_phi.append(compress_frames(transition[0]))
        self._actions.append(transition[1])
        self._rewards.append(transition[2])
        self._next_phi.append(compress_frames(transition[3]))
        self._not_done.append(not transition[4])

    def sample(self, n_samples, device=default_device):
        indices = rnd.randint(0, len(self), n_samples)

        prev_phi = map(decompress_frames, itemgetter(*indices)(self._prev_phi))
        prev_phi = torch.cat(list(prev_phi), dim=0).float().to(device)

        next_phi = map(decompress_frames, itemgetter(*indices)(self._next_phi))
        next_phi = torch.cat(list(next_phi), dim=0).float().to(device)

        actions = itemgetter(*indices)(self._actions)

        rewards = itemgetter(*indices)(self._rewards)
        rewards = torch.tensor(rewards).to(device)

        not_done = itemgetter(*indices)(self._not_done)
        not_done = torch.tensor(not_done).int().to(device)

        return rewards, prev_phi, next_phi, not_done, actions


class DQNReplayMemoryAtari(ReplayMemoryOpt):
    def __init__(self, size):
        super().__init__(size)
        self._phi = deque(maxlen=size+4)
        # because every phi is the set of 4 frames, and phi_next contains on extra frame

        self._actions = deque(maxlen=size)
        self._rewards = deque(maxlen=size)
        self._not_done = deque(maxlen=size)

    def append(self, transition):
        super().append(transition)
        prev_phi = transition[0]
        action = transition[1]
        reward = transition[2]
        next_phi = transition[3]
        done = transition[4]

        if len(self) == 0:
            for i in range(4):
                self._phi.append(compress_frames(prev_phi[None, :, i, :]))

        self._actions.append(action)
        self._rewards.append(reward)
        self._phi.append(compress_frames(next_phi[None, :, 3, :]))
        self._not_done.append(not done)

    def __sample_phi(self, index):
        return torch.tensor(np.concatenate([decompress_as_np(self._phi[i]) for i in range(index, index+4)], axis=1))

    def __sample_next_phi(self, index):
        return torch.tensor(np.concatenate([decompress_as_np(self._phi[i]) for i in range(index+1, index+5)], axis=1))

    def sample(self, n_samples, device=default_device):
        indices = rnd.randint(0, len(self)-4, n_samples)

        prev_phi = list(map(self.__sample_phi, indices))
        prev_phi = torch.cat(prev_phi, dim=0).to(device).float()

        next_phi = list(map(self.__sample_next_phi, indices))
        next_phi = torch.cat(list(next_phi), dim=0).to(device).float()

        actions = itemgetter(*indices)(self._actions)

        rewards = itemgetter(*indices)(self._rewards)
        rewards = torch.tensor(rewards).to(device)

        not_done = itemgetter(*indices)(self._not_done)
        not_done = torch.tensor(not_done).int().to(device)

        return rewards, prev_phi, next_phi, not_done, actions


class DQNReplayMemoryAtariV(ReplayMemoryOpt):
    def __init__(self, size):
        super().__init__(size)
        self._phi = deque(maxlen=size+4)
        # because every phi is the set of 4 frames, and phi_next contains on extra frame

        self._actions = deque(maxlen=size)
        self._rewards = deque(maxlen=size)
        self._not_done = deque(maxlen=size)

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

    def sample(self, n_samples, device=default_device):
        indices = self._get_indices(n_samples)

        prev_phi = list(map(self.__sample_phi, indices))
        prev_phi = torch.cat(prev_phi, dim=0).to(device).float()

        next_phi = list(map(self.__sample_next_phi, indices))
        next_phi = torch.cat(list(next_phi), dim=0).to(device).float()

        actions = itemgetter(*indices)(self._actions)

        rewards = itemgetter(*indices)(self._rewards)
        rewards = torch.tensor(rewards).to(device)

        not_done = itemgetter(*indices)(self._not_done)
        not_done = torch.tensor(not_done).int().to(device)

        return rewards, prev_phi, next_phi, not_done, actions


class DistillationReplayMemoryOpt(ReplayMemoryOpt):

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


class DistillationReplayMemoryAtariV(ReplayMemoryOpt):
    def __init__(self, size):
        super().__init__(size)

        # because every obs is the set of 4 frames
        self._obs = deque(maxlen=size+3)
        self._q_vals = deque(maxlen=size)
        self._not_done = deque(maxlen=size)

    def append(self, transition):
        obs = transition[0]
        q_vals = transition[1]
        done = not transition[2]

        if len(self) == 0 or not self._not_done[-1]:
            for i in range(4):
                self._phi.append(compress_frames(obs[None, :, i, :]))
        else:
            self._phi.append(compress_frames(obs[None, :, 3, :]))

        self._q_vals.append(q_vals)
        self._not_done.append(not done)
        super().append(transition)

    def _get_indices(self, sample_size):
        def sample_one(unused):
            # The argument 'unused' is just a dirty trick so that map can be used for this function

            # Do not check the first four frames, because there is not way of checking if any of them is terminal:
            # t[i] indicates if frame phi[i+4] is terminal
            # Do not check the last 3, because 4 frames are needed for a full transition:
            # obs <- (x0,x1,x2,x3)
            i = rnd.randint(4, len(self)-3)
            while not all([self._not_done[j] for j in range(i-3, i)]):
                i = rnd.randint(4, len(self)-3)
            return i
        return list(map(sample_one, range(sample_size)))

    def __sample_obs(self, index):
        return torch.tensor(np.concatenate([decompress_as_np(self._obs[i]) for i in range(index, index+4)], axis=1))

    def sample(self, n_samples, device=default_device):
        indices = self._get_indices(n_samples)

        obs = list(map(self.__sample_obs, indices))
        obs = torch.cat(obs, dim=0).to(device).float()

        q_vals = itemgetter(*indices)(self._rewards)
        q_vals = torch.cat(q_vals, dim=0).float().to(device)

        return obs, q_vals


