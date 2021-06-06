from collections import deque
import numpy.random as rnd
from operator import itemgetter
import torch
import numpy as np

from dqn.memory.replay_memory import ReplayMemory
from dqn.memory import compress_frames, decompress_as_np


class DQNReplayMemoryAtari(ReplayMemory):
    """
    A transition for this type of replay will be a tuple (reward, prev_phi, next_phi, not_done, actions). Each phi is a
    tuple containing 4 consecutive frames.
    """

    def __init__(self, size, device=("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__(size)

        # The deque _phi will contain the transitions that make up the observations of the game state. Thus, the length
        # of _phi is size+4 (by induction):
        #   * size = 1 (base case) -> a full transition needs 5 frames: (x1,x2,x3,x4) make up the observation of the
        #                             game state before the action is played and (x2,x3,x4,x5) make up the next
        #                             observation
        #   * size = n + 1 -> by induction hypothesis, if size = n, then len(__phi)=n+4. Thus, the last four frames
        #                     constitute an observation and, by adding the next frame of the game, we can consider the
        #                     last three of them and this new one as another observation. As a consequence, by adding
        #                     the capacity of 1 to the deque, the last 5 frames of it make up the observations of a new
        #                     transition (meaning n+1 transitions are stored)
        #
        # !!! HOWEVER, THIS FAILS !!! when we consider that, if in the sequence (x_{i},x_{i+1},x_{i+2},x_{i+3},x_{i+4}),
        # a terminal state is reached in frame x{i+3}. In this case (x_{i},x_{i+1},x_{i+2},x_{i+3}) and
        # (x_{i+1},x_{i+2},x_{i+3},x_{i+4}) do not make up a valid transition. Therefore, there will be a need to
        # validate each tuple of observations as part of valid transitions.
        #
        # What this means is that, in practise, not exactly 'size' transitions will be stored in the replay (at least
        # not valid ones).
        self._phi = deque(maxlen=size+4)

        self._actions = deque(maxlen=size)
        self._rewards = deque(maxlen=size)
        self._not_done = deque(maxlen=size)
        self._device = torch.device(device)

    def append(self, transition):
        prev_phi = transition[0]
        action = transition[1]
        reward = transition[2]
        next_phi = transition[3]
        done = transition[4]

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
            for i in range(4):
                # Store all the frames from previous phi
                self._phi.append(compress_frames(prev_phi[None, :, i, :]))

                # Add None values to the other deques so that the indices still match.
                self._actions.append(None)
                self._rewards.append(None)
                self._not_done.append(None)

        # add only the last frame of the next observation, since the first 3 are repeated from prev_phi
        self._phi.append(compress_frames(next_phi[None, :, 3, :]))
        self._actions.append(action)
        self._rewards.append(reward)
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

            #while not all([self._not_done[j] for j in range(i-4, i)]): # no longer valid.TODO:remove line after testing

            # While the transition is not valid, look for a new one
            while self._not_done[i] is None:
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
