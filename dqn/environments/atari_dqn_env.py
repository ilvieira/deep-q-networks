from baselines.common.atari_wrappers import EpisodicLifeEnv, NoopResetEnv, FireResetEnv
from typing import Union
from yaaf.environments.wrappers import NvidiaAtari2600Wrapper
from gym import Wrapper, Env
import numpy as np


class AtariDQNEnv(Wrapper):

    def __init__(self, env: Union[Env, Wrapper, str]):
        # starts in train mode
        """ Deep mind did not use the FireResetEnv wrapper. It turns out that the 0.05 random actions are enough to make
        sure that the game always starts."""

        self._train_env = EpisodicLifeEnv(NvidiaAtari2600Wrapper(env, channels_first=True))
        self._eval_env = NoopResetEnv(NvidiaAtari2600Wrapper(env, max_reward_clip=np.inf, min_reward_clip=-np.inf,
                                                             channels_first=True))

        super().__init__(self._train_env)

    def train(self):
        self.env = self._train_env

    def eval(self):
        self.env = self._eval_env

    def restart(self):
        # starts the game from the beginning (with all the lives)
        self.env.was_real_done = True
        self.reset()
