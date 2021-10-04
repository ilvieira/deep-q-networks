from abc import ABC, abstractmethod
import numpy as np


class Policy(ABC):

    @abstractmethod
    def choose_action(self, Q:np.ndarray=None) -> object:
        pass
