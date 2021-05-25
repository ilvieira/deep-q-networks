from abc import ABC, abstractmethod
import numpy as np


class Policy(ABC):

    @abstractmethod
    def choose_action(self, Q=None) -> object:
        pass
