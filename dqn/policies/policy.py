from abc import ABC, abstractmethod


class Policy(ABC):

    @abstractmethod
    def choose_action(self, Q=None) -> object:
        """Q should be an ndarray or a torch tensor containing in each of its n components the q-value for the
        corresponding action, given the current state of the environment. n denotes the number of actions available for
        the agent to perform."""
        pass

    # TODO: these two methods were added as a quick fix for an issue. In the future they shall be removed:
    def train(self):
        return

    def eval(self):
        return