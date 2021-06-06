from abc import ABC


class ReplayMemory(ABC):
    def __init__(self, size):
        # The size of a replay memory corresponds to the number of transitions it stores.
        self.__len = 0
        self.__max_len = size

    def append(self, transition):
        self.__len += 1
        self.__len = min(self.__len, self.__max_len)

    def __len__(self):
        return self.__len
