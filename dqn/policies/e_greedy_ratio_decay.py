from dqn.policies.e_greedy import EGreedy


class EGreedyRatioDecay(EGreedy):
    """ Epsilon-greedy policy whose epsilon decays to r*epsilon every time.
    decay should be a number between 0 and 1."""

    def __init__(self, epsilon=1, min_epsilon=0.1, ratio_of_decay=0.995, n_actions=None):
        super().__init__(epsilon, n_actions=n_actions)
        self.decay = ratio_of_decay
        self.min_epsilon = min_epsilon

    def choose_action(self, Q=None):
        """Q is an tensor with shape (N,) or an ndarray, containing the Q values of the N possible actions"""
        action_index = super().choose_action(Q=Q)

        # update the epsilon in the initial decay phase
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay

        return action_index
