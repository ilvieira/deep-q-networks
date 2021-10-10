from dqn.policies.e_greedy import EGreedy


class EGreedyLinearDecay(EGreedy):
    """ Epsilon-greedy policy"""

    def __init__(self, epsilon=1, min_epsilon=0.1, steps_of_decay=1_000_000, n_actions=None):
        super().__init__(epsilon, n_actions=n_actions)
        self.decay = (epsilon-min_epsilon)/steps_of_decay
        self.min_epsilon = min_epsilon

    def choose_action(self, Q=None):
        """Q is an tensor with shape (N,) or an ndarray, containing the Q values of the N possible actions"""
        action_index = super(EGreedyLinearDecay, self).choose_action(Q=Q)

        # update the epsilon in the initial decay phase
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.decay

        return action_index
