from dqn.policies.e_greedy import EGreedy


class EGreedyStepDecay(EGreedy):
    """ Epsilon-greedy policy whose epsilon decays step_value every deacy_frequency times it chooses an action."""

    def __init__(self, epsilon=1, min_epsilon=0.1, step_value=0.1, decay_frequency=1_000_000, n_actions=None):
        super().__init__(epsilon, n_actions=n_actions)
        self.decay = step_value
        self.min_epsilon = min_epsilon
        self.decay_frequency = decay_frequency
        self.timesteps_before_decay = decay_frequency

    def choose_action(self, Q=None):
        """Q is an tensor with shape (N,) or an ndarray, containing the Q values of the N possible actions"""
        action_index = super().choose_action(Q=Q)
        self.decay_frequency -= 1

        # update the epsilon in the initial decay phase
        if self.epsilon > self.min_epsilon and self.decay_frequency == 0:
            self.epsilon -= self.decay
            self.epsilon = max(self.min_epsilon, self.epsilon)
            self.decay_frequency = self.timesteps_before_decay

        return action_index
