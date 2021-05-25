from dqn.policies.policy import Policy
from dqn.policies.e_greedy_linear_decay import EGreedyLinearDecay
from dqn.policies.e_greedy import EGreedy


class AtariDQNPolicy(Policy):

    def __init__(self, eval_policy=EGreedy(epsilon=0.05),
                 train_policy=EGreedyLinearDecay(epsilon=1, min_epsilon=0.1, steps_of_decay=1_000_000),
                 start_training=True):

        self._train_policy = train_policy
        self._eval_policy = eval_policy
        self._training = start_training

    def train(self):
        self._training = True

    def eval(self):
        self._training = False

    def choose_action(self, Q=None):
        policy = self._train_policy if self._training else self._eval_policy
        return policy.choose_action(Q=Q)

