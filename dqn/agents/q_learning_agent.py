from dqn.agents.agent import Agent
from dqn.policies.e_greedy_linear_decay import EGreedyLinearDecay
from dqn.policies.e_greedy import EGreedy
import numpy as np


class QLearningAgent(Agent):
    """Uses Q-learning to obtain the Q-function as an array of dimensions (number of states)x(number of actions)"""

    def __init__(self, env, n_states, n_actions, gamma, alpha, obs_to_state_id=lambda x: x, episodic=False,
                 train_policy=EGreedyLinearDecay(epsilon=1,min_epsilon=0.05, steps_of_decay=5000),
                 eval_policy=EGreedy(0)):
        """
        - obs_to_state_id should be a function that, given a state observation obs returns the corresponding state id
        as the respective value.
        - episodic is whether we count an iteration as a full game episode or as a step taken in the environment.
        """
        super().__init__(env, n_actions)
        self.gamma = gamma
        self.alpha = alpha
        self.q_function = np.zeros((n_states, n_actions))
        self.obs_to_state_id = obs_to_state_id
        self.train_policy = train_policy
        self.eval_policy = eval_policy
        self.episodic = episodic

        self.train()
        # TODO: in the future experiment programing the Agent class to train() on __init__ instead of this subclass.
        #  This will imply checking if there is no impact in the other subclasses of Agent (including DQNAgent).

    def train(self):
        super().train()
        self.policy = self.train_policy

    def eval(self):
        super().eval()
        self.policy = self.eval_policy

    def action(self, observation):
        state = self.obs_to_state_id(observation)
        return self.policy.choose_action(self.q_function[state, :])

    def learn(self, iterations):
        self.train()
        state = self.obs_to_state_id(self.env.reset())

        iteration = 0
        while iteration < iterations:
            action = self.policy.choose_action(self.q_function[state, :])
            obs, reward, done, _ = self.env.step(action)
            next_state = self.obs_to_state_id(obs)

            q_t = self.q_function[state, action]
            next_q_opt = self.q_function[next_state, :].max() if not done else 0

            # q-function update
            self.q_function[state, action] = q_t + self.alpha * (reward + self.gamma * next_q_opt - q_t)

            # update state
            if not done:
                state = next_state
            else:
                state = self.env.reset()
                if self.episodic:
                    iteration += 1

            # if each timestep counts as an iteration:
            if not self.episodic:
                iteration += 1
