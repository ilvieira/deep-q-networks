from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
from dqn.agents.q_learning_agent import QLearningAgent
from dqn.policies.e_greedy_linear_decay import EGreedyLinearDecay

"""This a simple test to make sure that the QLearningAgent is working properly. It is for a very simple example and 
randomness was removed, since even a human player has trouble reaching the goal if it is not."""

env = FrozenLakeEnv(is_slippery=False)
agent = QLearningAgent(env, env.nS, env.nA, 0.9, 0.5,
                       train_policy=EGreedyLinearDecay(epsilon=1,min_epsilon=0.05, steps_of_decay=100))

agent.learn(500)
agent.play()
