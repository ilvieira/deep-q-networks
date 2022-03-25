from dqn.agents.dqn_agent import DQNAgent
from dqn.memory.dqn_replay_memory import DQNReplayMemory
from dqn.policies.train_eval_policy import TrainEvalPolicy, EGreedyLinearDecay, EGreedy
from dqn.agents.networks.custom_linear_dqn import CustomLinearDQN
from examples.pursuit.environment.Pursuit import Pursuit
import os
import torch
from torch.nn.functional import mse_loss

env = Pursuit(teammates="teammate aware", features="relative agent")
save_dir = os.getcwd()+"/Agents/pursuit_test"
replay = DQNReplayMemory(100_000)
policy = TrainEvalPolicy(eval_policy=EGreedy(epsilon=0),
                         train_policy=EGreedyLinearDecay(epsilon=0.5, min_epsilon=0.05, steps_of_decay=5_000))
agent = DQNAgent(env, replay, env.num_actions, CustomLinearDQN,
                 {"n_features": env.num_features, "n_actions": env.num_actions},
                 C=1, gamma=0.95, policy=policy,
                 optimizer=torch.optim.Adam, optimizer_parameters={"lr": 0.001}, loss=mse_loss)

# Create agent and populate the replay memory
agent.populate_replay_memory(10_000)

# Learning stage
agent.learn(save_dir, save_after_steps=1_000, max_steps=7500, max_time=24*3600, feedback_after_episodes=50)

input(" Press enter to watch the agent play")
agent.play()