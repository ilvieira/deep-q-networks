from dqn.agents.dqn_agent import DQNAtariAgent
import os
import torch

env = "PongDeterministic-v4"
save_dir = os.getcwd()+"/Agents/test_pong"

agent = DQNAtariAgent(env, replay_memory_size=1_000_000)

agent.populate_replay_memory(100)
agent.learn(save_dir, save_after_steps=600, max_steps=10_000_000, max_time=30, feedback_after_episodes=1)
agent.play()

new_dqn_agent = DQNAtariAgent.load(env, save_dir, import_replay=True)
new_dqn_agent.learn(save_dir, save_after_steps=600, max_steps=10_000_000, max_time=30, feedback_after_episodes=1)

