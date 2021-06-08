from dqn.agents.dqn_agent import DQNAtariAgent
import os

env = "PongDeterministic-v4"
save_dir = os.getcwd()+"/Agents/test_pong"

# Create agent and populate the replay memory
agent = DQNAtariAgent(env, replay_memory_size=1_000_000, C=10_000)
agent.populate_replay_memory(50_000)

# Learning stage
agent.learn(save_dir, save_after_steps=500_000, max_steps=3_000_000, max_time=24*3600, feedback_after_episodes=50)

input(" Press enter to watch the agent play")
agent.play()

# Load the agent from the directory where it was stored
new_dqn_agent = DQNAtariAgent.load(env, save_dir, import_replay=True)

input(" Press enter to watch the agent play")
new_dqn_agent.play()

# Resume learning
new_dqn_agent.learn(save_dir, save_after_steps=100_000, max_steps=3_500_000, feedback_after_episodes=5)

