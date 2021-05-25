from dqn.agents.dqn_agent import DQNAtariAgent
import os

env = "PongDeterministic-v4"
name = "test_pong"
directory = os.getcwd()+"/Agents/"

print("hello")
agent = DQNAtariAgent(env, name,
                      save_after_steps=100_000,
                      max_steps=10_000_000,
                      max_time=300,
                      replay_memory_size=1_000_000,
                      feed_back_after_episodes=10)

agent.populate_replay_memory(50_000)
agent.learn()
agent.play()

