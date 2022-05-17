from dqn.agents.networks.atari_multi_distillation_network import AtatariMultiDistillationNetwork
from dqn.agents.dqn_agent import DQNAtariAgent
from dqn.agents.double_dqn_agent import DoubleDQNAtariAgent
from dqn.agents.multi_distilled_agent import MultiDistilledAgent
from dqn.memory.distilation_replay_memory_atari import DistillationReplayMemoryAtari
from dqn.policies.train_eval_policy import TrainEvalPolicy

if __name__ == '__main__':

    name = "multi_distillation_30ep_seed73"
    env1 = "QbertDeterministic-v4"
    env2 = "PongDeterministic-v4"
    env3 = "FreewayDeterministic-v4"

    teacher1 = DoubleDQNAtariAgent.load(env1, "D:\\dqn_atari_runs\\Agents\\qbert_double", import_replay=False, policy=TrainEvalPolicy())
    teacher2 = DQNAtariAgent.load(env2, "D:\\dqn_atari_runs\\Agents\\pong", import_replay=False, policy=TrainEvalPolicy())
    teacher3 = DoubleDQNAtariAgent.load(env3, "D:\\dqn_atari_runs\Agents\\freeway_double\\best", import_replay=False, policy=TrainEvalPolicy())
    teachers = [teacher1, teacher2, teacher3]
    envs = [t.env for t in teachers]

    #episodes = int(input("For how many episodes are you planning to train the agent? (advised: 30)"))
    #device = input("Which device are you planning to use? (cuda:1 / cuda:0 / cuda)")

    """net = AtatariMultiDistillationNetwork([teacher1.n_actions, teacher2.n_actions, teacher3.n_actions],
                                          final_transformation=lambda vals: log_softmax(vals, dim=1))"""

    student = MultiDistilledAgent(envs, teachers, AtatariMultiDistillationNetwork,
                                  {"number_of_actions": [teacher1.n_actions, teacher2.n_actions, teacher3.n_actions]},
                                  memory_type=DistillationReplayMemoryAtari,
                                  memory_size=10 * 216_000,
                                  temperature=0.01,
                                  seed=73,
                                  optimizer_parameters={"lr": 0.001, "alpha": 0.95, "eps": 0.01}
                                  )

    print("learning started")

    student.learn(save_dir="DistilledAgents/atari_qbert_pong_freeway/debug", save_replay=True, verbose=True,
                  max_epochs=100, updates_per_episode=500, frames_per_episode=10_000)
    print("learning ended")
    #input("press enter to see the student play")
    #student.select_task(0)
    #student.play()
    input("press enter to see the student play 1")
    student.select_task(1)
    student.play()
    #input("press enter to see the student play 2")
    #student.select_task(2)
    #student.play()
