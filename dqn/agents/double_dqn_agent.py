from dqn.agents.dqn_agent import DQNAgent, DQNAtariAgent
from dqn.policies.train_eval_policy import TrainEvalPolicy
from dqn.torch_extensions import clip_mse3
import torch


class DoubleDQNAgent(DQNAgent):
    def __init__(self, env, replay, n_actions, net_type, net_parameters,
                 minibatch_size=32,
                 optimizer=torch.optim.RMSprop,
                 C=10_000,
                 update_frequency=1,
                 gamma=0.99,
                 loss=clip_mse3,
                 policy=TrainEvalPolicy(),
                 populate_policy=None,
                 seed=0,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 optimizer_parameters=None,
                 avg_loss_per_steps=10_000):
        super().__init__(env, replay, n_actions, net_type, net_parameters, minibatch_size=minibatch_size,
                         optimizer=optimizer, C=C, update_frequency=update_frequency, gamma=gamma, loss=loss,
                         policy=policy, populate_policy=populate_policy, seed=seed,
                         device=device, optimizer_parameters=optimizer_parameters,
                         avg_loss_per_steps=avg_loss_per_steps)

    def update_net(self, r, not_done, next_phi):
        with torch.no_grad():
            actions = self.Q_target(next_phi).max(axis=1, keepdim=True).indices.view(self.minibatch_size)
            q_estimate = torch.zeros(self.minibatch_size).to(self.device)
            q_phi = self.Q(next_phi)
            for i in range(self.minibatch_size):
                q_estimate[i] = q_phi[i, actions[i]]
            y = (r + self.gamma * not_done * q_estimate)
        return y


class DoubleDQNAtariAgent(DQNAtariAgent):
    def __init__(self, env, minibatch_size=32,
                 replay_memory_size=1_000_000,
                 C=10_000,
                 gamma=0.99,
                 loss=clip_mse3,
                 update_frequency=4,
                 seed=0,
                 device=("cuda" if torch.cuda.is_available() else "cpu"),
                 optimizer_parameters=None,
                 policy=TrainEvalPolicy()):
        super().__init__(env, minibatch_size=minibatch_size, replay_memory_size=replay_memory_size, C=C, gamma=gamma,
                         loss=loss, update_frequency=update_frequency, seed=seed, device=device,
                         optimizer_parameters=optimizer_parameters, policy=policy)

    def update_net(self, r, not_done, next_phi):
        with torch.no_grad():
            actions = self.Q_target(next_phi).max(axis=1, keepdim=True).indices.view(self.minibatch_size)
            q_estimate = torch.zeros(self.minibatch_size).to(self.device)
            q_phi = self.Q(next_phi)
            for i in range(self.minibatch_size):
                q_estimate[i] = q_phi[i, actions[i]]
            y = (r + self.gamma * not_done * q_estimate)
        return y
