import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import pickle
import time
import random as rnd
import matplotlib.pyplot as plt

from dqn.memory.distillation_replay_memory import DistillationReplayMemory as Memory
from dqn.agents.agent import Agent
from dqn.policies.e_greedy import EGreedy
from dqn.environments.atari_dqn_env import AtariDQNEnv
from yaaf.environments.wrappers import NvidiaAtari2600Wrapper
from torch.nn.functional import softmax


class MultiDistilledAgent(Agent):
    """ the authors did not mention minibatch size"""
    def __init__(self, envs, teachers, net_type, net_parameters,
                 memory_type=Memory,
                 memory_size=10 * 216_000,
                 minibatch_size=32,
                 optimizer=torch.optim.RMSprop,
                 loss=lambda x, y: F.kl_div(x, y, reduction='sum'),
                 sample_policy=EGreedy(epsilon=0.05),
                 policy=EGreedy(epsilon=0.05),
                 temperature=0.01,
                 seed=0,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 optimizer_parameters=None):
        if len(envs) != len(teachers):
            raise ValueError(f"The number of envs ({len(envs)}) must match the number of teachers ({len(teachers)}).")

        self.device = torch.device(device)
        self.set_seed(seed)
        self.minibatch_size = minibatch_size
        self.n_frames = 0
        self.n_epochs = 0

        # policy used by the agent while playing
        self.policy = policy
        # policy used to choose the action of the teacher while populating the replay memory
        self.sample_policy = sample_policy

        self.net_parameters = net_parameters
        self.net = net_type(**self.net_parameters).to(self.device)
        self.setup_optimizer(optimizer, optimizer_parameters)
        self.loss = loss
        self.temperature = temperature

        # initialize and store the environments, teachers and replay memories and select the first task by default
        self.n_tasks = len(envs)
        self.envs = envs
        self.initialize_teachers(teachers)
        self.memories = [memory_type(memory_size) for _ in range(self.n_tasks)]
        self.select_task(0)
        super().__init__(self.env, self.env.num_actions)

        self._losses = [[] for _ in range(self.n_tasks)]
        self._epochs = [[] for _ in range(self.n_tasks)]

    # ================================================================================================================
    # Setup Methods
    # ================================================================================================================

    def set_seed(self, seed: int):
        self.seed = seed
        torch.manual_seed(seed)
        rnd.seed(seed)
        np.random.seed(seed)

    def setup_optimizer(self, optimizer, optimizer_parameters):
        self.optimizer_parameters = optimizer_parameters if optimizer_parameters is not None \
            else {"lr": 0.00025, "alpha": 0.95, "eps": 0.01}
        self.optimizer = optimizer(self.net.parameters(), **self.optimizer_parameters)

    def initialize_teachers(self, teachers):
        self.teachers = teachers
        # turn on eval mode for each teacher
        for teacher in teachers:
            teacher.eval()

    # ================================================================================================================
    # Agent Methods
    # ================================================================================================================

    def action(self, observation: np.ndarray):
        """Chooses an action given an observation"""
        phi = torch.tensor(self.expand_obs(observation)).float()

        # phi is added to the gpu so that Q can make a prediction on it
        with torch.no_grad():
            q_vals = self.net(phi.to(self.device))
            action = self.policy.choose_action(q_vals.detach().cpu())
            return action

    # ================================================================================================================
    # Distillation Specific Methods
    # ================================================================================================================

    def select_task(self, task_id):
        self.task_id = task_id
        self.env = self.envs[task_id]
        self.teacher = self.teachers[task_id]
        self.memory = self.memories[task_id]
        self.net.choose_task(task_id)

    def populate_replay(self, n_frames):
        for task in range(self.n_tasks):
            self.select_task(task)
            print(f"populating replay {task}")
            obs = self.expand_obs(self.env.reset())
            done = False

            for i in range(n_frames):
                phi = torch.tensor(obs).float()

                # phi is added to the gpu so that Q can make a prediction on it
                with torch.no_grad():
                    q_vals = self.teacher.Q(phi.to(self.device))
                action = self.sample_policy.choose_action(q_vals.detach().cpu())

                transition = (obs, q_vals, done)
                self.memory.append(transition)

                next_obs, _, done, _ = self.env.step(action)
                next_obs = self.expand_obs(next_obs)

                # update obs
                obs = next_obs if not done else self.expand_obs(self.env.reset())

    def update_student(self, n_updates):
        losses = [0 for _ in range(self.n_tasks)]
        updates = [0 for _ in range(self.n_tasks)]
        for i in range(n_updates):
            # select random task
            self.select_task(rnd.randrange(self.n_tasks))

            obs, q_vals = self.memory.sample(self.minibatch_size, self.device)
            t_values = softmax(q_vals / self.temperature, dim=1)
            self.optimizer.zero_grad()
            s_values = self.net(obs)
            loss = self.loss(s_values, t_values)
            losses[self.task_id] += loss.detach().item()
            updates[self.task_id] += 1
            loss.backward()
            self.optimizer.step()
        avg_losses = [losses[t] / updates[t] for t in range(self.n_tasks)]
        return avg_losses

    def learn(self, save_dir, save_replay=False, verbose=True, updates_per_episode=10_000,
              max_epochs=1_000_000_000, max_frames=108_000_000, frames_per_episode=216_000):

        save_params = {"save_dir": save_dir, "save_replay": save_replay}
        self.save(**save_params)

        start = time.time()
        if verbose:
            print("... learning stage began ...")

        while self.n_frames < max_frames and self.n_epochs < max_epochs:
            # populate the replay and train for that task
            self.populate_replay(frames_per_episode)
            avg_losses = self.update_student(updates_per_episode)
            self.n_frames += updates_per_episode

            if verbose:
                print(f"EPOCH {self.n_epochs} COMPLETE:")

            # store the statistics and the resulting_agent
            for t in range(self.n_tasks):
                self._losses[t].append(avg_losses[t])
                self._epochs[t].append(self.n_epochs)
                self.save_stats(save_dir, self._losses[t], t)
                self.save_plot(save_dir, self._epochs[t], self._losses[t], t)
                if verbose:
                    print(f" - task {t}: avg loss: {avg_losses[t]}")
            self.save(**save_params)

            if verbose:
                print(f"  time: {time.time() - start}s")
            self.n_epochs += 1

    # ================================================================================================================
    # Statistics and Plot Methods
    # ================================================================================================================

    def save_stats(self, agent_dir, loss_per_epoch, task_id):
        agent_dir = agent_dir if (agent_dir[-1] == '/' or agent_dir[-1] == '\\') else agent_dir + '/'
        filename = agent_dir + f"stats{task_id}.csv"
        with open(filename, "w") as stats_file:
            stats_file.write("Loss")
        with open(filename, "a") as stats_file:
            for i in range(len(loss_per_epoch)):
                stats_file.write(f"\n{loss_per_epoch[i]}")

    def save_plot(self, agent_dir, epochs, losses, task_id, plot_name=None):
        agent_dir = agent_dir if (agent_dir[-1] == '/' or agent_dir[-1] == '\\') else agent_dir + '/'
        if plot_name is None:
            plot_name = agent_dir+"plot"

        x = epochs
        y = losses

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel="epochs", ylabel="loss value", title="Loss During the Learning Stage")
        ax.grid()
        fig.savefig(plot_name + str(task_id) + ".png")
        plt.close()

    # ================================================================================================================
    # Persistence Methods
    # ================================================================================================================
    def get_checkpoint(self):
        parameters = dict()
        parameters["n_tasks"] = self.n_tasks
        parameters["seed"] = self.seed
        parameters["temperature"] = self.temperature
        parameters["minibatch_size"] = self.minibatch_size
        parameters["policy"] = self.policy
        parameters["sample_policy"] = self.sample_policy
        parameters["n_epochs"] = self.n_epochs
        parameters["n_frames"] = self.n_frames

        # stats
        parameters["_losses"] = self._losses
        parameters["_epochs"] = self._epochs

        # checkpoint for the net
        parameters["net_type_name"] = self.net.__class__.__name__
        parameters["net_parameters"] = self.net_parameters
        parameters["net_state_dict"] = self.net.state_dict()

        # checkpoint for the optrimizer
        parameters["optimizer_name"] = self.optimizer.__class__.__name__
        parameters["optimizer_parameters"] = self.optimizer_parameters
        parameters["optimizer_state_dict"] = self.optimizer.state_dict()

        return parameters

    def save(self, save_dir, save_replay=True):
        agent_dir = save_dir if (save_dir[-1] == '/' or save_dir[-1] == '\\') else save_dir + '/'
        if not os.path.exists(agent_dir):
            os.makedirs(agent_dir)

        checkpoint = self.get_checkpoint()
        torch.save(checkpoint, agent_dir + "agent.tar")

        if save_replay:
            for i in range(self.n_tasks):
                with open(agent_dir+f"replay_{i}.p", "wb") as replay_file:
                    pickle.dump(self.memories[i], replay_file)

    @classmethod
    def load(cls, envs, teachers, agent_dir, net_type, loss=lambda x, y: F.kl_div(x, y, reduction='sum'),
             import_replay=True, optimizer=torch.optim.RMSprop,
             device=("cuda" if torch.cuda.is_available() else "cpu"), replay_type=Memory, replay_size=10 * 216_000):
        agent_dir = agent_dir if (agent_dir[-1] == '/' or agent_dir[-1] == '\\') else agent_dir + '/'
        checkpoint = torch.load(agent_dir + "agent.tar", map_location=device)
        if checkpoint["n_tasks"] != len(envs) or len(envs) != len(teachers):
            raise ValueError(f"Both the number of teachers ({len(teachers)}) and the number of envs ({len(envs)}) must "
                             f"match the number of tasks of the original agent ({checkpoint['n_tasks']}).")
        if net_type.__name__ != checkpoint["net_type_name"]:
            raise ValueError(f"The network for this agent is of the type '{checkpoint['Q_class_name']}', but there"
                             + f" was attempt to load it using a net of the type '{net_type}'.")
        if optimizer.__name__ != checkpoint["optimizer_name"]:
            raise ValueError(f"The optimizer for this agent is of the type '{checkpoint['optimizer_class_name']}', but"
                             + f" there was attempt to load it using an optimizer of the type '{optimizer}'.")

        agent = MultiDistilledAgent(envs, teachers, net_type, checkpoint["net_parameters"],
                                    memory_type=replay_type, memory_size=replay_size,
                                    minibatch_size=checkpoint["minibatch_size"],
                                    optimizer=optimizer,
                                    loss=loss,
                                    sample_policy=checkpoint["sample_policy"],
                                    policy=checkpoint["policy"],
                                    temperature=checkpoint["temperature"],
                                    seed=checkpoint["seed"],
                                    device=device,
                                    optimizer_parameters=checkpoint["optimizer_parameters"])

        agent.net.load_state_dict(checkpoint["net_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if import_replay:
            for i in range(checkpoint["n_tasks"]):
                with open(agent_dir + f"replay_{i}.p", "rb") as replay_file:
                    agent.memories = []
                    agent.memories.append(pickle.load(replay_file))

        #agent.n_epochs = checkpoint["n_epochs"]
        #agent.n_frames = checkpoint["n_frames"]
        agent._losses = checkpoint["_losses"]
        agent._epochs = checkpoint["_epochs"]
        return agent

    # ================================================================================================================
    # Other Auxiliary Methods
    # ================================================================================================================

    def expand_obs(self, screen: np.ndarray):
        return np.expand_dims(screen, axis=0)


"""class MultiDistilledAgentAtari(MultiDistilledAgent):
    #TODO : this class with the right parameters. Also, still have to checkout the replay memory for distillation
    def __init__(self, envs, teachers, net_type, net_parameters,
                 memory_type=Memory,
                 memory_size=10 * 216_000,
                 minibatch_size=32,
                 optimizer=torch.optim.RMSprop,
                 loss=lambda x, y: F.kl_div(x, y, reduction='sum'),
                 sample_policy=EGreedy(epsilon=0.05),
                 policy=EGreedy(epsilon=0.05),
                 temperature=0.01,
                 seed=0,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 optimizer_parameters=None):
        envs = [AtariDQNEnv(env) for env in envs]
        super().__init__()
        self._frames_per_hour = 216_000"""


