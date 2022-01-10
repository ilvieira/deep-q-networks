import torch.nn.functional as F
import torch.nn as nn


class LinearDQN(nn.Module):

    def __init__(self, n_features, number_of_actions):
        super().__init__()
        self.n_features = n_features

        # fully connected layers
        self.fc1 = nn.Linear(n_features, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, number_of_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, self.n_features)))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
