import torch.nn.functional as F
import torch.nn as nn


class LinearDQN(nn.Module):

    def __init__(self, n_features, n_actions, n_hidden_layers=2, n_units_per_layer=64):
        super().__init__()
        if n_hidden_layers < 1:
            raise ValueError("There must be at least one hidden layer.")

        self.n_features = n_features

        # fully connected layers
        self.fc1 = nn.Linear(n_features, n_units_per_layer)
        self.hidden = []
        for _ in range(n_hidden_layers-1):
            self.hidden.append(nn.Linear(n_units_per_layer, n_units_per_layer))
        self.fc_final = nn.Linear(n_units_per_layer, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, self.n_features)))
        for fc in self.hidden:
            x = F.relu(fc(x))
        return self.fc_final(x)

    def to(self, device):
        super().to(device)
        [fc.to(device) for fc in self.hidden]
        return self
