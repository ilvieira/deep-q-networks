import torch.nn.functional as F
import torch.nn as nn


class CustomLinearDQN(nn.Module):

    def __init__(self, n_features, n_actions, n_hidden_layer_units=(64, 64), activation_functions=(F.relu, F.relu)):
        """
         - n_hidden_layer_units should be an iterable object with the sequence of the number of units in the hidden
             layers
         - activation_functions should be an iterable object with each component being the activation function for the
             respective hidden layer from n_hidden_layers
        """
        super().__init__()
        n_hidden_layers = len(n_hidden_layer_units)
        if n_hidden_layers < 1:
            raise ValueError("There must be at least one hidden layer.")
        if len(activation_functions)!= n_hidden_layers:
            raise ValueError("There must be exactly one activation function for each hidden layer.")

        self.n_features = n_features
        self.activation_functions = activation_functions

        # fully connected layers
        self.fc1 = nn.Linear(n_features, n_hidden_layer_units[0])
        self.hidden = []
        for i in range(1, n_hidden_layers):
            self.hidden.append(nn.Linear(n_hidden_layer_units[i-1], n_hidden_layer_units[i]))
        self.fc_final = nn.Linear(n_hidden_layer_units[-1], n_actions)

    def forward(self, x):
        x = self.activation_functions[0](self.fc1(x.view(-1, self.n_features)))
        for i, fc in enumerate(self.hidden):
            x = self.activation_functions[i+1](fc(x))
        return self.fc_final(x)

    def to(self, device):
        super().to(device)
        [fc.to(device) for fc in self.hidden]
        return self
