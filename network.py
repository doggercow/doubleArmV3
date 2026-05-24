import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Input  : observation vector  (N_OBS = 6 floats)
    Output : Q-value for each action  (N_ACTIONS = 8 floats)

    Architecture: two hidden layers of 128 neurons with ReLU activations.
    The output layer is linear — Q-values are unbounded.
    """

    def __init__(self, n_obs, n_actions, hidden=128):
        super().__init__()

        self.layer1 = nn.Linear(n_obs,    hidden)
        self.layer2 = nn.Linear(hidden,   hidden)
        self.layer3 = nn.Linear(hidden,   n_actions)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x                         # shape: (batch, n_actions)
