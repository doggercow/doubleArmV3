# Responsible for defining the neural network architecture used by the DQN agent.

import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Input  : observation vector  (N_OBS = 8 floats)
    Output : Q-value for each action  (N_ACTIONS = 8 floats)

    Architecture: three hidden layers of 128 neurons with ReLU activations.
    The output layer is linear — Q-values are unbounded.
    """

    # Builds the network layers given the observation size, action count, and hidden layer width.
    def __init__(self, n_obs, n_actions, hidden=128):
        super().__init__()

        self.layer1 = nn.Linear(n_obs,    hidden)    # input layer: maps observation vector to first hidden layer
        self.layer2 = nn.Linear(hidden,   hidden)    # second hidden layer
        self.layer3 = nn.Linear(hidden,   hidden)    # third hidden layer
        self.layer4 = nn.Linear(hidden,   n_actions) # output layer: produces one Q-value per action (unbounded)

        self.relu = nn.ReLU()   # shared ReLU activation applied after each hidden layer

    # Runs a forward pass through the network and returns Q-values for every action.
    def forward(self, x):
        x = self.relu(self.layer1(x))   # first hidden layer with ReLU
        x = self.relu(self.layer2(x))   # second hidden layer with ReLU
        x = self.relu(self.layer3(x))   # third hidden layer with ReLU
        x = self.layer4(x)              # linear output — one Q-value per action
        return x                        # shape: (batch, n_actions)
