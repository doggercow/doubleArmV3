# Responsible for the DQN agent: experience replay buffer, action selection, learning, and model persistence.

import random
import numpy as np
import torch
import torch.nn as nn
from collections import deque

from network import DQN


# ── Replay Buffer ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    """
    Stores (obs, action, reward, next_obs, done) tuples.
    Once full, the oldest transitions are overwritten automatically (deque).
    """

    # Initialises the buffer with a fixed maximum capacity using a circular deque.
    def __init__(self, capacity=600_000):
        self.buf = deque(maxlen=capacity)   # circular buffer that automatically drops oldest entries when full

    # Appends a single transition tuple to the replay buffer.
    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, float(done)))   # done cast to float for tensor arithmetic later

    # Randomly samples a batch of transitions and returns them as stacked tensors.
    def sample(self, batch_size):
        batch                                  = random.sample(self.buf, batch_size)   # random subset of stored transitions
        obs, actions, rewards, next_obs, dones = zip(*batch)                           # unzip into separate per-field tuples

        return (
            torch.tensor(np.array(obs),      dtype=torch.float32),   # (B, N_OBS) observation batch
            torch.tensor(actions,            dtype=torch.long),       # (B,) action indices
            torch.tensor(rewards,            dtype=torch.float32),    # (B,) scalar rewards
            torch.tensor(np.array(next_obs), dtype=torch.float32),   # (B, N_OBS) next-observation batch
            torch.tensor(dones,              dtype=torch.float32),    # (B,) terminal flags (1.0 = done)
        )

    # Returns the current number of transitions stored in the buffer.
    def __len__(self):
        return len(self.buf)


# ── DQN Agent ─────────────────────────────────────────────────────────────────
class DQNAgent:
    """
    Standard DQN with:
      - Epsilon-greedy exploration (epsilon decays each episode)
      - Experience replay (random mini-batch sampling)
      - Target network (periodically synced from the policy network)
      - Bellman TD update  →  loss = MSE(Q(s,a),  r + γ · max_a' Q_target(s',a'))
    """

    # ── hyper-parameters ──────────────────────────────────────────────
    GAMMA       = 0.99    # discount factor: how much future rewards are worth relative to immediate ones
    LR          = 3e-4    # Adam learning rate (tuned down from default; default was too high for this task)
    BATCH_SIZE  = 64      # number of transitions sampled from the buffer per gradient update
    EPS_START   = 1.0     # initial epsilon: agent acts fully randomly at the start of training
    EPS_END     = 0.05    # minimum epsilon: agent always keeps at least 5% random exploration
    EPS_DECAY   = 0.9975  # multiplicative decay applied to epsilon at the end of each episode
    TARGET_SYNC = 500     # copy policy network weights into target network every N environment steps

    # Builds both networks, the optimiser, the loss function, and the replay buffer.
    def __init__(self, n_obs, n_actions):
        self.n_actions   = n_actions       # number of discrete actions available to the agent
        self.epsilon     = self.EPS_START  # current exploration rate, starts at 1.0 and decays over training
        self.total_steps = 0               # cumulative environment steps taken across all episodes

        # two identical networks — policy net is trained, target net is frozen
        self.policy_net = DQN(n_obs, n_actions)    # network that is actively trained and used for action selection
        self.target_net = DQN(n_obs, n_actions)    # frozen copy of the policy net used to compute stable TD targets
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()   # target net is never directly trained; eval mode disables dropout/batchnorm

        self.optimizer  = torch.optim.Adam(self.policy_net.parameters(), lr=self.LR)   # optimiser for the policy network
        self.loss_fn    = nn.MSELoss()   # mean squared error between predicted Q-values and Bellman targets
        self.buffer     = ReplayBuffer() # experience replay buffer storing past transitions

    # ── action selection ──────────────────────────────────────────────

    # Returns a random action with probability epsilon, otherwise returns the greedy best action.
    def select_action(self, obs):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)   # random action for exploration

        with torch.no_grad():
            obs_t    = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)   # add batch dimension: (1, N_OBS)
            q_values = self.policy_net(obs_t)                                # forward pass yields Q-value per action: (1, N_ACTIONS)
            return q_values.argmax(dim=1).item()                             # index of the highest Q-value action

    # ── store transition + maybe sync target ──────────────────────────

    # Stores a transition in the replay buffer and syncs the target network every TARGET_SYNC steps.
    def store(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)
        self.total_steps += 1   # increment the global step counter used to trigger target network syncs

        if self.total_steps % self.TARGET_SYNC == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())   # copy policy weights into target network

    # ── one gradient update ───────────────────────────────────────────

    # Samples a mini-batch, computes the Bellman TD loss, and runs one backprop step. Returns None if buffer is too small.
    def learn(self):
        if len(self.buffer) < self.BATCH_SIZE:
            return None   # not enough experience yet; skip update

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.BATCH_SIZE)

        # Q(s, a) — only the Q-value for the action that was actually taken
        q_values = self.policy_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)   # shape: (B,)

        # r + γ · max_a' Q_target(s', a')   (zeroed out on terminal transitions)
        with torch.no_grad():
            max_next_q = self.target_net(next_obs).max(dim=1).values   # best Q-value in the next state from the frozen target net
            targets    = rewards + self.GAMMA * max_next_q * (1.0 - dones)   # Bellman target (0 if done, so no future reward)

        loss = self.loss_fn(q_values, targets)   # MSE between current Q predictions and Bellman targets

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()   # return the scalar loss value for logging

    # ── epsilon decay (call once per episode) ─────────────────────────

    # Multiplies epsilon by EPS_DECAY and clamps it to EPS_END to gradually reduce exploration.
    def decay_epsilon(self):
        self.epsilon = max(self.EPS_END, self.epsilon * self.EPS_DECAY)

    # ── save / load ───────────────────────────────────────────────────

    # Saves only the policy network's weights to a .pth file.
    def save(self, path="model.pth"):
        torch.save(self.policy_net.state_dict(), path)

    # Loads a saved policy network from disk and returns an agent in greedy eval mode (epsilon = 0).
    @classmethod
    def load(cls, path, n_obs, n_actions):
        agent = cls(n_obs, n_actions)
        agent.policy_net.load_state_dict(torch.load(path, weights_only=True))
        agent.policy_net.eval()
        agent.epsilon = 0.0   # pure greedy when evaluating — no random exploration
        return agent
