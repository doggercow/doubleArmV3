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

    def __init__(self, capacity=10_000):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, float(done)))

    def sample(self, batch_size):
        batch                             = random.sample(self.buf, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        return (
            torch.tensor(np.array(obs),      dtype=torch.float32),  # (B, 6)
            torch.tensor(actions,            dtype=torch.long),      # (B,)
            torch.tensor(rewards,            dtype=torch.float32),   # (B,)
            torch.tensor(np.array(next_obs), dtype=torch.float32),  # (B, 6)
            torch.tensor(dones,              dtype=torch.float32),   # (B,)
        )

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
    GAMMA       = 0.99    # discount factor
    LR          = 1e-3    # Adam learning rate
    BATCH_SIZE  = 64      # transitions sampled per learning step
    EPS_START   = 1.0     # initial exploration rate
    EPS_END     = 0.05    # minimum exploration rate
    EPS_DECAY   = 0.995   # multiplied by epsilon at the end of each episode
    TARGET_SYNC = 200     # policy → target copy every N environment steps

    def __init__(self, n_obs, n_actions):
        self.n_actions  = n_actions
        self.epsilon    = self.EPS_START
        self.total_steps = 0

        # two identical networks — policy net is trained, target net is frozen
        self.policy_net = DQN(n_obs, n_actions)
        self.target_net = DQN(n_obs, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()           # target net is never directly trained

        self.optimizer  = torch.optim.Adam(self.policy_net.parameters(), lr=self.LR)
        self.loss_fn    = nn.MSELoss()
        self.buffer     = ReplayBuffer()

    # ── action selection ──────────────────────────────────────────────
    def select_action(self, obs):
        """Epsilon-greedy: random action with prob epsilon, greedy otherwise."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # (1, 6)
            q_values = self.policy_net(obs_t)                             # (1, 4)
            return q_values.argmax(dim=1).item()

    # ── store transition + maybe sync target ──────────────────────────
    def store(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)
        self.total_steps += 1

        if self.total_steps % self.TARGET_SYNC == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # ── one gradient update ───────────────────────────────────────────
    def learn(self):
        """
        Sample a mini-batch, compute the Bellman target, backprop, return loss.
        Returns None if the buffer isn't full enough yet.
        """
        if len(self.buffer) < self.BATCH_SIZE:
            return None

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.BATCH_SIZE)

        # Q(s, a)  — only the Q-value for the action that was actually taken
        q_values = self.policy_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # r + γ · max_a' Q_target(s', a')   (zeroed out on terminal transitions)
        with torch.no_grad():
            max_next_q = self.target_net(next_obs).max(dim=1).values
            targets    = rewards + self.GAMMA * max_next_q * (1.0 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # ── epsilon decay (call once per episode) ─────────────────────────
    def decay_epsilon(self):
        self.epsilon = max(self.EPS_END, self.epsilon * self.EPS_DECAY)

    # ── save / load ───────────────────────────────────────────────────
    def save(self, path="model.pth"):
        torch.save(self.policy_net.state_dict(), path)

    @classmethod
    def load(cls, path, n_obs, n_actions):
        agent = cls(n_obs, n_actions)
        agent.policy_net.load_state_dict(torch.load(path, weights_only=True))
        agent.policy_net.eval()
        agent.epsilon = 0.0   # pure greedy when evaluating
        return agent
