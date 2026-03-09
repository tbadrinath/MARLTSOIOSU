"""
agent.py
--------
DQN agent for the Intelligent Urban Traffic Management System (IUTMS).

Architecture (per PRD):
    Input(state_dim) → FC1(64) → ReLU → FC2(32) → ReLU → Output(action_dim)

Exploration:  ε-greedy with exponential decay from 1.0 → 0.05.
Learning:     Experience replay with a fixed-size circular buffer.
              Target network updated every ``target_update`` training calls.
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Neural-network model
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    Fully-connected Q-network.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the observation (input) vector.
    action_dim : int
        Number of discrete actions (output heads).
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Fixed-size circular experience replay buffer.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    """

    def __init__(self, capacity: int = 10_000) -> None:
        self._buf: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition."""
        self._buf.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return a random mini-batch of *batch_size* transitions."""
        batch = random.sample(self._buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Independent DQN agent controlling a single traffic-signal intersection.

    Parameters
    ----------
    ts_id : str
        Traffic-signal identifier (used for checkpointing).
    state_dim : int
        Observation vector size.
    action_dim : int
        Number of discrete actions (green-phase combinations).
    lr : float
        Adam learning rate.
    gamma : float
        Discount factor (RL γ, distinct from the reward γ).
    epsilon_start : float
        Initial ε for ε-greedy exploration.
    epsilon_min : float
        Minimum (floor) ε value.
    epsilon_decay : float
        Multiplicative decay applied to ε after each ``learn()`` call.
    batch_size : int
        Mini-batch size for gradient updates.
    buffer_capacity : int
        Replay buffer size.
    target_update : int
        Number of ``learn()`` calls between target-network hard updates.
    device : str | None
        ``"cuda"`` or ``"cpu"``; auto-detected when *None*.
    """

    def __init__(
        self,
        ts_id: str,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 10_000,
        target_update: int = 100,
        device: Optional[str] = None,
    ) -> None:
        self.ts_id = ts_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma_rl = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Online and target networks
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss – numerically stable

        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self._learn_steps = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """
        ε-greedy action selection.

        Parameters
        ----------
        state : np.ndarray
            Current observation vector.

        Returns
        -------
        int
            Chosen action index.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Push a transition into the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self) -> Optional[float]:
        """
        Sample a mini-batch and perform one gradient-descent step.

        Returns
        -------
        float | None
            Training loss (or *None* if the buffer is too small).
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device)

        # Current Q-values for chosen actions
        current_q = self.q_net(states_t).gather(1, actions_t).squeeze(1)

        # Target Q-values (Bellman equation)
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1).values
            target_q = rewards_t + self.gamma_rl * max_next_q * (1.0 - dones_t)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Periodic hard update of target network
        self._learn_steps += 1
        if self._learn_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist model weights and training state to *path*."""
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "learn_steps": self._learn_steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Restore model weights and training state from *path*."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self._learn_steps = checkpoint["learn_steps"]
