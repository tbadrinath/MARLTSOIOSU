"""
ppo_agent.py
------------
Proximal Policy Optimisation (PPO) agent for the Intelligent Urban Traffic
Management System (IUTMS).

Architecture (per-agent actor-critic):
    Shared backbone:  Input(state_dim) → FC(128) → ReLU → FC(64) → ReLU
    Actor head:       FC(64) → action_dim  (softmax → categorical policy)
    Critic head:      FC(64) → 1           (state-value estimate V(s))

Algorithm highlights
--------------------
* **Clipped surrogate objective** – prevents excessively large policy updates,
  keeping learning stable in the non-stationary multi-agent setting.
  Clip coefficient ε = 0.2 (default, following the original PPO paper).
* **Generalised Advantage Estimation (GAE)** – reduces variance while
  preserving a controllable bias via parameter λ (default 0.95).
* **Rollout buffer** – collect *n* on-policy steps per update, then perform
  *K* optimisation epochs over randomly shuffled mini-batches.
* **Entropy bonus** – encourages exploration by penalising over-confident
  policies.  Coefficient c₂ (default 0.01) is annealed as training matures.
* **Value-function clipping** – clips the value loss to avoid large swings in
  the critic that could destabilise the policy.

References
----------
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- maxbrenner-ai/Multi-Agent-Distributed-PPO-Traffc-light-control
- cts198859/deeprl_signal_control (actor-critic patterns)
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# Actor-Critic network
# ---------------------------------------------------------------------------

class ActorCriticNetwork(nn.Module):
    """
    Shared-backbone actor-critic network.

    Parameters
    ----------
    state_dim : int
        Observation vector dimensionality.
    action_dim : int
        Number of discrete actions.
    hidden1 : int
        First hidden-layer width (default 128).
    hidden2 : int
        Second hidden-layer width (default 64).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden1: int = 128,
        hidden2: int = 64,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )
        self.actor_head  = nn.Linear(hidden2, action_dim)
        self.critic_head = nn.Linear(hidden2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits : torch.Tensor, shape (batch, action_dim)
        value  : torch.Tensor, shape (batch,)
        """
        features = self.backbone(x)
        logits   = self.actor_head(features)
        value    = self.critic_head(features).squeeze(-1)
        return logits, value

    def get_action(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy π(·|s).

        Returns
        -------
        action      : long tensor (batch,)
        log_prob    : float tensor (batch,)
        entropy     : float tensor (batch,) – used for the entropy bonus
        """
        logits, value = self(x)
        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy

    def evaluate(
        self, x: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log-probabilities and value estimates for given state-action
        pairs (used during the optimisation phase).

        Parameters
        ----------
        x       : observation batch
        actions : action batch (long)

        Returns
        -------
        log_prob, value, entropy  – all of shape (batch,)
        """
        logits, value = self(x)
        dist     = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        return log_prob, value, entropy


# ---------------------------------------------------------------------------
# On-policy rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Circular buffer that stores on-policy trajectories for PPO updates.

    After ``collect()`` fills *n_steps* transitions, call ``compute_gae()``
    to populate advantage estimates, then iterate over ``mini_batches()``
    for the optimisation epochs.
    """

    def __init__(self, n_steps: int, state_dim: int) -> None:
        self.n_steps   = n_steps
        self.state_dim = state_dim
        self._ptr      = 0
        self._full     = False

        self.states    = np.zeros((n_steps, state_dim), dtype=np.float32)
        self.actions   = np.zeros(n_steps, dtype=np.int64)
        self.rewards   = np.zeros(n_steps, dtype=np.float32)
        self.values    = np.zeros(n_steps, dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.dones     = np.zeros(n_steps, dtype=np.float32)

        # Computed by compute_gae()
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns    = np.zeros(n_steps, dtype=np.float32)

    # ------------------------------------------------------------------
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        idx = self._ptr % self.n_steps
        self.states[idx]    = state
        self.actions[idx]   = action
        self.rewards[idx]   = reward
        self.values[idx]    = value
        self.log_probs[idx] = log_prob
        self.dones[idx]     = float(done)
        self._ptr += 1
        if self._ptr >= self.n_steps:
            self._full = True

    def is_full(self) -> bool:
        return self._full

    def reset(self) -> None:
        self._ptr  = 0
        self._full = False

    # ------------------------------------------------------------------
    def compute_gae(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """
        Compute Generalised Advantage Estimates in-place.

        GAE(λ):
            δₜ = rₜ + γ·V(sₜ₊₁)·(1−doneₜ) − V(sₜ)
            Aₜ = δₜ + (γλ)·Aₜ₊₁·(1−doneₜ)
        """
        gae = 0.0
        for t in reversed(range(self.n_steps)):
            next_value = last_value if t == self.n_steps - 1 else self.values[t + 1]
            next_done  = self.dones[t]
            delta      = self.rewards[t] + gamma * next_value * (1.0 - next_done) - self.values[t]
            gae        = delta + gamma * gae_lambda * (1.0 - next_done) * gae
            self.advantages[t] = gae
            self.returns[t]    = gae + self.values[t]

    # ------------------------------------------------------------------
    def mini_batches(
        self, batch_size: int
    ):
        """
        Yield randomly shuffled mini-batches as numpy arrays.

        Yields
        ------
        (states, actions, log_probs_old, advantages, returns)
        """
        indices = np.arange(self.n_steps)
        np.random.shuffle(indices)
        for start in range(0, self.n_steps, batch_size):
            idx = indices[start : start + batch_size]
            yield (
                self.states[idx],
                self.actions[idx],
                self.log_probs[idx],
                self.advantages[idx],
                self.returns[idx],
            )

    def __len__(self) -> int:
        return self.n_steps if self._full else self._ptr


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """
    Independent PPO agent controlling a single traffic-signal intersection.

    Inspired by:
      - maxbrenner-ai/Multi-Agent-Distributed-PPO-Traffc-light-control
      - Schulman et al. (2017) PPO paper

    Parameters
    ----------
    ts_id : str
        Traffic-signal identifier.
    state_dim : int
        Observation vector size.
    action_dim : int
        Number of discrete actions (green-phase combinations).
    lr : float
        Adam learning rate (shared for actor and critic).
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE smoothing parameter (λ).
    clip_epsilon : float
        PPO clipping coefficient (ε).
    value_loss_coef : float
        Weight of the critic loss in the combined objective.
    entropy_coef : float
        Weight of the entropy bonus (encourages exploration).
    n_steps : int
        Number of rollout steps before each policy update.
    n_epochs : int
        Number of optimisation epochs per update.
    batch_size : int
        Mini-batch size within each epoch.
    device : str | None
        ``"cuda"`` or ``"cpu"``; auto-detected when *None*.
    """

    def __init__(
        self,
        ts_id: str,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        n_steps: int = 512,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: Optional[str] = None,
    ) -> None:
        self.ts_id           = ts_id
        self.state_dim       = state_dim
        self.action_dim      = action_dim
        self.gamma           = gamma
        self.gae_lambda      = gae_lambda
        self.clip_epsilon    = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef    = entropy_coef
        self.n_steps         = n_steps
        self.n_epochs        = n_epochs
        self.batch_size      = batch_size

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.net = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

        self.buffer = RolloutBuffer(n_steps=n_steps, state_dim=state_dim)
        self._update_count = 0

    # ------------------------------------------------------------------
    # Public interface (compatible with DQNAgent's surface)
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        """Dummy attribute for API compatibility; PPO has no ε-greedy."""
        return 0.0

    def select_action(
        self, state: np.ndarray
    ) -> int:
        """
        Sample an action from the current policy π(·|s).

        Also stores the value estimate and log-probability in internal state
        so that the *next* ``store()`` call can read them.

        Parameters
        ----------
        state : np.ndarray  (float32, shape = (state_dim,))

        Returns
        -------
        int  – chosen action index
        """
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_t, log_prob_t, _ = self.net.get_action(state_t)
            _, value_t              = self.net(state_t)

        self._last_value    = value_t.item()
        self._last_log_prob = log_prob_t.item()
        return int(action_t.item())

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,   # unused by PPO (on-policy) but kept for API compat.
        done: bool,
    ) -> None:
        """
        Push the most-recent transition into the rollout buffer.

        The ``value`` and ``log_prob`` must be set by a preceding
        ``select_action()`` call for the same *state*.
        """
        self.buffer.push(
            state    = state,
            action   = action,
            reward   = reward,
            value    = self._last_value,
            log_prob = self._last_log_prob,
            done     = done,
        )

    def learn(self, last_state: Optional[np.ndarray] = None) -> Optional[float]:
        """
        Perform a PPO update when the rollout buffer is full.

        Parameters
        ----------
        last_state : np.ndarray | None
            The state *after* the last stored transition, used to bootstrap
            the value for the last step in GAE.  If None the bootstrap value
            is zero (terminal).

        Returns
        -------
        float | None
            Mean policy loss over all epochs/mini-batches (or None if the
            buffer is not yet full).
        """
        if not self.buffer.is_full():
            return None

        # Bootstrap value for GAE
        if last_state is not None:
            state_t = torch.as_tensor(
                last_state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                _, last_val_t = self.net(state_t)
            last_value = last_val_t.item()
        else:
            last_value = 0.0

        self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

        # Normalise advantages for numerical stability
        adv = self.buffer.advantages
        adv_mean, adv_std = adv.mean(), adv.std() + 1e-8
        self.buffer.advantages = (adv - adv_mean) / adv_std

        total_policy_loss = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            for batch in self.buffer.mini_batches(self.batch_size):
                states_b, actions_b, log_probs_old_b, advantages_b, returns_b = batch

                states_t      = torch.as_tensor(states_b,      dtype=torch.float32, device=self.device)
                actions_t     = torch.as_tensor(actions_b,     dtype=torch.int64,   device=self.device)
                log_probs_old = torch.as_tensor(log_probs_old_b, dtype=torch.float32, device=self.device)
                advantages_t  = torch.as_tensor(advantages_b,  dtype=torch.float32, device=self.device)
                returns_t     = torch.as_tensor(returns_b,     dtype=torch.float32, device=self.device)

                # Evaluate current policy
                log_probs_new, values_pred, entropy = self.net.evaluate(states_t, actions_t)

                # Probability ratio r(θ) = π_θ(a|s) / π_θ_old(a|s)
                ratio = torch.exp(log_probs_new - log_probs_old)

                # Clipped surrogate objective
                surr1 = ratio * advantages_t
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_t
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped, following the PPO paper)
                values_pred_clipped = (
                    torch.as_tensor(self.buffer.values[: len(states_b)], device=self.device)
                    + torch.clamp(
                        values_pred
                        - torch.as_tensor(self.buffer.values[: len(states_b)], device=self.device),
                        -self.clip_epsilon,
                        self.clip_epsilon,
                    )
                )
                value_loss = torch.max(
                    nn.functional.mse_loss(values_pred, returns_t),
                    nn.functional.mse_loss(values_pred_clipped, returns_t),
                )

                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_loss_coef  * value_loss
                    + self.entropy_coef     * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                n_updates += 1

        self.buffer.reset()
        self._update_count += 1
        return total_policy_loss / max(n_updates, 1)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist network weights and optimiser state."""
        torch.save(
            {
                "net":          self.net.state_dict(),
                "optimizer":    self.optimizer.state_dict(),
                "update_count": self._update_count,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Restore network weights and optimiser state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint["net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._update_count = checkpoint.get("update_count", 0)
