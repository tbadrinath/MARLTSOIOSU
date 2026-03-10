"""
tests/test_ppo_agent.py
-----------------------
Unit tests for simulation/ppo_agent.py
(ActorCriticNetwork, RolloutBuffer, PPOAgent).

No SUMO or GPU required.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.ppo_agent import ActorCriticNetwork, RolloutBuffer, PPOAgent


# ---------------------------------------------------------------------------
# ActorCriticNetwork
# ---------------------------------------------------------------------------

class TestActorCriticNetwork:
    def test_forward_shapes(self):
        net = ActorCriticNetwork(state_dim=12, action_dim=4)
        x = torch.zeros(8, 12)
        logits, value = net(x)
        assert logits.shape == (8, 4)
        assert value.shape  == (8,)

    def test_get_action_shapes(self):
        net = ActorCriticNetwork(state_dim=6, action_dim=3)
        x = torch.randn(1, 6)
        action, log_prob, entropy = net.get_action(x)
        assert action.shape   == (1,)
        assert log_prob.shape == (1,)
        assert entropy.shape  == (1,)

    def test_action_in_range(self):
        net = ActorCriticNetwork(state_dim=6, action_dim=4)
        x = torch.randn(100, 6)
        action, _, _ = net.get_action(x)
        assert action.min().item() >= 0
        assert action.max().item() < 4

    def test_evaluate_shapes(self):
        net = ActorCriticNetwork(state_dim=8, action_dim=3)
        x = torch.randn(16, 8)
        actions = torch.randint(0, 3, (16,))
        log_prob, value, entropy = net.evaluate(x, actions)
        assert log_prob.shape == (16,)
        assert value.shape    == (16,)
        assert entropy.shape  == (16,)

    def test_entropy_positive(self):
        """Entropy must be ≥ 0 (can be 0 for deterministic policies)."""
        net = ActorCriticNetwork(state_dim=4, action_dim=2)
        x = torch.randn(32, 4)
        _, _, entropy = net.get_action(x)
        assert (entropy >= 0).all()

    def test_custom_hidden_dims(self):
        net = ActorCriticNetwork(state_dim=10, action_dim=5, hidden1=64, hidden2=32)
        x = torch.randn(4, 10)
        logits, value = net(x)
        assert logits.shape == (4, 5)
        assert value.shape  == (4,)


# ---------------------------------------------------------------------------
# RolloutBuffer
# ---------------------------------------------------------------------------

class TestRolloutBuffer:
    STATE_DIM = 8
    N_STEPS   = 20

    def _fill_buffer(self, buf: RolloutBuffer, n: int = None) -> None:
        n = n or buf.n_steps
        for _ in range(n):
            buf.push(
                state    = np.random.randn(self.STATE_DIM).astype(np.float32),
                action   = int(np.random.randint(0, 4)),
                reward   = float(np.random.randn()),
                value    = float(np.random.randn()),
                log_prob = float(np.random.randn()),
                done     = bool(np.random.random() < 0.1),
            )

    def test_not_full_before_n_steps(self):
        buf = RolloutBuffer(n_steps=self.N_STEPS, state_dim=self.STATE_DIM)
        self._fill_buffer(buf, n=self.N_STEPS - 1)
        assert not buf.is_full()

    def test_full_after_n_steps(self):
        buf = RolloutBuffer(n_steps=self.N_STEPS, state_dim=self.STATE_DIM)
        self._fill_buffer(buf)
        assert buf.is_full()

    def test_len(self):
        buf = RolloutBuffer(n_steps=self.N_STEPS, state_dim=self.STATE_DIM)
        self._fill_buffer(buf, n=5)
        assert len(buf) == 5
        self._fill_buffer(buf, n=self.N_STEPS - 5)
        assert len(buf) == self.N_STEPS

    def test_reset(self):
        buf = RolloutBuffer(n_steps=self.N_STEPS, state_dim=self.STATE_DIM)
        self._fill_buffer(buf)
        assert buf.is_full()
        buf.reset()
        assert not buf.is_full()
        assert len(buf) == 0

    def test_compute_gae_shape(self):
        buf = RolloutBuffer(n_steps=self.N_STEPS, state_dim=self.STATE_DIM)
        self._fill_buffer(buf)
        buf.compute_gae(last_value=0.0)
        assert buf.advantages.shape == (self.N_STEPS,)
        assert buf.returns.shape    == (self.N_STEPS,)

    def test_compute_gae_terminal_value_zero(self):
        """When last_value=0 and done=True at last step, returns[last] = rewards[last]."""
        buf = RolloutBuffer(n_steps=5, state_dim=2)
        for i in range(5):
            buf.push(
                state    = np.zeros(2, dtype=np.float32),
                action   = 0,
                reward   = 1.0,
                value    = 0.0,
                log_prob = 0.0,
                done     = (i == 4),   # only last step is terminal
            )
        buf.compute_gae(last_value=0.0, gamma=1.0, gae_lambda=1.0)
        # With γ=1, λ=1, no value estimates, returns should equal cumulative future reward
        assert buf.returns[4] == pytest.approx(1.0, abs=1e-5)

    def test_mini_batches_cover_all(self):
        buf = RolloutBuffer(n_steps=self.N_STEPS, state_dim=self.STATE_DIM)
        self._fill_buffer(buf)
        buf.compute_gae(last_value=0.0)

        seen = 0
        for batch in buf.mini_batches(batch_size=5):
            states, actions, log_probs_old, advantages, returns = batch
            assert states.shape[1] == self.STATE_DIM
            seen += len(states)
        assert seen == self.N_STEPS


# ---------------------------------------------------------------------------
# PPOAgent
# ---------------------------------------------------------------------------

class TestPPOAgent:
    STATE_DIM  = 12
    ACTION_DIM = 4

    def _make_agent(self, **kwargs) -> PPOAgent:
        defaults = dict(
            ts_id      = "test_ts",
            state_dim  = self.STATE_DIM,
            action_dim = self.ACTION_DIM,
            n_steps    = 32,
            n_epochs   = 2,
            batch_size = 16,
            device     = "cpu",
        )
        defaults.update(kwargs)
        return PPOAgent(**defaults)

    def _random_state(self) -> np.ndarray:
        return np.random.randn(self.STATE_DIM).astype(np.float32)

    # --- select_action ---

    def test_select_action_in_range(self):
        agent = self._make_agent()
        for _ in range(50):
            a = agent.select_action(self._random_state())
            assert 0 <= a < self.ACTION_DIM

    def test_epsilon_always_zero(self):
        """PPO has no ε-greedy; epsilon property should always return 0."""
        agent = self._make_agent()
        assert agent.epsilon == 0.0

    # --- store ---

    def test_store_fills_buffer(self):
        agent = self._make_agent()
        state = self._random_state()
        for _ in range(agent.n_steps):
            agent.select_action(state)   # sets _last_value / _last_log_prob
            agent.store(state, 0, 0.0, state, False)
        assert agent.buffer.is_full()

    # --- learn ---

    def test_learn_returns_none_before_full(self):
        agent = self._make_agent()
        state = self._random_state()
        agent.select_action(state)
        agent.store(state, 0, 1.0, state, False)
        # Buffer not yet full
        result = agent.learn(last_state=state)
        assert result is None

    def test_learn_returns_float_when_full(self):
        agent = self._make_agent()
        state = self._random_state()
        for _ in range(agent.n_steps):
            agent.select_action(state)
            agent.store(state, 0, 0.5, state, False)
        result = agent.learn(last_state=state)
        assert isinstance(result, float)

    def test_buffer_cleared_after_learn(self):
        agent = self._make_agent()
        state = self._random_state()
        for _ in range(agent.n_steps):
            agent.select_action(state)
            agent.store(state, 0, 0.1, state, False)
        agent.learn(last_state=state)
        assert not agent.buffer.is_full()

    def test_update_count_increments(self):
        agent = self._make_agent()
        state = self._random_state()
        for _ in range(agent.n_steps):
            agent.select_action(state)
            agent.store(state, 0, 0.1, state, False)
        agent.learn(last_state=state)
        assert agent._update_count == 1

    # --- checkpoint ---

    def test_save_and_load(self, tmp_path):
        agent = self._make_agent()
        path = str(tmp_path / "ppo.pt")
        agent.save(path)

        agent2 = self._make_agent()
        agent2.load(path)

        for p1, p2 in zip(agent.net.parameters(), agent2.net.parameters()):
            assert torch.allclose(p1, p2)

    # --- compatibility with DQN trainer API surface ---

    def test_store_signature_matches_dqn(self):
        """store(state, action, reward, next_state, done) – same as DQNAgent."""
        agent = self._make_agent()
        s = self._random_state()
        agent.select_action(s)
        # Should not raise
        agent.store(s, 1, -0.5, s, False)
