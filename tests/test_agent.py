"""
tests/test_agent.py
-------------------
Unit tests for simulation/agent.py (DQNAgent, QNetwork, ReplayBuffer).
No SUMO or GPU required.
"""

import random
import sys
import os

import numpy as np
import pytest
import torch

# Ensure the repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.agent import DQNAgent, QNetwork, ReplayBuffer


# ---------------------------------------------------------------------------
# QNetwork tests
# ---------------------------------------------------------------------------

class TestQNetwork:
    def test_output_shape(self):
        net = QNetwork(state_dim=12, action_dim=4)
        x = torch.zeros(8, 12)   # batch of 8
        out = net(x)
        assert out.shape == (8, 4)

    def test_output_dtype(self):
        net = QNetwork(state_dim=6, action_dim=3)
        x = torch.randn(1, 6)
        out = net(x)
        assert out.dtype == torch.float32

    def test_different_dims(self):
        for state_dim, action_dim in [(1, 1), (32, 8), (100, 16)]:
            net = QNetwork(state_dim, action_dim)
            out = net(torch.zeros(2, state_dim))
            assert out.shape == (2, action_dim)


# ---------------------------------------------------------------------------
# ReplayBuffer tests
# ---------------------------------------------------------------------------

class TestReplayBuffer:
    def _make_transition(self, dim=4):
        state      = np.random.randn(dim).astype(np.float32)
        action     = random.randint(0, 3)
        reward     = random.uniform(-1, 1)
        next_state = np.random.randn(dim).astype(np.float32)
        done       = random.random() < 0.1
        return state, action, reward, next_state, done

    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=100)
        for _ in range(10):
            buf.push(*self._make_transition())
        assert len(buf) == 10

    def test_capacity_ring(self):
        buf = ReplayBuffer(capacity=5)
        for _ in range(20):
            buf.push(*self._make_transition())
        assert len(buf) == 5   # capped at capacity

    def test_sample_shapes(self):
        buf = ReplayBuffer(capacity=200)
        for _ in range(100):
            buf.push(*self._make_transition(dim=8))
        states, actions, rewards, next_states, dones = buf.sample(32)
        assert states.shape      == (32, 8)
        assert actions.shape     == (32,)
        assert rewards.shape     == (32,)
        assert next_states.shape == (32, 8)
        assert dones.shape       == (32,)

    def test_sample_raises_when_too_small(self):
        buf = ReplayBuffer(capacity=100)
        buf.push(*self._make_transition())
        with pytest.raises(ValueError):
            buf.sample(10)   # not enough items


# ---------------------------------------------------------------------------
# DQNAgent tests
# ---------------------------------------------------------------------------

class TestDQNAgent:
    STATE_DIM  = 12
    ACTION_DIM = 4

    def _make_agent(self, **kwargs):
        defaults = dict(
            ts_id="test",
            state_dim=self.STATE_DIM,
            action_dim=self.ACTION_DIM,
            buffer_capacity=500,
            batch_size=32,
            device="cpu",
        )
        defaults.update(kwargs)
        return DQNAgent(**defaults)

    # --- Action selection ---

    def test_select_action_range(self):
        agent = self._make_agent(epsilon_start=0.0)  # greedy
        state = np.zeros(self.STATE_DIM, dtype=np.float32)
        for _ in range(20):
            a = agent.select_action(state)
            assert 0 <= a < self.ACTION_DIM

    def test_select_action_random_with_epsilon_one(self):
        """With ε=1.0 every action should be random (uniform over actions)."""
        agent = self._make_agent(epsilon_start=1.0)
        state = np.zeros(self.STATE_DIM, dtype=np.float32)
        actions = {agent.select_action(state) for _ in range(200)}
        # At least 2 distinct actions should have been chosen (very likely)
        assert len(actions) > 1

    # --- Learning ---

    def test_learn_returns_none_when_buffer_small(self):
        agent = self._make_agent()
        assert agent.learn() is None

    def test_learn_returns_float_after_fill(self):
        agent = self._make_agent(batch_size=16, buffer_capacity=100)
        state = np.zeros(self.STATE_DIM, dtype=np.float32)
        for _ in range(30):
            agent.store(state, 0, 0.1, state, False)
        loss = agent.learn()
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_epsilon_decays(self):
        agent = self._make_agent(
            epsilon_start=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.9,
            batch_size=8,
            buffer_capacity=100,
        )
        state = np.zeros(self.STATE_DIM, dtype=np.float32)
        for _ in range(20):
            agent.store(state, 0, 0.0, state, False)
        initial_eps = agent.epsilon
        agent.learn()
        assert agent.epsilon < initial_eps
        assert agent.epsilon >= agent.epsilon_min

    def test_target_net_updated(self):
        agent = self._make_agent(
            target_update=2,
            batch_size=8,
            buffer_capacity=100,
        )
        state = np.zeros(self.STATE_DIM, dtype=np.float32)
        for _ in range(20):
            agent.store(state, 0, 1.0, state, False)

        # Modify online network weights
        with torch.no_grad():
            for p in agent.q_net.parameters():
                p.fill_(99.0)

        # Target network should be different at this point
        for qp, tp in zip(agent.q_net.parameters(), agent.target_net.parameters()):
            assert not torch.allclose(qp, tp)

        # After target_update steps the target should be synced
        for _ in range(agent.target_update):
            agent.learn()

        for qp, tp in zip(agent.q_net.parameters(), agent.target_net.parameters()):
            assert torch.allclose(qp, tp)

    # --- Checkpoint ---

    def test_save_and_load(self, tmp_path):
        agent = self._make_agent()
        path = str(tmp_path / "agent.pt")
        agent.save(path)

        agent2 = self._make_agent()
        agent2.load(path)

        for p1, p2 in zip(agent.q_net.parameters(), agent2.q_net.parameters()):
            assert torch.allclose(p1, p2)
        assert agent.epsilon == agent2.epsilon
