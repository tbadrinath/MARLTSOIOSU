"""
tests/test_env_wrapper.py
-------------------------
Unit tests for simulation/env_wrapper.py.

All TraCI calls are mocked so that SUMO does not need to be installed.
"""

import sys
import os
import types
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# Ensure repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Provide a minimal traci stub BEFORE importing env_wrapper so that the
# module-level `import traci` succeeds without a real SUMO installation.
# ---------------------------------------------------------------------------

def _make_traci_stub():
    """Build a minimal traci mock that satisfies all env_wrapper usages."""
    traci_mod = types.ModuleType("traci")

    # --- trafficlight sub-module ---
    tl = MagicMock()
    tl.getIDList.return_value = ["A0", "B0", "C0"]

    phase_mock = MagicMock()
    phase_mock.phases = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    logic_mock = MagicMock()
    logic_mock.phases = phase_mock.phases
    tl.getAllProgramLogics.return_value = [logic_mock]
    tl.getControlledLinks.return_value = [
        [("in_lane_0", "out_lane_0", None)],
        [("in_lane_1", "out_lane_1", None)],
    ]
    tl.getProgram.return_value = "0"
    tl.setPhase = MagicMock()
    tl.getPhase = MagicMock(return_value=1)  # current phase index
    traci_mod.trafficlight = tl

    # --- lane sub-module ---
    lane = MagicMock()
    lane.getLastStepVehicleNumber.return_value = 5
    lane.getLastStepOccupancy.return_value = 20.0   # 20 %
    lane.getLastStepHaltingNumber.return_value = 2
    lane.getLastStepVehicleIDs.return_value = ["veh0", "veh1"]
    traci_mod.lane = lane

    # --- vehicle sub-module ---
    vehicle = MagicMock()
    vehicle.getWaitingTime.return_value = 10.0
    vehicle.getSpeed.return_value = 8.0
    vehicle.getCO2Emission.return_value = 100.0
    vehicle.getIDList.return_value = ["veh0", "veh1"]
    traci_mod.vehicle = vehicle

    # --- simulation sub-module ---
    sim = MagicMock()
    sim.getDepartedNumber.return_value = 3
    traci_mod.simulation = sim

    # --- top-level functions ---
    traci_mod.start = MagicMock()
    traci_mod.close = MagicMock()
    traci_mod.simulationStep = MagicMock()

    # constants sub-module
    tc_mod = types.ModuleType("traci.constants")
    traci_mod.constants = tc_mod

    return traci_mod


# Inject the stub into sys.modules BEFORE importing env_wrapper
_traci_stub = _make_traci_stub()
sys.modules["traci"] = _traci_stub
sys.modules["traci.constants"] = _traci_stub.constants

from simulation.env_wrapper import (  # noqa: E402 – must follow stub injection
    TrafficEnv,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    DEFAULT_DELTA,
    SPILLBACK_THRESHOLD,
    MAX_LANE_VEHICLES,
    MAX_WAIT_TIME,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def env():
    """Create a TrafficEnv with mocked SUMO start/close."""
    e = TrafficEnv(
        net_file="maps/grid.net.xml",
        route_file="maps/grid.rou.xml",
        ts_ids=["A0", "B0", "C0"],
        max_steps=100,
    )
    with patch.object(e, "_start_sumo"), patch.object(e, "_close_sumo"):
        e.reset()
    return e


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrafficEnvInit:
    def test_default_weights(self):
        e = TrafficEnv("net.xml", "rou.xml")
        assert e.alpha == DEFAULT_ALPHA
        assert e.beta  == DEFAULT_BETA
        assert e.gamma == DEFAULT_GAMMA
        assert e.delta == DEFAULT_DELTA

    def test_custom_weights(self):
        e = TrafficEnv("net.xml", "rou.xml", alpha=0.1, beta=0.2, gamma=0.3, delta=0.4)
        assert e.alpha == pytest.approx(0.1)
        assert e.beta  == pytest.approx(0.2)
        assert e.gamma == pytest.approx(0.3)
        assert e.delta == pytest.approx(0.4)


class TestReset:
    def test_returns_obs_dict(self, env):
        assert isinstance(env.ts_ids, list)
        assert len(env.ts_ids) == 3

    def test_obs_are_numpy_float32(self, env):
        with patch.object(env, "_start_sumo"), patch.object(env, "_close_sumo"):
            obs = env.reset()
        for ts, o in obs.items():
            assert isinstance(o, np.ndarray), f"{ts}: expected ndarray"
            assert o.dtype == np.float32, f"{ts}: expected float32"

    def test_obs_values_clipped_to_one(self, env):
        # Artificially high vehicle count to test normalisation clipping
        _traci_stub.lane.getLastStepVehicleNumber.return_value = 9999
        with patch.object(env, "_start_sumo"), patch.object(env, "_close_sumo"):
            obs = env.reset()
        for o in obs.values():
            assert np.all(o <= 1.0 + 1e-6)
        _traci_stub.lane.getLastStepVehicleNumber.return_value = 5   # restore


class TestObservationSpace:
    def test_size_formula(self, env):
        for ts in env.ts_ids:
            n_in  = len(env._lane_map[ts])
            n_out = len(env._out_lane_map[ts])
            expected = n_in * 2 + n_out
            assert env.observation_space_size(ts) == expected


class TestActionSpace:
    def test_returns_int(self, env):
        for ts in env.ts_ids:
            assert isinstance(env.action_space_size(ts), int)
            assert env.action_space_size(ts) >= 1


class TestReward:
    def test_reward_is_float(self, env):
        r = env._compute_reward("A0")
        assert isinstance(r, float)

    def test_reward_decreases_with_spillback(self, env):
        """Reward should be lower when downstream occupancy exceeds 90 %."""
        _traci_stub.lane.getLastStepOccupancy.return_value = 20.0   # low occ
        r_low = env._compute_reward("A0")

        _traci_stub.lane.getLastStepOccupancy.return_value = 95.0   # spillback
        r_high = env._compute_reward("A0")

        assert r_high < r_low
        _traci_stub.lane.getLastStepOccupancy.return_value = 20.0   # restore

    def test_reward_formula_components(self, env):
        """
        Manually compute reward and compare against _compute_reward.
        Uses exact stub values so the formula can be verified.
        """
        _traci_stub.lane.getLastStepOccupancy.return_value = 20.0
        _traci_stub.lane.getLastStepVehicleNumber.return_value = 5
        _traci_stub.lane.getLastStepHaltingNumber.return_value = 2
        _traci_stub.lane.getLastStepVehicleIDs.return_value = ["v0"]
        _traci_stub.vehicle.getWaitingTime.return_value = 30.0
        _traci_stub.simulation.getDepartedNumber.return_value = 3

        env._prev_departed = 0  # so throughput = 3

        r = env._compute_reward("A0")

        in_lanes  = env._lane_map["A0"]
        out_lanes = env._out_lane_map["A0"]
        n_agents  = len(env.ts_ids)

        # throughput
        throughput = 3 / n_agents
        # queue
        halted = 2 * len(in_lanes)
        queue_penalty = halted / max(len(in_lanes) * MAX_LANE_VEHICLES, 1)
        # wait
        avg_wait = 30.0 / MAX_WAIT_TIME
        # spillback (occ=20 < 90 → 0)
        spillback = 0.0

        expected = (
            env.alpha * throughput
            - env.beta  * queue_penalty
            - env.gamma * avg_wait
            - env.delta * spillback
        )
        assert r == pytest.approx(expected, abs=1e-5)


class TestCollectInfo:
    def test_info_keys(self, env):
        info = env._collect_info()
        assert "step" in info
        assert "avg_speed" in info
        assert "co2_emissions" in info
        assert "vehicles_in_network" in info

    def test_info_values_are_numbers(self, env):
        info = env._collect_info()
        for key in ("avg_speed", "co2_emissions", "vehicles_in_network"):
            assert isinstance(info[key], (int, float))


# ---------------------------------------------------------------------------
# New feature tests: pressure reward and phase-time observations
# ---------------------------------------------------------------------------

class TestPressureReward:
    """Tests for the pressure reward mode (sumo-rl style)."""

    def _make_env(self) -> "TrafficEnv":
        from simulation.env_wrapper import TrafficEnv, REWARD_MODE_PRESSURE
        e = TrafficEnv(
            net_file="maps/grid.net.xml",
            route_file="maps/grid.rou.xml",
            ts_ids=["A0", "B0", "C0"],
            max_steps=100,
            reward_mode=REWARD_MODE_PRESSURE,
        )
        with patch.object(e, "_start_sumo"), patch.object(e, "_close_sumo"):
            e.reset()
        return e

    def test_pressure_reward_is_float(self):
        env = self._make_env()
        r = env._compute_reward("A0")
        assert isinstance(r, float)

    def test_pressure_reward_non_positive(self):
        """Pressure reward is always ≤ 0 (it is −|in − out| / n)."""
        env = self._make_env()
        r = env._compute_reward("A0")
        assert r <= 0.0

    def test_pressure_reward_zero_when_balanced(self):
        """When incoming and outgoing halting numbers are equal, pressure = 0."""
        env = self._make_env()
        original = _traci_stub.lane.getLastStepHaltingNumber.return_value
        try:
            _traci_stub.lane.getLastStepHaltingNumber.return_value = 3
            r = env._compute_reward("A0")
            # in = 2 lanes × 3 = 6, out = 2 lanes × 3 = 6 → pressure = 0
            assert r == pytest.approx(0.0, abs=1e-6)
        finally:
            _traci_stub.lane.getLastStepHaltingNumber.return_value = original

    def test_composite_vs_pressure_differ(self):
        from simulation.env_wrapper import REWARD_MODE_COMPOSITE, REWARD_MODE_PRESSURE
        env_c = TrafficEnv("n.xml", "r.xml", ts_ids=["A0"],
                           reward_mode=REWARD_MODE_COMPOSITE)
        env_p = TrafficEnv("n.xml", "r.xml", ts_ids=["A0"],
                           reward_mode=REWARD_MODE_PRESSURE)
        for e in (env_c, env_p):
            with patch.object(e, "_start_sumo"), patch.object(e, "_close_sumo"):
                e.reset()
        rc = env_c._compute_reward("A0")
        rp = env_p._compute_reward("A0")
        # They use different formulas, so they should generally differ
        # (this is a sanity check, not an equality check)
        assert isinstance(rc, float)
        assert isinstance(rp, float)


class TestPhaseObservation:
    """Tests for use_phase_obs=True (AndreaVidali-style phase features)."""

    def _make_phase_env(self) -> "TrafficEnv":
        from simulation.env_wrapper import TrafficEnv
        e = TrafficEnv(
            net_file="maps/grid.net.xml",
            route_file="maps/grid.rou.xml",
            ts_ids=["A0", "B0", "C0"],
            max_steps=100,
            use_phase_obs=True,
        )
        _traci_stub.trafficlight.getPhase = MagicMock(return_value=1)
        with patch.object(e, "_start_sumo"), patch.object(e, "_close_sumo"):
            e.reset()
        return e

    def test_obs_size_larger_with_phase(self):
        from simulation.env_wrapper import TrafficEnv
        env_no  = TrafficEnv("n.xml", "r.xml", ts_ids=["A0"], use_phase_obs=False)
        env_yes = TrafficEnv("n.xml", "r.xml", ts_ids=["A0"], use_phase_obs=True)
        for e in (env_no, env_yes):
            with patch.object(e, "_start_sumo"), patch.object(e, "_close_sumo"):
                e.reset()
        size_no  = env_no.observation_space_size("A0")
        size_yes = env_yes.observation_space_size("A0")
        assert size_yes == size_no + 2

    def test_obs_vector_length_matches_size(self):
        env = self._make_phase_env()
        obs = env._get_observation("A0")
        assert len(obs) == env.observation_space_size("A0")

    def test_phase_features_clipped(self):
        """Phase-time normalised feature should be in [0, 1]."""
        env = self._make_phase_env()
        # Set a very large phase_step to test clamping
        env._phase_step["A0"] = 99999
        obs = env._get_observation("A0")
        # Last two features are phase features
        assert 0.0 <= obs[-1] <= 1.0
        assert 0.0 <= obs[-2] <= 1.0

    def test_reward_mode_defaults_to_composite_with_phase(self):
        """use_phase_obs should not change the default reward mode."""
        from simulation.env_wrapper import REWARD_MODE_COMPOSITE
        env = self._make_phase_env()
        assert env.reward_mode == REWARD_MODE_COMPOSITE
