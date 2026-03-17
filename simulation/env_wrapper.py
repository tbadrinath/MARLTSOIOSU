"""
env_wrapper.py
--------------
TraCI environment wrapper for the Intelligent Urban Traffic Management System (IUTMS).

Connects Python to SUMO via the TraCI interface and exposes a Gym-like
multi-agent API so that each intersection can be treated as an independent
learner (decentralised MARL).

Observation space per agent (spatio-temporal):
    - Normalised vehicle count per incoming lane
    - Lane occupancy (density)
    - Upstream / downstream congestion status sampled from induction loops
      placed 100 m from each stopline.
    - [optional] current phase index (normalised) and time in current phase
      Enabled with ``use_phase_obs=True`` (inspired by AndreaVidali/
      Deep-QLearning-Agent-for-Traffic-Signal-Control).

Reward modes
------------
composite (default):
    R = α·Throughput − β·Queue − γ·WaitTime − δ·SpillbackPenalty
    Fairness-aware formula that prevents main-road bias.

pressure:
    R = −|incoming_queue − outgoing_queue|
    Inspired by LucasAlegre/sumo-rl — simple, scale-invariant, and
    effective in oversaturated networks.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# TraCI import guard – allow importing the module even when SUMO is absent
# (e.g. during unit-test runs that mock TraCI).
# ---------------------------------------------------------------------------
try:
    import traci  # type: ignore
    import traci.constants as tc  # type: ignore
    TRACI_AVAILABLE = True
except ImportError:
    traci = None  # type: ignore
    tc = None  # type: ignore
    TRACI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Reward hyper-parameters (can be overridden at construction time)
# ---------------------------------------------------------------------------
DEFAULT_ALPHA = 0.4   # throughput weight
DEFAULT_BETA = 0.3    # queue-length weight
DEFAULT_GAMMA = 0.2   # waiting-time weight
DEFAULT_DELTA = 0.5   # spillback-penalty weight

SPILLBACK_THRESHOLD = 0.90   # downstream occupancy fraction
LOOP_DETECTOR_DIST = 100.0   # metres – induction loop placement
MAX_LANE_VEHICLES = 40       # normalisation constant for vehicle counts
MAX_WAIT_TIME = 300.0        # seconds – normalisation constant
PHASE_DURATION = 10          # simulation steps per phase action

# Supported reward modes
REWARD_MODE_COMPOSITE = "composite"
REWARD_MODE_PRESSURE  = "pressure"


class TrafficEnv:
    """
    Multi-agent SUMO traffic environment.

    Parameters
    ----------
    net_file : str
        Path to the SUMO ``.net.xml`` network file.
    route_file : str
        Path to the SUMO ``.rou.xml`` route file.
    ts_ids : list[str]
        Traffic-signal IDs to control.  If *None*, all signals found in
        the network are used.
    max_steps : int
        Maximum simulation steps per episode.
    use_gui : bool
        Launch SUMO-GUI instead of the headless binary.
    sumo_port : int
        TraCI port (use different values for parallel training workers).
    alpha, beta, gamma, delta : float
        Reward weighting coefficients (composite mode only).
    seed : int
        Random seed forwarded to SUMO.
    reward_mode : str
        ``"composite"`` (default) – weighted sum of throughput, queue,
        wait-time, and spillback components.
        ``"pressure"`` – negative absolute pressure (|incoming − outgoing|
        queue lengths), inspired by LucasAlegre/sumo-rl.
    use_phase_obs : bool
        When *True*, append two extra features to every observation vector:
        (normalised current-phase index, normalised time-in-phase).
        Inspired by AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        net_file: str,
        route_file: str,
        ts_ids: Optional[List[str]] = None,
        max_steps: int = 3600,
        use_gui: bool = False,
        sumo_port: int = 8813,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
        delta: float = DEFAULT_DELTA,
        seed: int = 42,
        reward_mode: str = REWARD_MODE_COMPOSITE,
        use_phase_obs: bool = False,
    ) -> None:
        self.net_file = net_file
        self.route_file = route_file
        self._requested_ts_ids = ts_ids
        self.max_steps = max_steps
        self.use_gui = use_gui
        self.sumo_port = sumo_port
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.seed = seed
        self.reward_mode  = reward_mode
        self.use_phase_obs = use_phase_obs

        # Populated in reset()
        self.ts_ids: List[str] = []
        self._lane_map: Dict[str, List[str]] = {}   # ts_id -> incoming lanes
        self._out_lane_map: Dict[str, List[str]] = {}  # ts_id -> outgoing lanes
        self._step: int = 0
        self._running: bool = False
        self._prev_halted: Dict[str, float] = {}
        self._prev_departed: int = 0
        # Phase-observation state
        self._phase_step: Dict[str, int] = {}   # ts_id -> steps in current phase

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Start (or restart) the SUMO simulation and return initial observations.
        """
        if self._running:
            self._close_sumo()

        self._start_sumo()
        self._step = 0
        self._running = True

        # Discover controlled intersections
        all_tls = traci.trafficlight.getIDList()
        if self._requested_ts_ids is not None:
            self.ts_ids = [t for t in self._requested_ts_ids if t in all_tls]
        else:
            self.ts_ids = list(all_tls)

        # Build lane mappings
        self._lane_map = {}
        self._out_lane_map = {}
        for ts in self.ts_ids:
            links = traci.trafficlight.getControlledLinks(ts)
            incoming, outgoing = set(), set()
            for link_group in links:
                for link in link_group:
                    if link:
                        incoming.add(link[0])
                        outgoing.add(link[1])
            self._lane_map[ts] = list(incoming)
            self._out_lane_map[ts] = list(outgoing)

        # Initialise previous-step bookkeeping
        self._prev_halted = {ts: 0.0 for ts in self.ts_ids}
        self._prev_departed = 0
        self._phase_step = {ts: 0 for ts in self.ts_ids}

        return {ts: self._get_observation(ts) for ts in self.ts_ids}

    # ------------------------------------------------------------------
    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict]:
        """
        Apply *actions* (phase index per agent) and advance one phase.

        Parameters
        ----------
        actions : dict[str, int]
            Mapping from traffic-signal ID to chosen green-phase index.

        Returns
        -------
        observations, rewards, done, info
        """
        assert self._running, "Call reset() before step()."

        # Apply actions (set phase for each controlled intersection)
        for ts, phase in actions.items():
            prev_phase = traci.trafficlight.getPhase(ts) if self.use_phase_obs else None
            _ = traci.trafficlight.getProgram(ts)
            logic = traci.trafficlight.getAllProgramLogics(ts)
            if logic:
                num_phases = len(logic[0].phases)
                phase = int(phase) % num_phases
            traci.trafficlight.setPhase(ts, phase)
            # Track how many steps we have been in this phase
            if self.use_phase_obs:
                if prev_phase != phase:
                    self._phase_step[ts] = 0
                else:
                    self._phase_step[ts] = self._phase_step[ts] + PHASE_DURATION

        # Advance simulation by PHASE_DURATION steps
        for _ in range(PHASE_DURATION):
            traci.simulationStep()
            self._step += 1
            if self._step >= self.max_steps:
                break

        obs = {ts: self._get_observation(ts) for ts in self.ts_ids}
        rewards = {ts: self._compute_reward(ts) for ts in self.ts_ids}

        done = self._step >= self.max_steps
        info = self._collect_info()

        # Update previous-step stats
        for ts in self.ts_ids:
            self._prev_halted[ts] = self._count_halted(ts)
        self._prev_departed = traci.simulation.getDepartedNumber()

        return obs, rewards, done, info

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Shut down the SUMO process."""
        if self._running:
            self._close_sumo()

    # ------------------------------------------------------------------
    def observation_space_size(self, ts_id: str) -> int:
        """Dimensionality of the observation vector for *ts_id*."""
        lanes = self._lane_map.get(ts_id, [])
        # Per lane: vehicle count + occupancy = 2 features; plus 1 downstream flag
        base = len(lanes) * 2 + len(self._out_lane_map.get(ts_id, []))
        # Optional phase features: current phase (normalised) + time in phase (normalised)
        return base + (2 if self.use_phase_obs else 0)

    def action_space_size(self, ts_id: str) -> int:
        """Number of discrete actions (green-phase combinations) for *ts_id*."""
        try:
            logics = traci.trafficlight.getAllProgramLogics(ts_id)
            if logics:
                return len(logics[0].phases)
        except Exception:
            pass
        return 4  # sensible default for a 4-way intersection

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_sumo(self) -> None:
        """Launch SUMO subprocess and connect via TraCI."""
        binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1",
            "--seed", str(self.seed),
            "--quit-on-end", "true",
        ]
        traci.start(sumo_cmd, port=self.sumo_port)

    def _close_sumo(self) -> None:
        try:
            traci.close()
        except Exception:
            pass
        self._running = False

    # ------------------------------------------------------------------
    def _get_observation(self, ts_id: str) -> np.ndarray:
        """
        Build the spatio-temporal feature vector for *ts_id*.

        Base features:
            [vehicle_count_lane_0, occupancy_lane_0, ...,
             vehicle_count_lane_N, occupancy_lane_N,
             downstream_spillback_flag_0, ..., downstream_spillback_flag_M]

        Optional phase features (when ``use_phase_obs=True``):
            [normalised_phase_index, normalised_time_in_phase]
            Inspired by AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control.
        """
        incoming = self._lane_map.get(ts_id, [])
        outgoing = self._out_lane_map.get(ts_id, [])
        features: List[float] = []

        for lane in incoming:
            try:
                n_veh = traci.lane.getLastStepVehicleNumber(lane)
                occ = traci.lane.getLastStepOccupancy(lane)  # 0–100 %
            except Exception:
                n_veh, occ = 0, 0.0
            features.append(min(n_veh / MAX_LANE_VEHICLES, 1.0))
            features.append(occ / 100.0)

        for lane in outgoing:
            try:
                occ = traci.lane.getLastStepOccupancy(lane) / 100.0
            except Exception:
                occ = 0.0
            features.append(float(occ > SPILLBACK_THRESHOLD))

        # ── Phase-time features (optional) ──────────────────────────────
        if self.use_phase_obs:
            try:
                logic = traci.trafficlight.getAllProgramLogics(ts_id)
                current_phase = traci.trafficlight.getPhase(ts_id)
                num_phases    = len(logic[0].phases) if logic else 4
            except Exception:
                current_phase, num_phases = 0, 4
            phase_norm      = current_phase / max(num_phases - 1, 1)
            phase_time      = self._phase_step[ts_id]
            # Normalise phase-time against a generous maximum (10 × phase duration)
            phase_time_norm = min(phase_time / (PHASE_DURATION * 10), 1.0)
            features.append(phase_norm)
            features.append(phase_time_norm)

        return np.array(features, dtype=np.float32)

    # ------------------------------------------------------------------
    def _compute_reward(self, ts_id: str) -> float:
        """
        Dispatch to the active reward mode.

        Modes
        -----
        composite (default):
            R = α·Throughput − β·Queue − γ·WaitTime − δ·SpillbackPenalty
        pressure (sumo-rl style):
            R = −|incoming_queue − outgoing_queue|
        """
        if self.reward_mode == REWARD_MODE_PRESSURE:
            return self._compute_pressure_reward(ts_id)
        return self._compute_composite_reward(ts_id)

    # ------------------------------------------------------------------
    def _compute_pressure_reward(self, ts_id: str) -> float:
        """
        Pressure reward inspired by LucasAlegre/sumo-rl.

        R = −|sum(halted_incoming) − sum(halted_outgoing)|

        Minimising pressure drives the agent to balance queues across
        the intersection rather than greedily serving one direction.
        """
        incoming = self._lane_map.get(ts_id, [])
        outgoing = self._out_lane_map.get(ts_id, [])

        in_queue_total = 0.0
        for lane in incoming:
            try:
                in_queue_total += traci.lane.getLastStepHaltingNumber(lane)
            except Exception:
                pass

        out_queue = 0.0
        for lane in outgoing:
            try:
                out_queue += traci.lane.getLastStepHaltingNumber(lane)
            except Exception:
                pass

        pressure = abs(in_queue_total - out_queue)
        # Normalise by the total number of lanes
        norm = max(len(incoming) + len(outgoing), 1)
        return float(-pressure / norm)

    def _compute_composite_reward(self, ts_id: str) -> float:
        """
        Composite fairness-aware reward (original IUTMS formula):
            R = α·Throughput − β·Queue − γ·WaitTime − δ·SpillbackPenalty
        """
        incoming = self._lane_map.get(ts_id, [])
        outgoing = self._out_lane_map.get(ts_id, [])

        # --- Throughput: number of vehicles that passed during this phase ---
        departed = traci.simulation.getDepartedNumber()
        throughput = max(departed - self._prev_departed, 0) / max(len(self.ts_ids), 1)

        # --- Queue: normalised sum of halted vehicles on incoming lanes ---
        total_halted = self._count_halted(ts_id)
        queue_penalty = total_halted / max(len(incoming) * MAX_LANE_VEHICLES, 1)

        # --- Waiting time: average cumulative wait per vehicle ---
        wait_times = []
        for lane in incoming:
            try:
                for veh_id in traci.lane.getLastStepVehicleIDs(lane):
                    wait_times.append(traci.vehicle.getWaitingTime(veh_id))
            except Exception:
                pass
        avg_wait = (np.mean(wait_times) if wait_times else 0.0) / MAX_WAIT_TIME

        # --- Spillback penalty ---
        spillback = 0.0
        for lane in outgoing:
            try:
                occ = traci.lane.getLastStepOccupancy(lane) / 100.0
                if occ > SPILLBACK_THRESHOLD:
                    spillback += occ - SPILLBACK_THRESHOLD
            except Exception:
                pass
        spillback_norm = spillback / max(len(outgoing), 1)

        reward = (
            self.alpha * throughput
            - self.beta * queue_penalty
            - self.gamma * avg_wait
            - self.delta * spillback_norm
        )
        return float(reward)

    # ------------------------------------------------------------------
    def _count_halted(self, ts_id: str) -> float:
        total = 0.0
        for lane in self._lane_map.get(ts_id, []):
            try:
                total += traci.lane.getLastStepHaltingNumber(lane)
            except Exception:
                pass
        return total

    # ------------------------------------------------------------------
    def _collect_info(self) -> Dict:
        """Return aggregate simulation metrics for telemetry.

        Performance note: ``traci.vehicle.getIDList()`` is called once and
        cached in a local variable so that the three previously separate calls
        (avg_speed, co2_emissions, vehicles_in_network) share a single TraCI
        round-trip instead of three.
        """
        try:
            vehicle_ids = list(traci.vehicle.getIDList())
        except Exception:
            vehicle_ids = []

        try:
            avg_speed = float(np.mean([traci.vehicle.getSpeed(v) for v in vehicle_ids])) if vehicle_ids else 0.0
        except Exception:
            avg_speed = 0.0

        try:
            co2 = float(sum(traci.vehicle.getCO2Emission(v) for v in vehicle_ids))
        except Exception:
            co2 = 0.0

        return {
            "step": self._step,
            "avg_speed": avg_speed,
            "co2_emissions": co2,
            "vehicles_in_network": len(vehicle_ids),
        }
