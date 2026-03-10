"""
run_hyderabad.py
----------------
End-to-end simulation of MARL traffic-signal agents on the Hyderabad
HITEC City road network imported from OpenStreetMap.

This script:
  1. Converts the pre-embedded OSM file (``maps/hyderabad_hitec.osm``) into a
     SUMO road network using ``netconvert``.
  2. Generates realistic vehicle routes with ``randomTrips.py`` + ``duarouter``.
  3. Launches ``sumo`` (headless) or ``sumo-gui`` (with Xvfb) and connects via
     TraCI.
  4. Runs DQN-based multi-agent reinforcement learning agents — one per
     controlled intersection — for one episode.
  5. Captures per-step metrics (queue length, waiting time, throughput, CO₂).
  6. Generates a matplotlib-based animated screen recording (GIF) that renders
     the road network, moving vehicles, and real-time TLS states.
  7. Saves a metrics summary chart (PNG).

Usage
-----
  # Headless run — generates matplotlib-based GIF + metrics chart:
      python -m simulation.run_hyderabad

  # GUI mode — also captures sumo-gui screenshots via Xvfb:
      python -m simulation.run_hyderabad --gui

  # Custom number of steps:
      python -m simulation.run_hyderabad --steps 600

  # Re-use previously generated SUMO files (skip OSM conversion):
      python -m simulation.run_hyderabad --skip-convert

Output
------
  maps/hyderabad/map.net.xml               – SUMO road network
  maps/hyderabad/map.rou.xml               – Vehicle routes
  maps/hyderabad/simulation.sumocfg        – SUMO configuration
  maps/hyderabad/simulation_recording.gif  – Animated screen recording
  maps/hyderabad/metrics.png               – Metrics summary chart

References
----------
  - LucasAlegre/sumo-rl
  - AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control
  - SUMO TraCI documentation (https://sumo.dlr.de/docs/TraCI.html)
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT   = Path(__file__).resolve().parent.parent
MAPS_DIR    = REPO_ROOT / "maps"
OSM_FILE    = str(MAPS_DIR / "hyderabad_hitec.osm")
OUTPUT_DIR  = str(MAPS_DIR / "hyderabad")
NET_FILE    = str(Path(OUTPUT_DIR) / "map.net.xml")
ROUTE_FILE  = str(Path(OUTPUT_DIR) / "map.rou.xml")
CONFIG_FILE = str(Path(OUTPUT_DIR) / "simulation.sumocfg")
GIF_OUT     = str(Path(OUTPUT_DIR) / "simulation_recording.gif")
METRICS_PNG = str(Path(OUTPUT_DIR) / "metrics.png")

# Xvfb virtual display
XVFB_DISPLAY = ":99"

# Simulation parameters
DEFAULT_STEPS        = 500
DEFAULT_NUM_VEHICLES = 400
DEFAULT_SEED         = 42
SCREENSHOT_INTERVAL  = 25   # capture a frame every N steps


# ---------------------------------------------------------------------------
# SUMO environment setup
# ---------------------------------------------------------------------------

def _ensure_sumo_home() -> None:
    if os.environ.get("SUMO_HOME"):
        return
    for path in ["/usr/share/sumo", "/opt/sumo", "/usr/local/share/sumo"]:
        if Path(path).is_dir():
            os.environ["SUMO_HOME"] = path
            logger.info("SUMO_HOME auto-set to %s", path)
            return
    logger.warning("SUMO_HOME not found – some SUMO tools may be unavailable.")


def _start_xvfb() -> Optional["subprocess.Popen"]:
    """Start an Xvfb virtual framebuffer for off-screen rendering."""
    xvfb = shutil.which("Xvfb")
    if xvfb is None:
        logger.warning("Xvfb not found.")
        return None
    lock = Path(tempfile.gettempdir()) / f".X{XVFB_DISPLAY[1:]}-lock"
    lock.unlink(missing_ok=True)
    proc = subprocess.Popen(
        [xvfb, XVFB_DISPLAY, "-screen", "0", "1280x720x24"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = XVFB_DISPLAY
    time.sleep(1.5)
    logger.info("Xvfb started on %s (PID %d)", XVFB_DISPLAY, proc.pid)
    return proc


# ---------------------------------------------------------------------------
# OSM → SUMO conversion
# ---------------------------------------------------------------------------

def build_sumo_network(
    osm_file: str,
    output_dir: str,
    num_vehicles: int = DEFAULT_NUM_VEHICLES,
    seed: int = DEFAULT_SEED,
) -> None:
    """
    Convert the Hyderabad HITEC City OSM file to SUMO format.

    Uses netconvert + randomTrips + duarouter from the local SUMO installation.
    Writes ``map.net.xml``, ``map.rou.xml``, and ``simulation.sumocfg`` into
    *output_dir*.

    Falls back to the synthetic route writer if SUMO route tools are missing.
    """
    from simulation.osm_importer import (
        convert_to_sumo,
        generate_routes,
        generate_sumo_config,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Converting OSM → SUMO network …")
    convert_to_sumo(osm_file, NET_FILE)

    logger.info("Generating vehicle routes …")
    generate_routes(
        NET_FILE, ROUTE_FILE,
        num_vehicles=num_vehicles,
        seed=seed,
    )

    logger.info("Writing SUMO config …")
    generate_sumo_config(
        net_file    = NET_FILE,
        route_file  = ROUTE_FILE,
        config_file = CONFIG_FILE,
        begin       = 0,
        end         = DEFAULT_STEPS * 2,
        step_length = 1.0,
        seed        = seed,
    )

    logger.info("SUMO network ready in %s", output_dir)


# ---------------------------------------------------------------------------
# Simulation + MARL agents
# ---------------------------------------------------------------------------

def run_simulation(
    net_file: str,
    route_file: str,
    steps: int = DEFAULT_STEPS,
    use_gui: bool = False,
    seed: int = DEFAULT_SEED,
    output_dir: str = OUTPUT_DIR,
) -> Dict:
    """
    Run the Hyderabad HITEC City traffic simulation with DQN agents.

    Returns a dictionary of collected metrics.
    """
    try:
        import traci  # type: ignore
    except ImportError:
        raise RuntimeError(
            "TraCI not found.  Install SUMO and ensure 'traci' is importable."
        )

    from simulation.agent import DQNAgent

    # ── Start SUMO ──────────────────────────────────────────────────────────
    binary = "sumo-gui" if use_gui else "sumo"
    if shutil.which(binary) is None:
        binary = "sumo"
        use_gui = False

    sumo_cmd = [
        binary,
        "-n", net_file,
        "-r", route_file,
        "--no-step-log", "true",
        "--waiting-time-memory", "1000",
        "--time-to-teleport", "-1",
        "--seed", str(seed),
        "--quit-on-end", "true",
    ]
    if use_gui:
        sumo_cmd += ["--start", "true"]

    logger.info("Starting SUMO: %s", " ".join(sumo_cmd))
    traci.start(sumo_cmd, port=8814)

    try:
        return _run_episode(
            traci, steps, use_gui, output_dir
        )
    finally:
        try:
            traci.close()
        except Exception:
            pass


def _run_episode(
    traci,
    steps: int,
    use_gui: bool,
    output_dir: str,
) -> Dict:
    """Inner loop: discover TLS, create agents, step simulation."""
    from simulation.agent import DQNAgent
    from simulation.env_wrapper import (
        MAX_LANE_VEHICLES,
        MAX_WAIT_TIME,
        SPILLBACK_THRESHOLD,
        PHASE_DURATION,
        DEFAULT_ALPHA,
        DEFAULT_BETA,
        DEFAULT_GAMMA,
        DEFAULT_DELTA,
    )

    # ── Discover traffic signals ─────────────────────────────────────────
    all_tls = list(traci.trafficlight.getIDList())
    logger.info("Found %d traffic signals: %s", len(all_tls), all_tls)

    # Build lane maps
    lane_map: Dict[str, List[str]] = {}
    out_lane_map: Dict[str, List[str]] = {}
    for ts in all_tls:
        links = traci.trafficlight.getControlledLinks(ts)
        incoming, outgoing = set(), set()
        for link_group in links:
            for link in link_group:
                if link:
                    incoming.add(link[0])
                    outgoing.add(link[1])
        lane_map[ts] = list(incoming)
        out_lane_map[ts] = list(outgoing)

    # ── Build DQN agents ────────────────────────────────────────────────
    agents: Dict[str, DQNAgent] = {}
    for ts in all_tls:
        n_in  = len(lane_map[ts])
        n_out = len(out_lane_map[ts])
        state_dim  = n_in * 2 + n_out
        try:
            logics = traci.trafficlight.getAllProgramLogics(ts)
            action_dim = len(logics[0].phases) if logics else 4
        except Exception:
            action_dim = 4

        agents[ts] = DQNAgent(
            ts_id          = ts,
            state_dim      = max(state_dim, 1),
            action_dim     = max(action_dim, 2),
            lr             = 1e-3,
            gamma          = 0.99,
            epsilon_start  = 1.0,
            epsilon_min    = 0.05,
            epsilon_decay  = 0.995,
            batch_size     = 64,
            buffer_capacity= 10_000,
            target_update  = 100,
        )
        logger.info(
            "Agent %s: state_dim=%d action_dim=%d", ts, state_dim, action_dim
        )

    # ── Metrics accumulators ─────────────────────────────────────────────
    step_metrics: List[Dict] = []
    screenshots: List[str] = []
    screen_dir = Path(output_dir) / "frames"
    if use_gui:
        screen_dir.mkdir(parents=True, exist_ok=True)

    prev_halted: Dict[str, float] = {ts: 0.0 for ts in all_tls}
    total_reward: Dict[str, float] = {ts: 0.0 for ts in all_tls}

    sim_step = 0
    done = False

    while sim_step < steps:
        # ── Build observations ───────────────────────────────────────────
        obs: Dict[str, np.ndarray] = {}
        for ts in all_tls:
            features = []
            for lane in lane_map[ts]:
                try:
                    n_veh = traci.lane.getLastStepVehicleNumber(lane)
                    occ   = traci.lane.getLastStepOccupancy(lane)
                except Exception:
                    n_veh, occ = 0, 0.0
                features.append(min(n_veh / MAX_LANE_VEHICLES, 1.0))
                features.append(occ / 100.0)
            for lane in out_lane_map[ts]:
                try:
                    occ = traci.lane.getLastStepOccupancy(lane) / 100.0
                except Exception:
                    occ = 0.0
                features.append(float(occ > SPILLBACK_THRESHOLD))
            obs[ts] = np.array(features, dtype=np.float32) if features else np.zeros(1, dtype=np.float32)

        # ── Agent actions ────────────────────────────────────────────────
        actions: Dict[str, int] = {}
        for ts in all_tls:
            actions[ts] = agents[ts].select_action(obs[ts])

        # ── Apply actions & advance simulation ───────────────────────────
        for ts, phase in actions.items():
            try:
                logics = traci.trafficlight.getAllProgramLogics(ts)
                if logics:
                    num_phases = len(logics[0].phases)
                    phase = int(phase) % num_phases
                traci.trafficlight.setPhase(ts, phase)
            except Exception:
                pass

        for _ in range(PHASE_DURATION):
            traci.simulationStep()
            sim_step += 1
            if sim_step >= steps:
                break

        # ── Collect rewards ──────────────────────────────────────────────
        rewards: Dict[str, float] = {}
        for ts in all_tls:
            # Composite reward: throughput - queue - wait - spillback
            throughput = traci.simulation.getDepartedNumber()
            halted = sum(
                traci.lane.getLastStepHaltingNumber(lane)
                for lane in lane_map[ts]
                if _lane_exists(traci, lane)
            )
            wait = sum(
                traci.lane.getWaitingTime(lane)
                for lane in lane_map[ts]
                if _lane_exists(traci, lane)
            )
            spillback = sum(
                1 for lane in out_lane_map[ts]
                if _lane_exists(traci, lane) and
                traci.lane.getLastStepOccupancy(lane) / 100.0 > SPILLBACK_THRESHOLD
            )
            r = (
                DEFAULT_ALPHA * min(throughput / 10.0, 1.0)
                - DEFAULT_BETA  * min(halted / MAX_LANE_VEHICLES, 1.0)
                - DEFAULT_GAMMA * min(wait / MAX_WAIT_TIME, 1.0)
                - DEFAULT_DELTA * spillback
            )
            rewards[ts] = r
            total_reward[ts] = total_reward[ts] + r

        # ── Learn ────────────────────────────────────────────────────────
        next_obs: Dict[str, np.ndarray] = {}
        for ts in all_tls:
            features = []
            for lane in lane_map[ts]:
                try:
                    n_veh = traci.lane.getLastStepVehicleNumber(lane)
                    occ   = traci.lane.getLastStepOccupancy(lane)
                except Exception:
                    n_veh, occ = 0, 0.0
                features.append(min(n_veh / MAX_LANE_VEHICLES, 1.0))
                features.append(occ / 100.0)
            for lane in out_lane_map[ts]:
                try:
                    occ = traci.lane.getLastStepOccupancy(lane) / 100.0
                except Exception:
                    occ = 0.0
                features.append(float(occ > SPILLBACK_THRESHOLD))
            next_obs[ts] = np.array(features, dtype=np.float32) if features else np.zeros(1, dtype=np.float32)

        done_flag = sim_step >= steps
        for ts in all_tls:
            agents[ts].store(obs[ts], actions[ts], rewards[ts], next_obs[ts], done_flag)
            agents[ts].learn()

        # ── Per-step global metrics ──────────────────────────────────────
        try:
            vehicles_in_net = traci.vehicle.getIDCount()
            departed        = traci.simulation.getDepartedNumber()
            arrived         = traci.simulation.getArrivedNumber()
            avg_speed       = np.mean([
                traci.vehicle.getSpeed(v)
                for v in traci.vehicle.getIDList()
            ]) if vehicles_in_net > 0 else 0.0
            avg_wait = np.mean([
                traci.vehicle.getWaitingTime(v)
                for v in traci.vehicle.getIDList()
            ]) if vehicles_in_net > 0 else 0.0
            total_queue = sum(
                traci.lane.getLastStepHaltingNumber(lane)
                for ts in all_tls
                for lane in lane_map[ts]
                if _lane_exists(traci, lane)
            )
        except Exception:
            vehicles_in_net = 0
            departed = arrived = 0
            avg_speed = avg_wait = 0.0
            total_queue = 0

        step_metrics.append({
            "step":               sim_step,
            "vehicles_in_network": vehicles_in_net,
            "departed":           departed,
            "arrived":            arrived,
            "avg_speed_ms":       float(avg_speed),
            "avg_wait_s":         float(avg_wait),
            "total_queue":        int(total_queue),
            "total_reward":       float(sum(rewards.values())),
        })

        # ── Capture frame (GUI screenshot + matplotlib snapshot) ────────
        if sim_step % SCREENSHOT_INTERVAL == 0:
            # GUI screenshot (sumo-gui mode)
            if use_gui:
                frame_path = str(screen_dir / f"frame_{sim_step:05d}.png")
                try:
                    traci.gui.screenshot("View #0", frame_path)
                    logger.debug("GUI screenshot: %s", frame_path)
                except Exception as exc:
                    logger.debug("GUI screenshot failed at step %d: %s", sim_step, exc)

            # Collect vehicle positions and TLS states for matplotlib rendering
            try:
                veh_ids = traci.vehicle.getIDList()
                vehicle_positions = [
                    traci.vehicle.getPosition(v) for v in veh_ids
                ]
                vehicle_speeds = [
                    traci.vehicle.getSpeed(v) for v in veh_ids
                ]
                tls_states = {
                    ts: traci.trafficlight.getRedYellowGreenState(ts)
                    for ts in all_tls
                }
            except Exception:
                vehicle_positions = []
                vehicle_speeds = []
                tls_states = {}

            screenshots.append({
                "step":               sim_step,
                "vehicle_positions":  vehicle_positions,
                "vehicle_speeds":     vehicle_speeds,
                "tls_states":         tls_states,
                "metrics": {
                    "vehicles":  vehicles_in_net,
                    "avg_wait":  float(avg_wait),
                    "queue":     int(total_queue),
                    "reward":    float(sum(rewards.values())),
                },
            })

        if sim_step % 50 == 0:
            logger.info(
                "Step %4d/%d | vehicles=%d | avg_wait=%.1fs | queue=%d | reward=%.2f",
                sim_step, steps,
                vehicles_in_net, avg_wait, total_queue,
                sum(rewards.values()),
            )

    logger.info(
        "Episode complete. Total reward per agent: %s",
        {ts: f"{v:.2f}" for ts, v in total_reward.items()},
    )

    return {
        "step_metrics":  step_metrics,
        "screenshots":   screenshots,
        "total_reward":  total_reward,
        "num_agents":    len(all_tls),
        "agent_ids":     all_tls,
    }



def _lane_exists(traci, lane_id: str) -> bool:
    try:
        traci.lane.getLength(lane_id)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Network geometry helpers
# ---------------------------------------------------------------------------

def _load_network_geometry(net_file: str) -> Dict:
    """Parse the SUMO network for matplotlib rendering. Returns junctions/edges/bbox."""
    junctions = []
    edges = []
    tls_junctions = set()

    try:
        sumo_home = os.environ.get("SUMO_HOME", "/usr/share/sumo")
        tools_path = os.path.join(sumo_home, "tools")
        if tools_path not in sys.path:
            sys.path.insert(0, tools_path)
        import sumolib  # type: ignore
        net = sumolib.net.readNet(net_file, withInternal=False)
        for node in net.getNodes():
            junctions.append((node.getID(), node.getCoord()[0], node.getCoord()[1]))
            if node.getType() in ("traffic_light", "traffic_light_unregulated"):
                tls_junctions.add(node.getID())
        for edge in net.getEdges():
            if edge.getFunction() == "internal":
                continue
            shape = edge.getShape()
            if shape:
                edges.append(list(shape))
    except Exception as exc:
        logger.debug("sumolib parse failed (%s); using XML fallback.", exc)
        import xml.etree.ElementTree as ET
        tree = ET.parse(net_file)
        root = tree.getroot()
        for jn in root.findall("junction"):
            jid = jn.get("id", "")
            if jid.startswith(":"):
                continue
            try:
                x, y = float(jn.get("x", 0)), float(jn.get("y", 0))
            except ValueError:
                continue
            junctions.append((jid, x, y))
            if jn.get("type", "") in ("traffic_light", "traffic_light_unregulated"):
                tls_junctions.add(jid)
        for edge in root.findall("edge"):
            if edge.get("function", "") == "internal":
                continue
            for lane in edge.findall("lane"):
                shape_str = lane.get("shape", "")
                if not shape_str:
                    continue
                pts = []
                for pair in shape_str.strip().split():
                    try:
                        x2, y2 = map(float, pair.split(","))
                        pts.append((x2, y2))
                    except ValueError:
                        pass
                if pts:
                    edges.append(pts)
                    break

    if not junctions:
        return {"junctions": [], "edges": [], "tls_junctions": set(), "bbox": (0, 1, 0, 1)}
    xs2 = [j[1] for j in junctions]
    ys2 = [j[2] for j in junctions]
    return {
        "junctions":     junctions,
        "edges":         edges,
        "tls_junctions": tls_junctions,
        "bbox":          (min(xs2), max(xs2), min(ys2), max(ys2)),
    }


# ---------------------------------------------------------------------------
# Matplotlib frame renderer
# ---------------------------------------------------------------------------

def render_frame(
    net_geom: Dict,
    frame_data: Dict,
    output_path: str,
    total_steps: int,
) -> None:
    """Render one simulation frame showing road network, vehicles, TLS states."""
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.patches as mpatches  # type: ignore
    except ImportError:
        return

    BG = "#0D1117"; EDGE_CLR = "#2D3748"; JN_CLR = "#4A5568"
    TLS_GREEN = "#38A169"; TLS_RED = "#E53E3E"; TLS_YEL = "#D69E2E"
    VEH_FAST = "#63B3ED"; VEH_SLOW = "#F6AD55"; VEH_STOP = "#FC8181"

    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")

    bbox = net_geom["bbox"]
    pad  = max((bbox[1] - bbox[0]) * 0.07, 50)
    ax.set_xlim(bbox[0] - pad, bbox[1] + pad)
    ax.set_ylim(bbox[2] - pad, bbox[3] + pad)

    for shape in net_geom["edges"]:
        xs_e = [p[0] for p in shape]
        ys_e = [p[1] for p in shape]
        ax.plot(xs_e, ys_e, color=EDGE_CLR, linewidth=3.0, solid_capstyle="round", zorder=1)

    tls_ids    = net_geom["tls_junctions"]
    tls_states = frame_data.get("tls_states", {})
    for jid, jx, jy in net_geom["junctions"]:
        if jid in tls_ids:
            s = tls_states.get(jid, "")
            n_g = s.upper().count("G"); n_r = s.upper().count("R"); n_y = s.upper().count("Y")
            colour = TLS_GREEN if n_g >= n_r and n_g >= n_y else (TLS_YEL if n_y >= n_r else TLS_RED)
            ax.plot(jx, jy, "o", color=colour, markersize=13,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)
            ax.annotate(jid, (jx, jy), textcoords="offset points",
                        xytext=(5, 5), color="#CCC", fontsize=6, zorder=4)
        else:
            ax.plot(jx, jy, "o", color=JN_CLR, markersize=6, zorder=2)

    for (vx, vy), spd in zip(
        frame_data.get("vehicle_positions", []),
        frame_data.get("vehicle_speeds",    []),
    ):
        col = VEH_FAST if spd > 5.0 else (VEH_SLOW if spd > 0.5 else VEH_STOP)
        ax.plot(vx, vy, "s", color=col, markersize=5.5, alpha=0.9, zorder=5)

    handles = [
        mpatches.Patch(color=TLS_GREEN, label="TLS green phase"),
        mpatches.Patch(color=TLS_RED,   label="TLS red phase"),
        mpatches.Patch(color=VEH_FAST,  label="Vehicle (>5 m/s)"),
        mpatches.Patch(color=VEH_SLOW,  label="Vehicle (slow)"),
        mpatches.Patch(color=VEH_STOP,  label="Vehicle (stopped)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8,
              facecolor="#161B22", edgecolor="#333", labelcolor="white", framealpha=0.92)

    m    = frame_data.get("metrics", {})
    step = frame_data.get("step", 0)
    ax.set_title(
        f"Hyderabad HITEC City  ·  MARL DQN Traffic Signal Control\n"
        f"Step {step:4d}/{total_steps}  |  Vehicles: {m.get('vehicles', 0):3d}  "
        f"|  Avg Wait: {m.get('avg_wait', 0):5.1f} s  "
        f"|  Queue: {m.get('queue', 0):3d}  "
        f"|  Reward: {m.get('reward', 0):+.2f}",
        color="white", fontsize=10.5, pad=12, fontfamily="monospace",
    )
    ax.text(0.01, 0.01,
        "17.44–17.45°N  |  78.37–78.39°E  (HITEC City, Hyderabad, Telangana)",
        transform=ax.transAxes, color="#555", fontsize=7, va="bottom")

    progress = min(step / max(total_steps, 1), 1.0)
    axb = fig.add_axes([0.12, 0.03, 0.78, 0.018])
    axb.set_xlim(0, 1); axb.set_ylim(0, 1); axb.axis("off")
    axb.set_facecolor(BG)
    axb.axvspan(0, progress, color="#2196F3", alpha=0.85)
    axb.axvspan(progress, 1, color="#333", alpha=0.5)

    plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def make_gif(
    frames: List[Dict],
    gif_path: str,
    net_geom: Optional[Dict] = None,
    total_steps: int = DEFAULT_STEPS,
    fps: int = 3,
) -> bool:
    """Render frames with matplotlib and assemble into a GIF via ffmpeg."""
    try:
        import matplotlib  # type: ignore
    except ImportError:
        logger.warning("matplotlib not available – skipping GIF.")
        return False

    if not frames:
        logger.warning("No frame data to render.")
        return False

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        logger.warning("ffmpeg not found – skipping GIF generation.")
        return False

    if net_geom is None:
        net_geom = {"junctions": [], "edges": [], "tls_junctions": set(), "bbox": (0, 1, 0, 1)}

    frame_dir = Path(gif_path).parent / "mpl_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    png_paths: List[str] = []
    for frame_data in frames:
        step    = frame_data.get("step", 0)
        out_png = str(frame_dir / f"mpl_{step:05d}.png")
        render_frame(net_geom, frame_data, out_png, total_steps)
        if Path(out_png).exists():
            png_paths.append(out_png)

    if not png_paths:
        logger.warning("No rendered frames – GIF skipped.")
        return False

    list_file = str(Path(gif_path).parent / "ffmpeg_mpl_frames.txt")
    with open(list_file, "w") as fh:
        for png in png_paths:
            fh.write(f"file '{png}'\n")
            fh.write(f"duration {1.0/fps:.4f}\n")
        if png_paths:
            fh.write(f"file '{png_paths[-1]}'\n")
            fh.write("duration 1.0\n")

    cmd = [
        ffmpeg, "-y",
        "-f", "concat", "-safe", "0", "-i", list_file,
        "-vf",
        "scale=1200:-1:flags=lanczos,"
        "split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
        gif_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and Path(gif_path).exists():
        size_kb = Path(gif_path).stat().st_size // 1024
        logger.info("GIF saved: %s  (%d KB, %d frames)", gif_path, size_kb, len(png_paths))
        return True
    logger.warning("ffmpeg GIF failed: %s", result.stderr[:300])
    return False


def plot_metrics(step_metrics: List[Dict], png_path: str, total_reward: Dict) -> None:
    """Generate a dark-themed 6-panel metrics chart."""
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.gridspec as gridspec  # type: ignore
    except ImportError:
        logger.warning("matplotlib not available – skipping metrics chart.")
        return

    if not step_metrics:
        logger.warning("No step metrics to plot.")
        return

    xs       = [m["step"]               for m in step_metrics]
    vehicles = [m["vehicles_in_network"] for m in step_metrics]
    avg_wait = [m["avg_wait_s"]          for m in step_metrics]
    queue    = [m["total_queue"]         for m in step_metrics]
    reward   = [m["total_reward"]        for m in step_metrics]
    avg_spd  = [m["avg_speed_ms"]        for m in step_metrics]
    arrivals = list(np.cumsum([m["arrived"] for m in step_metrics]))

    C = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0D1117")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.36)

    for spec, x, y, colour, title, ylabel in [
        (gs[0, 0], xs, vehicles, C[0], "Vehicles in Network",    "Count"),
        (gs[0, 1], xs, avg_wait, C[1], "Avg Waiting Time (s)",   "Seconds"),
        (gs[0, 2], xs, queue,    C[2], "Total Queue Length",      "Halting veh."),
        (gs[1, 0], xs, reward,   C[3], "Per-Step Total Reward",   "Reward"),
        (gs[1, 1], xs, avg_spd,  C[4], "Avg Speed (m/s)",         "m/s"),
        (gs[1, 2], xs, arrivals, C[5], "Cumulative Arrivals",     "Vehicles"),
    ]:
        ax = fig.add_subplot(spec)
        ax.set_facecolor("#161B22")
        ax.plot(x, y, color=colour, linewidth=1.8, alpha=0.9)
        ax.fill_between(x, y, alpha=0.2, color=colour)
        ax.set_title(title, color="white", fontsize=11, pad=6)
        ax.set_xlabel("Simulation Step", color="#888", fontsize=9)
        ax.set_ylabel(ylabel, color="#888", fontsize=9)
        ax.tick_params(colors="#888", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.grid(True, color="#222", linestyle="--", linewidth=0.5)

    agent_str = "  |  ".join(f"{ts}: {v:.1f}" for ts, v in sorted(total_reward.items()))
    fig.suptitle(
        "Hyderabad HITEC City – MARL DQN Traffic Signal Control\n"
        f"Agent Cumulative Rewards │ {agent_str}",
        color="white", fontsize=12, y=0.98,
    )
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close(fig)
    logger.info("Metrics chart saved to %s", png_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate MARL DQN agents on the Hyderabad HITEC City OSM map.\n"
            "Produces a matplotlib animated GIF and a metrics PNG."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--osm-file",      default=OSM_FILE,      metavar="FILE")
    parser.add_argument("--output-dir",    default=OUTPUT_DIR,    metavar="DIR")
    parser.add_argument("--steps",         type=int, default=DEFAULT_STEPS,        metavar="N")
    parser.add_argument("--num-vehicles",  type=int, default=DEFAULT_NUM_VEHICLES, metavar="N")
    parser.add_argument("--seed",          type=int, default=DEFAULT_SEED)
    parser.add_argument("--gui",           action="store_true",
                        help="Also launch sumo-gui (Xvfb auto-started when DISPLAY unset).")
    parser.add_argument("--skip-convert",  action="store_true",
                        help="Skip OSM conversion and reuse existing SUMO network files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_sumo_home()

    output_dir  = args.output_dir
    net_file    = str(Path(output_dir) / "map.net.xml")
    route_file  = str(Path(output_dir) / "map.rou.xml")
    gif_out     = str(Path(output_dir) / "simulation_recording.gif")
    metrics_png = str(Path(output_dir) / "metrics.png")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if args.skip_convert and Path(net_file).exists() and Path(route_file).exists():
        logger.info("Skipping conversion – reusing existing files.")
    else:
        build_sumo_network(
            osm_file=args.osm_file, output_dir=output_dir,
            num_vehicles=args.num_vehicles, seed=args.seed,
        )

    net_geom = _load_network_geometry(net_file)
    logger.info(
        "Network geometry: %d junctions, %d edges, %d TLS",
        len(net_geom["junctions"]), len(net_geom["edges"]), len(net_geom["tls_junctions"]),
    )

    xvfb_proc = None
    use_gui   = args.gui
    if use_gui and not os.environ.get("DISPLAY", ""):
        xvfb_proc = _start_xvfb()
        if xvfb_proc is None:
            logger.warning("Xvfb unavailable – running headless.")
            use_gui = False

    try:
        result = run_simulation(
            net_file=net_file, route_file=route_file,
            steps=args.steps, use_gui=use_gui,
            seed=args.seed, output_dir=output_dir,
        )
    finally:
        if xvfb_proc is not None:
            xvfb_proc.terminate()
            logger.info("Xvfb stopped.")

    if result["screenshots"]:
        make_gif(
            frames=result["screenshots"], gif_path=gif_out,
            net_geom=net_geom, total_steps=args.steps,
        )
    else:
        logger.info("No frame data captured.")

    plot_metrics(result["step_metrics"], metrics_png, result["total_reward"])

    logger.info("=" * 62)
    logger.info("Simulation Summary – Hyderabad HITEC City (MARL DQN)")
    logger.info("  Agents         : %d  %s", result["num_agents"], result["agent_ids"])
    logger.info("  Steps run      : %d", args.steps)
    if result["total_reward"]:
        logger.info("  Mean reward    : %.3f",
                    float(np.mean(list(result["total_reward"].values()))))
    sm = result["step_metrics"]
    if sm:
        logger.info("  Avg wait (last 50): %.1f s",
                    float(np.mean([m["avg_wait_s"]  for m in sm[-50:]])))
        logger.info("  Avg queue (last 50): %.1f",
                    float(np.mean([m["total_queue"] for m in sm[-50:]])))
        logger.info("  Total arrivals : %d", int(sum(m["arrived"] for m in sm)))
    logger.info("  Metrics chart  : %s", metrics_png)
    if Path(gif_out).exists():
        logger.info("  Screen recording: %s", gif_out)
    logger.info("=" * 62)


if __name__ == "__main__":
    main()
