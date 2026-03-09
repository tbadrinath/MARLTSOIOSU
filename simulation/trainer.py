"""
trainer.py
----------
Multi-agent training loop for the Intelligent Urban Traffic Management System
(IUTMS).

Each episode:
1. Reset the SUMO environment → initial observations per intersection.
2. For every simulation step:
   a. Each DQN agent selects an action (ε-greedy).
   b. The environment advances by one phase.
   c. Rewards and next observations are collected.
   d. Transitions are stored in each agent's replay buffer.
   e. Each agent performs one learning step.
3. Episode metrics are logged and optionally sent to the Node.js telemetry
   server via HTTP POST.
4. Agent weights are saved periodically.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import requests  # type: ignore
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from simulation.env_wrapper import TrafficEnv
from simulation.agent import DQNAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # SUMO files
    "net_file": "maps/grid.net.xml",
    "route_file": "maps/grid.rou.xml",
    # Training
    "num_episodes": 200,
    "max_steps": 3600,
    "save_every": 20,           # save weights every N episodes
    "checkpoint_dir": "checkpoints",
    # DQN hyper-parameters
    "lr": 1e-3,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
    "batch_size": 64,
    "buffer_capacity": 10_000,
    "target_update": 100,
    # Reward weights
    "alpha": 0.4,
    "beta": 0.3,
    "gamma_reward": 0.2,
    "delta": 0.5,
    # Telemetry
    "telemetry_url": "http://localhost:3001/api/metrics",
    "use_gui": False,
    "sumo_port": 8813,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def post_metrics(url: str, payload: dict, timeout: float = 2.0) -> None:
    """Non-blocking POST to the Node.js telemetry server (best-effort)."""
    if not REQUESTS_AVAILABLE:
        return
    try:
        requests.post(url, json=payload, timeout=timeout)
    except Exception:
        pass  # telemetry loss is acceptable


def build_agents(
    env: TrafficEnv, cfg: dict
) -> Dict[str, DQNAgent]:
    """Instantiate one DQNAgent per traffic signal."""
    agents: Dict[str, DQNAgent] = {}
    for ts in env.ts_ids:
        state_dim = env.observation_space_size(ts)
        action_dim = env.action_space_size(ts)
        agents[ts] = DQNAgent(
            ts_id=ts,
            state_dim=state_dim,
            action_dim=action_dim,
            lr=cfg["lr"],
            gamma=cfg["gamma"],
            epsilon_start=cfg["epsilon_start"],
            epsilon_min=cfg["epsilon_min"],
            epsilon_decay=cfg["epsilon_decay"],
            batch_size=cfg["batch_size"],
            buffer_capacity=cfg["buffer_capacity"],
            target_update=cfg["target_update"],
        )
        logger.info(
            "Agent [%s] – state_dim=%d  action_dim=%d", ts, state_dim, action_dim
        )
    return agents


def save_agents(agents: Dict[str, DQNAgent], checkpoint_dir: str, episode: int) -> None:
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    for ts, agent in agents.items():
        path = os.path.join(checkpoint_dir, f"{ts}_ep{episode:04d}.pt")
        agent.save(path)
        logger.info("Saved checkpoint: %s", path)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: Optional[dict] = None) -> None:
    if cfg is None:
        cfg = DEFAULT_CONFIG

    env = TrafficEnv(
        net_file=cfg["net_file"],
        route_file=cfg["route_file"],
        max_steps=cfg["max_steps"],
        use_gui=cfg["use_gui"],
        sumo_port=cfg["sumo_port"],
        alpha=cfg["alpha"],
        beta=cfg["beta"],
        gamma=cfg["gamma_reward"],
        delta=cfg["delta"],
        seed=cfg["seed"],
    )

    # ------------------------------------------------------------------ #
    # First reset to discover intersections and build agents              #
    # ------------------------------------------------------------------ #
    logger.info("Starting SUMO and discovering intersections …")
    observations = env.reset()
    agents = build_agents(env, cfg)

    episode_rewards: list = []

    for episode in range(1, cfg["num_episodes"] + 1):
        if episode > 1:
            observations = env.reset()

        total_reward_per_agent: Dict[str, float] = {ts: 0.0 for ts in env.ts_ids}
        losses: Dict[str, list] = {ts: [] for ts in env.ts_ids}
        episode_start = time.time()

        done = False
        while not done:
            # ---------- Action selection ----------
            actions = {
                ts: agents[ts].select_action(observations[ts])
                for ts in env.ts_ids
            }

            # ---------- Environment step ----------
            next_obs, rewards, done, info = env.step(actions)

            # ---------- Store & learn ----------
            for ts in env.ts_ids:
                agents[ts].store(
                    observations[ts],
                    actions[ts],
                    rewards[ts],
                    next_obs[ts],
                    done,
                )
                loss = agents[ts].learn()
                if loss is not None:
                    losses[ts].append(loss)
                total_reward_per_agent[ts] += rewards[ts]

            observations = next_obs

            # ---------- Step-level telemetry ----------
            if cfg.get("telemetry_url"):
                step_metrics = {
                    "episode": episode,
                    "step": info.get("step", 0),
                    "avg_speed": info.get("avg_speed", 0.0),
                    "co2_emissions": info.get("co2_emissions", 0.0),
                    "total_reward": sum(total_reward_per_agent.values()),
                    "vehicles_in_network": info.get("vehicles_in_network", 0),
                }
                post_metrics(cfg["telemetry_url"], step_metrics)

        # ---------- Episode summary ----------
        mean_reward = np.mean(list(total_reward_per_agent.values()))
        mean_loss = np.mean(
            [np.mean(v) for v in losses.values() if v] or [0.0]
        )
        eps_sample = next(iter(agents.values())).epsilon if agents else 0.0
        elapsed = time.time() - episode_start

        episode_rewards.append(mean_reward)

        logger.info(
            "Episode %3d/%d | reward=%.3f | loss=%.4f | ε=%.3f | %.1fs",
            episode,
            cfg["num_episodes"],
            mean_reward,
            mean_loss,
            eps_sample,
            elapsed,
        )

        # ---------- Periodic checkpoints ----------
        if episode % cfg["save_every"] == 0:
            save_agents(agents, cfg["checkpoint_dir"], episode)

    # Final save
    save_agents(agents, cfg["checkpoint_dir"], cfg["num_episodes"])
    env.close()
    logger.info("Training complete. Mean episode reward: %.3f", np.mean(episode_rewards))


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MARL DQN agents for traffic signal control."
    )
    parser.add_argument("--net-file", default=DEFAULT_CONFIG["net_file"])
    parser.add_argument("--route-file", default=DEFAULT_CONFIG["route_file"])
    parser.add_argument("--episodes", type=int, default=DEFAULT_CONFIG["num_episodes"])
    parser.add_argument("--max-steps", type=int, default=DEFAULT_CONFIG["max_steps"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--gamma", type=float, default=DEFAULT_CONFIG["gamma"])
    parser.add_argument("--epsilon-start", type=float, default=DEFAULT_CONFIG["epsilon_start"])
    parser.add_argument("--epsilon-min", type=float, default=DEFAULT_CONFIG["epsilon_min"])
    parser.add_argument("--epsilon-decay", type=float, default=DEFAULT_CONFIG["epsilon_decay"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--alpha", type=float, default=DEFAULT_CONFIG["alpha"])
    parser.add_argument("--beta", type=float, default=DEFAULT_CONFIG["beta"])
    parser.add_argument("--gamma-reward", type=float, default=DEFAULT_CONFIG["gamma_reward"])
    parser.add_argument("--delta", type=float, default=DEFAULT_CONFIG["delta"])
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CONFIG["checkpoint_dir"])
    parser.add_argument("--telemetry-url", default=DEFAULT_CONFIG["telemetry_url"])
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--port", type=int, default=DEFAULT_CONFIG["sumo_port"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(
        {
            "net_file": args.net_file,
            "route_file": args.route_file,
            "num_episodes": args.episodes,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "gamma": args.gamma,
            "epsilon_start": args.epsilon_start,
            "epsilon_min": args.epsilon_min,
            "epsilon_decay": args.epsilon_decay,
            "batch_size": args.batch_size,
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma_reward": args.gamma_reward,
            "delta": args.delta,
            "checkpoint_dir": args.checkpoint_dir,
            "telemetry_url": args.telemetry_url,
            "use_gui": args.gui,
            "sumo_port": args.port,
            "seed": args.seed,
        }
    )
    train(cfg)
