"""
trainer.py
----------
Multi-agent training loop for the Intelligent Urban Traffic Management System
(IUTMS).

Supports two RL algorithms selectable via ``--algo``:
  * **dqn** (default) – Independent DQN with experience replay and target
    network.  Based on the approach in:
    - GameDisplayer/Deep-QLearning-Multi-Agent-Perspective-for-Traffic-Signal-Control
    - AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control
    - prajwal11660/-Intelligent-Traffic-Control-System
  * **ppo** – Independent PPO (actor-critic, clipped surrogate objective, GAE).
    Based on maxbrenner-ai/Multi-Agent-Distributed-PPO-Traffc-light-control and
    cts198859/deeprl_signal_control.

Supports two reward modes selectable via ``--reward``:
  * **composite** (default) – weighted combination of throughput, queue length,
    waiting time, and spillback penalty.
  * **pressure** – negative absolute pressure (|incoming − outgoing| queues),
    inspired by LucasAlegre/sumo-rl.

Each episode:
1. Reset the SUMO environment → initial observations per intersection.
2. For every simulation step:
   a. Each agent selects an action.
   b. The environment advances by one phase.
   c. Rewards and next observations are collected.
   d. Transitions are stored in each agent's buffer.
   e. Each agent performs one learning step (DQN: every step; PPO: when buffer full).
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
from typing import Dict, Optional, Union

import numpy as np

try:
    import requests  # type: ignore
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from simulation.env_wrapper import (
    TrafficEnv,
    REWARD_MODE_COMPOSITE,
    REWARD_MODE_PRESSURE,
)
from simulation.agent import DQNAgent
from simulation.ppo_agent import PPOAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Supported algorithm identifiers
ALGO_DQN = "dqn"
ALGO_PPO = "ppo"


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
    # Algorithm selection
    "algo": ALGO_DQN,           # "dqn" or "ppo"
    "reward_mode": REWARD_MODE_COMPOSITE,  # "composite" or "pressure"
    "use_phase_obs": False,     # include phase-time features in observation
    # DQN hyper-parameters
    "lr": 1e-3,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
    "batch_size": 64,
    "buffer_capacity": 10_000,
    "target_update": 100,
    # PPO hyper-parameters
    "ppo_lr": 3e-4,
    "ppo_n_steps": 512,
    "ppo_n_epochs": 10,
    "ppo_batch_size": 64,
    "ppo_clip_epsilon": 0.2,
    "ppo_gae_lambda": 0.95,
    "ppo_value_loss_coef": 0.5,
    "ppo_entropy_coef": 0.01,
    # Reward weights (composite mode)
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
) -> Dict[str, Union[DQNAgent, PPOAgent]]:
    """
    Instantiate one agent per traffic signal.

    The agent class is determined by ``cfg["algo"]``:
    * ``"dqn"`` → :class:`~simulation.agent.DQNAgent`
    * ``"ppo"`` → :class:`~simulation.ppo_agent.PPOAgent`
    """
    algo   = cfg.get("algo", ALGO_DQN).lower()
    agents: Dict[str, Union[DQNAgent, PPOAgent]] = {}

    for ts in env.ts_ids:
        state_dim  = env.observation_space_size(ts)
        action_dim = env.action_space_size(ts)

        if algo == ALGO_PPO:
            agents[ts] = PPOAgent(
                ts_id           = ts,
                state_dim       = state_dim,
                action_dim      = action_dim,
                lr              = cfg.get("ppo_lr", 3e-4),
                gamma           = cfg.get("gamma", 0.99),
                gae_lambda      = cfg.get("ppo_gae_lambda", 0.95),
                clip_epsilon    = cfg.get("ppo_clip_epsilon", 0.2),
                value_loss_coef = cfg.get("ppo_value_loss_coef", 0.5),
                entropy_coef    = cfg.get("ppo_entropy_coef", 0.01),
                n_steps         = cfg.get("ppo_n_steps", 512),
                n_epochs        = cfg.get("ppo_n_epochs", 10),
                batch_size      = cfg.get("ppo_batch_size", 64),
            )
        else:
            agents[ts] = DQNAgent(
                ts_id          = ts,
                state_dim      = state_dim,
                action_dim     = action_dim,
                lr             = cfg.get("lr", 1e-3),
                gamma          = cfg.get("gamma", 0.99),
                epsilon_start  = cfg.get("epsilon_start", 1.0),
                epsilon_min    = cfg.get("epsilon_min", 0.05),
                epsilon_decay  = cfg.get("epsilon_decay", 0.995),
                batch_size     = cfg.get("batch_size", 64),
                buffer_capacity= cfg.get("buffer_capacity", 10_000),
                target_update  = cfg.get("target_update", 100),
            )

        logger.info(
            "Agent [%s] algo=%s  state_dim=%d  action_dim=%d",
            ts, algo, state_dim, action_dim,
        )

    return agents


def save_agents(
    agents: Dict[str, Union[DQNAgent, PPOAgent]],
    checkpoint_dir: str,
    episode: int,
) -> None:
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

    algo        = cfg.get("algo", ALGO_DQN).lower()
    reward_mode = cfg.get("reward_mode", REWARD_MODE_COMPOSITE)

    env = TrafficEnv(
        net_file      = cfg["net_file"],
        route_file    = cfg["route_file"],
        max_steps     = cfg["max_steps"],
        use_gui       = cfg["use_gui"],
        sumo_port     = cfg["sumo_port"],
        alpha         = cfg["alpha"],
        beta          = cfg["beta"],
        gamma         = cfg["gamma_reward"],
        delta         = cfg["delta"],
        seed          = cfg["seed"],
        reward_mode   = reward_mode,
        use_phase_obs = cfg.get("use_phase_obs", False),
    )

    # ------------------------------------------------------------------ #
    # First reset to discover intersections and build agents              #
    # ------------------------------------------------------------------ #
    logger.info("Starting SUMO and discovering intersections (algo=%s reward=%s) …", algo, reward_mode)
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
                # PPO: pass last_state for GAE bootstrap; DQN: standard call
                if algo == ALGO_PPO:
                    loss = agents[ts].learn(last_state=next_obs[ts])
                else:
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
                    "algo": algo,
                    "reward_mode": reward_mode,
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
        description=(
            "Train MARL agents for traffic signal control.\n"
            "Supports DQN and PPO algorithms with composite or pressure reward."
        )
    )
    # SUMO files
    parser.add_argument("--net-file",    default=DEFAULT_CONFIG["net_file"])
    parser.add_argument("--route-file",  default=DEFAULT_CONFIG["route_file"])
    # Training
    parser.add_argument("--episodes",    type=int,   default=DEFAULT_CONFIG["num_episodes"])
    parser.add_argument("--max-steps",   type=int,   default=DEFAULT_CONFIG["max_steps"])
    # Algorithm & reward
    parser.add_argument(
        "--algo",
        choices=[ALGO_DQN, ALGO_PPO],
        default=DEFAULT_CONFIG["algo"],
        help="RL algorithm: 'dqn' (default) or 'ppo'",
    )
    parser.add_argument(
        "--reward",
        dest="reward_mode",
        choices=[REWARD_MODE_COMPOSITE, REWARD_MODE_PRESSURE],
        default=DEFAULT_CONFIG["reward_mode"],
        help="Reward mode: 'composite' (default) or 'pressure' (sumo-rl style)",
    )
    parser.add_argument(
        "--phase-obs",
        dest="use_phase_obs",
        action="store_true",
        help="Include current-phase index and time-in-phase in observation",
    )
    # DQN hyper-parameters
    parser.add_argument("--lr",            type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--gamma",         type=float, default=DEFAULT_CONFIG["gamma"])
    parser.add_argument("--epsilon-start", type=float, default=DEFAULT_CONFIG["epsilon_start"])
    parser.add_argument("--epsilon-min",   type=float, default=DEFAULT_CONFIG["epsilon_min"])
    parser.add_argument("--epsilon-decay", type=float, default=DEFAULT_CONFIG["epsilon_decay"])
    parser.add_argument("--batch-size",    type=int,   default=DEFAULT_CONFIG["batch_size"])
    # PPO hyper-parameters
    parser.add_argument("--ppo-lr",              type=float, default=DEFAULT_CONFIG["ppo_lr"])
    parser.add_argument("--ppo-n-steps",         type=int,   default=DEFAULT_CONFIG["ppo_n_steps"])
    parser.add_argument("--ppo-n-epochs",        type=int,   default=DEFAULT_CONFIG["ppo_n_epochs"])
    parser.add_argument("--ppo-batch-size",      type=int,   default=DEFAULT_CONFIG["ppo_batch_size"])
    parser.add_argument("--ppo-clip-epsilon",    type=float, default=DEFAULT_CONFIG["ppo_clip_epsilon"])
    parser.add_argument("--ppo-gae-lambda",      type=float, default=DEFAULT_CONFIG["ppo_gae_lambda"])
    parser.add_argument("--ppo-value-loss-coef", type=float, default=DEFAULT_CONFIG["ppo_value_loss_coef"])
    parser.add_argument("--ppo-entropy-coef",    type=float, default=DEFAULT_CONFIG["ppo_entropy_coef"])
    # Reward weights
    parser.add_argument("--alpha",       type=float, default=DEFAULT_CONFIG["alpha"])
    parser.add_argument("--beta",        type=float, default=DEFAULT_CONFIG["beta"])
    parser.add_argument("--gamma-reward",type=float, default=DEFAULT_CONFIG["gamma_reward"])
    parser.add_argument("--delta",       type=float, default=DEFAULT_CONFIG["delta"])
    # Infra
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CONFIG["checkpoint_dir"])
    parser.add_argument("--telemetry-url",  default=DEFAULT_CONFIG["telemetry_url"])
    parser.add_argument("--gui",  action="store_true")
    parser.add_argument("--port", type=int, default=DEFAULT_CONFIG["sumo_port"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(
        {
            "net_file":            args.net_file,
            "route_file":          args.route_file,
            "num_episodes":        args.episodes,
            "max_steps":           args.max_steps,
            "algo":                args.algo,
            "reward_mode":         args.reward_mode,
            "use_phase_obs":       args.use_phase_obs,
            "lr":                  args.lr,
            "gamma":               args.gamma,
            "epsilon_start":       args.epsilon_start,
            "epsilon_min":         args.epsilon_min,
            "epsilon_decay":       args.epsilon_decay,
            "batch_size":          args.batch_size,
            "ppo_lr":              args.ppo_lr,
            "ppo_n_steps":         args.ppo_n_steps,
            "ppo_n_epochs":        args.ppo_n_epochs,
            "ppo_batch_size":      args.ppo_batch_size,
            "ppo_clip_epsilon":    args.ppo_clip_epsilon,
            "ppo_gae_lambda":      args.ppo_gae_lambda,
            "ppo_value_loss_coef": args.ppo_value_loss_coef,
            "ppo_entropy_coef":    args.ppo_entropy_coef,
            "alpha":               args.alpha,
            "beta":                args.beta,
            "gamma_reward":        args.gamma_reward,
            "delta":               args.delta,
            "checkpoint_dir":      args.checkpoint_dir,
            "telemetry_url":       args.telemetry_url,
            "use_gui":             args.gui,
            "sumo_port":           args.port,
            "seed":                args.seed,
        }
    )
    train(cfg)
