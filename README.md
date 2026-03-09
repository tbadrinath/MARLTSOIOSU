# MARLTSOIOSU — Intelligent Urban Traffic Management System (IUTMS)

> **Multi-Agent Deep Reinforcement Learning for Traffic Signal Optimisation in
> Oversaturated Urban Networks**

[![Tests](https://img.shields.io/badge/tests-26%20passed-brightgreen)](#running-tests)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](#prerequisites)
[![SUMO](https://img.shields.io/badge/SUMO-1.15.0-orange)](#prerequisites)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#license)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Folder Structure](#folder-structure)
4. [Prerequisites](#prerequisites)
5. [Quick-Start](#quick-start)
6. [Configuration Reference](#configuration-reference)
7. [Reward Function](#reward-function)
8. [Dashboard](#dashboard)
9. [Running Tests](#running-tests)
10. [Success Metrics](#success-metrics)

---

## Overview

IUTMS replaces static fixed-time traffic timers with autonomous AI agents that
observe real-time traffic density and coordinate to prevent gridlock. Each
signalised intersection in a **3×3 urban grid** acts as an independent **Deep
Q-Network (DQN)** learner. Agents communicate implicitly through the shared
environment state; there is no global controller.

Key characteristics:

| Feature | Detail |
|---|---|
| Algorithm | Independent DQN with Experience Replay |
| Exploration | ε-greedy (1.0 → 0.05) |
| Observation | Spatio-temporal (lane queue + occupancy + downstream loops) |
| Reward | Composite fairness-aware formula (see [Reward Function](#reward-function)) |
| Simulation | SUMO 1.15.0 + TraCI |
| Telemetry | Express + Socket.io → React + Chart.js dashboard |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       SUMO Simulation                         │
│   ┌──────┐  ┌──────┐  ┌──────┐                              │
│   │  A0  │  │  B0  │  │  C0  │   ← 3×3 signalised grid     │
│   └──┬───┘  └──┬───┘  └──┬───┘                              │
│      │         │          │                                   │
│      ▼         ▼          ▼                                   │
│   env_wrapper.py  (TraCI interface + reward computation)      │
└──────────────────────┬───────────────────────────────────────┘
                       │ obs / reward
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  trainer.py  (multi-agent episode loop)                       │
│   DQNAgent(A0)  DQNAgent(B0)  …  DQNAgent(C2)               │
│       ↓ action         ↓ learn                                │
│   QNetwork  +  ReplayBuffer   (agent.py)                      │
└──────────────────────┬───────────────────────────────────────┘
                       │ POST /api/metrics
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  web/server/server.js   (Express + Socket.io)                 │
│   → broadcasts "step_metrics" to all ws clients              │
└──────────────────────┬───────────────────────────────────────┘
                       │ ws
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  web/client/  (React + Chart.js dashboard)                    │
│   • Reward per step    • Average vehicle speed               │
│   • Congestion Index   • CO₂ emissions                       │
└──────────────────────────────────────────────────────────────┘
```

### Neural network (per agent)

```
Input(state_dim)
    │
    ├─ FC(64) → ReLU
    ├─ FC(32) → ReLU
    └─ FC(action_dim)   ← Q-values for each green-phase combination
```

---

## Folder Structure

```
MARLTSOIOSU/
├── simulation/
│   ├── __init__.py
│   ├── env_wrapper.py   # TraCI environment + reward
│   ├── agent.py         # QNetwork, ReplayBuffer, DQNAgent
│   └── trainer.py       # Multi-agent training loop + CLI
├── maps/
│   ├── grid.net.xml     # 3×3 SUMO network
│   └── grid.rou.xml     # Vehicle flows (oversaturated scenario)
├── web/
│   ├── server/
│   │   ├── package.json
│   │   └── server.js    # Express + Socket.io telemetry server
│   └── client/
│       ├── package.json
│       ├── public/index.html
│       └── src/
│           ├── index.js
│           └── App.js   # React dashboard
├── tests/
│   ├── conftest.py
│   ├── test_agent.py
│   └── test_env_wrapper.py
├── requirements.txt
└── README.md
```

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | ≥ 3.9 | |
| SUMO | 1.15.0 | [sumo.dlr.de](https://sumo.dlr.de) |
| PyTorch | ≥ 2.0 | CPU build is sufficient for training |
| Node.js | ≥ 18 | Only needed for the dashboard |

After installing SUMO, make sure the `SUMO_HOME` environment variable is set:

```bash
export SUMO_HOME="/usr/share/sumo"          # Linux
export SUMO_HOME="/opt/homebrew/opt/sumo"   # macOS (Homebrew)
```

---

## Quick-Start

### 1 — Clone and install Python dependencies

```bash
git clone https://github.com/tbadrinath/MARLTSOIOSU.git
cd MARLTSOIOSU
pip install -r requirements.txt
```

### 2 — Install and start the telemetry server

```bash
cd web/server
npm install
npm start          # runs on http://localhost:3001
```

### 3 — Install and start the React dashboard

```bash
cd web/client
npm install
npm start          # opens http://localhost:3000
```

### 4 — Run the training loop

```bash
# From the repo root:
python -m simulation.trainer \
    --net-file  maps/grid.net.xml \
    --route-file maps/grid.rou.xml \
    --episodes  200 \
    --max-steps 3600 \
    --telemetry-url http://localhost:3001/api/metrics
```

Agent checkpoints are written to `checkpoints/` every 20 episodes.

To launch with the SUMO graphical interface:

```bash
python -m simulation.trainer --gui
```

---

## Configuration Reference

All CLI flags mirror the keys in `DEFAULT_CONFIG` in `simulation/trainer.py`.

| Flag | Default | Description |
|---|---|---|
| `--net-file` | `maps/grid.net.xml` | SUMO network file |
| `--route-file` | `maps/grid.rou.xml` | SUMO route file |
| `--episodes` | `200` | Number of training episodes |
| `--max-steps` | `3600` | Max simulation steps per episode |
| `--lr` | `1e-3` | Adam learning rate |
| `--gamma` | `0.99` | RL discount factor |
| `--epsilon-start` | `1.0` | Initial ε |
| `--epsilon-min` | `0.05` | Minimum ε |
| `--epsilon-decay` | `0.995` | ε multiplicative decay per learn step |
| `--batch-size` | `64` | Mini-batch size |
| `--alpha` | `0.4` | Reward: throughput weight |
| `--beta` | `0.3` | Reward: queue-length weight |
| `--gamma-reward` | `0.2` | Reward: waiting-time weight |
| `--delta` | `0.5` | Reward: spillback-penalty weight |
| `--checkpoint-dir` | `checkpoints` | Directory for weight files |
| `--telemetry-url` | `http://localhost:3001/api/metrics` | POST endpoint |
| `--gui` | `false` | Launch SUMO-GUI |
| `--port` | `8813` | TraCI port |
| `--seed` | `42` | Random seed |

---

## Reward Function

Each agent at intersection *i* receives:

$$R_i = \alpha \cdot \text{Throughput}_i - \beta \cdot \text{Queue}_i - \gamma \cdot \text{WaitTime}_i - \delta \cdot \text{SpillbackPenalty}_i$$

| Term | Description |
|---|---|
| **Throughput** | Fraction of vehicles that departed the network during this phase |
| **Queue** | Normalised count of halted vehicles on incoming lanes |
| **WaitTime** | Normalised average cumulative wait time per vehicle |
| **SpillbackPenalty** | Sum of excess occupancy on downstream lanes exceeding 90 % capacity — prevents *Main Road Bias* by forcing the agent to hold a red light even when the current approach is saturated |

---

## Dashboard

The React dashboard connects to the Socket.io server and displays four live
charts updated at every simulation step:

| Chart | Metric |
|---|---|
| Reward per Step | Combined agent reward |
| Average Vehicle Speed | Mean speed across all vehicles (m/s) |
| Congestion Index | Number of vehicles currently in the network |
| CO₂ Emissions | Total CO₂ per step (mg/s) |

A KPI row at the top shows the latest value of each metric at a glance.

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

Expected output: **26 passed**.

Tests cover `QNetwork`, `ReplayBuffer`, and `DQNAgent` in `test_agent.py`, and
the `TrafficEnv` observation, action-space, reward formula, and telemetry
helpers in `test_env_wrapper.py`. All tests run without a SUMO installation
(TraCI is fully mocked).

---

## Success Metrics

| KPI | Target |
|---|---|
| Throughput | > 15 % increase vs fixed-time baseline |
| Mean Waiting Time | < 45 s per vehicle during oversaturated peaks |
| Gridlock events | Zero over a simulated 24-hour cycle |

---

## License

MIT — see [LICENSE](LICENSE) for details.