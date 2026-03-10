# MARLTSOIOSU — Intelligent Urban Traffic Management System (IUTMS)

> **Multi-Agent Deep Reinforcement Learning for Traffic Signal Optimisation in
> Oversaturated Urban Networks**

[![Tests](https://img.shields.io/badge/tests-76%20passed-brightgreen)](#running-tests)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](#prerequisites)
[![SUMO](https://img.shields.io/badge/SUMO-1.15.0-orange)](#prerequisites)
[![Windows EXE](https://img.shields.io/badge/windows-exe-blue)](https://github.com/tbadrinath/MARLTSOIOSU/releases/latest/download/IUTMS-Setup.exe)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#license)

---

## Table of Contents

1. [Overview](#overview)
2. [Algorithms](#algorithms)
3. [Reward Modes](#reward-modes)
4. [OSM Map Import](#osm-map-import)
5. [Architecture](#architecture)
6. [Folder Structure](#folder-structure)
7. [Prerequisites](#prerequisites)
8. [Windows EXE](#windows-exe)
9. [Quick-Start](#quick-start)
10. [Configuration Reference](#configuration-reference)
11. [Dashboard](#dashboard)
12. [Download the Full Codebase ZIP](#download-the-full-codebase-zip)
13. [Running Tests](#running-tests)
14. [References](#references)

---

## Overview

IUTMS replaces static fixed-time traffic timers with autonomous AI agents that
observe real-time traffic density and coordinate to prevent gridlock.  Each
signalised intersection in a network acts as an independent learner.

Key characteristics:

| Feature | Detail |
|---|---|
| Algorithms | **DQN** (ε-greedy, replay, target network) · **PPO** (actor-critic, GAE, clipped objective) |
| Reward modes | **Composite** (throughput + queue + wait + spillback) · **Pressure** (sumo-rl style) |
| Observation | Spatio-temporal (lane queue + occupancy + downstream loops) + optional phase-time features |
| Map import | Any city via OpenStreetMap → automatic SUMO network + route generation |
| Simulation | SUMO 1.15.0 + TraCI |
| Telemetry | Express + Socket.io → React + Chart.js real-time dashboard |

---

## Algorithms

### DQN — Deep Q-Network
*Inspired by prajwal11660/-Intelligent-Traffic-Control-System, GameDisplayer/Deep-QLearning-Multi-Agent-Perspective-for-Traffic-Signal-Control, and AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control.*

```
Input(state_dim)
    │
    ├─ FC(64) → ReLU
    ├─ FC(32) → ReLU
    └─ FC(action_dim)   ← Q-values
```

- ε-greedy exploration (1.0 → 0.05 with exponential decay)
- Fixed-size circular experience replay buffer (default 10 000)
- Periodic hard update of target network (every 100 learn steps)
- Huber loss for numerical stability

**CLI:**
```bash
python -m simulation.trainer --algo dqn --episodes 200
```

### PPO — Proximal Policy Optimisation
*Inspired by maxbrenner-ai/Multi-Agent-Distributed-PPO-Traffc-light-control and cts198859/deeprl_signal_control.*

```
Shared backbone:  Input(state_dim) → FC(128) → ReLU → FC(64) → ReLU
Actor head:       FC(64) → action_dim  (softmax → categorical policy)
Critic head:      FC(64) → 1           (state-value V(s))
```

- On-policy rollout buffer (`n_steps = 512`)
- Generalised Advantage Estimation (GAE, λ = 0.95)
- Clipped surrogate objective (ε = 0.2) — prevents destructively large updates
- Entropy bonus (c₂ = 0.01) — sustains exploration
- Value-function clipping for stable critic training

**CLI:**
```bash
python -m simulation.trainer --algo ppo --ppo-n-steps 512 --ppo-n-epochs 10
```

---

## Reward Modes

### Composite (default)
*Original IUTMS formula — fairness-aware, prevents Main Road Bias.*

$$R_i = \alpha \cdot \text{Throughput}_i - \beta \cdot \text{Queue}_i - \gamma \cdot \text{WaitTime}_i - \delta \cdot \text{SpillbackPenalty}_i$$

| Term | Description |
|---|---|
| **Throughput** | Fraction of vehicles that departed the network during this phase |
| **Queue** | Normalised count of halted vehicles on incoming lanes |
| **WaitTime** | Normalised average cumulative wait time per vehicle |
| **SpillbackPenalty** | Excess occupancy on downstream lanes > 90 % capacity |

```bash
python -m simulation.trainer --reward composite --alpha 0.4 --beta 0.3
```

### Pressure
*Inspired by LucasAlegre/sumo-rl — simple, scale-invariant.*

$$R_i = -\frac{|\text{in\_queue} - \text{out\_queue}|}{\text{num\_lanes}}$$

Minimising pressure balances queues across the intersection rather than
greedily serving one approach.

```bash
python -m simulation.trainer --reward pressure
```

### Phase-Time Observations
*Inspired by AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control.*

When `--phase-obs` is set, two extra features are appended to each agent's
observation vector:
1. **Normalised current-phase index** — which green phase is currently active
2. **Normalised time-in-phase** — how many steps the current phase has been active

```bash
python -m simulation.trainer --phase-obs
```

---

## OSM Map Import

IUTMS can simulate traffic on **any real city** by downloading road-network
data from OpenStreetMap and converting it automatically to SUMO format.

### From the dashboard

1. Open the **🗺️ OSM Map Importer** panel in the dashboard.
2. Search for any city or location (e.g. *"Downtown Toronto, Canada"*).
3. Preview the area on the embedded OSM map.
4. Click **🚀 Import & Simulate**.

The pipeline runs server-side:
```
Nominatim geocoding
    ↓
Overpass API download (highway ways + nodes)
    ↓
netconvert → .net.xml
    ↓
randomTrips.py + duarouter → .rou.xml
```

### From the CLI

```python
from simulation.osm_importer import import_map

result = import_map("Manhattan, New York", output_dir="maps/osm/manhattan")
# → result["net_file"], result["route_file"]
```

Then run training on the imported map:
```bash
python -m simulation.trainer \
    --net-file  maps/osm/manhattan/map.net.xml \
    --route-file maps/osm/manhattan/map.rou.xml \
    --algo ppo --reward pressure
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       SUMO Simulation                         │
│   ┌──────┐  ┌──────┐  ┌──────┐                              │
│   │  A0  │  │  B0  │  │  C0  │   ← signalised intersections │
│   └──┬───┘  └──┬───┘  └──┬───┘                              │
│      │         │          │                                   │
│      ▼         ▼          ▼                                   │
│   env_wrapper.py  (TraCI · reward: composite|pressure)        │
└──────────────────────┬───────────────────────────────────────┘
                       │ obs / reward
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  trainer.py  (multi-agent episode loop, --algo dqn|ppo)       │
│   DQNAgent(A0)   PPOAgent(B0)   …                            │
│       ↓ action         ↓ learn                                │
│   agent.py (QNetwork + ReplayBuffer)                          │
│   ppo_agent.py (ActorCriticNetwork + RolloutBuffer)           │
└──────────────────────┬───────────────────────────────────────┘
                       │ POST /api/metrics
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  web/server/server.js   (Express + Socket.io)                 │
│   /api/metrics            – ingest step metrics              │
│   /api/osm/search         – Nominatim proxy                  │
│   /api/osm/import         – OSM → SUMO pipeline              │
└──────────────────────┬───────────────────────────────────────┘
                       │ ws
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  web/client/  (React + Chart.js dashboard)                    │
│   • Reward · Speed · Congestion · CO₂ (live charts)          │
│   • Algorithm & reward selector in ConfigPanel               │
│   • OSM Map Importer panel                                    │
└──────────────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
MARLTSOIOSU/
├── simulation/
│   ├── __init__.py
│   ├── env_wrapper.py   # TraCI environment + composite/pressure reward
│   ├── agent.py         # QNetwork, ReplayBuffer, DQNAgent
│   ├── ppo_agent.py     # ActorCriticNetwork, RolloutBuffer, PPOAgent
│   ├── trainer.py       # Multi-agent loop, --algo dqn|ppo, --reward ...
│   └── osm_importer.py  # Nominatim → Overpass → netconvert → routes
├── maps/
│   ├── grid.net.xml     # 3×3 SUMO network
│   ├── grid.rou.xml     # Vehicle flows (oversaturated scenario)
│   └── osm/             # Downloaded maps (created at runtime)
├── web/
│   ├── server/
│   │   ├── package.json
│   │   └── server.js    # Express + Socket.io + OSM endpoints
│   └── client/
│       ├── package.json
│       ├── public/index.html
│       └── src/
│           ├── index.js
│           └── App.js   # React dashboard (charts + OSM panel + config)
├── tests/
│   ├── conftest.py
│   ├── test_agent.py          # DQN tests
│   ├── test_ppo_agent.py      # PPO tests
│   ├── test_env_wrapper.py    # Environment + reward mode tests
│   └── test_osm_importer.py   # OSM pipeline tests (all mocked)
├── requirements.txt
└── README.md
```

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | ≥ 3.9 | |
| SUMO | 1.15.0 | [sumo.dlr.de](https://sumo.dlr.de) · required for simulation |
| PyTorch | ≥ 2.0 | CPU build is sufficient for training |
| Node.js | ≥ 18 | Only needed for the dashboard server |

After installing SUMO, set `SUMO_HOME`:

```bash
export SUMO_HOME="/usr/share/sumo"          # Linux
export SUMO_HOME="/opt/homebrew/opt/sumo"   # macOS (Homebrew)
```

---

## Windows EXE

- **Latest installer:** [Download `IUTMS-Setup.exe`](https://github.com/tbadrinath/MARLTSOIOSU/releases/latest/download/IUTMS-Setup.exe)
- **Portable bundle:** [Latest release assets](https://github.com/tbadrinath/MARLTSOIOSU/releases/latest)

The Windows release workflow now builds a self-contained installer that bundles:

- the Python-based SUMO launcher as `IUTMS-GUI.exe`
- the Node/Express telemetry server as `IUTMS-Server.exe`
- the pre-built React dashboard assets
- the bundled maps in this repository

This means Windows users do **not** need to download Python, Node.js, npm packages,
or pip packages every time they want to run the app. The only external runtime
prerequisite that still needs to be installed separately for live SUMO simulations
is **SUMO** itself.

To publish a fresh Windows installer, run the **Build Windows EXE** GitHub Actions
workflow or push a version tag (for example `v1.0.0`). The workflow uploads the
generated installer to the release page with the stable filename
`IUTMS-Setup.exe`, which keeps the download link above unchanged.

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
cd web/server && npm install && npm start   # http://localhost:3001
```

### 3 — Install and start the React dashboard

```bash
cd web/client && npm install && npm start   # http://localhost:3000
```

### 4 — Run training

```bash
# DQN with composite reward (default)
python -m simulation.trainer \
    --net-file maps/grid.net.xml --route-file maps/grid.rou.xml \
    --algo dqn --reward composite --episodes 200

# PPO with pressure reward
python -m simulation.trainer \
    --net-file maps/grid.net.xml --route-file maps/grid.rou.xml \
    --algo ppo --reward pressure --episodes 200

# PPO on an OSM-imported city map with phase observations
python -m simulation.trainer \
    --net-file  maps/osm/manhattan/map.net.xml \
    --route-file maps/osm/manhattan/map.rou.xml \
    --algo ppo --reward pressure --phase-obs
```

---

## Configuration Reference

| Flag | Default | Description |
|---|---|---|
| `--algo` | `dqn` | Algorithm: `dqn` or `ppo` |
| `--reward` | `composite` | Reward mode: `composite` or `pressure` |
| `--phase-obs` | off | Append phase-index + time-in-phase to observation |
| `--net-file` | `maps/grid.net.xml` | SUMO network file |
| `--route-file` | `maps/grid.rou.xml` | SUMO route file |
| `--episodes` | `200` | Training episodes |
| `--max-steps` | `3600` | Simulation steps per episode |
| `--lr` | `1e-3` | DQN Adam learning rate |
| `--epsilon-start/min/decay` | `1.0/0.05/0.995` | DQN ε-greedy schedule |
| `--ppo-lr` | `3e-4` | PPO learning rate |
| `--ppo-n-steps` | `512` | PPO rollout length |
| `--ppo-n-epochs` | `10` | PPO optimisation epochs |
| `--ppo-clip-epsilon` | `0.2` | PPO clip coefficient |
| `--ppo-gae-lambda` | `0.95` | GAE smoothing λ |
| `--alpha` | `0.4` | Composite reward: throughput weight |
| `--beta` | `0.3` | Composite reward: queue weight |
| `--gamma-reward` | `0.2` | Composite reward: wait-time weight |
| `--delta` | `0.5` | Composite reward: spillback weight |
| `--checkpoint-dir` | `checkpoints` | Where to save model weights |
| `--gui` | off | Launch SUMO graphical interface |
| `--seed` | `42` | Random seed |

---

## Dashboard

The React dashboard connects to the Socket.io server and displays four live
charts updated at every simulation step:

| Chart | Metric |
|---|---|
| Reward per Step | Combined agent reward |
| Average Vehicle Speed | Mean speed across all vehicles (m/s) |
| Congestion Index | Number of vehicles in the network |
| CO₂ Emissions | Total CO₂ per step (mg/s) |

Additional panels:
- **🗺️ OSM Map Importer** — search, preview, and import any city's road network
- **Training configuration** — interactive selector for DQN vs PPO and reward mode, with CLI command preview
- **About** — algorithm architecture summary
- **Export CSV** — download the current chart data

---

## Download the Full Codebase ZIP

You can package the complete IUTMS project source into a single zip archive in
either of these ways:

### From the server/dashboard

Start the telemetry server and use the **🗜 Download Project ZIP** button in the
dashboard, or download it directly from:

```text
GET /api/export/codebase
```

### From the CLI

```bash
python -m simulation.codebase_exporter --output /absolute/path/to/project-codebase.zip
```

The generated archive contains the project source and configuration files while
excluding transient directories such as `.git`, `node_modules`, and Python
cache files.

---

## Running Tests

```bash
pip install pytest torch numpy requests
python -m pytest tests/ -v
```

Expected output: **76 passed**.

| File | Coverage |
|---|---|
| `test_agent.py` | QNetwork, ReplayBuffer, DQNAgent |
| `test_ppo_agent.py` | ActorCriticNetwork, RolloutBuffer, PPOAgent |
| `test_env_wrapper.py` | TrafficEnv (obs space, reward modes, phase-obs) |
| `test_osm_importer.py` | OSM pipeline (all network calls mocked) |

All tests run without a SUMO installation (TraCI and HTTP calls are fully mocked).

---

## References

| Repository | Contribution |
|---|---|
| [prajwal11660/-Intelligent-Traffic-Control-System](https://github.com/prajwal11660/-Intelligent-Traffic-Control-System) | DQN baseline architecture |
| [GameDisplayer/Deep-QLearning-Multi-Agent-Perspective-for-Traffic-Signal-Control](https://github.com/GameDisplayer/Deep-QLearning-Multi-Agent-Perspective-for-Traffic-Signal-Control) | Multi-agent DQN patterns |
| [maxbrenner-ai/Multi-Agent-Distributed-PPO-Traffc-light-control](https://github.com/maxbrenner-ai/Multi-Agent-Distributed-PPO-Traffc-light-control) | PPO agent (ppo_agent.py) |
| [cts198859/deeprl_signal_control](https://github.com/cts198859/deeprl_signal_control) | Actor-critic patterns, training structure |
| [LucasAlegre/sumo-rl](https://github.com/LucasAlegre/sumo-rl) | Pressure reward, OSM conversion pattern |
| [AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control](https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control) | Phase-duration observations, OSM setup |

---

## License

MIT — see [LICENSE](LICENSE) for details.


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
