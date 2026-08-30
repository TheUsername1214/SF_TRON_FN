<div align="center">

# SF_TRON_FN 🤖

**Teach a bipedal robot to walk — 2,000 of them at once.**

Built on [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/) + [PyTorch](https://pytorch.org/), this repo trains a locomotion policy with **PPO** + a **privileged-state estimator** (teacher–student learning), then ships it to the real TRON1 robot with zero privileged sensors.

> *Simulation is the playground. The robot is the test. This repo is the bridge.*

</div>

---

## Overview

`SF_TRON_FN` is a GPU-accelerated RL pipeline that teaches the **TRON1** bipedal robot to walk — and it does so by throwing **2,000 digital twins** into the arena at once. The entire loop (observe → act → reward → update) runs in **vectorized PyTorch**, so not a single CPU-GPU round-trip slows things down.

Here's the clever part — **privileged learning**:

- 🎓 The **student (policy)** only ever sees what the real robot can feel: joint states, IMU, body orientation, clock signals, and a velocity command.
- 🧠 A lightweight **teacher (estimator network)** learns to *hallucinate* the things the robot can't measure — external forces, body linear velocity, foot contact & ground-reaction forces — from a 10-step window of those same humble observations.
- 🚀 At deployment, the estimator's guesses are stitched onto the observations and fed to the policy. The robot never misses the sensors it never had.

Top it off with **aggressive domain randomization** (mass, COM, inertia, friction, restitution, PD gains, action delay, and random shoves), and you get a policy that survives the leap from pixels to pavement.

<div align="center">

```mermaid
flowchart LR
    subgraph Sim["🎮 Isaac Lab Simulation (up to 2000 parallel envs)"]
        E[TRON1 Robot] --> O[Proprioceptive Obs]
        O --> A[Actor Network]
        A -->|PD targets| E
    end
    subgraph Train["🏋️ Training (train=true)"]
        O -->|history window| EST[Estimator Network]
        PRIV[Privileged State] -->|MSE target| EST
        EST --> F[Full State]
        F --> A
        A & C[Critic] --> PPO[PPO + GAE Update]
    end
    subgraph Deploy["🚀 Deployment (train=false)"]
        O -->|history window| EST2[Estimator]
        EST2 --> F2[Full State]
        F2 --> A2[Policy]
        A2 -->|ONNX| R[Real Robot / ONNX Runtime]
    end
```

</div>

## Why It's Cool ✨

- ⚡ **Massively parallel training** — 2,000 vectorized agents on one GPU, zero Python loops over environments.
- 🎯 **PPO with GAE** — actor-critic with a shared MLP backbone, clipped surrogate objective, entropy regularization, and an action-smoothing term to keep the gait fluid.
- 🧠 **Privileged-state estimator** — a teacher that reconstructs the unobservable from proprioception, so the deployed policy sheds privileged sensors entirely.
- 🍰 **Rich reward shaping** — velocity tracking, body height/orientation, foot constraints, single-support gait, air-time, stand-still, and a firm "no falling" penalty.
- 🎲 **Domain randomization** — every robot is a unique snowflake: randomized mass, COM, inertia, friction, PD gains, action delay, and surprise external forces.
- 🦾 **PD torque control** — position-based PD controllers with randomized gains and simulated actuation delay, so sim feels like the real thing.
- 📦 **ONNX export** — actor and estimator ship as ONNX for on-robot inference.

## Robot & Task

| Item | Detail |
| --- | --- |
| 🤖 Robot | TRON1 bipedal robot (`Model/Robot_Model/SF_TRON1A.usd`) |
| 🦾 Actuators | 8 joints (6 legs + 2 ankles), implicit PD torque control |
| ⏱️ Control frequency | 50 Hz policy (`dt = 0.02 s`), 500 Hz simulation (`sub_step = 10`) |
| 📡 Sensors | body IMU, ankle IMUs, ankle contact sensors |
| 🎛️ Command | forward velocity command `vel_cmd` (`0` = stand still, `1` = walk) |
| 👀 Observation | 33-D proprioceptive state + 8-D estimated privileged state |

## Project Structure

```
SF_TRON_FN/
├── Run_with_Estimator.py        # Main training / evaluation loop
├── torch2onnx.py                # Export actor + estimator to ONNX
├── model1.onnx / model2.onnx    # Exported actor / estimator
├── SRC/
│   ├── Config/
│   │   ├── Config.py            # All hyperparameters (Env / Robot / PPO / Estimator)
│   │   └── TS_Config.py         # Teacher-student state-dim adjustment
│   ├── Env/
│   │   ├── BaseEnv.py           # Base environment (state, reset, randomization)
│   │   ├── TronEnv.py           # Task-specific observation, reward, step logic
│   │   ├── SceneSetup.py        # Isaac Lab scene + domain randomization
│   │   └── SoftwareSetup.py     # Isaac Lab AppLauncher setup
│   ├── PPO/
│   │   ├── Actor_Critic.py      # Actor / Critic networks + PPO update
│   │   └── Buffer.py            # Rollout buffer + GAE computation
│   ├── Estimator/
│   │   └── Estimator.py         # Privileged-state estimator (teacher network)
│   ├── Plotter/
│   │   └── ImagePlotter.py      # Real-time plotting of estimated vs. real GRF
│   └── Utils/
│       └── Transformation.py    # Euler/quaternion, yaw transform helpers
└── Model/
    ├── Robot_Model/             # Robot USD + URDF/xacro description
    ├── NN_Model/                # Trained weights (.pth)
    └── tron1-rl-deploy-python-main/  # MuJoCo-based deployment controller
```

## Installation

### Prerequisites

- Ubuntu 20.04/22.04 with an NVIDIA GPU (tested with Isaac Lab)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) (or Isaac Sim 4.x)
- Python 3.10+
- PyTorch with CUDA support

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Yiyou6/SF_TRON_FN.git
cd SF_TRON_FN

# 2. Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies (adjust for your Isaac Lab install)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install onnx onnxruntime matplotlib

# 4. Make sure Isaac Lab is on your PYTHONPATH, or run from within its environment
```

> 💡 **Note:** `SceneSetup.py` imports `isaaclab.*` packages. Run the scripts inside your Isaac Lab virtual environment so the `isaaclab` package resolves correctly.

## Usage

### 🏋️ Train

Set `train = True` in `SRC/Config/Config.py` (both `EnvParam.train` and `EnvParam.headless`), then run:

```bash
python Run_with_Estimator.py
```

The loop runs 3,000 episodes, saves the best actor/critic/estimator to `Model/NN_Model/` (`*_f.pth` keeps the latest, `*.pth` the best), and prints a per-episode reward breakdown after every update — watch the velocity, gait, and "stay upright" rewards climb.

### 🎬 Evaluate

Set `train = False` and `headless = False` in `SRC/Config/Config.py` to load the best models and watch the robot walk in the GUI:

```bash
python Run_with_Estimator.py
```

### 📦 Export to ONNX

```bash
python torch2onnx.py
```

This exports:
- `model1.onnx` — the actor (input: 33-D proprioceptive + 8-D estimated privileged = 41-D),
- `model2.onnx` — the estimator (input: 10-step history = 330-D).

## Configuration

All hyperparameters live in `SRC/Config/Config.py`, grouped by concern:

| Group | Highlights |
| --- | --- |
| `EnvCfg.EnvParam` | `agents_num` (2000), `dt` (0.02), `sub_step` (10), `train`, `device` |
| `RobotCfg.ActuatorParam` | 8 actuators, `Kp`/`Kd` gains, default PD angles |
| `RobotCfg.InitialState` | spawn height, initial pose/velocity ranges |
| `RobotCfg.DomainRandomizationCfg` | mass / COM / inertia / friction / restitution / PD-gain ranges, action delay, external force |
| `PPOCfg.CriticParam` | `state_dim` (33+8), layer width, learning rate |
| `PPOCfg.ActorParam` | action scale, std scale, layer width, learning rate |
| `PPOCfg.PPOParam` | `gamma`, `lam`, `epsilon`, `maximum_step` (25), `episode` (3000), `batch_size` |
| `PPOCfg.EstimatorParam` | `history_length` (10), `output_dim` (8), layer width, learning rate |

## License

This project does not currently specify a license. Add a `LICENSE` file if you intend to open-source it.
