# SGU Team 2026 — Advanced AI Subject

This repository hosts coursework, classroom demonstrations, lab assignments, and research projects for the Advanced Artificial Intelligence course (SGU 2026).

---

## Repository Structure

```text
sgu-team-2026-advanced-ai-subject/
├── README.md                           # Course-wide overview and instructions
├── .gitignore                          # Repository ignore rules
├── requirements.txt                    # Base dependencies for course labs
├── FromClassroom/                      # Classroom lectures and code demos
├── lab01 - Agent/                      # Lab 01: Agent concepts, pursuit-evasion exercises
└── autonomous-driving-rl/              # Research subproject: Autonomous Driving in MetaDrive
    ├── README.md                       # Autonomous driving subproject overview
    ├── configs/                        # Experiment configurations & templates
    ├── docs/                           # Simulator docs, literature survey, and decisions
    ├── experiments/                    # Reproducible experiment runners
    ├── results/                        # Curated benchmark results
    ├── runs/                           # Local runtime artifacts (git-ignored)
    ├── scripts/                        # Utility scripts
    └── src/                            # Environments, agent policies, and logging
```

---

## 8-Stage Development Framework

To systematically study decision-making and learning methods, we propose an incremental 8-stage research and development framework:

- **Stage 0 — Baseline**: Random, naive, and sanity-check policies used to establish an empirical lower bound and validate the environment and evaluation pipeline.
- **Stage 1 — Rule-Based / Heuristic**: Handcrafted decision rules using observable state and domain knowledge (e.g., reactive safety filters, lane tracking).
- **Stage 2 — Planning / Search**: Explicit planning and search over discrete actions, trajectories, and state transitions.
- **Stage 3 — Simulation-Based Planning**: Forward simulation, trajectory rollouts, and Monte Carlo style planning. *(Note: "Simulation" refers to planning by evaluating simulated future outcomes, not merely using MetaDrive as the execution environment).*
- **Stage 4 — Learning from Data**: Supervised learning, imitation learning, and behavior cloning or other data-driven policies prior to full reinforcement learning.
- **Stage 5 — Model-Free Reinforcement Learning**: Learning policies and value functions directly through environment interaction without an explicit learned dynamics model.
- **Stage 6 — Search + Learning**: Combining learned policy/value functions with search/planning, inspired conceptually by systems such as AlphaZero. *(Note: This conceptual connection does not claim autonomous driving employs literal self-play).*
- **Stage 7 — Model-Based Reinforcement Learning**: Utilizing known or learned dynamics / world models for trajectory prediction and planning, conceptually related to model-based RL, MuZero, and world-model architectures.

> **Status Notice**: This 8-stage line represents a **proposed research/development framework**, not a finalized implementation plan. The project is currently **stopped at exploratory Stage 0 (Random Baseline)** until comprehensive literature survey and team design discussions are concluded.

---

## Autonomous Driving Subproject

The `autonomous-driving-rl/` subproject explores closed-loop driving control in urban and highway environments using the **MetaDrive** simulator.

For complete subproject documentation, observation/action specifications, and exploratory baseline findings, see:
👉 [autonomous-driving-rl/README.md](autonomous-driving-rl/README.md)

### Recommended Local Workspace Layout

To avoid Python package shadowing and keep the repository clean, MetaDrive source code resides outside the course repository as an external sibling:

```text
TTNTNC/
├── sgu-team-2026-advanced-ai-subject/   # Main course repository
└── metadrive-src/                       # MetaDrive source repository
```

### Quick MetaDrive Setup (Windows + Conda)

1. **Clone repositories side-by-side**:
   ```powershell
   cd D:\SGU\CNTT\TTNTNC
   git clone https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject.git
   git clone https://github.com/metadriverse/metadrive.git metadrive-src
   ```

2. **Pin verified MetaDrive commit** (for reproducibility):
   ```powershell
   cd D:\SGU\CNTT\TTNTNC\metadrive-src
   git checkout 85e5dadc6c7436d324348f6e3d8f8e680c06b4db
   ```

3. **Create Conda environment and install dependencies**:
   ```powershell
   conda create -n metadrive python=3.10 -y
   conda activate metadrive
   python -m pip install --upgrade pip setuptools wheel
   python -m pip install -e D:\SGU\CNTT\TTNTNC\metadrive-src
   ```

4. **Verify installation**:
   ```powershell
   cd D:\SGU\CNTT\TTNTNC\sgu-team-2026-advanced-ai-subject\autonomous-driving-rl
   python -c "import metadrive; print(metadrive.__file__); from metadrive import MetaDriveEnv; print('MetaDriveEnv import OK')"
   ```
   The printed path should point to `D:\SGU\CNTT\TTNTNC\metadrive-src\metadrive\__init__.py`.

For detailed troubleshooting and technical notes, refer to:
👉 [autonomous-driving-rl/docs/environment/METADRIVE_SETUP.md](autonomous-driving-rl/docs/environment/METADRIVE_SETUP.md)
