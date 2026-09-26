# Autonomous Driving Subproject (MetaDrive)

This subproject focuses on research and development of decision-making, planning, and learning algorithms for autonomous driving using the [MetaDrive](https://github.com/metadriverse/metadrive) simulator.

---

## 1. Project Purpose & Scope

The objective of this research track within the SGU Advanced AI course is to systematically investigate autonomous driving agents across an 8-stage progression:
- **Stage 0**: Random / Sanity-check Baseline
- **Stage 1**: Rule-Based / Heuristic
- **Stage 2**: Planning / Search
- **Stage 3**: Simulation-Based Planning
- **Stage 4**: Learning from Data (Imitation Learning / Behavior Cloning)
- **Stage 5**: Model-Free Reinforcement Learning
- **Stage 6**: Search + Learning (AlphaZero-inspired planning + value/policy networks)
- **Stage 7**: Model-Based Reinforcement Learning (Learned world models / MuZero)

### Current Implementation Status: Stopped at Exploratory Stage 0

> ⚠️ **Important Note**:
> The subproject is **strictly at Stage 0 (Baseline / Exploratory Prototype)**.
> **No Stage 1 (Rule-Based) or later agents are implemented.**
> The immediate next steps are literature review, requirement analysis, and architectural design discussions, **NOT** jumping straight into coding Stage 1 agents.

---

## 2. Directory Layout & MetaDrive Dependency

To keep the repository lightweight and prevent Python module shadowing conflicts, MetaDrive source code is hosted as an **external sibling repository**:

```text
D:\SGU\CNTT\TTNTNC\
├── sgu-team-2026-advanced-ai-subject\          # Main course repository
│   └── autonomous-driving-rl\                 # This subproject directory
│       ├── README.md                          # Subproject documentation
│       ├── configs/                           # Experiment configs and schema templates
│       ├── docs/                              # Environment guides, survey notes, ADRs
│       ├── experiments/                       # Experiment execution scripts
│       ├── results/                           # Curated benchmark results
│       ├── runs/                              # Transient execution logs (git-ignored)
│       ├── scripts/                           # Utility diagnostic scripts
│       ├── src/                               # Python packages: environments, agents, logging
│       │   ├── agents/                        # Agent implementations (Stage 0 only)
│       │   ├── environments/                  # Environment wrappers (CourseEnvV1)
│       │   ├── evaluation/                    # Evaluation harness (evaluate_random.py)
│       │   └── logging/                       # Lightweight logging scaffold
│       ├── inspect_metadrive.py               # Exploratory inspection script
│       ├── inspect_observation.py             # Vehicle state inspection
│       ├── inspect_observation_parts.py       # Decomposition inspection (Ego, Nav, LiDAR)
│       ├── inspect_dynamic_observation.py     # Stepping & dynamic sensor inspection
│       ├── inspect_reward_config.py           # MetaDrive default reward inspection
│       └── test_course_env.py                 # CourseEnvV1 test runner
└── metadrive-src\                              # External MetaDrive repository (0.4.3)
```

For complete installation steps and package shadowing guidance, see:
👉 [docs/environment/METADRIVE_SETUP.md](docs/environment/METADRIVE_SETUP.md)

---

## 3. Current Exploratory Environment Interface

The initial exploratory environment wrapper (`CourseEnvV1` in `src/environments/course_env_v1.py`) wraps MetaDrive with an experimental compact observation and discrete action space.

*Note: These interfaces are exploratory prototypes designed for initial pipeline validation, NOT finalized architectural commitments.*

### Observation Space
- **Raw MetaDrive observation**: `Box(..., (259,), float32)`
  - **Ego vehicle state** (9D): Left/right boundary distances, heading difference, velocity, steering, previous steering, previous throttle/brake, yaw rate, lateral position.
  - **Navigation checkpoints** (10D): Checkpoints forward/lateral projections, lane radius, curvature direction, lane angle.
  - **LiDAR cloud** (240D): 240 laser detection rays normalized in `[0.0, 1.0]`.
- **Experimental Course Observation** (35D):
  - 19D (9 Ego + 10 Navigation)
  - 16D compressed LiDAR sectors (taking the minimum distance across 16 equal partitions of the 240 rays).

### Action Space
- **Native MetaDrive action space**: Continuous `Box(-1.0, 1.0, (2,), float32)` representing `[steering, throttle_brake]`.
- **Experimental Course Action Space**: `Discrete(5)`
  - `0`: LEFT `[-0.35, 0.35]`
  - `1`: STRAIGHT `[0.00, 0.40]`
  - `2`: RIGHT `[0.35, 0.35]`
  - `3`: ACCELERATE `[0.00, 0.80]`
  - `4`: BRAKE `[0.00, -0.80]`

### Reward Function
- Native MetaDrive composite reward (forward driving progress + velocity bonuses − collision/out-of-road penalties).

---

## 4. Stage 0 Random Baseline (Exploratory Findings)

The random agent policy selects uniform random actions from `Discrete(5)`.

In one exploratory 20-episode validation run with `traffic_density=0.0` and single-scenario configuration, observed outcomes were approximately:
- **Episodes**: 20
- **Success rate**: 0.0%
- **Crash rate**: 0.0%
- **Out-of-road rate**: ~65.0% (remaining ~35% terminated on horizon timeout)
- **Mean route completion**: ~0.070
- **Mean episode reward**: ~28.058
- **Mean episode length**: ~624.8 steps

> ⚠️ **Notice & Stochasticity**: The current exploratory evaluator does not establish a finalized deterministic benchmark protocol. Because action sampling is uniform random and unseeded, individual runs will exhibit natural stochastic variation (for example, out-of-road termination rates typically vary between 60% and 75%). These figures describe **one illustrative exploratory run** intended solely to verify environment stepping, collision/boundary detection, and termination signals. They are **not** final scientific benchmark numbers.

---

## 5. Running the Exploratory Scripts

Activate the Conda environment:
```powershell
conda activate metadrive
```

Run headless inspection scripts:
```powershell
python inspect_metadrive.py
python inspect_observation.py
python inspect_observation_parts.py
python inspect_reward_config.py
```

Run the Stage-0 random baseline evaluation (from the `autonomous-driving-rl` project root):
```powershell
python -m src.evaluation.evaluate_random
```

---

## 6. Known Limitations & Next Steps

1. **Discrete Action Granularity**: The 5-action discrete mapping is rigid and lacks combined maneuvers (e.g. slight steering while coasting or hard braking while steering). Future revisions will evaluate discrete expansions (e.g., 9 actions) or continuous action controllers.
2. **LiDAR Reduction**: Aggregating 240 rays into 16 min-pooled sectors loses fine angular resolution of narrow obstacles.
3. **Reward Alignment**: The native reward was designed for dense model-free RL; reward shaping or multi-objective metrics may be required for planning algorithms.
4. **Immediate Next Step**: Conduct a systematic literature survey covering autonomous-driving decision making, rule/heuristic baselines, planning/search, simulation-based planning, learning from data/imitation learning, model-free RL, search+learning, model-based RL, simulator/benchmark design, and evaluation methodology. The purpose of the survey is to decide the final project direction and architecture before Stage 1 implementation. (Methods such as IDM/MOBIL may serve as illustrative survey references, but are not pre-selected as the final architecture).
