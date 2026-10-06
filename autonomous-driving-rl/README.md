# Autonomous Driving Research Platform (MetaDrive)

This directory is the autonomous-driving research subproject for the SGU Advanced AI course. MetaDrive is used as the simulation engine; the team has built a reproducible **Research Platform V1** on top of it so different agent families can be developed and compared under the same environment, information rights, benchmark cases, episode rules, metrics, and logging pipeline.

## Current status

**Platform foundation: complete.** The environment, benchmark protocol, agent interface, logging/provenance, launcher/preflight layer, and Research Workbench are implemented and hardened.

**Next research milestone: Stage 0 — Random / Naive Baseline Agent.**

**Known MapSuite V1 metadata erratum:** `declared_base_lane_num = 2`, but `effective_base_lane_num = 3` for the frozen geometry. See the [dedicated erratum](docs/environment/MAPSUITE_V1_LANE_METADATA_ERRATUM.md); do not apply a two-lane override based on the historical description.

The repository currently contains several fixture agents used to test the platform. They are intentionally **not benchmark agents** and are blocked from canonical VALIDATION/TEST runs. The old `CourseEnvV1` + `src/evaluation/evaluate_random.py` path is retained only as a legacy prototype/compatibility path; it is not the current scientific benchmark.

## What Platform V1 provides

- **MetaDrive pin:** version `0.4.3`, commit `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`.
- **Scenario suite:** 12 scenario families × 20 geometry seeds = **240 geometries**, grouped into Easy / Medium / Hard / Extreme.
- **Packaged benchmark dataset:** [`datasets/mapsuite_v1/`](datasets/mapsuite_v1/) packages the frozen 240-geometry procedural benchmark, split definitions, and canonical evaluation suites.
- **Fixed split:** 180 TRAIN, 48 VALIDATION geometries, 12 TEST geometries. Canonical suites contain **96 validation cases** and **60 test cases**.
- **Common agent input:** 259D core MetaDrive observation (9 ego + 10 navigation + 240 LiDAR), plus structured traffic context and route/task context.
- **Common actuator contract:** `[steering, throttle_brake]` in `[-1, 1]^2`, nominally at **10 Hz**. Certified continuous and discrete adapters are available.
- **Episode semantics:** clean success, safety failures, timeout, safety-first outcome precedence, and route-aware horizons.
- **Reward:** route progress + time cost + terminal outcome; reward is a training signal, not a benchmark ranking score.
- **Primary evaluation:** clean success rate, safety failure rate, mean/median final route completion, and mean time to clean success.
- **Reproducibility:** explicit geometry/environment/agent seed separation, fixed manifests, holdout protection, provenance, and run integrity checks.
- **Local-first logging:** local artifacts are authoritative; W&B is optional downstream tracking/visualization.
- **Launcher + preflight:** resolves an experiment plan and blocks invalid or scientifically inconsistent runs before simulation.
- **Research Workbench:** GUI for experiment setup, preflight, live monitoring, case/agent browsing, and verified result inspection.

## Start here

If this is your first time working on the project, read these in order:

1. [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) — install and verify the project on a new machine.
2. [`docs/PLATFORM_V1_OVERVIEW.md`](docs/PLATFORM_V1_OVERVIEW.md) — understand the environment and scientific contracts already built.
3. [`docs/EXPERIMENT_WORKFLOW.md`](docs/EXPERIMENT_WORKFLOW.md) — learn SANDBOX / VALIDATION / TEST and how agents are compared.
4. [`docs/AGENT_DEVELOPMENT_GUIDE.md`](docs/AGENT_DEVELOPMENT_GUIDE.md) — implement a new agent without bypassing the platform.
5. [`docs/STAGE_ROADMAP.md`](docs/STAGE_ROADMAP.md) — see how Stage 0 through Stage 7 fit on the same platform.
6. [`docs/WORKBENCH_GUIDE.md`](docs/WORKBENCH_GUIDE.md) — use the desktop Research Workbench.
7. [`datasets/mapsuite_v1/`](datasets/mapsuite_v1/) — explore the packaged procedural benchmark dataset.

For the verified history, hash registry, evidence matrix and boundaries, continue with the [Platform V1 knowledge guide](docs/README.md#platform-v1-knowledge-and-derived-artifacts). The [preview distribution guide](docs/MAP_ARTIFACT_DISTRIBUTION.md) explains the derived 240-map PNG export.

Detailed platform audit reports remain under [`docs/environment/`](docs/environment/) for anyone who needs implementation-level evidence.

## Quick start after installation

From `autonomous-driving-rl/`:

```powershell
# Inspect registered agents
python -m src.launcher.cli agents

# Inspect training cases
python -m src.launcher.cli cases --split TRAIN --tier Easy --limit 5

# Resolve a safe single-case sandbox plan without running the simulator
python -m src.launcher.cli plan --mode SANDBOX --agent fixture_seeded_random --tier Easy --agent-seed 101

# Run one sandbox episode (fixture only, not a benchmark result)
python -m src.launcher.cli run --mode SANDBOX --agent fixture_seeded_random --tier Easy --agent-seed 101 --render NATIVE

# Launch Research Workbench
python -m src.workbench.app
```

Trying to use the fixture agents for `VALIDATION` or `TEST` is expected to be blocked by preflight. That is a safety feature, not an error.

## Repository map

```text
autonomous-driving-rl/
├── configs/platform/       # Versioned scientific contracts and benchmark configuration
├── datasets/               # Packaged research datasets (e.g. MapSuite V1 procedural benchmark)
├── docs/                   # Human-facing guides + detailed audit/reference documents
├── experiments/            # Future experiment/training definitions and orchestration notes
├── results/audits/         # Committed machine-derived platform audit evidence
├── runs/                   # Local experiment outputs (git-ignored except .gitkeep)
├── scripts/                # Platform audit and diagnostic scripts
├── src/
│   ├── agents/             # Research agent implementations (Stage 0 onward)
│   ├── environments/       # Legacy CourseEnvV1 compatibility wrapper
│   ├── evaluation/         # Legacy exploratory evaluator; not the canonical benchmark path
│   ├── launcher/           # Plan resolver, registry, preflight, executor, CLI
│   ├── platform/           # Core scientific contracts: input/action/episode/reward/metrics/logging
│   └── workbench/          # PySide6 Research Workbench
└── tests/                  # Platform regression tests
```

## Project rule that matters most

**Algorithms change; the benchmark must not be changed merely because an algorithm performs poorly.**

TRAIN is for development, VALIDATION is for tuning/model selection, and TEST is the final holdout. If a future research stage truly requires a new information contract, action model, or benchmark capability, that change must be proposed and versioned explicitly instead of being introduced silently inside an agent.
