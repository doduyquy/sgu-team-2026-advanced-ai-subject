# Autonomous Driving Research Platform (MetaDrive)

This directory is the autonomous-driving research subproject for the SGU Advanced AI course. MetaDrive is used as the simulation engine; the team has built a reproducible **Research Platform V1** on top of it so different agent families can be developed and compared under the same environment, information rights, benchmark cases, episode rules, metrics, and logging pipeline.

## Native Race Mode · mở nhanh

Bấm đúp [`CHAY_DUA_XE.cmd`](../CHAY_DUA_XE.cmd) ở thư mục gốc để chọn A*, BFS,
Dijkstra, Hybrid A* hoặc MPPI thử nghiệm, rồi xem xe chạy ngay trong cửa sổ MetaDrive.
Có chế độ Chạy đơn hoặc Đua đồng thời: chọn 2–5 thuật toán, mỗi thuật toán điều
khiển một xe trong cùng cảnh. Có ba đường tham chiếu, đường Việt Nam và bảng
chọn 12 bản đồ / 60 kịch bản TEST, cùng giao thông, tạm dừng, chuyển xe theo dõi,
đổi camera, bảng trạng thái trực tiếp và kết quả riêng từng xe.
Xem [hướng dẫn tiếng Việt từng bước](docs/RACE_MODE_GUIDE.md). Các lượt đua tương
tác không phải benchmark chính thức; log giữ đúng split của bản đồ được chọn.
MPPI dùng `src/agents/MPPI.py` và cùng cấu hình JSON với CLI / Workbench.

## Current status

**Platform foundation: complete.** The environment, benchmark protocol, agent interface, logging/provenance, launcher/preflight layer, and Research Workbench are implemented and hardened.

**Stage 0:** `stage0_random_naive` is implemented for SANDBOX/AUDIT. Benchmark certification and Stage 1 remain the next research milestones. See the [agent policy and local readiness workflow](src/agents/README.md).

**Stage 2 / Stage 6 prototype:** `stage2_astar` and `stage6_astar_learned` share a public
local planning graph and controller. Stage 6 has a real supervised heuristic trainer,
fixed presets, checkpoint provenance and paired TRAIN evaluation in the Workbench.
Both remain benchmark-ineligible candidates. See [train and compare A*](docs/ASTAR_LEARNED_HEURISTIC.md).

**Stage 2 search methods:** BFS, Dijkstra, A* and forward Hybrid A* are implemented
with a shared public input/action interface and path follower. The Workbench can
select and run all four on the same fixed TRAIN case. See
[search planners and comparison commands](docs/SEARCH_PLANNERS.md).

The [624-episode evaluation report](results/search_planners_20261101/README.md)
records the frozen run results and their evidence hashes, including the current
zero clean-success rate on TEST.

**Known MapSuite V1 metadata erratum:** `declared_base_lane_num = 2`, but `effective_base_lane_num = 3` for the frozen geometry. See the [dedicated erratum](docs/environment/MAPSUITE_V1_LANE_METADATA_ERRATUM.md); do not apply a two-lane override based on the historical description.

The repository currently contains several fixture agents used to test the platform. They are intentionally **not benchmark agents** and are blocked from canonical VALIDATION/TEST runs. The old `CourseEnvV1` + `src/evaluation/evaluate_random.py` path is retained only as a legacy prototype/compatibility path; it is not the current scientific benchmark.

## Stage 5 SAC demo

Download or clone `main`, extract it if needed, and open PowerShell in its
`autonomous-driving-rl/` folder. The inference bundle is
[`checkpoints/stage5/sac-v2-5000/`](checkpoints/stage5/sac-v2-5000/).
It contains the original `model.zip`, metadata, and checksums for `stage-05000`.
The launcher verifies the checksum and observation profile at load time.

With Python 3.10 and Git installed, set up the dependencies once. This step
needs internet access for the pinned MetaDrive source and simulator assets.

```powershell
conda create -n sac-stage5 python=3.10 pip -y
conda activate sac-stage5
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-stage5-demo.txt
python -m metadrive.pull_asset
```

Run the Medium case using the native simulator window. Use `--render OFF`
for a headless run. A CPU is sufficient.

```powershell
python -m src.launcher.cli run --mode SANDBOX --agent stage5_sac_terminal_preview --checkpoint checkpoints/stage5/sac-v2-5000 --task-context-producer reference_route_station_proposal_v1 --tier Medium --sequence SCTCS --geom-seed 11 --env-seed 33701 --render NATIVE --wandb DISABLED
```

The demo uses the frozen 409D F09 and Terminal Preview V1 input path and raw
Box2 actions. Results are saved to the local `runs/` folder. The
[experiment evidence](results/stage5/README.md) includes paired VALIDATION
and TEST results with F09 recovery receipts. TEST Traffic ON achieved
19/60 clean successes. TEST Traffic OFF yielded 54/60 valid outcomes,
with 35 clean successes observed.

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

# Resolve a single-case TRAIN plan without running the simulator
python -X utf8 -m src.launcher.cli plan --mode SANDBOX --agent stage1_rule_based --tier Easy --agent-seed 42

# Run the rule-based agent, including traffic/junction/lane-change modules
python -X utf8 -m src.launcher.cli run --mode SANDBOX --agent stage1_rule_based --tier Medium --sequence SCXCS --geom-seed 14 --env-seed 0 --agent-seed 42 --render NATIVE --wandb DISABLED --finish-visual hologram

# Run the random/naive agent
python -X utf8 -m src.launcher.cli run --mode SANDBOX --agent stage0_random_naive --tier Easy --agent-seed 42 --render NATIVE --wandb DISABLED

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
├── experiments/            # Stage-specific training entry points and experiment notes
├── results/audits/         # Committed machine-derived platform audit evidence
├── runs/                   # Local experiment outputs and generated caches (git-ignored)
├── scripts/                # Platform audit and diagnostic scripts
├── src/
│   ├── agents/             # Research agent implementations (Stage 0 onward)
│   ├── environments/       # Legacy CourseEnvV1 compatibility wrapper
│   ├── evaluation/         # Legacy exploratory evaluator; not the canonical benchmark path
│   ├── launcher/           # Plan resolver, registry, preflight, executor, CLI
│   ├── platform/           # Core scientific contracts: input/action/episode/reward/metrics/logging
│   ├── logging/            # Agent source/configuration and per-step trace recording
│   ├── visualization/      # Finish hologram/minimap code; generated textures live in runs/.cache/
│   └── workbench/          # PySide6 Research Workbench
└── tests/                  # Platform regression tests
```

Source-maintenance record: [cleanup 0.6.1](docs/SOURCE_CLEANUP_V0_6_1.md).

Run all regressions with `python -B -X utf8 scripts/run_tests.py`; this isolates
backend and Qt tests in separate processes and records their commands/logs.

## Project rule that matters most

**Algorithms change; the benchmark must not be changed merely because an algorithm performs poorly.**

TRAIN is for development, VALIDATION is for tuning/model selection, and TEST is the final holdout. If a future research stage truly requires a new information contract, action model, or benchmark capability, that change must be proposed and versioned explicitly instead of being introduced silently inside an agent.
