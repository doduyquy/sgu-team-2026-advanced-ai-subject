# Research Platform V1 Overview

Platform V1 is the common scientific layer used by all future autonomous-driving agents. MetaDrive supplies the simulation engine; Platform V1 fixes the environment, interfaces, evaluation rules, logging, and experiment orchestration around it.

## 1. Architecture

```text
Research method / Agent
        │
        │ AgentPolicy: reset() / act() / close()
        ▼
AgentInputV1 + public episode context
        │
        ▼
Certified Action Adapter
        │
        ▼
Canonical [steering, throttle_brake]
        │
        ▼
MetaDrive 0.4.3 (pinned commit)
        │
        ├── episode classification
        ├── reward signal
        ├── evaluation metrics
        └── local experiment logging
                │
                ▼
         verified run artifacts
```

Launcher/preflight and the Research Workbench sit around this execution path; they do not give an algorithm additional information.

## 2. Scenario suite

Platform V1 defines 12 procedural scenario families, 20 geometry seeds per family, for **240 geometries**:

- Easy: `SCS`, `SCSS`, `SCCS`
- Medium: `SCXCS`, `SCTCS`, `SCXCCS`
- Hard: `SCXOCS`, `SCTXrCS`, `XTOCS`
- Extreme: `CrXROSTR`, `SCXOCrTYCS`, `SCTXORyCCS`

Difficulty is an environment property based on topology, decision structure, route length, and traffic density; it is not defined by how well a particular agent performs.

The geometry universe is deterministically split into:

- 180 TRAIN geometries;
- 48 VALIDATION geometries;
- 12 TEST geometries.

The canonical evaluation manifests expand these to **96 validation cases** and **60 test cases** through fixed environment seeds.

The key research rule is that the benchmark is fixed before comparing agents. A poor algorithm result is not a reason to replace TEST maps.

## 3. Information available to an agent

The main profile is `STATE_DECISION_V1`.

The core observation preserves the native 259D MetaDrive state:

- 9 ego-state values;
- 10 navigation values;
- 240 LiDAR rays.

Platform V1 adds structured local traffic context and task/route context. These give algorithms useful state-based information while keeping evaluator-private information outside the agent boundary.

An agent does **not** receive benchmark split identity, case ID, global route-completion percentage, success/failure verdict, aggregate metrics, or other evaluator-private fields as privileged inputs.

The principle is simple: **same information rights across stages**. A method may choose to ignore fields, but it must not receive hidden extra information because it is a more advanced algorithm.

## 4. Action and control

The physical actuator contract is continuous:

```text
[steering, throttle_brake] ∈ [-1, 1]²
```

Nominal control is **10 Hz** (`0.1 s` per decision). Platform-certified action adapters currently include:

- continuous 2D control;
- native 25-action discrete grid;
- low-branching 9-action discrete grid for search/planning-style methods.

The adapter validates actions and fails loudly on invalid output rather than silently changing an agent's decision.

## 5. Episode lifecycle and safety

Episodes distinguish task success, safety failure, and timeout. Clean success means the destination is reached without a higher-priority safety violation.

Safety events such as collisions and leaving the road take precedence over destination arrival. `terminated` and `truncated` retain their distinct Gymnasium semantics.

The episode horizon is route-aware rather than forcing every map to share one arbitrary step budget.

## 6. Reward and evaluation are different

The current reward design combines:

- signed route-progress shaping;
- a bounded time cost;
- terminal success/failure signals.

Reward exists to support learning. **Episode return is not the benchmark ranking score.** The reward specification remains a research specification until the main RL phase is frozen using TRAIN/VALIDATION only.

The five primary benchmark metrics are:

1. Clean Success Rate — higher is better.
2. Safety Failure Rate — lower is better.
3. Mean Final Route Completion — higher is better.
4. Median Final Route Completion — higher is better.
5. Mean Time to Clean Success — lower is better among successful episodes.

Results are also viewed per difficulty tier. The platform intentionally avoids collapsing all objectives into one mega-score because success/safety/progress trade-offs should remain visible.

## 7. Reproducible evaluation

Platform V1 separates different sources of randomness rather than reusing one ambiguous seed:

- geometry generation;
- simulator/environment stochasticity;
- agent stochasticity;
- training replicate randomness;
- protocol ordering.

TRAIN is available for development. VALIDATION is used for tuning/model/checkpoint decisions. TEST is a holdout for the final frozen evaluation.

VALIDATION and TEST execute their full fixed suites; the canonical path does not support cherry-picking a convenient subset of test cases.

## 8. Logging, provenance, and integrity

Experiment persistence is local-first. A run stores the information needed to reconstruct what was executed, per-episode results, aggregate metrics, timing, lifecycle state, and integrity fingerprints.

Local artifacts are the scientific source of truth. W&B may mirror them for team visualization/comparison, but a network failure must not invalidate an otherwise healthy local run.

The Results Browser trusts only complete runs whose required artifacts and integrity checks are consistent. Live Workbench telemetry is useful for observation, but it is provisional until the persisted run is verified.

## 9. Launcher and preflight

The launcher translates user intent into a concrete, immutable experiment plan and runs preflight before execution. Preflight checks that the requested experiment respects the platform contracts, case manifests, environment pin, agent eligibility, seed rules, and benchmark policies.

If a benchmark request violates those rules, execution is blocked before the simulator is created.

## 10. Current boundary

Platform V1 is ready for algorithm work. What is **not** yet implemented as the next research result is the canonical Stage 0 Random / Naive agent.

Current `fixture_*` agents exist to test the platform. They are not scientific baselines.

`CourseEnvV1` (35D observation + Discrete(5)) and `src/evaluation/evaluate_random.py` belong to the early prototype path. Keep them for compatibility/reference, but new research stages should integrate through the Platform V1 agent/launcher contracts.

## 11. Authoritative technical references

For implementation details, see the reports under [`environment/`](environment/), especially:

- `PLATFORM_V1_MAPSUITE_AUDIT.md`
- `PLATFORM_V1_EPISODE_AUDIT.md`
- `PLATFORM_V1_REWARD_METRICS_AUDIT.md`
- `PLATFORM_V1_EVALUATION_PROTOCOL_AUDIT.md`
- `PLATFORM_V1_AGENT_CONTRACT_AUDIT.md`
- `PLATFORM_V1_LOGGING_WANDB_AUDIT.md`
- `PLATFORM_V1_LAUNCHER_CORE_AUDIT.md`
- `PLATFORM_V1_WORKBENCH_AUDIT.md`
- `PLATFORM_V1_WORKBENCH_RESULTS_AUDIT.md`
- `PLATFORM_V1_WORKBENCH_HARDENING_AUDIT.md`
