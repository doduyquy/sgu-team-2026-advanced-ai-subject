# Platform V1 Research Launcher Core & Plan Resolver Audit

> **Status:** AUDIT & SCIENTIFIC SPECIFICATION (GATE 7.5A)  
> **Freeze Status:** LOCKED-FOR-PLATFORM-V1 (Launcher Core Verified)  
> **Target Scope:** Standardized experiment orchestration, plan resolution, preflight safety, and simulation pipeline across Stages 0 through 7  
> **Date:** October 2026  

---

## 1. Executive Summary & Objective

Gate 7.5A establishes the **Launcher Core, Experiment Plan Resolver, Preflight Engine, and Execution Harness** for Research Platform V1.

Before Gate 7.5, running an autonomous driving experiment required researchers to manually coordinate:
- Geometry selection, procedural generation seeds, and split restrictions;
- Route-aware horizons and environment stochasticity seeds;
- Observation profiles (`STATE_DECISION_V1`) and certified action adapters;
- Gate-7 local logger initialization, durable run states, and W&B mirroring.

Gate 7.5A encapsulates these responsibilities into an **algorithm-neutral, reproducible, and safe orchestration engine**:
$$\text{User Intent (LaunchRequestV1)} \longrightarrow \text{Resolver} \longrightarrow \text{ResolvedExperimentPlanV1} \longrightarrow \text{Preflight} \longrightarrow \text{Executor} \longrightarrow \text{Gate-7 Local Artifacts}$$

### Boundary Notice: Gate 7.5A vs. Gate 7.5B
- **Gate 7.5A (This Gate):** Pure headless Python orchestration engine, immutable data models, declarative registry, manifest resolvers, preflight safety battery, generic simulation loop, and thin CLI.
- **Gate 7.5B (Subsequent Gate):** Desktop graphical user interface (PySide6 / Qt), interactive dashboard, real-time telemetry charting, and desktop window management.

---

## 2. Orchestration Architecture & Mode Semantics

The launcher defines four user-facing orchestration modes (`LauncherMode`), cleanly mapped to lower-level Gate-7 execution classifications (`RunKind`):

| Launcher Mode | Mapped `RunKind` | Canonical Status | Case Scope | Render Policy | Default W&B Mode | Purpose |
|---|---|---|---|---|---|---|
| **`SANDBOX`** | `AUDIT` | Non-Canonical | 1 Episode (TRAIN only) | `OFF` or `NATIVE` | `DISABLED` | Interactive learning, visual inspection, algorithm debugging |
| **`VALIDATION`** | `VALIDATION_EVALUATION` | **Canonical** | 96 Cases (Gate-5 suite) | `OFF` (Headless mandatory) | `OFFLINE` | Checkpoint selection, parameter ablations, model comparison |
| **`TEST`** | `TEST_EVALUATION` | **Canonical** | 60 Cases (Gate-5 suite) | `OFF` (Headless mandatory) | `OFFLINE` | Final frozen scientific benchmark ranking |
| **`AUDIT`** | `AUDIT` | Non-Canonical | 1+ Non-TEST Cases | `OFF` or `NATIVE` | `DISABLED` | Platform regression, CI checks, contract calibration |

### Strict Holdout Safety Policy
- **Sandbox TRAIN-Only Enforcement:** Sandbox mode is strictly restricted to geometries in the `TRAIN` split (180 geometries).
- **Holdout Violations:** If a user requests a geometry belonging to the `VALIDATION` (48 geometries) or `TEST` (12 geometries) splits in Sandbox mode, the resolver raises a fatal `ValueError: FATAL HOLDOUT VIOLATION` immediately. No silent fallback or approximate matching is permitted.
- **Benchmark Full-Suite Lock:** Both `VALIDATION` (96 cases) and `TEST` (60 cases) resolve the exact, immutable sequence of cases declared in Gate-5 manifests (`validation_case_manifest.csv` and `test_case_manifest.csv`). Partial benchmark runs, case reordering, and "test smoke" runs are strictly prohibited in Gate 7.5A.

---

## 3. Data Flow: Intent vs. Resolved Plan

The architecture strictly distinguishes between what the user requested and what the platform will execute:

```text
       User CLI / Future GUI
                 ↓
          LaunchRequestV1   (Declarative intent: mode, agent_id, tier, seeds, render)
                 ↓
      resolve_experiment_plan()
                 ↓
     ResolvedExperimentPlanV1 (Concrete, immutable, fully-specified execution plan)
                 ↓
          run_preflight()     (Pure validation: 13 check categories)
                 ↓ (can_execute == True)
       ExperimentExecutor     (Generic 10 Hz simulation loop)
                 ↓
      MetaDrive Simulator + Gate-6 Policy + Gate-7 LocalExperimentLogger
```

### Deterministic Plan Fingerprinting (`resolved_plan_sha256`)
Every resolved plan computes an authoritative SHA-256 fingerprint covering:
- Mode, run kind, canonical status, and protocol scope;
- Agent identity, version, profile, adapter, and stochasticity;
- Ordered case suite (case IDs, seeds, hashes, horizons, traffic densities);
- Rendering mode and W&B mode;
- Platform contract hashes across Gates 5, 6, 7, and 7.5A.

*Exclusions:* Timestamps, user comments, warnings, terminal colors, and file system paths are excluded from `resolved_plan_sha256`.

---

## 4. Agent Registry & Fixture Boundaries

`AgentRegistryV1` decouples declarative capability metadata (`AgentRegistrationV1`) from runtime callable factories:
- **Declarative Metadata:** Serializable dataclass declaring input profile, action adapter, stochasticity, statefulness, and eligibility flags.
- **Runtime Factory:** Parameterless callable (`Callable[[], AgentPolicy]`) returning a fresh `AgentPolicy` instance.

### Initial Gate-7.5A Agents (Audit Fixtures Only):
Gate 7.5A provides verification fixtures from Gate 6:
1. `fixture_constant_continuous` (`DeterministicConstantFixtureAgent`): Constant steering=0.0, throttle=0.3.
2. `fixture_seeded_random` (`SeededRandomFixtureAgent`): Uniform random continuous actions seeded strictly via `agent_seed`.
3. `fixture_stateful_counter` (`StatefulCounterFixtureAgent`): Internal step counter verifying episodic state reset.
4. `fixture_discrete` (`DiscreteFixtureAgent`): Discrete action index 12 (stay still / idle).

*Scientific Boundary:* All initial fixtures are registered with `benchmark_eligible = False`. Preflight strictly blocks them from executing on `VALIDATION` or `TEST` suites. The true canonical Stage-0 Random baseline will be registered in a subsequent task.

---

## 5. Preflight Safety Battery

Before any simulator environment or logger directory is instantiated, `run_preflight()` evaluates a pure battery of 13 checks:
1. **MetaDrive Exact Pin:** Asserts installed version is `0.4.3` and commit is `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`.
2. **Git Cleanliness:** Requires clean working tree for `VALIDATION` and `TEST` benchmarks (blocks uncommitted benchmark runs).
3. **Agent Mode Eligibility:** Verifies agent `sandbox_eligible` / `audit_eligible` flags.
4. **Benchmark Agent Eligibility:** Asserts `benchmark_eligible == True` for `VALIDATION` and `TEST` (rejects fixtures).
5. **Input Profile Support:** Asserts profile is certified (`STATE_DECISION_V1`).
6. **Action Adapter Support:** Asserts adapter is certified (`continuous_box2_v1`, `discrete25_native_v1`, `discrete9_lowbranch_v1`).
7. **Sandbox Case Count & Holdout:** Asserts exactly 1 case in `TRAIN` split.
8. **Validation Suite Integrity:** Asserts exactly 96 cases in `VALIDATION` split matching manifest.
9. **Test Suite Integrity:** Asserts exactly 60 paired cases in `TEST` split matching manifest.
10. **Protocol Order Monotonicity:** Asserts sequential 1-indexed ordering.
11. **Stochastic Seed Policy:** Requires `agent_seed` $\in \{101, 202, 303\}$ for stochastic benchmark agents; requires `agent_seed is None` for deterministic benchmark agents.
12. **Benchmark Render Policy:** Asserts `render_mode == "OFF"` for benchmark suites.
13. **W&B Credential Presence:** Asserts `WANDB_API_KEY` exists in memory if `WandbMode.ONLINE` is requested.

*Execution Rule:* `can_execute == True` if and only if zero `FAIL` checks exist.

---

## 6. Execution Pipeline & Scientific Contracts

`ExperimentExecutor` implements an algorithm-neutral simulation loop:
- **Exact Geometry Loading:** Reconstructs road blocks using `MapGenerateMethod.PG_MAP_FILE`, recomputing `geometry_sha256` and verifying bit-for-bit equality against the Gate-5 manifest.
- **Information Parity Principle:** The agent receives strictly `AgentInputV1` and `AgentPublicEpisodeContext`. Evaluator metadata (split, tier, case ID, seeds, route completion, success status, W&B state) are strictly isolated from the agent.
- **10 Hz Control Cycle:** Measures agent decision latency strictly around `agent.act()` (excluding `AgentInput` assembly, action adaptation, rendering, physics, and logging).
- **Gate-3 Outcome Classification:** Classifies termination using `classify_episode_outcome()` with strict safety-first precedence.
- **Gate-4 Reward Specification:** Computes step rewards using `RewardSpecV1.compute_step_reward()` (MetaDrive native reward is completely discarded).
- **Gate-7 Local Persistence:** Appends per-episode rows to `episodes.csv`, records latency samples in `timing.csv`, builds authoritative summary in `summary.json`, and records cryptographic content fingerprints in `run_integrity.json`.
- **Gate-7 Provenance Embedding:** Embeds launcher provenance (`launcher_mode`, `launcher_contract_sha256`, `platform_execution_contract_sha256`, `resolved_plan_sha256`, `protocol_scope`) into `config.algorithm_hyperparameters["launcher"]` without mutating frozen Gate-7 schemas.

---

## 7. Event Telemetry Interface (Handoff to Gate 7.5B)

`LauncherEventV1` provides a clean, JSON-safe observer stream for future UI progress reporting:
- **Event Types:** `RUN_STARTED`, `EPISODE_STARTED`, `EPISODE_PROGRESS`, `EPISODE_FINISHED`, `RUN_FINISHED`, `RUN_INTERRUPTED`, `RUN_FAILED`.
- **Decoupled Performance:** Progress events emit at throttled intervals (every 10 steps), ensuring 10 Hz physical control is never impacted by listener overhead.
- **Strict Isolation:** Telemetry events are delivered exclusively to registered callbacks and never exposed to `AgentPolicy`.

---

## 8. CLI Usage Examples

The thin CLI (`python -m src.launcher.cli`) exposes standard launcher operations:

```bash
# 1. List registered agents and capabilities
python -m src.launcher.cli agents

# 2. Query available training geometries
python -m src.launcher.cli cases --split TRAIN --tier Medium --limit 5

# 3. Preview an exploratory Sandbox plan with preflight checks (zero side-effects)
python -m src.launcher.cli plan --mode SANDBOX --agent fixture_seeded_random --tier Easy --agent-seed 101

# 4. Preview structured JSON for GUI / automation integration
python -m src.launcher.cli plan --mode SANDBOX --agent fixture_seeded_random --json

# 5. Execute an exploratory Sandbox episode
python -m src.launcher.cli run --mode SANDBOX --agent fixture_constant_continuous --tier Easy --render OFF
```

---

## 9. Cryptographic Contract Hashes

Gate 7.5A introduces additive hashes without altering previously locked platform contracts:

| Contract | Hash (SHA-256) | Status |
|---|---|---|
| **`gate5_benchmark_contract_sha256`** | `9ddd889b84d8705fae618879e5035556c80d0276a3dc2a58a7963937ebb59f77` | **LOCKED (Untouched)** |
| **`gate6_agent_contract_sha256`** | `53aa37079ff44afa75d9a3f921b0c1f98c4600d882fced51fc8d9197795058eb` | **LOCKED (Untouched)** |
| **`platform_runtime_contract_sha256`** | `c7698768539a769c7b2bc6b90771ff234276a974b6e0353bbb03119faf2f79ad` | **LOCKED (Untouched)** |
| **`logging_contract_sha256`** | `0d17915556d83f49d9519576e95920c0853700f33a91593ef85b3aeef6cbb9f2` | **LOCKED (Untouched)** |
| **`platform_observability_contract_sha256`** | `f00c27fca8abf8145596576cd0dd2540181ff8ae7eccf7806c864921933d3581` | **LOCKED (Untouched)** |
| **`launcher_contract_sha256`** | `22718d9180b945aea801fe078e9204744dd063fc644301d310ac0db9d9ebd8cc` | **LOCKED (Gate 7.5A)** |
| **`platform_execution_contract_sha256`** | `fcc9d469466dd55b8a1a2d6f28235695b1130eacff88bcabc2ccc82db428c7a6` | **LOCKED (Gate 7.5A)** |

$$\text{platform\_execution\_contract\_sha256} = \text{SHA256}(\text{canonical\_json}(\{\text{"platform\_observability\_contract\_sha256"}, \text{"launcher\_contract\_sha256"}, \dots\}))$$

---

## 10. Audit Evidence & Artifacts

All verification artifacts are generated in `results/audits/launcher_core/`:
- `contract_hashes.json`: Locked platform hashes across Gates 5 through 7.5A.
- `mode_resolution_matrix.csv`: Verified mapping of LauncherMode to RunKind and default policies.
- `agent_registry_snapshot.json`: Machine-readable snapshot of all registered agent descriptors.
- `case_plan_parity.json`: Verified 100% manifest parity for Validation (96) and Test (60) suites.
- `benchmark_lock_checks.json`: Verified Sandbox holdout rejection and benchmark fixture rejection.
- `preflight_matrix.csv`: Empirical evaluation of 13 preflight decision scenarios.
- `plan_hash_determinism.json`: Proof of deterministic plan hashing and mutation sensitivity.
- `cli_core_parity.json`: Verified semantic equivalence between CLI JSON output and direct resolver output.
- `sandbox_execution_smoke.json`: Successful real MetaDrive simulation execution of Sandbox episode:
  - Run state: `COMPLETE`
  - All 7 Gate-7 local files present and verified.
  - Launcher provenance embedded in `run_manifest.json`.

---

## 11. Known Limitations & Handoff to Gate 7.5B

The following capabilities are deliberately excluded from Gate 7.5A and will be implemented in subsequent gates:
1. **Desktop Graphical User Interface (GUI):** PySide6 / Qt visual dashboard belongs to Gate 7.5B.
2. **Canonical Stage-0 Random Agent:** Final benchmark Random policy implementation belongs to Stage 0.
3. **Training & Multi-Seed Batch Execution:** Experiment queueing and learning loops belong to training infrastructure.
4. **Interactive Manual Driving / Teleoperation:** Visual keyboard teleoperation belongs to exploratory GUI tools.
5. **Map 2D/3D Visual Preview:** Visual road geometry rendering belongs to Gate 7.5B.

---

## 12. Regression & Verification Summary
- **Launcher Core Unit Tests:** 28 tests pass (`tests/test_launcher_core.py`, 2.6s).
- **Full Platform Regression Suite:** **158 total unit tests pass across repo** (`test_episode_lifecycle`: 12, `test_reward_metrics`: 14, `test_evaluation_protocol`: 20, `test_agent_contract`: 37, `test_logging_contract`: 47, `test_launcher_core`: 28).
- **Legacy Compatibility:** `CourseEnvV1` observation shape `(35,)` verified; `evaluate_random.py` completes 20 episodes cleanly.
- **Contract Integrity:** Gates 5, 6, and 7 hashes remain 100% untouched.
- **Privacy & Secret Hygiene:** Automated scans across all files confirmed zero credentials and zero machine-private paths committed.

STOP. PR remains open and unmerged.
