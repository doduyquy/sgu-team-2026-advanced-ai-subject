# Platform V1 Experiment Logging, Local Provenance, and Weights & Biases Integration Contract Audit

> **Status:** AUDIT & SCIENTIFIC SPECIFICATION (GATE 7)  
> **Freeze Status:** LOCKED-FOR-PLATFORM-V1 (Logging & Observability Contract Verified)  
> **Target Scope:** Standardized experiment tracking and W&B mirroring across Stages 0 through 7  
> **Date:** September 2026  

---

## 1. Executive Summary & Objective

Gate 7 establishes the **observability, provenance, local persistence, and remote mirroring contract** for Research Platform V1 across all research stages:
- **Stage 0:** Random Baseline
- **Stage 1:** Rule / Heuristic
- **Stage 2:** Planning / Search
- **Stage 3:** Simulation-Based Planning (MCTS)
- **Stage 4:** Learning from Data / Imitation Learning
- **Stage 5:** Model-Free Reinforcement Learning (PPO / SAC)
- **Stage 6:** Search + Learned Policy / Value
- **Stage 7:** Model-Based RL (World Models)

### Core Scientific Mandates:
1. **Local-First Authoritative Truth:** Local raw run records in `runs/<run_id>/` are the sole authoritative scientific source of truth. Weights & Biases acts strictly as a downstream mirror for remote indexing, comparison, and team visualization. An experiment remains 100% scientifically verifiable without network connectivity or W&B access.
2. **Observational Invariance:** Logging and tracking operations are strictly downstream observers. Logger construction, W&B initialization, or network calls NEVER alter agent decisions, environment dynamics, or seed schedules.
3. **Strict Ephemeral Secret Policy:** The `WANDB_API_KEY` credential is treated as an ephemeral secret read exclusively from the environment. Credentials, authorization tokens, and API keys are strictly forbidden from being written to code, configuration files, git commits, terminal outputs, or audit reports.
4. **Clean Failure Segregation:** Explicitly distinguishes task-level driving failures (classified by Gate-3/4 precedence) from technical failures (`AGENT_EXCEPTION`, `INVALID_AGENT_ACTION`, `LOCAL_LOG_WRITE_ERROR`, `WANDB_SYNC_ERROR`). Local write failures are fatal to the run, whereas remote W&B synchronization errors are non-fatal to the underlying scientific experiment.
5. **Exact Metric & Table Parity:** Gate-4 primary metrics (`clean_success_rate`, `safety_failure_rate`, `mean/median_final_route_completion`, `mean_time_to_clean_success_s`) and Gate-5 per-tier scorecards are mirrored bit-for-bit to W&B without recalculation or distortion.
6. **Timing Boundary Preservation:** Agent decision latency timer wraps `agent.act()` strictly, completely excluding downstream logger I/O and W&B network overhead.
7. **Additive Observability Hashing:** Gates 1 through 6 locked contract hashes remain untouched; Gate 7 computes an additive `platform_observability_contract_sha256`.

---

## 2. Source Contracts from Gates 1–6

Gate 7 ties together all previous platform contracts into a unified, reproducible logging pipeline:
- **Gate 1:** Root 259D observation and continuous actuator (`Box(2)`).
- **Gate 2 (`MapSuiteV1`):** 240 candidate geometries across 12 sequence families.
- **Gate 3 (`EpisodeSpecV1`):** Strict termination semantics, safety-first outcome precedence, and route-aware episode horizons.
- **Gate 4 (`RewardSpecV1` & `EvaluationMetricsV1`):** Decoupled reward training signal from evaluation scorecard; primary benchmark metrics.
- **Gate 5 (`EvaluationProtocolV1`):** Scenario splits (180 Train / 48 Validation / 12 Test), 5-channel seed taxonomy, and locked paired test manifest (`benchmark_contract_sha256 = 9ddd889b...`).
- **Gate 6 (`AgentContractV1`):** Information Parity Principle, `AgentInputV1`, certified action adapters, and agent runtime lifecycle (`agent_contract_sha256 = 53aa3707...`, `platform_runtime_contract_sha256 = c7698768...`).

---

## 3. Local-First Architecture: Authoritative Scientific Records

In Research Platform V1, remote tracking services (such as Weights & Biases) must never serve as the sole custodian of experimental evidence. Network outages, account deletions, rate limits, or API deprecations must not invalidate research findings.

### Data Flow Topology:
```text
AgentPolicy.act()
       ↓ (decision)
MetaDrive Simulator
       ↓ (step telemetry)
EpisodeRecord (Gate-3 / Gate-4 normalized)
       ↓
LocalExperimentLogger (AUTHORITATIVE SOURCE OF TRUTH)
  ├── run_manifest.json (Immutable provenance)
  ├── episodes.csv      (Streaming per-episode record)
  ├── summary.json      (Gate-4/5 aggregate scorecards)
  ├── timing.csv        (Per-episode agent decision latency)
  ├── technical_failures.jsonl (Optional technical failure traces)
  └── run_integrity.json (Cryptographic content hashes of local files)
       ↓ (downstream mirror)
WandbBackend (OPTIONAL VISUALIZATION & INDEXING)
  ├── W&B Config (Sanitized experiment configuration)
  ├── W&B Step Metrics (Per-episode evaluation event stream)
  ├── W&B Summary (Mirrored local summary metrics)
  └── W&B Table (evaluation_episodes projection of episodes.csv)
```

### Local Run Directory Layout (`runs/<run_id>/`):
- `run_id`: Unique identifier formatted as `run_<uuid4_12hex>` (e.g. `run_a1b2c3d4e5f6`).
- **Duplicate Protection:** `LocalExperimentLogger` asserts that `runs/<run_id>` does not exist prior to creation. Attempting to overwrite or append to an existing run raises `FileExistsError` immediately.
- **Atomic Writes:** All JSON records (`run_manifest.json`, `summary.json`, `run_integrity.json`, `wandb_sync.json`) are written via temporary `.tmp` files, flushed, fsynced to disk, and atomically renamed.

---

## 4. ExperimentRunV1 Granularity & Lifecycle

### Experiment Granularity:
$$\text{One ExperimentRunV1} = \text{One Fixed Policy / Replicate} + \text{One Config} + \text{One Suite of Episodes}$$
- **Evaluation Runs:** One complete evaluation across the 60 test cases (or 96 validation cases) forms exactly **one** experiment run and **one** W&B run. Individual episodes are logged as steps/rows inside that run, never as individual W&B runs.
- **Run Kinds (`RunKind`):**
  - `TRAINING`: Learning rollout iterations and model updates.
  - `VALIDATION_EVALUATION`: Checkpoint selection and ablation runs on 96 validation cases.
  - `TEST_EVALUATION`: Frozen final benchmark evaluation on 60 paired test cases.
  - `AUDIT`: Verification, smoke tests, and platform calibration runs.
- **Run Status (`RunStatus`):**
  `INITIALIZING` $\to$ `RUNNING` $\to$ `COMPLETE` (or `FAILED` / `INTERRUPTED`).
- **W&B Sync Status (`WandbSyncStatus`):**
  `DISABLED`, `OFFLINE`, `ONLINE`, `SYNCED`, `FAILED`.

---

## 5. Provenance, Privacy, and Secret Policies

### Provenance Tracking (`run_manifest.json`):
Every run captures an immutable manifest containing:
- `git_provenance`: commit SHA, branch, and `git_worktree_dirty` boolean. Benchmark evaluations (`TEST_EVALUATION`) executed with uncommitted modifications are prominently flagged.
- `environment_provenance`: Python version, MetaDrive version, MetaDrive commit, W&B SDK version (`0.30.0`), NumPy version, OS platform.
- `platform_contracts`: locked hashes of Gates 5, 6, and 7.
- `experiment_config_sha256`: canonical SHA-256 fingerprint of all semantic experiment parameters.

### Privacy & Ephemeral Secret Guarantees:
- **No API Keys Persisted:** `WANDB_API_KEY` is loaded in-memory from `os.environ`. Automated scanning verified **0 credentials** committed to repository files (`audit_secret_scan`).
- **Machine Privacy:** User home directories, machine hostnames, internal IP addresses, and personal paths are excluded from manifests. File references use normalized project-relative paths.

---

## 6. Metric Parity & Evaluator Telemetry Segregation

### Primary Benchmark Metrics (Reused from Gate 4):
The primary scorecard reported in `summary.json` and mirrored to `wandb.summary` strictly uses Gate-4 metrics:
1. `clean_success_rate`
2. `safety_failure_rate`
3. `mean_final_route_completion`
4. `median_final_route_completion`
5. `mean_time_to_clean_success_s` (serialized as `null` / `None` if zero successes; never converted to `0.0`)

### Per-Tier Namespaces & Macro Aggregation:
- `metrics/overall/<metric>`
- `metrics/tier/Easy/<metric>`, `metrics/tier/Medium/<metric>`, `metrics/tier/Hard/<metric>`, `metrics/tier/Extreme/<metric>`
- `metrics/macro/<metric>` (equal tier-weighted: $0.25$ per tier).
- **Rejection of Mega-Scores:** No weighted mega-score or geometric-mean scalar is generated.

### Diagnostic Classification of Episode Return:
`episode_return` is logged under `diagnostic/episode_return` (or `training_diagnostics/episode_return`). It is explicitly excluded from primary benchmark ranking scorecards.

---

## 7. Timing Boundary Preservation

Agent decision latency is measured using high-resolution monotonic timestamps (`time.perf_counter_ns()`):
$$\Delta t_{act} = t_{\text{post\_act}} - t_{\text{pre\_act}}$$
- **Exclusions:** Excludes `AgentInput` assembly, `ActionAdapter` conversion, MetaDrive physics simulation (`env.step()`), rendering, file I/O, and W&B network calls.
- **Empirical Exclusion Audit (`logging_latency_boundary.json`):** Verified that injecting $30\text{ ms}$ of simulated downstream logger overhead resulted in a measured agent act latency of $5.00\text{ ms}$, proving the timing boundary remains completely unpolluted by tracking infrastructure.

---

## 8. Observational Invariance & RNG Isolation

A critical requirement for scientific logging is that enabling tracking must not alter simulation dynamics.

*Simulator Trace Invariance Audit (`logging_rng_isolation.json`):* Executed two identical 20-step simulations on fixed geometry (`SCS` s13, env seed 5101) with `agent_policy=IDMPolicy`:
1. Run A: `WandbMode.DISABLED`
2. Run B: `WandbMode.OFFLINE` (active tracking backend logging metrics and tables)
- **Result:** Initial positions, intermediate trajectory coordinates, and final positions matched **100% bit-for-bit** across all 20 steps (`positions_match == True`). Logging is strictly observational.

---

## 9. Failure Policies & Degraded Modes

| Failure Mode | Impact on Local Scientific Run | Impact on W&B | Run Status Verdict |
|---|---|---|---|
| **Local Disk Write Failure** | FATAL. Local records cannot be written. | Halted. | `FAILED` (cannot be marked `COMPLETE`) |
| **W&B Network Outage** | NON-FATAL. Local run completes normally. | `WandbSyncStatus.FAILED` | `COMPLETE` (local records authoritative) |
| **W&B Init Timeout** | NON-FATAL. Local run completes normally. | `WandbSyncStatus.FAILED` | `COMPLETE` |
| **W&B Disabled** | None. Local run completes standalone. | `WandbSyncStatus.DISABLED` | `COMPLETE` |

---

## 10. Weights & Biases Online Smoke Test

Executed an online smoke test using the transient `WANDB_API_KEY`:
- **Run ID:** `audit_online_aaff2e38`
- **Run URL:** https://wandb.ai/phucga15062005/sgu-autonomous-driving-rl/runs/audit_online_aaff2e38
- **Project / Entity:** `sgu-autonomous-driving-rl` / `phucga15062005` (user: `irthn1311`)
- **Status:** `SYNCED`
- **Artifacts Mirrored:** Configuration parameters, step metrics stream, Gate-4 summary scorecards, and `evaluation_episodes` Table.
- **Credential Hygiene:** Zero secrets logged or persisted.

---

## 11. Additive Cryptographic Hashes

To preserve scientific provenance without mutating previously locked gates:
- Gate-5 `benchmark_contract_sha256`: `9ddd889b84d8705fae618879e5035556c80d0276a3dc2a58a7963937ebb59f77`
- Gate-6 `agent_contract_sha256`: `53aa37079ff44afa75d9a3f921b0c1f98c4600d882fced51fc8d9197795058eb`
- Gate-6 `platform_runtime_contract_sha256`: `c7698768539a769c7b2bc6b90771ff234276a974b6e0353bbb03119faf2f79ad`
- Gate-7 `logging_contract_sha256`: `02d480cbb78e876b4c91a4eb16d831106de4199e369c9b53b50f91b142636f9f`
- Gate-7 `platform_observability_contract_sha256`: `335d96590dd9bbcaf2eb1077d497ebb155acf3abae9bf299bb47087a8dbb81d8`

```json
{
  "gate5_benchmark_contract_sha256": "9ddd889b84d8705fae618879e5035556c80d0276a3dc2a58a7963937ebb59f77",
  "gate6_agent_contract_sha256": "53aa37079ff44afa75d9a3f921b0c1f98c4600d882fced51fc8d9197795058eb",
  "platform_runtime_contract_sha256": "c7698768539a769c7b2bc6b90771ff234276a974b6e0353bbb03119faf2f79ad",
  "logging_contract_sha256": "02d480cbb78e876b4c91a4eb16d831106de4199e369c9b53b50f91b142636f9f",
  "platform_observability_contract_sha256": "335d96590dd9bbcaf2eb1077d497ebb155acf3abae9bf299bb47087a8dbb81d8"
}
```

$$\text{platform\_observability\_contract\_sha256} = \text{SHA256}(\text{canonical\_json}(\{\text{"platform\_runtime\_contract\_sha256"}, \text{"logging\_contract\_sha256"}, \dots\}))$$

---

## 12. Unit Tests & Verification Summary

Implemented in `tests/test_logging_contract.py` (16 pure unit tests executing in $<4.0\text{ s}$ without network dependencies):
- Verified `RunManifest` construction, formatting, and schema.
- Verified duplicate run protection (`FileExistsError` on existing run directory).
- Verified episode logging, streaming CSV writes, and Gate-4 aggregate calculation.
- Verified null metric semantics (`null` in JSON, never converted to `0.0`).
- Verified expected episode count enforcement (mismatch causes `RunStatus.FAILED`).
- Verified fake W&B backend metric parity and `evaluation_episodes` Table parity.
- Verified W&B network failure is non-fatal to local run.
- Verified local disk write failure strictly prevents `RunStatus.COMPLETE`.
- Verified decision latency boundary excludes simulated logger delay.
- Verified secret absence across all serialized manifests and configs.
- Verified machine path privacy (no usernames or home directory paths).
- Verified `WandbMode.DISABLED` functions standalone without credentials or network.
- Verified `WandbMode.OFFLINE` sets offline mode without network.
- Verified technical failures logged to `technical_failures.jsonl`.
- Verified runtime dataclass and enum introspection against `logging_contract_v1.json`.
- Verified logging contract hash mutation sensitivity.

---

## 13. Regression & Integrity Validation
- Pure unit tests: **99 total tests pass across repo** (`test_episode_lifecycle`: 12, `test_reward_metrics`: 14, `test_evaluation_protocol`: 20, `test_agent_contract`: 37, `test_logging_contract`: 16).
- `CourseEnvV1`: Reset observation shape `(35,)` verified.
- `evaluate_random.py`: Completed 20 evaluation episodes cleanly.
- Gates 1 through 6 contracts remained completely untouched.
- Zero hardcoded credentials or drive letters in executable code.

STOP. PR remains open and unmerged.
