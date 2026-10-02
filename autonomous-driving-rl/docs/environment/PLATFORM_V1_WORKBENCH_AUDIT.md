# Research Platform V1: Workbench GUI Audit Report (Gate 7.5B Pass B1)

## Executive Summary
This document provides the authoritative architectural and audit report for **Gate 7.5B Pass B1: Research Workbench Foundation** on the autonomous-driving research platform (`autonomous-driving-rl`).

Gate 7.5B implements a clean, desktop presentation and control layer strictly on top of the already merged and frozen Gate 7.5A Launcher Core baseline (`0376da8b8bd3e1c3b60d43ed371b6713c0cb31f6`).

### Absolute Architectural Invariant
> **"The Research Workbench can disappear entirely and every scientific experiment remains fully definable, resolvable, validated, executable, and reproducible through Gate 7.5A alone."**

The Workbench GUI does **not** possess, duplicate, or alter scientific authority:
- All resolution of cases, splits, holdout locks, and configs is performed exclusively by Gate 7.5A (`src/launcher/resolver.py`).
- All pre-execution safety gates are enforced strictly by Gate 7.5A preflight (`src/launcher/preflight.py`).
- All simulator execution and telemetry emission are driven strictly by Gate 7.5A executor (`src/launcher/executor.py`).
- UI field disabling (e.g., locking case selection and forcing `Render=OFF` in `VALIDATION` / `TEST` modes) is purely UX convenience.

---

## 1. Process Isolation Architecture

MetaDrive experiment execution **must never** run directly in the Qt GUI thread. In Gate 7.5B Pass B1:
```
PySide6 GUI Main Thread
       │  (QProcess)
       ▼
Python Worker Subprocess (`src/workbench/worker.py`)
       │  (sys.executable / exact Conda env)
       ▼
Gate 7.5A Core Engine (`resolve_experiment_plan`, `run_preflight`, `ExperimentExecutor`)
       │
       ▼
MetaDrive Simulator (0.4.3 / commit 85e5dadc...)
```

1. **Subprocess Invocation:** `WorkbenchProcessRunner` invokes `src/workbench/worker.py` via `QProcess` using `sys.executable`.
2. **Sentinel JSONL Protocol:** The worker outputs structured messages over stdout prefixed with sentinel `@@WORKBENCH@@`.
3. **Panda3D / MetaDrive Output Isolation:** Raw engine/library logs on stdout/stderr pass through untouched and are never misparsed as protocol events.
4. **GUI Thread Safety:** All QWidget updates occur on Qt's main event thread via standard Qt signals (`planResolved`, `preflightReport`, `launcherEvent`, etc.).

---

## 2. Worker Sentinel Protocol (`WORKBENCH_PROTOCOL_V1`)

All structured messages adhere to:
```json
@@WORKBENCH@@{"protocol_version": "WORKBENCH_PROTOCOL_V1", "type": "<TYPE>", "timestamp": "<ISO8601>", "payload": { ... }}
```

### Supported Message Types
- `WORKER_READY`: Subprocess initialized, reporting PID and Python path.
- `PLAN_RESOLVED`: Contains full `ResolvedExperimentPlanV1` dict and `resolved_plan_sha256`.
- `PREFLIGHT_REPORT`: Contains full `PreflightReportV1` dictionary, checks list, and `can_execute` verdict.
- `LAUNCHER_EVENT`: Streams `LauncherEventV1` telemetry (`RUN_STARTED`, `EPISODE_STARTED`, `EPISODE_PROGRESS`, `EPISODE_FINISHED`, `RUN_FINISHED`).
- `EXECUTION_REPORT`: Contains final `ExecutionReportV1` summary record.
- `WORKER_ERROR`: Emitted upon unhandled worker exceptions with traceback.
- `WORKER_DONE`: Final lifecycle event with exit status and `can_execute` / `blocked` flag.

### Worker Operations
- **`PLAN`:** Reconstructs `LaunchRequestV1` -> resolves plan -> executes preflight -> emits `PLAN_RESOLVED` and `PREFLIGHT_REPORT` -> terminates with 0 simulator steps.
- **`RUN`:** Reconstructs `LaunchRequestV1` -> resolves plan -> executes preflight -> if blocked, terminates with `blocked=True`; if allowed, instantiates `ExperimentExecutor` and streams `LauncherEventV1` telemetry to GUI -> emits `EXECUTION_REPORT` -> terminates.

---

## 3. Native Rendering Policy

In Pass B1:
- For `SANDBOX` / `AUDIT` with `render=NATIVE`, MetaDrive opens its normal separate native OS rendering window.
- No Panda3D surface is embedded in a `QWidget`.
- No offscreen RGB frame streaming into Qt is implemented.
- The scientific observation profile remains completely unaltered.

---

## 4. Cancellation & Process Termination Limitation

In Pass B1:
- `LauncherEventV1` is strictly observational telemetry, never a control backchannel.
- Raising `KeyboardInterrupt` from event callbacks is strictly prohibited.
- An emergency termination button is labeled explicitly: **`FORCE TERMINATE`**.
- The warning dialog states: `Force termination is non-graceful and may leave the local run state marked RUNNING. It does not rewrite run_state or claim INTERRUPTED.`
- Window close attempts during active worker execution display an explicit confirmation dialog preventing accidental silent kills.

---

## 5. UI Capabilities Summary

The Workbench contains 5 tabs:
1. **Experiment Setup:** Exposes `LaunchRequestV1` parameters (Mode, Agent, Tier, Sequence, Geom/Env/Agent Seeds, Render, W&B, Custom ID). Dynamic invalidation ensures `Run Experiment` is disabled until `Resolve & Preflight` achieves `can_execute=True`.
2. **Plan & Preflight:** Displays resolved plan metadata, case boundaries, and preflight checks table with color-coded PASS, WARNING, and FAIL status.
3. **Run Monitor:** Live telemetry cards (Run ID, Episode, Step, Route Completion, Speed, Status), progress bar, and chronological event log.
4. **Agents Explorer:** Read-only inspection of registered agents in `AgentRegistryV1`.
5. **Case Explorer:** Read-only browsing and filtering of frozen Gate-5 manifest cases. **No partial VALIDATION or TEST execution controls exist.**

---

## 6. Privacy & Secret Handling

- Neither `WANDB_API_KEY` nor any authentication tokens are accepted via CLI or serialized into request JSON.
- `WANDB_API_KEY` is inherited exclusively from the environment.
- No private machine paths or tokens are committed in audit artifacts.

---

## 7. Cryptographic Contract Hashes

### Frozen Baseline Hashes (Verified Untouched)
| Contract Name | Expected Hash | Verified Status |
|---|---|---|
| `gate5_benchmark_contract_sha256` | `9ddd889b84d8705fae618879e5035556c80d0276a3dc2a58a7963937ebb59f77` | PASS |
| `gate6_agent_contract_sha256` | `53aa37079ff44afa75d9a3f921b0c1f98c4600d882fced51fc8d9197795058eb` | PASS |
| `gate6_platform_runtime_contract_sha256` | `c7698768539a769c7b2bc6b90771ff234276a974b6e0353bbb03119faf2f79ad` | PASS |
| `gate7_logging_contract_sha256` | `0d17915556d83f49d9519576e95920c0853700f33a91593ef85b3aeef6cbb9f2` | PASS |
| `gate7_platform_observability_contract` | `f00c27fca8abf8145596576cd0dd2540181ff8ae7eccf7806c864921933d3581` | PASS |
| `gate7_5a_launcher_contract_sha256` | `5e6269b6e32349bd3ee34ca52583c00f832bd0a737330c894e9a8d314653db04` | PASS |
| `gate7_5a_platform_execution_contract_sha256` | `442beafd86212419dc7b3edfa60e53715bf33877d3dd72e61c54091520b0563d` | PASS |
| `gate7_5a_canonical_agent_registry_sha256` | `9fac42d07949796d00b6dc869b98c945eae7a59355d7928f429e457f1ee7f26f` | PASS |

### Candidate Gate 7.5B Hashes
- **`workbench_contract_sha256`**: `e5711485e6571a04c336739ebc6f285213a1d9631fa29820a89707b37875a82b`
- **`platform_workbench_contract_sha256`**: `36f3a0891a9153df12a8b53b01f2064afbf848eafda462149a768ec989199e47`

---

## 8. Verification Results

### A. Unit Test Suite (`tests/test_workbench.py`)
23 tests executed in offscreen mode (`QT_QPA_PLATFORM=offscreen`):
- All 23 tests passed cleanly in ~13s.
- Total platform test suite: 250 tests (227 prior + 23 Workbench), 0 failures, 0 errors.

### B. Machine-Derived Audit Smokes (`scripts/audit_workbench.py`)
1. **Startup Smoke (`startup_smoke.json`):** Verified offscreen `MainWindow` initialization, 5 tabs, combobox population, Extreme tier availability, and exact manifest table bindings (TRAIN 180, VALIDATION 96, TEST 60).
2. **Worker PLAN Smoke (`plan_smoke.json`):** Verified plan resolution and preflight execution without simulator invocation. Sanitized `WORKER_READY` payload (basename only).
3. **Worker RUN Smoke (`run_smoke.json`):** Verified real MetaDrive execution under `SANDBOX`, streaming 10 Hz telemetry, `status="COMPLETE"`, `WORKER_DONE.success=True`, `run_id` parity, and writing 7/7 required Gate-7 artifacts (`run_manifest.json`, `run_state.json`, `episodes.csv`, `timing.csv`, `summary.json`, `wandb_sync.json`, `run_integrity.json`).
4. **Canonical Blocked Smoke (`canonical_blocked_smoke.json`):** Verified benchmark `TEST` request with non-benchmark fixture resolves plan, fails preflight with targeted reason `agent_benchmark_eligibility`, and exits with 0 simulator steps.
5. **Artifact Privacy Scan:** Verified 0 private absolute interpreter paths, user directories, or secrets across all committed audit artifacts.

### C. Manual Native Render Check
- Mode: `SANDBOX`
- Agent: `fixture_constant_continuous`
- Render: `NATIVE`
- Expected & Documented Manual Check:
  - MetaDrive opens a separate native window.
  - Workbench GUI remains responsive and continues logging telemetry.
  - Normal closing of native window does not corrupt Workbench state.
- **Evidence status:** Documented as manual operational verification. Automated test runs strictly headless (`render=OFF`, `QT_QPA_PLATFORM=offscreen`).

---

## 9. Non-Goals Maintained
- No Stage-0 Random agent implemented.
- No embedded MetaDrive viewport in QWidget.
- No offscreen camera streaming.
- No benchmark subset runs permitted.
- No graceful cancellation hacks.
- No binary packaging (`.exe`).
- Gate 7.5B status: **`AUDIT-CANDIDATE`**.
