# Research Platform V1: Workbench Hardening Audit Report (Gate 7.5B Pass B3)

## Executive Summary
This document provides the authoritative architectural and audit report for **Gate 7.5B Pass B3: Final Workbench Hardening & Platform Handoff** on the autonomous-driving research platform (`autonomous-driving-rl`).

Pass B3 is the final stabilizing and hardening pass for the Research Workbench presentation and control layer, built strictly on top of the merged baseline (`9c5d1fabe07e0c4cc24c28d0b5b2857f5496652d`).

### Explicit Anti-Scope-Creep Rule & Closure
Pass B3 is NOT a feature-development phase. Upon independent review and merge of PR #24, **Gate 7.5B is CLOSED**. There is NO planned Pass B4. The immediate next research milestone is **Stage 0 — Random / Naive Baseline Agent**.

---

## 1. Process Lifecycle Hardening
- **Single Worker Invariant:** `WorkbenchProcessRunner` strictly permits at most 1 active worker subprocess. Rapid double-starts raise `RuntimeError`.
- **Unexpected Worker Exit Handling:** Nonzero or zero process exit without structured `WORKER_DONE` emission is classified as `UnexpectedWorkerExit`. The GUI resets operation state deterministically without fabricating `COMPLETE` or `INTERRUPTED`.
- **Force Termination Semantics:** Labeled non-graceful `FORCE TERMINATE`. Kills worker process and cleans temp files without modifying on-disk `run_state.json`.
- **Bounded Termination Truthfulness:** `force_terminate_and_wait(timeout_ms)` kills the worker, waits boundedly, and reports success only when `QProcess.state()` positively reports `NotRunning`. Only then are running state, active operation and the temp request file cleared. If termination cannot be confirmed, it returns `False` and **retains** running state and active operation so the GUI keeps managing the potentially live process; a later `finished` signal is cleaned up by the normal `_on_process_finished()` path. No Gate-7 lifecycle state is ever fabricated.
- **Window Close While Running:** Close requires explicit confirmation. If bounded termination is unconfirmed, a diagnostic is shown and the close event is ignored; because the runner still reports active, any subsequent close attempt re-enters the running-worker confirmation guard.
- **Temp Request Lifecycle:** Temporary launch request JSON is unlinked on all terminal paths (success, blocked, error, unexpected exit, kill).

---

## 2. Operation State Machine Hardening
- **Synchronous UI Lock:** Clicking `Resolve` or `Run` locks all request-defining fields synchronously, eliminating the race window prior to `QProcess.started`.
- **Stale Preflight Invalidation:** Modifying any request field invalidates previous preflight. Late arriving signals during active operations cannot authorize modified requests.
- **Results Auto-Load Isolation:** Auto-navigation to Results occurs only upon successful `RUN` operations. `PLAN` operations and failed runs never alter `ResultsWidget` selection or root.

---

## 3. Protocol & Filesystem Resilience
- **Protocol Robustness:** Prefix-strict sentinel parsing buffers split chunks, handles multiple messages per read, rejects malformed sentinel JSON safely, and flushes trailing sentinels without newline.
- **Path Containment:** `RunArtifactRepository.load_run_snapshot` enforces strict immediate-child containment and rejects directory traversal or symlink escapes outside `runs_root`.
- **Discovery Error Surface:** Filesystem errors (e.g. `PermissionError`) surface in `last_discovery_error` without crashing the application.

---

## 4. GUI Resource Bounds
- **Bounded Event Log:** `RunMonitorWidget.text_event_log` document is capped at `MAX_EVENT_LOG_BLOCKS = 5000`. Excess lines are truncated from the head while preserving newest tail lines.
- **Live Chart Lifecycle:** Axes and lines reset cleanly upon `EPISODE_STARTED` and `reset_monitor`, preventing unbounded line artist growth.
- **Execution-Reported Labeling:** Telemetry overview in `RunMonitorWidget` is explicitly labeled `provisional / not integrity-verified` until loaded and verified in `ResultsWidget`.

---

## 5. Cryptographic Contract Hashes

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
| `workbench_contract_sha256` (B1) | `e5711485e6571a04c336739ebc6f285213a1d9631fa29820a89707b37875a82b` | PASS |
| `platform_workbench_contract_sha256` (B1) | `36f3a0891a9153df12a8b53b01f2064afbf848eafda462149a768ec989199e47` | PASS |
| `workbench_results_contract_sha256` (B2) | `21c2c541a891967d9206e2dc86781f1bfc62b794c13e7e1ee1c3732676e38429` | PASS |
| `platform_workbench_results_contract_sha256` (B2) | `a241e2890b65290e28d717714123625bc2809100fe3287f2fb3e14dba8d6e433` | PASS |

### Candidate Gate 7.5B Pass B3 Hashes
- **`workbench_hardening_contract_sha256`**: `adb11f5c205e0ee6651f1544f76deee36416fc49070a52a513cc5c786ceecdab`
- **`platform_workbench_hardening_contract_sha256`**: `b476749713e8a7afa2eafe50a5a89775550afe051a16b203a07d846c41f6972c`

---

## 6. Verification & Regression Evidence

### A. Pass B3 Unit Tests (`tests/test_workbench_hardening.py`)
- 49 tests executed: 48 passed, 1 skipped conditionally on Windows when non-admin symlink creation is prohibited.

### B. Total Platform Regression
- `tests.test_agent_contract`: 37 run, 37 passed, 0 skipped.
- `tests.test_episode_lifecycle`: 10 run, 10 passed, 0 skipped.
- `tests.test_evaluation_protocol`: 20 run, 20 passed, 0 skipped.
- `tests.test_reward_metrics`: 16 run, 16 passed, 0 skipped.
- `tests.test_workbench`: 29 run, 29 passed, 0 skipped.
- `tests.test_workbench_results`: 46 run, 46 passed, 0 skipped.
- `tests.test_workbench_hardening`: 49 run, 48 passed, 1 skipped.
- `tests.test_logging_contract`: 47 run, 47 passed, 0 skipped.
- `tests.test_launcher_core`: 97 run, 97 passed, 0 skipped.
- **Total Suite:** **351 tests run, 350 passed, 1 skipped, 0 failures, 0 errors.**

### C. Machine-Derived Audit Smokes (`scripts/audit_workbench_hardening.py`)
Every boolean PASS field in the B3 artifacts is derived from an observed value in the audit run and asserted before the artifact is written; no required PASS field is a literal constant.

1. **Process Lifecycle (`process_lifecycle_smoke.json`):** Verified single worker max, confirmed bounded termination of a real spawned worker (`force_term_confirmed_not_running`), unexpected exit classification, and temp file cleanup. `failed_to_start_real_executable_verified` is derived from the observed signals of a launch against a nonexistent executable (exactly one `workerError` with `error_type == FailedToStart`, exactly one `processFinished(-1)`, runner stopped, operation cleared, temp request removed) **before** any duplicate handler invocation; idempotence is checked only afterwards.
2. **Operation State (`operation_state_smoke.json`):** Verified synchronous lock, late signal protection, state recovery, and stale Results autoload protection.
3. **Protocol Resilience (`protocol_resilience_smoke.json`):** Split-chunk buffering, multiple messages per chunk, and trailing sentinel without newline observed through the real `WorkbenchProcessRunner._on_ready_read_stdout()` / `_on_process_finished()` paths; malformed input handling.
4. **Filesystem Resilience (`filesystem_resilience_smoke.json`):** Verified path containment, disappearing run handling, discovery error surfacing, and incomplete artifact handling.
5. **Resource Bounds (`resource_bounds_smoke.json`):** Verified document log bounded to 5000 blocks and observed chart line cleanup across episodes.
6. **Performance Smoke (`performance_smoke.json`):** 100 synthetic run directories and a 1000-row episode table are populated without exception; the episode table is observed read-only. Timings are **informational and non-gating**, vary on every run, and are intentionally not reproduced in this document — `performance_smoke.json` is the sole machine-derived source of measured values.
7. **Scientific Boundary (`scientific_boundary_regression.json`):** TEST + fixture run observed blocked by preflight (`agent_benchmark_eligibility`, `WORKER_DONE.blocked == True`) with no `LAUNCHER_EVENT` / `EXECUTION_REPORT` emitted.
8. **Real Sandbox Smoke (`real_sandbox_smoke.json`):** Verified real MetaDrive execution, 7/7 artifacts, and VERIFIED integrity.
9. **Artifact Privacy Scan:** Verified 0 private machine paths or secrets across all 10 B3 artifacts.

---

## 7. Manual Native Render Verification
- Manual native-render verification procedure documented; not part of automated evidence. Automated test and audit runs execute strictly headless (`render=OFF`, `QT_QPA_PLATFORM=offscreen`).
