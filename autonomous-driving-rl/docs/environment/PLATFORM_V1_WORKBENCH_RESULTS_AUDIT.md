# Research Platform V1: Workbench Results & Live UX Audit Report (Gate 7.5B Pass B2)

## Executive Summary
This document provides the authoritative architectural and audit report for **Gate 7.5B Pass B2: Results Browser, Artifact Integrity & Live UX** on the autonomous-driving research platform (`autonomous-driving-rl`).

Pass B2 extends the Research Workbench presentation and control layer strictly on top of the merged Gate 7.5B Pass B1 baseline (`99f0a66aa156004a5a1ec7993d2368ff83d7c325`).

### Scientific Invariant
> **"summary.json is authoritative aggregate result truth. episodes.csv is authoritative episode-row truth. timing.csv is authoritative episode timing truth. run_state.json is authoritative lifecycle truth. run_integrity.json is authoritative final fingerprint declaration. Workbench only reads, verifies, formats, filters, and visualizes; it never becomes the source of those truths."**

Deleting `src/workbench/` leaves every completed experiment run on disk completely self-contained, reproducible, verified, and accessible.

---

## 1. Results Architecture & Local-First Authority
Gate 7 local run directories are the sole scientific authority:
- `RunArtifactRepository` (`src/workbench/results_repository.py`) discovers and inspects runs on disk without modifying, repairing, appending to, or deleting any files.
- Consumes exact Gate-7 production schemas: `RunManifestV1`, `RunStateV1`, `ExperimentRunConfig.to_dict()`, and `RunIntegrityRecord`.
- Enforces strict run identity parity across directory basename, `run_state.json`, `run_manifest.json`, and `run_integrity.json`.
- Enforces strict path containment preventing directory traversal outside the configured runs root.
- `RunArtifactSnapshotV1` is an immutable, read-only view model representing a single run folder.
- All presentation tables in `ResultsWidget` (`src/workbench/widgets/results_widget.py`) enforce `QAbstractItemView.EditTrigger.NoEditTriggers`.

---

## 2. Integrity Status Model & Trust-Gating
The Workbench defines four mutually exclusive integrity display states:
1. **`VERIFIED`:** Run lifecycle status is `COMPLETE`, all 7 core Gate-7 artifacts are present, `run_integrity.json` exists, and every semantic hash matches observed disk contents via Gate-7 canonical semantic content hash verification (`canonical_json_file_sha256`, `canonical_csv_file_sha256`). Authoritative metric cards are displayed normally.
2. **`NOT_FINAL`:** Run lifecycle status is `INITIALIZING`, `RUNNING`, `INTERRUPTED`, or `FAILED`. Incomplete runs truthfully lack final summary or integrity records and are never misclassified as corrupted.
3. **`UNVERIFIED`:** Run claims `COMPLETE` status but lacks `run_integrity.json`.
4. **`FAILED`:** Run claims `COMPLETE` but one or more artifact fingerprints mismatch observed disk contents.
   - **Trust-Gating Behavior:** Potentially tampered summary values are never displayed under an authoritative heading. The UI surfaces an explicit `UNTRUSTED — INTEGRITY FAILED` state with prominent warning banner. No auto-repair or hash regeneration is ever performed.

---

## 3. Primary vs Diagnostic Metrics
- **Primary Benchmark Metrics:** Loaded directly from `summary.json["overall_metrics"]` without GUI recalculation:
  - `clean_success_rate`
  - `safety_failure_rate`
  - `mean_final_route_completion`
  - `median_final_route_completion`
  - `mean_time_to_clean_success_s` (displayed as `N/A — no clean-success samples` when `None`, never converted to zero).
- **Stored Macro Metrics:** When `summary.json["macro_metrics"]` is present, it is displayed directly without recomputation.
- **Diagnostic / Training Signals:** Clearly labeled as `Diagnostic / training signal — NOT a benchmark ranking score`:
  - `mean_episode_return`
  - `episode_return`
- **Zero Ranking Invariant:** Pass B2 does NOT implement leaderboards, agent rankings, multi-run composite scores, or automated model selection.

---

## 4. Live Sampled Telemetry UX
- **Cadence Clarification:** Control and decision loop executes at 10 Hz (`dt=0.1s`). The current `ExperimentExecutor` emits `EPISODE_PROGRESS` events every 10 decision steps. Thus progress telemetry is sampled approximately once per simulated second under the current executor.
- **`LiveTelemetryBufferV1`:** Consumes only passive `LauncherEventV1` events:
  - `RUN_STARTED` resets the entire run buffer.
  - `EPISODE_STARTED` resets the current episode trace.
  - `EPISODE_PROGRESS` appends `(step_index, route_completion, speed_kmh)` points.
  - `EPISODE_FINISHED` finalizes the episode trace.
  - `RUN_FINISHED` / `RUN_FAILED` / `RUN_INTERRUPTED` marks completion state.
- **Embedded Matplotlib Charts:** Real-time curves for *Route Completion vs Decision Step* and *Speed vs Decision Step* rendered on Qt canvas.
- **Scientific State Integrity:** Free-text `event.message` is never parsed to infer scientific outcome state.

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

### Candidate Gate 7.5B Pass B2 Hashes (Recomputed Semantic Wording)
- **`workbench_results_contract_sha256`**: `25716cc6aa0d93b2e6a282e51a7e23b37b050a975654ae2852726c2e86828244`
- **`platform_workbench_results_contract_sha256`**: `b53e56b6344c53299ffdee8f1402e4fccb16c375eeb08205897cf909effd89a3`

---

## 6. Verification & Regression Evidence

### A. Pass B2 Unit Tests (`tests/test_workbench_results.py`)
- 33 dedicated tests covering real Gate-7 schema parsing, run identity parity, path containment, tamper detection, trust-gated primary metrics, custom runs root auto-load, complete episodes/timing/provenance/macro columns, NoEditTriggers, live telemetry buffering, and contract determinism.
- All 33 tests passed cleanly.

### B. Total Platform Regression
- `tests.test_agent_contract`: 37 tests (PASS)
- `tests.test_episode_lifecycle`: 10 tests (PASS)
- `tests.test_evaluation_protocol`: 20 tests (PASS)
- `tests.test_reward_metrics`: 16 tests (PASS)
- `tests.test_workbench`: 29 tests (PASS)
- `tests.test_workbench_results`: 33 tests (PASS)
- `tests.test_logging_contract`: 47 tests (PASS)
- `tests.test_launcher_core`: 97 tests (PASS)
- **Total Suite:** **289 tests**, 0 failures, 0 errors.

### C. Machine-Derived Audit Smokes (`scripts/audit_workbench_results.py`)
1. **Startup Smoke (`results_startup_smoke.json`):** Offscreen construction with 6 main tabs, 6 read-only results subtabs, and 8-column runs browser.
2. **Complete Run Load Smoke (`complete_run_load_smoke.json`):** Real simulation run executed, loaded from disk, evaluated as `VERIFIED` with exact Gate-7 schema parity.
3. **Isolated Tamper Detection Smoke (`tamper_detection_smoke.json`):** Exact canonical semantic content hash mismatch detection on copied run without repairing or rewriting artifacts.
4. **Incomplete Run Semantics Smoke (`incomplete_run_semantics_smoke.json`):** `RUNNING` and `FAILED` runs evaluated truthfully as `NOT_FINAL`.
5. **Live Telemetry Smoke (`live_telemetry_smoke.json`):** In-memory trace buffering verified strictly from `LauncherEventV1`.
6. **Artifact Privacy Scan:** Verified 0 private machine paths or credentials persisted in audit artifacts.

---

## 7. Pass B2 Non-Goals Maintained
- No Stage-0 Random agent implemented.
- No embedded MetaDrive viewport in QWidget.
- No offscreen camera streaming.
- No multi-run ranking, leaderboards, or composite scores.
- No artifact deletion or editing controls.
- Gate 7.5B Status: **`AUDIT-CANDIDATE`**.
