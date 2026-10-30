# Platform V1 — Gate R0 Integration Recovery Audit

- **Document ID:** `AUDIT-R0-INTEGRATION-RECOVERY`
- **Version:** `1.0.0`
- **Date:** 2026-10-31
- **Status:** `AUDIT-COMPLETE / READY FOR INDEPENDENT REVIEW`
- **Subproject Root:** `autonomous-driving-rl/`
- **Baseline Commit:** `925164f4988e855f36006d586b5454f96ded4b4f` (PR #28 Merge)
- **Author:** AI Software Engineering Agent (OpenCode Gemini)

---

## 1. Executive Summary & Purpose

Gate R0 (Integration Recovery) restores the technical execution and simulation pipeline of Research Platform V1 following regressions introduced during early Stage 0 development. This audit documents the exact root causes, minimal non-invasive fixes, registry fingerprint preservation, test suite recovery, and real SANDBOX execution evidence.

### Summary of Recoveries:
1. **Blocker R0-A (Stage 0 Import):** Resolved `ImportError` where `src/launcher/registry.py` imported `Stage0RandomNaiveAgent` from `src.agents` without package export.
2. **Blocker R0-B (Missing MinimapOverlay):** Resolved `ModuleNotFoundError: No module named 'src.visualization'` in `src/launcher/executor.py` by restoring MetaDrive native 3D rendering semantics without uncommitted minimap overlay dependencies.
3. **Registry Fingerprint Reconciliation:** Preserved the locked Gate 7.5A Historical Fixture Registry Baseline (`9fac42d07949796d00b6dc869b98c945eae7a59355d7928f429e457f1ee7f26f`) via `build_historical_fixture_registry()` while verifying the active 5-agent canonical registry (`build_canonical_agent_registry()`).
4. **Test Suite Status:** Full unit test discovery across all 11 test suites recovered from blocked status to **400 tests passing (100% OK)**.
5. **Real Simulation Recovery:** Successfully executed a full end-to-end SANDBOX episode on TRAIN geometry, generating complete Gate-7 artifacts with `status: COMPLETE` and `IntegrityDisplayStatus.VERIFIED`.

---

## 2. Root Cause Analysis

### 2.1 Blocker R0-A: Stage 0 Package Export Failure
- **Symptom:** `src/launcher/registry.py` contained `from src.agents import Stage0RandomNaiveAgent`.
- **Root Cause:** When Stage 0 was introduced on 2026-10-09 (commit `8184d41`), the class was implemented in `src/agents/random_naive_agent.py`, but `src/agents/__init__.py` was left empty (containing only a docstring). Consequently, importing from package root `src.agents` threw `ImportError: cannot import name 'Stage0RandomNaiveAgent' from 'src.agents'`.
- **Resolution:** Exported `Stage0RandomNaiveAgent` in `src/agents/__init__.py`. This provides a clean package interface, avoids circular dependencies, and requires zero changes to agent implementation logic.

### 2.2 Blocker R0-B: Missing `src.visualization.minimap_overlay`
- **Symptom:** Executing or importing `src/launcher/executor.py` failed unconditionally with:
  ```text
  ModuleNotFoundError: No module named 'src.visualization'
  ```
- **Root Cause:** An incomplete development commit added `from src.visualization.minimap_overlay import MiniMapOverlay` to `src/launcher/executor.py`, instantiating and updating an overlay in the simulation loop. However, the `src/visualization/` directory was never committed to git. This caused the launcher to crash on import, even when headless execution (`render_mode="OFF"`) was requested.
- **Historical Comparison:** Inspection of historical commit `6a531b148b40cff0f7dd343552090aeeb5675957` confirmed that before the overlay was introduced, MetaDrive native rendering was cleanly handled via `"use_render": (render_mode.upper() == "NATIVE")` in `build_metadrive_case_config()`.
- **Resolution:** Removed the missing overlay import, initialization, step update, and cleanup hooks from `src/launcher/executor.py`. When `render_mode="NATIVE"` is requested, MetaDrive natively handles 3D Panda3D window creation, rendering, and destruction. When `render_mode="OFF"`, headless execution proceeds with zero graphical overhead.

---

## 3. Registry Fingerprint Reconciliation & Historical Baseline Preservation

### 3.1 The Architectural Tension
- At Gate 7.5A, the canonical registry contained exactly four audit fixture agents:
  1. `fixture_constant_continuous`
  2. `fixture_seeded_random`
  3. `fixture_stateful_counter`
  4. `fixture_discrete`
  Its SHA-256 fingerprint was locked in contract records as `9fac42d07949796d00b6dc869b98c945eae7a59355d7928f429e457f1ee7f26f`.
- When `stage0_random_naive` was registered as a 5th agent, the dynamic canonical registry fingerprint changed to `39cc8f88aea0e5d5bf65e74bd379660f6e9b233bfcd2d019de269221545f1d5a`.
- Older tests (`test_workbench.py`, `test_launcher_core.py`) and audit scripts asserted `len(reg) == 4` and hardcoded hash equality against `9fac42d0...`.

### 3.2 Implemented Architectural Distinction
To prevent modifying frozen historical records while accommodating active development:
1. **`HISTORICAL_FIXTURE_REGISTRY_SHA256` (`9fac42d07949796d00b6dc869b98c945eae7a59355d7928f429e457f1ee7f26f`):** Defined as an immutable constant in `src/launcher/registry.py`.
2. **`build_historical_fixture_registry() -> AgentRegistryV1`:** Explicitly populates the four Gate-7.5A fixture agents. Its computed fingerprint strictly equals `HISTORICAL_FIXTURE_REGISTRY_SHA256`.
3. **`build_canonical_agent_registry() -> AgentRegistryV1`:** Extends the historical fixture baseline by registering `stage0_random_naive`. Returns all five active agents.
4. **Test & Audit Reconciliation:**
   - `tests/test_launcher_core.py`: Asserts all 4 fixtures and `stage0_random_naive` are present (total 5); verifies `build_historical_fixture_registry()` matches `HISTORICAL_FIXTURE_REGISTRY_SHA256`.
   - `tests/test_workbench.py`: In `test_14`, verifies `build_canonical_agent_registry()` has 5 agents and `build_historical_fixture_registry()` has 4 agents; in `test_15`, verifies `build_historical_fixture_registry()` matches `HISTORICAL_FIXTURE_REGISTRY_SHA256` and canonical registry hash is dynamically deterministic.
   - `scripts/audit_workbench*.py`: Validates `hist_reg_hash == HISTORICAL_FIXTURE_REGISTRY_SHA256`.

---

## 4. Stage 0 Research Agent Integration

### 4.1 Registration Invariants
`stage0_random_naive` is registered in `src/launcher/registry.py` with:
- `agent_id`: `"stage0_random_naive"`
- `agent_version`: `"0.1.0"`
- `stage_label`: `"STAGE_0"`
- `method_family`: `"RANDOM"`
- `purpose`: `"RESEARCH_BASELINE"`
- `input_profile_id`: `"STATE_DECISION_V1"`
- `action_adapter_id`: `"discrete9_lowbranch_v1"`
- `inference_stochasticity`: `"stochastic"`
- `stateful_within_episode`: `True`
- `benchmark_eligible`: `False` (Blocked from `TEST` and `VALIDATION`)
- `sandbox_eligible`: `True`
- `audit_eligible`: `True`

### 4.2 Algorithm Observations for Stage 0 Lead (Khoa)
Static inspection identified two implementation issues in `src/agents/random_naive_agent.py` which are documented here for resolution during Stage 0 development (out of scope for Gate R0):
1. **Variable Typo:** `self._step_couter` is initialized and incremented in `__init__` and `act()`, whereas `self._step_counter` is reset in `reset()`.
2. **Action Index Annotation Discrepancy:** Line 28 notes `self._current_action = 7 # mặc định là thả trôi (coast)`. However, under `discrete9_lowbranch_v1`, index 7 corresponds to `steering = 0.0, throttle = 0.6` (Straight + Throttle), while coast (`throttle = 0.0`) is index 4 (`steering = 0.0, throttle = 0.0`).

---

## 5. Verification & Test Evidence

### 5.1 Import Smoke Tests
All primary entry points import cleanly with exit code 0:
```powershell
python -c "import src.launcher.registry; import src.launcher.executor; import src.launcher.cli; import src.workbench.core_adapter; print('OK')"
```
- **Result:** `OK` (Exit code: 0).

### 5.2 CLI Smoke Tests
1. **Agent Enumeration:**
   ```powershell
   python -m src.launcher.cli agents --json
   ```
   - **Result:** Output valid JSON listing all 5 agents (`fixture_constant_continuous`, `fixture_discrete`, `fixture_seeded_random`, `fixture_stateful_counter`, `stage0_random_naive`).
2. **SANDBOX Plan Resolution:**
   ```powershell
   python -m src.launcher.cli plan --mode SANDBOX --agent stage0_random_naive --tier Easy --render OFF --wandb DISABLED
   ```
   - **Result:** `Preflight Verdict: PROCEED_WITH_WARNINGS (Can Execute: True)`.
3. **TEST Benchmark Guard Enforcement:**
   ```powershell
   python -m src.launcher.cli plan --mode TEST --agent stage0_random_naive --render OFF --wandb DISABLED
   ```
   - **Result:** `[FAIL] (AGENT) Agent 'stage0_random_naive' is flagged benchmark_eligible=False. Preflight Verdict: BLOCKED (Can Execute: False)`.

### 5.3 Unit Test Suite Execution
Executed across all 11 repository test modules in active conda `metadrive` environment:

| Test Module | Tests Ran | Failures | Errors | Result |
|---|---|---|---|---|
| `tests.test_evaluation_protocol` | 20 | 0 | 0 | **PASS** |
| `tests.test_reward_metrics` | 16 | 0 | 0 | **PASS** |
| `tests.test_agent_contract` | 37 | 0 | 0 | **PASS** |
| `tests.test_episode_lifecycle` | 10 | 0 | 0 | **PASS** |
| `tests.test_logging_contract` | 47 | 0 | 0 | **PASS** |
| `tests.test_launcher_core` | 97 | 0 | 0 | **PASS** |
| `tests.test_workbench` | 29 | 0 | 0 | **PASS** |
| `tests.test_workbench_hardening` | 49 | 0 | 0 | **PASS** (1 skip) |
| `tests.test_workbench_results` | 46 | 0 | 0 | **PASS** |
| `tests.test_mapsuite_preview_export` | 27 | 0 | 0 | **PASS** |
| `tests.test_mapsuite_dataset` | 22 | 0 | 0 | **PASS** |
| **TOTAL** | **400** | **0** | **0** | **100% PASS** |

### 5.4 Real SANDBOX Execution Evidence
Executed real episode simulation with physical MetaDrive environment:
```powershell
python -m src.launcher.cli run --mode SANDBOX --agent fixture_constant_continuous --tier Easy --render OFF --wandb DISABLED --runs-root "<temp_dir>" --run-id "sandbox_recovery_smoke_001" --json
```
- **Execution Output:**
  - `run_id`: `"sandbox_recovery_smoke_001"`
  - `status`: `"COMPLETE"`
  - `completed_episodes`: `1 / 1`
  - `clean_success_rate`: `0.0`
  - `safety_failure_rate`: `1.0` (`out_of_road`)
  - `total_act_count`: `150`
  - `technical_failure_count`: `0`
- **Artifact Verification via `RunArtifactRepository`:**
  - Loaded snapshot: `sandbox_recovery_smoke_001`
  - `status`: `COMPLETE`
  - `integrity_status`: `IntegrityDisplayStatus.VERIFIED`
  - `canonical_run`: `False` (Correctly flagged as non-canonical sandbox)

### 5.5 Research Workbench GUI Smoke
Executed in offscreen Qt mode:
```powershell
$env:QT_QPA_PLATFORM="offscreen"
python -c "from PySide6.QtWidgets import QApplication; from src.workbench.main_window import MainWindow; app = QApplication([]); win = MainWindow(); print('Loaded:', win.windowTitle())"
```
- **Result:** `Loaded: Research Workbench V1 — Presentation & Control Layer (Gate 7.5B)`.

---

## 6. Files Changed in Gate R0

| File Path | Nature of Change |
|---|---|
| `src/agents/__init__.py` | Exported `Stage0RandomNaiveAgent` from `src.agents.random_naive_agent`. |
| `src/launcher/executor.py` | Removed missing `MiniMapOverlay` import and step/lifecycle hooks; restored native rendering path. |
| `src/launcher/registry.py` | Added `build_historical_fixture_registry()` and `HISTORICAL_FIXTURE_REGISTRY_SHA256`; canonical registry extends historical baseline. |
| `src/launcher/__init__.py` | Exported historical fixture registry helpers. |
| `tests/test_launcher_core.py` | Updated `test_initial_fixture_agents_registered` to verify all 5 agents and historical fixture baseline. |
| `tests/test_workbench.py` | Updated tests 14 & 15 to assert 5 canonical agents, 4 historical fixtures, and locked historical hash. |
| `scripts/audit_workbench.py` | Aligned historical fixture registry hash check. |
| `scripts/audit_workbench_results.py` | Aligned historical fixture registry hash check. |
| `scripts/audit_workbench_hardening.py` | Aligned historical fixture registry hash check. |
| `docs/environment/PLATFORM_V1_R0_INTEGRATION_RECOVERY_AUDIT.md` | Comprehensive Gate R0 recovery audit document (this file). |

### Strict Invariant Check
- Zero modifications to `configs/platform/`.
- Zero modifications to `datasets/mapsuite_v1/`.
- Zero modifications to locked `results/audits/`.
- Zero changes to frozen contract hashes of Gates 5, 6, 7, and 7.5A.
