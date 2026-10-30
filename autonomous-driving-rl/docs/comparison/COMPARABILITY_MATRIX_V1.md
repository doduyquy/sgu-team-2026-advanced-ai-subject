# XeTuHanh Platform V1 — Comparability Matrix V1

- **Document ID:** `MATRIX-COMP-V1`
- **Version:** `1.0.0-draft`
- **Date:** 2026-10-30
- **Status:** `DRAFT / TEAM REVIEW REQUIRED` — Traceable Decision Rules
- **Subproject Root:** `autonomous-driving-rl/`
- **Baseline Git Commit:** `dcd653306a0e072a2c6bee66eaeadf388101d679`

---

## 1. Framework and Decision Taxonomy

Comparing autonomous driving algorithms across stages requires verifying that observed performance differences reflect true algorithmic properties rather than disparities in evaluation splits, observation rights, action control spaces, or corrupted data artifacts.

Every pair of runs $(R_A, R_B)$ evaluated for scientific comparison must be mapped to exactly one of four compatibility verdicts:

```
+----------------------------------------------------------------------------------------------------+
|                                    VERDICT STATE DEFINITIONS                                       |
+----------------------------------------------------------------------------------------------------+
| 1. REJECTED_UNTRUSTED     | Hard block. At least one run is corrupted, incomplete, unverified, or |
|                           | executed in an untrusted sandbox/fixture state. Zero comparison.      |
+---------------------------+------------------------------------------------------------------------+
| 2. INCOMPATIBLE           | Block. Runs are valid individually, but evaluate fundamentally         |
|                           | different splits, manifests, simulator pins, or information rights.   |
+---------------------------+------------------------------------------------------------------------+
| 3. QUALIFIED_COMPARISON   | Allowed with mandatory caveats. Runs evaluate identical cases and      |
|                           | identical information rights, but differ in action adapter or timing   |
|                           | hardware. Valid as end-to-end system test, NOT as pure algorithm test.|
+---------------------------+------------------------------------------------------------------------+
| 4. STRICTLY_COMPARABLE    | Apples-to-apples benchmark comparison. Exact match across cases, seeds,|
|                           | input profile, action adapter, and verified platform contracts.       |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Systematic Comparability Matrix

| Row ID | Dimension / Scenario | Condition in Run A | Condition in Run B | Verdict | Scientific Rationale & Enforcement Rules |
|---|---|---|---|---|---|
| **M01** | **Artifact Integrity** | All 7 artifacts match `run_integrity.json` (`VERIFIED`) | One or more hash mismatches (`FAILED`) | `REJECTED_UNTRUSTED` | Tampered, modified, or corrupted local files. Hard block; zero scientific authority. |
| **M02** | **Lifecycle Completion** | `status == COMPLETE` | `status == RUNNING`, `INTERRUPTED`, `FAILED` | `REJECTED_UNTRUSTED` | Incomplete run (`NOT_FINAL`). Partial metrics cannot be compared against completed suites. |
| **M03** | **Integrity Declaration** | `run_integrity.json` exists & verifies | `run_integrity.json` missing (`UNVERIFIED`) | `REJECTED_UNTRUSTED` | Run lacks cryptographic finalization. Must not be used for authoritative benchmarking. |
| **M04** | **Clean Worktree** | `canonical_run == True` (`dirty == False`) | `dirty_override == True` (`dirty == True`) | `REJECTED_UNTRUSTED` | Uncommitted code changes introduce unverified side-effects. Blocked from canonical comparisons. |
| **M05** | **Agent Eligibility** | Research Agent (e.g., `ppo_stage5`) | Infrastructure Fixture (e.g., `fixture_constant_continuous`) | `REJECTED_UNTRUSTED` | Fixtures are sanity-check stubs, not valid scientific baselines. Preflight blocks them from canonical suites. |
| **M06** | **Dataset Split** | `split == "TEST"` ($N=60$) | `split == "VALIDATION"` ($N=96$) | `INCOMPATIBLE` | Different geometry distributions and case counts. Cross-split comparison is mathematically invalid. |
| **M07** | **Launcher Mode** | Mode `TEST` | Mode `SANDBOX` ($N=1$) or `AUDIT` | `INCOMPATIBLE` | Sandbox evaluates a single development case; audit evaluates contracts. Neither is a benchmark scorecard. |
| **M08** | **Episode Count & Coverage** | Complete expected count ($60/60$) | Missing episodes ($45/60$) or duplicate IDs | `REJECTED_UNTRUSTED` | Incomplete execution or duplicated episodes invalidate run-level statistics. |
| **M09** | **Case Set / Manifest Parity**| Canonical TEST Manifest (`0832c38e...`) | Different Case Manifest / Filtered Subset | `INCOMPATIBLE` | Both runs may be complete ($N=60$), but evaluating different scenario cases prevents paired comparison. |
| **M10** | **Simulator Pin** | MetaDrive `0.4.3` @ `85e5dadc` (`VERIFIED`) | Unverified commit / different MetaDrive (`MISMATCHED`) | `INCOMPATIBLE` | Physics, collision detection, or sensor rendering differences invalidate behavioral comparisons. |
| **M11** | **Information Rights** | Input Profile `STATE_DECISION_V1` | Input Profile `CORE_ONLY_V1` (Masked Context) | `INCOMPATIBLE` | Different observation spaces. Masked traffic/task context represents an ablation study, not a direct algorithmic contest. |
| **M12** | **Action Adapter** | `continuous_box2_v1` (Steering $[-1, 1]$, Accel $[-1, 1]$) | `discrete25_native_v1` (5x5 Steering-Throttle grid) | `QUALIFIED_COMPARISON` | Valid as an **end-to-end system comparison**. Cannot be claimed as pure algorithm superiority because discretization restricts vehicle dynamics. |
| **M13** | **Action Adapter** | `discrete25_native_v1` | `discrete9_lowbranch_v1` | `QUALIFIED_COMPARISON` | Control resolution disparity. Must disclose adapter details prominently in report. |
| **M14** | **Execution Hardware** | Workstation A (Intel i9 + RTX 4090) | Laptop B (AMD Ryzen 7 + Integrated GPU) | `QUALIFIED_COMPARISON` *(Timing Only)* | Task performance (success/safety/completion) remains `STRICTLY_COMPARABLE` under deterministic headless execution; **timing comparison is qualified/blocked**. |
| **M15** | **Sync Policy** | Latency Policy `FRAMEWORK_SYNCHRONIZED` | Latency Policy `NONE` | `QUALIFIED_COMPARISON` *(Timing Only)* | Asynchronous GPU kernel launches under `NONE` produce misleadingly optimistic `act()` latencies. Task metrics unaffected. |
| **M16** | **Identical Setup** | `STATE_DECISION_V1`, `continuous_box2_v1`, Clean, TEST ($N=60$), Verified | `STATE_DECISION_V1`, `continuous_box2_v1`, Clean, TEST ($N=60$), Verified | `STRICTLY_COMPARABLE` | True apples-to-apples algorithmic comparison across the five primary benchmark metrics. |

---

## 3. Detailed Decision Flow for Comparison Engines

*(Note: This logic specifies proposed comparison rules for future modules such as GUI V2 or offline comparison scripts. Current GUI V1/B2 only performs single-run inspection.)*

```
[START EVALUATION]
   |
   +--> Step 1: Check Layer A (Trustworthiness) for BOTH runs individually
   |    - Do all 7 required artifacts exist (run_manifest, run_state, episodes, timing, summary, wandb_sync, run_integrity)?
   |    - Does run_state.json status == COMPLETE?
   |    - Does _evaluate_integrity() yield IntegrityDisplayStatus.VERIFIED?
   |    - Is canonical_run == True (clean worktree, verified pin, no dirty/env overrides)?
   |    - Is recorded_episode_count == expected_episode_count with zero duplicate case_ids?
   |    - Is technical_failure_count == 0?
   |    IF NO  ===> Output: REJECTED_UNTRUSTED (Explain which run failed Layer A)
   |    IF YES ===> Proceed to Step 2
   |
   +--> Step 2: Check Layer B (Scientific Consistency) between Run A and Run B
   |    - Are splits identical? (e.g., both TEST or both VALIDATION)
   |    - Are manifest hashes identical? (exact same case IDs, geometries, and environment seeds)
   |    - Are simulator commit pins identical?
   |    - Are input profile IDs identical? (e.g., both STATE_DECISION_V1)
   |    IF NO  ===> Output: INCOMPATIBLE (Block comparison, explain mismatch)
   |    IF YES ===> Proceed to Step 3
   |
   +--> Step 3: Check Control Space and Hardware Context
   |    - Are action adapter IDs identical? (e.g., both continuous_box2_v1)
   |    - Are latency sync policies and hardware environments identical?
   |    IF ACTION ADAPTER DIFFERS:
   |        Output: QUALIFIED_COMPARISON (Valid as system comparison with mandatory disclosure)
   |    IF HARDWARE / TIMING DIFFERS:
   |        Output: STRICTLY_COMPARABLE for primary metrics; QUALIFIED_COMPARISON for decision latency
   |    IF ALL IDENTICAL:
   |        Output: STRICTLY_COMPARABLE (Apples-to-apples benchmark comparison)
```

---

## 4. Mandatory Reporting Disclosures by Verdict

### 4.1 When Verdict is `STRICTLY_COMPARABLE`
- Report the 5 primary metrics side-by-side (overall and across all 4 tiers).
- For learned agents, report mean $\pm$ sample standard deviation across independent training replicates ($R=3$).
- Show paired per-case difference histograms $(\Delta = B - A)$ across the 60 canonical cases.

### 4.2 When Verdict is `QUALIFIED_COMPARISON`
- **Action Adapter Disparity:**
  - Mandatory disclosure: *"Notice: Agent A uses action adapter `continuous_box2_v1` while Agent B uses `discrete25_native_v1`. Performance differences reflect the combined impact of policy learning and action space discretization, not algorithm capability alone."*
- **Hardware Disparity:**
  - Mandatory disclosure: *"Notice: Inference timings were measured on different hardware architectures. Task performance metrics remain comparable, but decision latencies cannot be directly compared."*

### 4.3 When Verdict is `INCOMPATIBLE` or `REJECTED_UNTRUSTED`
- Downstream tools must **refuse to display side-by-side benchmark delta tables** or leaderboard rankings.
- Visualizers must display a clear diagnostic warning detailing the specific failure point (e.g., *"Cannot compare Run A (TEST, N=60) with Run B (VALIDATION, N=96): dataset split mismatch"*).
- Visualizers must display a clear diagnostic warning detailing the specific failure point (e.g., *"Cannot compare Run A (TEST, N=60) with Run B (VALIDATION, N=96): dataset split mismatch"*).
