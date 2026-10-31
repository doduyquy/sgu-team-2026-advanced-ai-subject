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

To prevent contradictory single-state claims when driving performance and decision latency dimensions diverge, the proposed comparison specification defines a **Structured Pairwise Verdict Model**:
- `overall_verdict`: Umbrella status for the pair (`REJECTED_UNTRUSTED`, `INCOMPATIBLE`, `QUALIFIED_COMPARISON`, `STRICTLY_COMPARABLE`).
- `task_comparability`: Scoped status for the five primary driving benchmark metrics (`clean_success_rate`, `safety_failure_rate`, route completion, etc.).
- `latency_comparability`: Scoped status for agent decision latency (`timing.csv`).
- `qualification_flags`: Machine-readable flags (e.g., `ACTION_ADAPTER_DISPARITY`, `HARDWARE_LATENCY_UNVERIFIED`).

```
+----------------------------------------------------------------------------------------------------+
|                                    VERDICT STATE DEFINITIONS                                       |
+----------------------------------------------------------------------------------------------------+
| 1. REJECTED_UNTRUSTED     | Hard block. At least one run is corrupted, incomplete, unverified, or |
|                           | executed in a non-canonical/fixture state (Fails Layer A).            |
+---------------------------+------------------------------------------------------------------------+
| 2. INCOMPATIBLE           | Block. Both runs pass Layer A, but evaluate fundamentally different    |
|                           | splits (TEST vs VAL), benchmark versions, or observation profiles.     |
+---------------------------+------------------------------------------------------------------------+
| 3. QUALIFIED_COMPARISON   | Allowed with mandatory caveats. Both runs pass Layer A on same split,  |
|                           | but differ in action adapter or timing hardware. Valid as end-to-end   |
|                           | system comparison, NOT as an isolated algorithmic proof.               |
+---------------------------+------------------------------------------------------------------------+
| 4. STRICTLY_COMPARABLE    | Apples-to-apples benchmark comparison. Exact match across cases, seeds,|
|                           | input profile, action adapter, and verified platform contracts.       |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Systematic Comparability Matrix

| Row ID | Dimension / Scenario | Condition in Run A | Condition in Run B | Structured Verdict | Scientific Rationale & Enforcement Rules |
|---|---|---|---|---|---|
| **M01** | **Artifact Integrity** | 6 core digests + identity checks match (`VERIFIED`) | One or more hash mismatches (`FAILED`) | `overall: REJECTED_UNTRUSTED`<br>`task: REJECTED_UNTRUSTED`<br>`latency: REJECTED_UNTRUSTED` | Tampered, modified, or corrupted local files. Hard block; zero scientific authority. |
| **M02** | **Lifecycle Completion** | `status == COMPLETE` | `status == RUNNING`, `INTERRUPTED`, `FAILED` | `overall: REJECTED_UNTRUSTED`<br>`task: REJECTED_UNTRUSTED`<br>`latency: REJECTED_UNTRUSTED` | Incomplete run (`NOT_FINAL`). Partial metrics cannot be compared against completed suites. |
| **M03** | **Integrity Declaration** | `run_integrity.json` exists & verifies | `run_integrity.json` missing (`UNVERIFIED`) | `overall: REJECTED_UNTRUSTED`<br>`task: REJECTED_UNTRUSTED`<br>`latency: REJECTED_UNTRUSTED` | Run lacks cryptographic finalization. Must not be used for authoritative benchmarking. |
| **M04** | **Clean Worktree** | `canonical_run == True` (`dirty == False`) | `dirty_override == True` (`dirty == True`) | `overall: REJECTED_UNTRUSTED`<br>`task: REJECTED_UNTRUSTED`<br>`latency: REJECTED_UNTRUSTED` | Uncommitted code changes introduce unverified side-effects. Blocked from canonical comparisons. |
| **M05** | **Agent Eligibility** | Research Agent (e.g., `ppo_stage5`) | Infrastructure Fixture (e.g., `fixture_constant_continuous`) | `overall: REJECTED_UNTRUSTED`<br>`task: REJECTED_UNTRUSTED`<br>`latency: REJECTED_UNTRUSTED` | Fixtures are sanity-check stubs, not valid scientific baselines. Preflight blocks them from canonical suites. |
| **M06** | **Dataset Split** | Valid Canonical `TEST` ($N=60$) | Valid Canonical `VALIDATION` ($N=96$) | `overall: INCOMPATIBLE`<br>`task: INCOMPATIBLE`<br>`latency: INCOMPATIBLE` | Both pass Layer A, but evaluate different geometry distributions and case counts. Cross-split comparison is mathematically invalid. |
| **M07** | **Launcher Mode** | Mode `TEST` | Mode `SANDBOX` ($N=1$) or `AUDIT` | `overall: REJECTED_UNTRUSTED`<br>`task: REJECTED_UNTRUSTED`<br>`latency: REJECTED_UNTRUSTED` | Sandbox and audit modes are non-canonical (`canonical_run == False`). Presenting them as canonical benchmark evidence fails Layer A immediately. |
| **M08** | **Episode Count & Coverage** | Complete expected count ($60/60$) | Missing episodes ($45/60$) or duplicate IDs | `overall: REJECTED_UNTRUSTED`<br>`task: REJECTED_UNTRUSTED`<br>`latency: REJECTED_UNTRUSTED` | Incomplete execution or duplicated episodes fail Layer A run integrity. |
| **M09** | **Subsetting vs Version Parity**| Full Canonical TEST Manifest (`0832c38e...`) | Filtered / Partial Subset (e.g., 15 Easy cases) | `overall: REJECTED_UNTRUSTED`<br>`task: REJECTED_UNTRUSTED`<br>`latency: REJECTED_UNTRUSTED` | Subsetting a canonical benchmark suite is prohibited and fails Layer A. *(Note: Comparing two complete, individually valid but separately versioned canonical manifests reaches Layer B and yields `INCOMPATIBLE`)*. |
| **M10** | **Simulator Pin** | MetaDrive `0.4.3` @ `85e5dadc` (`VERIFIED`) | Unverified commit / different MetaDrive (`MISMATCHED`) | `overall: REJECTED_UNTRUSTED`<br>`task: REJECTED_UNTRUSTED`<br>`latency: REJECTED_UNTRUSTED` | Unverified simulator provenance fails Layer A canonical gating. |
| **M11** | **Information Rights** | Input Profile `STATE_DECISION_V1` | Input Profile `CORE_ONLY_V1` (Masked Context) | `overall: INCOMPATIBLE`<br>`task: INCOMPATIBLE`<br>`latency: INCOMPATIBLE` | Both pass Layer A, but observation spaces differ. Masked context represents an ablation study, not a direct algorithmic competition. |
| **M12** | **Action Adapter** | `continuous_box2_v1` (Steering $[-1, 1]$, Accel $[-1, 1]$) | `discrete25_native_v1` (5x5 Steering-Throttle grid) | `overall: QUALIFIED_COMPARISON`<br>`task: QUALIFIED_COMPARISON`<br>`latency: STRICTLY_COMPARABLE` (if same HW)<br>`flags: [ACTION_ADAPTER_DISPARITY]` | Valid as an **end-to-end system comparison**. Cannot be claimed as pure algorithm superiority because discretization restricts vehicle dynamics. |
| **M13** | **Action Adapter** | `discrete25_native_v1` | `discrete9_lowbranch_v1` | `overall: QUALIFIED_COMPARISON`<br>`task: QUALIFIED_COMPARISON`<br>`latency: STRICTLY_COMPARABLE` (if same HW)<br>`flags: [ACTION_ADAPTER_DISPARITY]` | Control resolution disparity. Must disclose adapter details prominently in report. |
| **M14** | **Execution Hardware** | Workstation A (Intel i9 + RTX 4090) | Laptop B (AMD Ryzen 7 + Integrated GPU) | `overall: QUALIFIED_COMPARISON`<br>`task: STRICTLY_COMPARABLE (Conditional on parity assumptions)`<br>`latency: QUALIFIED_COMPARISON`<br>`flags: [HARDWARE_LATENCY_UNVERIFIED]` | Different execution hardware qualifies decision latency (`timing.csv`). Task performance metrics remain comparable under declared environment parity assumptions (matching simulator pin, seed sequence, and dependencies), but cross-device numerical parity is not guaranteed without empirical verification. |
| **M15** | **Sync Policy** | Latency Policy `FRAMEWORK_SYNCHRONIZED` | Latency Policy `NONE` | `overall: QUALIFIED_COMPARISON`<br>`task: STRICTLY_COMPARABLE`<br>`latency: QUALIFIED_COMPARISON`<br>`flags: [HARDWARE_LATENCY_UNVERIFIED]` | Asynchronous GPU kernel launches under `NONE` produce misleadingly optimistic `act()` latencies. Latency comparison is qualified; task driving decisions remain comparable if the action executed by the simulator is identical. |
| **M16** | **Identical Setup** | `STATE_DECISION_V1`, `continuous_box2_v1`, Clean, TEST ($N=60$), Verified | `STATE_DECISION_V1`, `continuous_box2_v1`, Clean, TEST ($N=60$), Verified | `overall: STRICTLY_COMPARABLE`<br>`task: STRICTLY_COMPARABLE`<br>`latency: STRICTLY_COMPARABLE`<br>`flags: []` | True apples-to-apples algorithmic comparison across all five primary benchmark metrics and decision latency. |

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
   +--> Step 3: Check Control Space and Hardware Context (Structured Verdict Generation)
   |    - Are action adapter IDs identical? (e.g., both continuous_box2_v1)
   |    - Are latency sync policies and hardware environments identical?
   |    IF ADAPTER DIFFERS AND HARDWARE DIFFERS:
   |        Output: overall: QUALIFIED_COMPARISON | task: QUALIFIED_COMPARISON | latency: QUALIFIED_COMPARISON
   |                flags: [ACTION_ADAPTER_DISPARITY, HARDWARE_LATENCY_UNVERIFIED]
   |    IF ADAPTER DIFFERS AND HARDWARE IDENTICAL:
   |        Output: overall: QUALIFIED_COMPARISON | task: QUALIFIED_COMPARISON | latency: STRICTLY_COMPARABLE
   |                flags: [ACTION_ADAPTER_DISPARITY]
   |    IF ADAPTER IDENTICAL AND HARDWARE DIFFERS:
   |        Output: overall: QUALIFIED_COMPARISON | task: STRICTLY_COMPARABLE | latency: QUALIFIED_COMPARISON
   |                flags: [HARDWARE_LATENCY_UNVERIFIED]
   |    IF ALL IDENTICAL:
   |        Output: overall: STRICTLY_COMPARABLE | task: STRICTLY_COMPARABLE | latency: STRICTLY_COMPARABLE
   |                flags: []
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
  - Mandatory disclosure: *"Notice: Inference timings were measured on different hardware architectures. While task performance metrics are compared under matching simulator pins and canonical cases, cross-device numerical parity has not been empirically verified. Latency metrics reflect different compute environments and cannot be directly compared."*

### 4.3 When Verdict is `INCOMPATIBLE` or `REJECTED_UNTRUSTED`
- Downstream tools must **refuse to display side-by-side benchmark delta tables** or leaderboard rankings.
- Visualizers must display a clear diagnostic warning detailing the specific failure point (e.g., *"Cannot compare Run A (TEST, N=60) with Run B (VALIDATION, N=96): dataset split mismatch"*).
