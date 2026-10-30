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
| **M01** | **Artifact Integrity** | `integrity == VERIFIED` | `integrity == FAILED` (hash mismatch) | `REJECTED_UNTRUSTED` | Tampered, modified, or corrupted local files. Hard block; zero scientific authority. |
| **M02** | **Lifecycle Completion** | `status == COMPLETE` | `status == RUNNING` or `INTERRUPTED` | `REJECTED_UNTRUSTED` | Incomplete run (`NOT_FINAL`). Partial metrics cannot be compared against completed suites. |
| **M03** | **Integrity Declaration** | `integrity == VERIFIED` | `integrity == UNVERIFIED` (missing hash file) | `REJECTED_UNTRUSTED` | Run lacks cryptographic finalization. Must not be used for authoritative benchmarking. |
| **M04** | **Clean Worktree** | `canonical_run == True` (`dirty == False`) | `dirty_override == True` (`dirty == True`) | `REJECTED_UNTRUSTED` | Uncommitted code changes introduce unverified side-effects. Blocked from canonical comparisons. |
| **M05** | **Agent Eligibility** | Research Agent (e.g., `ppo_stage5`) | Infrastructure Fixture (e.g., `fixture_constant_continuous`) | `REJECTED_UNTRUSTED` | Fixtures are sanity-check stubs, not valid scientific baselines. Preflight blocks them from canonical suites. |
| **M06** | **Dataset Split** | `split == "TEST"` ($N=60$) | `split == "VALIDATION"` ($N=96$) | `INCOMPATIBLE` | Different geometry distributions and case counts. Cross-split comparison is mathematically invalid. |
| **M07** | **Launcher Mode** | Mode `TEST` | Mode `SANDBOX` ($N=1$) or `AUDIT` | `INCOMPATIBLE` | Sandbox evaluates a single development case; audit evaluates contracts. Neither is a benchmark scorecard. |
| **M08** | **Episode Count** | $60 / 60$ episodes completed | $45 / 60$ episodes completed (early exit) | `REJECTED_UNTRUSTED` | Missing episodes bias aggregate rates. Expected episode count must match recorded count exactly. |
| **M09** | **Manifest & Hash** | Test Manifest `0832c38e...` | Custom Manifest or Filtered Subset | `INCOMPATIBLE` | Benchmark gaming prevention: filtering subsets (e.g. Easy-only) is prohibited in canonical modes. |
| **M10** | **Simulator Pin** | MetaDrive `0.4.3` @ `85e5dadc` | Unverified commit / different MetaDrive | `INCOMPATIBLE` | Physics, collision detection, or sensor rendering differences invalidate behavioral comparisons. |
| **M11** | **Information Rights** | Input Profile `CORE_V1` | Input Profile `FULL_V1` (Oracle Traffic) | `INCOMPATIBLE` | Different observation spaces. Full oracle access confers an unfair information advantage; reportable only as an ablation study. |
| **M12** | **Action Adapter** | `CONTINUOUS_BOX2` (Steering $[-1, 1]$, Accel $[-1, 1]$) | `DISCRETE_25` (5x5 Steering-Throttle grid) | `QUALIFIED_COMPARISON` | Valid as an **end-to-end system comparison**. Cannot be claimed as pure algorithm superiority because discretization restricts vehicle dynamics. |
| **M13** | **Action Adapter** | `DISCRETE_25` | `DISCRETE_9` | `QUALIFIED_COMPARISON` | Control resolution disparity. Must disclose adapter details prominently in report. |
| **M14** | **Execution Hardware** | Workstation A (Intel i9 + RTX 4090) | Laptop B (AMD Ryzen 7 + Integrated GPU) | `QUALIFIED_COMPARISON` *(Timing Only)* | Task performance (success/safety) remains `STRICTLY_COMPARABLE` if simulation is headless & deterministic; **timing comparison is qualified/blocked**. |
| **M15** | **Sync Policy** | Latency Policy `FRAMEWORK_SYNCHRONIZED` | Latency Policy `NONE` | `QUALIFIED_COMPARISON` *(Timing Only)* | Asynchronous GPU kernel launches under `NONE` produce misleadingly optimistic `act()` latencies. |
| **M16** | **Identical Setup** | `CORE_V1`, `CONTINUOUS_BOX2`, Clean, TEST ($N=60$), Verified | `CORE_V1`, `CONTINUOUS_BOX2`, Clean, TEST ($N=60$), Verified | `STRICTLY_COMPARABLE` | True apples-to-apples algorithmic comparison across the five primary benchmark metrics. |

---

## 3. Detailed Decision Flow for Comparison Engines

When two runs $R_A$ and $R_B$ are selected for comparison (e.g., in GUI V2 or a comparison script), the evaluator must execute the following automated checklist:

```
[START EVALUATION]
   |
   +--> Step 1: Check Layer A (Trustworthiness) for BOTH runs
   |    - Does run_state.json exist and status == COMPLETE?
   |    - Does run_integrity.json exist and verify 100% of artifact hashes?
   |    - Is canonical_run == True?
   |    - Is recorded_episode_count == expected_episode_count?
   |    - Are technical failures == 0?
   |    IF NO  ===> Output: REJECTED_UNTRUSTED (Explain which run failed Layer A)
   |    IF YES ===> Proceed to Step 2
   |
   +--> Step 2: Check Layer B (Scientific Consistency)
   |    - Are splits identical? (e.g., both TEST or both VALIDATION)
   |    - Are manifest hashes identical?
   |    - Are simulator commit pins identical?
   |    - Are input profile IDs identical?
   |    IF NO  ===> Output: INCOMPATIBLE (Block comparison, explain mismatch)
   |    IF YES ===> Proceed to Step 3
   |
   +--> Step 3: Check Control Space and Hardware Context
   |    - Are action adapter IDs identical?
   |    - Are latency sync policies and hardware environments identical?
   |    IF NO  ===> Output: QUALIFIED_COMPARISON (Allow comparison with mandatory system-level caveats)
   |    IF YES ===> Output: STRICTLY_COMPARABLE (Apples-to-apples algorithmic benchmark)
```

---

## 4. Mandatory Reporting Disclosures by Verdict

### 4.1 When Verdict is `STRICTLY_COMPARABLE`
- Report the 5 primary metrics side-by-side (overall and across all 4 tiers).
- For learned agents, report mean $\pm$ sample standard deviation across independent training replicates ($R=3$).
- Show paired per-case difference histograms $(\Delta = B - A)$ across the 60 canonical cases.

### 4.2 When Verdict is `QUALIFIED_COMPARISON`
- **Action Adapter Disparity:**
  - Mandatory disclosure: *"Notice: Agent A uses action adapter `CONTINUOUS_BOX2` while Agent B uses `DISCRETE_25`. Performance differences reflect the combined impact of policy learning and action space discretization, not algorithm capability alone."*
- **Hardware Disparity:**
  - Mandatory disclosure: *"Notice: Inference timings were measured on different hardware architectures. Latency metrics cannot be directly compared."*

### 4.3 When Verdict is `INCOMPATIBLE` or `REJECTED_UNTRUSTED`
- Downstream tools must **refuse to display side-by-side benchmark delta tables** or leaderboard rankings.
- Visualizers must display a clear diagnostic warning detailing the specific failure point (e.g., *"Cannot compare Run A (TEST, N=60) with Run B (VALIDATION, N=96): dataset split mismatch"*).
