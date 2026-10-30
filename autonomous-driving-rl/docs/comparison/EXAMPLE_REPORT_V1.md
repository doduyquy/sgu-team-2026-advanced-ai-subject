# XeTuHanh Platform V1 — Synthetic Worked Comparison Examples V1

- **Document ID:** `REPORT-EXAMPLE-V1`
- **Version:** `1.0.0-draft`
- **Date:** 2026-10-30
- **Status:** `DRAFT / TEAM REVIEW REQUIRED` — Synthetic Worked Examples
- **Subproject Root:** `autonomous-driving-rl/`
**Notice:** **ALL NUMERICAL DATA IN THIS DOCUMENT IS SYNTHETIC.** It is constructed solely to test and demonstrate protocol decision rules, aggregation math, null handling, and reporting boundaries. It does not represent measured empirical performance of any real algorithm.

---

## 1. Structure of a Canonical Comparison Report

Every scientific comparison report produced on Platform V1 must structure its disclosures into eight required sections:
1. **Scope and Authority:** Split (`TEST` vs `VALIDATION`), manifest SHA-256, simulator commit, and integrity verification status.
2. **Replicate and Seed Taxonomy:** Number of independent training runs ($R$), inference stochasticity, and explicit seed mappings.
3. **Primary Benchmark Scorecards:** Exactly the five primary metrics, reported overall and per difficulty tier (`Easy`, `Medium`, `Hard`, `Extreme`).
4. **Primary Outcome and Safety Breakdown:** Mutually exclusive terminal outcome distribution (success, timeout, crashes, road departure).
5. **Route Progress and Diagnostic Breakdown:** Completion distributions, speeds, and cumulative RL return (diagnostic only).
6. **Decision Latency and Hardware Profile:** monotonic `agent.act()` timings and hardware execution environment.
7. **Run Compatibility Verdict:** Layer A/B/C checks and explicit compatibility label.
8. **Scientific Caveats and Unjustified Claims:** Explicit statements of what the evidence does *not* prove.

---

## 2. Synthetic Worked Examples

---

### Case 1: Zero Clean Successes (Null Time-to-Success Semantics)

**Context:** A naive random baseline (`agent_random_syn`) is evaluated on the canonical TEST suite ($N=60$). The vehicle terminates due to out-of-road or collisions on every single episode (60 safety failures out of 60 episodes).

#### Primary Benchmark Scorecard (SYNTHETIC)
| Difficulty Tier | Episodes ($N$) | Clean Success Rate | Safety Failure Rate | Mean Route Completion | Median Route Completion | Mean Time to Clean Success |
|---|---|---|---|---|---|---|
| **Easy** | 15 | 0.0% (0/15) | 100.0% (15/15) | 14.2% | 11.5% | `N/A — no clean-success samples` |
| **Medium** | 15 | 0.0% (0/15) | 100.0% (15/15) | 8.7% | 6.2% | `N/A — no clean-success samples` |
| **Hard** | 15 | 0.0% (0/15) | 100.0% (15/15) | 5.1% | 3.9% | `N/A — no clean-success samples` |
| **Extreme** | 15 | 0.0% (0/15) | 100.0% (15/15) | 3.4% | 2.1% | `N/A — no clean-success samples` |
| **Overall** | **60** | **0.0% (0/60)** | **100.0% (60/60)** | **7.85%** | **5.40%** | **`N/A — no clean-success samples`** |
| **Macro Summary**| **—** | **0.0%** | **100.0%** | **7.85%** | *[N/A per protocol]* | *[N/A per protocol]* |

#### Critical Protocol Validations
1. **Null Representation:** In `summary.json`, `"mean_time_to_clean_success_s": null`. In `episodes.csv`, column `time_to_clean_success_s` is an empty string for all 60 rows. In GUI and tables, it is rendered as `"N/A — no clean-success samples"`.
2. **Anti-Pattern Prevented:** Replacing null with `0.0 s` would make the worst agent appear to have instantaneous arrivals. Replacing null with `inf` or `9999.0` distorts mathematical calculations.

---

### Case 2: Unequal Success Counts and the Fallacy of "Faster" Policy

**Context:** Comparing two synthetic trained policies on canonical TEST ($N=60$):
- **Agent A (Conservative, High Success):** Solves 54 of 60 cases ($90.0\%$, $K=54$). It experiences 5 safety failures ($8.33\%$) and 1 step timeout ($1.67\%$). It drives carefully and reaches destination with mean time $38.50\,\text{s}$.
- **Agent B (Reckless, Low Success):** Solves only 6 of 60 cases ($10.0\%$, $K=6$). It experiences 53 safety failures ($88.33\%$) and 1 step timeout ($1.67\%$). The 6 solved cases were the simplest straight routes, completed in mean time $21.20\,\text{s}$. On the remaining 54 cases, Agent B crashed at high speed or timed out.

#### Primary Benchmark Comparison (SYNTHETIC)
| Metric Identifier | Preferred Direction | Agent A (Conservative) | Agent B (Reckless) | Comparison & Caveat |
|---|---|---|---|---|
| **Clean Success Rate** | Higher | **90.0% (54/60)** | 10.0% (6/60) | Agent A achieves $+80.0\text{ percentage points}$ higher success ($54/60$ vs $6/60$). |
| **Safety Failure Rate** | Lower | **8.33% (5/60)** | 88.33% (53/60) | Agent B exhibits massive collision rate ($+80.00\text{ percentage points}$ worse). |
| **Step Timeout Rate** | Diagnostic | 1.67% (1/60) | 1.67% (1/60) | Balanced timeout diagnostic. |
| **Mean Route Completion** | Higher | **95.2%** | 24.6% | Agent A reliably navigates entire routes. |
| **Median Route Completion**| Higher | **100.0%** | 12.8% | Agent B has bimodal collapse near starting line. |
| **Mean Time to Clean Success**| Lower *(conditional)*| 38.50 s ($K=54$) | **21.20 s ($K=6$)** | **NON-COMPARABLE OVERALL EFFICIENCY:** Descriptive times truthfully reflect respective subsets ($K=54$ vs $K=6$), but do not establish speed dominance. |

#### Critical Protocol Validations
1. **Conditional Denominator Disclosure:** Agent B's time is conditioned on $K=6$, representing only the easiest $10\%$ of scenarios. Agent A's time is conditioned on $K=54$, including complex curved routes and heavy traffic.
2. **Scientific Reporting Mandate:** It is scientifically invalid to assert that Agent B is "faster overall" or "more efficient". The report must explicitly state: *"Agent B's lower mean time-to-success is conditioned on completing only the simplest 10% of cases while failing the remaining 90%. Comparing conditional arrival times across disparate completion pools cannot establish overall operational efficiency."*

---

### Case 3: Safety vs Completion Trade-off (Prohibition of Mega-Scores)

**Context:** Demonstrating why geometric-mean success scores and scalar leaderboards are forbidden.
- **Agent X (Cautious Crawler):** Stalls or times out on challenging segments. `clean_success_rate = 50.0%` (30/60), `safety_failure_rate = 5.0%` (3/60), `timeout_rate = 45.0%` (27/60), `mean_final_route_completion = 72.0%`.
- **Agent Y (Aggressive Speeder):** Never yields. `clean_success_rate = 60.0%` (36/60), `safety_failure_rate = 38.33%` (23/60), `timeout_rate = 1.67%` (1/60), `mean_final_route_completion = 82.0%`.

#### Multi-Dimensional Scorecard (SYNTHETIC)
```
          [Clean Success Rate] (Higher is Better)
               Agent X: |====================> 50.0% (30/60)
               Agent Y: |========================> 60.0% (36/60) (+10.0 pp)

          [Safety Failure Rate] (Lower is Better - VITAL)
               Agent X: |==> 5.0% (3/60)
               Agent Y: |================> 38.33% (23/60) (+33.33 pp WORSE)

          [Timeout Rate] (Diagnostic)
               Agent X: |==================> 45.0% (27/60 - Passive hesitation)
               Agent Y: |> 1.67% (1/60)
```

#### Critical Protocol Validations
1. **Rejection of Naive Mega-Scores:** Under common naive composite formulations that weight completion and arrival (e.g., $0.5 \times \text{CSR} + 0.5 \times \text{Completion}$), Agent Y ($0.5 \times 0.60 + 0.5 \times 0.82 = 0.71$) would outscore Agent X ($0.5 \times 0.50 + 0.5 \times 0.72 = 0.61$). In autonomous driving, however, a policy causing physical collisions in nearly $40\%$ of scenarios is unviable.
2. **Scorecard Requirement:** Platform V1 explicitly prohibits scalar mega-scores precisely to force multi-objective inspection of both safety failure and clean success side-by-side across tiers.

---

### Case 4: Unmatched Case Manifests (Subsetting Anti-Pattern)

**Context:**
- Researcher A runs canonical TEST mode: 60 cases (Test Manifest SHA `0832c38e...`).
- Researcher B runs a custom script that executes only the 15 `Easy` cases of TEST.
- Researcher B attempts to compare their $80.0\%$ success rate against Researcher A's $65.0\%$ overall success rate.

#### Evaluator Audit Verdict
```
[COMPATIBILITY AUDIT ENGINE]
  Run A Manifest: test_case_manifest.csv (60 cases, SHA256: 0832c38e...)
  Run B Manifest: custom_easy_subset.csv  (15 cases, SHA256: e3b0c442...)
  Check Result:   FAIL — Case set mismatch (Layer B)
  VERDICT:        INCOMPATIBLE
```
#### Critical Protocol Validations
1. **Enforcement:** Comparing runs across different case sets is strictly forbidden. Subsetting is blocked by the launcher preflight in benchmark modes (`resolve_experiment_plan()` raises `ValueError` if tier or seed filters are supplied in `TEST` or `VALIDATION`).
2. **Action:** Run B must be discarded or evaluated on the full 60-case canonical manifest.

---

### Case 5: Infrastructure Fixture or Corrupted Integrity Run

**Context:**
- **Run A:** A valid candidate evaluated on TEST, `run_state.json` status is `COMPLETE`, `_evaluate_integrity()` yields `IntegrityDisplayStatus.VERIFIED`.
- **Run B:** An engineer tested `fixture_constant_continuous` in TEST mode using an unverified dirty worktree override, or manually edited `summary.json` to change the success rate.

#### Evaluator Audit Verdict
```
[INTEGRITY AUDIT ENGINE]
  Run B Target: runs/run_fixture_test_001/
  - Artifact run_integrity.json verification:
      episodes.csv computed SHA-256 mismatch!
      Recorded in run_integrity.json: a1b2c3...
      Observed on disk:               f9e8d7...
      Status:                         IntegrityDisplayStatus.FAILED
  - Canonical provenance check:
      canonical_run == False (dirty_override == True, unverified fixture agent)
  Check Result: FAIL — Cryptographic mismatch and non-canonical origin (Layer A)
  VERDICT:      REJECTED_UNTRUSTED
```
#### Critical Protocol Validations
- **Zero Scientific Standing:** Run B is immediately rejected. It cannot appear in any comparative report, plot, or table.

---

### Case 6: Learned Agent Replicates ($R=3$) vs Individual Episodes ($N=180$)

**Context:** Stage 5 RL policy (`ppo_stage5`) is trained with 3 independent training root seeds: `101`, `202`, and `303`. Each trained model checkpoint is evaluated once across the 60 canonical TEST cases.

#### Replicate Observations (SYNTHETIC)
| Training Replicate | Root Seed | Clean Success ($K/60$) | Clean Success Rate | Safety Failure ($S/60$) | Safety Failure Rate | Step Timeouts | Mean Route Completion | Mean Time to Success |
|---|---|---|---|---|---|---|---|---|
| Model Replicate 1 | `101` | 45 / 60 | 75.00% | 11 / 60 | 18.33% | 4 / 60 | 88.4% | 34.20 s |
| Model Replicate 2 | `202` | 42 / 60 | 70.00% | 14 / 60 | 23.33% | 4 / 60 | 84.1% | 36.10 s |
| Model Replicate 3 | `303` | 48 / 60 | 80.00% | 9 / 60 | 15.00% | 3 / 60 | 91.2% | 33.50 s |

#### Statistical Aggregation Math (Computed from Exact Counts)
**Clean Success Rate:**
- Replicate values: $[0.75, 0.70, 0.80]$
- Sample Mean: $\bar{m}_{\text{CSR}} = \frac{0.75 + 0.70 + 0.80}{3} = 75.00\%$
- Sample Standard Deviation ($R=3$):
  $$s_{\text{CSR}} = \sqrt{\frac{(0.75 - 0.75)^2 + (0.70 - 0.75)^2 + (0.80 - 0.75)^2}{3 - 1}} = \sqrt{\frac{0.0050}{2}} = 0.0500 = 5.00\text{ percentage points}$$

**Safety Failure Rate:**
- Underlying counts: $11/60$, $14/60$, $9/60$ (Total = $34/180$)
- Sample Mean: $\bar{m}_{\text{SFR}} = \frac{34}{180} \approx 18.89\%$
- Sample Standard Deviation ($R=3$):
  $$s_{\text{SFR}} = \sqrt{\frac{(11/60 - 34/180)^2 + (14/60 - 34/180)^2 + (9/60 - 34/180)^2}{3 - 1}} = \sqrt{\frac{114 / 32400}{2}} \approx 0.04194 = 4.19\text{ percentage points}$$

#### Primary Benchmark Reporting Format
$$\text{Clean Success Rate: } \mathbf{75.00\% \pm 5.00\text{ percentage points}} \quad (R=3 \text{ independent models; counts: } [45/60, 42/60, 48/60])$$
$$\text{Safety Failure Rate: } \mathbf{18.89\% \pm 4.19\text{ percentage points}} \quad (R=3 \text{ independent models; counts: } [11/60, 14/60, 9/60])$$

#### Critical Protocol Validations
1. **Statistical Independence:** The sample size for statistical inference across models is $R=3$, **NOT** $N = 3 \times 60 = 180$.
2. **Prohibited Fallacy:** An author must never pool all 180 episodes into a single spreadsheet and report an inflated $t$-test or significance claim based on $N=180$. Intra-policy episodes from the same model checkpoint share parameter weights and cannot be treated as independent training trials.

---

### Case 7: Action Adapter Mismatch (`continuous_box2_v1` vs `discrete25_native_v1`)

**Context:**
- **Agent A (RL):** Uses action adapter `continuous_box2_v1` (continuous steering and acceleration in $[-1.0, 1.0]$).
- **Agent B (Heuristic / Search):** Uses action adapter `discrete25_native_v1` (native 5-steering by 5-throttle action grid).
- Both agents evaluate identical canonical TEST cases ($N=60$) with identical `STATE_DECISION_V1` observations.

#### Evaluator Audit Verdict
```
[ACTION ADAPTER AUDIT]
  Agent A Adapter: continuous_box2_v1
  Agent B Adapter: discrete25_native_v1
  Check Result:    Disparate control action spaces (Layer B)
  VERDICT:         QUALIFIED_COMPARISON
```

#### Mandatory Disclosure Text in Final Report
> *"Comparative Status: QUALIFIED COMPARISON (System-Level Only)."*
>
> *"Notice: Agent A and Agent B operate in different action spaces (`continuous_box2_v1` vs `discrete25_native_v1`). While both control the identical vehicle physics and evaluate the identical 60 test cases, the 25-action discretization restricts steering finesse and throttle modulation. Consequently, performance differences reflect the integrated combination of policy intelligence AND action space discretization. This experiment does NOT demonstrate pure algorithm superiority in isolation."*

---

### Case 8: Deterministic Non-Learned Agent (Stage 1 Rule-Based)

**Context:** A deterministic rule-based controller (`rule_based_stage1`) evaluates the canonical TEST suite ($N=60$). The controller has no learned parameters, no neural network, and no stochastic sampling (`agent_seed = None`).

#### Correct Reporting Procedure
1. **Single Evaluation Run:** The agent evaluates the 60 canonical TEST cases exactly once.
2. **Deterministic Reproducibility:** Because the code and environment are deterministic, rerunning the suite with agent seeds `101`, `202`, and `303` would yield bit-identical trajectories.
3. **Prohibited Practice:** Fabricating 3 identical runs and reporting $\bar{m} \pm 0.0\%$ to mimic RL replicate tables is scientifically dishonest.
4. **Mandatory Reporting Format:**
   $$\text{Clean Success Rate: } \mathbf{68.3\%} \ (41/60 \text{ cases; single deterministic evaluation run})$$
   $$\text{Safety Failure Rate: } \mathbf{11.7\%} \ (7/60 \text{ cases})$$
   $$\text{Mean Route Completion: } \mathbf{84.2\%}$$
