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

**Context:** A naive random baseline (`agent_random_syn`) is evaluated on the canonical TEST suite ($N=60$). The vehicle terminates due to out-of-road or collisions on every single episode.

#### Primary Benchmark Scorecard (SYNTHETIC)
| Difficulty Tier | Episodes ($N$) | Clean Success Rate | Safety Failure Rate | Mean Route Completion | Median Route Completion | Mean Time to Clean Success |
|---|---|---|---|---|---|---|
| **Easy** | 15 | 0.0% (0/15) | 93.3% (14/15) | 14.2% | 11.5% | `N/A — no clean-success samples` |
| **Medium** | 15 | 0.0% (0/15) | 100.0% (15/15) | 8.7% | 6.2% | `N/A — no clean-success samples` |
| **Hard** | 15 | 0.0% (0/15) | 100.0% (15/15) | 5.1% | 3.9% | `N/A — no clean-success samples` |
| **Extreme** | 15 | 0.0% (0/15) | 100.0% (15/15) | 3.4% | 2.1% | `N/A — no clean-success samples` |
| **Overall** | **60** | **0.0% (0/60)** | **98.3% (59/60)** | **7.85%** | **5.40%** | **`N/A — no clean-success samples`** |
| **Macro Summary**| **—** | **0.0%** | **98.3%** | **7.85%** | *[N/A per protocol]* | *[N/A per protocol]* |

#### Critical Protocol Validations
1. **Null Representation:** In `summary.json`, `"mean_time_to_clean_success_s": null`. In `episodes.csv`, column `time_to_clean_success_s` is an empty string for all 60 rows. In GUI and tables, it is rendered as `"N/A — no clean-success samples"`.
2. **Anti-Pattern Prevented:** Replacing null with `0.0 s` would make the worst agent appear to have instantaneous arrivals. Replacing null with `inf` or `9999.0` distorts mathematical calculations.

---

### Case 2: Unequal Success Counts and the Fallacy of "Faster" Policy

**Context:** Comparing two synthetic trained policies on canonical TEST ($N=60$):
- **Agent A (Conservative, High Success):** Solves 54 of 60 cases ($90.0\%$). It drives carefully and reaches destination with mean time $38.5\,\text{s}$.
- **Agent B (Reckless, Low Success):** Solves only 6 of 60 cases ($10.0\%$). The 6 solved cases were the simplest straight routes, completed in mean time $21.2\,\text{s}$. On the remaining 54 cases, Agent B crashed at high speed within $5\,\text{s}$.

#### Primary Benchmark Comparison (SYNTHETIC)
| Metric Identifier | Preferred Direction | Agent A (Conservative) | Agent B (Reckless) | Comparison & Caveat |
|---|---|---|---|---|
| **Clean Success Rate** | Higher | **90.0% (54/60)** | 10.0% (6/60) | Agent A achieves $+80.0\%$ higher success ($p < 0.001$). |
| **Safety Failure Rate** | Lower | **8.3% (5/60)** | 88.3% (53/60) | Agent B exhibits massive collision rate. |
| **Mean Route Completion** | Higher | **95.2%** | 24.6% | Agent A reliably navigates entire routes. |
| **Median Route Completion**| Higher | **100.0%** | 12.8% | Agent B has bimodal collapse near starting line. |
| **Mean Time to Clean Success**| Lower *(conditional)*| 38.50 s ($K=54$) | **21.20 s ($K=6$)** | **NON-COMPARABLE DOMINANCE:** Agent B cannot be claimed as "faster". |

#### Critical Protocol Validations
1. **Conditional Denominator Disclosure:** Agent B's time is conditioned on $K=6$, representing only the easiest $10\%$ of scenarios. Agent A's time is conditioned on $K=54$, including complex curved routes and heavy traffic.
2. **Scientific Reporting Mandate:** It is scientifically fraudulent to state: *"Agent B drives $45\%$ faster to the goal than Agent A."* The report must explicitly state: *"Agent B's lower time-to-success is an artifact of severe conditioning bias, as it only completed the shortest, simplest 10% of cases and failed the remaining 90%."*

---

### Case 3: Safety vs Completion Trade-off (Prohibition of Mega-Scores)

**Context:** Demonstrating why geometric-mean success scores and scalar leaderboards are forbidden.
- **Agent X (Cautious Crawler):** Stalls or times out on challenging segments. `clean_success_rate = 50.0%`, `safety_failure_rate = 5.0%`, `timeout_rate = 45.0%`, `mean_final_route_completion = 72.0%`.
- **Agent Y (Aggressive Speeder):** Never yields. `clean_success_rate = 60.0%`, `safety_failure_rate = 38.3%`, `timeout_rate = 1.7%`, `mean_final_route_completion = 82.0%`.

#### Multi-Dimensional Scorecard (SYNTHETIC)
```
          [Clean Success Rate] (Higher is Better)
               Agent X: |====================> 50.0%
               Agent Y: |========================> 60.0% (+10.0%)

          [Safety Failure Rate] (Lower is Better - VITAL)
               Agent X: |==> 5.0%
               Agent Y: |================> 38.3% (+33.3% WORSE)

          [Timeout Rate] (Diagnostic)
               Agent X: |==================> 45.0% (Passive hesitation)
               Agent Y: |> 1.7%
```

#### Critical Protocol Validations
1. **Rejection of Scalar Ranking:** A scalar formula weighting completion and success would rank Agent Y ahead of Agent X. However, in autonomous driving, **safety is a non-negotiable constraint**. A policy causing physical collisions in nearly $40\%$ of scenarios is fundamentally unviable.
2. **Scorecard Requirement:** Platform V1 mandates side-by-side presentation of `clean_success_rate` and `safety_failure_rate` across tiers so that reckless behavior is immediately transparent.

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
- **Run A:** A valid candidate evaluated on TEST, `run_state.json` is `COMPLETE`, `run_integrity.json` status is `VERIFIED`.
- **Run B:** An engineer tested `fixture_constant_continuous` in TEST mode using an unverified dirty worktree override, or manually edited `summary.json` to change the success rate.

#### Evaluator Audit Verdict
```
[INTEGRITY AUDIT ENGINE]
  Run B Target: runs/run_fixture_test_001/
  - Artifact run_integrity.json check:
      episodes.csv SHA-256 mismatch!
      Recorded: a1b2c3...  vs  Observed: f9e8d7...
  - Canonical flag check:
      canonical_run == False (dirty_override == True, fixture agent)
  Check Result: FAIL — Cryptographic tamper detected (Layer A)
  VERDICT:      REJECTED_UNTRUSTED
```
#### Critical Protocol Validations
- **Zero Scientific Standing:** Run B is immediately rejected. It cannot appear in any comparative report, plot, or table.

---

### Case 6: Learned Agent Replicates ($R=3$) vs Individual Episodes ($N=180$)

**Context:** Stage 5 RL policy (`ppo_stage5`) is trained with 3 independent training root seeds: `101`, `202`, and `303`. Each trained model checkpoint is evaluated once across the 60 canonical TEST cases.

#### Replicate Observations (SYNTHETIC)
| Training Replicate | Root Seed | Clean Success ($K/60$) | Clean Success Rate | Safety Failure Rate | Mean Route Completion | Mean Time to Success |
|---|---|---|---|---|---|---|
| Model Replicate 1 | `101` | 45 / 60 | 75.0% | 18.3% | 88.4% | 34.20 s |
| Model Replicate 2 | `202` | 42 / 60 | 70.0% | 23.3% | 84.1% | 36.10 s |
| Model Replicate 3 | `303` | 48 / 60 | 80.0% | 15.0% | 91.2% | 33.50 s |

#### Statistical Aggregation Math
Sample Mean:
$$\bar{m}_{\text{CSR}} = \frac{75.0\% + 70.0\% + 80.0\%}{3} = 75.0\%$$
Sample Standard Deviation ($R=3$):
$$s_{\text{CSR}} = \sqrt{\frac{(0.75 - 0.75)^2 + (0.70 - 0.75)^2 + (0.80 - 0.75)^2}{3 - 1}} = \sqrt{\frac{0.0025 + 0.0025}{2}} = \sqrt{0.0025} = 0.050 = 5.0\%$$

#### Primary Benchmark Reporting Format
$$\text{Clean Success Rate: } \mathbf{75.0\% \pm 5.0\%} \quad (R=3 \text{ independent training seeds; individual: } [75.0\%, 70.0\%, 80.0\%])$$
$$\text{Safety Failure Rate: } \mathbf{18.87\% \pm 4.19\%} \quad (R=3 \text{ independent training seeds; individual: } [18.3\%, 23.3\%, 15.0\%])$$

#### Critical Protocol Validations
1. **Statistical Independence:** The sample size for statistical inference across models is $R=3$, **NOT** $N = 3 \times 60 = 180$.
2. **Prohibited Fallacy:** An author must never pool all 180 episodes into a single spreadsheet and report: *"Student's t-test with N=180 shows p=0.004"*. Such pooling treats intra-policy episodes as independent training trials, grossly inflating degrees of freedom and manufacturing spurious statistical significance.

---

### Case 7: Action Adapter Mismatch (Continuous Box2 vs Discrete 25)

**Context:**
- **Agent A (RL):** Uses action adapter `CONTINUOUS_BOX2` (continuous steering and acceleration $[-1.0, 1.0]$).
- **Agent B (Heuristic / Search):** Uses action adapter `DISCRETE_25` (discretized 5-steering by 5-throttle action grid).
- Both agents evaluate identical canonical TEST cases ($N=60$) with identical `CORE_V1` observations.

#### Evaluator Audit Verdict
```
[ACTION ADAPTER AUDIT]
  Agent A Adapter: CONTINUOUS_BOX2
  Agent B Adapter: DISCRETE_25
  Check Result:    Disparate control action spaces (Layer B)
  VERDICT:         QUALIFIED_COMPARISON
```

#### Mandatory Disclosure Text in Final Report
> *"Comparative Status: QUALIFIED COMPARISON (System-Level Only)."*
>
> *"Notice: Agent A and Agent B operate in different action spaces (`CONTINUOUS_BOX2` vs `DISCRETE_25`). While both control the identical vehicle physics and evaluate the identical 60 test cases, the 25-action discretization restricts steering finesse and throttle modulation. Consequently, performance differences reflect the integrated combination of policy intelligence AND action space discretization. This experiment does NOT demonstrate pure algorithm superiority in isolation."*

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
