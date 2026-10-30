# XeTuHanh Platform V1 — Scientific Comparison Specification V1

- **Document ID:** `SPEC-COMP-V1`
- **Version:** `1.0.0-draft`
- **Date:** 2026-10-30
- **Status:** `DRAFT / TEAM REVIEW REQUIRED` — Not an approved scientific lock
- **Subproject Root:** `autonomous-driving-rl/`
- **Baseline Git Commit:** `dcd653306a0e072a2c6bee66eaeadf388101d679`
- **Target Audience:** SGU Autonomous Driving Research Team (Q, Khoa, Phúc, Trí), Stage 5 RL Engineers, GUI V2 Implementers

---

## 1. Purpose, Scope, and Non-Objectives

### 1.1 Purpose
This specification establishes the normative rules, mathematical formulas, and scientific boundaries for comparing autonomous driving agents across Stages 0 through 7 on the Platform V1 benchmark. It standardizes how local execution artifacts are parsed, how primary benchmark metrics are interpreted, which experimental runs may be contrasted, and how statistical uncertainty must be reported before large-scale experiments commence.

### 1.2 Scope
This specification governs:
- Interpretation of aggregate benchmark metrics from `summary.json`.
- Per-episode telemetry analysis from `episodes.csv` (31 standardized columns).
- Agent decision latency accounting from `timing.csv`.
- Cryptographic provenance and trust verification from `run_manifest.json`, `run_state.json`, and `run_integrity.json`.
- Three-layer compatibility checks (Trustworthiness, Scientific Comparability, Statistical Interpretation).
- Statistical policies for independent training replicates ($R=3$) versus deterministic inference.

### 1.3 Non-Objectives and Anti-Patterns
- **No benchmark re-generation:** Does not alter frozen MapSuite V1 geometries, test cases, or hash anchors.
- **No contract mutation:** Does not modify `EvaluationMetricsV1`, `EvaluationProtocolV1`, or `LoggingContractV1`.
- **No code additions:** Does not implement new runtime metrics or training loops in this gate.
- **Prohibited Anti-Patterns:**
  - *No Unified Mega-Score:* Computing a scalar weighted average or geometric mean of success rates to declare a single "best agent" is strictly prohibited.
  - *No Reward Ranking:* Cumulative RL return (`episode_return`) is a learning diagnostic and must never be used to rank different algorithms.
  - *No Cherry-Picking:* Selecting favorable test subsets or reporting only Easy/Medium tiers after evaluating TEST is prohibited.
  - *No Fabricated Significance:* Treating individual scenario episodes ($N=60$ or $N=180$) from the same model as independent training samples to claim $p < 0.05$ is scientifically invalid and prohibited.

---

## 2. Authoritative Architecture and Hierarchy of Truth

### 2.1 The Single Source of Truth Hierarchy
To guarantee reproducibility across headless CLI runs, automated CI, and interactive GUI inspection, Platform V1 enforces an immutable data hierarchy:

```
+-----------------------------------------------------------------------------------+
|                        LOCAL RUN ARTIFACT DIRECTORY                                |
|                              runs/<run_id>/                                       |
+-----------------------------------------------------------------------------------+
  |
  +---> run_manifest.json     [Config, Git SHA, Environment, Contract SHA-256 Hashes]
  |
  +---> run_state.json        [Lifecycle: INITIALIZING -> RUNNING -> COMPLETE / FAILED]
  |
  +---> run_integrity.json    [SHA-256 Hashes of all artifacts generated at finalization]
  |
  +---> episodes.csv          [PRIMARY EPISODE TRUTH: 31 columns, 1 row per episode]
  |
  +---> timing.csv            [PRIMARY TIMING TRUTH: monotonic agent act latencies]
  |
  +---> summary.json          [AUTHORITATIVE AGGREGATE TRUTH: overall, tier, macro metrics]
  |
  +---> wandb_sync.json       [OPTIONAL MIRROR STATUS: local artifacts remain sovereign]
```

1. **`summary.json` is the sole authority for run-level aggregate metrics.** No consumer (CLI, script, or Qt GUI) may recalculate or alter these figures.
2. **`episodes.csv` is the sole authority for episode-level observations.** All statistical distributions and post-hoc analyses must trace directly to these rows.
3. **`timing.csv` is the sole authority for agent decision latencies.** Timings strictly isolate `agent.act()` execution from environment simulation and I/O.
4. **`run_integrity.json` is the cryptographic gate.** If any artifact digest mismatches the recorded hash, the entire run is marked `FAILED / UNTRUSTED`.
5. **W&B is a downstream mirror, not a primary data source.** Network drops or W&B API outages do not invalidate a verified local run; conversely, a W&B dashboard cannot authenticate a run whose local artifacts failed verification.

### 2.2 Invariant on Consumer Applications
The Research Workbench (GUI) and downstream comparison tools are **strictly read-only viewers**. They must never edit run files, delete failed runs, automatically patch broken hashes, or silently substitute fallback values.

---

## 3. The Five Locked Primary Benchmark Metrics

The Platform V1 benchmark evaluates performance across exactly five primary metrics defined in `configs/platform/evaluation_metrics_v1.json` and computed in `src/platform/metrics.py:compute_aggregate_metrics()`.

| Metric Identifier | Mathematical Definition | Unit / Range | Preferred Direction | Edge Case / Null Semantics |
|---|---|---|---|---|
| `clean_success_rate` | $\frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{clean\_success}_i = \text{True})$ | Unitless, $[0.0, 1.0]$ | **Higher** | $0.0$ if no episode reached destination cleanly. Requires `clean_success=True`, `raw_arrival=True`, and `primary_reason=SUCCESS`. |
| `safety_failure_rate` | $\frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{primary\_reason}_i \in \mathcal{S}_{\text{safety}})$ | Unitless, $[0.0, 1.0]$ | **Lower** | $0.0$ if zero safety terminations. $\mathcal{S}_{\text{safety}}$ includes human, vehicle, object, building, sidewalk collisions, and out-of-road. |
| `mean_final_route_completion` | $\frac{1}{N} \sum_{i=1}^N c_i^{\text{final}}$ | Unitless, $[0.0, 1.0]$ | **Higher** | Arithmetic mean of completion at termination. Never infer from `max_route_completion`. |
| `median_final_route_completion` | $\text{median}(\{c_i^{\text{final}}\}_{i=1}^N)$ | Unitless, $[0.0, 1.0]$ | **Higher** | Median completion fraction. Robust against bimodal early-crash distributions. |
| `mean_time_to_clean_success_s` | $\frac{1}{K} \sum_{i \in \mathcal{K}} t_i^{\text{success}}$ | Seconds, $[0.0, \infty)$ | **Lower** *(conditional)* | **`null` / `None` if $K = 0$.** In CSV it is empty; in GUI it displays as `"N/A — no clean-success samples"`. Never replace with $0.0$ or $\infty$. |

Where:
- $N$ is the total number of evaluated episodes in the suite (e.g., $N=60$ for canonical TEST, $N=96$ for canonical VALIDATION).
- $K = \sum_{i=1}^N \mathbb{I}(\text{clean\_success}_i = \text{True})$ is the clean success count.
- $\mathcal{K} = \{i \mid \text{clean\_success}_i = \text{True}\}$ is the index set of clean successes.
- $\mathcal{S}_{\text{safety}} = \{\text{CRASH\_HUMAN}, \text{CRASH\_VEHICLE}, \text{CRASH\_OBJECT}, \text{CRASH\_BUILDING}, \text{CRASH\_SIDEWALK}, \text{OUT\_OF\_ROAD}\}$.

### 3.1 Crucial Distinction: `safety_failure_rate` vs `raw_any_safety_event_rate`
- **`safety_failure_rate`** follows mutually exclusive primary outcome classification (`primary_terminal_reason`). An episode has exactly one primary terminal reason. If an agent runs out of time while brushing a curb, `primary_terminal_reason` may be `TIMEOUT`, not a safety failure.
- **`raw_any_safety_event_rate`** is a secondary diagnostic that records whether *any* physical collision or road departure flag was asserted during the episode, even if the episode ultimately terminated due to timeout or arrival.
- **Rule:** Never confuse or conflate these two metrics. Benchmark scorecards require `safety_failure_rate`.

### 3.2 Conditioning Semantics of `mean_time_to_clean_success_s`
- `mean_time_to_clean_success_s` is strictly conditioned on the subset $\mathcal{K}$.
- If Agent A achieves $10\%$ clean success with a mean time of $25.0\,\text{s}$, and Agent B achieves $90\%$ clean success with a mean time of $38.0\,\text{s}$, Agent A **cannot** be claimed as more efficient or faster. Agent A failed $90\%$ of the tasks; its time metric only reflects the easiest subset of scenarios it managed to solve.
- **Rule:** `mean_time_to_clean_success_s` must always be reported alongside `clean_success_rate` and the explicit sample count $K/N$.

---

## 4. Metric Aggregation Hierarchy: Overall, Per-Tier, and Macro

Platform V1 organizes metrics into three distinct aggregation levels:

### 4.1 Overall Metrics (`overall_metrics`)
- Computed across the entire evaluated suite ($N$ episodes pooled directly).
- Found under namespace `summary.json["overall_metrics"]` (and W&B `metrics/overall/*`).
- Because canonical suites are balanced across tiers (15 per tier in TEST, 24 per tier in VALIDATION), overall rate metrics equal the unweighted mean of tier rates when all tiers have identical counts. However, they remain conceptually distinct from macro metrics.

### 4.2 Per-Tier Scorecards (`tier_metrics`)
- Computed separately for each difficulty tier: `Easy`, `Medium`, `Hard`, and `Extreme`.
- Found under namespace `summary.json["tier_metrics"][<tier>]` (and W&B `metrics/tier/<tier>/*`).
- Reporting per-tier scorecards is **mandatory**. Aggregating across tiers without displaying the tier breakdown obscures critical performance cliffs (e.g., an agent excelling on straight roads in Easy but having $0\%$ success on complex intersections in Extreme).

### 4.3 Macro Metrics (`macro_metrics`)
Platform V1 implements macro summaries strictly in `src/platform/protocol.py:compute_macro_metrics()`:
```python
macro_success = sum(tier_scorecards[t].clean_success_rate for t in tiers) / 4.0
macro_safety_fail = sum(tier_scorecards[t].safety_failure_rate for t in tiers) / 4.0
macro_completion = sum(tier_scorecards[t].mean_final_route_completion for t in tiers) / 4.0
```
- **Exactly three macro metrics are implemented:**
  1. `macro_clean_success_rate`
  2. `macro_safety_failure_rate`
  3. `macro_mean_final_route_completion`
- **Equal-Tier Weighting:** Each of the 4 tiers is assigned an exact weight of $0.25$.
- **All 4 Tiers Required:** If any tier is missing, `compute_macro_metrics()` raises `ValueError`. In partial or custom runs, `macro_metrics` is set to `None`.
- **Prohibitions:**
  - *No Macro Median:* There is no `macro_median_final_route_completion` (medians cannot be averaged linearly).
  - *No Macro Time:* There is no `macro_mean_time_to_clean_success_s` (averaging conditioned times with unequal denominators is mathematically invalid).
  - *No Geometric-Mean Mega-Score:* The field `"geometric_mean_success_score"` is explicitly hardcoded to `None` per platform policy.

---

## 5. Diagnostic and Latency Metrics

### 5.1 Training Diagnostic: `episode_return`
- `episode_return` (cumulative reward return) is saved in `episodes.csv` and summarized as `mean_episode_return` in `summary.json["overall_metrics"]`.
- **Contract Rule:** Cumulative reward is solely an internal diagnostic signal for reinforcement learning dynamics. Because reward functions can be scaled, shaped, or shifted arbitrarily between algorithms, **`episode_return` must NEVER be used as a benchmark ranking metric or performance score across methods.**

### 5.2 Decision Timing and Hardware Latency
- Measured via high-resolution monotonic clock (`time.perf_counter_ns()`) in `src/platform/experiment_logging.py`.
- **Timing Boundary:** Strictly brackets `agent.act(agent_input)`. Excludes environment physics, observation assembly, action adapter conversion, rendering, disk logging, and network sync.
- Recorded per episode in `timing.csv` (`act_count`, `mean_act_ms`, `median_act_ms`, `p95_act_ms`, `max_act_ms`, `total_act_ms`, `latency_sync_policy`) and aggregated in `summary.json["timing_overall"]` (`total_act_count`, `weighted_mean_act_ms`, `max_act_ms`).
- **Hardware Caveat:** Decision latency is fundamentally hardware- and environment-dependent (CPU model, GPU architecture, thermal throttling, OS scheduling). Comparing inference latencies across different hardware platforms is scientifically invalid unless identical hardware environments and synchronization policies are proven.

---

## 6. Experimental Design, Splits, and Seed Taxonomy

Platform V1 defines an exact, reproducible geometry universe governed by `configs/platform/seed_plan_v1.json` and `configs/platform/evaluation_protocol_v1.json`.

```
+-----------------------------------------------------------------------------------+
|                        240 PROCEDURAL ROAD GEOMETRIES                              |
|                   (12 Scenario Families x 20 Seeds [0..19])                       |
+-----------------------------------------------------------------------------------+
       |                                   |                                  |
       v                                   v                                  v
+---------------+                   +---------------+                  +---------------+
|     TRAIN     |                   |  VALIDATION   |                  |     TEST      |
| 180 Geometries|                   | 48 Geometries |                  | 12 Geometries |
| (45 per tier) |                   | (12 per tier) |                  | (3 per tier)  |
+---------------+                   +---------------+                  +---------------+
       |                                   |                                  |
       v                                   v                                  v
Unconstrained RL,                   Tuning, Model Selection,           Final Scientific
Policy Exploration,                 Checkpoint Selection               Evaluation Only
Domain Randomization                (2 Seeds: 5101, 5102)              (5 Seeds: 9101-9105)
                                    ==> 96 Canonical Cases             ==> 60 Canonical Cases
```

### 6.1 Evaluation Suites
1. **Canonical VALIDATION Suite ($N=96$ cases):**
   - 48 geometries $\times$ 2 environment seeds (`5101`, `5102`).
   - Exactly 24 cases per tier (`Easy`: 24, `Medium`: 24, `Hard`: 24, `Extreme`: 24).
   - Dedicated exclusively to hyperparameter tuning, ablation studies, and checkpoint selection.
2. **Canonical TEST Suite ($N=60$ cases):**
   - 12 canonical geometries $\times$ 5 environment seeds (`9101`, `9102`, `9103`, `9104`, `9105`).
   - Exactly 15 cases per tier (`Easy`: 15, `Medium`: 15, `Hard`: 15, `Extreme`: 15).
   - Strictly held out. Evaluated once on frozen, finalized model candidates.
   - **Zero tuning on TEST:** Modifying hyperparameters, reward weights, or network architecture based on TEST outcomes is benchmark contamination.

### 6.2 Disambiguating the Seed Taxonomy
The word "seed" must never be used ambiguously. Platform V1 defines 5 distinct seed concepts:
1. `geometry_generation_seed` (int, `0..19`): Controls procedural road mesh generation. Frozen in geometry manifests.
2. `environment_seed` (int, `5101..5102` for val; `9101..9105` for test): Controls background traffic spawn, dynamic vehicle trajectories, and simulator stochasticity.
3. `training_run_seed` (int, `101, 202, 303`): Root seed initializing training environment instances, weight initializations, and replay buffer sampling for learned agents.
4. `agent_seed` (int, `101, 202, 303`): Controls agent-side decision stochasticity (e.g., action sampling in stochastic policies).
5. `protocol_order_seed` (int, `424242`): Fixed seed for deterministic pseudo-random shuffling of evaluation case execution order.

---

## 7. Statistical Policy and Reporting Requirements

### 7.1 Statistical Independence Unit
- **For Learned Agents (Stage 5 RL):** The fundamental unit of statistical independence is the **training run replicate**, NOT the individual evaluation episode.
  - Evaluating 1 trained model checkpoint over 60 test cases produces 60 correlated observations from a single policy parameter realization $\theta_1$.
  - Training 3 separate models from seeds `101`, `202`, and `303` yields 3 independent realizations $\{\theta_1, \theta_2, \theta_3\}$.
  - Evaluating each model on the 60 TEST cases yields $3 \times 60 = 180$ total episodes. These 180 episodes **must never be pooled as 180 independent samples** to compute statistical tests (e.g., claiming two-sample t-test with $N=180$).
- **For Genuinely Deterministic Agents (Stage 1 Rule-Based):**
  - A deterministic agent produces identical trajectories given identical inputs and environment seeds.
  - Reporting 3 identical evaluation runs for a deterministic agent and computing sample standard deviation ($0.0$) is meaningless.
  - Deterministic agents evaluate the canonical suite **once** ($N=60$ on TEST or $N=96$ on VALIDATION), reporting exact performance without fabricated replicate variance.

### 7.2 Aggregation Across Replicates ($R \ge 2$)
For learned algorithms evaluated across $R$ independent training replicates ($R=3$ recommended in Platform V1):
Let $m_r$ be the aggregate metric (e.g., `clean_success_rate`) of replicate $r \in \{1, \dots, R\}$.
$$\bar{m} = \frac{1}{R} \sum_{r=1}^R m_r$$
$$s = \sqrt{\frac{1}{R-1} \sum_{r=1}^R (m_r - \bar{m})^2} \quad (\text{for } R \ge 2)$$

**Mandatory Reporting Rules:**
1. State the number of independent training replicates explicitly ($R=3$).
2. Report the individual performance of each replicate alongside the sample mean $\bar{m}$ and sample standard deviation $s$.
3. Acknowledge that with $R=3$, the power of classical significance tests (e.g., Student's t-test, Mann-Whitney U) is extremely low. Avoid dogmatic claims of "statistically significant superiority" based solely on $R=3$.
4. Paired comparisons across two algorithms should match exact case IDs ($c_1, \dots, c_{60}$) and show paired per-case performance profiles descriptively.

---

## 8. Run Compatibility Decision Rules: Three-Layer Verification

Before comparing Run A and Run B, an automated or manual evaluator must evaluate three sequential verification layers:

```
+-----------------------------------------------------------------------------------+
| LAYER A: TRUSTWORTHY EVIDENCE (Integrity, Completeness, Provenance)                |
| If FAILS ==> REJECTED_UNTRUSTED (Hard Block: cannot use results)                  |
+-----------------------------------------------------------------------------------+
                                       | Passes
                                       v
+-----------------------------------------------------------------------------------+
| LAYER B: SCIENTIFIC COMPARABILITY (Protocols, Splits, Actions, Inputs)             |
| If FAILS ==> INCOMPATIBLE (Block) or QUALIFIED_COMPARISON (Caveated)              |
+-----------------------------------------------------------------------------------+
                                       | Passes
                                       v
+-----------------------------------------------------------------------------------+
| LAYER C: EVIDENCE STRENGTH (Statistical Power, Paired Matching, Inference)         |
| Determines validity of scientific superiority claims                              |
+-----------------------------------------------------------------------------------+
```

### 8.1 Layer A: Trustworthy Evidence (Integrity Gating)
A run is valid for benchmark comparison **only if all of the following hold**:
1. Directory contains all required artifacts (`run_manifest.json`, `run_state.json`, `episodes.csv`, `timing.csv`, `summary.json`, `run_integrity.json`).
2. `run_state.json` status is `COMPLETE`.
3. `run_integrity.json` verification status is `VERIFIED` (every artifact SHA-256 matches its disk content).
4. `canonical_run == True` (clean working tree, verified MetaDrive commit `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`, pinned version `0.4.3`).
5. Recorded episode count equals expected count ($N=60$ for TEST, $N=96$ for VALIDATION).
6. Zero unresolved technical failures (`technical_failure_count == 0` or verified benign).
*Failure at Layer A results in state `REJECTED_UNTRUSTED`.*

### 8.2 Layer B: Scientific Comparability (Fair Comparison Gating)
Two verified runs (Run A and Run B) can be compared **only under strict compatibility**:
1. **Mode and Split Match:** Both runs must be on the same split (TEST vs TEST, or VALIDATION vs VALIDATION). Comparing a VALIDATION run to a TEST run is strictly prohibited (`INCOMPATIBLE`).
2. **Case Manifest and Hash Match:** Both runs must evaluate identical `case_id` manifests and hashes (`0832c38e...` for TEST).
3. **Input Profile Alignment:** Both agents must share identical observation rights (e.g., both use `CORE_V1`). If Agent A uses `CORE_V1` (kinematics + lidar) and Agent B uses `FULL_V1` (oracle traffic context), they cannot be compared as an algorithmic benchmark (`INCOMPATIBLE` for algorithm comparison; reportable only as an informational information-ablation study).
4. **Action Adapter Alignment:**
   - *Same Action Adapter:* (e.g., both use `CONTINUOUS_BOX2`) $\rightarrow$ **`STRICTLY_COMPARABLE`** (direct algorithmic comparison).
   - *Different Action Adapter:* (e.g., Agent A uses `CONTINUOUS_BOX2`, Agent B uses `DISCRETE_25`) $\rightarrow$ **`QUALIFIED_COMPARISON`**. May be compared as end-to-end autonomous driving systems, but **NOT** as an isolated proof of algorithm superiority. The action discretization itself introduces structural control limitations.
*Failure at Layer B results in state `INCOMPATIBLE` or `QUALIFIED_COMPARISON`.*

### 8.3 Layer C: Evidence Strength and Scientific Interpretation
When interpreting comparable runs:
1. Show paired differences across identical case IDs.
2. Disclose replicate sample size ($R=3$) and report individual replicate variance.
3. Explicitly state that latency metrics are hardware-dependent.
4. Report all 5 primary metrics simultaneously; do not highlight success rate while ignoring a surge in safety failures.

---

## 9. Downstream System Alignment

### 9.1 Research Workbench GUI V2 Consumption
- GUI V2 must consume `summary.json` directly via `RunArtifactRepository`.
- GUI V2 must display integrity status (`VERIFIED`, `NOT_FINAL`, `UNVERIFIED`, `FAILED`) prominently.
- When comparing runs, GUI V2 must enforce Layer A and Layer B checks before rendering comparative visualizations, displaying explicit warnings if comparison is `QUALIFIED` or `INCOMPATIBLE`.
- GUI V2 must present `mean_time_to_clean_success_s` as `"N/A — no clean-success samples"` whenever clean successes equal zero.

### 9.2 Stage 5 RL Integration
- Stage 5 RL training pipelines must record training configuration digests and training run seeds (`101, 202, 303`).
- Evaluation of checkpoints must occur via headless launcher CLI in `VALIDATION` mode (for checkpoint selection) and `TEST` mode (once for final frozen evaluation).
- Stage 5 models must never be evaluated interactively in GUI with `NATIVE` rendering enabled during benchmark scoring.

---

## 10. References
- Platform V1 Contract Index: `docs/PLATFORM_V1_CONTRACT_INDEX.md`
- Experiment Workflow: `docs/EXPERIMENT_WORKFLOW.md`
- Evaluation Protocol V1: `configs/platform/evaluation_protocol_v1.json`
- Evaluation Metrics V1: `configs/platform/evaluation_metrics_v1.json`
- Logging Contract V1: `configs/platform/logging_contract_v1.json`
- Seed Plan V1: `configs/platform/seed_plan_v1.json`
- Henderson et al. (2018), *Deep Reinforcement Learning That Matters*, AAAI.
- Patterson et al. (2024), *Empirical Design in Reinforcement Learning*, JMLR.
- Agarwal et al. (2021), *Deep Reinforcement Learning at the Edge of the Statistical Precipice*, NeurIPS.
