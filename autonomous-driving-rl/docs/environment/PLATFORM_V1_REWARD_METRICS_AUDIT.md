# Platform V1 Reward Specification & Scientific Evaluation Metrics Audit

> **Status:** AUDIT & DESIGN RECOMMENDATION (GATE 4)  
> **Freeze Status:** NON-FROZEN — RESEARCH SPECIFICATION ONLY  
> **Target Scope:** RL Training Objective (RewardSpecV1) & Algorithm-Agnostic Benchmark Evaluation (EvaluationMetricsV1)  
> **Date:** September 2026  

---

## 1. Executive Summary & Foundational Distinction

Gate 4 establishes two **strictly independent, decoupled concepts** for Research Platform V1:

```text
+-----------------------------------------------------------------------------------------------+
|                                    EPISODE STEP OUTCOME                                       |
+-----------------------------------------------------------------------------------------------+
                             |                                     |
                             v                                     v
       [A. TRAINING REWARD (RewardSpecV1)]      [B. BENCHMARK METRICS (EvaluationMetricsV1)]
       - Scalar optimization objective          - Scientific benchmark scorecard
       - Consumed only by learning algorithms   - Algorithm-agnostic across Stages 0 through 7
       - Labeled: TRAINING DIAGNOSTIC           - Primary: Clean Success, Safety Failure, Progress
       - Bounded, length-invariant return       - Never ranks algorithms by cumulative return
```

### Core Invariants:
1. **Never Rank by Return:** The scientific benchmark must never rank algorithms primarily by episode return. Stages 0–3 (Random, Rules, Planners, MCTS) may ignore reward completely and remain fully evaluatable.
2. **Safety-First Precedence:** A vehicle arriving at the goal terminus while simultaneously crashing must **never** receive the clean success reward bonus.
3. **Length Invariance:** Net route progress across any completed map normalizes to $\approx +1.0$, eliminating the multi-hundred-point return scaling artifacts of native MetaDrive.

---

## 2. Source-of-Truth Verification

Authoritative source reference is the local pinned MetaDrive repository:
- **Local Path:** `D:\SGU\CNTT\TTNTNC\metadrive-src`
- **Pinned Git Commit:** `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`
- **Package Version:** `0.4.3` (`metadrive.constants.VERSION == "0.4.3"`)
- **Git Status:** Verified clean working tree via `git status --porcelain`.
- **Automated Validation:** Verified dynamically by `scripts/audit_reward_metrics.py`.

---

## 3. Source Audit: Native MetaDrive Reward & Cost Scheme

From source inspection of `metadrive/envs/metadrive_env.py` (lines 16–94, 246–290):

### Native Reward Equation:
```python
reward = 0.0
reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor * positive_road
reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h) * positive_road

step_info["step_reward"] = reward

if self._is_arrive_destination(vehicle):
    reward = +self.config["success_reward"]
elif self._is_out_of_road(vehicle):
    reward = -self.config["out_of_road_penalty"]
elif vehicle.crash_vehicle:
    reward = -self.config["crash_vehicle_penalty"]
elif vehicle.crash_object:
    reward = -self.config["crash_object_penalty"]
elif vehicle.crash_sidewalk:
    reward = -self.config["crash_sidewalk_penalty"]
```

### Default Parameters (`METADRIVE_DEFAULT_CONFIG`):
- `driving_reward = 1.0` (weight on meter-level longitudinal displacement along lane)
- `speed_reward = 0.1` (weight on speed fraction $v / v_{\text{max}}$)
- `success_reward = 10.0`
- `out_of_road_penalty = 5.0`
- `crash_vehicle_penalty = 5.0`
- `crash_object_penalty = 5.0`
- `crash_sidewalk_penalty = 0.0` (default is zero penalty!)
- `use_lateral_reward = False`

### Native Cost Function (`metadrive_env.py` lines 206–216):
```python
if self._is_out_of_road(vehicle):
    step_info["cost"] = self.config["out_of_road_cost"]  # default 1.0
elif vehicle.crash_vehicle:
    step_info["cost"] = self.config["crash_vehicle_cost"]  # default 1.0
elif vehicle.crash_object:
    step_info["cost"] = self.config["crash_object_cost"]  # default 1.0
```

---

## 4. Native Reward Pathology Findings

Our source audit and empirical tests revealed 5 fundamental pathologies in MetaDrive's native reward scheme:

1. **Explosive Distance-Scaling Defect:**
   - Because `driving_reward` rewards unnormalized meters traveled ($1.0 \times \text{meters}$), the return of a clean successful episode scales directly with corridor length:
     - Easy (`SCS` s11, $349.6\text{ m}$): Return $= \mathbf{350.75}$
     - Medium (`SCTCS` s0, $521.1\text{ m}$): Return $= \mathbf{543.29}$
     - Hard (`XTOCS` s19, $496.9\text{ m}$): Return $= \mathbf{516.16}$
     - Extreme (`CrXROSTR` s6, $938.6\text{ m}$): Return $= \mathbf{966.74}$
   - Based on an analytical estimate derived from meter distance scaling ($0.50 \times 938.6\text{ m} \times 1.0\text{ driving\_reward} + \dots$), an agent achieving 50% route completion on an Extreme map receives $\sim 480$ reward points—drastically outscoring an agent that achieves 100% clean success on an Easy map ($350.75$ points). Cumulative return is therefore fundamentally broken as a cross-map benchmark metric.

2. **Terminal Replacement / Override Semantics:**
   - On the terminal transition, the dense step reward is completely overwritten (`reward = +self.config["success_reward"]`), rather than acting as an additive bonus (`reward += ...`). The final step's progress and speed contribution are completely discarded.

3. **Precedence Inversion (Arrival Hacking):**
   - Native `reward_function()` evaluates `if self._is_arrive_destination(vehicle)` as the first branch.
   - If a policy crosses the destination line while simultaneously crashing into a traffic vehicle or curb, native MetaDrive blindly awards `+10.0` success reward and ignores the crash in the return.

4. **Missing Penalties for Critical Failures:**
   - In `done_function()`, `crash_human` and `crash_building` trigger immediate episode termination.
   - However, in `reward_function()`, `crash_human` and `crash_building` have no dedicated explicit reward penalty; reward falls through to dense step reward unless another earlier reward branch also triggers.

5. **Shadowed Sidewalk Penalty:**
   - In `reward_function()`, `elif self._is_out_of_road(vehicle)` precedes `elif vehicle.crash_sidewalk:`.
   - Under the audited default configuration (`out_of_route_done=False`, `on_continuous_line_done=True`), sidewalk contact triggers `_is_out_of_road()` first, making the sidewalk penalty branch unreachable/shadowed. Default `crash_sidewalk_penalty = 0.0`.

*Empirical evidence:* Committed in `results/audits/reward_metrics/native_reward_cases.csv`.

---

## 5. Audit of Route Completion as a Normalized Shaping Signal

Gate 1 strictly established that raw `route_completion` is evaluator-only telemetry and must never be exposed directly in `AgentInput`. However, we evaluated whether internal step differences:
$$\Delta \text{RC}_t = \text{RC}_t - \text{RC}_{t-1}$$
provide a robust, length-invariant scalar progress signal for RL training.

### Empirical Validation Findings (`route_completion_dynamics.csv`)

| Tier | Sequence | Seed | Steps | Route Length | Initial RC | Final RC | Telescoping Sum ($\sum \Delta \text{RC}$) | Actual Diff ($\text{RC}_T - \text{RC}_0$) | Telescoping Error | Negative Deltas | Max Step $\Delta$ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Easy** | `SCS` | 11 | 407 | $349.6\text{ m}$ | 0.0143 | 0.9879 | 0.973562 | 0.973562 | **$1.11\times 10^{-16}$** | 0 | 0.002725 |
| **Medium** | `SCXCS` | 11 | 383 | $523.8\text{ m}$ | 0.0095 | 0.6388 | 0.629234 | 0.629234 | **$0.00\times 10^0$** | 0 | 0.002696 |
| **Hard** | `SCXOCS` | 2 | 779 | $643.0\text{ m}$ | 0.0078 | 0.9937 | 0.985925 | 0.985925 | **$0.00\times 10^0$** | 0 | 0.001307 |
| **Extreme** | `CrXROSTR` | 6 | 1123 | $938.6\text{ m}$ | 0.0053 | 0.9951 | 0.989776 | 0.989776 | **$1.11\times 10^{-16}$** | 0 | 0.001428 |

### Key Properties Confirmed:
1. **Exact Telescoping Sum:** $\sum_{t=1}^T \Delta \text{RC}_t \equiv \text{RC}_T - \text{RC}_0$ down to float machine precision ($<10^{-15}$).
2. **Monotonic Forward Driving:** Zero negative deltas observed during forward reference driving.
3. **Smooth Step Magnitude:** Maximum step delta is bounded ($\Delta \text{RC} \in [0.00001, 0.0027]$ per 0.1s step). Zero transition spikes or discontinuities across lane, block, or intersection boundaries.
4. **Length Invariance:** Across all maps, total route progress sums to $\approx +0.98 - 0.99$, completely decoupling progress return from meter corridor distance.

---

## 6. Reward Candidate Comparison

We evaluated three reward architectures across benchmark trajectories:

- **Candidate A (Native MetaDrive):** Unnormalized displacement + speed bonus + terminal replacement.
- **Candidate B (Normalized Progress + Terminal):** $r_t = w_{\text{prog}} \times \Delta \text{RC}_t + r_{\text{terminal}}$.
- **Candidate C (Proposed RewardSpecV1):** $r_t = w_{\text{prog}} \times \Delta \text{RC}_t - \frac{\text{budget}}{H} + r_{\text{terminal}}$ with safety-first terminal resolution.

### Comparative Return Matrix (`reward_candidate_comparison.csv`)

| Trajectory Description | Tier | Route Length | Steps | Native Return (Cand A) | Cand B Return | Cand C Return (Proposed) | Cand C Progress | Cand C Time Cost | Cand C Terminal | Outcome |
|---|---|---|---|---|---|---|---|---|---|---|
| `easy_clean_success` | Easy | $349.6\text{ m}$ | 407 | **350.75** | 1.974 | **1.877** | +0.974 | -0.097 | +1.000 | SUCCESS |
| `medium_clean_success`| Medium | $521.1\text{ m}$ | 629 | **543.29** | 1.981 | **1.880** | +0.981 | -0.100 | +1.000 | SUCCESS |
| `hard_clean_success` | Hard | $496.9\text{ m}$ | 599 | **516.16** | 1.984 | **1.883** | +0.984 | -0.100 | +1.000 | SUCCESS |
| `extreme_clean_success`| Extreme | $938.6\text{ m}$ | 1123 | **966.74** | 1.990 | **1.890** | +0.990 | -0.100 | +1.000 | SUCCESS |
| `deliberate_out_of_road`| Easy | $349.6\text{ m}$ | 12 | -4.25 | -0.998 | **-1.000** | +0.003 | -0.003 | -1.000 | OUT_OF_ROAD |
| `stationary_timeout` | Easy | $349.6\text{ m}$ | 100 | **+0.01** | 0.000 | **-0.250** | 0.000 | -0.250 | 0.000 | TIMEOUT |
| `simultaneous_arrival_crash`| Synthetic | $500.0\text{ m}$ | 500 | **+10.00** | 0.000 | **-0.125** | +1.000 | -0.125 | -1.000 | CRASH_VEHICLE |

### Definitive Invariant Demonstration:
1. **Cross-Map Return Invariance:** Clean success in Candidate C produces **$1.877$ (Easy), $1.880$ (Medium), $1.883$ (Hard), and $1.890$ (Extreme)**—a uniform $\sim 1.88$ return across all tiers.
2. **Prevention of Stationary Exploitation:** A vehicle sitting stationary until timeout receives **$-0.250$** in Candidate C (penalized for wasting budget), whereas native MetaDrive awards $+0.01$.
3. **Arrival Hacking Blocked:** A vehicle crossing the line while crashing receives **$-0.125$** in Candidate C, whereas native MetaDrive awards $+10.00$.

---

## 7. Recommended Non-Frozen RewardSpecV1 Architecture (Candidate C)

```python
r_t = w_progress * delta_route_completion - (time_penalty_budget / horizon_steps) + r_terminal
```

### Calibrated Parameters:
- `progress_weight = 1.0`: Normalizes full-route progress return to $\approx +1.0$.
- `time_penalty_budget = 0.25`: Total possible time cost across the entire horizon $H$ is capped at $0.25$ ($H \times \frac{0.25}{H} = 0.25$).
- `success_bonus = 1.0`: Awarded if and only if `clean_success == True`.
- `safety_penalty = 1.0`: Uniform penalty applied upon any safety terminal failure (`CRASH_HUMAN`, `CRASH_VEHICLE`, `CRASH_OBJECT`, `CRASH_BUILDING`, `CRASH_SIDEWALK`, `OUT_OF_ROAD`).
- `timeout_penalty = 0.0`: Timeout is truncation; the accumulated time cost ($-0.25$) is sufficient disincentive without double-penalizing.

### Component Decomposition via `RewardBreakdown`:
Every step reward returns an explicit breakdown container:
```python
@dataclass(frozen=True)
class RewardBreakdown:
    progress_reward: float
    time_cost: float
    terminal_reward: float
    total_reward: float
```
Exposing components individually prevents silent reward hacking and facilitates policy loss diagnostics.

---

## 8. Uniform vs. Severity-Specific Safety Penalties

We explicitly considered whether to differentiate penalties (e.g. $-2.0$ for pedestrian/vehicle collision vs. $-0.5$ for sidewalk contact):
- **Decision:** Adopt a **uniform safety penalty ($-1.0$)** in RewardSpecV1.
- **Scientific Justification:** MapSuiteV1 currently maintains `accident_prob = 0.0` (zero deliberate pedestrian stress scenarios). Inventing a complex moral/ethical penalty hierarchy is unjustified by current benchmark evidence.
- Crucially, **the raw event taxonomy is preserved in full fidelity by EvaluationMetricsV1**, allowing researchers to track pedestrian, vehicle, and curb failure rates independently without distorting RL scalar gradients.

---

## 9. EvaluationMetricsV1: Primary, Secondary, and Diagnostic Hierarchy

Evaluation metrics are strictly segregated from training returns:

### 1. Primary Benchmark Metrics (The Evaluation Scorecard)
1. **`clean_success_rate`:** Proportion of episodes reaching destination terminus with zero collisions and zero boundary violations.
2. **`safety_failure_rate`:** Proportion of episodes terminating due to any physical collision or off-road departure.
3. **`mean_final_route_completion`:** Mean fraction of global route completed.
4. **`median_final_route_completion`:** Median route completion (robust against early outlier crashes).
5. **`mean_time_to_clean_success_s`:** Mean simulation time in seconds to reach clean arrival, **computed strictly over successful episodes** (null/None if 0 successes).

### 2. Secondary Diagnostic Metrics
- `raw_arrival_rate`: Raw goal line crossing frequency (including collisions).
- `timeout_rate`: Fraction of episodes truncated by step budget exhaustion.
- Mutually exclusive primary outcome rates: `crash_human_rate`, `crash_vehicle_rate`, `crash_object_rate`, `crash_building_rate`, `crash_sidewalk_rate`, `out_of_road_rate`, `unknown_termination_rate`.
- Raw individual safety event rates: `raw_crash_vehicle_rate`, `raw_crash_object_rate`, `raw_crash_building_rate`, `raw_crash_human_rate`, `raw_crash_sidewalk_rate`, `raw_out_of_road_rate`, and `raw_any_safety_event_rate` (individual event frequencies allowing overlapping/simultaneous events).
- `mean_max_route_completion`: Highest completion reached during the episode (surfaces reversing/backtracking).
- `mean_speed_kmh` and `max_speed_kmh`.

### 3. Training / Runtime Diagnostics Only
- `mean_episode_return`: Cumulative RL reward return. Strictly forbidden from primary benchmark ranking.
- `mean_episode_steps`: Average decision steps per episode.

*Validation:* Tested via `tests/test_reward_metrics.py` and `results/audits/reward_metrics/metrics_validation.csv`.

---

## 10. Raw Arrival vs. Clean Success Disentanglement

By tracking both `raw_arrival_rate` and `clean_success_rate`:
- An aggressive agent that crashes through traffic to reach the destination line achieves `raw_arrival_rate = 1.0` but `clean_success_rate = 0.0`.
- The benchmark immediately surfaces "arrival hacking" without conflating reckless driving with valid task completion.

---

## 11. Potential Reward Hacking & Pathology Analysis

We audited 8 potential reward hacking modes under Candidate C:

1. **Speed Without Progress:** Speed is not directly rewarded. Spinning wheels against an obstacle generates $\Delta \text{RC} = 0$, accumulating $-0.25/H$ per step.
2. **Long Episode Accumulation:** Time cost $-0.25/H$ ensures longer episodes have strictly lower return than fast completions of identical route progress.
3. **Near-Goal Crash:** Crashing at 99% progress yields $\approx +0.99 - \text{time} - 1.0 = -0.1$ to $-0.2$, never outscoring an early safe completion or even a conservative partial progression.
4. **Arrival + Crash Same Step:** Safety-first precedence overrides arrival, awarding $-1.0$ terminal penalty.
5. **Stationary Timeout:** Yields $-0.25$, making standing still strictly suboptimal.
6. **Oscillating / Reversing Progress:** In the benchmark vehicle configuration (`enable_reverse=False`), physical reversing is disabled. Mathematically and at the pure helper level, if $\Delta \text{RC} < 0$, RewardSpecV1 awards negative progress reward, ensuring backward motion cannot be exploited to accumulate reward. In an isolated audit-only test with `enable_reverse=True`, backward motion confirmed negative $\Delta \text{RC}$ accumulation.
7. **Route Transition Spikes:** Audited across intersections, roundabouts, and ramps; step deltas remain smooth ($<0.003$ per step), preventing transition exploitation.
8. **Lateral Boundary Exploitation:** Steering onto the sidewalk curb triggers `out_of_road` $\implies -1.0$ penalty.

---

## 12. Scientific Acceptance Questions Answered

1. **What exactly is MetaDrive's native reward equation?**  
   Dense driving displacement along lane plus speed bonus, with terminal replacement upon arrival or collision (`metadrive_env.py` lines 272–288).
2. **What happens to dense reward on terminal steps?**  
   It is completely replaced/overwritten by `reward = +success_reward` or `reward = -penalty`.
3. **Does native reward precedence conflict with Gate-3 safety-first semantics?**  
   Yes. Native reward awards `+10.0` for arrival even if the vehicle crashes on the same step.
4. **Which terminal categories lack explicit native penalties?**  
   `crash_human` and `crash_building` have zero penalties in `reward_function()`.
5. **Does native return scale with route length or episode time?**  
   Yes, strongly. Native return scales from 350 on Easy (350m) to 967 on Extreme (939m).
6. **Is route completion delta sufficiently stable as normalized shaping?**  
   Yes. Telescoping sum error is $<10^{-15}$, forward driving is monotonic, and step deltas are smooth ($<0.003$).
7. **Why was Candidate C selected?**  
   It provides length-invariant returns ($\approx 1.88$ across all tiers), bounds time cost, penalizes standing still, and strictly enforces safety-first precedence.
8. **Does clean success always outscore unsafe arrival?**  
   Yes. Clean success yields $\approx +1.88$; unsafe arrival yields $\le -0.125$.
9. **Can stationary timeout exploit reward?**  
   No. Stationary timeout yields $-0.250$.
10. **Is speed directly rewarded in RewardSpecV1?**  
    No. Efficiency is incentivized solely through the bounded time cost, leaving speed metrics to EvaluationMetricsV1.
11. **Are reward and evaluation metrics strictly separated?**  
    Yes. Implemented in separate modules (`src/platform/reward.py` vs. `src/platform/metrics.py`).
12. **What are the primary benchmark metrics?**  
    Clean success rate, safety failure rate, mean/median route completion, and conditional time-to-clean-success.
13. **Is raw arrival preserved separately from clean success?**  
    Yes, in both per-episode records and aggregate metrics.
14. **Is episode return explicitly diagnostic?**  
    Yes. Classified strictly under `TRAINING_DIAGNOSTICS_ONLY`.
15. **Can all reward/metrics logic be unit tested without simulator?**  
    Yes. 26 unit tests run in 0.001s without Panda3D.

---

## 13. Regression & Integrity Validation
- Pure unit tests: 26 passed in 0.001s (`test_episode_lifecycle.py` and `test_reward_metrics.py`).
- `CourseEnvV1`: Reset OK (shape `35,`).
- `evaluate_random.py`: Completed 20 evaluation episodes cleanly.
- Gate-1, Gate-2, and Gate-3 contracts remain completely untouched.
- Zero hardcoded drive letters in executable code.

STOP. PR remains open and unmerged.
