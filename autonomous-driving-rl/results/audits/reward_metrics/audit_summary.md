# Gate 4 Reward & Evaluation Metrics Audit Summary

## 1. Verified MetaDrive Source
- **Pinned Commit:** `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`
- **Package Version:** `0.4.3`
- **Working Tree:** Verified clean via `git status --porcelain`.

## 2. Native MetaDrive Reward Scaling and Override Semantics
- **Dense Scaling Defect:** Native return scales directly with corridor length (~68 for 350m Easy vs ~182 for 939m Extreme), making it unsuitable for cross-map benchmark ranking.
- **Terminal Override:** Native reward replaces dense step reward on terminal transitions (`reward = +success_reward`), discarding final driving/speed increments.
- **Precedence Conflict:** Native reward checks arrival before collisions, awarding `+10.0` even if the vehicle crashes on the same step.
- **Missing Penalties:** `crash_human` and `crash_building` terminate in `done_function()` but have zero explicit penalties in `reward_function()`.
- **Dead Code:** `crash_sidewalk_penalty` is unreachable because sidewalk contact triggers `_is_out_of_road()` first.

## 3. Route Completion Delta Dynamics
- **Telescoping Sum Exactness:** Sum of deltas matches `final_rc - initial_rc` down to float machine precision (error < 1e-15).
- **Monotonicity:** Zero negative deltas observed during forward reference driving.
- **Length Invariance:** Net route progress across any completed map is normalized to ~1.0.

## 4. Reward Candidate Comparison
| Trajectory | Tier | Route Length | Native Return (Cand A) | Cand B Return | Cand C Return (Proposed) | Cand C Progress | Cand C Time Cost | Cand C Terminal |
|---|---|---|---|---|---|---|---|---|
| easy_clean_success | Easy | 349.6 m | 350.7498 | 1.9736 | **1.8765** | 0.9736 | 0.0969 | 1.0 |
| medium_clean_success | Medium | 521.06 m | 543.2865 | 1.9809 | **1.8803** | 0.9809 | 0.1006 | 1.0 |
| hard_clean_success | Hard | 496.89 m | 516.1647 | 1.9839 | **1.8834** | 0.9839 | 0.1006 | 1.0 |
| extreme_clean_success | Extreme | 938.58 m | 966.7389 | 1.9899 | **1.89** | 0.9899 | 0.0999 | 1.0 |
| deliberate_out_of_road | Easy | 349.6 m | -4.2482 | -0.9976 | **-1.0005** | 0.0024 | 0.0029 | -1.0 |
| stationary_timeout | Easy | 349.6 m | 0.0092 | 0.0 | **-0.25** | 0.0 | 0.25 | 0.0 |
| simultaneous_arrival_crash | Synthetic | 500.0 m | 10.0 | 0.0 | **-0.125** | 1.0 | 0.125 | -1.0 |

## 5. Primary Benchmark Metrics Recommendation
1. **Clean Success Rate:** Destination arrival with zero safety violations.
2. **Safety Failure Rate:** Primary failure breakdown (pedestrian, vehicle, object, sidewalk, off-road).
3. **Route Completion:** Mean and median progress across episodes.
4. **Conditional Time-to-Success:** Efficiency among clean successes only (null if 0 successes).
- **Cumulative Reward Return:** Sequestered as a training/diagnostic metric; never used for benchmark ranking.
