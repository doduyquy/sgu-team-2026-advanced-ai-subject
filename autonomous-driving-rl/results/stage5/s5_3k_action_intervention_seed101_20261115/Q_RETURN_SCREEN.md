# S5.3K — Q versus soft-return screen

This is a derived diagnostic screen from the completed K1 episode records. It does not replace or modify the original run records.

Across 12 full episodes, mean initial online `min(Q1,Q2)` was **15.232** and target-critic `min(Q1,Q2)` was **15.281**. Mean observed soft return plus the separately labeled TIMEOUT target-critic bootstrap estimate was **9.095**. Mean online-Q difference `Q - return estimate` was **+6.137**; target-Q difference was **+6.186** (online range +6.022 to +6.259).

| Initial throttle condition | n | Mean online minQ | Mean target minQ | Mean soft-return estimate | Mean online Q − estimate |
|---|---:|---:|---:|---:|---:|
| brake_q15 | 4 | 15.241 | 15.288 | 9.095 | +6.147 |
| near_neutral_q50 | 4 | 15.232 | 15.280 | 9.095 | +6.137 |
| throttle_q85 | 4 | 15.223 | 15.273 | 9.096 | +6.127 |

## Interpretation

The critic predicts substantially higher absolute values than this small rollout screen. This is a candidate calibration discrepancy to investigate, not a confirmed critic error: each case-condition has one trajectory, stochastic variance is unestimated, and TIMEOUT bootstrap uses the checkpoint target critic. Relative action preference is also weak: Q ranks brake above throttle by about 0.019 on average, while the soft-return estimate favors throttle over brake by about 0.0008 paired mean. These tiny relative differences are not decisive with four geometries.

The intervention changed only the first throttle action. Subsequent stochastic SAC behavior returned to nearly identical low-speed trajectories, so this experiment does not test sustained acceleration or prove why training produced the policy.
