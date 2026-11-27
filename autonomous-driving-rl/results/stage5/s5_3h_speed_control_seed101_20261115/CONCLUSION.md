# S5.3H — Deterministic policy and speed-control isolation

Status: **COMPLETE**. 24 probes across the same eight internal TRAIN diagnostic geometries/seeds used in S5.3G. Each arm was capped at 300 steps; censored probes are not timeouts.

All arms used the same proportional longitudinal speed controller; actor throttle was recorded but overridden. The controller target is 12 km/h with `clip(0.08 * (12 - speed_km_h), -0.45, +0.60)`. Arms: deterministic SAC steering, stochastic SAC steering, and public-waypoint pure pursuit (8 m lookahead). This intervention is diagnostic only, not benchmark policy performance.

## Results

| Arm | n | Outcomes | Censored | Mean progress delta | Mean speed km/h | Mean abs lateral offset m | Raw actor throttle mean | Speed controller throttle mean |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| A_sac_deterministic | 8 | OUT_OF_ROAD 2, PROBE_CENSORED 6 | 6 | 0.1472 | 11.40 | 0.499 | -0.104 | 0.046 |
| B_sac_stochastic | 8 | OUT_OF_ROAD 1, PROBE_CENSORED 7 | 7 | 0.0795 | 6.01 | 0.788 | -0.096 | 0.473 |
| C_waypoint_pure_pursuit | 8 | PROBE_CENSORED 8 | 8 | 0.1556 | 11.49 | 0.159 | n/a | 0.039 |

## Same-observation policy action audit

At each of the 8 reset observations, deterministic action was compared with 32 stochastic samples on that exact observation before any step. Across reset states, deterministic actor mean throttle was -0.0927; mean stochastic-sample throttle was -0.0667; stochastic positive-throttle fraction averaged 0.422. Adapter output matched raw action on each deterministic check: True. Per-case samples are in `initial_action_audits.jsonl`.

## Interpretation

This compares steering sources under one shared speed controller, so it asks whether steering choice limits progress/safety when throttle is externally stabilized around 12 km/h. It still does not represent the SAC policy acting jointly. Compare terminal outcomes and route progress together; a censored rollout is not safe. The proportional speed controller is a diagnostic intervention and was not tuned as an agent or benchmark baseline.

Read-only checks: policy weights unchanged = `True`; optimizer updates 8999 -> 8999; replay size 0 -> 0. No VALIDATION/TEST access.

Checkpoint SHA-256 `6347ae4d82958bba89147ec39507ce3cb6abaa7e04d82f97101d1c09c2dea073`; internal split SHA-256 `082d116f393bb1b29dea59988087b3fb0f252705eefa97f601bce7550f2740fa`. Full records and source hashes are in adjacent JSON files.
