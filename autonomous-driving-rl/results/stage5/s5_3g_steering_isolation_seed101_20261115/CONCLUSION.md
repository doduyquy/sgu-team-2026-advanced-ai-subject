# S5.3G — Steering isolation diagnostic

## Protocol and scope

Completed 24 short probes: 8 fixed internal PILOT_DIAGNOSTIC TRAIN geometries (2 per tier, distinct sequence families), each run once in each arm with the same geometry and environment seed. The 10k throttle-hold SAC checkpoint was used only to sample steering in A/B. All arms used a fixed +0.5 throttle override to isolate steering. This is an actuator diagnostic, not an SAC policy evaluation or benchmark result.

A: stochastic SAC steering each step. B: same stochastic SAC steering with a 0.15 action-unit/step slew limit. C: waypoint pure-pursuit steering from public encoded waypoints, 8 m lookahead, with no speed governor. Each probe was capped at 300 steps. A cap is `PROBE_CENSORED`, never TIMEOUT. Simulator terminal outcomes use Platform V1 classification.

## Results

| Arm | Cases | Observed terminal outcomes | Censored | Mean route progress delta | Mean speed (km/h) | Mean absolute lateral offset (m) |
|---|---:|---|---:|---:|---:|---:|
| A_actor_stochastic_steering | 8 | OUT_OF_ROAD 1, PROBE_CENSORED 7 | 7 | 0.0836 | 6.78 | 0.606 |
| B_rate_limited_actor_steering | 8 | CRASH_VEHICLE 1, OUT_OF_ROAD 7 | 0 | 0.0797 | 18.50 | 0.523 |
| C_waypoint_pure_pursuit | 8 | CRASH_VEHICLE 6, OUT_OF_ROAD 1, SUCCESS 1 | 0 | 0.3558 | 40.90 | 0.247 |

### Paired route progress deltas

Positive means the first arm progressed farther on the same geometry and environment seed.

| Paired contrast | Mean difference | Median | Cases higher / lower / tied |
|---|---:|---:|---:|
| B_rate_limited_actor_steering_minus_A_actor_stochastic_steering_route_progress | -0.0039 | -0.0181 | 2 / 6 / 0 |
| C_waypoint_pure_pursuit_minus_A_actor_stochastic_steering_route_progress | +0.2722 | +0.1817 | 8 / 0 / 0 |
| C_waypoint_pure_pursuit_minus_B_rate_limited_actor_steering_route_progress | +0.2761 | +0.1870 | 8 / 0 / 0 |

## Interpretation

The steering slew limit did not make positive-throttle motion safe in this screening set: all 8 B probes ended in OUT_OF_ROAD or CRASH_VEHICLE. A reached the 300-step cap in 7/8 probes, so those are censored observations, not timeout or verified safe runs; the remaining A probe ended OUT_OF_ROAD. C made the greatest route progress on average and had the smallest lateral offset, but it was not safe at this throttle: 6/8 CRASH_VEHICLE, 1/8 OUT_OF_ROAD, and 1/8 SUCCESS. Its mean speed was about 40.9 km/h and mean maximum speed about 70.9 km/h. A steering-only pure-pursuit rule without speed control is therefore not evidence that steering alone solves the task.

This experiment does not identify why SAC learned a near-zero deterministic throttle. It shows that under a fixed +0.5 throttle intervention, steering rate-limiting alone did not prevent safety failures, and route following without a speed governor can yield more progress while still crashing. The original SAC policy jointly chooses throttle and steering, so this probe cannot attribute its learning failure to steering alone.

## Action fidelity audit status

The separate runtime action-fidelity audit exercised Stable-Baselines3 2.9.2 `collect_rollouts` on a synthetic Box(2) environment during warm-up and after learning starts. In both captured transitions, the environment action and replay action matched (within float32 representation). This establishes the custom sampler/collector code path; the historical S5.3G replay buffer was not saved, so its arrays cannot be inspected retrospectively. See `../s5_3g_throttle_hold5_seed101_20261115/action_fidelity_runtime_audit.json`.

## Provenance and limitations

Checkpoint SHA-256: `6347ae4d82958bba89147ec39507ce3cb6abaa7e04d82f97101d1c09c2dea073`. Internal split SHA-256: `082d116f393bb1b29dea59988087b3fb0f252705eefa97f601bce7550f2740fa`. Source commit recorded in checkpoint: `4f95af934f7ce9343eabdbca19e92c64f2ecb5c7`. Per-case records, paired analysis, and full provenance are adjacent JSON files. The sample is small (8 cases), diagnostic, and should not be generalized to all TRAIN geometries. No model training was performed.
