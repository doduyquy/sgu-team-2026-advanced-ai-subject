# S5.3L — Soft-Q Calibration replicate

Status: **COMPLETE**. Frozen SAC checkpoint; no training updates.

At each of four fixed internal TRAIN diagnostic reset states, 512 stochastic SAC actions were sampled. Three throttle values near conditional q15/q50/q85 were selected for brake / near-neutral / throttle. Initial steering was fixed to the deterministic actor output in all conditions. After this single intervention, the same frozen stochastic SAC policy sampled both action dimensions at every decision until a real Platform episode outcome/horizon.

Reset observations were checked for exact array equality before each condition. Only exact-parity triplets are included in paired contrasts. No external step cap was used; any incomplete episode is explicitly censored.

| Tier | Condition | Initial throttle | Q1 / Q2 / minQ | Steps | Outcome | Route progress delta | Discounted reward return | Discounted soft return | Speed km/h |
|---|---|---:|---|---:|---|---:|---:|---:|---:|
| Easy | brake_q15 | -0.753 | 15.234 / 15.240 / 15.234 | 1524 | TIMEOUT | +0.0125 | -0.0152 | +8.9965 | 0.48 |
| Easy | near_neutral_q50 | -0.080 | 15.226 / 15.231 / 15.226 | 1524 | TIMEOUT | +0.0125 | -0.0152 | +8.9965 | 0.48 |
| Easy | throttle_q85 | +0.675 | 15.216 / 15.224 / 15.216 | 1524 | TIMEOUT | +0.0126 | -0.0152 | +8.9971 | 0.48 |
| Medium | brake_q15 | -0.729 | 15.234 / 15.239 / 15.234 | 1604 | TIMEOUT | +0.0215 | -0.0140 | +9.1754 | 0.49 |
| Medium | near_neutral_q50 | -0.018 | 15.225 / 15.231 / 15.225 | 1604 | TIMEOUT | +0.0215 | -0.0140 | +9.1754 | 0.49 |
| Medium | throttle_q85 | +0.685 | 15.216 / 15.224 / 15.216 | 1604 | TIMEOUT | +0.0215 | -0.0140 | +9.1746 | 0.49 |
| Hard | brake_q15 | -0.787 | 15.272 / 15.288 / 15.272 | 2103 | TIMEOUT | +0.0207 | -0.0108 | +9.0662 | 0.52 |
| Hard | near_neutral_q50 | -0.059 | 15.261 / 15.277 / 15.261 | 2103 | TIMEOUT | +0.0206 | -0.0108 | +9.0661 | 0.52 |
| Hard | throttle_q85 | +0.659 | 15.251 / 15.265 / 15.251 | 2103 | TIMEOUT | +0.0205 | -0.0109 | +9.0652 | 0.52 |
| Extreme | brake_q15 | -0.748 | 15.226 / 15.230 / 15.226 | 3188 | TIMEOUT | +0.0170 | -0.0076 | +9.2375 | 0.50 |
| Extreme | near_neutral_q50 | -0.049 | 15.217 / 15.222 / 15.217 | 3188 | TIMEOUT | +0.0170 | -0.0076 | +9.2375 | 0.50 |
| Extreme | throttle_q85 | +0.658 | 15.208 / 15.215 / 15.208 | 3188 | TIMEOUT | +0.0170 | -0.0076 | +9.2378 | 0.50 |

## Paired differences

Positive values favor the left condition. n is the number of exact-reset-parity geometry pairs.

| Metric | Contrast | n | Mean | Median | Higher / lower / tie |
|---|---|---:|---:|---:|---|
| route_progress_delta | near_neutral_q50_minus_brake_q15 | 4 | +0.0000 | +0.0000 | 2 / 2 / 0 |
| route_progress_delta | throttle_q85_minus_brake_q15 | 4 | +0.0000 | +0.0000 | 3 / 1 / 0 |
| route_progress_delta | throttle_q85_minus_near_neutral_q50 | 4 | -0.0000 | -0.0000 | 1 / 3 / 0 |
| discounted_reward_return_observed | near_neutral_q50_minus_brake_q15 | 4 | +0.0000 | -0.0000 | 1 / 3 / 0 |
| discounted_reward_return_observed | throttle_q85_minus_brake_q15 | 4 | +0.0000 | +0.0000 | 2 / 2 / 0 |
| discounted_reward_return_observed | throttle_q85_minus_near_neutral_q50 | 4 | +0.0000 | +0.0000 | 2 / 2 / 0 |
| discounted_soft_return_observed | near_neutral_q50_minus_brake_q15 | 4 | -0.0000 | -0.0000 | 1 / 3 / 0 |
| discounted_soft_return_observed | throttle_q85_minus_brake_q15 | 4 | -0.0002 | -0.0003 | 2 / 2 / 0 |
| discounted_soft_return_observed | throttle_q85_minus_near_neutral_q50 | 4 | -0.0002 | -0.0003 | 2 / 2 / 0 |

## Interpretation limits

The bootstrapped timeout soft return uses the target critic and is reported separately from the realized episode return; it is not independent empirical evidence for critic calibration. Initial Q1/Q2 are critic predictions. One paired seed per geometry is a small diagnostic screen, not a statistical conclusion. If any row is PROBE_CENSORED, its observed return is partial and excluded from full-episode claims.

Exact reset parity: 4/4 case triplets. Policy weights unchanged: `True`; optimizer updates 8999 → 8999. No VALIDATION/TEST access. Checkpoint SHA-256 `6347ae4d82958bba89147ec39507ce3cb6abaa7e04d82f97101d1c09c2dea073`; split SHA-256 `082d116f393bb1b29dea59988087b3fb0f252705eefa97f601bce7550f2740fa`.
