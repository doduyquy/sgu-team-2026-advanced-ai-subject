# S5.3L — Soft-Q Calibration (L1)

Status: **COMPLETE** — 96/96 full episodes, 8/8 stochastic continuation replicates.

Checkpoint SHA-256 6347ae4d82958bba89147ec39507ce3cb6abaa7e04d82f97101d1c09c2dea073; pilot split SHA-256 082d116f393bb1b29dea59988087b3fb0f252705eefa97f601bce7550f2740fa; gamma 0.99; frozen alpha 0.06771417; SB3 2.9.2; Gymnasium 1.3.0.

Outcomes: TIMEOUT=96. Censored=0.

Mean online min(Q1,Q2) = **15.232**; mean full soft-return estimate = **9.097**; mean Q-return gap = **+6.135** (pooled SD 0.104, range +5.957 to +6.336).

## Per fixed reset state

95% t intervals use eight independent stochastic continuations (df=7); they describe variation conditional on one fixed geometry/environment reset, not across maps.

| Tier | Geometry | Initial action | n | minQ | Soft return | Q-return gap | Gap 95% CI |
|---|---|---|---:|---:|---:|---:|---:|
| Easy | geom_easy_SCCS_seed19 | brake_q15 | 8 | 15.234 | 9.064 | +6.170 | [+6.086, +6.253] |
| Easy | geom_easy_SCCS_seed19 | near_neutral_q50 | 8 | 15.226 | 9.064 | +6.161 | [+6.078, +6.245] |
| Easy | geom_easy_SCCS_seed19 | throttle_q85 | 8 | 15.216 | 9.064 | +6.152 | [+6.069, +6.235] |
| Extreme | geom_extreme_CrXROSTR_seed0 | brake_q15 | 8 | 15.226 | 9.113 | +6.112 | [+6.014, +6.211] |
| Extreme | geom_extreme_CrXROSTR_seed0 | near_neutral_q50 | 8 | 15.217 | 9.113 | +6.104 | [+6.005, +6.203] |
| Extreme | geom_extreme_CrXROSTR_seed0 | throttle_q85 | 8 | 15.208 | 9.113 | +6.095 | [+5.996, +6.194] |
| Hard | geom_hard_SCTXrCS_seed13 | brake_q15 | 8 | 15.272 | 9.072 | +6.200 | [+6.122, +6.278] |
| Hard | geom_hard_SCTXrCS_seed13 | near_neutral_q50 | 8 | 15.261 | 9.072 | +6.189 | [+6.111, +6.267] |
| Hard | geom_hard_SCTXrCS_seed13 | throttle_q85 | 8 | 15.251 | 9.074 | +6.177 | [+6.100, +6.253] |
| Medium | geom_medium_SCTCS_seed11 | brake_q15 | 8 | 15.234 | 9.139 | +6.095 | [+6.020, +6.169] |
| Medium | geom_medium_SCTCS_seed11 | near_neutral_q50 | 8 | 15.225 | 9.139 | +6.086 | [+6.012, +6.160] |
| Medium | geom_medium_SCTCS_seed11 | throttle_q85 | 8 | 15.216 | 9.139 | +6.077 | [+6.003, +6.151] |

## Paired contrasts

Positive values favor the condition on the left. Pairing is by geometry, environment seed and stochastic continuation seed.

| Metric | Contrast | n | Mean | Median | 95% CI | + / - / tie |
|---|---|---:|---:|---:|---|---|
| soft_return_for_analysis | near_neutral_q50_minus_brake_q15 | 32 | -0.000016 | -0.000016 | [-0.000031, -0.000001] | 11 / 21 / 0 |
| soft_return_for_analysis | throttle_q85_minus_brake_q15 | 32 | +0.000554 | +0.000207 | [-0.000100, +0.001207] | 23 / 9 / 0 |
| soft_return_for_analysis | throttle_q85_minus_near_neutral_q50 | 32 | +0.000570 | +0.000218 | [-0.000080, +0.001220] | 22 / 10 / 0 |
| discounted_reward_return_observed | near_neutral_q50_minus_brake_q15 | 32 | +0.000000 | -0.000000 | [-0.000000, +0.000000] | 16 / 16 / 0 |
| discounted_reward_return_observed | throttle_q85_minus_brake_q15 | 32 | +0.000017 | +0.000019 | [-0.000023, +0.000056] | 20 / 12 / 0 |
| discounted_reward_return_observed | throttle_q85_minus_near_neutral_q50 | 32 | +0.000017 | +0.000019 | [-0.000023, +0.000056] | 20 / 12 / 0 |
| route_progress_delta | near_neutral_q50_minus_brake_q15 | 32 | -0.000011 | -0.000001 | [-0.000079, +0.000057] | 12 / 20 / 0 |
| route_progress_delta | throttle_q85_minus_brake_q15 | 32 | +0.000104 | +0.000062 | [-0.000011, +0.000219] | 22 / 10 / 0 |
| route_progress_delta | throttle_q85_minus_near_neutral_q50 | 32 | +0.000115 | +0.000059 | [-0.000016, +0.000246] | 21 / 11 / 0 |

## Interpretation

L0 validated SB3 timeout/replay/soft-return semantics and found the timeout bootstrap tail negligible at these horizons. The positive Q-return gap persists across stochastic continuations: all 12 geometry-condition 95% intervals are above zero. This is evidence of local absolute critic overestimation/miscalibration on the tested state-action pairs, not a confirmed cause of the stationary policy. Relative action ranking is unresolved: mean throttle-minus-brake Q difference across four geometries is -0.0186, while the paired soft-return contrast is +0.000554 with CI [-0.000100, +0.001207], which includes zero. The data therefore do not establish that the critic chooses the wrong action ranking. L1 cannot attribute the absolute gap to critic approximation, changing alpha/policy during training, or replay coverage.

No training, replay writes, normalization updates, VALIDATION or TEST occurred. Full per-replicate hashes are in L1_ANALYSIS.json; run manifest is run_manifest.json.
