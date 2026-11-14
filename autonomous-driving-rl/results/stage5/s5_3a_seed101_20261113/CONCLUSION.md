# S5.3A SAC pilot — seed 101

## Scope and reproducibility

This is a development experiment on an internal split of the locked TRAIN universe. It is not official VALIDATION or TEST data and must not be described as a benchmark result. The authoritative geometry manifest remained unchanged.

- Internal split: 156 `PILOT_TRAIN` and 24 `PILOT_DIAGNOSTIC`, 39/6 per tier. Diagnostic selection used split seed `5301`; fixed environment seeds use schedule seed `5302` and are shared across checkpoints.
- Split identity: `082d116f393bb1b29dea59988087b3fb0f252705eefa97f601bce7550f2740fa`.
- Training: fresh SAC initialization, root seed `101`, continuous 10,000-step run with checkpoints at 0, 5,000 and 10,000. No S5.2 smoke checkpoint, replay buffer, normalization, or reward changes were used.
- Code commit: `a567164b77370cffda3934739850015a588933cf`. Runtime: Python 3.10.21, PyTorch 2.14.1+cpu, Stable-Baselines3 2.9.2, Gymnasium 1.3.0, MetaDrive 0.4.3.
- The seed controls the configured sources of randomness; it does not guarantee bit-for-bit reproduction across simulator, dependency, or hardware changes.

## Overall diagnostic results

All 24 cases completed at each checkpoint. Every episode was allowed to run to a Platform V1 terminal outcome; no interrupted case was counted as a timeout.

| Steps | Mean route completion | Median route completion | Clean success | Safety failure | Timeout | Stationary fraction | Mean return |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00890 | 0.00835 | 0/24 | 0/24 | 24/24 | 1.000 | -0.2411 |
| 5,000 | 0.01827 | 0.01035 | 0/24 | 1/24 | 23/24 | 0.929 | -0.2674 |
| 10,000 | 0.00887 | 0.00839 | 0/24 | 0/24 | 24/24 | 1.000 | -0.2411 |

The paired mean route-completion change from 0 to 5k was `+0.00937`; from 0 to 10k it was `-0.000025`. The 5k checkpoint also had a 4.17% safety-failure rate, caused by one `OUT_OF_ROAD` episode in the Hard tier. It produced a mean moving speed of 1.66 km/h among moving steps, but remained stationary for 92.9% of all diagnostic steps. At 10k, the policy returned to 100% stationary behavior and its route completion was effectively at baseline.

Tier breakdown is in [`scientific_summary.json`](scientific_summary.json). In every tier, the 5k policy showed some route increase, while the 10k route completion returned close to the 0-step value. No checkpoint achieved clean success. The untrained and 10k policies had zero throttle and near-total braking/stationarity; 5k reduced braking and moved on some geometries, with one safety failure. Steering saturation was zero at every checkpoint.

## Training and checkpoint evidence

- The continuous run completed exactly 10,000 steps in 703.8 seconds: 14.21 steps/s and 9,000 optimizer updates.
- Seven training episodes completed on seven unique geometries. All seven were Easy/SCCS geometries; this pilot therefore sampled only a small part of the 156-geometry training pool.
- End-of-run actor loss was -15.21, critic loss 0.001125 and entropy coefficient 0.06718. These losses are descriptive and are not evidence of driving improvement.
- RSS samples ranged from 354.6 MB to 410.6 MB; final sample was 410.6 MB.
- All checkpoint bundles passed checksum verification. Model SHA-256: 0 steps `5b36aacab8203756af25679184b0bb9c78a2d44f4da48068c2461ba1ef2d10f9`; 5k `3b21afb2b12e75fbabca08d1a73d19e1b31465e875494994fa934f0f4b74275a`; 10k `c6b51e979aefaae7af65a50912f4d640fa4038a15285b815f56a3594b11f4681`.

The evaluator ran each checkpoint in a separate process, used deterministic policy inference and the shared Stage5 encoder/action adapter, preserved the same geometry/environment schedule, and confirmed unchanged policy weights, optimizer update count and replay-buffer size. Diagnostic simulator memory/latency and per-episode controls are recorded with the per-case data under the ignored run directory.

## Interpretation and next step

This pilot measured a weak, non-monotonic learning signal: route completion improved at 5k, then regressed to baseline at 10k; success remained zero, stationary collapse returned at 10k, and 5k introduced one safety failure. Do not increase the budget to 25k/50k yet. First investigate the near-zero throttle/braking behavior, why only seven Easy geometries were visited, and the Hard out-of-road episode. Any action/reward or sampler change should be a separately reviewed experiment; this report does not select a winner or claim generalization.

## Artifacts

- [`pilot_split_v1.json`](pilot_split_v1.json): exact geometry membership, source manifest hash and internal split hash.
- [`scientific_summary.json`](scientific_summary.json): checkpoint, tier, paired and training/resource aggregates.
- [`learning_curve_route_completion.csv`](learning_curve_route_completion.csv) and [`learning_curve_route_completion.svg`](learning_curve_route_completion.svg): empirical curve from these three measured checkpoints.
- Raw checkpoints and per-case JSONL: `autonomous-driving-rl/runs/stage5_pilot/s5_3a_seed101_20261113/` (gitignored; not uploaded).
