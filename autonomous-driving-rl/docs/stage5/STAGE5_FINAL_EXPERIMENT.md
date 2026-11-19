# Stage 5 final development experiment — fresh SAC tau A/B

Run: `s5_3m_tau_ab_seed101_20261119_retry1`
Protocol: `stage5_tau_ab_v1`
Training source commit: `c23f5a270114147fc39fdb508b9aed0def6d16c6`
Protocol SHA-256: `851e740b37cbb01e30085174f32153b0a483c42c086dd4997cedb6631dc3f328`

## Question and design

S5.3L reported a positive Q-to-soft-return gap on its sampled state/action pairs. That result did not show that Q bias caused the stationary deterministic policy. This bounded experiment changed only SAC's target-network Polyak coefficient:

| Parameter | Control A | Treatment B |
|---|---:|---:|
| Algorithm | SAC | SAC |
| `tau` | 0.005 | 0.001 |
| `gamma` | 0.99 | 0.99 |
| Learning rate | 0.0003 | 0.0003 |
| Seed / budget | 101 / 10,000 steps | 101 / 10,000 steps |
| Sampler | `TIER_ROUND_ROBIN` over the same 156 PILOT_TRAIN geometries | Same |
| Observation / reward | Stage5ObservationEncoder 409D; RewardSpecV1 unchanged; no normalization | Same |
| Checkpoints | 0 / 5,000 / 10,000 | 0 / 5,000 / 10,000 |

One fresh model archive was loaded by both arms. Its SHA-256 and policy/entropy fingerprint were checked. Both began at 0 steps and 0 optimizer updates with identical weights, entropy state and RNG roots. The simulator arms ran sequentially in separate processes. Training trajectories were allowed to diverge. No historic checkpoint was used to initialize either arm.

The diagnostic schedule was fixed before training: two geometry IDs per tier, with locked geometry hashes and environment seeds from the internal `PILOT_DIAGNOSTIC` split. Each checkpoint ran the same 8 cases, capped at 300 decisions per case. A case that did not terminate within that cap is `PROBE_CENSORED`; it is excluded from official timeout and success semantics.

## Training results

| Arm | Steps | Optimizer updates | Completed training episodes | Unique training geometries | Throughput | Wall time | Final RAM | Final actor loss | Final critic loss | Final entropy coefficient |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A, `tau=.005` | 10,000 | 9,000 | 5 | 6 | 17.7 steps/s | 566.0 s | 423.1 MB | -15.2422 | 0.0003666 | 0.067133 |
| B, `tau=.001` | 10,000 | 9,000 | 5 | 6 | 18.2 steps/s | 550.4 s | 422.2 MB | -4.2880 | 0.0002859 | 0.067134 |

Training geometry step counts were identical by tier in both arms: Easy 2,807; Medium 2,369; Hard 2,208; Extreme 2,616. Each arm saw six unique geometries. Clean training successes: 0 in each arm. Training action summaries were also close: positive-throttle fractions 47.86% (A) and 47.89% (B); brake fractions 47.41% and 47.43%; near-zero fractions 4.73% and 4.68%.

Checkpoint-time telemetry is embedded in each checkpoint's `metadata.json`. Final values above come from the completed run status. Loss values are descriptive training telemetry only.

## Fixed diagnostic results

All route progress below is **change from route completion at reset**, not absolute route-completion value. Speed is mean km/h. `Stationary` is the fraction of steps below 1 km/h. `Censor / safety` is the number of 8 probes; safety failures are terminal outcomes, while censoring is only the 300-step diagnostic cap.

### Overall by checkpoint

| Step | Arm | Mean route-progress delta | Median delta | Mean speed | Stationary | Mean positive throttle | Mean brake | Clean success | Safety | Censor | Official timeout |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | A | 0.000016 | 0.000018 | 0.015 | 100% | 0.000000 | 0.1799 | 0/8 | 0/8 | 8/8 | 0/8 |
| 0 | B | 0.000016 | 0.000018 | 0.015 | 100% | 0.000000 | 0.1799 | 0/8 | 0/8 | 8/8 | 0/8 |
| 5,000 | A | 0.029453 | 0.030056 | 2.541 | 22.3% | 0.018727 | 0.0000 | 0/8 | 2/8 | 6/8 | 0/8 |
| 5,000 | B | 0.029378 | 0.028905 | 2.542 | 21.4% | 0.018620 | 0.0000 | 0/8 | 2/8 | 6/8 | 0/8 |
| 10,000 | A | 0.001050 | 0.001049 | 0.084 | 100% | 0.000534 | 0.0000 | 0/8 | 0/8 | 8/8 | 0/8 |
| 10,000 | B | 0.003725 | 0.004086 | 0.297 | 100% | 0.001897 | 0.0000 | 0/8 | 0/8 | 8/8 | 0/8 |

At 5k, both arms had safety failures on the same two geometries: `geom_hard_SCTXrCS_seed13` and `geom_medium_SCTCS_seed13`. At 10k neither arm had safety failures, but all probes were still censored and neither arm had a clean success.

### 10k per-tier summary

| Tier | Arm | Mean route-progress delta | Mean speed (km/h) | Stationary | Mean positive throttle | Mean steering | Safety / censor |
|---|---|---:|---:|---:|---:|---:|---:|
| Easy | A | 0.001109 | 0.074 | 100% | 0.000460 | -0.00905 | 0 / 2 |
| Easy | B | 0.004436 | 0.295 | 100% | 0.001846 | -0.00768 | 0 / 2 |
| Medium | A | 0.001325 | 0.092 | 100% | 0.000598 | -0.00881 | 0 / 2 |
| Medium | B | 0.004299 | 0.296 | 100% | 0.001936 | -0.00767 | 0 / 2 |
| Hard | A | 0.001177 | 0.094 | 100% | 0.000597 | -0.00881 | 0 / 2 |
| Hard | B | 0.003867 | 0.305 | 100% | 0.001930 | -0.00767 | 0 / 2 |
| Extreme | A | 0.000587 | 0.075 | 100% | 0.000479 | -0.00902 | 0 / 2 |
| Extreme | B | 0.002297 | 0.292 | 100% | 0.001877 | -0.00768 | 0 / 2 |

### Paired contrasts, B minus A

| Step | Metric | Mean paired difference | Median | Cases B > A / B < A / tie |
|---:|---|---:|---:|---:|
| 0 | Route-progress delta | 0.000000 | 0.000000 | 0 / 0 / 8 |
| 5,000 | Route-progress delta | -0.000076 | 0.002469 | 6 / 2 / 0 |
| 10,000 | Route-progress delta | +0.002675 | +0.002903 | 8 / 0 / 0 |
| 10,000 | Mean speed (km/h) | +0.213306 | +0.216668 | 8 / 0 / 0 |
| 10,000 | Stationary fraction | 0.000000 | 0.000000 | 0 / 0 / 8 |

The 8 cases are fixed development geometries under one training root seed. These descriptive paired results are not a generalization claim or statistical superiority test.

## Fixed-anchor action and critic diagnostics

The same eight reset observations were captured once and reused for checkpoints. The table reports means across those anchors; stochastic statistics use 64 actor samples per anchor. Q values are critic diagnostics, **not empirical returns**.

| Step | Arm | Deterministic throttle mean | Stochastic positive-throttle fraction | Online min Q mean | Target min Q mean | Entropy coefficient |
|---:|---|---:|---:|---:|---:|---:|
| 0 | A/B | -0.178866 | 37.5% / 39.6% | -0.2414 | -0.2414 | 1.0000 |
| 5,000 | A | +0.015412 | 50.0% | 14.0692 | 13.8439 | 0.300879 |
| 5,000 | B | +0.016362 | 48.0% | 3.7197 | 3.3469 | 0.300881 |
| 10,000 | A | +0.000532 | 44.1% | 15.2383 | 15.2850 | 0.067133 |
| 10,000 | B | +0.001878 | 49.4% | 4.2181 | 4.1873 | 0.067134 |

The lower B Q scale is consistent with `tau` changing target-network dynamics, but this run did not measure Q-versus-empirical-return calibration for the A/B checkpoints. It therefore cannot establish that B corrected overestimation. It also did not fix the practically stationary 10k behavior.

## Scientific conclusion

**Observed:** the treatment lowered online and target critic Q values on the fixed anchors. At 10k, its route-progress delta was higher on all eight paired diagnostic cases, with the same observed safety count as control. Both policies still had 100% stationary fraction, no clean success, and no official timeout; all eight cases per arm were censored at the diagnostic cap.

**Supported, not proven:** slower target updates may be associated with a different critic value scale and a small increase in short-horizon route progress at this seed and schedule.

**Unknown:** whether the value-scale difference represents improved calibration; whether the route-progress difference repeats across training seeds; whether either policy can sustain useful driving over a complete episode; and the root cause of the original stationary policy.

**Decision:** retain B (`tau=.001`) as a development candidate for independent confirmation because it meets the preregistered exploratory rule of greater paired progress without worse observed safety at 10k. Keep A as the historical control reference. Neither is benchmark-eligible. Do not increase the budget within this sprint.

## Artifacts

- Registered protocol: [`configs/experiments/stage5_tau_ab_v1.json`](../../configs/experiments/stage5_tau_ab_v1.json)
- Machine-readable split, checkpoints, records, paired metrics and critic diagnostics: [`results/stage5/s5_3m_tau_ab_seed101_20261119_retry1/`](../../results/stage5/s5_3m_tau_ab_seed101_20261119_retry1/)
- Run bundles, step logs and checkpoint files: local ignored `runs/stage5_tau_ab/s5_3m_tau_ab_seed101_20261119_retry1/`; not committed
- One technical launch attempt stopped before training because the log-directory setup collided with the arm-directory guard. It is preserved under the earlier run ID and excluded from all scientific results.
