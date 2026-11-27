# S5.3F — Deterministic vs stochastic policy diagnostic

## Scope

Read-only diagnostic of the 10,000-step SAC checkpoint from run `s5_3d_tierrr_gamma099_seed101_20261115`. It used two fixed geometries from the internal TRAIN-only `PILOT_DIAGNOSTIC` schedule, each with the same environment seed across policy modes:

| Geometry | Tier | Environment seed | Geometry SHA-256 prefix |
|---|---|---:|---|
| `geom_easy_SCCS_seed19` | Easy | 259818 | `e30180e8f3da` |
| `geom_extreme_CrXROSTR_seed0` | Extreme | 968985 | `7c392abe573c` |

For each geometry, one deterministic episode was paired with five full-horizon stochastic episodes. Two 25-step forced-throttle probes checked the physical action path; they are technical probes, not complete episodes. No VALIDATION or TEST cases were used.

## Measured episodes

`route delta` is final route completion minus route completion at reset. Values below are fractions (for example, `0.02185` means 2.185 percentage points of route completion).

| Geometry | Mode | n | Outcomes | Mean route delta | Mean speed km/h | Stationary fraction | Mean episode return |
|---|---|---:|---|---:|---:|---:|---:|
| Easy | Deterministic | 1 | 1 timeout | 0.00013 | 0.013 | 100.0% | -0.2499 |
| Easy | Stochastic | 5 | 5 timeouts | 0.02185 | 0.598 | 83.4% | -0.2281 |
| Extreme | Deterministic | 1 | 1 timeout | 0.00006 | 0.013 | 100.0% | -0.2499 |
| Extreme | Stochastic | 5 | 4 timeouts, 1 `OUT_OF_ROAD` | 0.01661 | 0.479 | 87.3% | -0.4246 |

There were no clean successes. Stochastic actions created measurable progress on both geometries, but motion remained intermittent. The Extreme stochastic group had one safety failure out of five; that episode's return was -1.1904. These small, fixed-case samples diagnose behavior and do not estimate generalization or benchmark rates.

## Action distribution

At the Easy reset observation, 2,048 stochastic actions were sampled from the same checkpoint. The deterministic action was `[-0.04088, -0.03826]`; the stochastic throttle/brake mean was `-0.03756`, median `-0.05173`, and 47.3% of samples were positive. The 5th/95th percentiles were `-0.897` and `+0.878`; 23.6% of samples exceeded `+0.5`.

The actor's Gaussian standard deviation before `tanh` is about `0.885`; the measured standard deviation of the transformed throttle/brake actions is `0.583`. This corrects an earlier shorthand that called `0.885` the action standard deviation. The transformed policy still has a broad distribution.

This also refines the earlier 64-sample probe that happened to show 59.4% positive throttle. The larger sample indicates that was ordinary sampling variation: the deterministic action is close to the distribution's center, while its broad tails produce occasional strong throttle actions.

Across the ten full stochastic episodes, 42.6% of steps had throttle/brake action above `+0.05`. Mean speed and route progress were higher than under deterministic inference, but stationary fractions remained 80.6–95.1% by episode.

## Temporal action pattern

Using `action[1] > +0.05` for throttle and `< -0.05` for braking, the 23,000 S5.3F stochastic steps had 42.6% throttle, 53.2% braking, and 4.2% near-neutral actions. The median consecutive throttle run was **one step**; the mean run was 1.84 steps, with a maximum of 12. Direct switches between positive throttle and braking occurred about **425 times per 1,000 steps**.

The S5.3D per-step training action log shows the same pattern after warm-up: over 9,000 steps, 47.0% throttle, 48.5% braking, a one-step median throttle run, and about **461 direct sign switches per 1,000 steps**. The telemetry logs actions, speed, route deltas, reward components, and geometry, but it does not contain the saved replay-buffer transitions or observations/next observations. This supports analysis of action/reward associations, not reconstruction of exactly which transitions each gradient update sampled.

The frequent throttle/brake switching is consistent with short control pulses that fail to maintain acceleration. It is descriptive evidence, not a causal isolation: the state changes also affect the action distribution.

The 25-step forced-throttle probes also worked on both geometries:

- Easy: speed rose from 0.74 to 18.58 km/h; route delta was `+0.01240`.
- Extreme: speed rose from 0.71 to 17.79 km/h; route delta was `+0.00568`.

This rules out a basic simulator/action-adapter inability to move the vehicle on these cases.

## Critic observations

Twin-critic values were sampled at reset and after 25 stochastic steps. Holding steering equal to the deterministic actor's steering, both critics monotonically lowered their predicted Q as throttle increased from the deterministic value near `-0.038` to `+0.7`. The change was about `-0.00325` on both geometries and both sampled states.

This is a concrete candidate mechanism: at these states the critic slightly favors near-zero throttle. It is not proof the critic is wrong. Q is a learned estimate, the action grid is sparse, and a fixed action's value is not the same experiment as a full stochastic rollout. The Easy episodes are suggestive of a mismatch: stochastic rollouts improved route delta and return without a safety failure, while the local Q grid still preferred less throttle. The Extreme result is less clear because one stochastic rollout went out of road.

For a proper calibration, compare against SAC's **entropy-regularized soft return**, not only raw environment return: SB3's SAC target includes `-alpha * log_prob` in the bootstrap term. Match the timeout/truncation bootstrap behavior too. See the pinned [SB3 2.9.2 SAC update](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.2/stable_baselines3/sac/sac.py) and [ReplayBuffer timeout handling](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.2/stable_baselines3/common/buffers.py).

## Conclusion

The diagnostic does **not** show a deterministic/stochastic inference bug. The deterministic action is near the stochastic distribution's center. The policy's learned center is near zero throttle; high action variance sometimes supplies enough throttle to move the vehicle. Those stochastic trajectories make small progress, remain stationary most of the time, and carry safety risk on Extreme.

The next research question is now narrower: why does the critic/actor center continue to favor near-zero throttle when stochastic motion can improve Easy-case return, and is that preference justified by the safety risk on harder geometry? The Q observations make critic value calibration a stronger lead, but do not yet establish a faulty critic or prove gamma is the cause. Do not treat these results as justification to switch benchmark inference to stochastic mode or to increase the training budget yet.

## Provenance and artifacts

- Training checkpoint SHA-256: `c52793eda0b75afb27c0934d1a0c1eae14c3c5399fd233c0a1989876cdb28140`.
- Internal split SHA-256: `082d116f393bb1b29dea59988087b3fb0f252705eefa97f601bce7550f2740fa`.
- Training code commit: `4f95af934f7ce9343eabdbca19e92c64f2ecb5c7`.
- Policy checksum and optimizer update count were unchanged by evaluation (`8999` before and after); replay buffer size was zero.
- Raw episode summaries and per-step traces: `runs/stage5_pilot/s5_3f_stochastic_policy_diag_seed101_20261115_v3/` (gitignored).
- Temporal action analysis: `runs/stage5_pilot/s5_3f_stochastic_policy_diag_seed101_20261115_v3/temporal_action_analysis.json`.
- Diagnostic runner: [`diagnose_stage5_sac_policy.py`](../../../scripts/diagnose_stage5_sac_policy.py), SHA-256 recorded in [`provenance.json`](provenance.json).
- The first two diagnostic attempts were stopped for performance and contain no episodes; they are marked `ABORTED.json` and were excluded from all results.
