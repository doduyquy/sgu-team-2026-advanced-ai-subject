# S5.5 P2 — Correlated Exploration Results

## Decision

**`NO_IMPROVEMENT` for the registered OU exploration treatment at 10,000 steps per arm.** The treatment produced the intended temporal correlation in the exploration noise, but did not preserve a useful deterministic driving policy. Both fresh SAC arms briefly moved at the 2k checkpoint, then returned to effectively stationary behavior by 5k and remained stationary at 10k. Neither arm had a clean success.

This result does not establish that SAC cannot learn to drive, nor does it identify a unique root cause.

## Frozen comparison

| Item | Control A | Treatment B |
|---|---|---|
| Training reward | RewardSpecV1 | RewardSpecV1 |
| Exploration | Standard SB3 SAC collection | `ou_ar1_v1`, added to both action dimensions after learning starts |
| OU definition | — | AR(1) coefficient 0.85; stationary standard deviation 0.15; innovation standard deviation 0.07905694; reset at episode end |
| Initialization | Same fresh actor/critics and entropy state | Same initialization fingerprint |
| Root seed / training pool | 101 / same 8 Easy `PILOT_TRAIN` geometries | Same |
| Evaluation | Same 6 fixed, disjoint Easy `PILOT_DIAGNOSTIC` cases | Same |
| Budget | 10,000 actual environment steps | 10,000 actual environment steps |

Both arms used the 409D observation, `continuous_box2_v1`, the same observation transform and corrected F09 producer. No VALIDATION or TEST data were accessed. Benchmark eligibility remains false.

## Frozen deterministic evaluation

The four probe rows below use the same six diagnostic cases and a 300-decision cap. All six cases are `PROBE_CENSORED`; they are not episode timeouts or safety successes. The 10k full-episode row runs to each case's actual horizon and classifies terminal outcomes.

| Steps | Arm | Mean speed km/h | Mean route progress delta | Stationary fraction | Clean | Safety failure | Timeout | Censored probes | Actor action std (steering, throttle) | First-layer nonpositive fraction |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | A/B | 0.015 | 0.000026 | 1.000 | 0 | 0 | — | 6/6 | 0.001946, 0.002484 | 0.466 |
| 2,000 | A | 3.437 | 0.070007 | 0.145 | 0 | 0 | — | 6/6 | 0.000244, 0.000630 | 0.956 |
| 2,000 | B | 3.402 | 0.069179 | 0.147 | 0 | 0 | — | 6/6 | 0.000807, 0.000247 | 0.961 |
| 5,000 | A | 0.284 | 0.005777 | 1.000 | 0 | 0 | — | 6/6 | 0, 0 | 1.000 |
| 5,000 | B | 0.014 | 0.000039 | 1.000 | 0 | 0 | — | 6/6 | ~0, ~0 | 1.000 |
| 10,000 probe | A | 0.014 | 0.000026 | 1.000 | 0 | 0 | — | 6/6 | 0, ~0 | 1.000 |
| 10,000 probe | B | 0.014 | 0.000029 | 1.000 | 0 | 0 | — | 6/6 | ~0, ~0 | 1.000 |
| 10,000 full | A | 0.013 | 0.000075 | 1.000 | 0 | 0 | 6/6 | — | 0, ~0 | 1.000 |
| 10,000 full | B | 0.013 | 0.000081 | 1.000 | 0 | 0 | 6/6 | — | ~0, ~0 | 1.000 |

At 10k, the paired mean B−A route-progress difference over the six cases was **+0.0000063**, effectively negligible at the observed scale. At 2k it was −0.000828, and at 5k it was −0.005738. These are descriptive paired differences from one training seed, not inferential statistics.

## Did OU change exploration?

Yes, at the noise-process level. Across B's 9,000 post-warmup noisy actions, measured OU noise lag-1 autocorrelation was 0.844 globally and 0.839 averaged within episodes, close to the registered 0.85 target. The noise reset eight times at episode boundaries. There were 1,028 clipped action components and zero requested/executed/replay action mismatches.

The replay behavior changed much less:

| Replay measure | A | B |
|---|---:|---:|
| Positive throttle fraction | 0.4886 | 0.4872 |
| Direct throttle/brake sign switches | 5,016 | 4,910 |
| Median positive-throttle run | 1 decision | 1 decision |
| Mean positive-throttle run | 1.945 decisions | 1.981 decisions |
| Executed throttle lag-1 autocorrelation within episodes | 0.0054 | 0.0540 |

Thus the treatment made the added noise temporally correlated, but did not produce long positive-throttle sequences or improve the learned deterministic policy in this run.

## Training and critic telemetry

Each arm completed 10,000 environment steps and 9,000 optimizer updates, visited all 8 training geometries, and completed 8 episodes: 6 `TIMEOUT`, 2 `OUT_OF_ROAD`, 0 clean success. One episode was partial at the 10k boundary. Throughput was 18.22 steps/s for A and 18.29 for B. Peak sampled RSS was 437 MB for A and 763 MB for B; runs were sequential in one process context, so this is not an independent memory comparison.

At the fixed recent replay sample at 10k, mean absolute TD error was 0.0570 (A) and 0.0220 (B); mean twin-Q gaps were 0.00656 and 0.01178, respectively. The corresponding online Q and target-Q diagnostics are in the machine-readable summary. These sample diagnostics do not establish critic calibration or explain stationary behavior. Entropy coefficients were nearly identical: 0.067201 (A), 0.067203 (B).

## Interpretation

- **Confirmed:** OU noise autocorrelation matched its target; action fidelity was exact in the recorded transitions; both arms covered all eight training geometries.
- **Observed:** both policies showed substantial movement on short probes at 2k, followed by near-zero action sensitivity and stationary behavior at 5k/10k. The full 10k diagnostics were six timeouts per arm.
- **Not demonstrated:** improvement in state-dependent deterministic control, repeatable clean driving, or a causal explanation for the collapse.
- **Not concluded:** that correlated exploration is ineffective in general. This one-seed, 10k comparison tests this particular OU process and setup only.

## Provenance and reproducibility

- Run: `runs/stage5_p2/s5_p2_correlated_exploration_seed101_20261121_retry1/` (gitignored; retain locally).
- Frozen protocol: [`S5_P2_PROTOCOL.md`](S5_P2_PROTOCOL.md); protocol SHA-256 `6a54151d0a3e52bcd237811f0b9a7d3912adadb512615e92eab71281f7564652`.
- Training source commit: `9234e50635b9d25093816c159ea948d0acacb41a`.
- Compact evidence: [`result_summary.json`](../../results/stage5/p2_correlated_exploration_seed101_20261121/result_summary.json), SHA-256 `e2548a297953ee9b608b975a356b7def97e9dc2401bdc0bfa41dc7584c1d7c50`.
- The summary records per-case results, paired deltas, checkpoint/model/replay hashes, evaluation hashes, geometry identities and run provenance. Raw checkpoint bundles and replay buffers are not committed.
