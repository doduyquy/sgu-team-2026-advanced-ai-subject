# Stage 5 P2-D — Actor-Only Activation Results

**Decision:** `REPRESENTATION_RECOVERED_NO_DRIVING_GAIN` (qualified: LeakyReLU preserves a nonzero derivative path, but useful observation sensitivity was not maintained and no safe driving skill emerged).
**Scope:** one fresh seed, TRAIN-only, development diagnostic; `benchmark_eligible=false`.
**Machine evidence:** [`result_summary.json`](../../results/stage5/p2d_actor_activation_seed101_20261121/result_summary.json), SHA-256 `d12f0e913cd8e4dd442e1caae8d6e41d1922095c4b752304573ad7728b19100b`.
**Layer-gradient evidence:** [`posthoc_layer_gradient_audit.json`](../../results/stage5/p2d_actor_activation_seed101_20261121/posthoc_layer_gradient_audit.json), SHA-256 `410dbad6b44cf45f585653898b873d9647a5698a8ebcfcc2fcf13ffd8db420a5`.
**Protocol:** [`S5_P2D_ACTOR_ACTIVATION_PROTOCOL.md`](S5_P2D_ACTOR_ACTIVATION_PROTOCOL.md), machine protocol SHA-256 `92f6180d3217f8133d214d929fc1315eb1c24cf538b4acbad8ddd92c060bacb8`.

## What was compared

Both fresh SAC arms used root seed 101, the same eight Easy `PILOT_TRAIN` geometries, same six disjoint Easy `PILOT_DIAGNOSTIC` geometries, 409D input, F09 producer `reference_route_station_proposal_v1`, `pilot_groupwise_physical_v1`, Box2 actions, RewardSpecV1, MLP 256×256, and unchanged SAC hyperparameters. Both completed 10,000 environment steps and 9,000 optimizer updates. Training was sequential on CPU; wall time was A 551.8 s and B 465.4 s.

| Arm | Actor | Critic / target critic | Training episodes | Geometries visited | Training episode outcomes | Action fidelity |
|---|---|---|---:|---:|---|---:|
| A | ReLU | ReLU | 8 | 8/8 | 6 TIMEOUT, 2 OUT_OF_ROAD | 0 mismatches |
| B | LeakyReLU(0.01) | ReLU | 7 | 8/8 | 7 TIMEOUT | 0 mismatches |

The treatment changed only the actor hidden activation. Initial parameter names, shapes, tensor values, critic weights and entropy state matched. Since the nonlinearity differs, checkpoint-0 actor outputs were measured, not assumed equal. The policy class and activation fingerprint were verified on save/load for every checkpoint. The 10k replay buffers and compact replay-state samples at 2k/5k were preserved with hashes.

Training exploration was nearly identical as expected: throttle was positive on 48.86% of A and 48.95% of B transitions; each arm had 5,016 direct throttle/brake sign switches, median positive-throttle run length 1 decision, and within-episode throttle lag-1 correlation near 0.006. The registered change was not an exploration treatment.

## Deterministic 300-step probes

All probes reached the 300-decision cap without a terminal event: each entry is `6 PROBE_CENSORED`, not six timeouts or safe completions.

| Steps | Arm | Mean speed km/h | Mean route progress delta | Stationary fraction | Clean | Safety failures | Deterministic action SD: steering / throttle |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0 | A | 0.015 | 0.000026 | 1.000 | 0 | 0 | 0.00222 / 0.00242 |
| 0 | B | 0.015 | 0.000029 | 1.000 | 0 | 0 | 0.00227 / 0.00235 |
| 2k | A | 3.437 | 0.07001 | 0.145 | 0 | 0 | 0.000253 / 0.000630 |
| 2k | B | 3.294 | 0.06701 | 0.150 | 0 | 0 | 0.000089 / 0.000312 |
| 5k | A | 0.284 | 0.00578 | 1.000 | 0 | 0 | 0 / 0 |
| 5k | B | 0.383 | 0.00777 | 1.000 | 0 | 0 | 0.000011 / 0.000045 |
| 10k | A | 0.014 | 0.000026 | 1.000 | 0 | 0 | 0 / 0 |
| 10k | B | 0.797 | 0.01620 | 0.628 | 0 | 0 | 0.000005 / 0.000021 |

The 10k paired probe mean route-progress difference was B−A `+0.01618` across the same six cases. This short capped signal did not expose terminal safety outcomes; it cannot be interpreted as safe completion.

## Full deterministic episodes at 10k

| Arm | Outcomes on six cases | Mean speed | Mean route-progress delta | Mean stationary fraction | Clean successes |
|---|---|---:|---:|---:|---:|
| A | 6 TIMEOUT | 0.0134 km/h | 0.000075 | 1.000 | 0/6 |
| B | 6 OUT_OF_ROAD | 2.363 km/h | 0.14678 | 0.228 | 0/6 |

The paired full-episode route-progress difference was B−A `+0.14670`; mean speed difference was `+2.3496 km/h`. Every B run ended in a safety failure, while every A run timed out stationary. Therefore B created more movement and route progress before failure, but this is not an improvement in driving competence. Do not describe the B result as safer or successful.

## Actor representation and gradients

The fixed 201 P2-C TRAIN anchors were unchanged across checkpoints. ReLU control A had 46.7% nonpositive first-layer preactivations initially, 95.7% at 2k, and 100% at 5k and 10k. Its first-layer derivative fraction and deterministic action Jacobian were zero at 5k/10k; all 256 units were inactive on these anchors.

Treatment B had zero derivative fraction throughout because LeakyReLU's negative-side derivative is 0.01. At 5k and 10k all first-layer preactivations were nonpositive, but the negative path remained differentiable. On the fixed anchors its first-linear-layer combined actor-objective gradient norm was `0.001158` at 5k and `0.000199` at 10k; A was zero at both checkpoints. This is direct evidence that the treatment kept a gradient path open, not evidence that the actor learned useful controls.

Practical action sensitivity still shrank sharply. B's deterministic action SD fell from about `0.0023` at initialization to `5.2e-6` steering and `2.1e-5` throttle at 10k. Its mean input Jacobian fell from about `0.055` per action to about `0.00033`. At 2k B's action SD was already lower than A's. The first-layer gradient remained nonzero but small; the policy was close to constant over the fixed anchors.

At 10k the auto entropy coefficient was `0.06720` for A and `0.06720` for B. Recent-replay Q1/Q2 and TD diagnostics are included in checkpoint metadata and the machine summary; those estimates do not establish critic calibration or correct action ranking.

For context, the fixed recent-replay snapshot at 10k gave mean Q1/Q2 `14.773/14.780`, mean absolute TD error `0.0570`, and twin-Q gap `0.00656` for A; B gave `14.852/14.853`, `0.01245`, and `0.01463`. These are internal critic diagnostics on each arm's own replay sample, not paired ground-truth value comparisons.

## Relation to S5.3C

S5.3C used LeakyReLU with the default 0.01 slope, but passed it through SB3's policy-wide `activation_fn`, affecting both actor and critic. It also used `pilot_waypoint_xy_div50m_v1`, not the group-wise transform used here. Its 10k full diagnostic reported 24/24 TIMEOUT, stationary fraction 1.0, zero mean throttle and no clean success. P2-D is the first matched actor-only check under the corrected P2 input pipeline. It again found no clean success, but the B policy produced movement followed by safety failures. These runs are not a direct paired comparison because they differ in protocol, geometry schedule and architecture scope.

## Interpretation

- **Confirmed:** the actor-only LeakyReLU policy reconstructs correctly, critic remains ReLU, weights/shapes align, replay actions match execution, and LeakyReLU gives nonzero first-layer derivatives on negative preactivations.
- **Confirmed:** B did not retain substantial deterministic observation sensitivity through 10k; its outputs were nearly constant on the fixed anchors.
- **Confirmed:** neither arm had a clean success. A was stationary in full evaluation; B failed out of road on all six full diagnostic episodes.
- **Not established:** whether the activation change caused the B full-episode failure pattern across other seeds, whether LeakyReLU improves optimization generally, or why the actor/objective converged to these policies.
- **Decision:** `REPRESENTATION_RECOVERED_NO_DRIVING_GAIN`, qualified as derivative-path recovery without practical policy sensitivity or safe driving gain. One seed/six cases is not a general conclusion.

Raw training/checkpoint/replay artifacts remain local under the ignored `runs/stage5_p2d/` directory. Their checkpoint and file hashes are recorded in the machine summary; cloning Git alone does not reproduce the binary bundles.
