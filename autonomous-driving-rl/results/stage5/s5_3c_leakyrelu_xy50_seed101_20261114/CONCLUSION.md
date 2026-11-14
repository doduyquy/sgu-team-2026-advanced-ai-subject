# S5.3C LeakyReLU pilot: findings

## Scope and provenance

This is an internal development diagnostic on the fixed 24-geometry `PILOT_DIAGNOSTIC` subset of TRAIN. It is not official VALIDATION or TEST. The run used seed 101, the fixed waypoint x/y divided-by-50 m transform, RewardSpecV1, and 10,000 continuous SAC environment steps. The only change from the S5.3B waypoint-scale run was replacing ReLU with LeakyReLU(0.01). The 0, 5k, and 10k checkpoints were all evaluated against the same 24-case schedule; all 72 episodes completed and none was interrupted.

- Code commit: `cbd85ab3dc4b2d55ef25bb3c72e5c749de4ae71c`
- Internal split SHA-256: `082d116f393bb1b29dea59988087b3fb0f252705eefa97f601bce7550f2740fa`
- Observation transform fingerprint: `5aef4b88da2a9384f82cc2a23a61ef8e1df41f711dbe75ac9d7e0e673e4447e1`
- Checkpoint SHA-256: 0 steps `8cd4f658b4e7da9c91054deec0afb4af2268c534d6ae2391421f02283c8e9f2f`; 5k `5fb23a31f3b9d739657631d4b6a6beb0c1a6fd0c1b6eaeb3e1dea3108e01537e`; 10k `758202fe0f7ba695e46229d0d3fc8cd4437499a5b3f3bcdd8d2489a55f8c0bd7`.
- Runtime: Windows 10, Python 3.10.21, MetaDrive 0.4.3, PyTorch 2.14.1 CPU.

The evaluator summary's `protocol` field names the parent waypoint-transform evaluator (`stage5_pilot_waypoint_xy50_ablation_v1`). The activation-specific identity is recorded in this run's `pilot_protocol.json` and checkpoint metadata as `LeakyReLU`, negative slope 0.01.

## Measured results

| Training steps | Mean route completion | Clean success | Safety failure | Timeout | Mean return | Stationary fraction | Mean speed while moving |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00891 | 0/24 | 0/24 | 24/24 | -0.2499 | 1.000 | — |
| 5,000 | 0.09081 | 0/24 | 23/24 | 1/24 | -1.0098 | 0.344 | 2.54 km/h |
| 10,000 | 0.00892 | 0/24 | 0/24 | 24/24 | -0.2499 | 1.000 | — |

The 5k checkpoint made more route progress by moving, but nearly every diagnostic episode ended in a safety failure. By 10k, the policy had returned to stationary behavior. The mean route completion at 10k was effectively unchanged from the untrained policy. No checkpoint achieved clean success.

The preceding ReLU run showed the same pattern: route completion 0.00889 / 0.08087 / 0.00890 and stationary fraction 1.000 / 0.319 / 1.000 at 0 / 5k / 10k. Its 5k checkpoint had 24/24 safety failures; this LeakyReLU run had 23/24. The single-factor activation change therefore did not materially alter the learning dynamics.

## Trace diagnosis and likely mechanism

Deterministic traces were collected on the same Easy geometry `geom_easy_SCCS_seed19` and Hard geometry `geom_hard_SCXOCS_seed0` for all three checkpoints. At 10k, each trace contained one distinct deterministic action after rounding actions to four decimals. The first actor layer had 256 units; every unit's preactivation was negative at every recorded step in both traces. LeakyReLU kept those units' outputs non-zero, so this is not literal dead-ReLU behavior. The actor nevertheless produced an effectively constant action and the car remained stationary. At 5k, activity was also strongly skewed negative, especially on the Easy trace.

Training completed 10,000 steps with 9,000 optimizer updates, but only seven episodes and seven unique geometries were completed. All seven were Easy geometries. The configured sampler is `ROUND_ROBIN`; because the pilot ends before cycling through the ordered 156-geometry pool, it never reaches the other tiers in this run. Throughput was about 16.7 steps/s, training wall time about 599 seconds, and final RAM about 412 MB.

The measured reward/outcome shift supports a likely mechanism: the moving 5k policy gains route progress but usually incurs the RewardSpecV1 terminal safety penalty, producing mean return around -1.01; the stationary 10k policy avoids safety failure and times out with return around -0.25. SAC can therefore prefer the locally less costly stationary behavior over unsafe movement. This is consistent with the observed collapse, but the pilot does not prove that reward preference is the only cause. Limited experience diversity (seven Easy geometries), unsafe exploratory behavior, and a policy whose first-layer responses become almost entirely negative are interacting candidate causes.

## Decision

Do not extend this exact setup directly to 25k/50k steps. The activation-only hypothesis has been tested and does not resolve the behavior. The next informative experiment should first correct training-case coverage so short pilots encounter all four TRAIN tiers (for example, a controlled tier-balanced sampler), while keeping RewardSpecV1, architecture, observation transform, seed, and budget otherwise fixed. Inspect whether early replay data contains safe-moving trajectories before drawing conclusions about the reward contract. Do not change reward or architecture together with the sampler; this internal diagnostic split is development data and must not be presented as official evaluation.

## Artifacts

- Aggregate metrics and paired comparisons: `scientific_summary.json`
- Empirical route learning curve: `learning_curve_route_completion.svg` and `.csv`
- Fixed split and protocol: `pilot_split_v1.json`, `pilot_protocol_v1.json`
- Full run, per-episode records, checkpoints, and trace probes: ignored local directory `runs/stage5_pilot/s5_3c_leakyrelu_xy50_seed101_20261114/`
