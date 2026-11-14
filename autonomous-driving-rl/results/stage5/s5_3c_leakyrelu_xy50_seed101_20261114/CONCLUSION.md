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

Training completed 10,000 steps with 9,000 optimizer updates. Seven episodes completed, and an eighth geometry had started by step 10,000. The step log contains transitions from eight unique geometries; all eight were Easy. The configured sampler is `ROUND_ROBIN`, and its canonical ordering puts all 39 PILOT_TRAIN Easy geometries before the other tiers. The 10k pilot therefore never sampled Medium, Hard, or Extreme. This is a confirmed pilot coverage defect, though not by itself proof of why the actor became stationary. Throughput was about 16.7 steps/s, training wall time about 599 seconds, and final RAM about 412 MB.

### Important correction: discounted objective versus episode return

The episode returns above are undiscounted diagnostics. They cannot be used directly to claim that SAC prefers the stationary policy, because SAC optimizes a discounted objective and the configured `gamma` is 0.99. The Stable-Baselines3 SAC documentation defines `gamma` as the discount factor. At `gamma=0.99`, a reward 780 steps away has weight 0.000394; at steps 1,524 and 1,860 its weights are approximately 0.000000223 and 0.0000000076. At 10 Hz, the usual effective horizon `1/(1-gamma)` is about 100 decisions, or 10 seconds, while these route-aware episodes last about 152–186 seconds.

Reconstructing per-step reward from the stored traces confirms the distinction. On `geom_easy_SCCS_seed19`, the 5k unsafe trajectory has an undiscounted return of about -1.035 but a gamma-discounted return of about -0.01375. The stationary 0-step and 10k trajectories have discounted returns around -0.01640 and -0.01639. Thus the 5k unsafe trajectory is actually slightly better under this narrow discounted calculation, despite its much worse episode return and safety outcome. This small difference does not predict SAC's learned policy, but it disproves my earlier explanation that the raw return comparison alone shows SAC has a reward incentive to stop.

The strong, confirmed concern is that terminal safety penalties and success bonuses are almost invisible to the configured SAC objective at the observed episode lengths. That can undermine safety learning and delayed-success credit assignment. It does not by itself explain why the 10k actor became stationary; the traces establish the collapse, not its unique cause. The other confirmed issue is training coverage: only eight Easy geometries were sampled and no other tier was reached. Replay-data diversity, Q-target magnitudes, and reward/entropy contributions still need a controlled diagnosis before naming a single cause for the stationary collapse.

## Decision

Do not extend this exact setup directly to 25k/50k steps. Two separate controlled checks are warranted before that: (1) fix the pilot sampler to draw from all four TRAIN tiers and hold the SAC hyperparameters fixed; (2) independently test a larger `gamma` while holding the sampler and all other settings fixed, since the current discount makes terminal outcomes negligible over the actual episode lengths. Keep RewardSpecV1 unchanged. Record discounted-return and Q-target diagnostics, not only episode return. These experiments would test whether broader replay coverage and longer credit assignment improve behavior; neither result should be assumed in advance. This internal diagnostic split is development data and must not be presented as official evaluation.

## Artifacts

- Aggregate metrics and paired comparisons: `scientific_summary.json`
- Empirical route learning curve: `learning_curve_route_completion.svg` and `.csv`
- Fixed split and protocol: `pilot_split_v1.json`, `pilot_protocol_v1.json`
- Full run, per-episode records, checkpoints, and trace probes: ignored local directory `runs/stage5_pilot/s5_3c_leakyrelu_xy50_seed101_20261114/`
