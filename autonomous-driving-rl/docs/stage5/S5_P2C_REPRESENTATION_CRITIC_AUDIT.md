# Stage 5 P2-C — Representation & Critic Audit

**Decision: `REPRESENTATION_INTERVENTION_JUSTIFIED`** for one controlled actor-gradient-flow experiment. This identifies a measured proximate mechanism for policy state-insensitivity; it does not establish the unique cause of SAC stationary behavior or show that changing activation will produce clean driving.

## Scope and provenance

This was a read-only audit. No SAC training or VALIDATION/TEST access occurred. RewardSpecV1, 409D AgentInputV1, `continuous_box2_v1`, actor/critic weights, and the P2 experiment were not modified.

The historical S5.5 checkpoints in P2-A remain interpreted using their recorded `legacy_v1` TaskContext producer. This additional audit uses the fresh P2 A/B checkpoints, both verified with `reference_route_station_proposal_v1`, fingerprint `a7df3942bb874b40ea16474be43c7b7a48796646c9d59cf61da5d71ba5291725`, and `pilot_groupwise_physical_v1`, fingerprint `265103242e3ba3ad6f8c493d8f0bad39644c375a2696dbcdad4857052796797e`.

- P2 run: `s5_p2_correlated_exploration_seed101_20261121_retry1`
- Training source commit: `9234e50635b9d25093816c159ea948d0acacb41a`
- Protocol SHA-256: `6a54151d0a3e52bcd237811f0b9a7d3912adadb512615e92eab71281f7564652`
- P2 result summary SHA-256: `e2548a297953ee9b608b975a356b7def97e9dc2401bdc0bfa41dc7584c1d7c50`
- Eight model checkpoints were loaded through the repository integrity loader; the A/B 10k replay buffers were loaded and verified against their checkpoint metadata.

The audit constructed 201 fixed anchors from actual saved observations on the six locked internal TRAIN diagnostic geometries. The same anchor array was evaluated against A/B at 0, 2k, 5k and 10k. It includes observations from both the short 2k movement probes and 10k full episodes. These anchors are not synthetic perturbations.

## Actor representation across checkpoints

| Arm/checkpoint | Deterministic steering SD | Deterministic throttle SD | First ReLU nonpositive | Neurons never active / 256 | Offline gradient norm through first linear layer |
|---|---:|---:|---:|---:|---:|
| A/B 0 | 0.002221 | 0.002423 | 46.7% | 115 | 0.3431 |
| A 2k | 0.000253 | 0.000630 | 95.7% | 244 | 0.1243 |
| B 2k | 0.000890 | 0.000253 | 96.1% | 245 | 0.1206 |
| A 5k | 0 | 0 | 100% | 256 | 0 |
| B 5k | 0 | 0 | 100% | 256 | 0 |
| A 10k | 0 | 0 | 100% | 256 | 0 |
| B 10k | 0 | 0 | 100% | 256 | 0 |

The zero first-layer gradient is expected when all first-layer ReLUs are inactive on the inputs being differentiated. Later layers and output heads still had nonzero offline gradients, but these gradients could not restore sensitivity through the inactive first layer on those states. The deterministic means at 10k were near neutral with mild braking (A: steering `+0.0015`, throttle/brake `−0.0253`; B: `+0.0108`, `−0.0153`), so this was not action `tanh` saturation at the bounds.

At 2k, 72 anchors had a normalized ego-speed feature above 0.02. The first layer already had 244/245 neurons never active on that subset. At 5k and 10k all 256 were inactive on both the higher-speed and lower-speed anchor subsets. This was not limited to stationary reset observations.

### Direct replay-state confirmation at 10k

The replay buffers contain 10,000 real transitions per arm. The first actor layer was inactive for all 10,000 replay observations in both arms, including the moving strata (1,680 A and 1,693 B transitions with measured speed at least 1 km/h), positive-progress transitions, and the sparse safety-failure transitions. Deterministic actor-action standard deviation on replay states was effectively zero.

The actual replay includes meaningful task-context variation: 4,351 A and 4,179 B transitions have a route turn-context score of at least 0.05 rad; more than 8,100 per arm have first-valid-waypoint lateral coordinate magnitude at least 0.25 m. The 10k deterministic actor remains constant on both subsets. This confirms lack of response on the observed states, but does not label what the correct steering action should have been.

The fixed diagnostic-anchor set itself was nearly straight (maximum turn-context score 0.0253 rad). The richer replay-state analysis avoids using that limited probe set as the only evidence about route geometry.

## Inputs and feature groups

Across the 10k replay observations, feature variation was:

| Input group | Nonconstant dimensions | Mean per-feature SD A | Mean per-feature SD B |
|---|---:|---:|---:|
| Ego 9D | 9/9 | 0.1126 | 0.1213 |
| Navigation 10D | 4/10 | 0.0280 | 0.0323 |
| LiDAR 240D | 0/240 | 0 | 0 |
| Traffic 66D | 0/66 | 0 | 0 |
| Task 84D | 63/84 | 0.0315 | 0.0465 |

LiDAR and Traffic features had zero observed variation in this Easy/no-traffic experiment. Therefore:

- The actor’s loss of response to Ego, Navigation and Task is directly measurable on actual varying inputs.
- LiDAR and Traffic usefulness is **UNKNOWN** here; their constant values do not show that the policy ignores useful LiDAR or traffic information.
- Although Task waypoints vary substantially, this experiment did not exercise traffic interaction and is not evidence of traffic-driving competence.

The fixed-anchor Jacobian, scaled by one empirical feature standard deviation, was exactly zero across all measured varying groups at 5k/10k. It was small but nonzero at initialization and 2k. The 201 anchors span six geometries and are identical across all eight checkpoint evaluations.

## Optimization and alpha

The P2 periodic training telemetry (median of recorded samples in the preceding 500 environment steps) shows a similar trajectory in both arms:

| Checkpoint | A actor grad norm | B actor grad norm | A critic grad norm | B critic grad norm | Entropy grad norm A/B | Alpha A/B |
|---:|---:|---:|---:|---:|---:|---:|
| 2k | 0.1931 | 0.1875 | 19.24 | 19.07 | 3.366 / 3.366 | 0.7873 / 0.7873 |
| 5k | 0.0509 | 0.0524 | 40.21 | 39.38 | 3.363 / 3.364 | 0.3201 / 0.3201 |
| 10k | 0.0117 | 0.0141 | 42.93 | 44.23 | 3.361 / 3.361 | 0.07142 / 0.07142 |

This supports that actor updates weakened while entropy-gradient magnitude stayed similar and critic gradient norms increased. Alpha decline alone does not explain the collapse. These are periodic aggregate logs, not per-update layer gradient histories. Checkpoint/offline snapshots bracket the loss of sensitivity between 2k and 5k; they cannot establish whether action insensitivity happened before the ReLU units died.

## Critic and replay audit

The replay action trace aligned with the 10k replay buffer for **all 10,000** actions, rewards, termination flags and timeout flags in both arms: zero mismatches. Timeout sampling used SB3’s replay-buffer sample path, which applies its timeout mask before bootstrapping the soft Bellman target.

| Replay group | A transitions | B transitions |
|---|---:|---:|
| Speed < 1 km/h | 8,320 (83.2%) | 8,307 (83.1%) |
| Speed ≥ 1 km/h | 1,680 (16.8%) | 1,693 (16.9%) |
| Positive route delta | 5,895 (59.0%) | 5,855 (58.6%) |
| Safety-failure transitions | 2 | 2 |

Both critics retain active units, but their hidden representations are sparse on replay states: about 173–190 of 256 units per hidden layer never activate across the observed replay batch. At recorded replay actions, average absolute local Q derivatives are small (A: Q1/Q2 throttle 0.00237/0.00276, steering 0.000303/0.000061; B: throttle 0.00562/0.00537, steering 0.00448/0.00458). The safety-failure transitions have large mean absolute TD errors (about 15.6–15.7), but there are only two per arm.

These results do not establish critic miscalibration or a wrong action ranking. Actor deterministic actions are at an average L2 distance of about 0.78–0.79 from the single recorded behavior action at the same replay state. That is not evidence that the actor’s alternative action is well supported. Replay buffers from 2k/5k were not saved; their state-action coverage, Q ranking and critic gradients remain **UNKNOWN**.

## Evidence classification

### CONFIRMED

- The actor’s first ReLU layer is fully inactive on the measured fixed TRAIN anchors by 5k and on all saved 10k replay states, including moving and route-turn contexts.
- Actor deterministic output Jacobians and offline gradients through the first layer are zero at 5k/10k on these real observations.
- The replay contains some movement and positive progress but is dominated by low speed; action/reward/terminal trace fidelity is exact.
- P2’s periodic actor-gradient norm declines considerably from 2k to 10k, while entropy-gradient magnitude remains nearly unchanged.

### SUPPORTED_HYPOTHESIS

- Dead first-layer ReLUs are a proximate mechanism blocking observation-dependent actor gradients on the sampled state distribution. This can explain why the deterministic policy becomes constant, but does not establish why optimization enters that regime or prove it is the only reason the car is stationary.
- Critic action sensitivity and state representation may also be weak. The measured critic evidence is insufficient to choose critic intervention first because alternative actor actions are not established as in-support and terminal failures are sparse.

### UNKNOWN

- Whether loss of actor sensitivity precedes or follows ReLU inactivity during the 2k–5k interval.
- Whether critic action rankings are correct for alternative supported actions; Q is not ground truth.
- Why the actor’s first layer becomes inactive, and whether a different activation alone will prevent policy collapse.
- Whether LiDAR or Traffic features help driving in environments where those features vary.
- Whether the stationary policy is initiated by representation, replay/action coverage, reward/entropy interaction, critic estimation, or a combination.

## Prior exploration intervention

The historical S5.3G throttle-hold test reduced throttle/brake sign switching and increased movement in training collection, but deterministic evaluation remained stationary and training had frequent out-of-road endings. P2’s OU treatment separately achieved the registered noise autocorrelation but also failed to retain deterministic sensitivity. Neither result shows that exploration is the only cause. P2’s treatment noise was temporally correlated; executed/replay action fidelity was exact.

## Decision and one proposed follow-up

`REPRESENTATION_INTERVENTION_JUSTIFIED` is a narrow priority decision, not proof of root cause. If a new experiment is separately authorized, compare P2 Control A against one fresh treatment that changes **actor latent activation only** from ReLU to LeakyReLU with slope 0.01. Keep critic, RewardSpecV1, 409D transform and corrected producer, action contract, pool, seed and 10k budget fixed. Copy compatible initial weights and report the activation difference as the only treatment. Track per-layer activation/gradient, fixed-anchor sensitivity, actual replay-state sensitivity and full-episode driving; activation recovery alone is not success. Stop at the registered 10k ceiling. S5.3C previously reported near-constant actions with LeakyReLU, so this is a controlled recheck under the corrected P2 protocol, not an expectation of success. No training is authorized by this audit.

## Reproduction and evidence

From `autonomous-driving-rl/` in `metadrive_sb3`:

```powershell
conda run -n metadrive_sb3 python scripts/audit_stage5_p2c_representation_critic.py
```

Machine-readable evidence: [`evidence.json`](../../results/stage5/p2c_representation_critic_audit_20261121/evidence.json), SHA-256 `489aa942d39806f5dfc33c48c6b0fcbaa7ea23dab7e687590df387be78dd00d1`.

The fixed observation matrix is preserved at [`fixed_train_anchors.npy`](../../results/stage5/p2c_representation_critic_audit_20261121/fixed_train_anchors.npy), SHA-256 `6ded06b27deec488dac1952b8a4a7d22755b8bbbca0d774854d53fe58ac6f13a`. The evidence includes checkpoint, replay, trace, evaluation-state, protocol and audit-script hashes. Large checkpoint bundles and full replay buffers remain local under the ignored `runs/` directory; they are not added to Git.
