# S5.3I — Policy mechanism audit

Status: **COMPLETE**. Read-only; no training run or frozen contract changes.

## I-1: action evolution across checkpoints

All three checkpoints were queried on the same eight reset observations from the internal TRAIN diagnostic schedule; each used 64 stochastic samples per observation.

| Checkpoint | Mean deterministic throttle | Median | Positive fraction | Stochastic throttle mean / SD | Stochastic steering SD | Q-grid argmax throttle (zero / actor steering) | ΔminQ(-1 minus +1), actor steering |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -0.117 | -0.152 | 0.000 | -0.071 / 0.590 | 0.427 | +1.00 / +1.00 | -0.0030 |
| 5000 | +0.001 | +0.001 | 0.500 | -0.002 / 0.607 | 0.576 | -1.00 / -1.00 | +0.0122 |
| 10000 | -0.093 | -0.091 | 0.000 | -0.030 / 0.583 | 0.582 | -1.00 / -1.00 | +0.0251 |

Q-grid argmax is the critic's predicted min(Q1,Q2) averaged across the eight observations, not empirical return. At 5k and 10k the average critic grid slopes toward braking at both fixed-zero and actor steering. The endpoint Q difference is small relative to the Q level and the grid includes actions that may be out of the replay distribution. The step-0 critic is randomly initialized and has no useful calibration interpretation.

## I-2: critic throttle preference

The complete per-observation Q1/Q2/min-Q grid is `critic_throttle_grid.jsonl`. It evaluates nine throttle values from -1 to +1 at zero steering and at each checkpoint's deterministic actor steering. Q is learned prediction, not ground truth. The missing 10k replay buffer prevents checking whether its data covered useful moving trajectories or calibrating Q against actual returns.

## I-3: reward and horizon audit

Audited 10000 recorded transitions across 11 logged episode segments. Maximum absolute row-level residual for `total = progress - time_cost + terminal` was 3.99e-08; maximum residual for `time_cost = 0.25 / route-aware horizon` was 0.

Positive route-progress transitions: 70.8%. Progress exceeded the per-step time cost on 15.0%; total step reward was positive on 15.0%. Summed terminal reward: -8.000. These are descriptive statistics from the existing pilot logs.

### Reward by action sign (descriptive, confounded)

| Action sign | Transitions | Mean progress reward | Mean time cost | Sum terminal reward | Mean total reward | P(progress > 0) | P(total reward > 0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| throttle | 4699 | +0.000091 | 0.000129 | -6.0 | -0.001315 | 97.7% | 27.8% |
| brake | 5301 | +0.000011 | 0.000126 | -2.0 | -0.000492 | 47.0% | 3.7% |

The pooled reversal is arithmetically explained: positive-throttle rows have a larger mean progress reward, but carry more terminal penalty (6 of the eight `-1` penalties versus 2 under brake). Because that penalty is assigned to the final transition, it lowers the pooled total reward for the throttle group. This association does not prove throttle caused the failures: action groups occur in different states, trajectories, and episode contexts.

Throttle versus brake: mean progress reward +0.000091 vs +0.000011; summed terminal reward -6.0 vs -2.0; mean total reward -0.001315 vs -0.000492.

Episode-level logged reward-only returns under gamma .99 and .9991 are included in `reward_horizon_audit.json`. They omit SAC entropy terms and bootstrap behavior, so they are not critic targets.

## Conclusion and limits

This audit can show whether deterministic throttle changed between the 0/5k/10k checkpoints and what the existing critic predicts across a throttle grid. It cannot prove why the actor changed: the replay buffer was not saved, only one seed is available, and Q values have not been calibrated against matched empirical returns. Reward decomposition matches the implemented formula on logged transitions, so this audit found no row-level reward-accounting mismatch in this run. Do not infer reward hacking, critic error, or a confirmed gamma cause from these data.

Checkpoint SHA-256: 0 `6d36f5969583f11515358b142919934150bfbf10adb204eb358bde142a94f522`, 5k `b6ac4bd4c85dbf17e69a8bdc9a6b1dcf19fed87d0a2a70f0981aa86c006fa4a7`, 10k `6347ae4d82958bba89147ec39507ce3cb6abaa7e04d82f97101d1c09c2dea073`. Internal split SHA-256: `082d116f393bb1b29dea59988087b3fb0f252705eefa97f601bce7550f2740fa`. Source SHA-256 values are in `source_hashes.json`. No VALIDATION/TEST access.
