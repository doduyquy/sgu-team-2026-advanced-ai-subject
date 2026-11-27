# S5.5 Mechanism Decision

**Status:** frozen before the S5.5 three-arm training run
**Scope:** read-only review of S5.3M/S5.4 evidence; no historical checkpoint was changed.

## Direct measurements reviewed

The six S5.3M checkpoints at 0, 5k and 10k for both tau arms passed the repository checkpoint-integrity verifier. Their model SHA-256 values are:

| Arm | 0 steps | 5k steps | 10k steps |
|---|---|---|---|
| A, tau 0.005 | `6def9818dc4ca4d97a597bc6479da7768f9b994706b9884be1550709bf8f65a0` | `ffc104bd942d0843623aabaa3e51459e7bfd1620f8ba27caa57b716a3d6bebf6` | `d074977e99852c796e46c7fbc1a284c1a382265e6d501e9af2812b5b53be0442` |
| B, tau 0.001 | `d431c9565ca90b2265c74b4a42b772d80866edbf14a424826aec96b13fb6c11e` | `0884cc368ab54fe06e7602d995c6b9575f0d38eb45762a91a6b586e76f13300a` | `a50d3c8dd755b81172be1a952e5bf1d91756842e9e84d86fd1fffce46ba1398c` |

The existing S5.4 analysis used eight fixed, real reset observations (`anchor_file_sha256=7a96ebd959681d239d120402cbb30840b209746002f9beff46de5dd64af567d2`). Across those anchors, deterministic action standard deviation was:

| Checkpoint | Steering std | Throttle/brake std | First-layer nonpositive fraction |
|---|---:|---:|---:|
| Shared 0-step | 0.02857 | 0.04569 | 48.73% |
| A, 5k | 0.000296 | 0.003234 | 95.51% |
| A, 10k | 0.000205 | 0.000117 | 99.61% |
| B, 5k | 0.003734 | 0.001174 | 94.38% |
| B, 10k | 0.000011 | 0.000071 | 99.61% |

This is direct evidence of near-constant actions on this small reset-anchor set, not proof that every replay state was collapsed. The historical S5.4 full episodes were stationary at 100k for both the raw and group-wise-normalized input arms.

## Observation-scale evidence and limits

In the eight-anchor S5.4 audit, raw task-waypoint coordinates (60D) ranged from about -0.084 to 50 and had mean per-feature across-anchor standard deviation 0.5067. The 259D core ranged 0 to 1 with corresponding mean standard deviation 0.00114. Traffic actor and traffic context values were all zero on these no-traffic Easy anchors. These ranges motivate careful input scaling, but the audit did not establish that scale mismatch caused collapse. Zero traffic values also say nothing about traffic avoidance.

The S5.4 transform was already tested as a fixed treatment against the raw observation arm through 100k, without clean successes or movement. S5.5 therefore holds the same versioned `pilot_groupwise_physical_v1` transform constant in every arm; it does not retest normalization or claim normalization solved the problem.

Other relevant historical measurements: alpha at S5.3M 10k was approximately 0.067 in both arms; the S5.3L mean critic Q minus estimated soft return was +6.135 over 96 TRAIN-only rollouts. That local Q gap does not establish that Q bias caused stationary behavior. The old replay buffers are unavailable, so the exact data distributions cannot be reconstructed.

## Frozen hypothesis and intervention

**Hypothesis:** fractional route progress in RewardSpecV1 may provide a weak or geometry-length-dependent learning signal at the observed early movement scale. A global multiplication of the whole reward and a signed progress reward measured in route metres are distinct alternatives; comparing all three on identical fresh SAC settings separates the general scale effect (Scale versus V1) from the metre parameterization (Metre versus Scale) within this bounded screening experiment.

**Single intervention family selected:** change only the training reward signal, in three pre-registered arms:

- A: unchanged V1 reward.
- B: V1 reward multiplied by the frozen median-route scale `k`.
- C: signed route-metre progress divided by 10, with terminal/time terms scaled by the same `k`.

The exact formulas, median route length, `k`, pools, seeds, and source revision are frozen in `results/stage5/s5_5_meter_reward_seed101_20261120/protocol.json`. Platform `RewardSpecV1`, observation contract, action contract, SAC architecture and hyperparameters remain unchanged. No teacher trajectory or demonstration transition enters SAC replay.

## What remains uncertain

- Whether any arm creates meaningful moving-state coverage in replay.
- Whether a reward change preserves safe steering and produces clean driving rather than only faster movement.
- Whether an outcome repeats across training roots; this sprint uses one root seed.
- Whether recurrent/history inputs are necessary. Existing current observations include ego motion/context, and a policy that fails to use them is not evidence that temporal memory is required.

The evidence and choice above are exploratory. S5.5 outcomes cannot establish general SAC learnability or benchmark eligibility.
