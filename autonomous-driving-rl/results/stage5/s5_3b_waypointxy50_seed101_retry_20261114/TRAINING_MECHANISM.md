# S5.3B mechanism diagnosis: why progress collapses

## Diagnosis

The pilot reached a **learning-signal and actor-representation failure**, not a
simulator/action-adapter wiring failure. On Easy diagnostic states, training
drives all 256 units in the actor's first ReLU layer inactive. The actor then
returns a fixed bias-driven action, so it cannot steer in response to changing
route observations. On Hard, the layer is sparse rather than fully dead and
the policy still leaves the road.

The most likely upstream driver is the combination of very sparse geometry
experience and weak useful progress feedback: the 10k-step run completed only
7 episodes on 7 Easy geometries, with 9,000 optimizer updates over that narrow
experience. Per-step progress reward was typically around `1e-5`, compared
with time costs around `1e-4`. The only moving diagnostic policy measured at
5k made progress but triggered the `-1.0` safety terminal reward on all 24
cases. SAC therefore had no demonstrated safe moving behavior to reinforce;
the constant, near-zero-action solution gave a higher return than this unsafe
policy.

This is a well-supported working diagnosis for the observed collapse, not
proof that one factor alone caused the weight changes. The run did not save
replay observations or intermediate gradients, so the exact actor update that
crossed the ReLU boundary cannot be reconstructed retrospectively.

## Direct measurements

### Training experience

- 10,000 environment steps, 9,000 optimizer updates.
- Seven completed episodes across seven Easy geometries; outcomes: six
  timeouts and one out-of-road failure.
- Mean training route completion: `0.0325`; mean episode return: `-0.3568`.
- Mean speed per 1k-step window was roughly `0.52–0.65 km/h`.
- Mean positive progress reward per step was about `1.1e-5–3.0e-5`; time cost
  was about `1.4e-4–2.3e-4` per step.
- The rollout action stream remained exploration-heavy (roughly 47% throttle
  samples and 48% brake samples overall); those are sampled actions, not a
  stable deterministic driving policy.

### Deterministic diagnostic return decomposition

RewardSpecV1 computes progress minus time cost plus a terminal outcome reward.
The measurements show why the observed unsafe motion does not persist:

| Checkpoint | Progress reward | Time cost | Safety terminal reward | Mean return |
|---:|---:|---:|---:|---:|
| 0 | +0.00889 | -0.25882 | 0 | -0.24993 |
| 5,000 | +0.08087 | -0.10073 | -1.00000 | -1.01985 |
| 10,000 | +0.00890 | -0.25882 | 0 | -0.24992 |

The stationary policy scores about `+0.77` return better than the measured
unsafe moving policy. Removing the safety terminal penalty from the 5k result
would make its return about `-0.0199`, better than the stationary return.
So the evidence does **not** show that RewardSpec intrinsically prefers zero
progress over safe driving. It shows that the only moving policy this pilot
found was unsafe, and the safety penalty correctly made that policy worse than
waiting out the episode. The learning pipeline did not discover the safe
middle ground.

### Actor collapse

On the same 766 Easy-case observations, first-layer activity averaged
`136.2/256` units in checkpoint 0 and `0/256` in both checkpoints 5k and 10k.
The 5k deterministic action was unchanged across every observation in that
episode. On the Hard trace, 5k/10k averaged only about `14/256` active units,
with no active unit on about 68% of steps; actions varied but the vehicle
still went out of road. See `ROOT_CAUSE_TRACE.md` and
`trace_activation_summary.json` for the per-case evidence.

## What is and is not the root cause

The immediate policy failure is directly established: dead/sparse first-layer
ReLU representation makes the actor state-insensitive on a meaningful part of
its diagnostic input space. A related contributing condition is established:
the training run had only seven completed, low-progress Easy episodes, while
the exploratory policy had no safe route-following behavior to reinforce.

The upstream reason actor weights move into this regime remains unresolved.
Likely contributors are the default ReLU MLP interacting with the feature
distribution, narrow replay coverage, and the relative strength/sparsity of
progress and safety feedback. It is not justified to label RewardSpecV1 alone
as the cause or to change it based on this one seed.

## Recommended handoff

Do not spend more compute repeating 10k or simply extend to 25k/50k. Before
another learning run, make the next short run observable: log actor first-layer
activation fractions and actor gradient norms at fixed intervals against a
fixed probe set, and retain the replay sample/feature-group statistics needed
to see when representation activity collapses. Then test one controlled
intervention. Keep Platform V1 metrics and contracts unchanged; any reward
change must be a separately versioned research ablation, not a silent edit.

No official VALIDATION or TEST data was used. The diagnosis comes from one
training seed and internal TRAIN diagnostic cases; it is enough to guide the
next engineering step, not to claim a general SAC result.
