# S5.3G — throttle persistence pilot result

## Question and intervention

Does reducing throttle sign jitter during SAC experience collection help the **deterministic learned policy** move? A fresh SAC model was trained for 10,000 steps with seed 101. At collection time, throttle was held within global 5-step collection blocks while steering was sampled every step. Throttle was resampled at episode boundaries; a block immediately after a boundary could therefore be shorter than 5 steps. RewardSpecV1, gamma 0.99, architecture, TRAIN-only split and tier-round-robin sampler were unchanged.

The collector changed both the action sent to MetaDrive and the normalized action stored in SB3 replay. Runtime training telemetry verifies **zero action mismatches across 2,007 five-step blocks**. No action adapter or benchmark contract was changed.

## Action fidelity audit

The suspected replay mismatch is **not present in the S5.3G code path**. `ThrottlePersistenceSAC._sample_action()` changes the normalized `buffer_actions` first, then reconstructs the environment `actions` from that same array. SB3 passes those actions to `env.step()` and subsequently passes the same `buffer_actions` to `_store_transition()` / `replay_buffer.add()`. The Stage5 environment adapter validates and forwards the canonical continuous action without substituting another throttle value.

I ran the actual SB3 2.9.2 `collect_rollouts()` path using the S5.3G sampler and a small synthetic Gymnasium Box(2) environment. In both uniform warm-up and post-`learning_starts`, the requested throttle differed from the held throttle, while the action received by the environment exactly matched the action stored in replay; steering was unchanged. The run output is in `action_fidelity_runtime_audit.json`.

The historical S5.3G replay buffer was not saved, so its arrays cannot be inspected retrospectively. The runtime probe validates the same implementation and SB3 collection path; it does not reconstruct that deleted in-memory buffer. The persistence state influences which action the behavior sampler chooses, but it is outside the environment dynamics: each `env.step()` receives the explicit executed action, so replay transitions still represent the ordinary environment transition for that state and action.

## Result

**This intervention did not solve the stationary deterministic policy.** After 10k steps, all 24 fixed diagnostic episodes timed out; none succeeded or had a safety failure. Stationary fraction remained 100%, moving-speed mean was undefined because no steps reached 1 km/h, and mean positive throttle at inference was 0. Mean brake magnitude was 0.0918, versus 0.0382 for the baseline checkpoint.

| Deterministic 10k metric | Baseline | Hold throttle 5 steps |
|---|---:|---:|
| Completed diagnostic cases | 24 | 24 |
| Clean success | 0/24 | 0/24 |
| Safety failures | 0/24 | 0/24 |
| Timeouts | 24/24 | 24/24 |
| Stationary fraction | 100.0% | 100.0% |
| Mean throttle | 0.0000 | 0.0000 |
| Mean brake magnitude | 0.0382 | 0.0918 |
| Mean absolute final route completion | 0.008888 | 0.008911 |

Both evaluations use the same 24 geometry/environment-seed pairs. Absolute route completion includes the nonzero reset value, so the relevant comparison is paired: the intervention's mean final-route difference was **+0.00002245** (17 cases higher, 7 lower; median +0.00005032). The corresponding mean reset-relative progress reconstructed from timeout reward was 0.00008834, versus 0.00006589 for baseline. This small numerical difference did not translate into movement or success.

## Did the intervention actually reduce jitter?

Yes, in the training actions. After warm-up, direct throttle/brake sign changes fell from **461.2** to **91.1 per 1,000 steps**. Median positive-throttle run length rose from **1** to **5 steps**. Thus the intervention changed the intended temporal property, but the deterministic evaluation did not improve.

Training yielded 10 completed episodes on 11 unique TRAIN geometries: **8 OUT_OF_ROAD and 2 TIMEOUT**, with no clean success. After warm-up, mean logged speed rose from 0.584 to 1.281 km/h and cumulative route progress across the 9,000 logged steps rose from 0.0979 to 0.4212. The historical baseline recorded 6 completed episodes (5 TIMEOUT, 1 OUT_OF_ROAD) on 7 unique geometries; the intervention recorded 9 post-warm-up terminal episodes (2 TIMEOUT, 7 OUT_OF_ROAD). These episode counts and geometry exposures differ, so treat the training contrast as descriptive. It shows the intervention generated more movement and progress, but the added movement often ended off-road.

## What happened during training

The intervention did change the collected experience: it reduced throttle sign flips and produced higher speed and route progress. But this did not turn into a safer deterministic actor. In the intervention run, 7 of the 9 post-warm-up terminal episodes were OUT_OF_ROAD; the baseline had 1 OUT_OF_ROAD among 6 completed episodes. The sample sizes and geometry sequence differ, so this is a warning signal rather than a clean estimate of the safety effect.

## Interpretation

This is evidence against **throttle sign jitter alone being a sufficient explanation or fix**. It is not proof that temporal exploration has no role: steering remained stochastic every step, the intervention sometimes held braking or near-zero throttle too, no successful trajectory was produced, and only one training seed was used. The increase in training out-of-road outcomes also means this intervention may have generated longer unsafe action segments rather than useful controlled acceleration.

The root cause of SAC's learned near-zero deterministic throttle remains unresolved. We have identified a more specific mechanism: per-step throttle jitter suppresses motion in collected experience, but reducing it alone produces more movement alongside many more off-road endings. SAC still selects near-zero throttle deterministically, plausibly because the additional movement was not safe enough to provide consistently favorable long-term value. This last explanation remains a hypothesis, not a proven critic diagnosis. The controlled result is that throttle persistence reduced measured sign flips by about 80% and increased logged progress, yet left the deterministic policy stationary. Continuing the same 5-step treatment or increasing its training budget is not justified by this result.

## Integrity and scope

- Training completed: 10,000/10,000 steps; 9,000 optimizer updates; throughput 11.52 steps/s; peak/final RAM about 424 MB.
- Checkpoints 0, 5k and 10k were saved and integrity-verified. Checkpoint SHA-256 values are in `provenance.json`.
- Full deterministic diagnostic completed for the 10k checkpoint (24/24 cases), with evaluation isolation checks passing.
- The separate full 0-step evaluation was stopped after 7/24 cases to limit cost; its partial records remain under `runs/` and are not represented as complete. 5k was not evaluated in this sprint.
- Internal TRAIN diagnostic only; no VALIDATION or TEST cases were used. This is not an official benchmark result.
- Run record status is `TRAINING_COMPLETE_DIAGNOSTICS_INCOMPLETE`: the original all-checkpoints diagnostic suite was intentionally interrupted. Training and the separate 10k evaluation are individually complete.
- Single seed and current runtime; seed alone does not imply bit-for-bit reproducibility.

## Artifacts

- Raw run and checkpoints: `runs/stage5_pilot/s5_3g_throttle_hold5_seed101_20261115/`
- Paired analysis: `paired_analysis.json`
- Provenance: `provenance.json`
