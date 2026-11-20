# S5.5 Reward Sanity — Full TRAIN Trajectory Replay

**Result:** PASS for reward-accounting integrity

**Source:** `results/stage5/s5_5_meter_reward_seed101_20261120/reward_preflight.json`

**Protocol SHA-256:** `ea7bd56c112498ca6d462510624a3e4e7130dd2290231969c444da19fe7a5d99`

**Raw trace:** `runs/stage5_meter_reward/s5_5_meter_reward_seed101_20261120/preflight/reward_traces.jsonl.gz`

**Raw trace SHA-256:** `fa19c1e9bb547f4abc5ce1371fe429603e8a1b65922c026df8dfd14adcc67ec3`

## Measurement checks

The preflight replayed four actual full TRAIN trajectories on `geom_easy_SCS_seed0`, environment seed 0: one Stage 1 v0.6.1 clean reference, one historical S5.4 stationary SAC episode, and two Stage 0 random safety failures (agent random seeds 42 and 101). Stage 1 and Stage 0 data were used only to validate reward and route-progress behavior; none was copied into an SAC replay buffer.

The episode reset route progress was `p0=0.012258074154877825`, not zero. MetaDrive's runtime `navigation.total_length` was `407.89441610698384 m`, exactly matching the locked manifest length for this geometry. Thus each transition used `delta_s_m = 407.89441610698384 * (p_t - p_(t-1))`. Signed negative progress was retained. Checks covered straight movement, curve entry, road-key/block transitions, terminal steps, and a stationary episode. There was no route-progress discontinuity at road boundaries; the maximum absolute progress jump on a road-key transition in the clean trace was `0.0030643`. RewardSpecV1 step reward matched the Stage5 environment on every transition. Terminal reward appeared once on each terminal episode and never on timeout. All four traces were complete episodes, not censored probes.

## Observed returns

The discounted column uses gamma `.99` and the same step reward sequence. Reward magnitudes across arms are not directly comparable as policy quality because B and C rescale the objective.

| TRAIN trajectory | Outcome | Route progress delta | V1 return (undiscounted / discounted) | Scale return | Metre return |
|---|---|---:|---:|---:|---:|
| Stage 1 reference | Clean success | +0.975488 | +1.8901 / +0.1814 | +80.2632 / +7.7038 | +78.6289 / +7.3915 |
| S5.4 SAC 100k | Timeout, stationary | -0.000061 | -0.2501 / -0.0205 | -10.6188 / -0.8697 | -10.6187 / -0.8696 |
| Stage 0 random seed 42 | Safety failure | +0.202513 | -0.8185 / -0.2650 | -34.7585 / -11.2547 | -35.0977 / -11.4338 |
| Stage 0 random seed 101 | Safety failure | +0.199274 | -0.8212 / -0.2769 | -34.8702 / -11.7575 | -35.2041 / -11.9347 |

Both unsafe episodes received the terminal `-1` exactly once. Their total returns were lower than the clean reference for all three reward forms. The clean route earned more than the stationary policy, while reverse movement contributed negative progress rather than being clipped away.

## Interpretation and limits

Reward accounting passed the protocol's integrity checks: reset parity, signed metre conversion, route continuity, no clipping, terminal-once semantics, timeout semantics, and exact V1 parity. The trace set demonstrates that the meter form can represent forward and reverse movement in signed distance and does not, on these cases, rank a crash above a clean drive.

This is a sanity check over four trajectories on one Easy geometry, not proof that the three rewards induce the same long-horizon policy or that one is optimal. On this `407.894 m` route, Scale's positive clean return is slightly larger than Metre's because the route is shorter than the training-pool median. At exactly `424.64794675619254 m`, Scale and Metre are equal by construction. The training comparison, not this replay alone, tests whether the different signals alter SAC behavior.
