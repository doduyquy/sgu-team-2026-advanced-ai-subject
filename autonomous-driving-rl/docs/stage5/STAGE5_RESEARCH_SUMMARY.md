# Stage 5 research summary for group presentation

## Objective and method

Stage 5 tests model-free reinforcement learning for continuous vehicle control in MetaDrive under Platform V1 observation, action, reward, safety and evaluation contracts. The current baseline uses Stable-Baselines3 Soft Actor-Critic (SAC), a 409D `AgentInputV1` encoder and continuous action adapter `continuous_box2_v1`.

The completed development pilot partitioned the locked 180 TRAIN geometries into 156 `PILOT_TRAIN` and 24 `PILOT_DIAGNOSTIC` geometries. S5.3M trained two fresh, paired-config SAC arms from seed 101 for 10,000 environment steps: control `tau=.005`, treatment `tau=.001`. Only target-network update coefficient `tau` changed. Eight fixed internal TRAIN diagnostic cases were evaluated at checkpoints 0, 5k and 10k.

## Observed result

At 10k, the lower-tau arm had mean reset-relative route progress `0.003725`, versus `0.001050` for control. The paired difference was `+0.002675` on all 8 cases. Treatment also averaged `0.297 km/h` versus `0.084 km/h` for control. Both remained below 1 km/h throughout all diagnostic decisions, had zero clean successes, zero observed safety failures at 10k and 8/8 `PROBE_CENSORED` probes. Censoring is the 300-step diagnostic cap and is neither a canonical timeout nor success.

## What the evidence supports

- MetaDrive training and the SAC update/checkpoint/inference pipeline execute end to end.
- Changing `tau` changed the critic Q scale on fixed reset observations and coincided with a small, paired increase in route progress in this one-seed pilot.
- Neither checkpoint demonstrated useful clean driving performance at 10k.
- Lower Q alone is not critic calibration; the A/B did not estimate empirical soft return for its checkpoints.
- The findings are limited to one training root and eight internal development cases. They do not establish generalization, statistical superiority, safety, or a root cause for stationary behavior.

## Current decision

Keep `tau=.005` as the reference baseline. Keep `tau=.001` only as a development candidate for a separately approved independent-root confirmation. Stage 5 remains CANDIDATE and benchmark-ineligible. No canonical VALIDATION or TEST run was performed. Before official evaluation, the project needs an approved Stage 5 checkpoint-binding extension, independent candidate confirmation, protocol lock and scientific sign-off.

Detailed per-tier values and hashes: [final experiment report](STAGE5_FINAL_EXPERIMENT.md). GUI instructions: [comparison guide](STAGE5_COMPARISON_GUIDE.md).
