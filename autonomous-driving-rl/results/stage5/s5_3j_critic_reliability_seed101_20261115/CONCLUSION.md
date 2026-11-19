# S5.3J — Action support and critic reliability audit

Status: **COMPLETE diagnostic**. Frozen 10k SAC checkpoint only; no learning updates.

## J1 — Action support

The historical telemetry contains 10000 executed action rows and confirms marginal support: steering `-0.998..0.999`, throttle/brake `-0.998..0.994`. It does not contain state/next-state vectors, behavior log-probabilities, replay indices, or target-Q samples; replay buffer was not saved. Therefore exact historical state-conditional replay support and historical Bellman residual cannot be reconstructed.

New frozen-policy probes collected 2400 transitions across the same eight TRAIN diagnostic geometries, deterministic and stochastic modes (cap 150 steps per case/mode). Raw arrays are local under `runs/stage5_pilot/s5_3j_critic_reliability_seed101_20261115/transitions.npz`; this is a fresh diagnostic sample, not replay history.

## J2 — Soft Bellman residual on fresh transitions

Checkpoint entropy coefficient `alpha=0.067714`, `gamma=0.9900`. The target uses SB3's soft target critic and stochastic next action, with 8 next-action samples per transition. TIMEOUT truncations bootstrap (done mask 0), safety/success terminations do not, and the external cap is censored rather than converted to timeout.

| Policy mode | transitions | mean |Q1−Q2| | mean |minQ−target| | residual q50 / q90 / q99 | mean next-target MC SD |
|---|---:|---:|---:|---|---:|
| deterministic | 1200 | 0.0060 | 0.0158 | 0.0159 / 0.0218 / 0.0267 | 0.0102 |
| stochastic | 1200 | 0.0124 | 0.0225 | 0.0156 / 0.0371 / 0.1702 | 0.0114 |

These are one-step, sampled soft Bellman residuals on transitions generated now by a frozen policy. They are not residuals on the historical replay and do not prove critic miscalibration. High residuals can reflect stochastic target sampling, state/action distribution shift, or actual critic error; this probe alone does not separate those causes.

## J3 decision gate

Historical replay support is unrecoverable, but a new state/action sample was available and J2 was completed. A further controlled action intervention should proceed only if these fresh residuals/disagreements indicate a material and repeatable concern; it must use reset-and-seed parity and verify identical initial observations before comparing interventions. This audit does not itself assert an intervention result.

Read-only check: weights unchanged `True`, optimizer updates 8999 → 8999, replay size 0 → 0. No VALIDATION/TEST. Checkpoint SHA-256 `6347ae4d82958bba89147ec39507ce3cb6abaa7e04d82f97101d1c09c2dea073`, split SHA-256 `082d116f393bb1b29dea59988087b3fb0f252705eefa97f601bce7550f2740fa`.
