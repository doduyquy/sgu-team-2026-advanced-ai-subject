# S5.5 Protocol — Reward Scale and Route-Metre Signal

**Status:** completed, TRAIN-only development experiment

**Run:** `s5_5_meter_reward_seed101_20261120`

**Protocol SHA-256:** `ea7bd56c112498ca6d462510624a3e4e7130dd2290231969c444da19fe7a5d99`

**Training source revision:** `1baaeedf9bd9d69f4507d5ff9b27b75dcc9d5639`

## Question and preregistered comparison

S5.4 showed that the 409D actor with the shared `pilot_groupwise_physical_v1` transform could still become nearly observation-insensitive and stationary. S5.5 therefore held that transform fixed and compared three reward signals. The experiment tests whether a global reward scale helps, and whether expressing progress as signed route metres adds anything beyond that scale. It does not presume either change helps.

| Arm | Training reward |
|---|---|
| A — V1 | `delta_p - 0.25/H + T_t` |
| B — Scale | `k * (delta_p - 0.25/H + T_t)` |
| C — Metre | `L_effective_m * delta_p / 10 + k * (T_t - 0.25/H)` |

`delta_p` is the signed change in MetaDrive route completion, initialized from reset progress. `H` is the registered case horizon. `T_t` is the RewardSpecV1 terminal component applied on its terminal transition. `L_effective_m` is MetaDrive `navigation.total_length`, checked against the locked geometry manifest. The manifest median for the eight training geometries is `424.64794675619254 m`; therefore `k = 42.46479467561925`. At route length `10*k = 424.64794675619254 m`, B and C are algebraically identical step by step. No reward clipping is used. Platform `RewardSpecV1` itself is unchanged.

## Maps, seeds, and information boundary

All geometries are from the locked MapSuite TRAIN split. The eight training geometries are the S5.4 `PILOT_TRAIN` Easy cases: SCCS seeds 0, 2, 3; SCSS seeds 0, 3, 4; SCS seeds 0, 1. Their IDs, geometry hashes, route lengths, round-robin order, and seed derivation are in `results/stage5/s5_5_meter_reward_seed101_20261120/protocol.json` (`training_pool_sha256=3f388a82be58b3cd773ab7c6f7c52e9f5622a9f5b29f2f9d36979beb6b93b73f`).

The six fixed, unseen internal diagnostic geometries are SCCS seeds 4 and 19, SCS seeds 12 and 16, and SCSS seeds 2 and 18. Their geometry IDs, SHA-256 values, and environment seeds are locked in the same protocol (`diagnostic_pool_sha256=e326928bdb02f36367786c8acd0bc96a3a9f843e5676a1ff56eede0d7e69281c`). No diagnostic geometry was allowed into the training sampler. No VALIDATION or TEST data was opened.

Each arm used a fresh SAC model with root seed 101 and the same initial actor, twin-critic, and entropy state (`f91771d432a8c99e8ba4f9e36bbf2f47716ac1331adbdbc5027fa6e35050ea68`). Replay buffers and optimizer states were independent. The only arm difference was training reward. Common settings were SB3 SAC, 2x256 ReLU MLP, learning rate `3e-4`, gamma `.99`, tau `.005`, buffer `100,000`, batch `256`, learning starts `1,000`, automatic entropy coefficient, one environment step and one gradient update per training step. The 409D encoder and `continuous_box2_v1` action contract remained fixed. No Stage 1/MPPI transitions, labels, or demonstrations entered SAC training.

Training checkpoints were saved at 0, 2k, 5k, 10k, and 30k steps. Each checkpoint was verified with the repository integrity verifier. Deterministic probes were capped at 300 decisions on all six diagnostic cases at every checkpoint; a probe without a terminal outcome is `PROBE_CENSORED`. Full episodes were run at 10k and 30k on the same six cases and environment seeds. Full episode outcomes use the existing terminal classifier.

## Reproduction and evidence

From `autonomous-driving-rl/`, the protocol, preflight, training, evaluator, and analyzer scripts are:

```powershell
conda run -n metadrive_sb3 python scripts/register_stage5_s55_protocol.py
conda run -n metadrive_sb3 python scripts/run_stage5_s55_reward_preflight.py
conda run -n metadrive_sb3 python scripts/run_stage5_s55_training.py --arm ALL
conda run -n metadrive_sb3 python scripts/evaluate_stage5_s55.py --arm ALL
conda run -n metadrive_sb3 python scripts/summarize_stage5_s55_results.py
```

The run artifacts are under ignored `runs/stage5_meter_reward/s5_5_meter_reward_seed101_20261120/`. Compact verified results are in the matching `results/stage5/` folder, with file sizes and hashes in `artifact_manifest.json`. The commands above describe the recorded run; the scripts refuse to overwrite an existing run identity, so reproduction requires a fresh output identity/directory. The three arms completed 30,000 actual environment steps and 29,000 optimizer updates each. This single-seed CPU run is protocol-reproducible; bitwise replay of simulator trajectories or model weights is not claimed.

The training runner did not retain per-episode training outcome rows in its final compact summary. The verified summary marks these outcomes `NOT_PERSISTED_BY_RUNNER`; aggregate completed-episode counts, per-geometry step exposure, checkpoint telemetry, and the separate fixed-schedule evaluation records are preserved. This limits analysis of the training episode outcome mix but does not alter the independently evaluated diagnostic outcomes.
