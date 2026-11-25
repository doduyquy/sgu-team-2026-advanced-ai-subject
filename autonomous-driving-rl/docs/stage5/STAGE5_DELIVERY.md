# Stage 5 Guided SAC delivery

The frozen fallback is J11–5.000, `s5-e2938ff0e622-j11-a1`. The model SHA-256 is `f82ca94d27ebd9fc3468081e20631536c959990a708398f5474bebee986e6d07`. The full-state SHA-256 is `253fe755507ca860f93a0a881e09d795d9c4d8920d5bd621e69f08947b9de591`. Keep model bytes and full state in the private campaign workspace or authorized private artifact transfer.

## Inference and demo

Use the project's MetaDrive and Stable-Baselines3 environment. The package preserves the verified model bytes, pins the encoder, transform, reference route station producer and Terminal Preview adapter, and contains checksums for the launcher. It uses 409D observations and continuous Box2 steering plus throttle/brake. The inference export does not load or copy replay.

```powershell
python -B -m cloud.kaggle.delivery --state runs/kaggle/harvest/s5-e2938ff0e622-j11-a1 --sha 253fe755507ca860f93a0a881e09d795d9c4d8920d5bd621e69f08947b9de591 --out runs/kaggle/delivery/j11-5000
python -B -m src.launcher.cli run --mode SANDBOX --agent stage5_sac_terminal_preview --checkpoint runs/kaggle/delivery/j11-5000 --task-context-producer reference_route_station_proposal_v1 --tier Easy --sequence SCS --geom-seed 12 --env-seed 30101 --render OFF --wandb DISABLED --runs-root runs/kaggle/delivery/demo
```

This represents one Easy demonstration case. Use `--render NATIVE` on a desktop for the visual demo. The launcher records the model and training-state digests, adapter fingerprint, episode outcome, speed, route progress and timing. The verified headless run completed cleanly with 98.8% route progress. The launcher remains SANDBOX-only for this policy. Official benchmark eligibility requires the project's governed checkpoint binding and protocol approval.

## Verified scope

| Tier | Frozen fallback result |
| --- | --- |
| Easy | 12/12 clean, zero safety failures on diagnostic and a seed-independent confirmation of the same six geometries |
| Medium | 0/12 clean, 12 vehicle collisions on the internal diagnostic panel |
| Hard | Unmeasured in this delivery |
| Extreme | Unmeasured in this delivery |

The seed confirmation does not estimate generalization to unseen geometry. Keep the confirmation panel out of subsequent tuning.

## Continue training

Preserve the private full-state checkpoint for training. Resume at a reset boundary with actor, critic, target critic, optimizer, alpha, RNG, sampler and replay state. A CPU-to-GPU migration preserves state while using a CUDA RNG stream. Trajectories across devices need not be bitwise equal.

An operator with a harvested candidate can plan 25k, 50k or 100k new decisions with explicit CPU or T4 and a wall-clock budget. The planner rejects a replay capacity whose conservative size estimate exceeds the long-parent transfer budget. Periodic checkpoints remain available through version-aware harvest when Kaggle publishes their outputs.

```powershell
python -B -m cloud.kaggle.control package --from-job VERIFIED_PARENT_JOB
python -B -m cloud.kaggle.control learning-resume-plan --campaign stage5 --round curriculum50 --parent VERIFIED_PARENT_JOB --transitions 50000 --device cuda --max-seconds 5400
python -B -m cloud.kaggle.control launch --campaign stage5 --round curriculum50 --job j10 --fingerprint PLAN_SHA_PRINTED_BY_PLANNER --server-admission
```

Use `--phase medium`, `--phase hard` or `--phase extreme` only for an evidence-backed next-phase transition. Omit it to continue the current phase. Use `--server-admission` when concurrency is unknown and the operator has no unresolved submission. Refer to the [operator guide](STAGE5_PARALLEL_OPERATOR.md) for account setup and artifact harvest.
