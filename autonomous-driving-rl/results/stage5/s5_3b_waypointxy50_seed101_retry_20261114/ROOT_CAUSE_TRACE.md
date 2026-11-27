# Follow-up trace: why did the 5k policy leave the road?

This follow-up uses only two cases from the same internal TRAIN diagnostic
schedule: `geom_easy_SCCS_seed19` and `geom_hard_SCXOCS_seed0`. It replays the
5k checkpoint deterministically and records the full 409D policy input, action,
speed, route progress, reward, ego position/heading and safety flags at every
step. The trace is development-only; no official VALIDATION/TEST data was used.

Raw per-step JSONL traces and the checkpoint activation analysis are retained
under the ignored `runs/stage5_pilot/s5_3b_waypointxy50_seed101_retry_20261114/`
directory. The reusable instrumentation is
`scripts/evaluate_stage5_pilot.py --trace-dir --geometry-id` and
`scripts/analyze_stage5_pilot_trace.py`.

## Observations

| Geometry | Steps | Steering std | Throttle std | Unique actions (4 decimals) | Route progress | First ReLU active units on 0 / 5k / 10k observations |
|---|---:|---:|---:|---:|---:|---:|
| Easy, SCCS seed 19 | 766 | 0.00000007 | 0.00000002 | 1 | 0.00985 → 0.10668 | 136.2 / **0** / **0** |
| Hard, SCXOCS seed 0 | 368 | 0.0443 | 0.0489 | 49 | 0.00807 → 0.05414 | 135.4 / 14.3 / 14.1 |

Both episodes ended with the authoritative `out_of_road` flag. On Easy, the
5k action was identical for all 766 observations:
steering `-0.010949`, throttle `0.005712`. The actor's first 256-unit ReLU
layer had zero active units on every observation in this episode. The same
observation sequence activates about 136 units per step in checkpoint 0. This
directly explains the constant action on this case: with its first hidden
representation all zero, the actor reduces to its downstream biases and
cannot respond to changing route state.

On Hard, the first layer was active more often and actions varied, but about
68% of observations still had no active first-layer unit. The policy still
left the road. So dead-ReLU output collapse explains the Easy case and much of
the 10k stationary behavior, but it does not by itself explain every unsafe
trajectory; the partially active Hard policy also maps state to unsafe
control.

Across both traces the mean first-layer preactivation changed from about
`+0.009` in the untrained checkpoint to `-0.35` at 5k and `-0.36` at 10k. The
mean first-layer bias changed only from about `-0.0012` to `-0.0025` on these
traces. This points toward learned weights interacting poorly with the visited
observation distribution, rather than a simple uniform bias shift.

## What this establishes and what remains open

Established: after training, this actor has a dead first ReLU layer on the
Easy trace and a sparse one on the Hard trace; on Easy this produces a
constant, non-responsive action. The waypoint scale adjustment alone did not
prevent it.

Not established: why the SAC updates push the first-layer representation into
this regime. Candidate contributors include the narrow replay experience (the
10k run completed only seven episodes on seven Easy training geometries), the
feature distribution and SAC actor/critic gradients. This two-case trace does
not prove which contributor dominates, nor does it establish a general result
across training seeds.

The next diagnostic should instrument replay batches and actor updates around
the onset of collapse: feature-group preactivation contributions, active-unit
fractions, actor gradient norms, and action variation on a fixed observation
probe set at regular checkpoints. Keep reward, architecture, and all other
training settings fixed. Only after locating when and how the representation
collapses should we choose a controlled intervention.
