# S5.3B waypoint-scale ablation — observed outcome

Classification: internal development diagnostic on the fixed PILOT_DIAGNOSTIC
subset of TRAIN. This is not official VALIDATION or TEST evidence.

## Design and provenance

Starting at seed 101, SAC was trained continuously for 10,000 steps with the
S5.2 architecture and unchanged RewardSpecV1. The only policy-input change was
to divide the 20 waypoint relative x/y values by 50 m after the frozen 409D
encoder. Heading and every other feature remained unchanged. Training used
the fixed 156-geometry PILOT_TRAIN membership; the same 24 PILOT_DIAGNOSTIC
geometries and environment seeds were evaluated deterministically at 0, 5k,
and 10k. All 72 episodes completed. No official holdout was read.

Run artifacts are under `runs/stage5_pilot/s5_3b_waypointxy50_seed101_retry_20261114/`.
The primary machine-readable report and empirical route curve are beside this
file. Checkpoint bundle SHA-256 values and the transform fingerprint are in
`scientific_summary.json` and the run manifest.

## Full diagnostic measurements

| Steps | Mean route | Median route | Clean success | Safety failure | Timeout | Stationary fraction | Mean speed | Mean return |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00889 | 0.00836 | 0/24 | 0/24 | 24/24 | 1.000 | 0.013 km/h | -0.250 |
| 5,000 | 0.08087 | 0.06586 | 0/24 | 24/24 | 0/24 | 0.319 | 2.384 km/h | -1.020 |
| 10,000 | 0.00890 | 0.00838 | 0/24 | 0/24 | 24/24 | 1.000 | 0.013 km/h | -0.250 |

At 5k, mean throttle was 0.0129, mean brake was effectively zero, and mean
steering was -0.0157 (steering standard deviation 0.0160); steering saturation
was zero. The policy did move and advanced farther, but every case ended in a
safety failure and return became much worse. At 10k the actor returned to a
near-constant stationary policy. The paired per-geometry differences are in
`scientific_summary.json`; the route learning curve is based on these measured
episodes.

## Interpretation

This supports a narrower mechanism claim: fixed waypoint scaling can prevent
the early stationary collapse long enough for SAC to produce a moving policy,
but that policy is unsafe and the change does not persist through 10k. It does
not establish waypoint scale as the sole cause of dead-ReLU collapse. The
training run still completed only seven episodes on seven Easy geometries,
with no clean success. A single seed and one scale value cannot establish a
general scaling rule.

The next useful experiment is not a longer budget. Inspect the 5k policy's
first out-of-road transitions and action/observation traces, then design a
separate controlled safety/coverage study. Keep reward and official metrics
fixed until that mechanism is understood. Do not promote this checkpoint to
benchmark use.

## Runtime and recovery

Training completed 10,000 steps, 9,000 optimizer updates, in 704.4 s
(14.2 steps/s), with RAM sampled from 353.6 to 409.1 MB. The first full
diagnostic attempt stopped during case 1 because Windows denied replacing the
progress JSON; no incomplete episode was treated as an outcome. The writer was
made resilient, the failed artifacts retained under
`diagnostics_failed_attempt1/`, and all three schedules were then run to 24/24.
Each evaluator confirmed weights, optimizer count, replay size and training
sampler state were unchanged.
