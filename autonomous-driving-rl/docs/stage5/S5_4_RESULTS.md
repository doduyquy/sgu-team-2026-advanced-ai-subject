# S5.4 SAC Learnability Recovery — Results

**Status:** both training arms completed 100,000 environment steps. This is a TRAIN-only development experiment, not a benchmark evaluation.

## Protocol identity

- Run: `s5_4_learnability_seed101_20261119`
- Source revision: `f7c7ccef2c9f35e57ab67228f119908709b838a7`
- Protocol SHA-256: `f3ca16a6ab452b53735e59bd03d59dab28a34cc70ca4d573d53fa72f7eebca25`
- Root seed: `101`; sampler: episode-level round robin; 100,000 steps per arm.
- Training pool: eight Easy `PILOT_TRAIN` geometries (3 SCCS, 3 SCSS, 2 SCS). All eight were encountered by both arms.
- Evaluation pool: the same six Easy `PILOT_DIAGNOSTIC` geometry/environment-seed pairs at every checkpoint. Geometry identity and SHA-256 are recorded in `protocol.json`.
- No VALIDATION or TEST data were read. `benchmark_eligible` remains false.

Control A used the unchanged 409D encoder output. Treatment B applied only the versioned `pilot_groupwise_physical_v1` post-encoder transform: traffic actor fields were scaled by declared physical reference units, waypoint x/y by 50m and heading by π; core features, mask semantics and task mask/scalars were unchanged. The canonical 409D encoder and RewardSpecV1 were not changed. Both arms used fresh initialization from the shared initial parameter state, root seed 101, and independent optimizers/replay buffers. The only registered arm difference was the observation transform.

Common SAC settings: SB3 MlpPolicy with two 256-unit ReLU layers, batch 256, buffer 100,000, learning rate 0.0003, gamma 0.99, tau 0.005, train frequency 1, one gradient step per environment step, and automatic entropy coefficient. Each arm performed 99,000 optimizer updates after the 1,000-step learning-start period.

## Training record

| Arm | Steps | Updates | Completed episodes | Geometries encountered | Training wall time | Throughput | Final sampled RSS | Clean successes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A — control | 100,000 | 99,000 | 79 | 8/8 | 6,101.6 s | 16.4 steps/s | 622.3 MB | 0 |
| B — physical group scaling | 100,000 | 99,000 | 78 | 8/8 | 2,665.3 s | 37.5 steps/s | 642.9 MB | 0 |

Both arms reached every scheduled checkpoint (0, 10k, 30k, 50k, 100k). Runtime differed substantially; wall-clock throughput is reported as measured and is not evidence of better learning. Per-geometry actual step counts are in `scientific_results.json` and `protocol.json`.

## Fixed diagnostic results

Each fast diagnostic is six 300-decision probes. Every probe was `PROBE_CENSORED` at the cap (6/6); this is not a timeout, success or evidence of safety. “Progress Δ” is measured from reset. `Stationary` is the fraction of decisions below 1 km/h. Full episodes ran only at 30k and 100k.

| Steps | Arm | Probe speed (km/h) | Probe progress Δ | Stationary | Full mean route Δ | Full clean success | Full safety failure | Full timeout |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | A | 0.01473 | +0.0000304 | 1.000 | N/A | N/A | N/A | N/A |
| 0 | B | 0.01486 | +0.0000256 | 1.000 | N/A | N/A | N/A | N/A |
| 10,000 | A | 0.01476 | +0.0000637 | 1.000 | N/A | N/A | N/A | N/A |
| 10,000 | B | 0.48274 | +0.0098057 | 1.000 | N/A | N/A | N/A | N/A |
| 30,000 | A | 0.02126 | −0.0001050 | 1.000 | −0.0000890 | 0/6 | 0/6 | 6/6 |
| 30,000 | B | 0.02410 | −0.0001038 | 1.000 | +0.0000512 | 0/6 | 0/6 | 6/6 |
| 50,000 | A | 0.02094 | −0.0001065 | 1.000 | N/A | N/A | N/A | N/A |
| 50,000 | B | 0.02302 | −0.0000959 | 1.000 | N/A | N/A | N/A | N/A |
| 100,000 | A | 0.01951 | −0.0000825 | 1.000 | −0.0000826 | 0/6 | 0/6 | 6/6 |
| 100,000 | B | 0.02357 | −0.0000995 | 1.000 | +0.0000183 | 0/6 | 0/6 | 6/6 |

At 100k, the mean final route completion was 0.012134 for A and 0.012234 for B. All 12 full episodes (six per arm) ended by the environment's true TIMEOUT classifier; none ended in clean success or safety failure. Both policies remained stationary throughout those episodes. The absence of a safety terminal while the vehicle hardly moved must not be described as safe driving.

B's 10k probe showed a temporary movement/progress signal, but all six probes were still stationary under the `<1 km/h` definition, and that signal did not persist at later checkpoints. At 30k and 100k the full-episode driving metrics remained near the reset route-completion value.

## Paired A/B comparison

Pairs share both geometry ID and environment seed (`n=6`). On full episodes, B−A mean route-progress difference was `+0.0001402` at 30k (B higher on 6/6 pairs) and `+0.0001009` at 100k (B higher on 5/6, lower on 1/6). The 100k difference is about **0.0101 percentage points** of route completion. Mean speed differences were only `+0.000905 km/h` at 30k and `+0.001393 km/h` at 100k. Stationary fraction, clean success and safety-failure outcomes were identical: stationary 1.0, clean success 0/6 and safety failure 0/6 for both.

This small paired route difference is not a useful-driving improvement and does not establish treatment superiority. There is one training root seed and six paired diagnostic cases; no statistical superiority claim is warranted.

## Actor responsiveness

On the fixed anchor observations, both arms' first actor layer became inactive. The measured nonpositive fraction was 98.8% for A and 100% for B at 10k; it was 100% for both by 50k and 100k. At 100k, the standard deviation of deterministic actor actions across the six observations was effectively zero for both arms. At 30k, A's steering varied across anchors, but its throttle remained nearly constant and the vehicle still failed to make route progress. These measurements support state-insensitive policy behavior on the sampled anchors; they do **not** prove that dead ReLUs are the sole root cause.

## Conclusion

The existing SAC stack completed the full 100k budget in both arms, saved and verified all ten checkpoints, exercised all eight selected training geometries, and performed isolated evaluations on the fixed unseen internal TRAIN diagnostic set. However, neither arm learned useful autonomous driving in this experiment: there were no clean successes, all diagnostic full episodes timed out, and both policies were stationary. The observation-scaling treatment did not recover persistent movement or actor sensitivity. This is a negative result for this configuration and seed, not proof that SAC is incapable of solving the task.

**Decision: `NO_GO_MULTI_MAP` for these candidates.** Do not transfer either checkpoint to multi-map training or official evaluation based on this run.

## Artifacts

- [Machine-readable protocol](../../results/stage5/s5_4_learnability_seed101_20261119/protocol.json)
- [Mechanism analysis](../../results/stage5/s5_4_learnability_seed101_20261119/mechanism_analysis.json)
- [Checkpoint metrics CSV](../../results/stage5/s5_4_learnability_seed101_20261119/checkpoint_metrics.csv)
- [Paired per-case comparisons](../../results/stage5/s5_4_learnability_seed101_20261119/paired_comparisons.json)
- [Scientific results and checkpoint identities](../../results/stage5/s5_4_learnability_seed101_20261119/scientific_results.json)
- [Summary artifact checksums](../../results/stage5/s5_4_learnability_seed101_20261119/summary_artifact_checksums.json)
- [Speed curve](../../results/stage5/s5_4_learnability_seed101_20261119/plots/probe_mean_speed_km_h.png), [progress curve](../../results/stage5/s5_4_learnability_seed101_20261119/plots/probe_mean_route_progress_delta.png), [stationary fraction](../../results/stage5/s5_4_learnability_seed101_20261119/plots/probe_stationary_fraction.png), [actor sensitivity](../../results/stage5/s5_4_learnability_seed101_20261119/plots/actor_sensitivity.png), [paired route differences](../../results/stage5/s5_4_learnability_seed101_20261119/plots/paired_route_progress_difference.png), and [training geometry coverage](../../results/stage5/s5_4_learnability_seed101_20261119/plots/training_geometry_coverage.png).

Raw model bundles and episode logs remain in the ignored `runs/stage5_learnability/s5_4_learnability_seed101_20261119/` directory. A Git clone contains the manifests, summary evidence and plot files, but not the checkpoints or raw logs.
