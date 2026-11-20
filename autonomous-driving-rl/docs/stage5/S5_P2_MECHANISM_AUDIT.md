# S5 P2-A — Historical learning-mechanism audit

**Status:** complete, read-only analysis. No training was performed in this phase.

## Evidence and scope

The audit loads all 15 S5.5 checkpoints (A/B/C at 0, 2k, 5k, 10k, and 30k) through the integrity-checked checkpoint loader, explicitly requesting `legacy_v1`, which is the producer recorded by those historical runs. It evaluates 619 saved 409D observations from the fixed six-case TRAIN diagnostic schedule. The raw observation file SHA-256 is `3ac58093aebb2ea33f1f03ff02669bbe05d0a882820413d293aa9508ae739e85`; the audit JSON SHA-256 is `9e5452ed83b11887c06d1dde75f17e10319898c9fcd0d9cc01441280981a1bd2`.

These checkpoints predate versioned F09 integration. The audit does not reinterpret them as corrected-producer checkpoints and does not use them as the P2 control. The new P2 experiment uses only fresh initialization and `reference_route_station_proposal_v1` in both arms.

## Measured behavior on saved TRAIN observations

| S5.5 step | Deterministic action SD range across three arms (steering / throttle) | First actor layer nonpositive | Alpha range | Offline actor-gradient proxy range |
|---:|---:|---:|---:|---:|
| 0 | 0.00194–0.00194 / 0.00249–0.00249 | 46.6% | 1.000 | 1.271 |
| 2k | 0.00030–0.00055 / 0.00025–0.00050 | 96.3–97.1% | 0.7408 | 0.238–0.466 |
| 5k | 0 for all arms / 0 for all arms | 100% | 0.3012 | 0.106–0.107 |
| 10k | 0 / 0 | 100% | 0.0672 | 0.0234–0.0250 |
| 30k | 0–0.635 / 0 for all arms | 100% | 0.000218–0.000890 | 0.000258–0.00371 |

At 30k the mean deterministic throttle was approximately -0.90 (A), -0.97 (B), and -0.97 (C). The action SD values are computed across saved observations per checkpoint and arm; the observation samples are small and do not cover every encountered training state. The activation and action summaries show when collapse-like behavior is present on this diagnostic sample, not its cause.

The gradient values are recomputed offline by differentiating the actor objective on saved observations. They are **not** historical optimizer gradient logs. Historical per-update gradients, exact replay action support, and replay TD errors are unavailable because S5.5 did not save replay buffers or those telemetry streams.

## Critic comparisons and their limits

The audit evaluated twin Q values for deterministic actions and local action substitutions (throttle/brake and steering at -0.5, 0, +0.5) on the saved observations. Twin-critic gaps varied by checkpoint: large at initialization on this sample, then generally much smaller after updates. The ranking was not consistently a preference for braking at later checkpoints. These substitutions may be outside historical replay support; support is **UNKNOWN**. Q values are model predictions, not environment ground truth, so this audit does not label the critics calibrated or miscalibrated.

Alpha declined monotonically in these saved checkpoints, and first-layer ReLU inactivity rose. Neither observation alone proves that entropy tuning or ReLU inactivity caused stationary behavior. The sampled observations also contained no variation in LiDAR or traffic features in this Easy no-traffic schedule, so their utility cannot be assessed here.

## Prior throttle persistence result

The S5.3G hold-five diagnostic reduced throttle/brake sign switching and increased movement/progress in its training trajectories. Executed and replay-stored actions were checked to match in that implementation. Yet deterministic evaluation remained stationary on all 24 cases, and the intervention produced several out-of-road training endings with no clean success. This tests an action persistence intervention in that historical run; it does not test an AR(1) exploration process that perturbs both steering and longitudinal action and records exact replay fidelity in a fresh corrected-producer experiment.

## P2 mechanism decision

The best bounded mechanism to test next is temporally correlated exploration. The reason is specific: prior stochastic sampling produced short positive-throttle runs, while fixed throttle holding changed data collection but did not make the deterministic actor learn movement. OU AR(1) is a distinct, pre-registered treatment whose realized temporal correlation, clipping, episode reset, executed action, and replay action can all be measured. This is an exploratory hypothesis, not a claim that exploration is the root cause.

Evidence: [`historical_mechanism_audit.json`](../../results/stage5/p2_mechanism_audit/historical_mechanism_audit.json) and its adjacent `.sha256` file.
