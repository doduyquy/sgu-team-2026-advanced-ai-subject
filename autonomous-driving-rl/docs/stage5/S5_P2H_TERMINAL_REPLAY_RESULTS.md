# Stage 5 P2-H — Results

**Decision:** `P2H_INCONCLUSIVE` for the terminal-prioritization hypothesis; both 10k policies failed to demonstrate clean driving.

**TRAIN only:** no VALIDATION/TEST access; `benchmark_eligible=false`.

**Protocol SHA-256:** `4ed9ae34692e5da2a01953b85c692722cd1a32c3e52dc723e1dbc1472e6026bd`.

**Training source commit:** `875453e18d7f7c82002c1b0ee799fc7f6f4abab8`.

## What the experiment actually tested

Both arms used fresh SAC initialization, the same initial actor/critic/entropy tensors, seed 101, eight Easy TRAIN geometries, corrected F09 producer, 409D transformed observation, Continuous Box2, RewardSpecV1, gamma 0.99, and the P2-D actor-only LeakyReLU(0.01)/ReLU critic architecture. Each reached exactly 10,000 environment decisions and 9,000 optimizer updates.

Arm A used SB3's standard uniform batch shared by critic, actor and entropy-temperature loss. Arm B sampled a separate uniform actor/temperature batch and a critic batch from the registered mixture. The intended critic mixture prioritized true clean/safety terminals. However, B recorded **zero true terminals** at every checkpoint. Its sampler therefore used the uniform fallback for the entire run. B had seven timeout truncations, which SB3 correctly bootstrapped and the priority sampler correctly excluded.

Because B's actor/temperature batch was independent while A reused its uniform batch for all losses, the arms also differed in minibatch coupling. With no terminal-priority exposure, any observed metric difference cannot be attributed to terminal prioritization. The P2-H treatment contrast was not realized in the training data.

## Training and replay evidence

| Metric | A — uniform | B — terminal-aware |
|---|---:|---:|
| Actual environment steps | 10,000 | 10,000 |
| Optimizer updates | 9,000 | 9,000 |
| Completed training episodes | 7 | 7 |
| Training geometry IDs visited | 8/8 | 8/8 |
| Training outcomes | 7 TIMEOUT | 7 TIMEOUT |
| Clean / safety terminal rows | 0 / 0 | 0 / 0 |
| Timeout truncations in replay | 7 | 7 |
| B prioritized critic draws | — | 2,304,000, all uniform fallback |
| B eligible terminal priority draws | — | 0 |
| Action-to-replay mismatch | 0 | 0 |
| Training wall time | 201.5 s | 213.7 s |

Exploration remained stochastic: throttle was positive in 48.95% (A) and 48.92% (B) of replay actions; median positive run length was one decision in both arms, with 5,016/5,018 sign switches. This is training-action telemetry, not evidence that deterministic inference learned to drive.

## Deterministic probes and full episodes

The six fixed Easy TRAIN diagnostic cases were evaluated at checkpoints 0/2k/5k/10k. Every 300-decision probe without terminal was labeled `PROBE_CENSORED`. Full 10k evaluation used the same six geometry/seed pairs.

| Checkpoint | Arm | Mean speed km/h | Mean route-progress delta | Stationary fraction (<1 km/h) | Clean | Safety | Timeout | Censored |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | A | 0.015 | 0.00003 | 100.0% | 0 | 0 | 0 | 6 |
| 0 | B | 0.015 | 0.00003 | 100.0% | 0 | 0 | 0 | 6 |
| 2k | A | 3.294 | 0.06701 | 15.0% | 0 | 0 | 0 | 6 |
| 2k | B | 3.301 | 0.06713 | 15.1% | 0 | 0 | 0 | 6 |
| 5k | A | 0.383 | 0.00777 | 100.0% | 0 | 0 | 0 | 6 |
| 5k | B | 3.840 | 0.07377 | 13.3% | 0 | 2 | 0 | 4 |
| 10k | A | 0.797 | 0.01620 | 62.8% | 0 | 0 | 0 | 6 |
| 10k | B | 1.126 | 0.02286 | 44.4% | 0 | 0 | 0 | 6 |

At 10k, all six **full episodes** in each arm ended `OUT_OF_ROAD`: A 0/6 clean and 6/6 safety failures; B 0/6 clean and 6/6 safety failures. Mean route-progress delta was 0.14678 for A and 0.14413 for B. Mean speed was 2.363 vs 2.785 km/h and stationary fraction was 22.8% vs 19.4%. Paired B−A differences across the six cases were −0.002651 route progress, +0.4217 km/h, and −0.03446 stationary fraction. The faster movement was accompanied by failure on every full episode, so it is not a driving improvement.

## Observation dependence and learning signal

Across 360 real diagnostic observations, deterministic action standard deviation fell from `[0.00204, 0.00239]` at step 0 to `[0.0000063, 0.0000271]` for A and `[0.0000118, 0.0000264]` for B at 10k (steering, throttle/brake). At 5k A was effectively constant; B retained slightly more variation but had two safety outcomes in the short probes. By 10k, both actors were nearly constant on the tested states. This confirms weak state sensitivity in these checkpoints; it does not identify whether the cause is critic learning, representation, entropy tuning or another interaction.

At 10k, alpha was approximately 0.0672 in both arms. Fixed recent-replay Q/TD snapshots contained no true terminal rows. They therefore do not answer how either critic would learn from terminal safety/success outcomes.

## Scientific conclusion

**Confirmed:** the sampler implementation can oversample genuine terminals, exclude timeouts from priority eligibility, retain SB3 timeout bootstrapping, and preserve action/replay fidelity. The A/B runs were complete, but B encountered no terminal safety or clean-success transition; its priority branch never activated. Both deterministic policies lacked robust state-dependent actions by 10k, and neither completed a clean diagnostic episode.

**Not established:** whether terminal-aware replay improves SAC, whether critic error caused the stationary behavior, or whether a different reward/entropy objective would solve it. The paired speed/progress differences are non-causal because terminal prioritization had zero exposure and the minibatch coupling differed.

**Decision:** `P2H_INCONCLUSIVE`. This result does not support extending the same experiment or declaring a replay-method win. For a future bounded decision, prioritize one preregistered SAC learning-objective comparison around reward/entropy/critic interaction; do not infer a fix from lower Q error, speed alone, or these non-exposed replay runs.

## Artifacts

- Machine-readable summary and paired cases: [`result_summary.json`](../../results/stage5/p2h_terminal_replay_seed101_20261121/result_summary.json)
- Summary SHA-256: [`result_summary.sha256`](../../results/stage5/p2h_terminal_replay_seed101_20261121/result_summary.sha256)
- Preflight evidence: [`preflight_evidence.json`](../../results/stage5/p2h_terminal_replay_seed101_20261121/preflight_evidence.json)
- Artifact hashes: [`CHECKSUMS.sha256`](../../results/stage5/p2h_terminal_replay_seed101_20261121/CHECKSUMS.sha256)
- Ignored full checkpoints, replay buffers and per-step logs: `runs/stage5_p2h/s5_p2h_terminal_replay_seed101_20261121/`. They are not committed; their hashes and sizes are listed in `artifact_inventory.json`.
