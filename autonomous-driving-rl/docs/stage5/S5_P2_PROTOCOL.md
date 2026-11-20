# S5 P2 — Correlated Exploration Protocol

**Protocol ID:** `stage5_p2_ou_ar1_exploration_v1`
**Status:** preregistered before training
**Decision scope:** development-only, TRAIN geometries; `benchmark_eligible=false`.

## Question and registered contrast

Does a temporally correlated exploration perturbation produce more useful replay coverage and a more observation-dependent deterministic SAC policy than standard SAC collection, under the same RewardSpecV1, observation pipeline, geometry pool, seed rules, and optimizer budget?

- **A — Control:** standard Stable-Baselines3 SAC collection: random warm-up, then stochastic policy actions.
- **B — Treatment:** same collection plus a zero-mean stationary AR(1) action perturbation after warm-up, applied to both steering and throttle/brake.

The perturbation is `n[t] = 0.85 n[t-1] + epsilon[t]`, with stationary standard deviation 0.15 and innovation standard deviation `0.15 * sqrt(1 - 0.85^2) = 0.07905694`. It is added in the normalized Box(2) action domain, clipped componentwise to `[-1, 1]`, and the same resulting action is passed to MetaDrive and recorded in replay. Noise state resets at both termination and truncation. It does not hold a sampled throttle command for a fixed number of decisions; it is a distinct experiment from historical S5.3G throttle-hold.

This changes the exploration sampler only. It does not change RewardSpecV1, gamma, tau, architecture, 409D encoder, the shared groupwise observation transform, Box2 action contract, or policy outputs during frozen deterministic evaluation.

## Data and seeds

The training pool is the exact eight Easy `PILOT_TRAIN` geometries in the S5.5 locked protocol. The diagnostic pool is its six fixed Easy `PILOT_DIAGNOSTIC` geometries, each with a frozen environment seed. Their IDs and geometry hashes are embedded in [`protocol.json`](../../results/stage5/p2_correlated_exploration_seed101_20261120/protocol.json) and revalidated against the locked TRAIN manifest before runner startup. The pools are disjoint. Neither pool contains VALIDATION or TEST cases.

Both arms use root seed 101 and the same ordered `ROUND_ROBIN` geometry sampler with the existing deterministic environment-seed derivation. Actual episode geometry/seed exposure is logged. If arm trajectories produce different episode counts, the run records that divergence rather than forcing the simulator through artificial paired transitions.

Both arms use fresh initialization. One initial actor/critic state and entropy state is created and hashed, then copied to each arm; replay buffers and optimizer states are independently initialized. No historical trained checkpoint initializes either arm. The registered SAC settings are MLP `[256,256]` ReLU, learning rate `3e-4`, gamma `.99`, tau `.005`, buffer 100,000, batch 256, learning starts 1,000, auto entropy, one gradient update per environment step, CPU. Each arm is capped at 10,000 environment steps with checkpoints at 0, 2,000, 5,000, and 10,000.

Both arms explicitly use `reference_route_station_proposal_v1`. Historical S5.5 checkpoints remain on `legacy_v1` and are only used for offline P2-A analysis.

## Evaluation

Every saved checkpoint is evaluated with a frozen deterministic actor in a separate MetaDrive environment on all six locked diagnostic cases, with a 300-decision cap. At 10k, each arm also receives a full episode on all six cases using the resolved route-aware horizon and the existing terminal classifier. Evaluation does not touch the training sampler, replay buffer, optimizer, or weights. A cap without terminal outcome is `PROBE_CENSORED`; it is not a timeout or a safety completion.

Primary development measurements: reset-relative route progress, speed, stationary fraction, sustained movement, clean success, safety failure, timeout, throttle/brake and steering behavior, actor action variation across real observations, first-layer activation, training episodes and geometry exposure. Training telemetry records action samples, executed actions, replay actions, losses, alpha, gradient norms, twin-Q and TD-error snapshots, memory, throughput, checkpoint integrity, and per-episode outcomes.

## Decision limits

One root seed and six diagnostic geometries cannot establish generalization or statistical superiority. This experiment can establish whether the registered treatment changed exploration/coverage and whether a learning signal appeared under this narrow setup. Acceleration, lower Q values, or shorter censored probes do not count as learned driving competence. The runner must stop at 10k per arm; no automatic budget extension or additional treatment is authorized.

Reproduction:

```powershell
conda activate metadrive_sb3
python scripts/run_stage5_p2_experiment.py --arm ALL
python scripts/evaluate_stage5_p2.py --arm BOTH
```

The first command refuses to overwrite an existing run. The second reads only saved checkpoints and the locked TRAIN diagnostic schedule. Protocol SHA-256 is stored in `protocol.json`; all implementation source file hashes are also checked before training.
