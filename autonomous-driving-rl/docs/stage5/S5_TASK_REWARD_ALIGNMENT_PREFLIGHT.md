# Stage 5 — P1-R Task & Reward Alignment Preflight

**Decision: `REWARD_V1_RETAIN`.** Reward integrity passed on the available full TRAIN traces. This decision retains RewardSpecV1 for now; it does not show that the reward is sufficient, that SAC learned driving, or that stationary behavior is solved. No SAC training, validation/test access, or checkpoint modification occurred.

## Scope and provenance

- Analysis base: PR #42 HEAD `2c572ea620e4254667665814f74485ddf57815e6`.
- Historical trajectories: S5.5 protocol source commit `1baaeedf9bd9d69f4507d5ff9b27b75dcc9d5639`; TRAIN geometry `geom_easy_SCS_seed0`, SHA-256 `3bbbf995e00279527132ce1594ffe36e8b0236c47afdbec7fdb6bb287e9687d8`, environment seed 0.
- Raw trace: `runs/stage5_meter_reward/s5_5_meter_reward_seed101_20261120/preflight/reward_traces.jsonl.gz`, SHA-256 `fa19c1e9bb547f4abc5ce1371fe429603e8a1b65922c026df8dfd14adcc67ec3` (local, gitignored), 1,845 transitions across four complete episodes.
- The historic S5.5 protocol predates F09 producer integration and does not record a TaskContext producer identity. Reward replay uses recorded signed progress, horizon and terminal component, so its arithmetic is independently checkable; the historic policy observations cannot be retroactively attributed to the corrected producer. Current registered producer: `reference_route_station_proposal_v1`, fingerprint `a7df3942bb874b40ea16474be43c7b7a48796646c9d59cf61da5d71ba5291725`.
- Current implementation remains AgentInputV1 409D, Continuous Box2, corrected producer versioning, and Platform RewardSpecV1. No canonical contract or historic artifact changed.
- Machine evidence: [`evidence.json`](../../results/stage5/task_reward_alignment_p1r/evidence.json), SHA-256 `d81bf7c7375246d76343cec101439490a51daaa9f0d81497f0db4dd352627156`; checksum file [`evidence.sha256`](../../results/stage5/task_reward_alignment_p1r/evidence.sha256).

## Objective and metric separation

The task is safe route completion in as little time as practical. Clean success requires reaching the destination without a safety violation. Safety failures and timeouts remain separate outcomes. Time-to-clean-success is defined only over clean completed episodes; mean speed is descriptive, and a conditional cruising speed must be reported with its condition and denominator. Route progress is signed change from reset, not absolute route completion.

RewardSpecV1 does not set a universal speed target. It rewards signed route progress, subtracts a horizon-normalized time cost, and applies a terminal success or failure component. The ~60 km/h figure in the task note is a conditional development objective, not a reward term or benchmark threshold. Lane tracking, heading, curve-entry speed and control smoothness are not explicit reward terms in RewardSpecV1; they are intended to emerge through progress and safety outcomes and must be inspected as behavior/diagnostic evidence.

## Reward integrity and observed ranking

The audit replayed the stored transitions through the authoritative `compute_step_route_progress_and_reward()` helper and independently recomputed V1, Scale and Metre. It verified contiguous steps, reset progress, signed deltas, route-length conversion, terminal-at-end semantics, and actual reward passed in the trace. All 1,845 rows matched with zero observed numeric difference; clipping was disabled. Each of the four full episodes had exactly one terminal transition; the timeout had zero terminal reward, and each of the other three had one nonzero terminal component.

| Full TRAIN episode | Outcome | Steps | Progress Δ | Mean speed | V1 return, undiscounted | V1 return, γ=.99 |
|---|---|---:|---:|---:|---:|---:|
| Stage 1 reference trace | CLEAN_SUCCESS | 418 | +0.975488 | 34.23 km/h | +1.890112 | +0.181417 |
| S5.4 A 100k frozen policy | TIMEOUT, stationary | 1,224 | −0.000061 | 0.014 km/h | −0.250061 | −0.020481 |
| Stage 0 random seed 42 | CRASH_SIDEWALK | 103 | +0.202513 | 29.88 km/h | −0.818524 | −0.265037 |
| Stage 0 random seed 101 | OUT_OF_ROAD | 100 | +0.199268 | 29.63 km/h | −0.821157 | −0.276876 |

The clean trip took 41.8 seconds (418 decisions at 10 Hz). Its mean speed while at or above 1 km/h was 34.31 km/h; this single trajectory does not estimate a general cruising-speed distribution. Only one authentic clean trajectory exists. There is no matched clean-fast/clean-slow pair and no real unsafe-slow trace; both comparisons are `UNKNOWN`.

All three recorded reward forms ranked the clean trace above stationary and both failures over these complete episodes. This is a narrow sanity result from one geometry, seed and four trajectories, not a general policy ranking. The Scale and Metre magnitudes are not directly comparable as policy quality because they rescale reward; on this 407.894 m route they are not numerically identical.

## Credit assignment and termination

For horizon `H=1,224`, the per-decision time cost is `0.25/H = 0.000204248`. An ideal signed forward progress rate must exceed about `0.8333 m/s` (3.0 km/h) for progress reward alone to exceed that time cost, assuming velocity projected along the route and no terminal component. This does not mean the environment requires the car to cruise at 3 km/h; it is the pointwise reward break-even implied by this horizon.

| γ (analysis only) | Effective horizon `1/(1−γ)` | Weight at clean terminal (step 418) | Weight at timeout (step 1,224) | Stationary V1 return, discounted |
|---:|---:|---:|---:|---:|
| 0.90 | 10 decisions / 1 s | 8.30e−20 | 1.09e−56 | −0.002042 |
| 0.99 | 100 decisions / 10 s | 0.015131 | 4.59e−6 | −0.020481 |
| 0.999 | 1,000 decisions / 100 s | 0.658883 | 0.294166 | −0.144286 |
| 0.9999 | 10,000 decisions / 1,000 s | 0.959155 | 0.884877 | −0.235378 |
| 1.0 | undiscounted | 1 | 1 | −0.250061 |

These are trajectory replay calculations only; training gamma remains `0.99`. At γ=.90 the two observed unsafe-fast episodes have slightly positive discounted V1 returns (+0.001674 and +0.001666), marginally above the clean trace (+0.001384), because early progress is retained while late terminal outcomes are almost entirely discounted. At the actual γ=.99, the two failures are negative and the clean trace is positive. This confirms that discount choice can change ranking on these particular returns; it does not establish that changing gamma is warranted.

In the common zero-based window `[49, 99)`, unsafe traces earned V1 returns +0.1341/+0.1379 versus +0.1210 for the clean trace. Their first public-task-waypoint curve indications occurred at steps 93/96 versus 104 for the clean trace, so the window is not matched by distance-to-curve or geometry phase. The unsafe traces terminate before or near the curve and do not prove that curve overspeed caused failure. They do show how transient progress can precede a later safety penalty.

Runtime source inspection and the installed SB3 2.9.2 source confirm the distinction:

- `Stage5TrainingEnv.step()` classifies max-step exhaustion as Gymnasium truncation and safety/success outcomes as termination. Timeout reward is zero under the current contract.
- `DummyVecEnv` records `TimeLimit.truncated` and preserves `terminal_observation` before autoreset.
- SB3 `ReplayBuffer` stores the timeout flag and multiplies done by `(1 − timeout)` when sampling. Thus the SAC critic bootstraps from the terminal observation after timeout; its target also has the SAC entropy term. The observed finite episode return is not itself the full SAC Bellman target.
- Safety failures and clean success are true terminals and are not bootstrapped through as ordinary timeouts.

## Classification

### CONFIRMED

- RewardSpecV1 and all recorded reward variants match the stored transitions and authoritative helper on all 1,845 rows.
- On these matched full episodes, terminal-once accounting is correct; timeout contributes no terminal penalty; signed progress is preserved.
- The recorded examples show clean success above stationary and safety failures at full-episode return for V1, Scale and Metre.
- γ changes discounted trajectory ranking; at γ=.90 the observed unsafe-fast traces slightly outrank the clean trace, while at γ=.99 they do not.
- Timeouts are truncated and bootstrapped by the installed SB3 replay semantics; terminal failures/successes are not.

### HYPOTHESIS

- Sparse successful experience plus delayed terminal safety feedback may make learning turn-entry and safe-speed behavior difficult. The unsafe prefixes receive positive progress reward before the failure penalty; an agent must learn a sufficiently long-horizon association between its earlier controls and later outcome.
- Reward magnitude, SAC entropy and critic estimation may interact with this credit-assignment problem. This offline audit does not isolate those factors.

### UNKNOWN

- Whether the objective rewards clean-fast more than clean-slow for matched geometry and environment conditions; only one clean episode was available.
- Whether V1 ranks unsafe-slow behavior appropriately; no authentic unsafe-slow full trace was available.
- Whether an alternative Stage 5 reward improves SAC or resolves the observed stationary behavior.
- Whether historic S5.5 trajectories used corrected F09 TaskContext semantics; the protocol predates and does not record the producer ID.
- Whether the current reward is sufficient for safe autonomous driving on geometry beyond this one Easy case.

## Decision and limitations

**`REWARD_V1_RETAIN`.** No arithmetic/integrity bug or full-episode ranking inversion at the actual training γ=.99 was demonstrated in the available trajectories. A new reward candidate is not justified yet because clean-fast/clean-slow and unsafe-slow matched outcomes are missing. RewardSpecV1 remains unchanged. The low-γ inversion and transient unsafe progress are concrete credit-assignment risks to remember, not proof of a bad reward formula.

This preflight does not resolve SAC stationary behavior, actor collapse, or driving competence. It does not authorize P2, training, a gamma change, or benchmark use. `benchmark_eligible` remains false.

## Reproduction and checks

From `autonomous-driving-rl/`:

```powershell
conda run -n metadrive_sb3 python scripts/audit_stage5_task_reward_alignment.py
conda run -n metadrive_sb3 python -m unittest tests.test_reward_metrics tests.test_stage5_meter_reward tests.test_stage5_task_producer -q
```

Observed in this run: offline replay script completed with 4 full TRAIN episodes / 1,845 rows and zero integrity mismatches. Targeted regression: **29 tests passed**, exit code 0. `git diff --check` passed. No scientific training or simulator was run for P1-R.
