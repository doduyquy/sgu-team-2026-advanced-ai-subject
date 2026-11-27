# S5.4 — SAC learnability recovery protocol

Status before run: **pre-registered; not yet a scientific result**. The run driver is `scripts/run_stage5_learnability.py`; the exact machine-readable protocol generated before the first training step is `results/stage5/s5_4_learnability_seed101_20261119/protocol.json`. Raw checkpoints and runtime records belong under the ignored `runs/stage5_learnability/` directory.

## Question and treatment

Does a fixed, versioned group-wise physical-unit scaling after the frozen 409D encoder recover observation-dependent SAC control and useful movement compared with the current SAC input? The sole arm difference is Treatment B's `pilot_groupwise_physical_v1` transform. The measured rationale and limits are documented in [`S5_4_MECHANISM_DECISION.md`](S5_4_MECHANISM_DECISION.md).

Both arms start from one newly initialized, shared SAC model state (including actor, critics, entropy state and fresh optimizers), root seed 101, and the same declared hyperparameters. A uses no transform; B uses the fixed transform. Neither arm loads a historical trained checkpoint. RewardSpecV1, action contract, observation encoder schema, 409D dimension, ReLU architecture and training schedule stay unchanged.

## Geometry and seeds

The existing verified internal split is `results/stage5/pilot_split_v1.json`, derived from the locked MapSuite V1 geometry manifest. Training uses exactly eight `PILOT_TRAIN` Easy IDs, selected before training by sorting within each family by generation seed then ID and taking 3 SCCS, 3 SCSS, and 2 SCS geometries. Diagnostic evaluation uses all six Easy `PILOT_DIAGNOSTIC` IDs. IDs, geometry SHA-256 values, split identity and environment seeds are recorded in `protocol.json`.

The sampler is episode-level ROUND_ROBIN across the frozen eight-ID pool. Its environment seed is separately derived from `SHA256("train_env_seed:101:{episode_index}")`, taking the first 32 bits modulo 1,000,000; the sampler episode index begins at one. The fixed primary diagnostic uses schedule seed 5404, with a second seed 5405 reserved only for a candidate satisfying the early-transfer primary gate. Geometry generation seed is never used as the unique geometry identity.

No VALIDATION or TEST geometry is accessed. `PROBE_CENSORED` is a distinct outcome label and is excluded from full-episode success, safety and timeout rates.

## Training budget and checkpoints

Each arm has a maximum of 100,000 environment decisions and mandatory snapshots at 0, 10k, 30k, 50k and 100k unless the pre-registered early-transfer criterion is met. The model learns continuously within one `learn()` call per arm; evaluation runs in a separate Python process against saved checkpoint weights. Evaluation does not update weights, optimizers, replay buffer, normalization statistics or training sampler.

Common SAC settings: learning rate 0.0003; buffer 100,000; batch 256; gamma 0.99; tau 0.005; learning starts 1,000; automatic entropy coefficient; one gradient step per environment step after warm-up. The model has the existing two 256-unit ReLU layers. Actual optimizer updates, completed episodes, sampled geometry steps, throughput, RAM and checkpoint hashes are recorded.

An interrupted arm is marked incomplete. Model ZIP alone is not treated as an exact resume point; the protocol does not concatenate a new segment onto a partially saved run.

## Evaluation

Every checkpoint receives deterministic probes on the same six unseen internal TRAIN geometries and fixed environment seeds, capped at 300 decisions. These probes record movement, stationary fraction, sustained moving run, action distributions, route-progress delta from reset, actor sensitivity and hidden-layer activation diagnostics. Nonterminal caps remain `PROBE_CENSORED`.

Full episodes use the actual route-aware horizon and existing Platform V1 terminal classifier at 30k and 100k. A 50k full evaluation is run only when its preregistered quick-probe criteria are met. Outcomes include clean success, safety failure and timeout separately. Full episodes, not fast probes, are the source for those outcome rates.

At 30k, lack of clean success alone is not a stop. At 100k, the learnability budget is exhausted; no automatic extension is allowed. If primary and confirmation evaluation meet the frozen early-transfer criteria, the Easy-only campaign stops at that checkpoint and the result is labeled `GO_MULTI_MAP`; this task never starts multi-map training.

## Candidate/transfer rule

Early `GO_MULTI_MAP` requires at least two distinct diagnostic geometry clean successes in full episodes at the primary fixed schedule, mean full-episode stationary fraction no greater than 0.70, no more safety failures than clean successes, deterministic action standard deviation of at least 0.01 in one action dimension across fixed observations, and at least one clean success under the second fixed environment seed without safety failures dominating. Speed alone cannot qualify a candidate. Otherwise training continues to its registered budget.

One root seed is a screening experiment, not evidence of statistical superiority. An early GO is an engineering development decision, not benchmark eligibility. `stage5_sac` remains `CANDIDATE` and `benchmark_eligible=false`.

## Reproduction

From `autonomous-driving-rl/` in the project Python environment:

```powershell
python -m unittest tests.test_stage5_learnability tests.test_stage5_sac tests.test_stage5_pilot -v
python scripts/run_stage5_learnability.py --technical-smoke
python scripts/run_stage5_learnability.py
```

The smoke is technical only and is not included in scientific results. The full run refuses to overwrite existing run and result folders. Its output protocol hash must be preserved alongside the run and checkpoint manifests.
