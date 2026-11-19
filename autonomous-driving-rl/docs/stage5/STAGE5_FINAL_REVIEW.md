# Stage 5 final technical review

Review date: 2026-11-19
Branch: `feat/stage5-sac-pilot-diagnostics`
Reviewed HEAD: `483f3107ecd97348e475fb17303f5e8582b8851b`
PR: [#37 — S5.3A: SAC pilot and learning diagnostics](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/pull/37)

## Decision

**GO for human merge review.** The current change set has no merge-blocking correctness, integrity, or frozen-contract issue identified in this review. This is a code-review decision only: it does not certify SAC as a successful driving policy, does not grant benchmark eligibility, and does not merge the PR. The PR remains open for its owner/reviewer to decide.

**NO-GO for official Stage 5 benchmark certification at this time.** `stage5_sac` is still `CANDIDATE` with `benchmark_eligible=false`; the resolved experiment plan does not bind the SAC checkpoint SHA, the evidence is from one training root, and the policy has not shown clean driving success.

## Review scope and findings

### Correctness

- `Stage5TrainingEnv.reset()` initializes the previous route completion from MetaDrive's reset information. `step()` calls the canonical `continuous_box2_v1` adapter, makes one MetaDrive decision step, computes signed route delta and delegates reward calculation to the Platform V1 reward helper.
- The two action dimensions remain `[steering, throttle_brake]` in `[-1,1]`. The adapter preserves the values and rejects invalid shape, non-finite, and out-of-range actions; no hidden throttle remapping was found.
- MetaDrive uses 0.02 s physics substeps and `decision_repeat=5`, so the policy decision interval is 0.1 s (10 Hz). A single Gymnasium transition therefore corresponds to one control decision, with the same action applied over the five internal physics substeps.
- The timeout path remains Gymnasium truncation; safety outcomes use the existing safety-first episode classifier. SB3's timeout-aware replay handling bootstraps from the terminal observation for pure time limits. Reward computation and action-fidelity regression tests pass.
- `Stage5SacAgent` loads a checksum-verified bundle, verifies the encoder fingerprint and predicts deterministically from public `AgentInputV1`. It does not update weights, optimizer, replay buffer, or observation normalization during inference.
- The reset-progress reward defect found during early development is fixed in the current source. The first S5.3A result predates that fix and is explicitly retained as historical evidence, not treated as directly interchangeable with the fresh S5.3M experiment.

### Scientific integrity

- Training sampling is restricted to verified TRAIN manifest records. Pilot mode limits the sampler to the 156 `PILOT_TRAIN` geometry IDs and rejects diagnostic geometry IDs even through explicit reset options.
- The S5.3M diagnostic schedule is a fixed eight-geometry subset of the internal 24-case diagnostic pool, with environment seeds recorded separately from geometry-generation seeds. The reported 8/8 nonterminal probes at 10k are `PROBE_CENSORED`, not timeout, success, or evidence of safety.
- Canonical and development results remain separate in Results → Comparison. Paired deltas are suppressed when case/environment-seed schedules, metric identity, protocol, inference mode, input profile, or adapter differ. Stage 5 records are visibly marked `TRAIN ONLY — NOT BENCHMARK`.
- S5.3L's Q-to-soft-return gap is an observed discrepancy on a bounded set of frozen-policy states and interventions. It does not establish that critic overestimation caused the stationary deterministic policy.
- S5.3M changed only `tau` between two fresh arms under the registered protocol. B's paired route-progress delta was higher by `+0.002675` at 10k, but both arms remained stationary for 100% of diagnostic steps and had zero clean successes. A single training root does not support a superiority claim.
- No Stage 5 official VALIDATION or TEST result is claimed; no TEST data were used for tuning or selection.

### Reproducibility and provenance

- S5.3M records the registered config/protocol, split identity, seed, training source commit, per-case schedule, checkpoint steps and SHA-256 values in compact committed artifacts. All six local 0/5k/10k bundles pass `verify_checkpoint_integrity()`.
- The independent local backup contains 26 checksum-verified Stage 5 checkpoint bundles, including the six S5.3M bundles and a 5k plumbing-smoke bundle. Its per-file hash manifest is at `D:\SGU\Backups\XeTuHanh_Stage5_20261119\STAGE5_BACKUP_MANIFEST.json`.
- Matching raw checkpoints and diagnostic traces are gitignored and cannot be recovered by cloning GitHub alone. The training artifacts state that matching seeds do not guarantee bitwise retraining; model and replay persistence also do not restore every PRNG, optimizer, sampler cursor, or simulator state.
- Protocol canonical hashes, configuration hashes, S5.3L per-replicate result hashes, raw trace byte counts and hashes, and S5.3M artifact hashes were checked against the actual files during handoff.

### Platform and team integration

- The full repository regression at this HEAD completed with 602 tests: 601 passed, 1 skipped, 0 failures/errors (exit code 0). The focused Stage 5/Workbench suite completed 157/157 (exit code 0).
- Frozen Platform V1/Launcher/results contract hash tests passed. Changes in this PR add a presentation subtab and tests; they do not alter the scientific result contract, canonical metrics, action contract, reward contract, official manifests, or TEST policy.
- No implementation changes to Stage 1, 2, 4, or 6 agent logic were found in the Prompt 2 integration commit.
- Workbench startup, historic pilot telemetry, A/B comparison, JSON export, and deterministic viewer startup on a frozen TRAIN checkpoint were exercised on the Windows desktop. The viewer was stopped during the demo preview; no result was recorded from that partial preview.

## Non-blocking limitations

1. Stage 5 resolved-plan checkpoint binding needs a reviewed, versioned contract extension. Existing `CheckpointBindingV1` carries Stage-6/A* semantics and was not repurposed.
2. A/B evidence uses one root seed and eight diagnostic geometries; it is exploratory and not a benchmark result.
3. Raw bundles are backed up on the same D: volume. This protects against accidental loss in the source checkout but not a disk/volume failure. A team-approved separate-volume or institutional artifact location is still needed for durable archival.
4. There is no current eligible canonical comparison source on this checkout. The GUI can admit verified eligible runs, but Track A is empty here.

## Verification evidence

- Full repository suite: **602 tests, 601 PASS, 1 SKIP, 0 FAIL/ERROR**, exit 0; 871.327 s; Windows `metadrive_sb3` Python 3.10.21 environment with Qt preloaded and `KMP_DUPLICATE_LIB_OK=TRUE` for this regression invocation.
- Focused Stage 5/Workbench suite: **157 PASS**, exit 0.
- `py_compile` on changed Python files, `git diff --check`, contract-hash regressions, 25 checkpoint bundles under `runs/stage5_pilot` and `runs/stage5_tau_ab` verified before backup, plus the separate `runs/checkpoints/stage5_sac_smoke` bundle verified after inclusion.
- Backup verification: 349 files, 1,144,514,034 bytes; all source/copy file SHA-256 values match; all 26 checkpoint bundles pass the repository verifier.
- Live Workbench smoke: seven expected top-level tabs, COMPLETE 10k monitor run with telemetry and checkpoint available, eight paired S5.3M cases, censored classification preserved, JSON export verified with `official_leaderboard=false`; application closed cleanly.
- Frozen viewer smoke: native MetaDrive reported `Render Mode: onscreen`, loaded the frozen 10k TRAIN checkpoint and fixed internal TRAIN schedule, and was stopped before an episode outcome. No training or evaluation result was added.
