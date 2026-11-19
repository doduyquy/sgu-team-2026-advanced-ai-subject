# Stage 5 status

Updated: 2026-11-19 (Prompt 2 integration)

## Current state

Stage 5 has a working SAC training and inference path, but no checkpoint has demonstrated clean driving performance sufficient for canonical evaluation. `stage5_sac` remains **CANDIDATE** with `benchmark_eligible=false`.

S5.1 and S5.2 are complete and merged. The S5.3 development work now has pilot and diagnostic evidence through S5.3L, plus one bounded fresh A/B experiment (S5.3M). The current data still do not identify a root cause for SAC's weak deterministic driving behavior.

| Area | Status | Evidence |
|---|---|---|
| Training environment, 409D encoder, TRAIN sampler | Complete | S5.1 regression and sampler isolation tests |
| SAC implementation, checkpoint integrity, inference | Complete | S5.2 tests, save/load and frozen inference |
| Pilot split | Complete | 156 `PILOT_TRAIN` / 24 `PILOT_DIAGNOSTIC`; 39/6 per tier; geometry IDs and SHA-256 |
| S5.3A–D pilot/ablations | Historical local results consolidated | `results/stage5/s5_3a*` through `s5_3d*`; keep their differing protocols explicit |
| S5.3E | Not verified | No matching local result artifact was found during this inventory |
| S5.3F–L diagnostics | Reported complete, artifact-checked | `results/stage5/s5_3f*` through `s5_3l*`; see each report for limits |
| S5.3M fresh tau A/B | Complete, one seed | `results/stage5/s5_3m_tau_ab_seed101_20261119_retry1/FINAL_REPORT.md` |
| Workbench Results Comparison V1 | Complete for verified canonical sources and TRAIN-only Stage 5 diagnostics | `STAGE5_COMPARISON_GUIDE.md`; no current canonical agent is eligible |
| SAC monitor Qt lifecycle | Hardened and covered by close/process-race tests | Stage 5 pilot monitor regression |
| Checkpoint binding to resolved plan | NO-GO pending versioned contract review | Existing `CheckpointBindingV1` is Stage-6/A*-specific; Stage 5 remains ineligible |
| Canonical VALIDATION/TEST | Not performed | Holdouts remained unused in these Stage 5 development runs |
| Benchmark certification | Not eligible | No certification or benchmark binding was activated |
| Prompt 2 regression | Complete | Full suite: 602 tests, 601 passed, 1 skipped, 0 failed/errors; exit 0. Stage 5/Workbench focus: 157 passed, exit 0. |

## S5.3M result at a glance

Both arms trained from the same fresh initial model for 10,000 steps. At 10k the treatment (`tau=0.001`) gained more route completion than control (`tau=0.005`) on all eight paired internal diagnostic cases, with no safety failure in either arm. The mean paired difference was `+0.002675` route-completion units. Both arms nevertheless remained below 1 km/h on average, were stationary for 100% of diagnostic steps under the `<1 km/h` rule, and recorded zero clean successes. These are development probes capped at 300 steps; nonterminal probes are censored, not official timeouts.

The treatment's online and target Q values were lower on the fixed reset observations. This shows a change in Q scale, not verified calibration: the A/B diagnostic did not estimate empirical soft returns for these checkpoints. Full per-case, per-tier, training and Q records are linked from the [final experiment report](STAGE5_FINAL_EXPERIMENT.md).

## Decision

Keep `tau=0.005` as the reference baseline. Retain the fresh `tau=0.001` 10k checkpoint as a **development candidate for independent confirmation only**. Do not use either run for official benchmark claims. Before canonical evaluation, confirm the candidate across independent training roots, freeze and bind the exact checkpoint SHA to the resolved experiment plan, and satisfy the Platform V1 evaluation protocol.

## Collaboration boundary

Stage 5 owns its training environment, SAC experiments, checkpoint provenance and Stage 5 diagnostics. It consumes Platform V1 contracts and must not change frozen contracts, RewardSpecV1, MapSuite manifests, Launcher semantics, canonical metrics or other teams' Stage implementations.

The Research Workbench has a SAC training monitor and a Results → Comparison subtab. The latter reads hash-verified Platform V1 canonical runs only when their registered agent is benchmark eligible, and separately reads manifest-verified S5.3M development records. It suppresses paired deltas when tracks, protocol, case/environment-seed schedule, metric identity, inference mode, input profile or action adapter differ. Stage 5 comparison is visibly labeled TRAIN-only and not benchmark data.

The only eligible comparison sources currently available are the six S5.3M checkpoints; the local model bundles are verified on the development machine. No registered method is currently benchmark-eligible, so Track A has no eligible source. See the [comparison guide](STAGE5_COMPARISON_GUIDE.md), [checkpoint-binding proposal](STAGE5_CHECKPOINT_BINDING_PROPOSAL.md), and [team handoff](STAGE5_HANDOFF.md).
