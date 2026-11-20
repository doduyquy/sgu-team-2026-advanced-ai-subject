# Workbench Comparison View V1

## Open and select results

1. Start the Research Workbench from `autonomous-driving-rl/` with the project's `metadrive_sb3` Python environment: `python -m src.workbench.app`.
2. Open **6. Results → Comparison**.
3. Select one source in each dropdown. Use **Refresh sources** after new result artifacts are added.
4. Read the comparison status banner before interpreting any difference. A paired difference is shown only when both sources pass their track's trust checks and share a compatible metric contract, input profile, action adapter, inference mode, split/protocol and exact case/environment-seed schedule.
5. Use **Open A/B result folder** for the local source directory or **Export comparison JSON…** for a compact comparison record.

## Two separate result tracks

### Canonical evaluation

Track A discovers only complete `canonical_run` artifacts accepted by the existing `RunArtifactRepository` as `VERIFIED` and whose registered agent is marked benchmark eligible. The five Platform V1 primary aggregate metrics are passed through from `summary.json`; the GUI does not recompute or rank them. Paired per-case differences require exact case ID and environment-seed schedule equality, matching platform contract fingerprints, input profile, action adapter and inference mode.

The current canonical agent registry has no benchmark-eligible methods, so this track may show no source. Completed but ineligible runs remain available in the existing Results browser and are not admitted here as official comparisons.

### Stage 5 development diagnostics

The view reads `results/stage5/<run>/protocol.json`, `artifact_manifest.json`, `analysis.json`, `anchor_observations.json`, and `q_anchor_diagnostics.json`. It verifies each declared file hash, the protocol's canonical self-hash, registered config SHA, internal TRAIN split SHA, arm status, checkpoint identities in case records, and fixed case/environment-seed schedule. Checkpoint bundles, when present locally, are checked against their checksum manifest and result-manifest hashes. If model bytes are absent, the result record stays visible with `MODEL BUNDLE MISSING`; it cannot be used to load/infer from that checkpoint.

The S5.3M A/B records are labeled **TRAIN ONLY — NOT BENCHMARK**. `PROBE_CENSORED` remains separate from `TIMEOUT`, clean success and safety outcomes. Route progress, action behavior and critic values are diagnostics. Q values are not empirical returns. They must not be copied into an official leaderboard.

## Reading `NOT DIRECTLY COMPARABLE`

The view suppresses paired deltas when tracks, metric contracts, split/protocol, case IDs, environment seeds, input profile, action adapter or inference mode differ; when a canonical agent is ineligible; or when a checkpoint SHA is absent from a Stage 5 result. It still shows source summaries side by side where available, with no `B − A` claim. A hash failure or malformed artifact is reported as `SOURCE VALIDATION FAILED`; the GUI does not repair it.

## Current local S5.3M sources

The view loads six real sources: A/B at 0, 5k and 10k from `s5_3m_tau_ab_seed101_20261119_retry1`. On the development machine the six model bundles are present and verified. Other clones may have only the compact result artifacts; those clones display the checkpoint SHA from the manifest and report the model bundle as missing. `runs/` is intentionally ignored and is not included in GitHub.

## View identity and limits

This is an additive `WorkbenchComparisonViewV1` presentation extension. It does not modify `WorkbenchResultsContractV1`, Launcher contracts, canonical metrics, eligibility, artifact authority, or the internal Stage 5 evaluator. It is not a leaderboard and does not select a winning checkpoint. The exact S5.3M values and limitations are in [STAGE5_FINAL_EXPERIMENT.md](STAGE5_FINAL_EXPERIMENT.md).
