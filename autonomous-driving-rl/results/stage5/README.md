# Stage 5 SAC ? experiment evidence

- [Final VALIDATION and TEST report](../../docs/stage5/final_product/FINAL_SAC_EVALUATION.md)
- [Paired VAL/TEST raw episode CSV](final-sac-evaluation/episodes.csv) and [packaged evidence manifest](final-sac-evaluation/manifest.json)
- [Recovery receipts](final-sac-evaluation-recovery/recovery-receipts.jsonl), [F09 diagnostics](final-sac-evaluation-recovery/localization-diagnostics.csv), and [regression proof](final-sac-evaluation-recovery/regression-proof.json)
- Publication tables for [Word](final-sac-evaluation-recovery/report-table.csv), [slides](final-sac-evaluation-recovery/slide-table.csv), and [poster](final-sac-evaluation-recovery/poster-table.csv)
- [Parallel training architecture](../../docs/stage5/STAGE5_KAGGLE_ARCHITECTURE.md) and [12-hour campaign results](../../docs/stage5/STAGE5_WINDOW12_RESULTS.md)
- Consolidation results [V1](../../docs/stage5/final_product/LOCAL_CONSOLIDATION.md), [V2](../../docs/stage5/final_product/LOCAL_CONSOLIDATION_V2.md), and [V3](../../docs/stage5/final_product/LOCAL_CONSOLIDATION_V3.md)
- [Inference checkpoint and demo](../../README.md#stage-5-sac-demo)

The paired evaluation used one deterministic checkpoint with `canonical_run=false`.
It produced 296 valid outcomes from 312 planned episodes. The 16 unresolved
Traffic OFF cases have diagnostic receipts and are excluded from complete-suite
rates. The model and CSV measurements retain their original bytes. Exported
JSON and JSONL use portable paths or redact local dependency paths. The package
manifest separates the original evidence checksums from the packaged file
checksums. Historical full-state checkpoints and replay are kept in the
research workspace.

[Parallel campaign workload data](parallel-training/workloads.csv) ? [Consolidation V3 comparison CSV](consolidation-v3/comparison.csv)
