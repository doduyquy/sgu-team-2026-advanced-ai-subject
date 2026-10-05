# Experiments

This directory is reserved for stage-specific training/experiment definitions that do not belong in the common platform runtime.

Canonical evaluation itself is orchestrated by `src/launcher` and the Research Workbench. Future learned stages may add training scripts/configurations here, but they should produce agents/checkpoints that are evaluated through the same Platform V1 contracts.

Rules:

- develop/train on TRAIN scenarios;
- use VALIDATION for selection/tuning;
- freeze the method before TEST;
- do not create alternate private benchmark definitions here;
- runtime scientific outputs belong under `runs/<run_id>/`;
- curated report-level summaries may be committed under `results/` when appropriate.
