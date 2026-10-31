# Experiments

This directory contains stage-specific training/experiment entry points outside the common evaluation runtime.

`train_astar_heuristic.py --preset smoke|standard` collects public local graphs on
TRAIN and fits a Stage 6 cost-to-go heuristic. `--plan` shows its budget without
starting the simulator. Fixed settings live in `configs/experiments/astar_learning_v1.json`.
See [A* learning workflow](../docs/ASTAR_LEARNED_HEURISTIC.md).

Canonical evaluation is orchestrated by `src/launcher` and the Research Workbench.
Learned stages produce frozen agents/checkpoints evaluated through the same Platform V1 contracts.

Rules:

- develop/train on TRAIN scenarios;
- use VALIDATION for selection/tuning;
- freeze the method before TEST;
- do not create alternate private benchmark definitions here;
- runtime scientific outputs belong under `runs/<run_id>/`;
- curated report-level summaries may be committed under `results/` when appropriate.
