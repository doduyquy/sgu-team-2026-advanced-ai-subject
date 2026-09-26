# Experiments

This directory contains reproducible experiment definitions, runner scripts, and parameter sets for evaluating agent policies.

## Guidelines

- Scripts here should execute specific benchmarking sweeps using configurations from `configs/`.
- Runtime outputs will be saved to `runs/<run_id>/` (which is git-ignored).
- Significant and milestone results should be curated and placed in `results/`.
