# Logging Design & Templates

This directory documents the lightweight, technology-neutral logging specification for autonomous driving experiments.

## Core Principles

1. **Simplicity & Portability**: Logging relies strictly on standardized plain-text formats (`JSON` and `CSV`). Heavy external logging frameworks (e.g., Weights & Biases, MLflow, TensorBoard, external database services) are intentionally avoided at this stage.
2. **Source of Truth**: Raw `CSV` episode records and `JSON` metadata/summaries serve as the canonical source of truth. Plots and markdown summaries are derived visualizations.
3. **Reproducibility**: Every run captures its exact configuration parameters (environment, seeds, hyperparameters) and execution metadata (git commit, git branch, dirty status, Python version, OS, MetaDrive version and commit).
4. **Hierarchical Granularity**: Run-level metadata and episode-level metrics are standard. Step-level logging is heavy and should be strictly optional, disabled by default.

## Proposed Run Artifact Structure

Each experiment run directory in `runs/<run_id>/` will follow this structure:

```text
runs/<run_id>/
├── config.json         # Exact experiment configuration used
├── metadata.json       # Environment, runtime, and git reproducibility metadata
├── episodes.csv        # Per-episode tabular metrics
├── summary.json        # Aggregate benchmark statistics across all episodes
├── stdout.log          # Console output log
└── plots/              # Optional derived visualization plots
```

## Template Files

Standard example templates are provided in `configs/templates/` and referenced here:
- `experiment.example.json`: Experiment configuration parameters.
- `metadata.example.json`: Environment and execution metadata.
- `summary.example.json`: Aggregate evaluation statistics.
- `episodes.example.csv`: Per-episode evaluation record format.
