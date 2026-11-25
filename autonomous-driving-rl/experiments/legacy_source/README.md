# Retired source references

`pre_cleanup_20261031/manifest.json` maps each old path to an exact source copy
and SHA-256. These inspection scripts and the old partial-trace agent audit were
retired during cleanup 0.6.1. They are historical references, not active tools;
relative project-root calculations in the copied files retain their original
meaning and are not rewritten to make them runnable from this archive.

Run current agents through `python -X utf8 -m src.launcher.cli run ...`. Use
`scripts/run_rule_based_experiment.py` and the maintained probe/replay/report
tools for reproducible TRAIN experiments. See `scripts/README.md`.
