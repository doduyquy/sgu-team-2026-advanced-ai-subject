# Maintained scripts

Run agents through `python -X utf8 -m src.launcher.cli run --agent <id> ...`.
Scripts below are tools supporting that runtime.

| Purpose | Script |
|---|---|
| Paired TRAIN experiments with source/configuration/input/action logs | `run_rule_based_experiment.py` |
| Verify and summarize completed experiments | `summarize_rule_based_experiment.py` |
| Replay recorded decisions | `replay_rule_based_trace.py` |
| Controlled lane-change safety probes | `probe_rule_based_lane_change.py` |
| Controlled LiDAR stop/restart probes | `probe_rule_based_obstacle.py` |
| Finish-visual manual preview | `demo_finish_visual.py` |
| Platform and Workbench contract audits | `audit_*.py`, `run_observation_action_audit.py` |
| Dataset export/verification and previews | `export_mapsuite_v1_*.py`, `verify_mapsuite_v1_dataset.py` |
| Full backend/GUI regression, isolated native processes and logs | `run_tests.py` |

Use a fresh run ID for every driving attempt. See the
[experiment workflow](../experiments/rule_based/README.md).

Old root `inspect_*.py`, the manual `test_course_env.py`, and the earlier
partial-trace `audit_rule_based_agent.py` were retired during 0.6.1 cleanup.
Their exact bytes/hashes are retained under
`experiments/legacy_source/pre_cleanup_20261031/` as historical references.
They are not maintained run commands. Frozen scientific snapshots and audit
generators stay in their original locations.

Run all regressions with `python -B -X utf8 scripts/run_tests.py`. Optional
`--group backend` or `--group gui` selects one group; `--output-dir` must be a
new directory. Qt tests use offscreen rendering. Separate processes avoid the
Windows Qt/Panda3D import-order conflict and keep source bytecode caches disabled.
