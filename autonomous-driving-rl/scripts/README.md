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
| Kiểm định ngoại tuyến và xuất bằng chứng G0 của Stage 4 (chưa thu dữ liệu/huấn luyện) | `stage4.py audit`, `stage4.py publish-audit` |
| G1 bước 1: kiểm chứng vận tốc và dấu lái trên TRAIN, hồ sơ nguồn/bản đồ/số đo từng bước | `audit_stage4_telemetry.py --run-id <new-id> --expected-logging corrected` |
| G1 bước 2: thu thử Stage 1, kiểm tra nguồn gốc/ghép bước và phát lại; xuất tổng hợp có băm | `collect_stage4_pilot.py collect --run-id <new-id>`, `collect_stage4_pilot.py publish --run-id <id> --output results/audits/stage4/g1/<new-dir>` |
| G1 bước 3: đo thời gian/RAM/dung lượng toàn cây collector, kiểm tra môi trường, ngoại suy theo tier | `profile_stage4_resources.py profile --run-id <new-id>`, `profile_stage4_resources.py publish --run-id <id> --output results/audits/stage4/g1/<new-dir>` |

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
