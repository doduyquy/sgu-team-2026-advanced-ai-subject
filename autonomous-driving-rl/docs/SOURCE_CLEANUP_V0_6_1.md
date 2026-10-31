# Source cleanup — 0.6.1

Started 2026-10-31 (Asia/Saigon), after P4/0.6.0. This is code organization and
maintenance, not P5 or a new driving benchmark. The existing CLI agent IDs remain
the entrypoints. Source cleanup must preserve policy actions/diagnostics and
public contract semantics, with checks recorded below.

## Scope

- Remove obsolete root inspection/manual-run scripts superseded by maintained
  audits and launcher tests. Retire the old partial-trace rule-based audit tool,
  preserving its original bytes for historical experiment reproducibility.
- Remove unused calculations/state/imports after checking consumers. Keep
  controller helpers, public compatibility exports and platform fixture agents.
- Put generated hologram textures in ignored runtime cache instead of `src`.
- Guard the legacy evaluator's main entrypoint against simulations on import.
- Update the project map, tool/agent guides and version changelog.
- Delete regenerable source/test bytecode caches, preserving experiment records,
  benchmark data, pinned contracts and all released source snapshots.

## Evidence

- Original sources of seven retired scripts copied verbatim to
  `experiments/legacy_source/pre_cleanup_20261031/manifest.json`, with old/new
  paths and SHA-256. Active root inspection/manual scripts and partial-trace
  audit removed. Existing immutable research snapshots remain untouched.
- Removed dead preview-curvature calculation and unused lane-change elapsed
  counter. Planned 84 unused imports across 32 runtime files; kept five imports
  in the pinned action adapter so its exact file hash remains compatible with
  historical replay. Actual cleanup removes 79 imports across 31 files. Public
  exports/known consumers/string annotations were excluded from that plan.
- Backend regression attempt 03: **423 tests passed**. Live 0.6.1 replay:
  **1992/1992 steps** match full decisions/diagnostics and canonical actions across
  six physical traces (929 steps) and Hard/XTOCS TRAIN (1063 steps). All public
  policy parameters equal 0.6.0. This is decision replay, not a new driving score.
- Two aggregate discovery attempts crashed in native QtWidgets loading after
  MetaDrive/Panda3D imports (Windows 0xC0000005). Standalone Qt/QApplication
  succeeds. Tests are therefore run in backend and GUI processes separately.
- First isolated GUI attempt: 124 tests, four failures in stale assertions:
  expecting only four registered agents, treating the historical fixture-registry
  hash as the hash of the expanded live registry, and two Windows short/long
  temp-path comparisons. Updated tests compare the full registry before/after
  import, retain the exact historical four-fixture hash, and resolve path aliases.
  No locked hash or benchmark contract changed to make tests pass.
- Isolated GUI attempt 05: **124 tests passed**. The maintained one-command
  runner then passed the complete suite again: **423 backend + 124 GUI = 547**
  tests, with commands/source hashes/UTF-8 logs under `final_checks/`. The runner
  keeps native processes separate and disables bytecode caches in child workers.
- Official CLI smoke `rule_based_v061_cleanup_smoke_20261031`: COMPLETE,
  Easy/SCS/geom0/env0/agent42, clean success in **418 steps**, no safety event.
  Per-step public inputs/actions and exact source/configuration are logged.
- Three source PNG textures were regenerated in `runs/.cache/finish_visual/`
  and verified byte-for-byte identical before removal. Removed 10 regenerable
  bytecode-cache directories and 3 generated PNGs from source; cleanup recorded
  890334 bytes. No experiment/result directory was deleted.
- Seven old entry scripts retired and one maintained `scripts/run_tests.py`
  added. Active Python source count: **83 → 77** (excluding tests/archives).
  The project root now contains README/AGENTS and the actual project folders.
- Raw commands,
  source inventories, test logs and per-step replay outputs are retained under
  `runs/source_cleanup_v061_20261031_01/`; official CLI smoke has its own fresh
  run ID. No historical driving result is relabeled as a new experiment.

## Commands and scope

Normal Stage 1 execution still uses `--agent stage1_rule_based`; the registered
version is now 0.6.1. The five `rule_based_*.py` runtime files form one controller.
Stage 0 remains `--agent stage0_random_naive`. The legacy CourseEnvV1 wrapper is
retained for its documented compatibility role, with an import-safe evaluator.

```powershell
python -B -X utf8 scripts/run_tests.py
python -X utf8 -m src.launcher.cli agents
```

Maintenance evidence: `results/audits/source_cleanup/v0.6.1/` and the immutable
`experiments/rule_based/versions/0.6.1/` source snapshot. Parent 0.6.0 archives
and prior records remain intact. The 16-case P4 driving score remains attributed
to 0.6.0; this maintenance check has one new TRAIN smoke and exact recorded-input
replays. P5, full 108 TRAIN and benchmark/contract audits remain pending.
