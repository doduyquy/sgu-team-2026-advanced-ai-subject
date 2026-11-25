# P4 / 0.6.0 development record

Started 2026-10-30. This is TRAIN development, not canonical validation or a
benchmark certification. Version 0.5.0 is preserved under
`experiments/rule_based/versions/0.5.0`; before edits, the policy, extractor,
traffic, memory and junction files were SHA-256 checked against that archive.
All five matched. Its published SHA-256 values remain in the 0.5 release record.

## Intended change

- Shared immutable bounded lane geometry and actual rendered boundary metadata;
  unchanged 259D vector, new structured schema pending audit.
- Route-required adjacent changes before discretionary overtaking.
- Front/rear stopping gap and TTC, predicted swept body, LiDAR and overflow veto.
- Persistent maneuver target, precommit safe abort, postcommit least-risk stop,
  completion and cooldown. Never force a maneuver because waiting times out.
- Preserve P2/P3 guards and log nested lane-change decisions in every trace.

## Evidence and attempts

Implementation and measured experiments will be appended here. Each simulator
attempt has a fresh `runs/` directory containing exact TRAIN case/seeds,
configuration/source hashes, runtime, public inputs/actions/diagnostics and final
status, including unsuccessful attempts. Controlled probes are interventions,
not ordinary driving scores.

- Initial pure tests: 80 existing tests exposed a 51-key diagnostics payload
  during junction states. Removed redundant top-level `curvature_per_m` instead
  of raising the shared 50-key budget. Re-run: 80/80 passed.
- Added 16 P4 tests; first run had two assertion/fixture failures (assumed exact
  key count and a leader too far away to trigger slow-follow). Corrected fixtures;
  combined policy/contract/replay suite: 133/133 passed.
- `rule_based_v060_draft01_easy3_20261030`: COMPLETED, paired 0.5/current,
  three Easy TRAIN/env0, 3/3 success each, zero crash/road/solid-boundary flags.
  This draft predates the return-after-pass addition; do not call it final evidence.
- `rule_based_v060_probe01_20261030`: FAILED at setup, before simulation;
  MetaDrive package lacks `__version__`. Runner now records optional module
  version and installed package/runtime information without assuming that field.
- `rule_based_v060_probe02_20261030`: COMPLETED_WITH_FAILED_EXPECTATIONS,
  neither scenario reached its spawn condition (required an 80 m single straight
  edge, absent in the selected TRAIN geometry). Both drove to destination without
  safety flags, but this does not validate a maneuver. Probe now requires 50 m;
  the policy's own maneuver length/end-buffer conditions remain unchanged.
- `rule_based_v060_probe03_20261030`: COMPLETED, 2/2 physical scenarios passed.
  Parked-leader pass completed at step 150; fast-rear scenario observed a rear
  veto then completed a change after clearance (170 steps). Neither produced
  crash, road departure or solid-boundary contact. This draft precedes the final
  between-sample motion margin and speed/alignment admission guard.
- First full integrated suite ran 268 tests; four launcher tests hit sandbox
  filesystem write denial for their temporary `runs/run_*` directories. This is
  recorded as an environment-limited attempt, not a passing suite. Re-run with
  approved workspace-write execution will be recorded separately.
- Review added relative travel inflation across prediction samples, an initial
  <=30 km/h/aligned admission guard (prepare at <=24), and measured road-frame
  body-envelope checks while tracking a maneuver-frame path. A permitted broken
  divider is not treated as the outer road boundary.
- Expanded P4 suite to 20 tests. A regression test caught the mandatory-end
  speed bound overwriting the 24 km/h prepare cap; changed composition to min.
  All 20 passed. Integrated approved-write run: 272/272 passed in 32.806 s,
  console `runs/v060_integrated_checks_02.log`.
- Probes 04 and 05: four scenarios passed each (overtake, rear veto, required
  route change, safe precommit abort), no crash/road/solid-boundary flags.
  04 used raw action floats, 05 canonical float32. Mandatory probe intentionally
  restricts availability of the next route edge and stops before it; original
  physics/rendered lanes stay intact, so this is not a normal driving score.
- Started final-family12 and final-env1 attempts, then requested clean STOP
  before a safety-review fix. Both are INTERRUPTED, not acceptance runs. The
  family12 draft continues separately to preserve the complete paired baseline.
- Probe 06 revealed an overly conservative persistent heading envelope could
  immobilize a nearly completed change despite a clear target gap. Kept actual
  center-offset inflation and an explicit current-body check; the initial
  heading envelope now settles over the configured 0.4 s controller response,
  with future orientation following the planned tangent. New physical runs are
  required; do not count earlier probes as evidence for this source.
- Probe 06 final status COMPLETED_WITH_FAILED_EXPECTATIONS: parked-leader pass
  stalled at 82.9% progress for 1100 steps; no crash/road/solid-boundary flags.
  The other three scenario expectations passed (fast-rear expectation alone does
  not certify completion; its actual maneuver_completed=false remains in JSON).
- Probe 07: COMPLETED, four expectations passed, 586 steps replayed exactly
  with frozen source/input types, zero action/diagnostics mismatches.
- `rule_based_v060_acceptance_env1_20261030`: a real extraction error occurred
  on Hard/SCTXrCS/env1 when lane endpoint rounded to a regular sample distance.
  Logged error/partial trace preserved. Added post-rounding endpoint deduplication
  and regression for both sides of the rounding threshold. Associated acceptance
  runs stopped cleanly and are not used as final driving evidence.
- Final checks 03: 276/276 passed in 49.040 s (23 P4 tests plus report integrity
  coverage included); console `runs/v060_integrated_checks_03.log`.
- Probe 08 uses the final policy and deduplicating extractor: COMPLETED, 6/6
  physical scenarios, no crash/road/solid-boundary flags. Steps: pass 150, rear
  veto then pass 170, mandatory right 171, mandatory left 171, precommit abort 95,
  postcommit blocker/stop/clear/resume 173. Six frozen replays cover all 930 steps.
- Final driving acceptance is split into roads9/env0, three distinct Extreme
  env0 shards, and targeted4/env1. Case coverage stays the same 16 unique TRAIN
  cases used for 0.5.0. Reports select the original completed 0.5 current variant
  explicitly; they verify hashes, trace counts, physical case definitions and
  simulator configuration, rather than renaming/copying raw folders. The earlier
  duplicated draft paired-family run was cleanly stopped, keeping partial data.
- First split release candidate reached 9/9 ordinary-case success, but Extreme
  SCXOCrTYCS exposed a lifecycle bug: approach entry moved inside 35 m during an
  existing change, stopping it at ~79% progress. Its completed TIMEOUT result
  remains in `rule_based_v060_release_extreme_scxo_20261030` (completion .307565,
  no crash/road/solid-boundary flags). Init-only approach veto is now separate
  from active guard; actual junction lock/sensors/boundaries/traffic still veto.
  Added a dedicated regression, 24 P4 tests now pass. Earlier incomplete shards
  were stopped; final acceptance reruns use the `accept02_*` prefix.
- Integrated checks 04 on final lifecycle source: **277/277 passed**, 79.750 s,
  console `runs/v060_integrated_checks_04.log`. Probe 09: **6/6 passed**, all 930
  steps replay exactly against frozen policy/helpers/input, no action/diagnostic
  mismatches. The later count supersedes the earlier 276-test provisional count.
- The accept02 SCXOCrTYCS attempt completed its change, then an observed
  oncoming actor drove into the stopped ego upstream of the junction. Result:
  CRASH_VEHICLE at step 717, completion .335753. This is an acceptance failure,
  not hidden by the six passing controlled probes. Record retained verbatim.
- Conservative response: optional passing/return is prohibited whenever a
  junction/passage is observed within the full 50 m window. Mandatory positioning
  keeps its upstream 35 m gate. Added the 49 m discretionary-veto regression.
  Separately, commit now accounts for actual rotated body reaching the divider
  with .05 m buffer, even before center progress reaches .22. Its regression
  initially exposed a NumPy bool in diagnostics; normalized to native bool.
  Final pure P4 suite: 26/26 passing. Interim accept03 runs were cleanly stopped.
- Integrated checks 05 on the final policy/helper/extractor: **279/279 passed**
  in 42.946 s; console `runs/v060_integrated_checks_05.log`. Probe 10: **6/6**
  passed, 929 recorded canonical-action steps; frozen replays stored with
  `accept04_replay_*` names. Final driving evidence uses only `accept04_*`.
- CLI smoke 01 (earlier helper): COMPLETE, Hard/XTOCS TRAIN/env0/agent42, clean
  success. Smoke 02 repeats the official launcher on the final helper and is
  recorded separately; it is not added as a seventeenth unique training case.
  It completed clean success in 1063 steps, source/configuration and 1063-line
  launcher trace retained. Probe10's final postcommit scenario is 172 steps,
  one less than probe09 after earlier body-based commit; final total is 929.
- Evidence collection attempt 01 failed before creating report/snapshot folders:
  PowerShell redirected console logs as UTF-16, while the collector assumed
  UTF-8. Added BOM-aware decoding; original console and its SHA-256 remain
  unchanged. Failure console: `runs/v060_evidence_collection_console.log`.

## Final acceptance and freeze

Finalized 2026-10-31 Asia/Saigon; source snapshot UTC
`2026-10-30T17:02:55.695235+00:00`. Evidence collection attempt 02 completed.

- Final accept04 coverage: 16 unique TRAIN cases, 13 success / 3 timeout,
  zero crash/road/solid-boundary and no technical errors. Baseline 0.5 also
  13/16; no gained or lost success. Exact paired definitions, simulator configs,
  source/config hashes and trace rows verified before publication.
- Natural traces contain six completed lane changes with passing reason across
  five cases. These count completed lane-change FSM maneuvers; physical passing
  of the blocker is measured separately in controlled pass probes.
- Hard SCXOCS/env0 completion .9919631543 -> .9917536386 (150.1 ->148.5 s), env1
  .9923589937 -> .9920315255 (148.2 ->146.8 s); both still SUCCESS. Differences
  retained. No attribution to policy alone because public extractor also changed.
- Three unresolved Extreme timeouts: CrXROSTR .4031247509, SCTXORyCCS
  .4726938856, SCXOCrTYCS .4514835611, all geom0/env0. No forced launch/lane change.
- Frozen 0.6.0 source/config at `experiments/rule_based/versions/0.6.0/`.
  Nine earlier source groups copied verbatim from their raw snapshots to
  `versions/dev060_*`, clearly marked retrospective archives and not releases.
  The full index has 37 attempts: 16 COMPLETED, 2 CLI COMPLETE, 16 INTERRUPTED,
  1 FAILED, 2 COMPLETED_WITH_FAILED_EXPECTATIONS. There are no unclosed runs.
- `release_verification.json` rechecks every recorded source hash in both 0.5
  and 0.6 archives and actually imports flat archived 0.6 policy/helpers/input
  to replay all 929 physical steps exactly. No release snapshot was modified.
- Curated JSON/CSV, test verification, raw-run provenance index, source-group
  index and maneuver events are under `results/audits/rule_based/v0.6.0/`.
  Raw simulator/CLI outputs stay under their original `runs/` folders. Tests
  command: conda metadrive python, `-B -X utf8 -m unittest`, the 11 named modules
  from integrated-check calls, `-q`; original UTF-16 console/hash retained.
- P4 is implemented and accepted with the documented scope. P5, full 108 TRAIN,
  controller/latency consolidation, legal signal/sign source, contract audit and
  canonical VALIDATION/TEST remain. Benchmark eligibility stays false.

Exact final integrated check, from `autonomous-driving-rl` with the conda
`metadrive` interpreter (`C:/Users/DANG KHOA/.conda/envs/metadrive/python.exe`):

```powershell
python -B -X utf8 -m unittest tests.test_finish_visual tests.test_random_naive_agent tests.test_rule_based_agent tests.test_rule_based_corridor tests.test_agent_experiment_logging tests.test_agent_contract tests.test_launcher_core tests.test_rule_based_report tests.test_rule_based_p2_p3 tests.test_rule_based_replay tests.test_rule_based_p4 -q
```

Final evidence export and archive check (original executions, not commands to
overwrite existing artifacts):

```powershell
python -B -X utf8 experiments/rule_based/collect_v060_evidence.py --output-dir results/audits/rule_based/v0.6.0 --freeze
python -B -X utf8 results/audits/rule_based/v0.6.0/verify_release_archive.py
```

## Simulator inspection

Pinned MetaDrive PGBlock constructs each lane's LEFT boundary and only the
rightmost lane's RIGHT boundary (`pg_block.py:create_in_world`). Thus the shared
boundary to the right is rendered from the right neighbor's left metadata.
Reading the current lane's nominal right line alone can authorize crossing a
solid rendered line. P4 extraction must respect this ownership. Lane lateral
coordinates are positive RIGHT; public ego y and steering positive LEFT.
