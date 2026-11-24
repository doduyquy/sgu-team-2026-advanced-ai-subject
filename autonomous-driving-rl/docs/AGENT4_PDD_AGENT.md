# Agent 4 PDD hybrid — release 1.0.0

This release packages the best verified PDD candidate as a normal launcher agent.
It preserves the byte-identical supported40 runtime recorded on 23 November 2026,
the learned weights, and its Stage 1 assistance. It does not retrain the model.

| Item | Value |
|---|---|
| Agent ID | `stage4_pdd_hybrid` |
| Entrypoint | `src/agents/agent4_pdd_agent.py:Stage4PDDHybridAgent` |
| Method | ExtraTrees trajectory/speed prediction + controller + Stage 1 |
| Input/output | `STATE_DECISION_V1` / `continuous_box2_v1`, 10 Hz |
| Model | `artifacts/agent4_pdd/1.0.0/model_bundle.joblib` |
| Model SHA-256 | `3893f571441d9613031b8388b46e5b9db3cf6e364a2742f28eae9327be5e6b66` |
| Offline status | `DEV_NEEDS_FIX`; WAIT/geometry qualification remains open |
| Eligibility | Sandbox/audit; benchmark eligibility is false |

## Running

Use the project’s MetaDrive Python environment. If the model dependencies are missing, install them in that environment.

```powershell
python -m pip install scikit-learn==1.7.2 joblib==1.5.2
```

From the project root, list the agent or run the integration tests and simulator.

```powershell
python -X utf8 -B -m unittest discover -s tests -p test_agent4_pdd_agent.py -v
python -X utf8 -B -m src.launcher.cli agents
python -X utf8 -B -m src.launcher.cli run --mode SANDBOX --agent stage4_pdd_hybrid --tier Medium --sequence SCXCS --geom-seed 14 --env-seed 1 --agent-seed 42 --render NATIVE --wandb DISABLED
```

Use `--render OFF` for headless evaluation. Each launcher run creates its own
run directory and records public inputs, decisions, provenance, and episode results.
The agent appears through the canonical registry, like Stage 0 and Stage 1.

The trusted joblib checkpoint was trained with scikit-learn 1.7.2. Recorded runtime
dependencies were NumPy 2.2.6, joblib 1.5.2, MetaDrive 0.4.3; the preserved trainer
module also imports pandas. The wrapper fails on a scikit-learn version mismatch
instead of silently loading incompatible weights. No PyTorch/CARLA/PDD download
or connection to the sibling dataset workspace is needed for inference.

## Historical driving evidence and limits

The supported profile reached the destination cleanly in five inspected TRAIN
Medium/Hard cases. Times were 62.3, 99.1, 84.0, 135.5 and 118.5 seconds; maximum
observed speed was 38.82 km/h. All five were slower than the paired Stage 1 runs.
The three inspected Extreme cases had zero clean arrivals: two crashes and one
timeout. A later optional-passing ablation was rejected and is not selected here.
These historical development results come from TRAIN cases.

Stage 1 still handles maneuvers/unknown routes and lane-margin handoff, and caps
longitudinal commands. Pace/cruise/flow assistance is controller logic. This agent
is neither a neural network nor an independently qualified replacement for Stage 1.

The release contains its original training report, successful Medium/Hard report,
failed Extreme report, and a 32-frame consecutive recorded decision fixture.
The integration tests load project-local files and compare the packaged
agent's actions with the recorded fixture. The replay checks implementation consistency.

## Source packaging and records

`src/agents/_pdd_frozen/` holds byte-identical inference dependencies, including
the original audit runtime source and the two legacy flow-lineage receipts it
actually verifies on load. They are preserved as compatibility code; their
training/audit command-line paths are not this release's interface.
The private loader resolves their imports without adding paths or replacing
legacy module names in `sys.modules`. The frozen source/model bindings remain
active. Do not edit those files to tune a release; create a new version instead.

`artifacts/agent4_pdd/1.0.0/manifest.json` records project-relative paths and hashes.
The wrapper pins the manifest checksum and verifies files before unpickling weights.
Changes to the current Stage 1/input/tracker dependencies intentionally require a
new compatibility review. Launcher provenance also snapshots all agent helpers.

The local archive preserves the pre-integration source snapshot and hashes.
The release uses the existing Stage 1 policy and platform input extraction.

## Changelog

- 24 November 2026 — Archived the earlier Agent 4 workspaces. Verified all 21
  release file hashes and passed both integration tests, including the 32-frame
  replay. SANDBOX run `pdd_cleanup_smoke_01` completed Medium / SCXCS with
  geometry seed 14, environment seed 1 and agent seed 42. The agent reached
  the destination in 623 steps (62.3 simulated seconds), with a maximum speed
  of 38.80 km/h. The run is a TRAIN smoke test.
- 1.0.0, 23 November 2026 — First project-local package and canonical registration.
  The release uses the existing learned weights and driving controller. Offline
  qualification, Extreme driving and independent evaluation remain open.
