# Research Workbench Guide

Research Workbench V1 is a desktop control/inspection layer over the same launcher, preflight, executor, logging, and result contracts used by the CLI.

Launch it from `autonomous-driving-rl/`:

```powershell
python -m src.workbench.app
```

## 1. What the Workbench is — and is not

The Workbench is an experiment control room. It does not contain the intelligence of an agent and does not replace the scientific backend.

```text
GUI request
  → resolved plan
  → preflight
  → worker process
  → ExperimentExecutor + MetaDrive
  → local artifacts
  → Results verification
```

The GUI can disappear and the same experiments remain definable through the launcher core/CLI.

## 2. Experiment Setup

Configure:

- mode: SANDBOX / AUDIT / VALIDATION / TEST;
- registered agent;
- tier/sequence/geometry/environment seed where the mode permits it;
- agent seed;
- render mode;
- W&B mode;
- optional runs root and run ID.

For learning/debugging, start with `SANDBOX`, a fixture agent, an Easy case, and `NATIVE` rendering.

VALIDATION and TEST intentionally lock case selection to the fixed full suites.

## 3. Plan & Preflight

`Resolve & Preflight` answers two separate questions:

1. **Resolve:** exactly what would be executed?
2. **Preflight:** is that experiment allowed under Platform V1?

The page shows the resolved agent/case/execution information and a table of PASS/WARNING/FAIL checks.

If the verdict is `BLOCKED`, do not bypass it. Fix the failed requirement.

A common expected example today is `agent_benchmark_eligibility`: fixture agents are allowed for SANDBOX/AUDIT but intentionally blocked from VALIDATION/TEST.

Changing the experiment request invalidates the previous preflight; resolve again before running.

## 4. Run Monitor

The Workbench runs simulation in a separate worker process so a simulator/worker failure does not silently become a valid GUI result.

The monitor shows:

- run/episode progress;
- sampled route completion and speed telemetry;
- chronological launcher events;
- an execution-reported summary.

Important: the live summary is **provisional**. It is not yet the authoritative persisted result.

`FORCE TERMINATE` is a non-graceful emergency control. It kills the worker but the GUI does not fabricate a scientific lifecycle state such as COMPLETE/INTERRUPTED on behalf of the logger.

## 5. Agents Explorer

Use this tab to inspect registered agent capabilities:

- method/stage identity;
- input profile;
- action adapter;
- deterministic/stochastic inference;
- sandbox/audit/benchmark eligibility;
- checkpoint requirement.

`benchmark_eligible=True` means the platform allows canonical benchmark execution. It does **not** mean the agent has good performance.

## 6. Cases Explorer

Browse TRAIN, VALIDATION, and TEST manifests by tier/sequence.

This is a read-only inspection tool. It intentionally does not provide a “run selected TEST rows” workflow because canonical VALIDATION/TEST use the fixed complete suites.

## 7. Results

Results reads persisted run artifacts; it does not recompute scientific metrics in the GUI.

Trust states include:

- `VERIFIED` — complete run with valid integrity evidence;
- `NOT_FINAL` — run is not in a final trustworthy lifecycle state;
- `UNVERIFIED` — complete-looking data without sufficient integrity proof;
- `FAILED` — integrity verification failed.

Only verified complete runs may be labeled with authoritative benchmark metrics.

Use the result views for:

- overall primary metrics;
- episode-level records;
- agent decision timing;
- provenance;
- artifact integrity;
- W&B/technical diagnostics.

## 8. Suggested first hands-on exercise

1. Select `SANDBOX`.
2. Choose `fixture_seeded_random`.
3. Select an Easy case or leave the case selector on its default TRAIN choice.
4. Set Agent Seed to `101`.
5. Set Render to `NATIVE`.
6. Resolve & Preflight.
7. Read the resolved plan and checks.
8. Run one episode.
9. Observe the MetaDrive window and Run Monitor.
10. Open Results and inspect integrity/provenance.

After doing this once, a teammate should understand the operational flow before implementing Stage 0.
