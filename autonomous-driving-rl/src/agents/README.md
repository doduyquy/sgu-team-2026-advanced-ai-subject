# Research Agents

This package is the intended home for real Stage 0+ agent implementations.

## Current state

`Stage0RandomNaiveAgent` (`stage0_random_naive`, version `0.1.1`) is implemented and registered for **SANDBOX/AUDIT**. Benchmark eligibility remains disabled. Existing `fixture_*` policies exercise infrastructure and are not scientific baselines.

`AStarAgent` (`stage2_astar`) and `LearnedAStarAgent` (`stage6_astar_learned`), version
`0.1.0`, are SANDBOX/AUDIT candidates with the same discrete9 adapter and local route
graph. Stage 6 requires a trained MLP heuristic checkpoint; see the
[training and comparison workflow](../../docs/ASTAR_LEARNED_HEURISTIC.md).

## Stage 0 policy

The agent uses `STATE_DECISION_V1` and the certified `discrete9_lowbranch_v1` adapter.
It ignores observation fields and does not learn. On reset, its local NumPy `RandomState`
uses `agent_seed` (fallback 42), the decision counter returns to zero, and action 7
(straight + throttle) is held for the first 20 decisions.

Thereafter it independently samples steering (left/straight/right: 0.2/0.6/0.2) and
throttle (brake/coast/throttle: 0.1/0.15/0.75), then holds that action for an integer
20--40 decisions inclusive. At 10 Hz these holds last 2--4 simulated seconds.
This is a biased Random/Naive baseline; it is not uniform random control at every step.

Version 0.1.1 fixes the per-episode diagnostic counter reset and package export without
changing the action sequence for a given seed.

## Local verification and TRAIN episode

From the repository root, use the project-local interpreter:

```powershell
& '.\.venv\metadrive\python.exe' -B autonomous-driving-rl\scripts\check_stage0_readiness.py --run-smoke
```

This verifies the package and MetaDrive pin, preloads Qt for the full regression suite,
then executes one non-canonical TRAIN case (`SCS`, geometry seed 0, environment seed 0,
agent seed 101). Results Browser must verify its persisted artifacts.
Logs are saved under `artifacts/stage0_readiness/`; episodes are saved under `runs/`.
Git ownership trust is scoped to the helper process and the named local checkouts.

Use `--normalize-dataset-eol` if package checksums fail solely because LF files were
checked out as CRLF. The helper rejects mismatches that cannot be restored to the recorded
hash by newline normalization. It preserves source audit artifacts and checksum values.

After a successful regression, `--skip-tests --run-smoke --render NATIVE` starts another
exploratory episode with the standard MetaDrive window. The additional minimap is optional
and reports a warning if its module is absent. Native rendering still needs visual review.

## Before adding an agent

Read:

- [`../../docs/AGENT_DEVELOPMENT_GUIDE.md`](../../docs/AGENT_DEVELOPMENT_GUIDE.md)
- [`../../docs/STAGE_ROADMAP.md`](../../docs/STAGE_ROADMAP.md)
- [`../../docs/EXPERIMENT_WORKFLOW.md`](../../docs/EXPERIMENT_WORKFLOW.md)

New agents should implement the common `AgentPolicy` lifecycle, consume `AgentInputV1`, emit actions through a certified adapter, and be registered in `src/launcher/registry.py`.

Do not implement private environment wrappers or read evaluator internals from an agent merely to simplify one method.
