# Legacy and scientific boundaries

Platform V1, the packaged MapSuite V1 benchmark and the hardened Workbench are the current foundation. A canonical Stage 0 agent is still the next research milestone. Use the [build record](PLATFORM_V1_BUILD_RECORD.md) to understand history, the [contract index](PLATFORM_V1_CONTRACT_INDEX.md) to identify current locks, and the [evidence index](PLATFORM_V1_EVIDENCE_INDEX.md) to find their verification.

## Legacy prototype

[`src/environments/course_env_v1.py`](../src/environments/course_env_v1.py) retains the old 35D observation and Discrete(5) action path. [`src/evaluation/evaluate_random.py`](../src/evaluation/evaluate_random.py) is its exploratory evaluator. Both are retained for compatibility/reference only. Their outputs are not current benchmark results and must not be mixed with canonical agent comparisons.

Launcher Core must not silently fall back to that path. Current execution uses [`src/launcher/executor.py`](../src/launcher/executor.py), AgentInputV1 and certified adapters. The current core input is 259D plus structured traffic/task context; the canonical actuator is `[steering, throttle_brake]` in `[-1, 1]^2`. Agent development must follow the [agent guide](AGENT_DEVELOPMENT_GUIDE.md).

## Audit fixtures

The `fixture_*` registrations in [`src/launcher/registry.py`](../src/launcher/registry.py) and fixture policies in [`src/platform/agent.py`](../src/platform/agent.py) exercise seed determinism, state reset, action mapping, invalid output and process/logging behavior. Their implementation or smoke results do not establish a scientific baseline.

Fixtures are not Stage 0. They are intentionally blocked from canonical VALIDATION/TEST by [`src/launcher/preflight.py`](../src/launcher/preflight.py). Keep their SANDBOX/infrastructure role explicit. Stage 0 requires a separately implemented and reviewed scientific agent with canonical registration, reproducible agent seeds and the same frozen contracts.

## Live versus authoritative results

| Surface | Authority and interpretation |
|---|---|
| Workbench live telemetry | Provisional, possibly incomplete or dropped; useful for monitoring. |
| Persisted local run artifacts with verified identity/integrity | Scientific source of truth; assess completion, provenance and eligibility before comparison. |
| W&B | Downstream mirror/visualization; cannot repair or override local run identity or integrity. |
| Incomplete, tampered, unverified or fixture runs | Inspectable diagnostics; never silently treat as canonical benchmark results. |

The Results Browser enforces these trust rules in [`results_repository.py`](../src/workbench/results_repository.py). See [workflow](EXPERIMENT_WORKFLOW.md) and [Workbench guide](WORKBENCH_GUIDE.md). A complete artifact from a fixture remains a fixture result.

## Candidate generation versus frozen dataset

Gate 2's [`mapsuite_v1_candidates.json`](../configs/maps/mapsuite_v1_candidates.json), candidate metrics and 12 representative previews document calibration. Gate 5 then locks the selected 240 geometries, splits and evaluation manifests. The [`datasets/mapsuite_v1/`](../datasets/mapsuite_v1/) package distributes that frozen benchmark: 12 families × 20 geometry seeds, 180 TRAIN / 48 VALIDATION / 12 TEST geometries, 96 validation cases and 60 test cases.

Do not reopen candidate selection, adjust difficulty or invent new geometry IDs during export. The 12 committed family previews and the 240 derived Level 1 previews have distinct purposes. Neither PNG set determines geometry identity; block-sequence hashes and the frozen package do. See [distribution policy](MAP_ARTIFACT_DISTRIBUTION.md).

## Future work

Algorithm code, learning schedules and development experiments may change within existing information/action rights and TRAIN/VALIDATION rules. TEST is a final holdout, never a tuning or checkpoint-selection source. This branch authorizes geometry-only visualization of all 240 maps, including TEST geometry; it supplies no TEST agent rollouts, rewards or performance measurements. Exported images must not become an extra agent input or a selection signal.

The simulator pin, geometry universe/families/seeds, splits/canonical manifests, benchmark hashes, lifecycle/safety precedence, reward/metrics, AgentInputV1 rights, adapters, preflight, provenance and Workbench trust rules remain frozen. Capabilities that need different rights or semantics require a reviewed, versioned extension. No fixture promotion or implicit contract exception is allowed.
