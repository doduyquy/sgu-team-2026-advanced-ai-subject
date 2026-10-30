# XeTuHanh Platform V1 — Field Catalog and Machine Mapping V1

- **Document ID:** `CATALOG-FIELDS-V1`
- **Version:** `1.0.0-draft`
- **Date:** 2026-10-30
- **Status:** `DRAFT / TEAM REVIEW REQUIRED` — Traceable Machine Mapping
- **Subproject Root:** `autonomous-driving-rl/`
- **Baseline Git Commit:** `dcd653306a0e072a2c6bee66eaeadf388101d679`

---

## 1. Catalog Overview and Conventions

This document catalogs every scientific, diagnostic, provenance, timing, and lifecycle field produced or consumed across Platform V1. Every field is mapped to its authoritative code implementation and configuration contract.

### Field Attributes
- **Name:** Exact identifier in JSON, CSV header, or Python dataclass.
- **Source of Truth:** Code module and file defining or generating the field.
- **Type & Unit:** Python data type and physical/mathematical unit.
- **Granularity:** `Episode` (one value per episode in `episodes.csv`), `Aggregate` (run-level or tier-level in `summary.json`), or `Run-Level` (in `run_manifest.json` / `run_state.json`).
- **Null Policy:** How missing, non-applicable, or undefined states are represented.
- **Presentation Rule:** How formatting and GUI display layers must render the value.
- **Status:** `EXISTING` (implemented and verified in Platform V1) or `PROPOSED` (Stage 5 research proposal).

---

## 2. Primary Benchmark Metrics (Locked)

These five fields constitute the primary benchmark scorecard. They are computed in `src/platform/metrics.py:compute_aggregate_metrics()`, defined in `configs/platform/evaluation_metrics_v1.json`, stored in `summary.json["overall_metrics"]` and `summary.json["tier_metrics"][<tier>]`, and displayed in Research Workbench Results tab.

| Field Name | Source of Truth | Type & Unit | Granularity | Null Policy | Presentation Rule | Status |
|---|---|---|---|---|---|---|
| `clean_success_rate` | `src/platform/metrics.py:AggregateMetrics` | `float`, unitless $[0.0, 1.0]$ | Aggregate (Overall, Tier) | Never null; $0.0$ if no clean success | Percentage format: `{val * 100:.1f}%` (e.g., `85.0%`) | `EXISTING` |
| `safety_failure_rate` | `src/platform/metrics.py:AggregateMetrics` | `float`, unitless $[0.0, 1.0]$ | Aggregate (Overall, Tier) | Never null; $0.0$ if no safety terminations | Percentage format: `{val * 100:.1f}%` (e.g., `12.5%`) | `EXISTING` |
| `mean_final_route_completion` | `src/platform/metrics.py:AggregateMetrics` | `float`, unitless $[0.0, 1.0]$ | Aggregate (Overall, Tier) | Never null; arithmetic mean over all episodes | Percentage format: `{val * 100:.1f}%` (e.g., `92.4%`) | `EXISTING` |
| `median_final_route_completion` | `src/platform/metrics.py:AggregateMetrics` | `float`, unitless $[0.0, 1.0]$ | Aggregate (Overall, Tier) | Never null; median over all episodes | Percentage format: `{val * 100:.1f}%` (e.g., `100.0%`) | `EXISTING` |
| `mean_time_to_clean_success_s` | `src/platform/metrics.py:AggregateMetrics` | `float` or `None`, seconds $[0.0, \infty)$ | Aggregate (Overall, Tier) | **`None` / `null` if clean success count = 0** | If float: `{val:.2f} s`. If null: `"N/A — no clean-success samples"` | `EXISTING` |

---

## 3. Macro Metrics (Locked Equal-Tier Aggregation)

These fields represent the equal-tier-weighted macro summaries across the four difficulty tiers (`Easy`, `Medium`, `Hard`, `Extreme`). They are computed in `src/platform/protocol.py:compute_macro_metrics()` and stored in `summary.json["macro_metrics"]`.

| Field Name | Source of Truth | Type & Unit | Granularity | Null Policy | Presentation Rule | Status |
|---|---|---|---|---|---|---|
| `macro_clean_success_rate` | `src/platform/protocol.py:compute_macro_metrics()` | `float`, unitless $[0.0, 1.0]$ | Aggregate (Macro) | `None` if not all 4 tiers present in run | Unweighted mean of 4 tiers: $\frac{1}{4} \sum \text{CSR}_t$. Render as `{val * 100:.1f}%` | `EXISTING` |
| `macro_safety_failure_rate` | `src/platform/protocol.py:compute_macro_metrics()` | `float`, unitless $[0.0, 1.0]$ | Aggregate (Macro) | `None` if not all 4 tiers present in run | Unweighted mean of 4 tiers: $\frac{1}{4} \sum \text{SFR}_t$. Render as `{val * 100:.1f}%` | `EXISTING` |
| `macro_mean_final_route_completion` | `src/platform/protocol.py:compute_macro_metrics()` | `float`, unitless $[0.0, 1.0]$ | Aggregate (Macro) | `None` if not all 4 tiers present in run | Unweighted mean of 4 tiers: $\frac{1}{4} \sum \text{MFRC}_t$. Render as `{val * 100:.1f}%` | `EXISTING` |
| `aggregation_weighting` | `src/platform/protocol.py:compute_macro_metrics()` | `str` | Aggregate (Macro) | Constant `"equal_tier_weighted_0.25_each"` | Explanatory label | `EXISTING` |
| `geometric_mean_success_score` | `src/platform/protocol.py:compute_macro_metrics()` | `None` | Aggregate (Macro) | Always `None` / `null` | Displayed as `"PROHIBITED"` | `EXISTING` |

*Note: Macro metrics exist ONLY for the three primary rates above. There is no macro median completion, no macro time to success, and no composite scalar score.*

---

## 4. Secondary Diagnostic Aggregate Metrics

Stored in `summary.json["overall_metrics"]` and `summary.json["tier_metrics"][<tier>]` from `src/platform/metrics.py:AggregateMetrics`.

### 4.1 Mutually Exclusive Primary Outcome Rates (Sum to 1.0)
| Field Name | Description | Type & Unit | Preferred Direction | Status |
|---|---|---|---|---|
| `success_rate` | Reached terminus cleanly with zero safety events | `float` $[0.0, 1.0]$ | Higher | `EXISTING` |
| `timeout_rate` | Truncated by horizon step budget exhaustion | `float` $[0.0, 1.0]$ | Diagnostic / Lower | `EXISTING` |
| `crash_human_rate` | Terminated by pedestrian collision | `float` $[0.0, 1.0]$ | Lower | `EXISTING` |
| `crash_vehicle_rate` | Terminated by background vehicle collision | `float` $[0.0, 1.0]$ | Lower | `EXISTING` |
| `crash_object_rate` | Terminated by traffic barrier / obstacle collision | `float` $[0.0, 1.0]$ | Lower | `EXISTING` |
| `crash_building_rate` | Terminated by structure / building collision | `float` $[0.0, 1.0]$ | Lower | `EXISTING` |
| `crash_sidewalk_rate` | Terminated by sidewalk barrier impact | `float` $[0.0, 1.0]$ | Lower | `EXISTING` |
| `out_of_road_rate` | Terminated by road boundary departure | `float` $[0.0, 1.0]$ | Lower | `EXISTING` |
| `unknown_termination_rate` | Undetermined or unhandled simulator termination | `float` $[0.0, 1.0]$ | Lower (Target 0.0) | `EXISTING` |

### 4.2 Raw Safety Event Rates (Allowing Overlap / Simultaneous Events)
| Field Name | Description | Type & Unit | Status |
|---|---|---|---|
| `raw_crash_vehicle_rate` | Fraction of episodes experiencing any vehicle collision | `float` $[0.0, 1.0]$ | `EXISTING` |
| `raw_crash_object_rate` | Fraction of episodes experiencing any object collision | `float` $[0.0, 1.0]$ | `EXISTING` |
| `raw_crash_building_rate` | Fraction of episodes experiencing any building collision | `float` $[0.0, 1.0]$ | `EXISTING` |
| `raw_crash_human_rate` | Fraction of episodes experiencing any human collision | `float` $[0.0, 1.0]$ | `EXISTING` |
| `raw_crash_sidewalk_rate` | Fraction of episodes experiencing any sidewalk collision | `float` $[0.0, 1.0]$ | `EXISTING` |
| `raw_out_of_road_rate` | Fraction of episodes experiencing any out-of-road event | `float` $[0.0, 1.0]$ | `EXISTING` |
| `raw_any_safety_event_rate` | Fraction experiencing at least one physical safety violation | `float` $[0.0, 1.0]$ | `EXISTING` |

### 4.3 Route, Motion, and Diagnostic Rates
| Field Name | Description | Type & Unit | Null Policy | Status |
|---|---|---|---|---|
| `raw_arrival_rate` | Crossed terminus line regardless of collision state | `float` $[0.0, 1.0]$ | Never null | `EXISTING` |
| `median_time_to_clean_success_s` | Median time among clean successes | `float` or `None`, seconds | `None` if successes = 0 | `EXISTING` |
| `mean_max_route_completion` | Mean peak completion attained (reveals backtracking) | `float` $[0.0, 1.0]$ | Never null | `EXISTING` |
| `mean_speed_kmh` | Average vehicle speed across all episodes | `float`, km/h | Never null | `EXISTING` |
| `max_speed_kmh` | Peak vehicle speed recorded across all episodes | `float`, km/h | Never null | `EXISTING` |
| `mean_episode_return` | Mean cumulative RL reward return | `float`, reward units | Never null (Diagnostic only) | `EXISTING` |
| `mean_episode_steps` | Mean decision steps taken per episode | `float`, steps | Never null | `EXISTING` |
| `total_episodes` | Total count of evaluated episodes in the aggregation | `int` | Never null | `EXISTING` |

---

## 5. Episode-Level Telemetry Columns (All 31 in `episodes.csv`)

Governed by `src/platform/experiment_logging.py:EpisodeLogRowV1` and `configs/platform/logging_contract_v1.json`. Recorded row-by-row into `runs/<run_id>/episodes.csv`.

| # | Column Name | Python Type | Physical Unit | Null Policy in CSV | Semantic Description |
|---|---|---|---|---|---|
| 1 | `episode_index` | `int` | Index ($1..N$) | Non-null positive | Sequential execution counter for the run |
| 2 | `protocol_order_index` | `int` | Index ($1..N$) | Non-null positive | Case position in deterministic shuffled protocol |
| 3 | `case_id` | `str` | Identifier | Non-null string | Unique scenario case ID (e.g., `case_test_001_easy`) |
| 4 | `split` | `str` | Role | `"TRAIN"`, `"VALIDATION"`, `"TEST"` | Benchmark dataset split role |
| 5 | `tier` | `str` | Difficulty | `"Easy"`, `"Medium"`, `"Hard"`, `"Extreme"` | Difficulty tier of the road geometry |
| 6 | `sequence` | `str` | Geometry Family | `"SCS"`, `"SCS_ROUNDABOUT"`, etc. | Procedural scenario sequence structure |
| 7 | `geometry_generation_seed` | `int` | Seed ($0..19$) | Non-null integer | Seed used to sample procedural road geometry |
| 8 | `environment_seed` | `int` | Seed | Non-null integer | Seed for background traffic & simulator noise |
| 9 | `horizon_steps` | `int` | Steps | Non-null positive | Maximum step budget for the route |
| 10 | `agent_seed` | `int` or empty | Seed | Empty string if deterministic / not set | Seed for agent-side stochasticity (preserves 0 as valid) |
| 11 | `primary_terminal_reason` | `str` | Categorical | Non-null string | Mutually exclusive primary cause of termination |
| 12 | `terminal_reason` | `str` | Categorical | Non-null string | Synonymous with primary terminal reason in V1 |
| 13 | `clean_success` | `bool` | Boolean | `"True"`, `"False"` | Clean success flag (safe arrival at terminus) |
| 14 | `raw_arrival` | `bool` | Boolean | `"True"`, `"False"` | Destination line reached (even if collided) |
| 15 | `final_route_completion` | `float` | Fraction $[0.0, 1.0]$ | Non-null finite float | Fraction of route completed at final step |
| 16 | `max_route_completion` | `float` | Fraction $[0.0, 1.0]$ | Non-null finite float | Highest route completion achieved during episode |
| 17 | `episode_steps` | `int` | Steps | Non-null non-negative | Number of decision steps taken |
| 18 | `episode_time_s` | `float` | Seconds | Non-null non-negative | Total elapsed simulation time in seconds |
| 19 | `time_to_clean_success_s` | `float` or empty | Seconds | **Empty string if `clean_success=False`** | Elapsed simulation time to reach clean success |
| 20 | `mean_speed_kmh` | `float` | km/h | Non-null float | Average vehicle speed over episode steps |
| 21 | `max_speed_kmh` | `float` | km/h | Non-null float | Maximum vehicle speed attained during episode |
| 22 | `episode_return` | `float` | Reward units | Non-null float | Cumulative RL reward return (diagnostic only) |
| 23 | `crash_human` | `bool` | Boolean | `"True"`, `"False"` | Collision with pedestrian occurred |
| 24 | `crash_vehicle` | `bool` | Boolean | `"True"`, `"False"` | Collision with traffic vehicle occurred |
| 25 | `crash_object` | `bool` | Boolean | `"True"`, `"False"` | Collision with road obstacle/cone occurred |
| 26 | `crash_building` | `bool` | Boolean | `"True"`, `"False"` | Collision with building/structure occurred |
| 27 | `crash_sidewalk` | `bool` | Boolean | `"True"`, `"False"` | Sidewalk barrier collision occurred |
| 28 | `out_of_road` | `bool` | Boolean | `"True"`, `"False"` | Drove off drivable road surface |
| 29 | `timeout` | `bool` | Boolean | `"True"`, `"False"` | Episode terminated due to step budget exhaustion |
| 30 | `has_technical_failure` | `bool` | Boolean | `"True"`, `"False"` | Non-scientific technical exception occurred |
| 31 | `technical_failure_reason`| `str` | Categorical | Empty string if no technical failure | Category of failure (e.g., `AGENT_EXCEPTION`) |

---

## 6. Timing and Latency Fields

### 6.1 Per-Episode Timing (`timing.csv` / `EpisodeTimingRecord`)
Governed by `src/platform/experiment_logging.py:EpisodeTimingRecord`.
| Column Name | Type | Unit | Null Policy | Semantic Description |
|---|---|---|---|---|
| `episode_index` | `int` | Index ($1..N$) | Non-null | Matches `episodes.csv` episode index |
| `act_count` | `int` | Invocations | Non-null non-negative | Number of `agent.act()` calls during episode |
| `mean_act_ms` | `float` | Milliseconds | $0.0$ if `act_count = 0` | Arithmetic mean decision time |
| `median_act_ms` | `float` | Milliseconds | $0.0$ if `act_count = 0` | Median decision time |
| `p95_act_ms` | `float` | Milliseconds | $0.0$ if `act_count = 0` | 95th percentile decision time |
| `max_act_ms` | `float` | Milliseconds | $0.0$ if `act_count = 0` | Maximum decision time in episode |
| `total_act_ms` | `float` | Milliseconds | $0.0$ if `act_count = 0` | Sum of all decision latencies in episode |
| `latency_sync_policy` | `str` | Categorical | Default `"NONE"` | `"NONE"` or `"FRAMEWORK_SYNCHRONIZED"` |

### 6.2 Run-Level Aggregate Timing (`summary.json["timing_overall"]`)
| Field Name | Type | Unit | Aggregation Formula |
|---|---|---|---|
| `total_act_count` | `int` | Invocations | $\sum_{i=1}^N \text{act\_count}_i$ |
| `weighted_mean_act_ms` | `float` or `None` | Milliseconds | $\frac{\sum \text{total\_act\_ms}_i}{\sum \text{act\_count}_i}$ (Weighted mean across all decisions) |
| `max_act_ms` | `float` or `None` | Milliseconds | $\max_{i=1}^N (\text{max\_act\_ms}_i)$ (Worst-case decision latency across entire run) |

---

## 7. Provenance, Metadata, and Integrity Fields

Recorded in `run_manifest.json`, `run_state.json`, and `run_integrity.json`.

| Field Name | Location | Type | Semantic Description |
|---|---|---|---|
| `run_id` | All artifacts | `str` | Deterministic unique run folder name (e.g., `run_test_ppo_001`) |
| `run_kind` | `run_manifest.json`, `run_state.json` | `str` | Enum: `TRAINING`, `VALIDATION_EVALUATION`, `TEST_EVALUATION`, `AUDIT` |
| `status` | `run_state.json` | `str` | Lifecycle: `INITIALIZING`, `RUNNING`, `COMPLETE`, `FAILED`, `INTERRUPTED` |
| `canonical_run` | `run_state.json` | `bool` | True only if run had clean worktree, verified pin, and locked contracts |
| `git_commit_sha` | `run_manifest.json:git_provenance` | `str` | Git SHA of repository at launch |
| `git_worktree_dirty` | `run_manifest.json:git_provenance` | `bool` | True if untracked/modified files existed at launch |
| `metadrive_pinned_commit` | `run_manifest.json:environment_provenance` | `str` | Locked pin: `85e5dadc6c7436d324348f6e3d8f8e680c06b4db` |
| `metadrive_pinned_version`| `run_manifest.json:environment_provenance` | `str` | Locked version: `0.4.3` |
| `environment_verification_status`| `run_state.json` | `str` | Verification verdict: `VERIFIED`, `UNVERIFIED_OVERRIDE`, `MISMATCH` |
| `benchmark_contract_sha256` | `run_manifest.json:platform_contracts` | `str` | Locked Gate 5 hash: `9ddd889b84d8705fae618879e5035556c80d0276a3dc2a58a7963937ebb59f77` |
| `agent_contract_sha256` | `run_manifest.json:platform_contracts` | `str` | Locked Gate 6 hash: `53aa37079ff44afa75d9a3f921b0c1f98c4600d882fced51fc8d9197795058eb` |
| `logging_contract_sha256`| `run_manifest.json:platform_contracts` | `str` | Locked Gate 7 hash: `0d17915556d83f49d9519576e95920c0853700f33a91593ef85b3aeef6cbb9f2` |
| `recorded_episode_count` | `run_state.json` | `int` | Number of completed episodes committed to disk |
| `expected_episode_count` | `run_state.json` | `int` | Predeclared expected episodes ($60$ for TEST, $96$ for VALIDATION) |
| `episodes_sha256` | `run_integrity.json` | `str` | Cryptographic canonical CSV hash of `episodes.csv` |
| `summary_sha256` | `run_integrity.json` | `str` | Cryptographic canonical JSON hash of `summary.json` |

---

## 8. Proposed Stage 5 Training and Checkpoint Fields (`PROPOSED`)

The following fields are **NOT part of Platform V1 frozen evaluation artifacts**. They are proposed for the upcoming Stage 5 Model-Free RL training gate and must be finalized in a future ADR before implementation.

| Proposed Field Name | Target Location | Type & Unit | Proposed Semantic | Status |
|---|---|---|---|---|
| `training_wall_time_s` | `training_manifest.json` | `float`, seconds | Total wall-clock time spent training the model | `PROPOSED` |
| `total_env_steps` | `training_manifest.json` | `int`, steps | Cumulative environment simulation interaction steps | `PROPOSED` |
| `total_gradient_updates` | `training_manifest.json` | `int`, steps | Number of policy/value optimizer update steps | `PROPOSED` |
| `training_seed_root` | `training_manifest.json` | `int` | Seed initializing training stochasticity (`101`, `202`, `303`) | `PROPOSED` |
| `checkpoint_sha256` | `training_manifest.json` | `str` | SHA-256 digest of serialized policy network weights | `PROPOSED` |
| `hyperparameter_config_sha256` | `training_manifest.json` | `str` | SHA-256 digest of learning rates, buffer size, clip range | `PROPOSED` |
| `eval_checkpoint_selection_step` | `training_manifest.json` | `int`, steps | Environment step where best VALIDATION checkpoint was saved | `PROPOSED` |
| `hardware_gpu_descriptor` | `training_manifest.json` | `str` | GPU device name and driver version (e.g., `RTX 4090 / 535.12`) | `PROPOSED` |

*Rule: Never assert or assume that these `PROPOSED` fields are already recorded by the Platform V1 runner.*
