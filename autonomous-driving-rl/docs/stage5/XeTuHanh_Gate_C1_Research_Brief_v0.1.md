# XeTuHanh — Gate C1: Scientific Comparison Specification V1
**Type:** Research specification + OpenCode Gemini handoff  
**Prepared:** 2026-10-30  
**Status:** `DRAFT / TEAM REVIEW REQUIRED` — not an approved scientific lock  
**Repository:** `https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject`  
**Observed main HEAD:** `dcd653306a0e072a2c6bee66eaeadf388101d679`  
**Subproject root:** `autonomous-driving-rl/`

## 0. Objective and non-objective

**Objective:** Specify a truthful, traceable, common interpretation and comparison protocol for scientific benchmark results from Stage 0–7, before substantial experiments. Deliver specification/field registry/comparability matrix/stats policy/example report, reviewed by the team. Prepare downstream consumption by GUI V2 and Stage 5 without implementing either.

**Non-objectives:** No RL training, checkpoints, agent changes, MetaDrive changes, GUI changes, new scientific metric implementations, redefinition of existing metrics, benchmark regeneration, split changes, or direct edits to frozen contracts. Do not change benchmark hashes to make tests pass. Do not upgrade dependencies.

The gate is successful when the team can answer: *What is measured? How is it computed? Which runs may be compared? What statistical claim is justified? What remains uncertain?*

## 1. Authoritative current sources

Inspect these *before* drafting. Cite precise file paths and definitions in docs:
- `configs/platform/evaluation_metrics_v1.json`
- `configs/platform/evaluation_protocol_v1.json`
- `configs/platform/logging_contract_v1.json`
- `configs/platform/seed_plan_v1.json`
- `configs/platform/workbench_results_contract_v1.json`
- `src/platform/metrics.py`, `src/platform/protocol.py`, `src/platform/experiment_logging.py`
- `src/launcher/models.py`, `src/launcher/resolver.py`, `src/launcher/preflight.py`
- `src/workbench/results_repository.py`, `src/workbench/widgets/results_widget.py`
- `docs/PLATFORM_V1_CONTRACT_INDEX.md`, `docs/EXPERIMENT_WORKFLOW.md`, `docs/COURSE_RESEARCH_SCOPE.md`

**Source-of-truth rule:** `summary.json` provides per-run aggregate metrics; `episodes.csv` is the per-episode evidence; `timing.csv` is the source of agent decision timing; `run_manifest.json`/`run_state.json`/`run_integrity.json` provide provenance, lifecycle and fingerprint evidence. The GUI must not silently reimplement the frozen metric formulas. Cross-run analytical aggregates, if ever implemented, must be distinctly labeled derived analysis and preserve raw source links.

**Provenance note:** `EvaluationMetricsV1` was historically labeled NON-FROZEN at Gate 4, but selected definitions were bound into the frozen Gate 5 benchmark. Do not mistake historical wording for permission to alter the locked benchmark.

## 2. Locked primary benchmark fields

Name | Meaning | Direction | Scope / edge case
---|---|---|---
`clean_success_rate` | fraction of clean successes | higher | uses safety-first success semantics
`safety_failure_rate` | fraction whose **primary terminal reason** is a safety failure | lower | distinct from raw event flags
`mean_final_route_completion` | arithmetic mean of final completion [0,1] | higher | never infer from best progress
`median_final_route_completion` | median of final completion [0,1] | higher | distinguish from mean
`mean_time_to_clean_success_s` | average simulation time among **clean successes** | lower conditionally | `null`/N/A when zero successes; always report success denominator

The `Safety Failure Rate` and `raw_any_safety_event_rate` are not synonymous. The first follows mutually exclusive primary outcome classification. Raw safety flags can overlap.

Macro fields currently implemented in `src/platform/protocol.py` are only:
`macro_clean_success_rate`, `macro_safety_failure_rate`, `macro_mean_final_route_completion`, with **equal weighting of 0.25 per tier**. Do not create a "macro time to success" or composite score by assumption. Overall metrics and macro metrics are distinct even if their values occasionally coincide because suites are balanced.

Primary metrics must be reported for both **overall** and each tier (Easy/Medium/Hard/Extreme). Also retain episode count, clean-success count, and valid time-to-success count (as explanatory presentation counts, carefully label as derived if not persisted in the summary).

## 3. Diagnostic, timing, provenance, and future training fields

**Existing episode fields (31; exact schema from LoggingContractV1):**
`episode_index`, `protocol_order_index`, `case_id`, `split`, `tier`, `sequence`, `geometry_generation_seed`, `environment_seed`, `horizon_steps`, `agent_seed`, `primary_terminal_reason`, `terminal_reason`, `clean_success`, `raw_arrival`, `final_route_completion`, `max_route_completion`, `episode_steps`, `episode_time_s`, `time_to_clean_success_s`, `mean_speed_kmh`, `max_speed_kmh`, `episode_return`, `crash_human`, `crash_vehicle`, `crash_object`, `crash_building`, `crash_sidewalk`, `out_of_road`, `timeout`, `has_technical_failure`, `technical_failure_reason`.

**Existing timing fields:** `act_count`, `mean_act_ms`, `median_act_ms`, `p95_act_ms`, `max_act_ms`, `total_act_ms`, `latency_sync_policy` (plus `episode_index`). Agent timing excludes environment step, rendering, logger, adapter and input construction. Timing comparisons require disclosing hardware, synchronization assumptions, platform and runtime; do not claim hardware-independent latency superiority.

**Existing run/provenance fields to inventory:** agent id/version, declared method, input profile, action adapter, agent seed, training run seed (if present), mode/split, contract/manifest fingerprints, simulator pin, git SHA, dirty-worktree flag, canonical flag, expected and recorded episode counts, integrity state, checkpoint identity (if available), run state and W&B mirror status.

**Stage 5 *proposals only*, not part of existing Gate-7 benchmark metrics:** training wall time, environment steps, training seed root, training updates, checkpoint digest, training hyperparameter config digest, optimizer/preprocessing state, hardware descriptors, and evaluation checkpoint selection trail. Mark `PROPOSED`, provide a separate later ADR and schema, and never assert these are already recorded by the platform.

## 4. Immutable experimental design and statistical policy

Current Gate 5 rules:
- 240 geometries from 12 families; 180 TRAIN, 48 VALIDATION, 12 TEST.
- Canonical VALIDATION = 96 ordered cases, 24 per tier; canonical TEST = 60 ordered cases, 15 per tier.
- Environment test seeds 9101–9105; validation seeds 5101–5102.
- Agent replicate seeds 101, 202, 303 for stochastic inference.
- Learned methods train **3 independent models** from training root seeds 101, 202, 303.
- Genuinely deterministic inference agents without independently trained replicates use one evaluation run per environment case, not fake replicate variance.
- TEST held out for frozen final evaluation only; no tuning/selection after seeing TEST.

**Distinguish these concepts:** training run seed; stochastic inference seed; fixed environment seed; geometry generation seed; paired case identity. These are not interchangeable. A stochastic trained policy may require a clearly documented inference-seed strategy **in addition to** independent training seeds; do not invent combinations until team approval. Mark any unresolved 3×3 factorial design decision `OPEN`.

**Minimum reporting:** a table for each algorithm/model replicate (overall + by tier), then a clearly distinguished across-independent-training-run mean and sample standard deviation when R>=2. For R=3, disclose the small sample size and show individual replicate scores, rather than pretending 180 episodes of three models are 180 independent training experiments. For deterministic non-learned agents, report their canonical case coverage without fabricated replicate std.

For vector of independent replicate-level metrics `m_r`:
`mean = Σ m_r / R`
`sample_std = sqrt(Σ(m_r - mean)^2 / (R-1))` when `R>=2`.
Do **not** compute an inferential p-value or 95% CI by treating individual environment cases from the same trained checkpoint as independent training replicates. Any bootstrap proposal must state the resampling unit, dependence structure, paired-case blocking and limitations with only three independently trained models.

**Paired comparisons:** match case IDs and environment seeds across compared agents, with valid coverage and compatible protocol. Pairing by case improves comparability but does not make two models' training randomness identical. Report per-tier effects; optional delta statistics must state orientation (`B-A` for higher-is-better, `A-B` for lower-is-better) and whether they are exploratory.

**Conditional success time:** If an agent has zero clean successes, its time-to-success is N/A rather than zero or infinity. If success counts differ, mean conditional time does **not** establish overall efficiency dominance; interpret together with success rate.

Do not use a universal weighted/geometric "mega-score" or a leaderboard that hides safety trade-offs. Reward/return is a training diagnostic, not the cross-method scientific ranking metric.

## 5. Run compatibility decision rules: three layers

### Layer A — trustworthy evidence (HARD BLOCK for authoritative comparisons)
- Both run snapshots exist, `run_state=COMPLETE`, full evidence and integrity fingerprints verify.
- `canonical_run=True`, no dirty/unverified override, matching verified MetaDrive pin.
- No fixture or AUDIT/SANDBOX run promoted to canonical.
- Correct split/mode; full exact manifest case IDs, ordering/coverage, unique episode identities and expected counts.
- No technical failure masquerading as successful evaluation.
- Avoid deriving final claims from W&B only, live telemetry, missing/failed/partial/inconsistent artifacts.
- CHECK **all** required conditions; `integrity=VERIFIED` is necessary but may not alone prove benchmark scientific eligibility or exact match of both runs.

### Layer B — scientific comparability
- Same benchmark contract/version, metric definitions, case manifest, environment seeds, episode lifecycle, simulator and observation rights.
- A mismatch of split, benchmark hash, source case IDs, environment pin or primary metric semantics is **BLOCK** for direct apples-to-apples comparison.
- A different `action_adapter_id` or permitted `input_profile_id` must be surfaced prominently. In particular:
  - Same action adapter + same input profile + frozen protocol: closer to an algorithmic comparison.
  - Different action adapter but common canonical physical actuators: may be presented as **end-to-end system** comparison, NOT as an isolated test of algorithm superiority. Label `QUALIFIED`, not silent equivalent.
  - Different information rights/profile: do not present as pure same-information benchmark; report separate ablation/qualified experiment only if governance permits.
- Mismatched hardware affects **inference timing claims** rather than necessarily invalidating success/safety comparisons; flag latency comparison separately.

### Layer C — evidence strength and interpretation
- If two valid models share exact test cases, show paired per-case differences descriptively.
- With only three training replicates, reserve strong superiority/significance claims.
- Provide full sample sizes and caveats; avoid cherry-picking tiers/cases after seeing TEST.

**State machine suggestion for future comparison module (not implemented in C1):** `REJECTED_UNTRUSTED`, `INCOMPATIBLE`, `QUALIFIED_COMPARISON`, `STRICTLY_COMPARABLE`, plus explicit reasons. Never conflate technical validity with statistical confidence. These enum names are proposals, not additions to existing runtime contracts.

## 6. Example report contract

Every report must include:
1. Claim scope: exploratory TRAIN/SANDBOX, VALIDATION selection, or frozen TEST.
2. Benchmark/environment/contract SHA, source code/checkpoint provenance, policy input+action profile.
3. Cases/replicates: exact counts and seed taxonomy; independent training replicates and inference stochasticity separately.
4. Five primary metrics with correct units and directions; overall and tier views.
5. Outcome and safety breakdown, success denominators and conditional-time N/A behavior.
6. Technical failures, timing context, and diagnostic rewards separately.
7. Comparative inference: supported observation vs unsupported causal algorithm claims.
8. Limitations, resource budgets (if available) and references.
9. GUI future-use notes: verification gating; no mutation of local run data; no reinterpretation of reward as performance.

Use **clearly labeled SYNTHETIC DATA ONLY** for worked examples. Do not invent real run IDs, empirical agent performance or citations.

## 7. Gate C1 deliverables and acceptance

Recommended output under `autonomous-driving-rl/docs/comparison/`:
- `COMPARISON_SPEC_V1.md`: normative proposal with precise source references, version, status `DRAFT-TEAM-REVIEW`, decision rules, units, formulae, caveats.
- `FIELD_CATALOG_V1.md`: machine-source mapped field catalog (`name`, source, unit, type, granularity, null policy, stage eligibility, intended presentation, status existing/proposed), without inventing authoritative field semantics.
- `COMPARABILITY_MATRIX_V1.md`: tested-by-reason paper matrix covering verified vs unverified, TEST vs VALIDATION, missing episodes, different split, profile/adapter/seed case and hardware latency.
- `EXAMPLE_REPORT_V1.md`: clearly synthetic example covering zero success, safety trade-offs, tier balance, deterministic/stochastic/learned replicate distinctions.
- `OPEN_DECISIONS_V1.md`: sign-off checklist for Q/Khoa/P/Tri; identify unresolved choices and owner; do not assert sign-off was obtained.

A short link can be added to `docs/README.md` or `docs/EXPERIMENT_WORKFLOW.md` without rewriting frozen content.

**Acceptance criteria:**
- All existing primary metrics exact spellings and semantics, with citations to repository files.
- 31 episode fields / timing fields correctly inventoried.
- No schema changes, no source edits to frozen contracts, no dependency changes.
- Clear data authority, comparison compatibility policy, train/val/test boundaries.
- Worked examples include edge cases: 0 successes, unequal success rates, unmatched case IDs, fixture data, action adapter mismatch, three training seeds vs episode count, no fake variance for deterministic non-learned agents.
- `git diff --check` passes; documentation links resolve; no synthetic data misrepresented as measured.
- Manual cross-check every field against `configs/platform/*` and `src/platform/*`; any contradictions must be reported before authoring claims.
- Final status remains DRAFT until team sign-off. Git merge alone does not mean scientific approval.

## 8. Gate C1 exclusions and known independent blocker

Current main contains probable `src.agents` export and `src.visualization.minimap_overlay` import failures from Stage 0, observed in code review after 2026-10-09. These are **R0 integration issues**: note separately; do not modify/import-fix in this docs-only PR. If tests fail due to them, report exact commands and traceback truthfully instead of changing unrelated files. Handle R0 on a separate branch/PR.

## 9. Planned explanation after a gate is implemented

Before moving to the next gate, produce a Vietnamese walkthrough:
- What user-level problem was solved? Why does it matter for comparing Stage 0–7?
- What files and rules were introduced? Which ones already existed and were not modified?
- One concrete worked example of the decision rules and a misleading interpretation they prevent.
- Which acceptance tests passed, which did not run, what code provenance was checked, whether PR is merge-ready.
- What remains undecided/at risk and which subsequent gate is unblocked.
Never label C1 complete until independent GitHub PR review and team scientific sign-off.

## 10. Research references (for rationale, not replacement for code-derived contract)

- Henderson et al. (2018), *Deep Reinforcement Learning That Matters*, AAAI. https://ojs.aaai.org/index.php/AAAI/article/view/11694
- Patterson et al. (2024), *Empirical Design in Reinforcement Learning*, JMLR. https://jmlr.org/papers/v25/23-0183.html
- Agarwal et al., *RLiable: Towards Reliable Evaluation & Reporting in RL*. https://research.google/blog/rliable-towards-reliable-evaluation-reporting-in-reinforcement-learning/
- Stable Baselines3, *RL Tips and Tricks*. https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html

Repository source definitions always take precedence for *what Platform V1 actually implements*.
