# Platform V1 build record

Verified against Git merge objects on 2026-10-06. Baseline `origin/main`: `00a34859fe6832695f90310b4af2026e13b83b4c` (PR #26). Every merge below is an ancestor of that baseline. The PR order records the driving subproject milestones; unrelated course commits are outside this record.

Platform V1 is complete, MapSuite V1 is packaged, and Workbench is hardened. The canonical Stage 0 agent remains unimplemented. Read [boundaries](LEGACY_AND_BOUNDARIES.md), [contracts](PLATFORM_V1_CONTRACT_INDEX.md) and [evidence](PLATFORM_V1_EVIDENCE_INDEX.md) alongside this record.

Early Gate 1-4 reports deliberately retain their historical NON-FROZEN recommendation status. Their selected artifacts are frozen by the Gate 5 benchmark contract; that later lock is the current authority. A merge proves integration, not that every historical empirical claim was rerun today.

## Bootstrap prototype -- [PR #2](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/pull/2)

- Merge: [`2b93cc7799084ee98b773a84310d133cc18909c9`](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/commit/2b93cc7799084ee98b773a84310d133cc18909c9).
- Purpose: Separate coursework from reproducible driving research.
- Key decision: Retain the exploratory wrapper and evaluator as compatibility references; this was not a canonical benchmark.
- Implementation: [`src/environments/course_env_v1.py`](../src/environments/course_env_v1.py).
- Human report: [`docs/environment/METADRIVE_SETUP.md`](../docs/environment/METADRIVE_SETUP.md).
- Machine evidence: no separate machine audit introduced; bootstrap/handoff documentation describes the implementation and points to earlier evidence.
- Authority and dependency: No scientific benchmark freeze; depends on the course repository.

## Gate 1 -- [PR #4](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/pull/4)

- Merge: [`3fdb883950e8afaf2c6b99bf94949a444b6e9444`](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/commit/3fdb883950e8afaf2c6b99bf94949a444b6e9444).
- Purpose: Audit observable information, action ordering and control timing.
- Key decision: Recommend the 259D observation and continuous two-actuator control; resolve source/docstring discrepancies.
- Implementation: [`scripts/run_observation_action_audit.py`](../scripts/run_observation_action_audit.py).
- Human report: [`docs/environment/PLATFORM_V1_OBSERVATION_ACTION_AUDIT.md`](../docs/environment/PLATFORM_V1_OBSERVATION_ACTION_AUDIT.md).
- Audit: [`scripts/run_observation_action_audit.py`](../scripts/run_observation_action_audit.py); committed machine evidence: [`results/audits/observation_action/`](../results/audits/observation_action/), especially [`results/audits/observation_action/observation_schema.json`](../results/audits/observation_action/observation_schema.json).
- Authority and dependency: Historical recommendation; schemas subsequently bound by Gate 5. Depends on bootstrap and the simulator pin.

## Gate 2 -- [PR #6](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/pull/6)

**Known historical metadata defect:** Gate 2 recorded `declared_base_lane_num = 2` without applying that override; pinned MetaDrive used `effective_base_lane_num = 3`. See the [dedicated lane erratum](environment/MAPSUITE_V1_LANE_METADATA_ERRATUM.md). Frozen geometry/hash identity remains authoritative.

- Merge: [`19fffa63b59e232dba1a703f69458ea9b87db005`](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/commit/19fffa63b59e232dba1a703f69458ea9b87db005).
- Purpose: Calibrate scenarios of increasing geometry and traffic complexity.
- Key decision: Choose 12 families, 20 geometry seeds each; retain exact blocks for representative canonical geometries.
- Implementation: [`configs/maps/mapsuite_v1_candidates.json`](../configs/maps/mapsuite_v1_candidates.json).
- Human report: [`docs/environment/PLATFORM_V1_MAPSUITE_AUDIT.md`](../docs/environment/PLATFORM_V1_MAPSUITE_AUDIT.md).
- Audit: [`scripts/audit_mapsuite_candidates.py`](../scripts/audit_mapsuite_candidates.py); committed machine evidence: [`results/audits/mapsuite/`](../results/audits/mapsuite/), especially [`results/audits/mapsuite/candidate_metrics.csv`](../results/audits/mapsuite/candidate_metrics.csv).
- Authority and dependency: Candidate calibration is historical; the selected universe becomes frozen in Gate 5. Depends on Gate 1.

## Gate 3 -- [PR #8](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/pull/8)

- Merge: [`a066e609746f5589ff966b982540dc4f2b77f6c1`](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/commit/a066e609746f5589ff966b982540dc4f2b77f6c1).
- Purpose: Define reproducible episode lifecycle and route-aware budgets.
- Key decision: Safety failures take precedence over arrival; distinguish termination from timeout truncation.
- Implementation: [`src/platform/episode.py`](../src/platform/episode.py).
- Human report: [`docs/environment/PLATFORM_V1_EPISODE_AUDIT.md`](../docs/environment/PLATFORM_V1_EPISODE_AUDIT.md).
- Audit: [`scripts/audit_episode_contract.py`](../scripts/audit_episode_contract.py); committed machine evidence: [`results/audits/episode/`](../results/audits/episode/), especially [`results/audits/episode/termination_cases.csv`](../results/audits/episode/termination_cases.csv).
- Authority and dependency: EpisodeSpecV1 recommendations subsequently bound by Gate 5. Depends on Gate 2 route calibration.

## Gate 4 -- [PR #10](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/pull/10)

- Merge: [`2c0aa767d5599d67de18f7de69f434dbff71bf34`](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/commit/2c0aa767d5599d67de18f7de69f434dbff71bf34).
- Purpose: Separate a training reward from scientific evaluation.
- Key decision: Use route progress, time cost and terminal outcomes; compare agents by success, safety, completion and success time.
- Implementation: [`src/platform/reward.py`](../src/platform/reward.py).
- Human report: [`docs/environment/PLATFORM_V1_REWARD_METRICS_AUDIT.md`](../docs/environment/PLATFORM_V1_REWARD_METRICS_AUDIT.md).
- Audit: [`scripts/audit_reward_metrics.py`](../scripts/audit_reward_metrics.py); committed machine evidence: [`results/audits/reward_metrics/`](../results/audits/reward_metrics/), especially [`results/audits/reward_metrics/metrics_aggregate_validation.json`](../results/audits/reward_metrics/metrics_aggregate_validation.json).
- Authority and dependency: RewardSpecV1 and EvaluationMetricsV1 subsequently bound by Gate 5. Depends on Gate 3 outcome semantics.

## Gate 5 -- [PR #12](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/pull/12)

- Merge: [`001645edd4cc66ef538762d3fcc7d878eefcb608`](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/commit/001645edd4cc66ef538762d3fcc7d878eefcb608).
- Purpose: Freeze scientifically comparable evaluation.
- Key decision: Separate geometry/environment/agent seeds; lock 180/48/12 geometry splits and 96/60 evaluation cases.
- Implementation: [`src/platform/protocol.py`](../src/platform/protocol.py).
- Human report: [`docs/environment/PLATFORM_V1_EVALUATION_PROTOCOL_AUDIT.md`](../docs/environment/PLATFORM_V1_EVALUATION_PROTOCOL_AUDIT.md).
- Audit: [`scripts/audit_evaluation_protocol.py`](../scripts/audit_evaluation_protocol.py); committed machine evidence: [`results/audits/evaluation_protocol/`](../results/audits/evaluation_protocol/), especially [`results/audits/evaluation_protocol/protocol_hashes.json`](../results/audits/evaluation_protocol/protocol_hashes.json).
- Authority and dependency: Authoritative benchmark lock, schemas, manifests, seeds, reward and episode policies. Depends on Gates 1-4.

## Gate 6 -- [PR #14](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/pull/14)

- Merge: [`72cb534c73cb289a007a0cd4ce13e8c9a4259e4a`](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/commit/72cb534c73cb289a007a0cd4ce13e8c9a4259e4a).
- Purpose: Give all algorithm stages a common information and action boundary.
- Key decision: AgentInputV1 excludes evaluator-only information; certified adapters map agent output to the frozen actuator.
- Implementation: [`src/platform/agent.py`](../src/platform/agent.py).
- Human report: [`docs/environment/PLATFORM_V1_AGENT_CONTRACT_AUDIT.md`](../docs/environment/PLATFORM_V1_AGENT_CONTRACT_AUDIT.md).
- Audit: [`scripts/audit_agent_contract.py`](../scripts/audit_agent_contract.py); committed machine evidence: [`results/audits/agent_contract/`](../results/audits/agent_contract/), especially [`results/audits/agent_contract/contract_hashes.json`](../results/audits/agent_contract/contract_hashes.json).
- Authority and dependency: Authoritative agent/runtime contracts; fixture agents remain infrastructure only. Depends on Gate 5.

## Gate 7 -- [PR #16](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/pull/16)

- Merge: [`734e260480ef7a714ef9cbe66620a85518dfdd7b`](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/commit/734e260480ef7a714ef9cbe66620a85518dfdd7b).
- Purpose: Make local experiment records reproducible and failure-aware.
- Key decision: Local artifacts are the source of truth; W&B is an optional downstream mirror.
- Implementation: [`src/platform/experiment_logging.py`](../src/platform/experiment_logging.py).
- Human report: [`docs/environment/PLATFORM_V1_LOGGING_WANDB_AUDIT.md`](../docs/environment/PLATFORM_V1_LOGGING_WANDB_AUDIT.md).
- Audit: [`scripts/audit_logging_contract.py`](../scripts/audit_logging_contract.py); committed machine evidence: [`results/audits/logging_contract/`](../results/audits/logging_contract/), especially [`results/audits/logging_contract/contract_hashes.json`](../results/audits/logging_contract/contract_hashes.json).
- Authority and dependency: Authoritative logging/observability contracts. Depends on Gate 6 runtime identity.

## Gate 7.5A -- [PR #18](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/pull/18)

- Merge: [`0376da8b8bd3e1c3b60d43ed371b6713c0cb31f6`](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/commit/0376da8b8bd3e1c3b60d43ed371b6713c0cb31f6).
- Purpose: Resolve and validate experiments before simulation.
- Key decision: Use one launcher core for CLI and GUI; fail closed on benchmark, registry or implementation mismatches.
- Implementation: [`src/launcher/`](../src/launcher/).
- Human report: [`docs/environment/PLATFORM_V1_LAUNCHER_CORE_AUDIT.md`](../docs/environment/PLATFORM_V1_LAUNCHER_CORE_AUDIT.md).
- Audit: [`scripts/audit_launcher_core.py`](../scripts/audit_launcher_core.py); committed machine evidence: [`results/audits/launcher_core/`](../results/audits/launcher_core/), especially [`results/audits/launcher_core/contract_hashes.json`](../results/audits/launcher_core/contract_hashes.json).
- Authority and dependency: Authoritative launcher/execution contracts and canonical registry hash. Depends on Gate 7 and all earlier locks.

## Gate 7.5B Pass B1 -- [PR #20](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/pull/20)

- Merge: [`99f0a66aa156004a5a1ec7993d2368ff83d7c325`](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/commit/99f0a66aa156004a5a1ec7993d2368ff83d7c325).
- Purpose: Provide a teammate-facing experiment Workbench.
- Key decision: Keep workers isolated and delegate scientific decisions to Launcher Core.
- Implementation: [`src/workbench/core_adapter.py`](../src/workbench/core_adapter.py).
- Human report: [`docs/environment/PLATFORM_V1_WORKBENCH_AUDIT.md`](../docs/environment/PLATFORM_V1_WORKBENCH_AUDIT.md).
- Audit: [`scripts/audit_workbench.py`](../scripts/audit_workbench.py); committed machine evidence: [`results/audits/workbench/`](../results/audits/workbench/), especially [`results/audits/workbench/contract_hashes.json`](../results/audits/workbench/contract_hashes.json).
- Authority and dependency: Authoritative Workbench presentation/isolation contract; UI confers no new information rights. Depends on 7.5A.

## Gate 7.5B Pass B2 -- [PR #22](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/pull/22)

- Merge: [`9c5d1fabe07e0c4cc24c28d0b5b2857f5496652d`](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/commit/9c5d1fabe07e0c4cc24c28d0b5b2857f5496652d).
- Purpose: Browse results while protecting artifact identity and integrity.
- Key decision: Live telemetry is provisional; only verified persisted artifacts can be treated as authoritative.
- Implementation: [`src/workbench/results_repository.py`](../src/workbench/results_repository.py).
- Human report: [`docs/environment/PLATFORM_V1_WORKBENCH_RESULTS_AUDIT.md`](../docs/environment/PLATFORM_V1_WORKBENCH_RESULTS_AUDIT.md).
- Audit: [`scripts/audit_workbench_results.py`](../scripts/audit_workbench_results.py); committed machine evidence: [`results/audits/workbench_results/`](../results/audits/workbench_results/), especially [`results/audits/workbench_results/contract_hashes.json`](../results/audits/workbench_results/contract_hashes.json).
- Authority and dependency: Authoritative results/trust contract. Depends on B1 and local logging.

## Gate 7.5B Pass B3 -- [PR #24](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/pull/24)

- Merge: [`f68ea806cadfca85236bfe9c09546008dfe75811`](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/commit/f68ea806cadfca85236bfe9c09546008dfe75811).
- Purpose: Harden worker lifecycle, bounded resources and recovery.
- Key decision: Enforce operation state and filesystem/protocol resilience without changing scientific semantics.
- Implementation: [`src/workbench/process_runner.py`](../src/workbench/process_runner.py).
- Human report: [`docs/environment/PLATFORM_V1_WORKBENCH_HARDENING_AUDIT.md`](../docs/environment/PLATFORM_V1_WORKBENCH_HARDENING_AUDIT.md).
- Audit: [`scripts/audit_workbench_hardening.py`](../scripts/audit_workbench_hardening.py); committed machine evidence: [`results/audits/workbench_hardening/`](../results/audits/workbench_hardening/), especially [`results/audits/workbench_hardening/scientific_boundary_regression.json`](../results/audits/workbench_hardening/scientific_boundary_regression.json).
- Authority and dependency: Authoritative final Workbench hardening contract. Depends on B2; closes the Platform V1 foundation.

## Team handoff/onboarding -- [PR #25](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/pull/25)

- Merge: [`d455c1d3a363e0a226f2f2c352b53df67bfc37b0`](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/commit/d455c1d3a363e0a226f2f2c352b53df67bfc37b0).
- Purpose: Make the completed foundation usable by new teammates.
- Key decision: Explain canonical vs fixture agents, setup, workflows and future stages.
- Implementation: [`docs/GETTING_STARTED.md`](../docs/GETTING_STARTED.md).
- Human report: [`docs/PLATFORM_V1_OVERVIEW.md`](../docs/PLATFORM_V1_OVERVIEW.md).
- Machine evidence: no separate machine audit introduced; bootstrap/handoff documentation describes the implementation and points to earlier evidence.
- Authority and dependency: Navigation and interpretation only; no new scientific contract. Depends on B3.

## MapSuite V1 dataset packaging -- [PR #26](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/pull/26)

- Merge: [`00a34859fe6832695f90310b4af2026e13b83b4c`](https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/commit/00a34859fe6832695f90310b4af2026e13b83b4c).
- Purpose: Distribute the frozen benchmark in a self-describing package.
- Key decision: Derive CSVs/config/provenance from frozen artifacts; preserve the 12 representative family previews.
- Implementation: [`scripts/export_mapsuite_v1_dataset.py`](../scripts/export_mapsuite_v1_dataset.py).
- Human report: [`datasets/mapsuite_v1/README.md`](../datasets/mapsuite_v1/README.md).
- Machine evidence: the package [`datasets/mapsuite_v1/dataset_manifest.json`](../datasets/mapsuite_v1/dataset_manifest.json) and [`datasets/mapsuite_v1/CHECKSUMS.sha256`](../datasets/mapsuite_v1/CHECKSUMS.sha256); verify with [`scripts/verify_mapsuite_v1_dataset.py`](../scripts/verify_mapsuite_v1_dataset.py).
- Authority and dependency: Authoritative GitHub package of the Gate 5 benchmark, not a redesigned dataset. Depends on Gate 5 and the completed platform.

## Next milestone and change control

Stage 0 Random / Naive is the next algorithm milestone, outside this documentation/export branch. Algorithm implementations may evolve within AgentInputV1 and certified adapters. Changing the simulator pin, geometry universe, families, seeds, splits, manifests, reward, metrics, lifecycle, holdout policy or trust/provenance rules requires an explicitly reviewed and versioned contract change. See [course scope](COURSE_RESEARCH_SCOPE.md).
