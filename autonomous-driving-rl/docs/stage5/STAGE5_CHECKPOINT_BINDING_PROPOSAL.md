# Stage 5 checkpoint binding — NO-GO for official certification

## Current verified behavior

`make_sac_agent_factory()` pins an expected `model_sha256` and `Stage5SacAgent` verifies the checkpoint bundle before inference. The executor records that provenance under `run_manifest.json → config.algorithm_hyperparameters.sac_checkpoint`. This protects the loaded candidate and the run record.

The frozen `ResolvedExperimentPlanV1` identity does not include that SAC model digest. Two checkpoints can therefore share the same resolved plan identity. Stage 5 remains registered as `CANDIDATE`, `benchmark_eligible=false`; official VALIDATION/TEST evaluation is not permitted.

## Why the current checkpoint extension cannot be reused as-is

`CheckpointBindingV1` currently carries `planner_fingerprint`, a Stage 6 A* configuration path and a restricted training-seed contract. `validate_checkpoint_binding()` loads A* config/planner semantics and checks Stage 6 metadata. Using it for SAC would either misstate model semantics or require changes to frozen fields and validators. `CheckpointResolvedExperimentPlanV1` is therefore not a compatible Stage 5 binding merely because it is a plan subclass.

## Compatible proposal for governance review

Design a separate versioned `Stage5CheckpointBindingV1` extension with at least:

- model SHA-256 and checkpoint format/version;
- encoder ID, version and fingerprint;
- algorithm and exact training configuration fingerprint;
- RewardSpecV1 identity and internal TRAIN split identity;
- training root seed, actual environment steps and checkpoint step;
- code revision and required runtime/dependency provenance.

The semantic identity must exclude local filesystem paths and include the content/protocol fields above. Resolver serialization, plan fingerprinting, preflight verification and run-manifest persistence must all use that same versioned extension. Existing Stage 6 binding semantics and frozen hashes must remain unchanged. The extension needs contract tests showing that changing only checkpoint SHA changes the resolved identity and that path relocation does not.

## Decision

**NO-GO:** this sprint does not add the Stage 5 binding to `LaunchRequestV1`, `ResolvedExperimentPlanV1`, Preflight or frozen contract hashes. The change needs a separately reviewed versioned contract and governance approval. Until then, do not promote SAC to benchmark eligibility or use it for canonical holdout scoring. The Workbench comparison view displays checkpoint SHA and its available verification state, but it cannot certify identity through the resolved plan.
