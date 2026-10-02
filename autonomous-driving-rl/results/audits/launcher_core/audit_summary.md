# Gate 7.5A Research Launcher Core Audit Summary

## 1. Verified MetaDrive Source & Pinned Versions
- **Pinned MetaDrive Commit:** `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`
- **Package Version:** `0.4.3`
- **Working Tree Policy:** Strict clean worktree required for VALIDATION and TEST.

## 2. Orchestration Architecture & Modes
- **Launcher Modes:** `SANDBOX`, `VALIDATION`, `TEST`, `AUDIT`.
- **RunKind Mapping:** SANDBOX/AUDIT $\to$ `AUDIT`; VALIDATION $\to$ `VALIDATION_EVALUATION`; TEST $\to$ `TEST_EVALUATION`.
- **Holdout Isolation:** Sandbox is strictly restricted to TRAIN split geometries. Requests for VALIDATION or TEST geometries raise holdout violations loudly.
- **Benchmark Case Parity:** Exactly 96 cases for VALIDATION and 60 paired cases for TEST with 100% manifest parity.

## 3. Agent Registry & Fixture Boundaries
- **AgentRegistryV1:** Segregates declarative `AgentRegistrationV1` metadata from runtime factory callables.
- **Audit Fixtures:** 4 Gate-6 fixtures registered (`fixture_constant_continuous`, `fixture_seeded_random`, `fixture_stateful_counter`, `fixture_discrete`). All flagged `benchmark_eligible = False`.
- **Benchmark Protection:** Fixtures are strictly blocked from executing on TEST or VALIDATION suites.

## 4. Preflight Validation Battery
- Pure preflight engine evaluates 7 distinct scientific check categories across 13 decision matrix scenarios.
- Blocks dirty worktrees, unverified environments, missing/invalid stochastic seeds, and native rendering on benchmark runs.
- Execution is allowed if and only if zero blocking failures occur.

## 5. Execution Pipeline & Gate-7 Integration
- Generic `ExperimentExecutor` connects MetaDrive simulator, Gate-6 AgentPolicy, Gate-3 outcome classification, Gate-4 reward metrics, and Gate-7 logging.
- Information Parity Principle strictly preserved: agent receives only `AgentInputV1` and `AgentPublicEpisodeContext`.
- Launcher provenance embedded cleanly into `run_manifest.json` under `config.algorithm_hyperparameters['launcher']` without mutating Gate-7 schemas.

## 6. Cryptographic Contract Hashes
- **`gate5_benchmark_contract_sha256`:** `9ddd889b84d8705fae618879e5035556c80d0276a3dc2a58a7963937ebb59f77` (locked, untouched)
- **`gate6_agent_contract_sha256`:** `53aa37079ff44afa75d9a3f921b0c1f98c4600d882fced51fc8d9197795058eb` (locked, untouched)
- **`platform_runtime_contract_sha256`:** `c7698768539a769c7b2bc6b90771ff234276a974b6e0353bbb03119faf2f79ad` (locked, untouched)
- **`logging_contract_sha256`:** `0d17915556d83f49d9519576e95920c0853700f33a91593ef85b3aeef6cbb9f2` (locked, untouched)
- **`platform_observability_contract_sha256`:** `f00c27fca8abf8145596576cd0dd2540181ff8ae7eccf7806c864921933d3581` (locked, untouched)
- **`canonical_agent_registry_sha256`:** `9fac42d07949796d00b6dc869b98c945eae7a59355d7928f429e457f1ee7f26f`
- **`launcher_contract_sha256`:** `5e6269b6e32349bd3ee34ca52583c00f832bd0a737330c894e9a8d314653db04`
- **`platform_execution_contract_sha256`:** `442beafd86212419dc7b3edfa60e53715bf33877d3dd72e61c54091520b0563d`

## 7. Additional Verification Artifacts
- `canonical_registry_integrity.json`: Verified canonical registry SHA-256 sensitivity and description exclusion.
- `registry_authority_negative_checks.json`: Verified negative exploit defenses against forged benchmark eligibility and custom registry authority.
- `implementation_binding_checks.json`: Verified strict runtime implementation class and factory identity binding.
- `canonicality_matrix.json`: Machine-derived representation of demonstrated mode capabilities across Platform V1.
- `reward_passthrough_parity.json`: Verified signed progress delta passthrough without clamping.
- `execution_config_lock.json`: Verified 10 Hz physical control, 0.02 step, decision repeat 5, Trigger mode.
- `sandbox_execution_smoke.json`: Successful real MetaDrive simulation execution of Sandbox episode.
- `audit_execution_smoke.json`: Successful real MetaDrive simulation execution of Audit episode.
