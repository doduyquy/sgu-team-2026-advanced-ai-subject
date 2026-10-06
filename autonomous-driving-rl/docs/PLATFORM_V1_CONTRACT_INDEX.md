# Platform V1 contract and hash index

Verified registry for the PR #26 baseline. Values below are taken from committed machine evidence, not copied from a chat. The 0.4.3 simulator pin is `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`. See [build record](PLATFORM_V1_BUILD_RECORD.md) for the merges that introduced each lock.

## Hash semantics

Contract hashes use `src/platform/protocol.py:canonical_json_sha256`: parsed JSON, sorted keys, compact separators, `allow_nan=False`. Manifest hashes use `canonical_csv_file_sha256`: an ordered list of row dictionaries with sorted keys. They are insensitive to CRLF/LF. Dataset `CHECKSUMS.sha256` and preview image hashes instead cover raw file bytes. A raw file digest may differ from the semantic digest; they must not be substituted for one another. Composite platform hashes bind the preceding gate, new contract, pin and gate metadata as defined by the producing module.

## Gate 5

Benchmark schemas, MapSuite, lifecycle, reward, metrics, splits and seed plan.

- Producing implementation: [`src/platform/protocol.py`](../src/platform/protocol.py).
- Machine config: [`configs/platform/evaluation_protocol_v1.json`](../configs/platform/evaluation_protocol_v1.json).
- Audit: [`scripts/audit_evaluation_protocol.py`](../scripts/audit_evaluation_protocol.py).
- Evidence: [`results/audits/evaluation_protocol/protocol_hashes.json`](../results/audits/evaluation_protocol/protocol_hashes.json).
- Downstream: Gate 6 runtime and all canonical evaluation.

| Hash key | Verified value |
|---|---|
| `benchmark_contract_sha256` | `9ddd889b84d8705fae618879e5035556c80d0276a3dc2a58a7963937ebb59f77` |

## Gate 6

Agent API, AgentInputV1 rights and certified action adapters; runtime joins Gate 5 and agent contract.

- Producing implementation: [`src/platform/agent.py`](../src/platform/agent.py).
- Machine config: [`configs/platform/agent_contract_v1.json`](../configs/platform/agent_contract_v1.json).
- Audit: [`scripts/audit_agent_contract.py`](../scripts/audit_agent_contract.py).
- Evidence: [`results/audits/agent_contract/contract_hashes.json`](../results/audits/agent_contract/contract_hashes.json).
- Downstream: Gate 7 observability and launcher.

| Hash key | Verified value |
|---|---|
| `agent_contract_sha256` | `53aa37079ff44afa75d9a3f921b0c1f98c4600d882fced51fc8d9197795058eb` |
| `platform_runtime_contract_sha256` | `c7698768539a769c7b2bc6b90771ff234276a974b6e0353bbb03119faf2f79ad` |

## Gate 7

Local logs, provenance, failure states and downstream W&B; observability joins runtime and logging.

- Producing implementation: [`src/platform/experiment_logging.py`](../src/platform/experiment_logging.py).
- Machine config: [`configs/platform/logging_contract_v1.json`](../configs/platform/logging_contract_v1.json).
- Audit: [`scripts/audit_logging_contract.py`](../scripts/audit_logging_contract.py).
- Evidence: [`results/audits/logging_contract/contract_hashes.json`](../results/audits/logging_contract/contract_hashes.json).
- Downstream: Gate 7.5A execution.

| Hash key | Verified value |
|---|---|
| `logging_contract_sha256` | `0d17915556d83f49d9519576e95920c0853700f33a91593ef85b3aeef6cbb9f2` |
| `platform_observability_contract_sha256` | `f00c27fca8abf8145596576cd0dd2540181ff8ae7eccf7806c864921933d3581` |

## Gate 7.5A

Plan resolution, preflight and registration identity; execution joins observability and launcher.

- Producing implementation: [`src/launcher/contracts.py`](../src/launcher/contracts.py) and [`src/launcher/registry.py`](../src/launcher/registry.py).
- Machine config: [`configs/platform/launcher_contract_v1.json`](../configs/platform/launcher_contract_v1.json).
- Audit: [`scripts/audit_launcher_core.py`](../scripts/audit_launcher_core.py).
- Evidence: [`results/audits/launcher_core/contract_hashes.json`](../results/audits/launcher_core/contract_hashes.json).
- Downstream: Workbench B1 and plan identity.

| Hash key | Verified value |
|---|---|
| `launcher_contract_sha256` | `5e6269b6e32349bd3ee34ca52583c00f832bd0a737330c894e9a8d314653db04` |
| `platform_execution_contract_sha256` | `442beafd86212419dc7b3edfa60e53715bf33877d3dd72e61c54091520b0563d` |
| `canonical_agent_registry_sha256` | `9fac42d07949796d00b6dc869b98c945eae7a59355d7928f429e457f1ee7f26f` |

## Workbench B1

UI/worker isolation and protocol; platform hash joins execution and Workbench.

- Producing implementation: [`src/workbench/contracts.py`](../src/workbench/contracts.py).
- Machine config: [`configs/platform/workbench_contract_v1.json`](../configs/platform/workbench_contract_v1.json).
- Audit: [`scripts/audit_workbench.py`](../scripts/audit_workbench.py).
- Evidence: [`results/audits/workbench/contract_hashes.json`](../results/audits/workbench/contract_hashes.json).
- Downstream: Workbench B2.

| Hash key | Verified value |
|---|---|
| `workbench_contract_sha256` | `e5711485e6571a04c336739ebc6f285213a1d9631fa29820a89707b37875a82b` |
| `platform_workbench_contract_sha256` | `36f3a0891a9153df12a8b53b01f2064afbf848eafda462149a768ec989199e47` |

## Workbench B2

Verified local results, integrity and provisional live telemetry; platform hash joins B1 and B2.

- Producing implementation: [`src/workbench/results_contracts.py`](../src/workbench/results_contracts.py).
- Machine config: [`configs/platform/workbench_results_contract_v1.json`](../configs/platform/workbench_results_contract_v1.json).
- Audit: [`scripts/audit_workbench_results.py`](../scripts/audit_workbench_results.py).
- Evidence: [`results/audits/workbench_results/contract_hashes.json`](../results/audits/workbench_results/contract_hashes.json).
- Downstream: Workbench B3.

| Hash key | Verified value |
|---|---|
| `workbench_results_contract_sha256` | `21c2c541a891967d9206e2dc86781f1bfc62b794c13e7e1ee1c3732676e38429` |
| `platform_workbench_results_contract_sha256` | `a241e2890b65290e28d717714123625bc2809100fe3287f2fb3e14dba8d6e433` |

## Workbench B3

Bounded resources, lifecycle/recovery and trust; platform hash joins B2 and B3.

- Producing implementation: [`src/workbench/hardening_contracts.py`](../src/workbench/hardening_contracts.py).
- Machine config: [`configs/platform/workbench_hardening_contract_v1.json`](../configs/platform/workbench_hardening_contract_v1.json).
- Audit: [`scripts/audit_workbench_hardening.py`](../scripts/audit_workbench_hardening.py).
- Evidence: [`results/audits/workbench_hardening/contract_hashes.json`](../results/audits/workbench_hardening/contract_hashes.json).
- Downstream: All subsequent Workbench integration.

| Hash key | Verified value |
|---|---|
| `workbench_hardening_contract_sha256` | `adb11f5c205e0ee6651f1544f76deee36416fc49070a52a513cc5c786ceecdab` |
| `platform_workbench_hardening_contract_sha256` | `b476749713e8a7afa2eafe50a5a89775550afe051a16b203a07d846c41f6972c` |

The canonical registry is produced by `src/launcher/registry.py`, recorded in [`results/audits/launcher_core/agent_registry_snapshot.json`](../results/audits/launcher_core/agent_registry_snapshot.json) and [`results/audits/launcher_core/canonical_registry_integrity.json`](../results/audits/launcher_core/canonical_registry_integrity.json). Its hash covers canonical registrations; fixture registration never makes an agent a scientific baseline.

## Gate 5 manifest and component hashes

All values are from [`results/audits/evaluation_protocol/protocol_hashes.json`](../results/audits/evaluation_protocol/protocol_hashes.json). Producing code/audit: [`src/platform/protocol.py`](../src/platform/protocol.py) / [`scripts/audit_evaluation_protocol.py`](../scripts/audit_evaluation_protocol.py). The benchmark payload binds every component below, and downstream runtime/launcher checks depend on them.

| Payload key | Semantic SHA-256 | Config/schema/manifest |
|---|---|---|
| `observation_schema_sha256` | `3b76413fb7c191f47a16f59fb879cc92dd3ea379d340011290dd6cf482ed07ac` | [`results/audits/observation_action/observation_schema.json`](../results/audits/observation_action/observation_schema.json) |
| `action_schema_sha256` | `f042c413f1ad9fab849434fe1c5394ce1bd43616f6fcf09d6b94e02e91d138d6` | [`results/audits/observation_action/action_schema.json`](../results/audits/observation_action/action_schema.json) |
| `mapsuite_manifest_sha256` | `ae048a80c1bfc475a15ababc64f51a41ca8f7494a4b3d93bc2af6fcf7aebba08` | [`configs/maps/mapsuite_v1_candidates.json`](../configs/maps/mapsuite_v1_candidates.json) |
| `canonical_candidates_sha256` | `ea44f5525b76bcd6339fb42c41a20d5f0482fd314c42cbed5385a619b51f3084` | [`results/audits/mapsuite/canonical_candidates.json`](../results/audits/mapsuite/canonical_candidates.json) |
| `episode_spec_sha256` | `b51bd75545a894ffa35e92149d0f666bcc7d9e449d6b84d4632540461aac823e` | [`configs/platform/episode_spec_v1.json`](../configs/platform/episode_spec_v1.json) |
| `reward_spec_sha256` | `673f9e18f8384ed9b981f3099ea9eb53082bc40c0ac5ab9a7d95a732db392388` | [`configs/platform/reward_spec_v1.json`](../configs/platform/reward_spec_v1.json) |
| `evaluation_metrics_sha256` | `4b3bd1ea18c1bb6cd1fd91675948eb7f0f907bcb0bb063abeade57cfdda2e751` | [`configs/platform/evaluation_metrics_v1.json`](../configs/platform/evaluation_metrics_v1.json) |
| `scenario_split_spec_sha256` | `4e77783c5685d021babe5c00d7a03bebacc10eb252e5e056285c65b9e3f986b6` | [`configs/platform/scenario_split_v1.json`](../configs/platform/scenario_split_v1.json) |
| `seed_plan_spec_sha256` | `61a9e0c9106f415aa3d5bae0523069035f5e027eda7273382f93cee72ee5d57a` | [`configs/platform/seed_plan_v1.json`](../configs/platform/seed_plan_v1.json) |
| `geometry_split_manifest_sha256` | `3b07e94b99766f409b000455304ae2980a05467156734ee519ccf27068bdc481` | [`results/audits/evaluation_protocol/geometry_split_manifest.csv`](../results/audits/evaluation_protocol/geometry_split_manifest.csv) |
| `test_case_manifest_sha256` | `0832c38e2e8a0a3bb0c6cafbb2ba63cfcd1dcbbf84b4c7d6b31b9eaf5dc8ec77` | [`results/audits/evaluation_protocol/test_case_manifest.csv`](../results/audits/evaluation_protocol/test_case_manifest.csv) |
| `validation_case_manifest_sha256` | `10afc2afcfd1c3e8e6f201448bdeb79ee5b07b3c0ec92e8e6e1adb7c10f42ac1` | [`results/audits/evaluation_protocol/validation_case_manifest.csv`](../results/audits/evaluation_protocol/validation_case_manifest.csv) |
| `evaluation_protocol_core_sha256` | `30b78e6e1b2d0b84d2c74326080a59dc57d5a42d0d400a60edd161d76ad5209e` | [`configs/platform/evaluation_protocol_v1.json`](../configs/platform/evaluation_protocol_v1.json) |

`test_manifest_sha256` in the evidence is an alias for payload `test_case_manifest_sha256`. The geometry, validation and test manifest semantic digests are respectively `3b07e94b99766f409b000455304ae2980a05467156734ee519ccf27068bdc481`, `10afc2afcfd1c3e8e6f201448bdeb79ee5b07b3c0ec92e8e6e1adb7c10f42ac1`, and `0832c38e2e8a0a3bb0c6cafbb2ba63cfcd1dcbbf84b4c7d6b31b9eaf5dc8ec77`.

## Packaged dataset and derived previews

[`datasets/mapsuite_v1/dataset_manifest.json`](../datasets/mapsuite_v1/dataset_manifest.json) records raw source-file digests and the `mapsuite_v1` / `1.0.0` identity. [`datasets/mapsuite_v1/CHECKSUMS.sha256`](../datasets/mapsuite_v1/CHECKSUMS.sha256) covers the committed package. Run `python scripts/verify_mapsuite_v1_dataset.py` to compare the package to frozen sources. Individual `geometry_sha256` values fingerprint serialized block sequences, not PNG pixels. Preview `image_sha256` values identify derived files and confer no benchmark authority.

## Verify without mutating locks

From `autonomous-driving-rl/`, run `python -m unittest discover -s tests -p "test_*.py"` and `python scripts/verify_mapsuite_v1_dataset.py`. Existing contract tests recompute locked contract/registry identities. Review `git diff <baseline> -- configs/platform datasets/mapsuite_v1 results/audits src/platform src/launcher src/workbench` for unintended changes. Do not rerun an audit into committed evidence directories as a way to update a lock. A legitimate semantic change requires a reviewed new version.
