# Gate 7.5B Workbench Audit Summary (Pass B1 Correction 1)

- Gate: Gate 7.5B (Research Workbench GUI Pass B1 Correction 1)
- Status: AUDIT-CANDIDATE
- Baseline Main Merge SHA: `0376da8b8bd3e1c3b60d43ed371b6713c0cb31f6`
- PySide6 Pinned Version: `6.11.2`
- Protocol Version: `WORKBENCH_PROTOCOL_V1`

## Contract Cryptographic Fingerprints
- `gate5_benchmark_contract_sha256`: `9ddd889b84d8705fae618879e5035556c80d0276a3dc2a58a7963937ebb59f77` (VERIFIED UNCHANGED)
- `gate6_agent_contract_sha256`: `53aa37079ff44afa75d9a3f921b0c1f98c4600d882fced51fc8d9197795058eb` (VERIFIED UNCHANGED)
- `gate6_platform_runtime_contract_sha256`: `c7698768539a769c7b2bc6b90771ff234276a974b6e0353bbb03119faf2f79ad` (VERIFIED UNCHANGED)
- `gate7_logging_contract_sha256`: `0d17915556d83f49d9519576e95920c0853700f33a91593ef85b3aeef6cbb9f2` (VERIFIED UNCHANGED)
- `gate7_platform_observability_contract`: `f00c27fca8abf8145596576cd0dd2540181ff8ae7eccf7806c864921933d3581` (VERIFIED UNCHANGED)
- `launcher_contract_sha256`: `5e6269b6e32349bd3ee34ca52583c00f832bd0a737330c894e9a8d314653db04` (VERIFIED UNCHANGED)
- `platform_execution_contract_sha256`: `442beafd86212419dc7b3edfa60e53715bf33877d3dd72e61c54091520b0563d` (VERIFIED UNCHANGED)
- `canonical_agent_registry_sha256`: `9fac42d07949796d00b6dc869b98c945eae7a59355d7928f429e457f1ee7f26f` (VERIFIED UNCHANGED)
- `workbench_contract_sha256`: `e5711485e6571a04c336739ebc6f285213a1d9631fa29820a89707b37875a82b` (CANDIDATE)
- `platform_workbench_contract_sha256`: `36f3a0891a9153df12a8b53b01f2064afbf848eafda462149a768ec989199e47` (CANDIDATE)

## Empirical Evidence
1. **Startup Smoke:** Offscreen construction successful; 5 tabs and initial tables populated. Authoritative manifest counts verified: TRAIN 180, VALIDATION 96, TEST 60. Extreme tier available.
2. **PLAN Smoke:** Subprocess resolved plan & preflight, 0 simulator steps executed. Sanitized WORKER_READY (basename only).
3. **RUN Smoke:** Real MetaDrive execution in subprocess; streamed telemetry via `LauncherEventV1`; status `COMPLETE`; `WORKER_DONE.success == True`; run_id parity verified; 7/7 Gate-7 artifacts generated.
4. **Canonical Blocked Smoke:** Benchmark TEST with non-benchmark fixture failed preflight on `agent_benchmark_eligibility`, with 0 simulator steps.
5. **Thin-Client Invariant:** GUI contains 0 scientific evaluation authority; core resolver, preflight, and executor remain authoritative.
6. **Privacy Invariant:** Zero private absolute machine paths or credentials persisted in audit artifacts.
