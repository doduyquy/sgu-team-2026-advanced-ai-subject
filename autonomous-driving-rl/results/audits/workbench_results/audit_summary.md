# Gate 7.5B Pass B2 Results & Live UX Audit Summary

- Gate: Gate 7.5B Pass B2 (Results Browser, Artifact Integrity & Live UX)
- Status: AUDIT-CANDIDATE
- Baseline Main Merge SHA: `99f0a66aa156004a5a1ec7993d2368ff83d7c325`
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
- `workbench_contract_sha256`: `e5711485e6571a04c336739ebc6f285213a1d9631fa29820a89707b37875a82b` (VERIFIED UNCHANGED)
- `platform_workbench_contract_sha256`: `36f3a0891a9153df12a8b53b01f2064afbf848eafda462149a768ec989199e47` (VERIFIED UNCHANGED)
- `workbench_results_contract_sha256`: `25716cc6aa0d93b2e6a282e51a7e23b37b050a975654ae2852726c2e86828244` (CANDIDATE)
- `platform_workbench_results_contract_sha256`: `b53e56b6344c53299ffdee8f1402e4fccb16c375eeb08205897cf909effd89a3` (CANDIDATE)

## Empirical Evidence
1. **Startup Smoke:** Offscreen construction with 6 main tabs. Results tab includes 6 read-only subtabs (Overview, Episodes, Timing, Provenance, Integrity, W&B) and 8-column runs browser.
2. **Complete Run Load:** Real simulator-backed run executed, persisted 7/7 Gate-7 files, loaded via Results Repository, evaluated as VERIFIED with bit-for-bit Gate-7 schema parity.
3. **Isolated Tamper Detection:** Exact canonical semantic content hash mismatch detection on copied run without repairing or rewriting artifacts.
4. **Incomplete Run Semantics:** RUNNING and FAILED runs evaluated as NOT_FINAL without fabricating missing summaries.
5. **Live Telemetry Buffer:** In-memory trace buffering verified strictly from `LauncherEventV1` events sampled every 10 decision steps.
6. **No-Ranking Invariant:** Inspects individual runs only; zero multi-run leaderboard or ranking semantics.
