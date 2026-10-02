# Gate 7.5B Workbench Audit Summary

- Gate: Gate 7.5B (Research Workbench GUI Pass B1)
- Status: AUDIT-CANDIDATE
- Baseline Main Merge SHA: `0376da8b8bd3e1c3b60d43ed371b6713c0cb31f6`
- PySide6 Pinned Version: `6.11.2`
- Protocol Version: `WORKBENCH_PROTOCOL_V1`

## Contract Cryptographic Fingerprints
- `launcher_contract_sha256`: `5e6269b6e32349bd3ee34ca52583c00f832bd0a737330c894e9a8d314653db04` (VERIFIED UNCHANGED)
- `platform_execution_contract_sha256`: `442beafd86212419dc7b3edfa60e53715bf33877d3dd72e61c54091520b0563d` (VERIFIED UNCHANGED)
- `canonical_agent_registry_sha256`: `9fac42d07949796d00b6dc869b98c945eae7a59355d7928f429e457f1ee7f26f` (VERIFIED UNCHANGED)
- `workbench_contract_sha256`: `e5711485e6571a04c336739ebc6f285213a1d9631fa29820a89707b37875a82b` (CANDIDATE)
- `platform_workbench_contract_sha256`: `36f3a0891a9153df12a8b53b01f2064afbf848eafda462149a768ec989199e47` (CANDIDATE)

## Empirical Evidence
1. **Startup Smoke:** Offscreen construction successful; 5 tabs and initial tables populated.
2. **PLAN Smoke:** Subprocess resolved plan & preflight, 0 simulator steps executed.
3. **RUN Smoke:** Real MetaDrive execution in subprocess; streamed telemetry via `LauncherEventV1`; 7/7 Gate-7 artifacts generated.
4. **Canonical Blocked Smoke:** Benchmark TEST with non-benchmark fixture resolved plan, failed preflight on `agent_benchmark_eligibility`, and exited with 0 simulator steps.
5. **Thin-Client Invariant:** GUI contains 0 scientific evaluation authority; core resolver, preflight, and executor remain authoritative.
