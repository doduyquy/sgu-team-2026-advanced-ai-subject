# Platform V1 MapSuite Level 1 preview export audit

Correction audit measured on 2026-10-06 using Python 3.10.21 and clean pinned MetaDrive 0.4.3 at `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`. This is local export/infrastructure evidence, not CI verification or a scientific agent milestone.

Machine evidence: [`verification.json`](../../results/audits/mapsuite_previews/verification.json). Read the prominently linked [lane metadata erratum](MAPSUITE_V1_LANE_METADATA_ERRATUM.md), [distribution policy](../MAP_ARTIFACT_DISTRIBUTION.md) and [Windows onboarding workaround](../GETTING_STARTED.md#known-windows-native-import-order-issue).

## Source and regenerated real artifact

- Frozen main baseline: `00a34859fe6832695f90310b4af2026e13b83b4c` (PR #26).
- Previous audited PR #27 HEAD: `9b4fd4c4c9e17f6b40a72897cf258f624ae347cc`.
- Provenance correction commit: `d53f01665fa206e7ceadd7cb4445b4af8d646713`.
- Documentation/export-source commit: `432a6e971ffbbfd10cce2c53e060fdfdc207ba10`; it contains the corrected exporter used for every newly rendered image. Later evidence-only changes record this measured run.
- Exporter version: `1.0.1`; exact recorded Git blob SHA-256: `bf3c271d11b93f6a0999883aee9c14301dba73ce0139780f69e8a970f78377d8`.
- Output: `artifacts/pr27_correction/verified_export/mapsuite_v1_previews` (gitignored).
- **240/240** non-empty **1024 x 1024** PNGs; **zero reconstruction/hash failures**; 240 unique geometry IDs and filenames; both manifests have 240 rows.
- Package size: **4,537,606 bytes**.
- ZIP: `artifacts/pr27_correction/verified_export/mapsuite_v1_previews_1.0.0_432a6e971ffb.zip`; **3,771,798 bytes**.
- Duration: **233.717 seconds**.
- All 244 package-member checksums and all 245 ZIP members verified. A separate CLI verification also passes.
- Internal integrity, recorded-source Git provenance and current checkout compatibility each report **VERIFIED**.
- All 240 newly rendered image hashes match the prior frozen-geometry previews. The previous package was not reused as the final artifact.

## Provenance blocker correction

The old verifier could accept coherent local mutations to source SHA/exporter hash fields. The correction requires `source_git_sha` to be an exact existing commit object in this repository. Tree/blob/tag objects and arbitrary valid-looking nonexistent hashes are rejected. It retrieves the exporter from **that recorded commit** using Git plumbing and hashes the exact blob bytes. The recorded exporter SHA-256 must match; its version must match the literal assignment parsed from that blob without executing historical code.

It also compares the four recorded canonical package hashes (geometries, generation config, dataset manifest and checksum file) to the corresponding exact blobs at the recorded commit. Git blobs, not EOL-converted checkout bytes, define those provenance digests. Independently using `git cat-file -e <sha>^{commit}` and `git show <sha>:<path>` confirmed the commit, exporter digest and all four package digests for this real export.

Current checkout compatibility remains separate: the current canonical package and frozen sources must pass the dataset verifier, package metadata and image rows must agree with the current geometry table, and current package bytes must agree with the recorded canonical digests. Internal package integrity still checks PNG hashes/dimensions, exact member coverage, matching CSV/JSON manifests, SOURCE_REF and package checksums.

An older legitimate export need not use current HEAD or today's exporter version. The verifier checks its recorded commit/blob/version and current dataset compatibility. The previous real archive passes the stronger verifier as a backward-compatibility probe; it is not the final reviewed artifact. Consistency with Git objects is not a signed attestation of who performed execution.

## Commands and observed regression results

From `autonomous-driving-rl/` in the pinned environment:

```powershell
python -m unittest tests.test_mapsuite_dataset -v
python -m unittest tests.test_mapsuite_preview_export -v
python scripts/verify_mapsuite_v1_dataset.py
python -m unittest discover -s tests -p "test_*.py"
```

Dataset tests: **22 passed**. Preview tests: **27 passed**, including seven new provenance regressions. Tests mutate source commit/exporter/source-package hashes and regenerate SOURCE_REF plus package checksums, then require Git-backed rejection. Positive cases verify the real recorded commit/blob and an older legitimate exporter version. A blob object used as a commit is rejected.

Ordinary full discovery still exits with **0xC0000005** when MetaDrive is imported before Qt on this Windows environment, before completing the suite. The crash-dialog suppression used when recording its exit code does not change test selection or import order. The documented Qt-preloaded command discovers the same full pattern:

```powershell
python -X faulthandler -u -c "from PySide6.QtCore import QProcess; from PySide6.QtWidgets import QApplication; import unittest; suite=unittest.defaultTestLoader.discover('tests',pattern='test_*.py'); result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not result.wasSuccessful())"
```

Observed: **400 discovered, 399 passed, 0 failures, 0 errors, 1 existing Windows symlink-related skip**, **81.480 seconds**. Coverage is the original 393 tests plus all seven new tests; none are excluded. The prior audit's 393/392/0/1 result remains historical onboarding context, not the current count. No Launcher/Workbench redesign or frozen source fix is made for the native environment issue.

```powershell
python scripts/export_mapsuite_v1_previews.py --output artifacts/pr27_correction/verified_export/mapsuite_v1_previews --zip
python scripts/export_mapsuite_v1_previews.py --verify artifacts/pr27_correction/verified_export/mapsuite_v1_previews
```

Both real export and separate verification pass. Raw local logs remain in ignored `artifacts/pr27_correction/` because they contain local paths.

## Lane erratum and scientific integrity

`declared_base_lane_num = 2` and `effective_base_lane_num = 3` remain distinct. The generated manifest records the historical declaration as `dataset_declared_base_lane_num = 2`, effective runtime value 3 and an explicit warning. Gate 2's shorthand environment creation did not apply the intended two-lane override; its descriptive writers recorded 2 regardless. All 240 actual serialized geometry hashes match the effective three-lane path. See the [dedicated erratum](MAPSUITE_V1_LANE_METADATA_ERRATUM.md) for affected fields, source evidence and requirements for a future versioned correction.

PR #27 does not rewrite canonical geometry/config/checksum files, Gate 2 evidence or Gate 5 manifests. All **17/17 scientific/registry/manifest hashes** were recomputed unchanged; exact before/after values are in machine evidence. All 24 canonical package checksums pass unchanged. The baseline comparison `git diff 00a34859fe6832695f90310b4af2026e13b83b4c -- autonomous-driving-rl/datasets/mapsuite_v1` is empty. Frozen source/config and prior scientific audit evidence are unchanged.

No split, seed, action, observation, reward, metric, lifecycle, input right, holdout, preflight, logging or trust semantics change. Stage 0 is not implemented; no fixture is promoted. PNGs/ZIPs are not committed or uploaded. GitHub remains authoritative. The 25 accidental question-mark punctuation artifacts were replaced with safe ASCII, and local-link/privacy/encoding scans pass. PR #27 remains for human review and must not be merged as part of this task.
