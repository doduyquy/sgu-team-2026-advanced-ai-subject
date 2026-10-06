# Platform V1 MapSuite Level 1 preview export audit

Measured on 2026-10-06 using Python 3.10.21, MetaDrive 0.4.3 at the clean pinned commit `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`. This verifies derived visualization infrastructure; no scientific algorithm milestone is implemented.

Machine evidence: [`verification.json`](../../results/audits/mapsuite_previews/verification.json). Source package: [`datasets/mapsuite_v1/`](../../datasets/mapsuite_v1/). Export/distribution contract: [distribution policy](../MAP_ARTIFACT_DISTRIBUTION.md).

## Identity and measured full export

- Baseline main: `00a34859fe6832695f90310b4af2026e13b83b4c` (PR #26).
- Full export source: `de1eb236969b4d94175fbd3ff6bbf1e06aef81b3`.
- Final code/test validation source: `34e1cfb724df779e410dea37046ffb7750375513`. Later changes add evidence/documentation and Git line-ending rules only. The full export used the earlier committed source; subsequent code adds output protection for Git metadata and additional fast regression tests, with no change to geometry reconstruction/rendering.
- Output: `artifacts/verified_preview_export/mapsuite_v1_previews` (gitignored).
- Images: **240/240**, each 1024 ? 1024; **0 reconstruction failures**, 240 unique IDs and filenames, all non-empty.
- Package size: **4,537,606 bytes**.
- ZIP: `artifacts/verified_preview_export/mapsuite_v1_previews_1.0.0_de1eb236969b.zip`; **3,771,796 bytes**.
- Full export duration: **284.453 seconds**.
- All image digests and 244 package-member checksums verified. The ZIP's 245 members match the extracted package byte for byte.
- One geometry per tier was rendered twice in fresh renderer lifecycles; all four pairs and their corresponding full-export images are byte-identical. This is same-environment evidence, not a claim about arbitrary platforms/library versions.
- Easy and Extreme samples were visually inspected for complete geometry fit, margins and lack of traffic overlays.

## Export architecture and source boundary

The exporter discovers all identities from the committed geometry table after the existing package verifier passes. It uses Gate 5's pinned procedural generation path, geometry seeds from the table, no stochastic traffic and no driving steps. Every serialized block sequence must match its frozen geometry hash before MetaDrive's native top-down road renderer writes a PNG. Environments close on success/failure. A staging directory becomes the final package only after complete verification. Existing output and ZIP paths are rejected.

Names: `<geometry_id>__<tier>__<sequence>__gseed-<NN>.png`, for example `geom_easy_SCS_seed0__Easy__SCS__gseed-00.png`. Unsafe characters are reversibly escaped. Both manifests, SOURCE_REF, sorted raw checksums and deterministic ZIP metadata keep the package self-describing. Synthetic tests are explicitly marked and rejected by the real verifier.

## Verification commands and observed counts

Run from `autonomous-driving-rl/` in the pinned project environment:

```powershell
python -m unittest tests.test_mapsuite_dataset -v
python -m unittest tests.test_mapsuite_preview_export -v
python -m unittest discover -s tests -p "test_*.py"
python scripts/verify_mapsuite_v1_dataset.py
python scripts/export_mapsuite_v1_previews.py --output artifacts/verified_preview_export/mapsuite_v1_previews --zip
python scripts/export_mapsuite_v1_previews.py --verify artifacts/verified_preview_export/mapsuite_v1_previews
```

MapSuite tests: **22 passed**. Preview tests: **20 passed**, including synthetic complete export, checksums/ZIP, provenance/identity mutation, source immutability, path protection, native reconstruction settings and pre-write geometry hash rejection.

The plain discovery command hit a native Windows access violation (`0xC0000005`) importing QtWidgets after MetaDrive, before completing tests. It also occurs when importing the existing launcher tests before Qt; preview-module import alone does not cause it. Importing Qt first allows the same entire suite to run:

```powershell
python -X faulthandler -u -c "from PySide6.QtCore import QProcess; from PySide6.QtWidgets import QApplication; import unittest; suite=unittest.defaultTestLoader.discover('tests',pattern='test_*.py'); result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not result.wasSuccessful())"
```

Observed: **393 tests, 392 passed, 0 failed, 1 skipped**, **58.837 seconds**. The skip is the existing Windows symlink escape test; symlink creation requires privileges unavailable to the test process. Tests are not removed, mocked out or weakened by the import-order workaround. Raw local logs remain under gitignored `artifacts/` and are not committed because they contain local paths.

## Scientific integrity

All **17** recomputed contract/registry/manifest hashes equal their baseline values; exact before/after values are in machine evidence and the [contract index](../PLATFORM_V1_CONTRACT_INDEX.md). Frozen code, configs, prior evidence and canonical dataset Git contents have no diff. All 24 canonical package-member checksums pass unchanged. No geometry, family, seed, split, evaluation case, action, observation, reward, metric, lifecycle, information right, preflight, logging, trust or TEST policy is changed. Stage 0 is not implemented and no fixture is promoted.

Git attributes address a pre-existing checkout-byte issue without changing canonical blobs or checksums: packaged text uses its recorded LF bytes; the five raw-provenance source inputs use their recorded CRLF bytes on all operating systems. This avoids Windows package-checksum failure and LF-only source-provenance failure. Values were checked against Git blobs and recorded digests before selecting those attributes.

## Human review items

1. **Historical lane metadata discrepancy:** packaged `generation_config.json` and geometry descriptive metadata declare two base lanes; the pinned Gate 5 reconstruction defaults to three. All 240 frozen block-sequence hashes match the three-lane reconstruction. This branch preserves those geometries and records both declared/effective values plus a warning in the generated manifest. Any correction to the frozen descriptive metadata requires a separate reviewed change; the exporter does not alter the source package.
2. **Windows native import order:** the ordinary discovery command needs environment follow-up for Qt after MetaDrive. The full suite passes with Qt imported first as recorded above. No frozen launcher/Workbench source is patched here.

The generated ZIP is local and ready for distribution. No Drive destination was supplied; no upload was performed. GitHub MapSuite V1 remains authoritative. No 240-image set is committed. Review and merge remain human actions; this branch must not be merged by the exporter task.
