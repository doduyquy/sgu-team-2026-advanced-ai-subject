# MapSuite V1 derived artifact distribution

**GitHub MapSuite V1 remains the authoritative scientific source.** Google Drive or similar storage is a distribution mirror for derived human-facing assets. It must never silently redefine geometry identity, split, seed, test manifest, difficulty or benchmark semantics. No team Drive URL or upload destination is configured by this document.

## Level 1 scope

[`scripts/export_mapsuite_v1_previews.py`](../scripts/export_mapsuite_v1_previews.py) exports exactly one deterministic top-down road-geometry PNG for each of the 240 records in [`datasets/mapsuite_v1/geometries.csv`](../datasets/mapsuite_v1/geometries.csv). It verifies the committed dataset against its frozen sources, requires the clean MetaDrive 0.4.3 Git checkout at `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`, reconstructs using Gate 5 settings and checks every serialized block-sequence hash before rendering.

The renderer fits the whole road network into a consistent square view with margin and no traffic/trajectory overlays. It uses MetaDrive's native top-down road renderer on an offscreen surface; no driving policy or environment steps are needed. Reset only initializes geometry and the engine. Every environment is closed even on failure. Traffic is disabled for visualization, while the scientific package's traffic settings remain unchanged.

This level exports no mesh, world serialization, binary cache, videos, GIFs, driving traces or RL transitions. PNGs do not add information rights to any agent. TEST geometry previews are authorized human-facing geometry inspection, not TEST agent evaluation or tuning evidence.

## Reproduce and verify

Activate the project Python 3.10 / pinned MetaDrive environment, commit exporter changes so provenance identifies the actual source, and run from `autonomous-driving-rl/`:

```powershell
python scripts/verify_mapsuite_v1_dataset.py
python scripts/export_mapsuite_v1_previews.py --output artifacts/mapsuite_v1_previews --zip
python scripts/export_mapsuite_v1_previews.py --verify artifacts/mapsuite_v1_previews
```

Export requires a clean tracked source tree and refuses existing output/ZIP paths. For another export choose a fresh directory. Failure leaves an explicitly incomplete staging directory for diagnosis, never a successful package. No failed geometry is skipped. The default image size is 1024 × 1024; renderer version/configuration is recorded in the JSON manifest. Bitwise reproducibility is checked for repeated exports in the same pinned environment; cross-platform renderer/library differences may affect image hashes without changing geometry hashes.

Fast tests do not initialize MetaDrive:

```powershell
python -m unittest tests.test_mapsuite_preview_export -v
python -m unittest tests.test_mapsuite_dataset -v
python -m unittest discover -s tests -p "test_*.py"
```

The full export command above is the separate 240-map integration check. The verifier checks exact file coverage, 240 unique IDs/names, dataset identity, geometry/source metadata, PNG dimensions, image digests and package checksums. Export with `--zip` also verifies the archive against the package before reporting success.

## Package and provenance

```text
artifacts/mapsuite_v1_previews/
├── README.md
├── preview_manifest.csv
├── preview_manifest.json
├── CHECKSUMS.sha256
├── SOURCE_REF.txt
└── previews/                       # Exactly 240 PNG files
```

Names are `<geometry_id>__<tier>__<sequence>__gseed-<NN>.png`, with the packaged ID retained. Unsafe filename characters are reversibly escaped; path traversal and Windows name collisions are rejected. For example: `geom_easy_SCS_seed0__Easy__SCS__gseed-00.png`.

Both manifests record filename, geometry ID, sequence, generation seed, tier, split, geometry/image SHA-256, width/height, dataset ID/version and MetaDrive version/commit. The JSON also records source Git SHA, exporter version/hash, renderer configuration and source package digests. `SOURCE_REF.txt` states the source repository/commit and simulator pin, and explicitly says:

> GitHub MapSuite V1 is authoritative. This package contains derived visualization artifacts only.

Sorted raw-file checksums cover all images, both manifests, README and SOURCE_REF; the checksum file excludes itself. ZIP entries have sorted names, fixed timestamps and fixed permissions. The archive is named `mapsuite_v1_previews_1.0.0_<source-short-sha>.zip` beside the output directory. It contains the entire self-describing package. The source SHA refers to the committed exporter source used for that export, which may precede a later documentation-only evidence commit.

Recipients can extract the ZIP and run `sha256sum -c CHECKSUMS.sha256` from its package directory, or use the repository verifier command above with the extracted directory. Checksum consistency alone is not proof of trusted origin; compare SOURCE_REF and source digests to the reviewed GitHub checkout. Do not replace canonical CSVs, splits, test cases or generation configs with mirror files.

## Mirror layout and retention

Recommended external layout (documentation only):

```text
MetaDrive_MapSuite_V1/
└── v1.0.0__source-<git-short-sha>/
    ├── README.md
    ├── SOURCE_REF.txt
    ├── CHECKSUMS.sha256
    ├── preview_manifest.csv
    ├── preview_manifest.json
    └── previews/                   # 240 derived images
```

Upload the verified ZIP as a unit, preserve version/source directories, and avoid ambiguous in-place replacement. This task performs no Drive upload. Generated packages and ZIPs live under gitignored `artifacts/`; the 240 generated images are intentionally absent from normal Git history. The 12 committed family representatives under `datasets/mapsuite_v1/previews/` remain unchanged.
