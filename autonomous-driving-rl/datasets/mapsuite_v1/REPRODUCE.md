# Reproducing & Verifying MetaDrive MapSuite V1

This guide explains how any teammate can reproducibly export, inspect, and verify the `mapsuite_v1` dataset package on a clean installation.

---

## 1. Prerequisites

MapSuite V1 requires Python 3.10 and the pinned MetaDrive simulator baseline:

- **Python:** `3.10`
- **MetaDrive Version:** `0.4.3`
- **Pinned MetaDrive Commit:** `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`

See [`docs/environment/METADRIVE_SETUP.md](../../docs/environment/METADRIVE_SETUP.md) for full installation and virtual environment instructions.

---

## 2. Inspecting the Authoritative Source Artifacts

MapSuite V1 is derived deterministically from existing, frozen Platform V1 audit artifacts. You can inspect the source files directly:

```powershell
# In autonomous-driving-rl/
# Inspect candidate metrics (240 geometry rows)
Get-Content results/audits/mapsuite/candidate_metrics.csv -TotalCount 5

# Inspect canonical geometry partition (180 TRAIN / 48 VAL / 12 TEST)
Get-Content results/audits/evaluation_protocol/geometry_split_manifest.csv -TotalCount 5

# Inspect canonical evaluation suites (96 validation cases, 60 test cases)
Get-Content results/audits/evaluation_protocol/validation_case_manifest.csv -TotalCount 5
Get-Content results/audits/evaluation_protocol/test_case_manifest.csv -TotalCount 5
```

---

## 3. Re-exporting the Dataset Package

To re-export the entire dataset package programmatically from the source artifacts:

```powershell
# From autonomous-driving-rl/
python scripts/export_mapsuite_v1_dataset.py
```

### Optional: Export to a Custom / Temporary Directory
You can test export into a temporary path without touching the committed repository files:

```powershell
python scripts/export_mapsuite_v1_dataset.py --output /tmp/test_mapsuite_v1
```

The exporter:
1. Validates all source artifacts and their cryptographic hashes;
2. Joins geometric metadata and partitions;
3. Ensures strict value agreement across sources;
4. Writes `geometries.csv`, `scenario_families.csv`, splits, and evaluation cases;
5. Copies the 12 representative preview images;
6. Computes deterministic SHA-256 fingerprints in `CHECKSUMS.sha256`.

---

## 4. Verifying Package Integrity

To independently verify the committed dataset package against all Platform V1 contracts and checksums:

```powershell
python scripts/verify_mapsuite_v1_dataset.py
```

The verifier executes comprehensive structural and cryptographic checks:
- Verifies exact counts (240 geometries, 12 families, 20 seeds/family, 180/48/12 splits, 96/60 evaluation cases, 12 previews);
- Validates split disjointness and completeness;
- Confirms validation and test cases reference only their respective splits;
- Validates source artifact SHA-256 values recorded in `dataset_manifest.json`;
- Validates all file hashes in `CHECKSUMS.sha256`.

Any discrepancy causes the verifier to exit with a non-zero exit code.

---

## 5. Running the Automated Test Suite

To run the automated regression tests covering dataset integrity, exporter determinism, and path safety:

```powershell
python -m unittest tests.test_mapsuite_dataset -v
```

---

## 6. Important Conceptual Rule

**Regenerating the dataset package does NOT redesign the benchmark.**
MapSuite V1 is an export view of the frozen Platform V1 benchmark. The geometry IDs, block compositions, scenario seeds, split assignments, environment seeds, and test case manifests are permanently locked by Platform V1 contracts (`benchmark_contract_sha256`).
