#!/usr/bin/env python3
"""
MetaDrive MapSuite V1 Dataset Verifier.

Independently validates the structural, semantic, and cryptographic integrity
of the MapSuite V1 procedural benchmark dataset package against frozen Platform V1
contracts and source artifacts.

Exits with code 0 on complete pass, or code 1 on any failure.
Never attempts silent auto-repair.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple

# Pinned expectations
EXPECTED_METADRIVE_VERSION = "0.4.3"
EXPECTED_METADRIVE_COMMIT = "85e5dadc6c7436d324348f6e3d8f8e680c06b4db"

EXPECTED_TIERS = ["Easy", "Medium", "Hard", "Extreme"]
EXPECTED_FAMILIES = {
    "Easy": ["SCS", "SCSS", "SCCS"],
    "Medium": ["SCXCS", "SCTCS", "SCXCCS"],
    "Hard": ["SCXOCS", "SCTXrCS", "XTOCS"],
    "Extreme": ["CrXROSTR", "SCXOCrTYCS", "SCTXORyCCS"],
}

REQUIRED_FILES = [
    "README.md",
    "SCHEMA.md",
    "REPRODUCE.md",
    "CHECKSUMS.sha256",
    "dataset_manifest.json",
    "generation_config.json",
    "geometries.csv",
    "scenario_families.csv",
    "splits/train_geometries.csv",
    "splits/validation_geometries.csv",
    "splits/test_geometries.csv",
    "evaluation_cases/validation_cases.csv",
    "evaluation_cases/test_cases.csv",
]

EXPECTED_PREVIEWS = [
    "easy_rank1_SCS_seed11.png",
    "easy_rank2_SCSS_seed9.png",
    "easy_rank3_SCCS_seed9.png",
    "medium_rank1_SCXCS_seed11.png",
    "medium_rank2_SCTCS_seed0.png",
    "medium_rank3_SCXCCS_seed13.png",
    "hard_rank1_SCXOCS_seed2.png",
    "hard_rank2_SCTXrCS_seed1.png",
    "hard_rank3_XTOCS_seed19.png",
    "extreme_rank1_CrXROSTR_seed6.png",
    "extreme_rank2_SCXOCrTYCS_seed16.png",
    "extreme_rank3_SCTXORyCCS_seed4.png",
]


def find_project_root() -> Path:
    """Finds the autonomous-driving-rl project root."""
    curr = Path(__file__).resolve().parent
    for p in [curr, curr.parent, curr.parent.parent]:
        if (p / "results" / "audits" / "mapsuite").exists():
            return p
    cwd = Path.cwd()
    if (cwd / "results" / "audits" / "mapsuite").exists():
        return cwd
    if (cwd / "autonomous-driving-rl" / "results" / "audits" / "mapsuite").exists():
        return cwd / "autonomous-driving-rl"
    raise RuntimeError("Could not find autonomous-driving-rl project root.")


def compute_file_sha256(path: Path) -> str:
    """Computes SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_dataset(dataset_dir: Path, project_root: Path) -> bool:
    """Runs all verification stages. Returns True on success, False on failure."""
    print("============================================================")
    print("VERIFYING METADRIVE MAPSUITE V1 DATASET PACKAGE")
    print("============================================================")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Project root:      {project_root}")

    errors: List[str] = []

    # 1. Structure Verification
    print("\n--- 1. Structure & Required Files ---")
    for req in REQUIRED_FILES:
        fp = dataset_dir / req
        if not fp.is_file():
            errors.append(f"Missing required file: {req}")

    previews_dir = dataset_dir / "previews"
    if not previews_dir.is_dir():
        errors.append(f"Missing required directory: previews/")
    else:
        found_previews = sorted([p.name for p in previews_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
        if found_previews != sorted(EXPECTED_PREVIEWS):
            errors.append(f"Preview images mismatch: found {len(found_previews)}, expected {len(EXPECTED_PREVIEWS)}")
        else:
            print(f"  [OK] Exactly 12 expected preview images verified.")

    if errors:
        for e in errors:
            print(f"  [FAIL] {e}")
        return False
    print("  [OK] All required files and directories exist.")

    # 2. Geometries Master Table Verification
    print("\n--- 2. Geometries Master Table (geometries.csv) ---")
    geoms_path = dataset_dir / "geometries.csv"
    with open(geoms_path, newline="", encoding="utf-8") as f:
        geoms_rows = list(csv.DictReader(f))

    if len(geoms_rows) != 240:
        errors.append(f"geometries.csv row count is {len(geoms_rows)}, expected 240.")

    geom_ids = [r["geometry_id"] for r in geoms_rows]
    if len(geom_ids) != len(set(geom_ids)):
        errors.append("Duplicate geometry_id values found in geometries.csv.")
    else:
        print(f"  [OK] Exactly 240 unique geometry_id values verified.")

    geom_keys = [(r["tier"], r["sequence"], int(r["geometry_generation_seed"])) for r in geoms_rows]
    if len(geom_keys) != len(set(geom_keys)):
        errors.append("Duplicate (tier, sequence, geometry_generation_seed) tuples found in geometries.csv.")
    else:
        print(f"  [OK] Zero duplicate (tier, sequence, seed) keys.")

    # Family counts and membership
    family_counts: Dict[Tuple[str, str], int] = {}
    for r in geoms_rows:
        key = (r["tier"], r["sequence"])
        family_counts[key] = family_counts.get(key, 0) + 1

    expected_all_families = []
    for tier, seqs in EXPECTED_FAMILIES.items():
        for s in seqs:
            expected_all_families.append((tier, s))

    if set(family_counts.keys()) != set(expected_all_families):
        errors.append(f"Families mismatch: {set(family_counts.keys()) ^ set(expected_all_families)}")
    else:
        print(f"  [OK] Exactly 12 expected scenario families verified.")

    for f_key, count in family_counts.items():
        if count != 20:
            errors.append(f"Family {f_key} has {count} geometries, expected 20.")
    print(f"  [OK] Exactly 20 geometries per scenario family verified across all 12 families.")

    for r in geoms_rows:
        if r["generation_success"].strip().lower() not in ("true", "1"):
            errors.append(f"Geometry {r['geometry_id']} has generation_success={r['generation_success']}")

    # 3. Splits Verification
    print("\n--- 3. Splits Partitioning (splits/) ---")
    splits_expected = {"TRAIN": 180, "VALIDATION": 48, "TEST": 12}
    split_files = {
        "TRAIN": dataset_dir / "splits" / "train_geometries.csv",
        "VALIDATION": dataset_dir / "splits" / "validation_geometries.csv",
        "TEST": dataset_dir / "splits" / "test_geometries.csv",
    }

    split_geoms_union: Set[str] = set()
    split_id_sets: Dict[str, Set[str]] = {}

    for split_name, expected_count in splits_expected.items():
        sp = split_files[split_name]
        with open(sp, newline="", encoding="utf-8") as f:
            sp_rows = list(csv.DictReader(f))

        if len(sp_rows) != expected_count:
            errors.append(f"{split_name} split has {len(sp_rows)} rows, expected {expected_count}")

        sp_ids = set(r["geometry_id"] for r in sp_rows)
        split_id_sets[split_name] = sp_ids

        # Ensure all rows declare the matching split column
        for r in sp_rows:
            if r["split"] != split_name:
                errors.append(f"Row {r['geometry_id']} in {split_name} file has split='{r['split']}'")

        # Verify that split rows exactly match geometries.csv projection
        geom_split_subset = [r for r in geoms_rows if r["split"] == split_name]
        if sp_rows != geom_split_subset:
            errors.append(f"{split_name} split file content does not match exact filtered projection of geometries.csv")

    # Disjointness and completeness
    train_ids = split_id_sets["TRAIN"]
    val_ids = split_id_sets["VALIDATION"]
    test_ids = split_id_sets["TEST"]

    if train_ids & val_ids:
        errors.append(f"TRAIN and VALIDATION overlap by {len(train_ids & val_ids)} geometries!")
    if train_ids & test_ids:
        errors.append(f"TRAIN and TEST overlap by {len(train_ids & test_ids)} geometries!")
    if val_ids & test_ids:
        errors.append(f"VALIDATION and TEST overlap by {len(val_ids & test_ids)} geometries!")

    union_ids = train_ids | val_ids | test_ids
    if union_ids != set(geom_ids):
        errors.append(f"Splits union does not match all 240 geometries: union={len(union_ids)}, all={len(geom_ids)}")
    else:
        print("  [OK] Strict split partitioning verified: 180 TRAIN / 48 VALIDATION / 12 TEST disjoint, union = 240.")

    # 4. Evaluation Cases Verification
    print("\n--- 4. Evaluation Cases (evaluation_cases/) ---")
    val_cases_path = dataset_dir / "evaluation_cases" / "validation_cases.csv"
    test_cases_path = dataset_dir / "evaluation_cases" / "test_cases.csv"

    with open(val_cases_path, newline="", encoding="utf-8") as f:
        val_cases = list(csv.DictReader(f))
    with open(test_cases_path, newline="", encoding="utf-8") as f:
        test_cases = list(csv.DictReader(f))

    if len(val_cases) != 96:
        errors.append(f"validation_cases.csv has {len(val_cases)} cases, expected 96.")
    else:
        print("  [OK] Exactly 96 validation cases verified.")

    if len(test_cases) != 60:
        errors.append(f"test_cases.csv has {len(test_cases)} cases, expected 60.")
    else:
        print("  [OK] Exactly 60 test cases verified.")

    # Ensure validation cases only reference VALIDATION geometries
    val_geom_keys = {(r["tier"], r["sequence"], int(r["geometry_generation_seed"])) for r in geoms_rows if r["split"] == "VALIDATION"}
    for vc in val_cases:
        key = (vc["tier"], vc["sequence"], int(vc["geometry_generation_seed"]))
        if key not in val_geom_keys:
            errors.append(f"Validation case {vc['case_id']} references non-VALIDATION geometry: {key}")

    # Ensure test cases only reference TEST geometries
    test_geom_keys = {(r["tier"], r["sequence"], int(r["geometry_generation_seed"])) for r in geoms_rows if r["split"] == "TEST"}
    for tc in test_cases:
        key = (tc["tier"], tc["sequence"], int(tc["geometry_generation_seed"]))
        if key not in test_geom_keys:
            errors.append(f"Test case {tc['case_id']} references non-TEST geometry: {key}")
    print("  [OK] Evaluation cases reference strictly their designated split geometries.")

    # 5. Source Artifact Provenance & Dataset Manifest
    print("\n--- 5. Source Artifact Provenance ---")
    manifest_path = dataset_dir / "dataset_manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest_data = json.load(f)

    if manifest_data.get("dataset_id") != "mapsuite_v1":
        errors.append(f"Manifest dataset_id mismatch: {manifest_data.get('dataset_id')}")

    source_artifacts = manifest_data.get("source_artifacts", [])
    if len(source_artifacts) != 5:
        errors.append(f"Expected 5 source artifact entries in manifest, found {len(source_artifacts)}")

    for sa in source_artifacts:
        rel_p = sa.get("repo_relative_path", "")
        expected_h = sa.get("sha256", "")
        actual_fp = project_root / rel_p
        if not actual_fp.is_file():
            errors.append(f"Source artifact declared in manifest not found on disk: {rel_p}")
        else:
            actual_h = compute_file_sha256(actual_fp)
            if actual_h != expected_h:
                errors.append(f"Source artifact hash mismatch for {rel_p}: recorded={expected_h} vs disk={actual_h}")
            else:
                print(f"  [OK] Source artifact verified: {rel_p}")

    # 6. Simulator Version & Commit Verification
    print("\n--- 6. Simulator Metadata ---")
    sim_info = manifest_data.get("simulator", {})
    if sim_info.get("version") != EXPECTED_METADRIVE_VERSION:
        errors.append(f"Simulator version mismatch in manifest: {sim_info.get('version')} vs {EXPECTED_METADRIVE_VERSION}")
    if sim_info.get("commit") != EXPECTED_METADRIVE_COMMIT:
        errors.append(f"Simulator commit mismatch in manifest: {sim_info.get('commit')} vs {EXPECTED_METADRIVE_COMMIT}")

    gen_config_path = dataset_dir / "generation_config.json"
    with open(gen_config_path, encoding="utf-8") as f:
        gen_config = json.load(f)
    gsim = gen_config.get("simulator", {})
    if gsim.get("version") != EXPECTED_METADRIVE_VERSION:
        errors.append(f"Simulator version mismatch in generation_config: {gsim.get('version')}")
    if gsim.get("pinned_commit") != EXPECTED_METADRIVE_COMMIT:
        errors.append(f"Simulator commit mismatch in generation_config: {gsim.get('pinned_commit')}")
    print("  [OK] MetaDrive 0.4.3 / commit 85e5dadc verified across manifest and generation config.")

    # 7. Package Checksums (CHECKSUMS.sha256)
    print("\n--- 7. Package Checksums (CHECKSUMS.sha256) ---")
    checksums_path = dataset_dir / "CHECKSUMS.sha256"
    recorded_checksums: Dict[str, str] = {}
    with open(checksums_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                file_hash, rel_path = parts[0], parts[1].strip()
                recorded_checksums[rel_path] = file_hash

    # Check that all actual files in dataset_dir match recorded checksums
    actual_files: Set[str] = set()
    for item in dataset_dir.rglob("*"):
        if item.is_file() and item.name != "CHECKSUMS.sha256":
            rel_p = item.relative_to(dataset_dir).as_posix()
            actual_files.add(rel_p)
            if rel_p not in recorded_checksums:
                errors.append(f"File {rel_p} is present on disk but missing from CHECKSUMS.sha256")
            else:
                curr_h = compute_file_sha256(item)
                rec_h = recorded_checksums[rel_p]
                if curr_h != rec_h:
                    errors.append(f"Checksum mismatch for {rel_p}: recorded={rec_h} vs computed={curr_h}")

    for rec_p in recorded_checksums:
        if rec_p not in actual_files:
            errors.append(f"File {rec_p} recorded in CHECKSUMS.sha256 is missing from disk")

    if not any(f"Checksum mismatch" in e or "missing" in e for e in errors):
        print(f"  [OK] All {len(recorded_checksums)} package checksums strictly verified.")

    # Final verdict
    print("\n============================================================")
    if errors:
        print(f"VERIFICATION FAILED WITH {len(errors)} ERROR(S):")
        for e in errors:
            print(f"  [ERROR] {e}")
        print("============================================================")
        return False

    print("MAPSUITE V1 DATASET VERIFICATION SUCCESSFUL (100% PASSED)!")
    print("============================================================")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify MetaDrive MapSuite V1 Dataset Package")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Path to datasets/mapsuite_v1 directory",
    )
    args = parser.parse_args()

    project_root = find_project_root()
    dataset_dir = args.dataset_dir if args.dataset_dir is not None else project_root / "datasets" / "mapsuite_v1"
    dataset_dir = dataset_dir.resolve()

    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    success = verify_dataset(dataset_dir, project_root)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
