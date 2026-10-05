#!/usr/bin/env python3
"""
MetaDrive MapSuite V1 Dataset Exporter.

Deterministic, offline exporter that derives the versioned MapSuite V1
dataset package from existing, frozen Platform V1 benchmark artifacts.

Source artifacts:
- results/audits/mapsuite/candidate_metrics.csv
- results/audits/mapsuite/canonical_candidates.json
- results/audits/mapsuite/previews/ (12 PNG files)
- results/audits/evaluation_protocol/geometry_split_manifest.csv
- results/audits/evaluation_protocol/validation_case_manifest.csv
- results/audits/evaluation_protocol/test_case_manifest.csv

This exporter strictly adheres to the Source-of-Truth rule:
it NEVER invents new geometries, splits, seeds, or evaluation cases.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Tuple

# Pinned metadata matching Platform V1 contracts
DATASET_ID = "mapsuite_v1"
DATASET_NAME = "MetaDrive MapSuite V1"
DATASET_VERSION = "1.0.0"
DATASET_TYPE = "procedural_scenario_benchmark"
SCIENTIFIC_STATUS = "DERIVED_FROM_FROZEN_PLATFORM_V1_BENCHMARK"

SIMULATOR_NAME = "MetaDrive"
PINNED_METADRIVE_VERSION = "0.4.3"
PINNED_METADRIVE_COMMIT = "85e5dadc6c7436d324348f6e3d8f8e680c06b4db"

TIER_ORDER = ["Easy", "Medium", "Hard", "Extreme"]

SEQUENCE_ORDER = {
    "Easy": ["SCS", "SCSS", "SCCS"],
    "Medium": ["SCXCS", "SCTCS", "SCXCCS"],
    "Hard": ["SCXOCS", "SCTXrCS", "XTOCS"],
    "Extreme": ["CrXROSTR", "SCXOCrTYCS", "SCTXORyCCS"],
}

TIER_RANK_MAP = {
    ("Easy", "SCS"): 1,
    ("Easy", "SCSS"): 2,
    ("Easy", "SCCS"): 3,
    ("Medium", "SCXCS"): 1,
    ("Medium", "SCTCS"): 2,
    ("Medium", "SCXCCS"): 3,
    ("Hard", "SCXOCS"): 1,
    ("Hard", "SCTXrCS"): 2,
    ("Hard", "XTOCS"): 3,
    ("Extreme", "CrXROSTR"): 1,
    ("Extreme", "SCXOCrTYCS"): 2,
    ("Extreme", "SCTXORyCCS"): 3,
}

GEOMETRY_COLUMNS = [
    "geometry_id",
    "split",
    "tier",
    "sequence",
    "geometry_generation_seed",
    "candidate_role",
    "geometry_sha256",
    "geometry_hash_source",
    "generation_success",
    "block_ids",
    "block_count",
    "route_length_m",
    "bbox_width_m",
    "bbox_height_m",
    "lane_width_m",
    "base_lane_num",
    "straight_count",
    "curve_count",
    "intersection_count",
    "t_intersection_count",
    "roundabout_count",
    "ramp_count",
    "merge_split_count",
    "decision_block_count",
    "branching_choice_score",
    "traffic_density",
    "planned_traffic_vehicle_count",
    "episode_budget_seconds",
    "required_avg_speed_kmh_for_horizon_1000",
]

SCENARIO_FAMILY_COLUMNS = [
    "tier",
    "family_rank_within_tier",
    "sequence",
    "geometry_seed_count",
    "geometry_seed_min",
    "geometry_seed_max",
    "traffic_density",
    "representative_geometry_seed",
    "representative_preview",
]


def find_project_root() -> Path:
    """Finds the autonomous-driving-rl project root."""
    curr = Path(__file__).resolve().parent
    for p in [curr, curr.parent, curr.parent.parent]:
        if (p / "results" / "audits" / "mapsuite").exists():
            return p
    # Fallback to current working directory
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


def load_source_artifacts(project_root: Path) -> Dict[str, Any]:
    """Loads and validates all frozen source artifacts."""
    artifacts = {}

    cm_path = project_root / "results" / "audits" / "mapsuite" / "candidate_metrics.csv"
    cc_path = project_root / "results" / "audits" / "mapsuite" / "canonical_candidates.json"
    sm_path = project_root / "results" / "audits" / "evaluation_protocol" / "geometry_split_manifest.csv"
    vm_path = project_root / "results" / "audits" / "evaluation_protocol" / "validation_case_manifest.csv"
    tm_path = project_root / "results" / "audits" / "evaluation_protocol" / "test_case_manifest.csv"
    prev_dir = project_root / "results" / "audits" / "mapsuite" / "previews"

    for p in [cm_path, cc_path, sm_path, vm_path, tm_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing required source artifact: {p}")
    if not prev_dir.is_dir():
        raise FileNotFoundError(f"Missing required previews directory: {prev_dir}")

    # Read candidate_metrics.csv
    with open(cm_path, newline="", encoding="utf-8") as f:
        cm_rows = list(csv.DictReader(f))
    if len(cm_rows) != 240:
        raise ValueError(f"Expected 240 rows in candidate_metrics.csv, found {len(cm_rows)}")

    # Read geometry_split_manifest.csv
    with open(sm_path, newline="", encoding="utf-8") as f:
        sm_rows = list(csv.DictReader(f))
    if len(sm_rows) != 240:
        raise ValueError(f"Expected 240 rows in geometry_split_manifest.csv, found {len(sm_rows)}")

    # Read canonical_candidates.json
    with open(cc_path, encoding="utf-8") as f:
        canonical_candidates = json.load(f)

    # Read validation_case_manifest.csv
    with open(vm_path, newline="", encoding="utf-8") as f:
        vm_reader = csv.DictReader(f)
        vm_header = vm_reader.fieldnames or []
        vm_rows = list(vm_reader)
    if len(vm_rows) != 96:
        raise ValueError(f"Expected 96 rows in validation_case_manifest.csv, found {len(vm_rows)}")

    # Read test_case_manifest.csv
    with open(tm_path, newline="", encoding="utf-8") as f:
        tm_reader = csv.DictReader(f)
        tm_header = tm_reader.fieldnames or []
        tm_rows = list(tm_reader)
    if len(tm_rows) != 60:
        raise ValueError(f"Expected 60 rows in test_case_manifest.csv, found {len(tm_rows)}")

    # Read previews
    previews = sorted([p for p in prev_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
    if len(previews) != 12:
        raise ValueError(f"Expected exactly 12 representative preview images, found {len(previews)}")

    artifacts["cm_rows"] = cm_rows
    artifacts["sm_rows"] = sm_rows
    artifacts["canonical_candidates"] = canonical_candidates
    artifacts["vm_rows"] = vm_rows
    artifacts["vm_header"] = vm_header
    artifacts["tm_rows"] = tm_rows
    artifacts["tm_header"] = tm_header
    artifacts["previews"] = previews

    # Source artifact hashes
    source_manifest_entries = [
        {"repo_relative_path": "results/audits/mapsuite/candidate_metrics.csv", "sha256": compute_file_sha256(cm_path)},
        {"repo_relative_path": "results/audits/mapsuite/canonical_candidates.json", "sha256": compute_file_sha256(cc_path)},
        {"repo_relative_path": "results/audits/evaluation_protocol/geometry_split_manifest.csv", "sha256": compute_file_sha256(sm_path)},
        {"repo_relative_path": "results/audits/evaluation_protocol/validation_case_manifest.csv", "sha256": compute_file_sha256(vm_path)},
        {"repo_relative_path": "results/audits/evaluation_protocol/test_case_manifest.csv", "sha256": compute_file_sha256(tm_path)},
    ]
    source_preview_entries = [
        {"repo_relative_path": f"results/audits/mapsuite/previews/{p.name}", "sha256": compute_file_sha256(p)}
        for p in previews
    ]
    artifacts["source_manifest_entries"] = source_manifest_entries
    artifacts["source_preview_entries"] = source_preview_entries

    return artifacts


def build_geometries(cm_rows: List[Dict[str, str]], sm_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Joins candidate_metrics.csv and geometry_split_manifest.csv deterministically."""
    cm_map: Dict[Tuple[str, str, int], Dict[str, str]] = {}
    for r in cm_rows:
        key = (r["difficulty_tier"], r["sequence"], int(r["scenario_seed"]))
        if key in cm_map:
            raise ValueError(f"Duplicate key in candidate_metrics: {key}")
        cm_map[key] = r

    sm_map: Dict[Tuple[str, str, int], Dict[str, str]] = {}
    for r in sm_rows:
        key = (r["tier"], r["sequence"], int(r["geometry_generation_seed"]))
        if key in sm_map:
            raise ValueError(f"Duplicate key in geometry_split_manifest: {key}")
        sm_map[key] = r

    if set(cm_map.keys()) != set(sm_map.keys()):
        diff = set(cm_map.keys()) ^ set(sm_map.keys())
        raise ValueError(f"Key mismatch between candidate_metrics and geometry_split_manifest: {diff}")

    joined_geometries: List[Dict[str, Any]] = []

    for key, sm_r in sm_map.items():
        cm_r = cm_map[key]
        tier, seq, seed = key

        # Strict cross-source consistency checks
        if sm_r["block_ids"] != cm_r["block_ids"]:
            raise ValueError(f"Inconsistent block_ids for {key}: sm={sm_r['block_ids']} vs cm={cm_r['block_ids']}")
        if float(sm_r["traffic_density"]) != float(cm_r["traffic_density"]):
            raise ValueError(f"Inconsistent traffic_density for {key}: sm={sm_r['traffic_density']} vs cm={cm_r['traffic_density']}")
        sm_len = float(sm_r["route_length_m"])
        cm_len = float(cm_r["route_total_length_m"])
        if abs(sm_len - cm_len) > 0.05:
            raise ValueError(f"Inconsistent route length for {key}: sm={sm_len} vs cm={cm_len}")

        row = {
            "geometry_id": sm_r["geometry_id"],
            "split": sm_r["split"],
            "tier": sm_r["tier"],
            "sequence": sm_r["sequence"],
            "geometry_generation_seed": int(sm_r["geometry_generation_seed"]),
            "candidate_role": sm_r["candidate_role"],
            "geometry_sha256": sm_r["geometry_sha256"],
            "geometry_hash_source": sm_r["geometry_hash_source"],
            "generation_success": cm_r["generation_success"].strip().lower() in ("true", "1"),
            "block_ids": sm_r["block_ids"],
            "block_count": int(cm_r["block_count"]),
            "route_length_m": float(sm_r["route_length_m"]),
            "bbox_width_m": float(cm_r["bbox_width_m"]),
            "bbox_height_m": float(cm_r["bbox_height_m"]),
            "lane_width_m": float(cm_r["lane_width"]),
            "base_lane_num": int(cm_r["base_lane_num"]),
            "straight_count": int(cm_r["straight_count"]),
            "curve_count": int(cm_r["curve_count"]),
            "intersection_count": int(cm_r["intersection_count"]),
            "t_intersection_count": int(cm_r["t_intersection_count"]),
            "roundabout_count": int(cm_r["roundabout_count"]),
            "ramp_count": int(cm_r["ramp_count"]),
            "merge_split_count": int(cm_r["merge_split_count"]),
            "decision_block_count": int(cm_r["decision_block_count"]),
            "branching_choice_score": int(cm_r["branching_choice_score"]),
            "traffic_density": float(sm_r["traffic_density"]),
            "planned_traffic_vehicle_count": int(cm_r["planned_traffic_vehicle_count"]),
            "episode_budget_seconds": float(cm_r["episode_budget_seconds"]),
            "required_avg_speed_kmh_for_horizon_1000": float(cm_r["required_avg_speed_kmh_for_horizon_1000"]),
        }
        joined_geometries.append(row)

    # Sort deterministically: Tier order, Sequence order, Seed ascending
    def sort_key(g: Dict[str, Any]) -> Tuple[int, int, int]:
        t_idx = TIER_ORDER.index(g["tier"])
        s_idx = SEQUENCE_ORDER[g["tier"]].index(g["sequence"])
        return (t_idx, s_idx, g["geometry_generation_seed"])

    joined_geometries.sort(key=sort_key)
    return joined_geometries


def build_scenario_families(canonical_candidates: Dict[str, List[Dict[str, Any]]], geometries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Builds the 12 scenario families table from canonical candidates and geometries."""
    families: List[Dict[str, Any]] = []

    # Map candidate selections
    cand_by_seq: Dict[str, Dict[str, Any]] = {}
    for tier, clist in canonical_candidates.items():
        for c in clist:
            cand_by_seq[c["sequence"]] = c

    for tier in TIER_ORDER:
        for seq in SEQUENCE_ORDER[tier]:
            rank = TIER_RANK_MAP[(tier, seq)]
            cand = cand_by_seq[seq]

            # Collect geometry seeds for this family
            family_geoms = [g for g in geometries if g["tier"] == tier and g["sequence"] == seq]
            if len(family_geoms) != 20:
                raise ValueError(f"Family {tier}/{seq} has {len(family_geoms)} geometries; expected 20.")

            seeds = [g["geometry_generation_seed"] for g in family_geoms]
            min_seed = min(seeds)
            max_seed = max(seeds)
            traffic_density = family_geoms[0]["traffic_density"]

            preview_file = Path(cand["preview_image"]).name
            row = {
                "tier": tier,
                "family_rank_within_tier": rank,
                "sequence": seq,
                "geometry_seed_count": len(family_geoms),
                "geometry_seed_min": min_seed,
                "geometry_seed_max": max_seed,
                "traffic_density": traffic_density,
                "representative_geometry_seed": cand["scenario_seed"],
                "representative_preview": f"previews/{preview_file}",
            }
            families.append(row)

    return families


def build_generation_config() -> Dict[str, Any]:
    """Constructs generation_config.json containing verified procedural generation parameters."""
    families_meta = []
    for tier in TIER_ORDER:
        for seq in SEQUENCE_ORDER[tier]:
            rank = TIER_RANK_MAP[(tier, seq)]
            families_meta.append({
                "tier": tier,
                "sequence": seq,
                "family_rank_within_tier": rank,
                "block_count": len(seq) + 1,  # +1 for FirstPGBlock 'I'
                "first_block_id": "I",
                "sequence_block_ids": seq,
            })

    return {
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "simulator": {
            "name": SIMULATOR_NAME,
            "version": PINNED_METADRIVE_VERSION,
            "pinned_commit": PINNED_METADRIVE_COMMIT,
            "upstream_repository": "https://github.com/metadriverse/metadrive",
        },
        "generation_method": "MetaDrive BIG (Block-Intersection Generator)",
        "generator_source_references": [
            "metadrive/component/map/pg_map.py",
            "metadrive/component/algorithm/BIG.py",
            "metadrive/component/algorithm/blocks_prob_dist.py",
            "metadrive/component/pg_space.py",
        ],
        "lane_configuration": {
            "lane_width_m": 3.5,
            "base_lane_num": 2,
            "random_lane_width": False,
            "random_lane_num": False,
        },
        "block_parameter_bounds": {
            "straight_length_range_m": [40.0, 80.0],
            "curve_radius_range_m": [25.0, 60.0],
            "curve_angle_range_deg": [45.0, 135.0],
            "max_block_generation_trials": 5,
        },
        "seeds": {
            "geometry_seed_min": 0,
            "geometry_seed_max": 19,
            "geometry_seeds_per_family": 20,
            "total_geometries": 240,
        },
        "traffic_density_policy": {
            "Easy": 0.0,
            "Medium": 0.08,
            "Hard": 0.15,
            "Extreme": 0.25,
            "traffic_mode": "Trigger",
        },
        "scenario_families": families_meta,
    }


def build_dataset_manifest(
    source_manifest_entries: List[Dict[str, str]],
    source_preview_entries: List[Dict[str, str]],
    geometries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Constructs dataset_manifest.json with counts and source provenance."""
    train_count = sum(1 for g in geometries if g["split"] == "TRAIN")
    val_count = sum(1 for g in geometries if g["split"] == "VALIDATION")
    test_count = sum(1 for g in geometries if g["split"] == "TEST")

    return {
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "dataset_type": DATASET_TYPE,
        "scientific_status": SCIENTIFIC_STATUS,
        "description": "Fixed, reproducible 240-geometry procedural benchmark dataset for autonomous driving research on MetaDrive 0.4.3.",
        "counts": {
            "scenario_families": 12,
            "geometries": len(geometries),
            "train_geometries": train_count,
            "validation_geometries": val_count,
            "test_geometries": test_count,
            "validation_cases": 96,
            "test_cases": 60,
            "representative_previews": 12,
        },
        "split_summary": {
            "TRAIN": train_count,
            "VALIDATION": val_count,
            "TEST": test_count,
        },
        "simulator": {
            "name": SIMULATOR_NAME,
            "version": PINNED_METADRIVE_VERSION,
            "commit": PINNED_METADRIVE_COMMIT,
        },
        "source_artifacts": source_manifest_entries,
        "source_previews": source_preview_entries,
    }


def write_csv(path: Path, columns: List[str], rows: List[Dict[str, Any]]) -> None:
    """Writes a list of dicts to CSV deterministically with UTF-8 and LF newlines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in columns})


def export_dataset(project_root: Path, output_dir: Path) -> None:
    """Main export routine."""
    print("============================================================")
    print("EXPORTING METADRIVE MAPSUITE V1 DATASET PACKAGE")
    print("============================================================")
    print(f"Project root: {project_root}")
    print(f"Output directory: {output_dir}")

    # 1. Load and validate source artifacts
    artifacts = load_source_artifacts(project_root)
    print("  [OK] Source artifacts loaded and verified.")

    # 2. Build joined geometries
    geometries = build_geometries(artifacts["cm_rows"], artifacts["sm_rows"])
    if len(geometries) != 240:
        raise ValueError(f"Expected 240 joined geometries, got {len(geometries)}")
    print("  [OK] 240 geometries joined and sorted.")

    # 3. Build scenario families
    families = build_scenario_families(artifacts["canonical_candidates"], geometries)
    if len(families) != 12:
        raise ValueError(f"Expected 12 scenario families, got {len(families)}")
    print("  [OK] 12 scenario families constructed.")

    # 4. Filter splits
    train_geoms = [g for g in geometries if g["split"] == "TRAIN"]
    val_geoms = [g for g in geometries if g["split"] == "VALIDATION"]
    test_geoms = [g for g in geometries if g["split"] == "TEST"]

    if len(train_geoms) != 180 or len(val_geoms) != 48 or len(test_geoms) != 12:
        raise ValueError(f"Invalid split counts: TRAIN={len(train_geoms)}, VAL={len(val_geoms)}, TEST={len(test_geoms)}")
    print("  [OK] Split partitions verified: 180 TRAIN / 48 VALIDATION / 12 TEST.")

    # 5. Build generation config & dataset manifest
    gen_config = build_generation_config()
    manifest = build_dataset_manifest(
        artifacts["source_manifest_entries"],
        artifacts["source_preview_entries"],
        geometries,
    )

    # 6. Ensure target directory structure
    splits_dir = output_dir / "splits"
    eval_cases_dir = output_dir / "evaluation_cases"
    previews_dir = output_dir / "previews"

    splits_dir.mkdir(parents=True, exist_ok=True)
    eval_cases_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    # 7. Write dataset files
    write_csv(output_dir / "geometries.csv", GEOMETRY_COLUMNS, geometries)
    write_csv(output_dir / "scenario_families.csv", SCENARIO_FAMILY_COLUMNS, families)
    write_csv(splits_dir / "train_geometries.csv", GEOMETRY_COLUMNS, train_geoms)
    write_csv(splits_dir / "validation_geometries.csv", GEOMETRY_COLUMNS, val_geoms)
    write_csv(splits_dir / "test_geometries.csv", GEOMETRY_COLUMNS, test_geoms)

    # Write evaluation cases directly from source manifests (preserving order and columns)
    write_csv(eval_cases_dir / "validation_cases.csv", artifacts["vm_header"], artifacts["vm_rows"])
    write_csv(eval_cases_dir / "test_cases.csv", artifacts["tm_header"], artifacts["tm_rows"])

    # Write JSON configs with stable indentation and LF newlines
    with open(output_dir / "generation_config.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(gen_config, f, indent=2)
        f.write("\n")

    with open(output_dir / "dataset_manifest.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    # 8. Copy representative previews
    for prev_path in artifacts["previews"]:
        dest = previews_dir / prev_path.name
        shutil.copy2(prev_path, dest)
    print(f"  [OK] Copied {len(artifacts['previews'])} representative preview images.")

    # 9. Compute CHECKSUMS.sha256 over all committed files in mapsuite_v1 (excluding CHECKSUMS.sha256 itself)
    checksums: List[Tuple[str, str]] = []
    for item in sorted(output_dir.rglob("*")):
        if item.is_file() and item.name != "CHECKSUMS.sha256":
            rel = item.relative_to(output_dir).as_posix()
            file_hash = compute_file_sha256(item)
            checksums.append((file_hash, rel))

    checksums.sort(key=lambda x: x[1])
    with open(output_dir / "CHECKSUMS.sha256", "w", encoding="utf-8", newline="\n") as f:
        for file_hash, rel in checksums:
            f.write(f"{file_hash}  {rel}\n")

    print(f"  [SAVED] CHECKSUMS.sha256 ({len(checksums)} entries).")
    print("============================================================")
    print("MAPSUITE V1 DATASET EXPORT COMPLETE!")
    print("============================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MetaDrive MapSuite V1 Dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Target output directory (default: <project_root>/datasets/mapsuite_v1)",
    )
    args = parser.parse_args()

    project_root = find_project_root()
    output_dir = args.output if args.output is not None else project_root / "datasets" / "mapsuite_v1"
    output_dir = output_dir.resolve()

    export_dataset(project_root, output_dir)


if __name__ == "__main__":
    main()
