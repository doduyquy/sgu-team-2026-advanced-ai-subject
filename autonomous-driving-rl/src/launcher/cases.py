"""
Case Resolution and Exact Geometry Verification for Research Platform V1 Launcher (Gate 7.5A).

This module implements:
- Authoritative loading from Gate-5 locked CSV manifests:
  - geometry_split_manifest.csv (240 geometries universe)
  - validation_case_manifest.csv (96 validation cases)
  - test_case_manifest.csv (60 paired benchmark test cases)
- Strict holdout isolation: Sandbox is restricted strictly to TRAIN geometries.
- Exact geometry loading and SHA-256 verification via MetaDrive headless generation.
- Full manifest parity with zero reimplementation drift.
"""

import copy
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.envs.metadrive_env import MetaDriveEnv

from src.launcher.models import LaunchRequestV1, ResolvedCaseV1
from src.platform import (
    canonical_json_sha256,
    compute_route_aware_horizon,
)


def make_jsonable(obj: Any) -> Any:
    """Recursively converts numpy structures and primitives to JSON-serializable types."""
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if hasattr(obj, "tolist"):
        return make_jsonable(obj.tolist())
    if hasattr(obj, "item"):
        return make_jsonable(obj.item())
    if isinstance(obj, dict):
        return {str(k): make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_jsonable(v) for v in obj]
    return obj


def load_manifest_csv(file_path: Path) -> List[Dict[str, str]]:
    """Loads CSV manifest into list of dictionary records."""
    if not file_path.exists():
        raise FileNotFoundError(f"Required Gate-5 manifest file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_default_project_root() -> Path:
    """Resolves autonomous-driving-rl project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def load_locked_geometry_block_sequence(
    record: Dict[str, Any],
    project_root: Optional[Path] = None
) -> Tuple[List[Any], str, bool]:
    """
    Regenerates and verifies exact locked road geometry block sequence from Gate-5 record.
    Asserts bit-for-bit geometry_sha256 equality with Gate-5 manifest before returning blocks.
    Fails loudly if any geometry drift is detected.
    """
    seq = record["sequence"]
    geom_seed = int(record["geometry_generation_seed"])
    expected_hash = record["geometry_sha256"]

    # Regenerate geometry under clean, zero-traffic headless environment
    env_gen = MetaDriveEnv(dict(
        use_render=False,
        num_scenarios=1,
        start_seed=geom_seed,
        map=seq,
        traffic_density=0.0
    ))
    env_gen.reset(seed=geom_seed)
    raw_blocks = env_gen.current_map.get_meta_data()["block_sequence"]
    clean_blocks = make_jsonable(raw_blocks)
    env_gen.close()

    regen_hash = canonical_json_sha256(clean_blocks)
    if regen_hash != expected_hash:
        raise ValueError(
            f"FATAL GEOMETRY DRIFT: Geometry hash mismatch for sequence '{seq}' seed {geom_seed}! "
            f"Regenerated={regen_hash} != Manifest={expected_hash}"
        )
    return clean_blocks, regen_hash, True


def resolve_sandbox_case(
    request: LaunchRequestV1,
    project_root: Optional[Path] = None
) -> ResolvedCaseV1:
    """
    Resolves exactly ONE episode for Sandbox mode.
    Strictly restricted to TRAIN split geometries. Rejects TEST and VALIDATION geometries loudly.
    """
    root = project_root or get_default_project_root()
    geom_manifest_path = root / "results" / "audits" / "evaluation_protocol" / "geometry_split_manifest.csv"
    geom_records = load_manifest_csv(geom_manifest_path)

    # 1. If explicit geometry_generation_seed is provided, find exact record and verify TRAIN split
    if request.geometry_generation_seed is not None:
        target_seed = int(request.geometry_generation_seed)
        target_seq = request.sequence
        matched = []
        for r in geom_records:
            if int(r["geometry_generation_seed"]) == target_seed:
                if target_seq is None or r["sequence"] == target_seq:
                    matched.append(r)

        if not matched:
            raise ValueError(
                f"No geometry found in Gate-5 universe matching seed {target_seed}"
                f"{f' and sequence {target_seq}' if target_seq else ''}"
            )

        selected = matched[0]
        # Strict holdout check: reject VALIDATION and TEST geometries immediately
        actual_split = selected["split"].upper()
        if actual_split in ("VALIDATION", "TEST"):
            raise ValueError(
                f"FATAL HOLDOUT VIOLATION: Sandbox requested geometry {selected['geometry_id']} "
                f"which belongs to {actual_split} split! Sandbox mode is strictly restricted to TRAIN geometries."
            )
    else:
        # 2. Filter eligible TRAIN records
        train_records = [r for r in geom_records if r["split"].upper() == "TRAIN"]

        if request.tier is not None:
            train_records = [r for r in train_records if r["tier"].lower() == request.tier.lower()]
            if not train_records:
                raise ValueError(f"No TRAIN geometries available for tier '{request.tier}'")

        if request.sequence is not None:
            train_records = [r for r in train_records if r["sequence"] == request.sequence]
            if not train_records:
                raise ValueError(f"No TRAIN geometries available for sequence '{request.sequence}'")

        # Stable deterministic ordering: tier order (Easy, Medium, Hard, Extreme), sequence, geom_seed
        tier_weights = {"easy": 1, "medium": 2, "hard": 3, "extreme": 4}
        train_records.sort(
            key=lambda r: (
                tier_weights.get(r["tier"].lower(), 99),
                r["sequence"],
                int(r["geometry_generation_seed"])
            )
        )
        selected = train_records[0]

    # Resolve case properties
    tier = selected["tier"]
    sequence = selected["sequence"]
    geom_seed = int(selected["geometry_generation_seed"])
    geom_hash = selected["geometry_sha256"]
    env_seed = int(request.environment_seed) if request.environment_seed is not None else 0
    traffic_density = float(selected["traffic_density"])
    route_len = float(selected["route_length_m"])
    horizon_steps = compute_route_aware_horizon(route_len)

    case_id = f"sandbox/{tier}/{sequence}/{geom_seed}/{env_seed}"

    return ResolvedCaseV1(
        case_id=case_id,
        case_index=1,
        protocol_order_index=1,
        split="TRAIN",
        tier=tier,
        sequence=sequence,
        geometry_generation_seed=geom_seed,
        geometry_sha256=geom_hash,
        environment_seed=env_seed,
        traffic_density=traffic_density,
        horizon_steps=horizon_steps
    )


def resolve_validation_cases(project_root: Optional[Path] = None) -> List[ResolvedCaseV1]:
    """
    Resolves the complete, immutable Gate-5 validation suite (exactly 96 cases).
    Sources directly from validation_case_manifest.csv with zero deviation.
    """
    root = project_root or get_default_project_root()
    manifest_path = root / "results" / "audits" / "evaluation_protocol" / "validation_case_manifest.csv"
    records = load_manifest_csv(manifest_path)

    if len(records) != 96:
        raise ValueError(f"Expected exactly 96 validation cases, got {len(records)} in {manifest_path}")

    cases = []
    for r in records:
        cases.append(ResolvedCaseV1(
            case_id=r["case_id"],
            case_index=int(r["case_index"]),
            protocol_order_index=int(r["protocol_order_index"]),
            split="VALIDATION",
            tier=r["tier"],
            sequence=r["sequence"],
            geometry_generation_seed=int(r["geometry_generation_seed"]),
            geometry_sha256=r["geometry_sha256"],
            environment_seed=int(r["environment_seed"]),
            traffic_density=float(r["traffic_density"]),
            horizon_steps=int(r["horizon_steps"])
        ))
    return cases


def resolve_test_cases(project_root: Optional[Path] = None) -> List[ResolvedCaseV1]:
    """
    Resolves the complete, immutable Gate-5 test suite (exactly 60 cases).
    Sources directly from test_case_manifest.csv with zero deviation.
    """
    root = project_root or get_default_project_root()
    manifest_path = root / "results" / "audits" / "evaluation_protocol" / "test_case_manifest.csv"
    records = load_manifest_csv(manifest_path)

    if len(records) != 60:
        raise ValueError(f"Expected exactly 60 test cases, got {len(records)} in {manifest_path}")

    cases = []
    for r in records:
        cases.append(ResolvedCaseV1(
            case_id=r["case_id"],
            case_index=int(r["case_index"]),
            protocol_order_index=int(r["protocol_order_index"]),
            split="TEST",
            tier=r["tier"],
            sequence=r["sequence"],
            geometry_generation_seed=int(r["geometry_generation_seed"]),
            geometry_sha256=r["geometry_sha256"],
            environment_seed=int(r["environment_seed"]),
            traffic_density=float(r["traffic_density"]),
            horizon_steps=int(r["horizon_steps"])
        ))
    return cases


def resolve_audit_cases(
    request: LaunchRequestV1,
    project_root: Optional[Path] = None
) -> List[ResolvedCaseV1]:
    """
    Resolves case suite for platform audit execution.
    By default resolves 1 deterministic TRAIN or VALIDATION case. Rejects TEST cases.
    """
    root = project_root or get_default_project_root()
    geom_manifest_path = root / "results" / "audits" / "evaluation_protocol" / "geometry_split_manifest.csv"
    geom_records = load_manifest_csv(geom_manifest_path)

    # Filter to non-TEST geometries
    eligible = [r for r in geom_records if r["split"].upper() in ("TRAIN", "VALIDATION")]
    if request.tier:
        eligible = [r for r in eligible if r["tier"].lower() == request.tier.lower()]
    if request.sequence:
        eligible = [r for r in eligible if r["sequence"] == request.sequence]
    if request.geometry_generation_seed is not None:
        eligible = [r for r in eligible if int(r["geometry_generation_seed"]) == int(request.geometry_generation_seed)]

    if not eligible:
        raise ValueError("No eligible audit geometry found matching request")

    selected = eligible[0]
    tier = selected["tier"]
    sequence = selected["sequence"]
    geom_seed = int(selected["geometry_generation_seed"])
    geom_hash = selected["geometry_sha256"]
    env_seed = int(request.environment_seed) if request.environment_seed is not None else 9101
    traffic_density = float(selected["traffic_density"])
    route_len = float(selected["route_length_m"])
    horizon_steps = compute_route_aware_horizon(route_len)

    case_id = f"audit/{tier}/{sequence}/{geom_seed}/{env_seed}"

    return [ResolvedCaseV1(
        case_id=case_id,
        case_index=1,
        protocol_order_index=1,
        split=selected["split"].upper(),
        tier=tier,
        sequence=sequence,
        geometry_generation_seed=geom_seed,
        geometry_sha256=geom_hash,
        environment_seed=env_seed,
        traffic_density=traffic_density,
        horizon_steps=horizon_steps
    )]
