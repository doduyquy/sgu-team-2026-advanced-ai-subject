"""Fixed, TRAIN-only internal split and diagnostic schedule for S5.3A."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from src.launcher.cases import LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256, load_manifest_csv


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_pilot_split(project_root: Path, split_seed: int = 5301) -> Dict[str, Any]:
    """Select 39 train and 6 diagnostic geometries per tier from verified TRAIN only."""
    manifest_path = project_root / "results/audits/evaluation_protocol/geometry_split_manifest.csv"
    rows = load_manifest_csv(manifest_path, expected_sha256=LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256)
    train = [row for row in rows if row["split"].upper() == "TRAIN"]
    if len(train) != 180:
        raise ValueError(f"Expected 180 TRAIN geometries, got {len(train)}")
    if any(len(row["geometry_sha256"]) != 64 for row in train):
        raise ValueError("TRAIN manifest contains malformed geometry SHA-256")

    rng = random.Random(int(split_seed))
    diagnostic: List[Dict[str, str]] = []
    tiers = sorted({row["tier"] for row in train}, key=str.lower)
    for tier in tiers:
        families: Dict[str, List[Dict[str, str]]] = {}
        for row in train:
            if row["tier"] == tier:
                families.setdefault(row["sequence"], []).append(row)
        for candidates in families.values():
            candidates.sort(key=lambda row: row["geometry_id"])
            rng.shuffle(candidates)
        family_order = list(families)
        rng.shuffle(family_order)
        while len([row for row in diagnostic if row["tier"] == tier]) < 6:
            progressed = False
            for family in family_order:
                if families[family]:
                    diagnostic.append(families[family].pop())
                    progressed = True
                    if len([row for row in diagnostic if row["tier"] == tier]) == 6:
                        break
            if not progressed:
                raise ValueError(f"Tier {tier} contains fewer than six geometries")

    diag_ids = {row["geometry_id"] for row in diagnostic}
    def membership(row: Dict[str, str]) -> Dict[str, Any]:
        return {
            "geometry_id": row["geometry_id"], "tier": row["tier"], "sequence": row["sequence"],
            "geometry_generation_seed": int(row["geometry_generation_seed"]),
            "geometry_sha256": row["geometry_sha256"],
        }
    result: Dict[str, Any] = {
        "protocol": "stage5_pilot_split_v1", "split_seed": int(split_seed),
        "source_manifest": "results/audits/evaluation_protocol/geometry_split_manifest.csv",
        "source_manifest_sha256": LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256,
        "pilot_train": [membership(row) for row in sorted(train, key=lambda x: x["geometry_id"]) if row["geometry_id"] not in diag_ids],
        "pilot_diagnostic": [membership(row) for row in sorted(diagnostic, key=lambda x: x["geometry_id"])],
    }
    if len(result["pilot_train"]) != 156 or len(result["pilot_diagnostic"]) != 24:
        raise AssertionError("Internal split cardinality mismatch")
    for tier in tiers:
        if sum(row["tier"] == tier for row in result["pilot_train"]) != 39:
            raise AssertionError(f"Expected 39 PILOT_TRAIN geometries for {tier}")
        if sum(row["tier"] == tier for row in result["pilot_diagnostic"]) != 6:
            raise AssertionError(f"Expected six PILOT_DIAGNOSTIC geometries for {tier}")
    result["split_sha256"] = canonical_sha256(result)
    return result


def save_pilot_split(project_root: Path, split_seed: int = 5301) -> Path:
    destination = project_root / "results/stage5/pilot_split_v1.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = build_pilot_split(project_root, split_seed)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def load_pilot_split(path: Path) -> Dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    declared = result.get("split_sha256")
    payload = dict(result)
    payload.pop("split_sha256", None)
    actual = canonical_sha256(payload)
    if declared != actual:
        raise ValueError(f"Pilot split SHA-256 mismatch: declared={declared}, actual={actual}")
    train = result.get("pilot_train", [])
    diagnostic = result.get("pilot_diagnostic", [])
    train_ids = {item["geometry_id"] for item in train}
    diagnostic_ids = {item["geometry_id"] for item in diagnostic}
    if len(train) != 156 or len(diagnostic) != 24 or train_ids & diagnostic_ids:
        raise ValueError("Pilot split membership is incomplete or overlapping")
    return result


def diagnostic_schedule(split: Dict[str, Any], schedule_seed: int = 5302) -> List[Dict[str, Any]]:
    """Return one fixed case per diagnostic geometry with distinct environment seed channel."""
    schedule = []
    for item in split["pilot_diagnostic"]:
        digest = hashlib.sha256(f"stage5-diagnostic:{schedule_seed}:{item['geometry_id']}".encode()).hexdigest()
        schedule.append({**item, "environment_seed": int(digest[:8], 16) % 1_000_000})
    return schedule
