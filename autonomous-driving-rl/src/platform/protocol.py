"""
Platform V1 Scientific Evaluation Protocol, Scenario Splits, and Seed Taxonomy.
Gate 5 of Research Platform V1.

Pure, unit-testable module independent of Panda3D / simulator engine state.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Set, Tuple

from src.platform.metrics import AggregateMetrics


class SplitRole(str, Enum):
    """Dataset split roles for scenario geometries."""
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"


def canonical_json_sha256(data: Any) -> str:
    """Computes a deterministic SHA-256 fingerprint from a JSON-serializable object."""
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def derive_seed(
    root_seed: int,
    namespace: str,
    index: int,
    optional_geometry_id: Optional[str] = None
) -> int:
    """
    Derives a deterministic, 31-bit non-negative integer seed from root seed and namespace.
    Uses SHA-256 to ensure platform-independent stability across Python processes.
    """
    key = f"{root_seed}_{namespace}_{index}_{optional_geometry_id or ''}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed_int = int.from_bytes(digest[:4], byteorder="big") % (2**31 - 1)
    return seed_int


@dataclass(frozen=True)
class GeometryRecord:
    """Metadata record for a procedural road geometry in the 240-geometry universe."""
    geometry_id: str
    tier: str
    sequence: str
    geometry_generation_seed: int
    split: SplitRole
    candidate_role: str
    block_ids: str
    route_length_m: float
    geometry_sha256: str
    exact_block_sequence: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["split"] = self.split.value
        return data


@dataclass(frozen=True)
class EvaluationCase:
    """Specific evaluation scenario instance combining geometry, environment seed, and budget."""
    case_id: str
    case_index: int
    protocol_order_index: int
    split: SplitRole
    tier: str
    sequence: str
    candidate_role: str
    geometry_generation_seed: int
    geometry_sha256: str
    environment_seed: int
    traffic_density: float
    horizon_steps: int

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["split"] = self.split.value
        return data


@dataclass(frozen=True)
class SeedPlanV1:
    """
    Explicit seed taxonomy and declared seed pools for Platform V1.
    Never overloads 'seed' into a single ambiguous integer.
    """
    spec_version: str = "1.0.0"
    status: str = "LOCKED-FOR-PLATFORM-V1"
    geometry_split_salt: str = "platform-v1-geometry-split-v1"
    protocol_order_seed: int = 424242
    test_environment_seeds: List[int] = field(default_factory=lambda: [9101, 9102, 9103, 9104, 9105])
    validation_environment_seeds: List[int] = field(default_factory=lambda: [5101, 5102])
    agent_replicate_seeds: List[int] = field(default_factory=lambda: [101, 202, 303])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioSplitV1:
    """Specification of geometry partition counts across Train, Validation, and Test."""
    spec_version: str = "1.0.0"
    status: str = "LOCKED-FOR-PLATFORM-V1"
    total_geometries: int = 240
    train_count: int = 180
    validation_count: int = 48
    test_count: int = 12
    train_per_tier: int = 45
    validation_per_tier: int = 12
    test_per_tier: int = 3
    train_per_sequence: int = 15
    validation_per_sequence: int = 4
    test_per_sequence: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def assign_geometry_splits(
    candidates_metrics: List[Dict[str, Any]],
    canonical_candidates_manifest: Dict[str, List[Dict[str, Any]]],
    split_salt: str = "platform-v1-geometry-split-v1"
) -> List[GeometryRecord]:
    """
    Deterministic stratified split of the 240 candidate geometries:
    - 12 Gate-2 human-reviewed canonical geometries -> TEST (1 primary + 2 alternates per tier).
    - For each sequence, the remaining 19 seeds are sorted by SHA-256 hash using split_salt.
    - First 15 -> TRAIN (180 total, 45 per tier).
    - Final 4 -> VALIDATION (48 total, 12 per tier).
    """
    # 1. Identify canonical test geometries
    test_canonical_lookup = {}
    for tier, cands in canonical_candidates_manifest.items():
        for c in cands:
            key = (tier, c["sequence"], int(c["scenario_seed"]))
            test_canonical_lookup[key] = c

    # Group metrics by sequence
    seq_map = {}
    for m in candidates_metrics:
        tier = m["difficulty_tier"]
        seq = m["sequence"]
        seed = int(m["scenario_seed"])
        seq_map.setdefault((tier, seq), []).append(m)

    records = []

    for (tier, seq), metrics_list in sorted(seq_map.items()):
        # Sort by seed
        metrics_list_sorted = sorted(metrics_list, key=lambda x: int(x["scenario_seed"]))

        # Find canonical test candidate for this sequence
        test_entries = [m for m in metrics_list_sorted if (tier, seq, int(m["scenario_seed"])) in test_canonical_lookup]
        if len(test_entries) != 1:
            raise ValueError(f"Expected exactly 1 test canonical candidate for {tier} {seq}, found {len(test_entries)}")
        test_metric = test_entries[0]
        test_seed = int(test_metric["scenario_seed"])
        test_cand_meta = test_canonical_lookup[(tier, seq, test_seed)]

        # The remaining 19 candidates
        remaining = [m for m in metrics_list_sorted if int(m["scenario_seed"]) != test_seed]
        if len(remaining) != 19:
            raise ValueError(f"Expected exactly 19 non-test candidates for {tier} {seq}, found {len(remaining)}")

        # Deterministic SHA-256 sorting for stratified train/val assignment
        def compute_sort_key(item):
            s = int(item["scenario_seed"])
            hash_input = f"{split_salt}_{tier}_{seq}_{s}"
            return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

        remaining_sorted = sorted(remaining, key=compute_sort_key)
        train_metrics = remaining_sorted[:15]
        val_metrics = remaining_sorted[15:]

        # Create GeometryRecord for TEST
        test_geom_id = f"geom_{tier.lower()}_{seq}_seed{test_seed}"
        exact_blocks = test_cand_meta.get("exact_block_sequence")
        geom_hash = canonical_json_sha256(exact_blocks) if exact_blocks else hashlib.sha256(f"{seq}_{test_seed}".encode()).hexdigest()

        records.append(GeometryRecord(
            geometry_id=test_geom_id,
            tier=tier,
            sequence=seq,
            geometry_generation_seed=test_seed,
            split=SplitRole.TEST,
            candidate_role=test_cand_meta.get("candidate_role", "canonical_test"),
            block_ids=test_metric["block_ids"],
            route_length_m=float(test_metric["route_total_length_m"]),
            geometry_sha256=geom_hash,
            exact_block_sequence=exact_blocks
        ))

        # Create GeometryRecords for TRAIN
        for m in train_metrics:
            s = int(m["scenario_seed"])
            records.append(GeometryRecord(
                geometry_id=f"geom_{tier.lower()}_{seq}_seed{s}",
                tier=tier,
                sequence=seq,
                geometry_generation_seed=s,
                split=SplitRole.TRAIN,
                candidate_role="pool_candidate",
                block_ids=m["block_ids"],
                route_length_m=float(m["route_total_length_m"]),
                geometry_sha256=hashlib.sha256(f"{seq}_{s}_{m['block_ids']}_{m['route_total_length_m']}".encode()).hexdigest(),
                exact_block_sequence=None
            ))

        # Create GeometryRecords for VALIDATION
        for m in val_metrics:
            s = int(m["scenario_seed"])
            records.append(GeometryRecord(
                geometry_id=f"geom_{tier.lower()}_{seq}_seed{s}",
                tier=tier,
                sequence=seq,
                geometry_generation_seed=s,
                split=SplitRole.VALIDATION,
                candidate_role="pool_candidate",
                block_ids=m["block_ids"],
                route_length_m=float(m["route_total_length_m"]),
                geometry_sha256=hashlib.sha256(f"{seq}_{s}_{m['block_ids']}_{m['route_total_length_m']}".encode()).hexdigest(),
                exact_block_sequence=None
            ))

    return records


def validate_split_integrity(records: List[GeometryRecord]) -> Dict[str, Any]:
    """
    Verifies that split assignment invariants strictly hold.
    Fails loudly if any overlap, leakage, count mismatch, or duplicate occurs.
    """
    if len(records) != 240:
        raise ValueError(f"Total geometries must be 240, got {len(records)}")

    train_recs = [r for r in records if r.split == SplitRole.TRAIN]
    val_recs = [r for r in records if r.split == SplitRole.VALIDATION]
    test_recs = [r for r in records if r.split == SplitRole.TEST]

    if len(train_recs) != 180:
        raise ValueError(f"Expected 180 TRAIN geometries, got {len(train_recs)}")
    if len(val_recs) != 48:
        raise ValueError(f"Expected 48 VALIDATION geometries, got {len(val_recs)}")
    if len(test_recs) != 12:
        raise ValueError(f"Expected 12 TEST geometries, got {len(test_recs)}")

    # Check tier counts
    for tier in ("Easy", "Medium", "Hard", "Extreme"):
        t_tr = sum(1 for r in train_recs if r.tier == tier)
        t_vl = sum(1 for r in val_recs if r.tier == tier)
        t_ts = sum(1 for r in test_recs if r.tier == tier)
        if t_tr != 45 or t_vl != 12 or t_ts != 3:
            raise ValueError(f"Tier {tier} balance mismatch: {t_tr} train, {t_vl} val, {t_ts} test")

    # Sequence + Seed pair overlap check
    train_pairs = {(r.sequence, r.geometry_generation_seed) for r in train_recs}
    val_pairs = {(r.sequence, r.geometry_generation_seed) for r in val_recs}
    test_pairs = {(r.sequence, r.geometry_generation_seed) for r in test_recs}

    if train_pairs.intersection(val_pairs):
        raise ValueError("Leakage detected: sequence+seed pair in both TRAIN and VALIDATION")
    if train_pairs.intersection(test_pairs):
        raise ValueError("Leakage detected: sequence+seed pair in both TRAIN and TEST")
    if val_pairs.intersection(test_pairs):
        raise ValueError("Leakage detected: sequence+seed pair in both VALIDATION and TEST")

    # Geometry hash overlap check
    test_hashes = {r.geometry_sha256 for r in test_recs}
    train_hashes = {r.geometry_sha256 for r in train_recs}
    val_hashes = {r.geometry_sha256 for r in val_recs}

    if test_hashes.intersection(train_hashes):
        raise ValueError("Leakage detected: test geometry hash present in TRAIN")
    if test_hashes.intersection(val_hashes):
        raise ValueError("Leakage detected: test geometry hash present in VALIDATION")

    return {
        "status": "VALID",
        "total_geometries": 240,
        "train_count": len(train_recs),
        "validation_count": len(val_recs),
        "test_count": len(test_recs),
        "leakage_detected": False,
        "split_balance_verified": True
    }


def build_test_cases(
    test_geometries: List[GeometryRecord],
    test_env_seeds: List[int],
    protocol_order_seed: int = 424242,
    density_map: Optional[Dict[str, float]] = None,
    horizons_map: Optional[Dict[str, int]] = None
) -> List[EvaluationCase]:
    """
    Builds the 60 canonical test evaluation cases (12 geometries x 5 environment seeds)
    ordered deterministically by protocol_order_seed.
    """
    default_densities = {"Easy": 0.0, "Medium": 0.08, "Hard": 0.15, "Extreme": 0.25}
    densities = density_map or default_densities

    raw_cases = []
    case_idx = 1
    for geom in sorted(test_geometries, key=lambda g: (g.tier, g.sequence, g.geometry_generation_seed)):
        for env_seed in sorted(test_env_seeds):
            case_id = f"test/{geom.tier}/{geom.sequence}/geom-{geom.geometry_generation_seed}/env-{env_seed}"
            density = densities.get(geom.tier, 0.0)

            # Route-aware horizon
            if horizons_map and geom.geometry_id in horizons_map:
                horizon = horizons_map[geom.geometry_id]
            else:
                horizon = max(1000, math.ceil(geom.route_length_m * 3.0))

            raw_cases.append({
                "case_id": case_id,
                "case_index": case_idx,
                "split": SplitRole.TEST,
                "tier": geom.tier,
                "sequence": geom.sequence,
                "candidate_role": geom.candidate_role,
                "geometry_generation_seed": geom.geometry_generation_seed,
                "geometry_sha256": geom.geometry_sha256,
                "environment_seed": env_seed,
                "traffic_density": density,
                "horizon_steps": horizon,
            })
            case_idx += 1

    # Deterministic shuffling via protocol_order_seed
    def order_key(case_dict):
        k = f"{protocol_order_seed}_{case_dict['case_id']}"
        return hashlib.sha256(k.encode("utf-8")).hexdigest()

    sorted_cases = sorted(raw_cases, key=order_key)

    evaluation_cases = []
    for order_idx, c in enumerate(sorted_cases, 1):
        evaluation_cases.append(EvaluationCase(
            case_id=c["case_id"],
            case_index=c["case_index"],
            protocol_order_index=order_idx,
            split=c["split"],
            tier=c["tier"],
            sequence=c["sequence"],
            candidate_role=c["candidate_role"],
            geometry_generation_seed=c["geometry_generation_seed"],
            geometry_sha256=c["geometry_sha256"],
            environment_seed=c["environment_seed"],
            traffic_density=c["traffic_density"],
            horizon_steps=c["horizon_steps"],
        ))

    return evaluation_cases


def build_validation_cases(
    validation_geometries: List[GeometryRecord],
    val_env_seeds: List[int],
    protocol_order_seed: int = 424242,
    density_map: Optional[Dict[str, float]] = None,
) -> List[EvaluationCase]:
    """
    Builds the 96 validation cases (48 geometries x 2 environment seeds).
    """
    default_densities = {"Easy": 0.0, "Medium": 0.08, "Hard": 0.15, "Extreme": 0.25}
    densities = density_map or default_densities

    raw_cases = []
    case_idx = 1
    for geom in sorted(validation_geometries, key=lambda g: (g.tier, g.sequence, g.geometry_generation_seed)):
        for env_seed in sorted(val_env_seeds):
            case_id = f"val/{geom.tier}/{geom.sequence}/geom-{geom.geometry_generation_seed}/env-{env_seed}"
            density = densities.get(geom.tier, 0.0)
            horizon = max(1000, math.ceil(geom.route_length_m * 3.0))

            raw_cases.append({
                "case_id": case_id,
                "case_index": case_idx,
                "split": SplitRole.VALIDATION,
                "tier": geom.tier,
                "sequence": geom.sequence,
                "candidate_role": geom.candidate_role,
                "geometry_generation_seed": geom.geometry_generation_seed,
                "geometry_sha256": geom.geometry_sha256,
                "environment_seed": env_seed,
                "traffic_density": density,
                "horizon_steps": horizon,
            })
            case_idx += 1

    def order_key(case_dict):
        k = f"{protocol_order_seed}_{case_dict['case_id']}"
        return hashlib.sha256(k.encode("utf-8")).hexdigest()

    sorted_cases = sorted(raw_cases, key=order_key)

    eval_cases = []
    for order_idx, c in enumerate(sorted_cases, 1):
        eval_cases.append(EvaluationCase(
            case_id=c["case_id"],
            case_index=c["case_index"],
            protocol_order_index=order_idx,
            split=c["split"],
            tier=c["tier"],
            sequence=c["sequence"],
            candidate_role=c["candidate_role"],
            geometry_generation_seed=c["geometry_generation_seed"],
            geometry_sha256=c["geometry_sha256"],
            environment_seed=c["environment_seed"],
            traffic_density=c["traffic_density"],
            horizon_steps=c["horizon_steps"],
        ))

    return eval_cases


def compute_macro_metrics(tier_scorecards: Dict[str, AggregateMetrics]) -> Dict[str, Any]:
    """
    Computes transparent, equal-tier-weighted macro summaries across the four tiers.
    Does NOT compute an arbitrary geometric-mean mega-score.
    """
    tiers = ("Easy", "Medium", "Hard", "Extreme")
    for t in tiers:
        if t not in tier_scorecards:
            raise ValueError(f"Missing tier scorecard for macro aggregation: {t}")

    macro_success = sum(tier_scorecards[t].clean_success_rate for t in tiers) / 4.0
    macro_safety_fail = sum(tier_scorecards[t].safety_failure_rate for t in tiers) / 4.0
    macro_completion = sum(tier_scorecards[t].mean_final_route_completion for t in tiers) / 4.0

    return {
        "macro_clean_success_rate": macro_success,
        "macro_safety_failure_rate": macro_safety_fail,
        "macro_mean_final_route_completion": macro_completion,
        "aggregation_weighting": "equal_tier_weighted_0.25_each",
        "geometric_mean_success_score": None,  # Explicitly rejected per Platform V1 policy
    }
