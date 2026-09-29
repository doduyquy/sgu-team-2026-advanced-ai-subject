"""
Platform V1 Scientific Evaluation Protocol, Scenario Splits, and Seed Taxonomy.
Gate 5 of Research Platform V1.

Pure, unit-testable module independent of Panda3D / simulator engine state.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import csv
import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Set, Tuple

from src.platform.metrics import AggregateMetrics
from src.platform.specs import compute_route_aware_horizon


class SplitRole(str, Enum):
    """Dataset split roles for scenario geometries."""
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"


def canonical_json_sha256(data: Any) -> str:
    """Computes a deterministic SHA-256 fingerprint from a JSON-serializable object."""
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_json_file_sha256(file_path: Any) -> str:
    """Computes deterministic SHA-256 fingerprint for a JSON file, invariant to whitespace and CRLF/LF newlines."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return canonical_json_sha256(data)


def canonical_csv_file_sha256(file_path: Any) -> str:
    """
    Computes deterministic SHA-256 fingerprint for a CSV file, invariant to CRLF/LF newlines.
    Normalizes each row to a dictionary with sorted keys.
    """
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(sorted(row.items())) for row in reader]
    return canonical_json_sha256(rows)


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
    traffic_density: float
    geometry_sha256: str
    geometry_hash_source: str = "pinned_regeneration"
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


@dataclass(frozen=True)
class HorizonPolicy:
    """Immutable parameter container for route-aware episode horizon calculation from Gate-3 EpisodeSpec."""
    reference_floor_speed_kmh: float = 18.0
    safety_margin: float = 1.5
    control_frequency_hz: int = 10
    min_horizon_steps: int = 1000
    max_horizon_steps: int = 4000

    @classmethod
    def from_episode_spec(cls, spec_dict: Dict[str, Any]) -> "HorizonPolicy":
        ep_spec = spec_dict.get("episode_specification", spec_dict)
        return cls(
            reference_floor_speed_kmh=float(ep_spec["reference_floor_speed_kmh"]),
            safety_margin=float(ep_spec["safety_margin"]),
            control_frequency_hz=int(ep_spec["control_frequency_hz"]),
            min_horizon_steps=int(ep_spec["min_horizon_steps"]),
            max_horizon_steps=int(ep_spec["max_horizon_steps"]),
        )

    def compute_horizon(self, route_length_m: float) -> int:
        return compute_route_aware_horizon(
            route_length_m=route_length_m,
            reference_floor_speed_kmh=self.reference_floor_speed_kmh,
            safety_margin=self.safety_margin,
            control_frequency_hz=self.control_frequency_hz,
            min_horizon_steps=self.min_horizon_steps,
            max_horizon_steps=self.max_horizon_steps,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def assign_geometry_splits(
    candidates_metrics: List[Dict[str, Any]],
    canonical_candidates_manifest: Dict[str, List[Dict[str, Any]]],
    split_salt: str = "platform-v1-geometry-split-v1",
    geometry_hashes_lookup: Optional[Dict[Tuple[str, int], Any]] = None,
    geometry_data_lookup: Optional[Dict[Tuple[str, int], Dict[str, Any]]] = None
) -> List[GeometryRecord]:
    """
    Deterministic stratified split of the 240 candidate geometries:
    - 12 Gate-2 human-reviewed canonical geometries -> TEST (1 primary + 2 alternates per tier).
    - For each sequence, the remaining 19 seeds are sorted by SHA-256 hash using split_salt.
    - First 15 -> TRAIN (180 total, 45 per tier).
    - Final 4 -> VALIDATION (48 total, 12 per tier).
    Supports full-precision route lengths and true exact-block geometry hashes via geometry_data_lookup.
    """
    test_canonical_lookup = {}
    for tier, cands in canonical_candidates_manifest.items():
        for c in cands:
            key = (tier, c["sequence"], int(c["scenario_seed"]))
            test_canonical_lookup[key] = c

    seq_map = {}
    for m in candidates_metrics:
        tier = m["difficulty_tier"]
        seq = m["sequence"]
        seq_map.setdefault((tier, seq), []).append(m)

    records = []

    for (tier, seq), metrics_list in sorted(seq_map.items()):
        metrics_list_sorted = sorted(metrics_list, key=lambda x: int(x["scenario_seed"]))

        test_entries = [m for m in metrics_list_sorted if (tier, seq, int(m["scenario_seed"])) in test_canonical_lookup]
        if len(test_entries) != 1:
            raise ValueError(f"Expected exactly 1 test canonical candidate for {tier} {seq}, found {len(test_entries)}")
        test_metric = test_entries[0]
        test_seed = int(test_metric["scenario_seed"])
        test_cand_meta = test_canonical_lookup[(tier, seq, test_seed)]

        remaining = [m for m in metrics_list_sorted if int(m["scenario_seed"]) != test_seed]
        if len(remaining) != 19:
            raise ValueError(f"Expected exactly 19 non-test candidates for {tier} {seq}, found {len(remaining)}")

        def compute_sort_key(item):
            s = int(item["scenario_seed"])
            hash_input = f"{split_salt}_{tier}_{seq}_{s}"
            return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

        remaining_sorted = sorted(remaining, key=compute_sort_key)
        train_metrics = remaining_sorted[:15]
        val_metrics = remaining_sorted[15:]

        # Lookup entry for test candidate if provided
        test_data_entry = None
        if geometry_data_lookup and (seq, test_seed) in geometry_data_lookup:
            test_data_entry = geometry_data_lookup[(seq, test_seed)]
        elif geometry_hashes_lookup and (seq, test_seed) in geometry_hashes_lookup:
            val = geometry_hashes_lookup[(seq, test_seed)]
            if isinstance(val, dict):
                test_data_entry = val

        # TEST Record
        test_geom_id = f"geom_{tier.lower()}_{seq}_seed{test_seed}"
        exact_blocks = test_cand_meta.get("exact_block_sequence")
        if exact_blocks:
            test_geom_hash = canonical_json_sha256(exact_blocks)
            test_source = "gate2_stored_exact"
        elif test_data_entry and "geometry_sha256" in test_data_entry:
            test_geom_hash = test_data_entry["geometry_sha256"]
            test_source = test_data_entry.get("geometry_hash_source", "pinned_regeneration")
        elif geometry_hashes_lookup and (seq, test_seed) in geometry_hashes_lookup and isinstance(geometry_hashes_lookup[(seq, test_seed)], str):
            test_geom_hash = geometry_hashes_lookup[(seq, test_seed)]
            test_source = "pinned_regeneration"
        else:
            test_geom_hash = canonical_json_sha256(f"{seq}_{test_seed}")
            test_source = "fallback_identity"

        test_route_len = (
            float(test_data_entry["route_length_m"])
            if test_data_entry and "route_length_m" in test_data_entry
            else float(test_metric["route_total_length_m"])
        )
        test_block_ids = (
            test_data_entry["block_ids"]
            if test_data_entry and "block_ids" in test_data_entry
            else test_metric["block_ids"]
        )

        records.append(GeometryRecord(
            geometry_id=test_geom_id,
            tier=tier,
            sequence=seq,
            geometry_generation_seed=test_seed,
            split=SplitRole.TEST,
            candidate_role=test_cand_meta.get("candidate_role", "canonical_test"),
            block_ids=test_block_ids,
            route_length_m=test_route_len,
            traffic_density=float(test_metric["traffic_density"]),
            geometry_sha256=test_geom_hash,
            geometry_hash_source=test_source,
            exact_block_sequence=exact_blocks
        ))

        # TRAIN Records
        for m in train_metrics:
            s = int(m["scenario_seed"])
            data_entry = None
            if geometry_data_lookup and (seq, s) in geometry_data_lookup:
                data_entry = geometry_data_lookup[(seq, s)]
            elif geometry_hashes_lookup and (seq, s) in geometry_hashes_lookup:
                val = geometry_hashes_lookup[(seq, s)]
                if isinstance(val, dict):
                    data_entry = val

            if data_entry and "geometry_sha256" in data_entry:
                g_hash = data_entry["geometry_sha256"]
                g_src = data_entry.get("geometry_hash_source", "pinned_regeneration")
            elif geometry_hashes_lookup and (seq, s) in geometry_hashes_lookup and isinstance(geometry_hashes_lookup[(seq, s)], str):
                g_hash = geometry_hashes_lookup[(seq, s)]
                g_src = "pinned_regeneration"
            else:
                g_hash = canonical_json_sha256(f"{seq}_{s}")
                g_src = "fallback_identity"

            route_len = (
                float(data_entry["route_length_m"])
                if data_entry and "route_length_m" in data_entry
                else float(m["route_total_length_m"])
            )
            b_ids = (
                data_entry["block_ids"]
                if data_entry and "block_ids" in data_entry
                else m["block_ids"]
            )

            records.append(GeometryRecord(
                geometry_id=f"geom_{tier.lower()}_{seq}_seed{s}",
                tier=tier,
                sequence=seq,
                geometry_generation_seed=s,
                split=SplitRole.TRAIN,
                candidate_role="pool_candidate",
                block_ids=b_ids,
                route_length_m=route_len,
                traffic_density=float(m["traffic_density"]),
                geometry_sha256=g_hash,
                geometry_hash_source=g_src,
                exact_block_sequence=None
            ))

        # VALIDATION Records
        for m in val_metrics:
            s = int(m["scenario_seed"])
            data_entry = None
            if geometry_data_lookup and (seq, s) in geometry_data_lookup:
                data_entry = geometry_data_lookup[(seq, s)]
            elif geometry_hashes_lookup and (seq, s) in geometry_hashes_lookup:
                val = geometry_hashes_lookup[(seq, s)]
                if isinstance(val, dict):
                    data_entry = val

            if data_entry and "geometry_sha256" in data_entry:
                g_hash = data_entry["geometry_sha256"]
                g_src = data_entry.get("geometry_hash_source", "pinned_regeneration")
            elif geometry_hashes_lookup and (seq, s) in geometry_hashes_lookup and isinstance(geometry_hashes_lookup[(seq, s)], str):
                g_hash = geometry_hashes_lookup[(seq, s)]
                g_src = "pinned_regeneration"
            else:
                g_hash = canonical_json_sha256(f"{seq}_{s}")
                g_src = "fallback_identity"

            route_len = (
                float(data_entry["route_length_m"])
                if data_entry and "route_length_m" in data_entry
                else float(m["route_total_length_m"])
            )
            b_ids = (
                data_entry["block_ids"]
                if data_entry and "block_ids" in data_entry
                else m["block_ids"]
            )

            records.append(GeometryRecord(
                geometry_id=f"geom_{tier.lower()}_{seq}_seed{s}",
                tier=tier,
                sequence=seq,
                geometry_generation_seed=s,
                split=SplitRole.VALIDATION,
                candidate_role="pool_candidate",
                block_ids=b_ids,
                route_length_m=route_len,
                traffic_density=float(m["traffic_density"]),
                geometry_sha256=g_hash,
                geometry_hash_source=g_src,
                exact_block_sequence=None
            ))

    return records


def validate_split_integrity(
    records: List[GeometryRecord],
    expected_test_identities: Optional[Set[Any]] = None
) -> Dict[str, Any]:
    """
    Verifies that split assignment invariants strictly hold across counts,
    balance, sequence+seed identities, canonical test sequestering, and all
    pairwise geometry hash intersections.
    Fails loudly if any violation occurs.
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

    # Tier counts
    for tier in ("Easy", "Medium", "Hard", "Extreme"):
        t_tr = sum(1 for r in train_recs if r.tier == tier)
        t_vl = sum(1 for r in val_recs if r.tier == tier)
        t_ts = sum(1 for r in test_recs if r.tier == tier)
        if t_tr != 45 or t_vl != 12 or t_ts != 3:
            raise ValueError(f"Tier {tier} balance mismatch: {t_tr} train, {t_vl} val, {t_ts} test")

    # Sequence counts
    sequences = set(r.sequence for r in records)
    for seq in sequences:
        s_tr = sum(1 for r in train_recs if r.sequence == seq)
        s_vl = sum(1 for r in val_recs if r.sequence == seq)
        s_ts = sum(1 for r in test_recs if r.sequence == seq)
        if s_tr != 15 or s_vl != 4 or s_ts != 1:
            raise ValueError(f"Sequence {seq} balance mismatch: {s_tr} train, {s_vl} val, {s_ts} test")

    # Identity pair overlap check
    train_pairs = {(r.sequence, r.geometry_generation_seed) for r in train_recs}
    val_pairs = {(r.sequence, r.geometry_generation_seed) for r in val_recs}
    test_pairs = {(r.sequence, r.geometry_generation_seed) for r in test_recs}

    if train_pairs.intersection(val_pairs):
        raise ValueError("Leakage detected: sequence+seed pair in both TRAIN and VALIDATION")
    if train_pairs.intersection(test_pairs):
        raise ValueError("Leakage detected: sequence+seed pair in both TRAIN and TEST")
    if val_pairs.intersection(test_pairs):
        raise ValueError("Leakage detected: sequence+seed pair in both VALIDATION and TEST")

    # Authoritative canonical test identity verification
    canonical_test_verified = False
    if expected_test_identities is not None:
        normalized_expected = set()
        for item in expected_test_identities:
            if len(item) == 3:
                normalized_expected.add((item[1], int(item[2])))
            else:
                normalized_expected.add((item[0], int(item[1])))

        missing_canonical = normalized_expected - test_pairs
        if missing_canonical:
            raise ValueError(f"Expected canonical test geometries missing from TEST split: {sorted(missing_canonical)}")
        unexpected_test = test_pairs - normalized_expected
        if unexpected_test:
            raise ValueError(f"Unexpected geometries occupying TEST split: {sorted(unexpected_test)}")
        train_canonical_leak = train_pairs.intersection(normalized_expected)
        if train_canonical_leak:
            raise ValueError(f"Canonical test geometries leaked into TRAIN split: {sorted(train_canonical_leak)}")
        val_canonical_leak = val_pairs.intersection(normalized_expected)
        if val_canonical_leak:
            raise ValueError(f"Canonical test geometries leaked into VALIDATION split: {sorted(val_canonical_leak)}")
        canonical_test_verified = True

    # Pairwise split hash intersection checks
    train_hashes = {r.geometry_sha256 for r in train_recs}
    val_hashes = {r.geometry_sha256 for r in val_recs}
    test_hashes = {r.geometry_sha256 for r in test_recs}

    leakage_train_val = train_hashes.intersection(val_hashes)
    leakage_train_test = train_hashes.intersection(test_hashes)
    leakage_val_test = val_hashes.intersection(test_hashes)

    if leakage_train_val:
        raise ValueError(f"Leakage detected between TRAIN and VALIDATION: {len(leakage_train_val)} duplicate hashes")
    if leakage_train_test:
        raise ValueError(f"Leakage detected between TRAIN and TEST: {len(leakage_train_test)} duplicate hashes")
    if leakage_val_test:
        raise ValueError(f"Leakage detected between VALIDATION and TEST: {len(leakage_val_test)} duplicate hashes")

    # Scan universe for duplicate geometry hashes
    hash_to_geoms = {}
    for r in records:
        hash_to_geoms.setdefault(r.geometry_sha256, []).append(r)

    duplicate_groups = []
    for h, group in hash_to_geoms.items():
        if len(group) > 1:
            duplicate_groups.append({
                "geometry_sha256": h,
                "count": len(group),
                "geometries": [f"{g.sequence}_seed{g.geometry_generation_seed} ({g.split.value})" for g in group]
            })

    return {
        "status": "VALID",
        "total_geometries": 240,
        "train_count": len(train_recs),
        "validation_count": len(val_recs),
        "test_count": len(test_recs),
        "leakage_detected": False,
        "split_balance_verified": True,
        "canonical_test_identities_verified": canonical_test_verified,
        "duplicate_geometry_groups_count": len(duplicate_groups),
        "duplicate_geometry_groups": duplicate_groups,
    }


def compute_manifest_sha256(cases: List[EvaluationCase]) -> str:
    """
    Computes production canonical SHA-256 fingerprint from an ordered evaluation case list.
    Matches canonical_csv_file_sha256 by normalizing each case to sorted string-keyed dictionary.
    """
    rows = []
    for c in sorted(cases, key=lambda x: x.protocol_order_index):
        d = {k: str(v) for k, v in c.to_dict().items() if k != "split"}
        rows.append(dict(sorted(d.items())))
    return canonical_json_sha256(rows)


def build_test_cases(
    test_geometries: List[GeometryRecord],
    test_env_seeds: List[int],
    protocol_order_seed: int = 424242,
    horizon_policy: Optional[HorizonPolicy] = None,
) -> List[EvaluationCase]:
    """
    Builds the 60 canonical test evaluation cases (12 geometries x 5 environment seeds)
    using Gate-3 compute_route_aware_horizon via HorizonPolicy and propagating geometry traffic_density.
    """
    policy = horizon_policy or HorizonPolicy()
    raw_cases = []
    case_idx = 1
    for geom in sorted(test_geometries, key=lambda g: (g.tier, g.sequence, g.geometry_generation_seed)):
        for env_seed in sorted(test_env_seeds):
            case_id = f"test/{geom.tier}/{geom.sequence}/geom-{geom.geometry_generation_seed}/env-{env_seed}"

            # Gate-3 canonical route-aware horizon computation via HorizonPolicy
            horizon = policy.compute_horizon(geom.route_length_m)

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
                "traffic_density": geom.traffic_density,
                "horizon_steps": horizon,
            })
            case_idx += 1

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
    horizon_policy: Optional[HorizonPolicy] = None,
) -> List[EvaluationCase]:
    """
    Builds the 96 validation cases (48 geometries x 2 environment seeds).
    """
    policy = horizon_policy or HorizonPolicy()
    raw_cases = []
    case_idx = 1
    for geom in sorted(validation_geometries, key=lambda g: (g.tier, g.sequence, g.geometry_generation_seed)):
        for env_seed in sorted(val_env_seeds):
            case_id = f"val/{geom.tier}/{geom.sequence}/geom-{geom.geometry_generation_seed}/env-{env_seed}"

            horizon = policy.compute_horizon(geom.route_length_m)

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
                "traffic_density": geom.traffic_density,
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


def build_evaluation_protocol_core(
    test_cases: List[EvaluationCase],
    val_cases: List[EvaluationCase],
    test_environment_seeds: Optional[List[int]] = None,
    validation_environment_seeds: Optional[List[int]] = None,
    custom_rules: Optional[Dict[str, str]] = None,
    pinned_commit: str = "85e5dadc6c7436d324348f6e3d8f8e680c06b4db",
    pinned_version: str = "0.4.3"
) -> Dict[str, Any]:
    """
    Builds the authoritative EvaluationProtocolV1 core content WITHOUT hash fingerprints.
    Captures protocol metadata, evaluation rules, and case suite summaries for canonical hashing.
    """
    default_rules = {
        "paired_evaluation": "All algorithms must evaluate the exact same ordered test cases from test_case_manifest.csv.",
        "test_set_holdout": "TEST geometries (12 canonicals) are strictly held out. No training, hyperparameter search, or checkpoint selection allowed on TEST.",
        "validation_usage": "VALIDATION cases (96 cases) exist solely for hyperparameter tuning, ablation studies, and model checkpoint selection.",
        "replicate_reporting": "Stochastic inference methods report mean \u00b1 std across 3 independent replicates (agent seeds 101, 202, 303). Learned methods train 3 independent models on training run seeds 101, 202, 303. Genuinely deterministic methods report 1 run per environment case.",
        "tier_scorecards": "Primary benchmark reporting uses per-tier scorecards (Easy, Medium, Hard, Extreme) on Gate-4 primary metrics. Geometric-mean success mega-score is explicitly prohibited."
    }
    rules = dict(default_rules)
    if custom_rules:
        rules.update(custom_rules)

    t_seeds = test_environment_seeds or sorted(list(set(c.environment_seed for c in test_cases)))
    v_seeds = validation_environment_seeds or sorted(list(set(c.environment_seed for c in val_cases)))

    return {
        "metadata": {
            "protocol_name": "EvaluationProtocolV1",
            "spec_version": "1.0.0",
            "status": "LOCKED-FOR-PLATFORM-V1",
            "gate": "Gate 5 (Research Platform V1)",
            "pinned_metadrive_commit": pinned_commit,
            "pinned_metadrive_version": pinned_version,
        },
        "evaluation_rules": rules,
        "test_suite_summary": {
            "test_cases_count": len(test_cases),
            "test_geometries_count": len(set((c.sequence, c.geometry_generation_seed) for c in test_cases)),
            "test_environment_seeds": t_seeds,
            "cases_per_tier": {
                "Easy": sum(1 for c in test_cases if c.tier == "Easy"),
                "Medium": sum(1 for c in test_cases if c.tier == "Medium"),
                "Hard": sum(1 for c in test_cases if c.tier == "Hard"),
                "Extreme": sum(1 for c in test_cases if c.tier == "Extreme"),
            }
        },
        "validation_suite_summary": {
            "validation_cases_count": len(val_cases),
            "validation_geometries_count": len(set((c.sequence, c.geometry_generation_seed) for c in val_cases)),
            "validation_environment_seeds": v_seeds,
            "cases_per_tier": {
                "Easy": sum(1 for c in val_cases if c.tier == "Easy"),
                "Medium": sum(1 for c in val_cases if c.tier == "Medium"),
                "Hard": sum(1 for c in val_cases if c.tier == "Hard"),
                "Extreme": sum(1 for c in val_cases if c.tier == "Extreme"),
            }
        },
        "training_suite_summary": {
            "training_geometries_count": 180,
            "seed_derivation": "derive_seed(training_run_seed, 'environment', episode_index, geometry_id)"
        }
    }


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
