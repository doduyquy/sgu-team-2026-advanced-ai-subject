"""
Reproducible MapSuite V1 TRAIN Split Case Sampler (Gate S5.1).

Selects geometries strictly from the 180 verified TRAIN split records in
geometry_split_manifest.csv. Strictly isolates VALIDATION and TEST holdouts.

Key Design Requirements:
- Separate training root seed from environment seed and geometry seed.
- Support fixed case selection (for debugging/parity tests), sequential round-robin,
  uniform random, and tier-balanced curriculum sampling.
- Cryptographically verifies manifest against locked Gate-5 SHA-256 digest.
- Output authentic ResolvedCaseV1 instances with route-aware horizon steps.
"""

from __future__ import annotations

import csv
from enum import Enum
import hashlib
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.launcher.cases import (
    LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256,
    compute_route_aware_horizon,
    get_default_project_root,
    load_manifest_csv,
)
from src.launcher.models import ResolvedCaseV1


class SamplerStrategy(str, Enum):
    FIXED = "FIXED"                    # Repeats a designated single TRAIN case
    ROUND_ROBIN = "ROUND_ROBIN"        # Deterministic sequential cycle through TRAIN cases
    UNIFORM_RANDOM = "UNIFORM_RANDOM"  # Uniform pseudorandom selection across eligible TRAIN cases
    TIER_BALANCED = "TIER_BALANCED"    # Uniform selection across 4 tiers (Easy, Medium, Hard, Extreme)


class TrainCaseSampler:
    """
    Deterministic sampler providing training cases from MapSuite V1 TRAIN split.
    Guarantees zero leakage of VALIDATION (48) or TEST (12) holdout geometries.
    """

    def __init__(
        self,
        training_root_seed: int = 101,
        strategy: str | SamplerStrategy = SamplerStrategy.ROUND_ROBIN,
        tier_filter: Optional[str] = None,
        sequence_filter: Optional[str] = None,
        fixed_geometry_seed: Optional[int] = None,
        project_root: Optional[Path] = None,
    ) -> None:
        """
        Initializes the TRAIN case sampler.

        Args:
            training_root_seed: Master seed governing pseudorandom sampling sequence.
            strategy: SamplerStrategy enum or string ("FIXED", "ROUND_ROBIN", "UNIFORM_RANDOM", "TIER_BALANCED").
            tier_filter: Optional difficulty filter ("Easy", "Medium", "Hard", "Extreme").
            sequence_filter: Optional scenario sequence family filter (e.g. "SCS", "SCXCCS").
            fixed_geometry_seed: Required if strategy == FIXED; specific geometry generation seed.
            project_root: Subproject root directory path.
        """
        self.project_root = project_root or get_default_project_root()
        self.training_root_seed = int(training_root_seed)
        self.strategy = SamplerStrategy(strategy)
        self.tier_filter = tier_filter.lower() if tier_filter else None
        self.sequence_filter = sequence_filter
        self.fixed_geometry_seed = int(fixed_geometry_seed) if fixed_geometry_seed is not None else None

        # 1. Load and cryptographically verify geometry manifest
        manifest_path = self.project_root / "results" / "audits" / "evaluation_protocol" / "geometry_split_manifest.csv"
        all_records = load_manifest_csv(manifest_path, expected_sha256=LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256)

        # 2. Strict holdout isolation: filter ONLY TRAIN geometries
        train_records = []
        for r in all_records:
            split = r["split"].upper()
            if split in ("VALIDATION", "TEST"):
                continue  # Strictly isolated
            if split == "TRAIN":
                train_records.append(r)
            else:
                raise ValueError(f"Unrecognized split in manifest: '{split}'")

        if len(train_records) != 180:
            raise ValueError(f"Expected exactly 180 TRAIN split records, got {len(train_records)}")

        # 3. Apply optional tier and sequence filters
        if self.tier_filter is not None:
            train_records = [r for r in train_records if r["tier"].lower() == self.tier_filter]
            if not train_records:
                raise ValueError(f"No TRAIN records match tier filter '{self.tier_filter}'")

        if self.sequence_filter is not None:
            train_records = [r for r in train_records if r["sequence"] == self.sequence_filter]
            if not train_records:
                raise ValueError(f"No TRAIN records match sequence filter '{self.sequence_filter}'")

        # Stable canonical sorting: Tier, Sequence, geometry_generation_seed
        tier_weights = {"easy": 1, "medium": 2, "hard": 3, "extreme": 4}
        train_records.sort(
            key=lambda r: (
                tier_weights.get(r["tier"].lower(), 99),
                r["sequence"],
                int(r["geometry_generation_seed"]),
            )
        )
        self._records = train_records

        # Group by tier for TIER_BALANCED strategy
        self._records_by_tier: Dict[str, List[Dict[str, str]]] = {}
        for r in self._records:
            t = r["tier"].lower()
            self._records_by_tier.setdefault(t, []).append(r)

        # 4. Comprehensive TRAIN manifest lookup for holdout protection (unfiltered 180 records)
        self._train_manifest_lookup: Dict[Tuple[str, str, int], Dict[str, str]] = {
            (r["tier"].lower(), r["sequence"], int(r["geometry_generation_seed"])): r
            for r in all_records
            if r["split"].upper() == "TRAIN"
        }

        # 5. Fixed strategy validation: must unambiguously resolve to exactly ONE geometry
        if self.strategy == SamplerStrategy.FIXED:
            matches = list(self._records)
            if self.fixed_geometry_seed is not None:
                matches = [r for r in matches if int(r["geometry_generation_seed"]) == self.fixed_geometry_seed]

            if len(matches) == 0:
                raise ValueError(
                    f"Fixed strategy failed: No TRAIN geometry matches filters "
                    f"(tier={self.tier_filter}, sequence={self.sequence_filter}, seed={self.fixed_geometry_seed})"
                )
            if len(matches) > 1:
                matching_ids = [m["geometry_id"] for m in matches]
                raise ValueError(
                    f"Ambiguous fixed geometry request: seed {self.fixed_geometry_seed} matches {len(matches)} TRAIN geometries ({matching_ids}). "
                    f"Please specify tier_filter and sequence_filter to unambiguously identify exactly one geometry."
                )
            self._fixed_record = matches[0]
        else:
            self._fixed_record = None

        # 6. Initialize sampling state
        self._rng = random.Random(self.training_root_seed)
        self._np_rng = np.random.RandomState(self.training_root_seed)
        self._cycle_index = 0
        self._sampled_count = 0
        self._last_sampled_case: Optional[ResolvedCaseV1] = None

    @property
    def total_eligible_geometries(self) -> int:
        """Returns the number of distinct TRAIN geometries in the active pool."""
        return len(self._records)

    @property
    def sampled_count(self) -> int:
        """Total number of cases sampled since initialization/reset."""
        return self._sampled_count

    @property
    def last_sampled_case(self) -> Optional[ResolvedCaseV1]:
        """Most recently sampled ResolvedCaseV1."""
        return self._last_sampled_case

    def validate_train_case(self, case: Any) -> None:
        """
        Validates that a given case matches an authentic, verified MapSuite V1 TRAIN geometry.
        Loudly rejects forged split labels, holdout geometries (VALIDATION, TEST), and mismatched parameters.
        """
        if not isinstance(case, ResolvedCaseV1):
            raise TypeError(f"Expected ResolvedCaseV1 instance, got {type(case).__name__}")

        if str(case.split).upper() != "TRAIN":
            raise ValueError(f"FATAL HOLDOUT VIOLATION: Case '{case.case_id}' split '{case.split}' is not TRAIN!")

        geom_seed = int(case.geometry_generation_seed)
        key = (case.tier.lower(), case.sequence, geom_seed)
        if key not in self._train_manifest_lookup:
            raise ValueError(
                f"FATAL HOLDOUT VIOLATION: Case '{case.case_id}' geometry "
                f"(tier='{case.tier}', sequence='{case.sequence}', geom_seed={geom_seed}) "
                f"does not exist in verified MapSuite V1 TRAIN manifest!"
            )

        record = self._train_manifest_lookup[key]
        if case.geometry_sha256 != record["geometry_sha256"]:
            raise ValueError(
                f"FATAL GEOMETRY HASH MISMATCH for case '{case.case_id}': "
                f"expected={record['geometry_sha256']}, got={case.geometry_sha256}"
            )

        if abs(float(case.traffic_density) - float(record["traffic_density"])) > 1e-4:
            raise ValueError(
                f"FATAL TRAFFIC DENSITY MISMATCH for case '{case.case_id}': "
                f"expected={record['traffic_density']}, got={case.traffic_density}"
            )

        expected_horizon = compute_route_aware_horizon(float(record["route_length_m"]))
        if case.horizon_steps != expected_horizon:
            raise ValueError(
                f"FATAL HORIZON STEPS MISMATCH for case '{case.case_id}': "
                f"expected={expected_horizon}, got={case.horizon_steps}"
            )

    def get_case(
        self,
        geometry_seed: int,
        tier: Optional[str] = None,
        sequence: Optional[str] = None,
        environment_seed: Optional[int] = None,
    ) -> ResolvedCaseV1:
        """
        Unambiguously retrieves an authentic TRAIN case by (tier, sequence, geometry_seed).
        Raises ValueError if the specification matches multiple geometries or zero geometries.
        """
        target_seed = int(geometry_seed)
        matches = [
            r for r in self._train_manifest_lookup.values()
            if int(r["geometry_generation_seed"]) == target_seed
        ]

        if tier is not None:
            matches = [r for r in matches if r["tier"].lower() == tier.lower()]
        elif self.tier_filter is not None:
            matches = [r for r in matches if r["tier"].lower() == self.tier_filter]

        if sequence is not None:
            matches = [r for r in matches if r["sequence"] == sequence]
        elif self.sequence_filter is not None:
            matches = [r for r in matches if r["sequence"] == self.sequence_filter]

        if len(matches) == 0:
            raise ValueError(
                f"No TRAIN geometry found matching (tier={tier or self.tier_filter}, sequence={sequence or self.sequence_filter}, seed={target_seed})."
            )
        if len(matches) > 1:
            matching_ids = [m["geometry_id"] for m in matches]
            raise ValueError(
                f"Ambiguous geometry request: seed {target_seed} matches {len(matches)} TRAIN geometries ({matching_ids}). "
                f"Please supply 'tier' and 'sequence' to specify an exact geometry."
            )

        record = matches[0]
        env_seed = int(environment_seed) if environment_seed is not None else 0
        tier_str = record["tier"]
        seq_str = record["sequence"]
        geom_hash = record["geometry_sha256"]
        traffic_density = float(record["traffic_density"])
        route_len = float(record["route_length_m"])
        horizon_steps = compute_route_aware_horizon(route_len)

        case_id = f"train/{tier_str}/{seq_str}/geom-{target_seed}/env-{env_seed}"

        return ResolvedCaseV1(
            case_id=case_id,
            case_index=1,
            protocol_order_index=1,
            split="TRAIN",
            tier=tier_str,
            sequence=seq_str,
            geometry_generation_seed=target_seed,
            geometry_sha256=geom_hash,
            environment_seed=env_seed,
            traffic_density=traffic_density,
            horizon_steps=horizon_steps,
        )

    def get_case_for_geometry_seed(
        self,
        geom_seed: int,
        environment_seed: Optional[int] = None,
        tier: Optional[str] = None,
        sequence: Optional[str] = None,
    ) -> ResolvedCaseV1:
        """Forwarding alias to get_case() ensuring backwards compatibility with ambiguity validation."""
        return self.get_case(
            geometry_seed=geom_seed,
            tier=tier,
            sequence=sequence,
            environment_seed=environment_seed,
        )

    def reset_seed(self, seed: int) -> None:
        """Resets the pseudorandom sampling sequence with a new training root seed."""
        self.training_root_seed = int(seed)
        self._rng = random.Random(self.training_root_seed)
        self._np_rng = np.random.RandomState(self.training_root_seed)
        self._cycle_index = 0
        self._sampled_count = 0

    def derive_environment_seed(self, episode_index: int) -> int:
        """
        Derives a deterministic, reproducible environment seed for MetaDrive simulator
        given the master training seed and the current episode index.
        """
        # 32-bit unsigned integer derived via SHA-256
        seed_str = f"train_env_seed:{self.training_root_seed}:{episode_index}"
        digest = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % 1_000_000

    def get_case_for_seed(self, seed: int) -> ResolvedCaseV1:
        """
        Deterministically maps an explicit seed to an exact TRAIN case.
        Guarantees that identical seed selects identical case and parameters,
        fulfilling the Gymnasium reset(seed=...) determinism contract.
        """
        seed_val = int(seed)
        idx = seed_val % len(self._records)
        record = self._records[idx]

        if record["split"].upper() != "TRAIN":
            raise ValueError(f"FATAL HOLDOUT LEAKAGE: Selected case from '{record['split']}' split! Must be strictly TRAIN.")

        tier = record["tier"]
        sequence = record["sequence"]
        geom_seed = int(record["geometry_generation_seed"])
        geom_hash = record["geometry_sha256"]
        traffic_density = float(record["traffic_density"])
        route_len = float(record["route_length_m"])
        horizon_steps = compute_route_aware_horizon(route_len)

        case_id = f"train/{tier}/{sequence}/geom-{geom_seed}/env-{seed_val}"

        resolved = ResolvedCaseV1(
            case_id=case_id,
            case_index=idx + 1,
            protocol_order_index=idx + 1,
            split="TRAIN",
            tier=tier,
            sequence=sequence,
            geometry_generation_seed=geom_seed,
            geometry_sha256=geom_hash,
            environment_seed=seed_val,
            traffic_density=traffic_density,
            horizon_steps=horizon_steps,
        )

        self._last_sampled_case = resolved
        return resolved

    def sample_case(self, environment_seed: Optional[int] = None) -> ResolvedCaseV1:
        """
        Samples the next training case according to the designated sampling strategy.

        Args:
            environment_seed: Optional explicit simulator seed override. If None, derived deterministically.

        Returns:
            ResolvedCaseV1 instance referencing strictly TRAIN geometry.
        """
        record: Dict[str, str]

        if self.strategy == SamplerStrategy.FIXED:
            assert self._fixed_record is not None
            record = self._fixed_record

        elif self.strategy == SamplerStrategy.ROUND_ROBIN:
            record = self._records[self._cycle_index % len(self._records)]
            self._cycle_index += 1

        elif self.strategy == SamplerStrategy.UNIFORM_RANDOM:
            idx = self._rng.randint(0, len(self._records) - 1)
            record = self._records[idx]

        elif self.strategy == SamplerStrategy.TIER_BALANCED:
            tiers = sorted(self._records_by_tier.keys())
            chosen_tier = self._rng.choice(tiers)
            tier_recs = self._records_by_tier[chosen_tier]
            record = self._rng.choice(tier_recs)

        else:
            raise ValueError(f"Unsupported sampling strategy: '{self.strategy}'")

        # Verify holdout protection strictly
        if record["split"].upper() != "TRAIN":
            raise ValueError(f"FATAL HOLDOUT LEAKAGE: Sampled case from '{record['split']}' split! Must be strictly TRAIN.")

        # Determine environment seed
        self._sampled_count += 1
        env_seed = int(environment_seed) if environment_seed is not None else self.derive_environment_seed(self._sampled_count)

        tier = record["tier"]
        sequence = record["sequence"]
        geom_seed = int(record["geometry_generation_seed"])
        geom_hash = record["geometry_sha256"]
        traffic_density = float(record["traffic_density"])
        route_len = float(record["route_length_m"])
        horizon_steps = compute_route_aware_horizon(route_len)

        case_id = f"train/{tier}/{sequence}/geom-{geom_seed}/env-{env_seed}"

        resolved = ResolvedCaseV1(
            case_id=case_id,
            case_index=self._sampled_count,
            protocol_order_index=self._sampled_count,
            split="TRAIN",
            tier=tier,
            sequence=sequence,
            geometry_generation_seed=geom_seed,
            geometry_sha256=geom_hash,
            environment_seed=env_seed,
            traffic_density=traffic_density,
            horizon_steps=horizon_steps,
        )

        self._last_sampled_case = resolved
        return resolved
