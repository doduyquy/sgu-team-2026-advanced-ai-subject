"""
Unit tests for Platform V1 scientific evaluation protocol, scenario splits, and seed taxonomy.
Pure, unit-testable tests running in <0.05s without Panda3D / MetaDrive.
"""

import csv
import json
from pathlib import Path
import unittest

from src.platform import (
    AggregateMetrics,
    EvaluationCase,
    GeometryRecord,
    ScenarioSplitV1,
    SeedPlanV1,
    SplitRole,
    assign_geometry_splits,
    build_test_cases,
    build_validation_cases,
    canonical_json_sha256,
    compute_macro_metrics,
    derive_seed,
    validate_split_integrity,
)


class TestEvaluationProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project_root = Path(__file__).resolve().parent.parent
        metrics_csv = project_root / "results" / "audits" / "mapsuite" / "candidate_metrics.csv"
        canon_json = project_root / "results" / "audits" / "mapsuite" / "canonical_candidates.json"

        with open(metrics_csv, "r", encoding="utf-8") as f:
            cls.metrics = list(csv.DictReader(f))
        with open(canon_json, "r", encoding="utf-8") as f:
            cls.canon_manifest = json.load(f)

        cls.split_salt = "platform-v1-geometry-split-v1"
        cls.geometries = assign_geometry_splits(cls.metrics, cls.canon_manifest, cls.split_salt)

    def test_240_geometry_universe(self):
        self.assertEqual(len(self.metrics), 240)
        self.assertEqual(len(self.geometries), 240)

    def test_split_counts_180_48_12(self):
        train = [g for g in self.geometries if g.split == SplitRole.TRAIN]
        val = [g for g in self.geometries if g.split == SplitRole.VALIDATION]
        test = [g for g in self.geometries if g.split == SplitRole.TEST]

        self.assertEqual(len(train), 180)
        self.assertEqual(len(val), 48)
        self.assertEqual(len(test), 12)

    def test_per_tier_and_per_sequence_balance(self):
        tiers = ("Easy", "Medium", "Hard", "Extreme")
        for t in tiers:
            t_train = sum(1 for g in self.geometries if g.tier == t and g.split == SplitRole.TRAIN)
            t_val = sum(1 for g in self.geometries if g.tier == t and g.split == SplitRole.VALIDATION)
            t_test = sum(1 for g in self.geometries if g.tier == t and g.split == SplitRole.TEST)
            self.assertEqual(t_train, 45, f"Tier {t} train count mismatch")
            self.assertEqual(t_val, 12, f"Tier {t} val count mismatch")
            self.assertEqual(t_test, 3, f"Tier {t} test count mismatch")

        # Per sequence: 15 train, 4 val, 1 test
        sequences = set(g.sequence for g in self.geometries)
        self.assertEqual(len(sequences), 12)
        for seq in sequences:
            s_train = sum(1 for g in self.geometries if g.sequence == seq and g.split == SplitRole.TRAIN)
            s_val = sum(1 for g in self.geometries if g.sequence == seq and g.split == SplitRole.VALIDATION)
            s_test = sum(1 for g in self.geometries if g.sequence == seq and g.split == SplitRole.TEST)
            self.assertEqual(s_train, 15, f"Sequence {seq} train mismatch")
            self.assertEqual(s_val, 4, f"Sequence {seq} val mismatch")
            self.assertEqual(s_test, 1, f"Sequence {seq} test mismatch")

    def test_all_12_canonical_geometries_in_test_only(self):
        canon_keys = set()
        for tier, cands in self.canon_manifest.items():
            for c in cands:
                canon_keys.add((tier, c["sequence"], int(c["scenario_seed"])))

        self.assertEqual(len(canon_keys), 12)

        test_keys = set((g.tier, g.sequence, g.geometry_generation_seed) for g in self.geometries if g.split == SplitRole.TEST)
        train_keys = set((g.tier, g.sequence, g.geometry_generation_seed) for g in self.geometries if g.split == SplitRole.TRAIN)
        val_keys = set((g.tier, g.sequence, g.geometry_generation_seed) for g in self.geometries if g.split == SplitRole.VALIDATION)

        # Exact match with test keys
        self.assertEqual(test_keys, canon_keys)
        # Zero canonical keys in train or val
        self.assertEqual(len(canon_keys.intersection(train_keys)), 0)
        self.assertEqual(len(canon_keys.intersection(val_keys)), 0)

    def test_zero_leakage_and_integrity_validation(self):
        report = validate_split_integrity(self.geometries)
        self.assertEqual(report["status"], "VALID")
        self.assertFalse(report["leakage_detected"])
        self.assertTrue(report["split_balance_verified"])

    def test_deterministic_split_reproducibility(self):
        # Running assign_geometry_splits twice with same salt produces bit-for-bit identical records
        geoms_run2 = assign_geometry_splits(self.metrics, self.canon_manifest, self.split_salt)
        self.assertEqual(
            [g.to_dict() for g in self.geometries],
            [g.to_dict() for g in geoms_run2]
        )

    def test_derive_seed_deterministic(self):
        # derive_seed must be 100% deterministic and produce 31-bit integers
        s1 = derive_seed(101, "environment", 0, "geom_easy_SCS_seed11")
        s2 = derive_seed(101, "environment", 0, "geom_easy_SCS_seed11")
        s3 = derive_seed(101, "environment", 1, "geom_easy_SCS_seed11")
        s4 = derive_seed(202, "environment", 0, "geom_easy_SCS_seed11")

        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, s3)
        self.assertNotEqual(s1, s4)
        self.assertTrue(0 <= s1 < 2**31 - 1)

    def test_test_case_manifest_generation(self):
        test_geoms = [g for g in self.geometries if g.split == SplitRole.TEST]
        test_env_seeds = [9101, 9102, 9103, 9104, 9105]
        cases = build_test_cases(test_geoms, test_env_seeds, protocol_order_seed=424242)

        self.assertEqual(len(cases), 60)
        # All case IDs must be globally unique
        case_ids = [c.case_id for c in cases]
        self.assertEqual(len(case_ids), len(set(case_ids)))

        # Exactly 15 cases per tier
        for t in ("Easy", "Medium", "Hard", "Extreme"):
            tier_cases = [c for c in cases if c.tier == t]
            self.assertEqual(len(tier_cases), 15)

    def test_validation_case_manifest_generation(self):
        val_geoms = [g for g in self.geometries if g.split == SplitRole.VALIDATION]
        val_env_seeds = [5101, 5102]
        cases = build_validation_cases(val_geoms, val_env_seeds, protocol_order_seed=424242)

        self.assertEqual(len(cases), 96)
        case_ids = [c.case_id for c in cases]
        self.assertEqual(len(case_ids), len(set(case_ids)))

        for t in ("Easy", "Medium", "Hard", "Extreme"):
            tier_cases = [c for c in cases if c.tier == t]
            self.assertEqual(len(tier_cases), 24)

    def test_macro_metrics_equal_tier_weighted(self):
        # Create mock tier scorecards with known values
        def mock_scorecard(succ_rate, fail_rate, comp):
            return AggregateMetrics(
                total_episodes=15,
                clean_success_rate=succ_rate,
                safety_failure_rate=fail_rate,
                mean_final_route_completion=comp,
                median_final_route_completion=comp,
                mean_time_to_clean_success_s=50.0,
                success_rate=succ_rate,
                timeout_rate=0.0,
                crash_human_rate=0.0,
                crash_vehicle_rate=fail_rate,
                crash_object_rate=0.0,
                crash_building_rate=0.0,
                crash_sidewalk_rate=0.0,
                out_of_road_rate=0.0,
                unknown_termination_rate=0.0,
                raw_arrival_rate=succ_rate,
                median_time_to_clean_success_s=50.0,
                mean_max_route_completion=comp,
                raw_crash_vehicle_rate=fail_rate,
                raw_crash_object_rate=0.0,
                raw_crash_building_rate=0.0,
                raw_crash_human_rate=0.0,
                raw_crash_sidewalk_rate=0.0,
                raw_out_of_road_rate=0.0,
                raw_any_safety_event_rate=fail_rate,
                mean_speed_kmh=25.0,
                max_speed_kmh=30.0,
                mean_episode_return=1.8,
                mean_episode_steps=500.0,
            )

        cards = {
            "Easy": mock_scorecard(1.0, 0.0, 1.0),
            "Medium": mock_scorecard(0.8, 0.2, 0.9),
            "Hard": mock_scorecard(0.6, 0.4, 0.8),
            "Extreme": mock_scorecard(0.2, 0.8, 0.5),
        }
        macro = compute_macro_metrics(cards)
        # Expected macro clean success: (1.0 + 0.8 + 0.6 + 0.2) / 4 = 2.6 / 4 = 0.65
        self.assertAlmostEqual(macro["macro_clean_success_rate"], 0.65, places=5)
        # Expected macro safety fail: (0.0 + 0.2 + 0.4 + 0.8) / 4 = 1.4 / 4 = 0.35
        self.assertAlmostEqual(macro["macro_safety_failure_rate"], 0.35, places=5)
        # Explicit rejection of geometric-mean mega-score
        self.assertIsNone(macro["geometric_mean_success_score"])

    def test_manifest_lock_detects_mutation(self):
        test_geoms = [g for g in self.geometries if g.split == SplitRole.TEST]
        test_env_seeds = [9101, 9102, 9103, 9104, 9105]
        cases = build_test_cases(test_geoms, test_env_seeds, protocol_order_seed=424242)

        # Baseline manifest hash
        hash_1 = canonical_json_sha256([c.to_dict() for c in cases])

        # Mutate one environment seed in one test case
        mutated_cases = list(cases)
        c0 = mutated_cases[0]
        mutated_cases[0] = EvaluationCase(
            case_id=c0.case_id,
            case_index=c0.case_index,
            protocol_order_index=c0.protocol_order_index,
            split=c0.split,
            tier=c0.tier,
            sequence=c0.sequence,
            candidate_role=c0.candidate_role,
            geometry_generation_seed=c0.geometry_generation_seed,
            geometry_sha256=c0.geometry_sha256,
            environment_seed=9999,  # Mutated seed!
            traffic_density=c0.traffic_density,
            horizon_steps=c0.horizon_steps
        )
        hash_2 = canonical_json_sha256([c.to_dict() for c in mutated_cases])

        # Any mutation must immediately change the lock hash
        self.assertNotEqual(hash_1, hash_2)


if __name__ == "__main__":
    unittest.main()
