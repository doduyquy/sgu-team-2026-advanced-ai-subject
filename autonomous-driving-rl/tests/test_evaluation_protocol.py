"""
Unit tests for Platform V1 scientific evaluation protocol, scenario splits, and seed taxonomy.
Pure, unit-testable tests running in <0.05s without Panda3D / MetaDrive.
"""

import copy
import csv
import json
from pathlib import Path
import unittest

from src.platform import (
    AggregateMetrics,
    EvaluationCase,
    GeometryRecord,
    HorizonPolicy,
    ScenarioSplitV1,
    SeedPlanV1,
    SplitRole,
    assign_geometry_splits,
    build_evaluation_protocol_core,
    build_test_cases,
    build_validation_cases,
    canonical_csv_file_sha256,
    canonical_json_file_sha256,
    canonical_json_sha256,
    compute_macro_metrics,
    compute_manifest_sha256,
    compute_route_aware_horizon,
    derive_seed,
    validate_split_integrity,
)


class TestEvaluationProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project_root = Path(__file__).resolve().parent.parent
        metrics_csv = project_root / "results" / "audits" / "mapsuite" / "candidate_metrics.csv"
        canon_json = project_root / "results" / "audits" / "mapsuite" / "canonical_candidates.json"
        manifest_csv = project_root / "results" / "audits" / "evaluation_protocol" / "geometry_split_manifest.csv"

        with open(metrics_csv, "r", encoding="utf-8") as f:
            cls.metrics = list(csv.DictReader(f))
        with open(canon_json, "r", encoding="utf-8") as f:
            cls.canon_manifest = json.load(f)

        cls.canon_keys = set()
        for tier, cands in cls.canon_manifest.items():
            for c in cands:
                cls.canon_keys.add((c["sequence"], int(c["scenario_seed"])))

        # Load production exact-hash lookup from geometry_split_manifest.csv if available
        geometry_data_lookup = {}
        if manifest_csv.exists():
            with open(manifest_csv, "r", encoding="utf-8") as f:
                manifest_rows = list(csv.DictReader(f))
            for r in manifest_rows:
                seq = r["sequence"]
                s = int(r["geometry_generation_seed"])
                geometry_data_lookup[(seq, s)] = {
                    "geometry_sha256": r["geometry_sha256"],
                    "route_length_m": float(r["route_length_m"]),
                    "block_ids": r["block_ids"],
                    "geometry_hash_source": r["geometry_hash_source"]
                }

        cls.split_salt = "platform-v1-geometry-split-v1"
        cls.geometry_data_lookup = geometry_data_lookup if geometry_data_lookup else None
        cls.geometries = assign_geometry_splits(
            cls.metrics,
            cls.canon_manifest,
            cls.split_salt,
            geometry_data_lookup=cls.geometry_data_lookup
        )

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

        self.assertEqual(test_keys, canon_keys)
        self.assertEqual(len(canon_keys.intersection(train_keys)), 0)
        self.assertEqual(len(canon_keys.intersection(val_keys)), 0)

    def test_zero_leakage_and_integrity_validation(self):
        report = validate_split_integrity(self.geometries, expected_test_identities=self.canon_keys)
        self.assertEqual(report["status"], "VALID")
        self.assertFalse(report["leakage_detected"])
        self.assertTrue(report["split_balance_verified"])
        self.assertTrue(report["canonical_test_identities_verified"])
        self.assertEqual(report["duplicate_geometry_groups_count"], 0)

    def test_canonical_test_identity_validation_enforcement(self):
        # Corrupt test geometries by swapping one canonical for a non-canonical
        corrupted = list(self.geometries)
        test_idx = next(i for i, g in enumerate(corrupted) if g.split == SplitRole.TEST)
        g_orig = corrupted[test_idx]
        corrupted[test_idx] = GeometryRecord(
            geometry_id=g_orig.geometry_id,
            tier=g_orig.tier,
            sequence=g_orig.sequence,
            geometry_generation_seed=999,  # Invalid seed not in canonicals!
            split=g_orig.split,
            candidate_role=g_orig.candidate_role,
            block_ids=g_orig.block_ids,
            route_length_m=g_orig.route_length_m,
            traffic_density=g_orig.traffic_density,
            geometry_sha256=g_orig.geometry_sha256,
            geometry_hash_source=g_orig.geometry_hash_source,
            exact_block_sequence=g_orig.exact_block_sequence
        )
        with self.assertRaises(ValueError):
            validate_split_integrity(corrupted, expected_test_identities=self.canon_keys)

    def test_pairwise_split_leakage_detection(self):
        # Corrupt one train geometry to share the hash of a test geometry
        corrupted = list(self.geometries)
        test_hash = next(g.geometry_sha256 for g in corrupted if g.split == SplitRole.TEST)
        train_idx = next(i for i, g in enumerate(corrupted) if g.split == SplitRole.TRAIN)
        g_orig = corrupted[train_idx]
        corrupted[train_idx] = GeometryRecord(
            geometry_id=g_orig.geometry_id,
            tier=g_orig.tier,
            sequence=g_orig.sequence,
            geometry_generation_seed=g_orig.geometry_generation_seed,
            split=g_orig.split,
            candidate_role=g_orig.candidate_role,
            block_ids=g_orig.block_ids,
            route_length_m=g_orig.route_length_m,
            traffic_density=g_orig.traffic_density,
            geometry_sha256=test_hash,  # Leaked test hash!
            geometry_hash_source=g_orig.geometry_hash_source,
            exact_block_sequence=g_orig.exact_block_sequence
        )
        with self.assertRaises(ValueError):
            validate_split_integrity(corrupted)

    def test_canonical_json_formatting_invariance(self):
        # Semantically identical JSON objects with different whitespace/newlines
        data_compact = '{"a":1,"b":[2,3],"c":{"d":"value"}}'
        data_formatted = '{\n  "c": {\n    "d": "value"\n  },\n  "a": 1,\n  "b": [\n    2,\n    3\n  ]\n}\r\n'

        obj1 = json.loads(data_compact)
        obj2 = json.loads(data_formatted)

        h1 = canonical_json_sha256(obj1)
        h2 = canonical_json_sha256(obj2)
        self.assertEqual(h1, h2)

    def test_canonical_csv_formatting_invariance(self):
        # Test CRLF vs LF invariance in CSV content hashing
        import tempfile
        content_lf = "col_a,col_b\nval1,val2\nval3,val4\n"
        content_crlf = "col_a,col_b\r\nval1,val2\r\nval3,val4\r\n"

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False) as f_lf:
            f_lf.write(content_lf)
            path_lf = Path(f_lf.name)

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False) as f_crlf:
            f_crlf.write(content_crlf)
            path_crlf = Path(f_crlf.name)

        try:
            h_lf = canonical_csv_file_sha256(path_lf)
            h_crlf = canonical_csv_file_sha256(path_crlf)
            self.assertEqual(h_lf, h_crlf)
        finally:
            path_lf.unlink(missing_ok=True)
            path_crlf.unlink(missing_ok=True)

    def test_exact_block_geometry_hashing(self):
        block_seq_1 = [
            {"id": "I", "pre_block_socket_index": None},
            {"id": "S", "length": 50.0, "pre_block_socket_index": "0I-socket0"}
        ]
        block_seq_2 = copy.deepcopy(block_seq_1)

        # Same geometry content must yield identical hash
        h1 = canonical_json_sha256(block_seq_1)
        h2 = canonical_json_sha256(block_seq_2)
        self.assertEqual(h1, h2)

        # Mutating one parameter must change hash
        block_seq_mutated = copy.deepcopy(block_seq_1)
        block_seq_mutated[1]["length"] = 55.0
        h_mut = canonical_json_sha256(block_seq_mutated)
        self.assertNotEqual(h1, h_mut)

    def test_deterministic_split_reproducibility(self):
        geoms_run2 = assign_geometry_splits(
            self.metrics,
            self.canon_manifest,
            self.split_salt,
            geometry_data_lookup=self.geometry_data_lookup
        )
        self.assertEqual(
            [g.to_dict() for g in self.geometries],
            [g.to_dict() for g in geoms_run2]
        )

    def test_derive_seed_deterministic(self):
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
        case_ids = [c.case_id for c in cases]
        self.assertEqual(len(case_ids), len(set(case_ids)))

        for t in ("Easy", "Medium", "Hard", "Extreme"):
            tier_cases = [c for c in cases if c.tier == t]
            self.assertEqual(len(tier_cases), 15)

        # Check Gate-3 horizon helper reuse
        for c in cases:
            geom = next(g for g in test_geoms if g.sequence == c.sequence and g.geometry_generation_seed == c.geometry_generation_seed)
            expected_horizon = compute_route_aware_horizon(geom.route_length_m)
            self.assertEqual(c.horizon_steps, expected_horizon)
            # Check traffic density propagation
            self.assertEqual(c.traffic_density, geom.traffic_density)

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
        self.assertAlmostEqual(macro["macro_clean_success_rate"], 0.65, places=5)
        self.assertAlmostEqual(macro["macro_safety_failure_rate"], 0.35, places=5)
        self.assertIsNone(macro["geometric_mean_success_score"])

    def test_production_manifest_hashing_detects_mutation(self):
        test_geoms = [g for g in self.geometries if g.split == SplitRole.TEST]
        test_env_seeds = [9101, 9102, 9103, 9104, 9105]
        cases = build_test_cases(test_geoms, test_env_seeds, protocol_order_seed=424242)

        # Baseline hash via production helper compute_manifest_sha256
        hash_1 = compute_manifest_sha256(cases)

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
        hash_2 = compute_manifest_sha256(mutated_cases)

        self.assertNotEqual(hash_1, hash_2)

    def test_committed_manifest_integrity(self):
        project_root = Path(__file__).resolve().parent.parent
        manifest_csv = project_root / "results" / "audits" / "evaluation_protocol" / "geometry_split_manifest.csv"
        if not manifest_csv.exists():
            self.skipTest("geometry_split_manifest.csv does not exist yet")

        with open(manifest_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 240)
        train_count = sum(1 for r in rows if r["split"].upper() == "TRAIN")
        val_count = sum(1 for r in rows if r["split"].upper() == "VALIDATION")
        test_count = sum(1 for r in rows if r["split"].upper() == "TEST")

        self.assertEqual(train_count, 180)
        self.assertEqual(val_count, 48)
        self.assertEqual(test_count, 12)

        # Ensure all 240 true hashes are distinct (0 duplicate geometry groups)
        hashes = [r["geometry_sha256"] for r in rows]
        self.assertEqual(len(hashes), len(set(hashes)))

    def test_evaluation_protocol_core_rule_change_alters_hash(self):
        test_geoms = [g for g in self.geometries if g.split == SplitRole.TEST]
        val_geoms = [g for g in self.geometries if g.split == SplitRole.VALIDATION]
        test_cases = build_test_cases(test_geoms, [9101, 9102])
        val_cases = build_validation_cases(val_geoms, [5101])

        core_default = build_evaluation_protocol_core(test_cases, val_cases)
        h_default = canonical_json_sha256(core_default)

        # Mutate paired_evaluation rule
        core_mutated = build_evaluation_protocol_core(
            test_cases,
            val_cases,
            custom_rules={"paired_evaluation": "Mutated paired evaluation rule text."}
        )
        h_mutated = canonical_json_sha256(core_mutated)

        self.assertNotEqual(h_default, h_mutated)

    def test_case_builders_use_supplied_horizon_policy(self):
        test_geoms = [g for g in self.geometries if g.split == SplitRole.TEST]
        policy_default = HorizonPolicy(safety_margin=1.5)
        policy_larger = HorizonPolicy(safety_margin=2.0)

        cases_default = build_test_cases(test_geoms, [9101], horizon_policy=policy_default)
        cases_larger = build_test_cases(test_geoms, [9101], horizon_policy=policy_larger)

        # Safety margin 2.0 must produce larger horizon steps than 1.5, proving no hardcoded 1.5
        for cd, cl in zip(cases_default, cases_larger):
            self.assertGreater(cl.horizon_steps, cd.horizon_steps)

    def test_episode_spec_v1_values_reproduce_locked_horizons(self):
        project_root = Path(__file__).resolve().parent.parent
        ep_spec_path = project_root / "configs" / "platform" / "episode_spec_v1.json"
        manifest_csv = project_root / "results" / "audits" / "evaluation_protocol" / "test_case_manifest.csv"

        if not ep_spec_path.exists() or not manifest_csv.exists():
            self.skipTest("episode_spec_v1.json or test_case_manifest.csv missing")

        with open(ep_spec_path, "r", encoding="utf-8") as f:
            ep_spec_data = json.load(f)
        policy = HorizonPolicy.from_episode_spec(ep_spec_data)

        test_geoms = [g for g in self.geometries if g.split == SplitRole.TEST]
        cases = build_test_cases(test_geoms, [9101, 9102, 9103, 9104, 9105], horizon_policy=policy)

        with open(manifest_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            locked_horizons = {r["case_id"]: int(r["horizon_steps"]) for r in reader}

        for c in cases:
            self.assertEqual(c.horizon_steps, locked_horizons[c.case_id])


if __name__ == "__main__":
    unittest.main()
