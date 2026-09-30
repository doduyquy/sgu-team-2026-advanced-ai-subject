"""
Unit tests for Gate 7: Experiment Logging, Local Provenance, and W&B Integration Contract.
Pure unit tests running in <0.20s without network / cloud dependencies.
"""

import copy
import csv
import dataclasses
import json
import math
import os
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from src.platform.episode import EpisodeOutcome, TerminalReason
from src.platform.experiment_logging import (
    EpisodeLogRowV1,
    EpisodeTimingRecord,
    ExperimentRunConfig,
    LocalExperimentLogger,
    RunIntegrityRecord,
    RunKind,
    RunManifestV1,
    RunStateV1,
    RunStatus,
    WandbMode,
    WandbSyncStatus,
    build_logging_contract_core,
    capture_environment_provenance,
    capture_git_provenance,
    get_utc_now_iso,
)
from src.platform.metrics import EpisodeRecord
from src.platform.protocol import (
    canonical_csv_file_sha256,
    canonical_json_file_sha256,
    canonical_json_sha256,
)
from src.platform.wandb_backend import FakeTrackingBackend, WandbBackend


def create_mock_episode_record(
    index: int,
    clean_success: bool = True,
    route_completion: float = 1.0,
    tier: str = "Easy"
) -> EpisodeRecord:
    """Constructs a deterministic mock EpisodeRecord for test fixtures."""
    return EpisodeRecord(
        tier=tier,
        sequence="SCS" if tier == "Easy" else "SCXCS",
        scenario_seed=11,
        terminated=True,
        truncated=False,
        primary_reason=TerminalReason.SUCCESS if clean_success else TerminalReason.CRASH_VEHICLE,
        raw_arrival=clean_success,
        clean_success=clean_success,
        final_route_completion=route_completion,
        max_route_completion=route_completion,
        raw_crash_vehicle=not clean_success,
        raw_crash_object=False,
        raw_crash_building=False,
        raw_crash_human=False,
        raw_crash_sidewalk=False,
        raw_out_of_road=False,
        episode_steps=400 if clean_success else 200,
        simulation_time_s=40.0 if clean_success else 20.0,
        mean_speed_kmh=31.5,
        max_speed_kmh=42.0,
        episode_return=85.0 if clean_success else 15.0,
        time_to_clean_success_s=40.0 if clean_success else None,
        agent_seed=42
    )


class TestLoggingContractCore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.runs_root = Path(self.temp_dir.name)

        self.config = ExperimentRunConfig(
            run_kind=RunKind.TEST_EVALUATION,
            benchmark_contract_sha256="9ddd889b84d8705fae618879e5035556c80d0276a3dc2a58a7963937ebb59f77",
            agent_contract_sha256="53aa37079ff44afa75d9a3f921b0c1f98c4600d882fced51fc8d9197795058eb",
            platform_runtime_contract_sha256="c7698768539a769c7b2bc6b90771ff234276a974b6e0353bbb03119faf2f79ad",
            logging_contract_sha256="02d480cbb78e876b4c91a4eb16d831106de4199e369c9b53b50f91b142636f9f",
            platform_observability_contract_sha256="335d96590dd9bbcaf2eb1077d497ebb155acf3abae9bf299bb47087a8dbb81d8",
            agent_id="test_fixture_agent",
            agent_version="1.0.0",
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic",
            stateful_within_episode=False,
            agent_seed=42,
            protocol_order_seed=424242,
            expected_episode_count=4,
            allow_dirty_worktree_override=True
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_run_initialization_and_manifest(self):
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)
        self.assertEqual(logger.status, RunStatus.RUNNING)
        self.assertTrue(logger.manifest_path.exists())
        self.assertTrue(logger.run_state_path.exists())
        self.assertTrue(logger.episodes_csv_path.exists())
        self.assertTrue(logger.timing_csv_path.exists())

        # Inspect manifest
        with open(logger.manifest_path, "r", encoding="utf-8") as f:
            manifest_dict = json.load(f)

        self.assertEqual(manifest_dict["run_id"], logger.run_id)
        self.assertEqual(manifest_dict["run_kind"], "TEST_EVALUATION")
        self.assertIn("git_provenance", manifest_dict)
        self.assertIn("environment_provenance", manifest_dict)
        self.assertIn("platform_contracts", manifest_dict)

        # Check Gate-7 contract hashes in manifest
        contracts = manifest_dict["platform_contracts"]
        self.assertIn("logging_contract_sha256", contracts)
        self.assertIn("platform_observability_contract_sha256", contracts)

    def test_duplicate_run_protection(self):
        logger1 = LocalExperimentLogger(self.config, runs_root=self.runs_root, custom_run_id="run_fixed_001")
        self.assertEqual(logger1.run_id, "run_fixed_001")

        with self.assertRaises(FileExistsError):
            LocalExperimentLogger(self.config, runs_root=self.runs_root, custom_run_id="run_fixed_001")

    def test_durable_run_state_lifecycle(self):
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root, custom_run_id="run_state_test")
        self.assertTrue(logger.run_state_path.exists())

        # Initial state is RUNNING
        with open(logger.run_state_path, "r", encoding="utf-8") as f:
            st1 = json.load(f)
        self.assertEqual(st1["status"], "RUNNING")
        self.assertIsNone(st1["finished_at_utc"])
        self.assertEqual(st1["recorded_episode_count"], 0)

        # Log 4 episodes
        for idx in range(1, 5):
            rec = create_mock_episode_record(idx)
            logger.log_episode(rec, episode_index=idx, protocol_order_index=idx, case_id=f"case_{idx}", split="TEST", environment_seed=9101, geometry_generation_seed=11, horizon_steps=1000)

        # State updated during run
        with open(logger.run_state_path, "r", encoding="utf-8") as f:
            st2 = json.load(f)
        self.assertEqual(st2["recorded_episode_count"], 4)

        logger.finalize_run()
        # Final state is COMPLETE with finished timestamp
        with open(logger.run_state_path, "r", encoding="utf-8") as f:
            st3 = json.load(f)
        self.assertEqual(st3["status"], "COMPLETE")
        self.assertIsNotNone(st3["finished_at_utc"])

    def test_canonical_dirty_worktree_policy(self):
        # Without allow_dirty_worktree_override, benchmark on dirty tree must fail loudly
        clean_config = ExperimentRunConfig(
            run_kind=RunKind.TEST_EVALUATION,
            benchmark_contract_sha256="hash",
            agent_contract_sha256="hash",
            platform_runtime_contract_sha256="hash",
            logging_contract_sha256="hash",
            platform_observability_contract_sha256="hash",
            agent_id="test",
            agent_version="1.0",
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic",
            stateful_within_episode=False,
            allow_dirty_worktree_override=False  # No override
        )

        git_prov = capture_git_provenance()
        if git_prov.get("git_worktree_dirty"):
            with self.assertRaises(RuntimeError):
                LocalExperimentLogger(clean_config, runs_root=self.runs_root)

    def test_episode_logging_and_summary_aggregation(self):
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)

        tiers = ["Easy", "Medium", "Hard", "Extreme"]
        for idx, t in enumerate(tiers, 1):
            rec = create_mock_episode_record(idx, clean_success=(idx % 2 == 1), route_completion=1.0 if (idx % 2 == 1) else 0.5, tier=t)
            timing = EpisodeTimingRecord.from_latencies(idx, [1.25, 1.5, 2.0])
            logger.log_episode(
                rec,
                timing=timing,
                episode_index=idx,
                protocol_order_index=idx,
                case_id=f"case_{t}_{idx}",
                split="TEST",
                environment_seed=9101,
                geometry_generation_seed=11,
                horizon_steps=1000
            )

        integrity = logger.finalize_run()
        self.assertEqual(logger.status, RunStatus.COMPLETE)
        self.assertTrue(logger.summary_json_path.exists())
        self.assertTrue(logger.integrity_json_path.exists())
        self.assertTrue(logger.wandb_sync_json_path.exists())

        with open(logger.summary_json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        self.assertEqual(summary["total_episodes"], 4)
        self.assertAlmostEqual(summary["overall_metrics"]["clean_success_rate"], 0.5)
        self.assertAlmostEqual(summary["macro_metrics"]["macro_clean_success_rate"], 0.5)

    def test_null_metric_semantics(self):
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)
        for idx in range(1, 5):
            rec = create_mock_episode_record(idx, clean_success=False, route_completion=0.3, tier="Hard")
            logger.log_episode(
                rec,
                episode_index=idx,
                protocol_order_index=idx,
                case_id=f"case_{idx}",
                split="TEST",
                environment_seed=9101,
                geometry_generation_seed=11,
                horizon_steps=1000
            )

        logger.finalize_run()
        with open(logger.summary_json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        self.assertIsNone(summary["overall_metrics"]["mean_time_to_clean_success_s"])

    def test_expected_episode_count_mismatch_fails(self):
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)
        rec = create_mock_episode_record(1)
        logger.log_episode(rec, episode_index=1, protocol_order_index=1, case_id="c1", split="TEST", environment_seed=9101, geometry_generation_seed=11, horizon_steps=1000)

        with self.assertRaises(ValueError):
            logger.finalize_run()
        self.assertEqual(logger.status, RunStatus.FAILED)

    def test_technical_failure_jsonl_logging(self):
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)
        rec = create_mock_episode_record(1)
        tech_fail = {"reason": "INVALID_AGENT_ACTION", "details": "Action contained NaN"}
        logger.log_episode(rec, episode_index=1, protocol_order_index=1, case_id="c1", split="TEST", environment_seed=9101, geometry_generation_seed=11, horizon_steps=1000, technical_failure=tech_fail)
        self.assertTrue(logger.failures_jsonl_path.exists())

        with open(logger.failures_jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        loaded = json.loads(lines[0])
        self.assertEqual(loaded["reason"], "INVALID_AGENT_ACTION")

    def test_secret_absence_and_path_privacy(self):
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)
        serialized_manifest = json.dumps(logger.manifest.to_dict())

        self.assertNotIn("api_key", serialized_manifest.lower())
        self.assertNotIn("secret", serialized_manifest.lower())

        user_home = str(Path.home())
        if user_home and len(user_home) > 3:
            self.assertNotIn(user_home, serialized_manifest)

    def test_disabled_mode_requires_no_cloud(self):
        backend = WandbBackend(mode=WandbMode.DISABLED)
        self.assertEqual(backend.sync_status, WandbSyncStatus.DISABLED)
        backend.start_run(RunManifestV1("test", "AUDIT", get_utc_now_iso(), "hash", {}, {}, {}, {}), self.config)
        row = EpisodeLogRowV1.from_episode(create_mock_episode_record(1), 1, 1, "c1", "TEST", 11, 9101, 1000)
        backend.log_episode(row)
        backend.log_summary({"test": 1.0})
        res = backend.finish(RunStatus.COMPLETE)
        self.assertEqual(res["sync_status"], WandbSyncStatus.DISABLED.value)

    def test_offline_mode_operates_without_network(self):
        backend = WandbBackend(mode=WandbMode.OFFLINE)
        self.assertEqual(backend.mode, WandbMode.OFFLINE)

    def test_local_write_failure_simulation_prevents_complete(self):
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)
        logger.episodes_csv_path.chmod(0o444)
        rec = create_mock_episode_record(1)

        try:
            with self.assertRaises(IOError):
                logger.log_episode(rec, episode_index=1, protocol_order_index=1, case_id="c1", split="TEST", environment_seed=9101, geometry_generation_seed=11, horizon_steps=1000)
            self.assertEqual(logger.status, RunStatus.FAILED)
            with self.assertRaises(RuntimeError):
                logger.finalize_run()
        finally:
            logger.episodes_csv_path.chmod(0o666)


class TestWandbBackendAndParity(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.runs_root = Path(self.temp_dir.name)

        self.config = ExperimentRunConfig(
            run_kind=RunKind.TEST_EVALUATION,
            benchmark_contract_sha256="9ddd889b84d8705fae618879e5035556c80d0276a3dc2a58a7963937ebb59f77",
            agent_contract_sha256="53aa37079ff44afa75d9a3f921b0c1f98c4600d882fced51fc8d9197795058eb",
            platform_runtime_contract_sha256="c7698768539a769c7b2bc6b90771ff234276a974b6e0353bbb03119faf2f79ad",
            logging_contract_sha256="02d480cbb78e876b4c91a4eb16d831106de4199e369c9b53b50f91b142636f9f",
            platform_observability_contract_sha256="335d96590dd9bbcaf2eb1077d497ebb155acf3abae9bf299bb47087a8dbb81d8",
            agent_id="test_agent",
            agent_version="1.0.0",
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic",
            stateful_within_episode=False,
            expected_episode_count=2,
            wandb_mode=WandbMode.ONLINE,
            allow_dirty_worktree_override=True
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_canonical_episode_log_row_unified_consumption(self):
        local_logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)
        backend = FakeTrackingBackend()
        backend.start_run(local_logger.manifest, self.config)

        # Log Episode 1 and Episode 2 using canonical row
        ep1 = create_mock_episode_record(1, clean_success=True, route_completion=1.0, tier="Easy")
        timing1 = EpisodeTimingRecord.from_latencies(1, [1.0, 2.0])
        row1 = local_logger.log_episode(ep1, timing=timing1, episode_index=1, protocol_order_index=1, case_id="test/Easy/SCS/1/9101", split="TEST", environment_seed=9101, geometry_generation_seed=11, horizon_steps=1049)
        backend.log_episode(row1, timing=timing1)

        ep2 = create_mock_episode_record(2, clean_success=False, route_completion=0.4, tier="Medium")
        timing2 = EpisodeTimingRecord.from_latencies(2, [1.5, 2.5])
        row2 = local_logger.log_episode(ep2, timing=timing2, episode_index=2, protocol_order_index=2, case_id="test/Medium/SCXCS/1/9101", split="TEST", environment_seed=9101, geometry_generation_seed=11, horizon_steps=1572)
        backend.log_episode(row2, timing=timing2)

        sync_meta = backend.finish(RunStatus.COMPLETE)
        local_logger.finalize_run(sync_meta)

        # 1. Step axis parity: event step == episode_index
        self.assertEqual(backend.episodes_logged[0]["eval/episode_index"], 1)
        self.assertEqual(backend.episodes_logged[1]["eval/episode_index"], 2)
        self.assertEqual(backend.episodes_logged[0]["eval/protocol_order_index"], 1)
        self.assertEqual(backend.episodes_logged[1]["eval/protocol_order_index"], 2)

        # 2. Table row parity against CSV rows
        with open(local_logger.episodes_csv_path, "r", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))

        self.assertEqual(len(csv_rows), len(backend.table_logged))
        self.assertEqual(csv_rows[0]["case_id"], "test/Easy/SCS/1/9101")
        self.assertEqual(csv_rows[0]["horizon_steps"], "1049")
        self.assertEqual(csv_rows[0]["protocol_order_index"], "1")

        # 3. Reject logging after finish
        with self.assertRaises(RuntimeError):
            backend.log_episode(row1)

    def test_wandb_failure_matrix_simulation(self):
        # 1. Init failure
        b_init_fail = FakeTrackingBackend(simulate_init_failure=True)
        with self.assertRaises(ConnectionError):
            b_init_fail.start_run(RunManifestV1("r", "AUDIT", get_utc_now_iso(), "h", {}, {}, {}, {}), self.config)
        self.assertEqual(b_init_fail.sync_status, WandbSyncStatus.FAILED)

        # 2. Log failure
        b_log_fail = FakeTrackingBackend(simulate_log_failure=True)
        b_log_fail.start_run(RunManifestV1("r", "AUDIT", get_utc_now_iso(), "h", {}, {}, {}, {}), self.config)
        row = EpisodeLogRowV1.from_episode(create_mock_episode_record(1), 1, 1, "c1", "TEST", 11, 9101, 1000)
        with self.assertRaises(IOError):
            b_log_fail.log_episode(row)
        self.assertEqual(b_log_fail.sync_status, WandbSyncStatus.FAILED)

        # 3. Finish failure
        b_finish_fail = FakeTrackingBackend(simulate_finish_failure=True)
        b_finish_fail.start_run(RunManifestV1("r", "AUDIT", get_utc_now_iso(), "h", {}, {}, {}, {}), self.config)
        with self.assertRaises(RuntimeError):
            b_finish_fail.finish(RunStatus.COMPLETE)
        self.assertEqual(b_finish_fail.sync_status, WandbSyncStatus.FAILED)


class TestTimingPrecisionAndWeightedAggregation(unittest.TestCase):
    def test_raw_floating_point_precision_preserved(self):
        # Preserve full precision without rounding
        raw_latencies = [1.23456789, 2.34567891, 3.45678912]
        rec = EpisodeTimingRecord.from_latencies(1, raw_latencies)
        self.assertAlmostEqual(rec.mean_act_ms, float(np.mean(raw_latencies)), places=7)

    def test_weighted_mean_act_latency(self):
        # Episode 1: 10 acts at 5.0ms (total = 50ms)
        # Episode 2: 90 acts at 1.0ms (total = 90ms)
        # Weighted mean: 140ms / 100 acts = 1.40ms
        # Unweighted mean: (5.0 + 1.0) / 2 = 3.0ms (WRONG)
        t1 = EpisodeTimingRecord.from_latencies(1, [5.0] * 10)
        t2 = EpisodeTimingRecord.from_latencies(2, [1.0] * 90)

        total_acts = t1.act_count + t2.act_count
        weighted_mean = (t1.total_act_ms + t2.total_act_ms) / total_acts
        self.assertEqual(total_acts, 100)
        self.assertAlmostEqual(weighted_mean, 1.40, places=5)

    def test_logger_latency_excluded_from_agent_act_timing(self):
        latencies = []
        for _ in range(5):
            t0 = time.perf_counter_ns()
            target_ns = t0 + 5_000_000
            while time.perf_counter_ns() < target_ns:
                pass
            t1 = time.perf_counter_ns()
            act_ms = (t1 - t0) / 1e6
            latencies.append(act_ms)
            time.sleep(0.030)

        timing_record = EpisodeTimingRecord.from_latencies(1, latencies)
        self.assertLess(timing_record.mean_act_ms, 10.0)
        self.assertGreater(timing_record.mean_act_ms, 4.0)


class TestLoggingContractHashingAndIntrospection(unittest.TestCase):
    def test_runtime_dataclass_and_enum_introspection(self):
        core = build_logging_contract_core()

        # Check all 6 dataclasses
        dc_classes = [ExperimentRunConfig, RunManifestV1, RunStateV1, EpisodeLogRowV1, EpisodeTimingRecord, RunIntegrityRecord]
        for cls in dc_classes:
            name = cls.__name__
            self.assertIn(name, core["runtime_dataclass_schemas"])
            actual = [f.name for f in dataclasses.fields(cls)]
            declared = core["runtime_dataclass_schemas"][name]
            self.assertEqual(actual, declared, f"Field drift in {name}")

        # Check all 4 enums
        enum_classes = [RunKind, RunStatus, WandbMode, WandbSyncStatus]
        for ecls in enum_classes:
            ename = ecls.__name__
            self.assertIn(ename, core["runtime_enum_schemas"])
            e_actual = [e.value for e in ecls]
            e_declared = core["runtime_enum_schemas"][ename]
            self.assertEqual(e_actual, e_declared, f"Enum drift in {ename}")

    def test_logging_contract_hash_mutation_sensitivity(self):
        core_default = build_logging_contract_core()
        h_default = canonical_json_sha256(core_default)

        # Mutate timing schema
        core_mut = build_logging_contract_core(custom_latency_schema={"nominal_realtime_budget_ms": 50.0})
        h_mut = canonical_json_sha256(core_mut)
        self.assertNotEqual(h_default, h_mut)

        # Mutate secret policy
        core_sec = build_logging_contract_core(custom_secret_policy={"persistence_rule": "MUTATED_RULE"})
        h_sec = canonical_json_sha256(core_sec)
        self.assertNotEqual(h_default, h_sec)


if __name__ == "__main__":
    unittest.main()
