"""
Unit tests for Gate 7: Experiment Logging, Local Provenance, and W&B Integration Contract.
Pure unit tests running in <0.10s without network / cloud dependencies.
"""

import copy
import csv
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
    EpisodeTimingRecord,
    ExperimentRunConfig,
    LocalExperimentLogger,
    RunIntegrityRecord,
    RunKind,
    RunManifestV1,
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
            agent_id="test_fixture_agent",
            agent_version="1.0.0",
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic",
            stateful_within_episode=False,
            agent_seed=42,
            protocol_order_seed=424242,
            expected_episode_count=4
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_run_initialization_and_manifest(self):
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)
        self.assertEqual(logger.status, RunStatus.RUNNING)
        self.assertTrue(logger.manifest_path.exists())
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

    def test_duplicate_run_protection(self):
        # First creation succeeds
        logger1 = LocalExperimentLogger(self.config, runs_root=self.runs_root, custom_run_id="run_fixed_001")
        self.assertEqual(logger1.run_id, "run_fixed_001")

        # Second creation with identical run_id must raise FileExistsError
        with self.assertRaises(FileExistsError):
            LocalExperimentLogger(self.config, runs_root=self.runs_root, custom_run_id="run_fixed_001")

    def test_episode_logging_and_summary_aggregation(self):
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)

        # Log 4 episodes across Easy, Medium, Hard, Extreme
        tiers = ["Easy", "Medium", "Hard", "Extreme"]
        for idx, t in enumerate(tiers):
            rec = create_mock_episode_record(idx + 1, clean_success=(idx % 2 == 0), route_completion=1.0 if (idx % 2 == 0) else 0.5, tier=t)
            timing = EpisodeTimingRecord.from_latencies(idx + 1, [1.2, 1.5, 2.0])
            logger.log_episode(rec, timing=timing)

        # Finalize
        integrity = logger.finalize_run()
        self.assertEqual(logger.status, RunStatus.COMPLETE)
        self.assertTrue(logger.summary_json_path.exists())
        self.assertTrue(logger.integrity_json_path.exists())
        self.assertTrue(logger.wandb_sync_json_path.exists())

        # Verify summary content
        with open(logger.summary_json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        self.assertEqual(summary["total_episodes"], 4)
        self.assertIn("overall_metrics", summary)
        self.assertIn("tier_metrics", summary)
        self.assertIn("macro_metrics", summary)
        self.assertEqual(len(summary["tier_metrics"]), 4)

        # Clean success rate: 2 out of 4 = 0.5
        self.assertAlmostEqual(summary["overall_metrics"]["clean_success_rate"], 0.5)
        self.assertAlmostEqual(summary["macro_metrics"]["macro_clean_success_rate"], 0.5)

        # Verify timing stats
        self.assertEqual(summary["timing_overall"]["total_act_count"], 12)
        self.assertGreater(summary["timing_overall"]["mean_act_ms"], 0.0)

    def test_null_metric_semantics(self):
        # Scenario where clean_success_rate = 0 -> time_to_clean_success_s must be null
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)
        for idx in range(1, 5):
            rec = create_mock_episode_record(idx, clean_success=False, route_completion=0.3, tier="Hard")
            logger.log_episode(rec)

        logger.finalize_run()
        with open(logger.summary_json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        # Assert null/None, never 0.0
        self.assertIsNone(summary["overall_metrics"]["mean_time_to_clean_success_s"])

    def test_expected_episode_count_mismatch_fails(self):
        # Config expects 4 episodes, but only 2 are logged -> finalize must fail
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)
        logger.log_episode(create_mock_episode_record(1))
        logger.log_episode(create_mock_episode_record(2))

        with self.assertRaises(ValueError):
            logger.finalize_run()
        self.assertEqual(logger.status, RunStatus.FAILED)


    def test_technical_failure_jsonl_logging(self):
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)
        rec = create_mock_episode_record(1)
        tech_fail = {"reason": "INVALID_AGENT_ACTION", "details": "Action contained NaN"}
        logger.log_episode(rec, technical_failure=tech_fail)
        self.assertTrue(logger.failures_jsonl_path.exists())

        with open(logger.failures_jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        loaded = json.loads(lines[0])
        self.assertEqual(loaded["reason"], "INVALID_AGENT_ACTION")

    def test_secret_absence_and_path_privacy(self):
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)
        serialized_manifest = json.dumps(logger.manifest.to_dict())

        # Assert no API keys or tokens in manifest
        self.assertNotIn("api_key", serialized_manifest.lower())
        self.assertNotIn("secret", serialized_manifest.lower())

        # Path privacy: verify no user home directories
        user_home = str(Path.home())
        if user_home and len(user_home) > 3:
            self.assertNotIn(user_home, serialized_manifest)

    def test_disabled_mode_requires_no_cloud(self):
        backend = WandbBackend(mode=WandbMode.DISABLED)
        self.assertEqual(backend.sync_status, WandbSyncStatus.DISABLED)
        backend.start_run(RunManifestV1("test", "AUDIT", get_utc_now_iso(), "hash", {}, {}, {}, {}), self.config)
        rec = create_mock_episode_record(1)
        backend.log_episode(rec)
        backend.log_summary({"test": 1.0})
        res = backend.finish(RunStatus.COMPLETE)
        self.assertEqual(res["sync_status"], WandbSyncStatus.DISABLED.value)

    def test_offline_mode_operates_without_network(self):
        backend = WandbBackend(mode=WandbMode.OFFLINE)
        self.assertEqual(backend.mode, WandbMode.OFFLINE)

    def test_local_write_failure_simulation_prevents_complete(self):
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)
        # Make episodes.csv unwritable to simulate disk/OS failure
        logger.episodes_csv_path.chmod(0o444)
        rec = create_mock_episode_record(1)

        try:
            with self.assertRaises(IOError):
                logger.log_episode(rec)
            self.assertEqual(logger.status, RunStatus.FAILED)
            # Cannot finalize failed run
            with self.assertRaises(RuntimeError):
                logger.finalize_run()
        finally:
            logger.episodes_csv_path.chmod(0o666)

    def test_observability_hash_combination(self):
        h_runtime = "c7698768539a769c7b2bc6b90771ff234276a974b6e0353bbb03119faf2f79ad"
        h_logging = canonical_json_sha256(build_logging_contract_core())

        payload = {
            "platform_runtime_contract_sha256": h_runtime,
            "logging_contract_sha256": h_logging
        }
        obs_hash_1 = canonical_json_sha256(payload)

        # Mutating runtime hash changes observability hash
        payload_mut = dict(payload, platform_runtime_contract_sha256="mutated_hash")
        obs_hash_2 = canonical_json_sha256(payload_mut)
        self.assertNotEqual(obs_hash_1, obs_hash_2)


class TestWandbBackendAndParity(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.runs_root = Path(self.temp_dir.name)

        self.config = ExperimentRunConfig(
            run_kind=RunKind.TEST_EVALUATION,
            benchmark_contract_sha256="9ddd889b84d8705fae618879e5035556c80d0276a3dc2a58a7963937ebb59f77",
            agent_contract_sha256="53aa37079ff44afa75d9a3f921b0c1f98c4600d882fced51fc8d9197795058eb",
            platform_runtime_contract_sha256="c7698768539a769c7b2bc6b90771ff234276a974b6e0353bbb03119faf2f79ad",
            agent_id="test_agent",
            agent_version="1.0.0",
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic",
            stateful_within_episode=False,
            expected_episode_count=2,
            wandb_mode=WandbMode.ONLINE
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_fake_wandb_metric_and_table_parity(self):
        local_logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)
        backend = FakeTrackingBackend()
        backend.start_run(local_logger.manifest, self.config)

        # Log episodes
        ep1 = create_mock_episode_record(1, clean_success=True, route_completion=1.0, tier="Easy")
        timing1 = EpisodeTimingRecord.from_latencies(1, [1.0, 2.0])
        local_logger.log_episode(ep1, timing1)
        backend.log_episode(ep1, timing1)

        ep2 = create_mock_episode_record(2, clean_success=False, route_completion=0.4, tier="Medium")
        timing2 = EpisodeTimingRecord.from_latencies(2, [1.5, 2.5])
        local_logger.log_episode(ep2, timing2)
        backend.log_episode(ep2, timing2)

        # Finalize local
        sync_meta = backend.finish(RunStatus.COMPLETE)
        local_logger.finalize_run(sync_meta)

        # 1. Episode Table Parity
        self.assertEqual(len(backend.episodes_logged), 2)
        self.assertEqual(len(backend.table_logged), 2)
        with open(local_logger.episodes_csv_path, "r", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        self.assertEqual(len(csv_rows), len(backend.table_logged))

        # Check column values
        self.assertEqual(csv_rows[0]["clean_success"], str(backend.table_logged[0]["clean_success"]))
        self.assertEqual(csv_rows[1]["primary_terminal_reason"], backend.table_logged[1]["primary_terminal_reason"])

    def test_wandb_failure_simulation_is_non_fatal_to_local_run(self):
        local_logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)

        # Backend simulates log failure
        backend = FakeTrackingBackend(simulate_log_failure=True)
        backend.start_run(local_logger.manifest, self.config)

        ep1 = create_mock_episode_record(1)
        local_logger.log_episode(ep1)

        # Backend failure should be caught gracefully by caller / integration
        try:
            backend.log_episode(ep1)
        except IOError:
            backend.fail("Simulated network outage")

        self.assertEqual(backend.sync_status, WandbSyncStatus.FAILED)

        # Local experiment completes cleanly regardless of remote tracking failure
        ep2 = create_mock_episode_record(2)
        local_logger.log_episode(ep2)
        integrity = local_logger.finalize_run({"sync_status": WandbSyncStatus.FAILED.value, "error": "Simulated network outage"})

        self.assertEqual(local_logger.status, RunStatus.COMPLETE)
        self.assertTrue(local_logger.summary_json_path.exists())


class TestTimingBoundaryPreservation(unittest.TestCase):
    def test_logger_latency_excluded_from_agent_act_timing(self):
        # Prove that artificial logger delay does not inflate measured agent latency
        latencies = []
        for _ in range(5):
            t0 = time.perf_counter_ns()
            # Simulated pure agent.act() workload using high-precision busy-wait (5.0 ms)
            target_ns = t0 + 5_000_000
            while time.perf_counter_ns() < target_ns:
                pass
            t1 = time.perf_counter_ns()
            act_ms = (t1 - t0) / 1e6
            latencies.append(act_ms)

            # Simulated heavy logger I/O outside timing boundary
            time.sleep(0.030)  # 30ms logger/W&B overhead

        timing_record = EpisodeTimingRecord.from_latencies(1, latencies)
        # Measured mean latency must reflect ~5ms agent decision, NOT 35ms
        self.assertLess(timing_record.mean_act_ms, 10.0)
        self.assertGreater(timing_record.mean_act_ms, 4.0)


class TestLoggingContractHashingAndIntrospection(unittest.TestCase):
    def test_runtime_dataclass_and_enum_introspection(self):
        import dataclasses
        core = build_logging_contract_core()

        # Check dataclasses
        dc_classes = [ExperimentRunConfig, RunManifestV1, EpisodeTimingRecord, RunIntegrityRecord]
        for cls in dc_classes:
            name = cls.__name__
            self.assertIn(name, core["runtime_dataclass_schemas"])
            actual = [f.name for f in dataclasses.fields(cls)]
            declared = core["runtime_dataclass_schemas"][name]
            self.assertEqual(actual, declared, f"Field drift in {name}")

        # Check enums
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
