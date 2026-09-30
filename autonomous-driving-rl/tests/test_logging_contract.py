"""
Unit tests for Gate 7: Experiment Logging, Local Provenance, and W&B Integration Contract.
Comprehensive unit test suite covering:
  A. Benchmark metadata strictness (no fallbacks for TEST/VALIDATION)
  B. Seed semantics (agent_seed=0 preservation)
  C. Backend contract (EpisodeLogRowV1 only, raw EpisodeRecord/dict rejected)
  D. MetaDrive exact pin verification (version 0.4.3 & commit 85e5dad...)
  E. Finalize_run keyword-only API and single authoritative summary
  F. W&B operational modes and environment restoration
  G. Failure isolation (W&B init/log/finish failure leaves local run COMPLETE)
  H. Table parity (shared fields value-equivalence + timing projection)
  I. Cryptographic hash sensitivity across all schema dimensions
  J. Error message sanitization & credential protection
  K. Run integrity hashing including wandb_sync.json
  L. Interruption semantics
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
from unittest.mock import patch

import numpy as np

from src.platform.episode import EpisodeOutcome, TerminalReason
from src.platform.experiment_logging import (
    EPISODE_CSV_COLUMNS,
    PINNED_METADRIVE_COMMIT,
    PINNED_METADRIVE_VERSION,
    WANDB_DYNAMIC_KEY_TEMPLATES,
    WANDB_STATIC_EPISODE_EVENT_KEYS,
    WANDB_SUMMARY_MAPPING_RULES,
    WANDB_TABLE_COLUMNS,
    WANDB_TIMING_EVENT_KEYS,
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
    sanitize_error_message,
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
    tier: str = "Easy",
    agent_seed: int = 42
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
        agent_seed=agent_seed
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

        with open(logger.manifest_path, "r", encoding="utf-8") as f:
            manifest_dict = json.load(f)

        self.assertEqual(manifest_dict["run_id"], logger.run_id)
        self.assertEqual(manifest_dict["run_kind"], "TEST_EVALUATION")
        self.assertIn("git_provenance", manifest_dict)
        self.assertIn("environment_provenance", manifest_dict)
        self.assertIn("platform_contracts", manifest_dict)

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

        with open(logger.run_state_path, "r", encoding="utf-8") as f:
            st1 = json.load(f)
        self.assertEqual(st1["status"], "RUNNING")
        self.assertIsNone(st1["finished_at_utc"])
        self.assertEqual(st1["recorded_episode_count"], 0)

        for idx in range(1, 5):
            rec = create_mock_episode_record(idx)
            logger.log_episode(rec, episode_index=idx, protocol_order_index=idx, case_id=f"case_{idx}", split="TEST", environment_seed=9101, geometry_generation_seed=11, horizon_steps=1000)

        with open(logger.run_state_path, "r", encoding="utf-8") as f:
            st2 = json.load(f)
        self.assertEqual(st2["recorded_episode_count"], 4)

        summary = logger.prepare_summary()
        logger.finalize_run(summary_payload=summary)

        with open(logger.run_state_path, "r", encoding="utf-8") as f:
            st3 = json.load(f)
        self.assertEqual(st3["status"], "COMPLETE")
        self.assertIsNotNone(st3["finished_at_utc"])

    def test_canonical_dirty_worktree_policy(self):
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
            allow_dirty_worktree_override=False
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

        summary = logger.prepare_summary()
        integrity = logger.finalize_run(summary_payload=summary)
        self.assertEqual(logger.status, RunStatus.COMPLETE)
        self.assertTrue(logger.summary_json_path.exists())
        self.assertTrue(logger.integrity_json_path.exists())
        self.assertTrue(logger.wandb_sync_json_path.exists())

        with open(logger.summary_json_path, "r", encoding="utf-8") as f:
            loaded_summary = json.load(f)

        self.assertEqual(loaded_summary["total_episodes"], 4)
        self.assertAlmostEqual(loaded_summary["overall_metrics"]["clean_success_rate"], 0.5)
        self.assertAlmostEqual(loaded_summary["macro_metrics"]["macro_clean_success_rate"], 0.5)

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

        summary = logger.prepare_summary()
        logger.finalize_run(summary_payload=summary)
        with open(logger.summary_json_path, "r", encoding="utf-8") as f:
            loaded_summary = json.load(f)

        self.assertIsNone(loaded_summary["overall_metrics"]["mean_time_to_clean_success_s"])

    def test_expected_episode_count_mismatch_fails(self):
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)
        rec = create_mock_episode_record(1)
        logger.log_episode(rec, episode_index=1, protocol_order_index=1, case_id="c1", split="TEST", environment_seed=9101, geometry_generation_seed=11, horizon_steps=1000)

        summary = logger.prepare_summary()
        with self.assertRaises(ValueError):
            logger.finalize_run(summary_payload=summary)
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
        backend.log_summary({"overall_metrics": {"clean_success_rate": 1.0}})
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
                logger.finalize_run(summary_payload={"overall_metrics": {}})
        finally:
            logger.episodes_csv_path.chmod(0o666)


class TestBenchmarkMetadataStrictness(unittest.TestCase):
    """Blocker 2: Strict Benchmark Metadata Requirements."""
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.runs_root = Path(self.temp_dir.name)
        self.test_config = ExperimentRunConfig(
            run_kind=RunKind.TEST_EVALUATION,
            benchmark_contract_sha256="dummy",
            agent_contract_sha256="dummy",
            platform_runtime_contract_sha256="dummy",
            logging_contract_sha256="dummy",
            platform_observability_contract_sha256="dummy",
            agent_id="test",
            agent_version="1.0",
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic",
            stateful_within_episode=False,
            allow_dirty_worktree_override=True
        )
        self.val_config = ExperimentRunConfig(
            run_kind=RunKind.VALIDATION_EVALUATION,
            benchmark_contract_sha256="dummy",
            agent_contract_sha256="dummy",
            platform_runtime_contract_sha256="dummy",
            logging_contract_sha256="dummy",
            platform_observability_contract_sha256="dummy",
            agent_id="test",
            agent_version="1.0",
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic",
            stateful_within_episode=False,
            allow_dirty_worktree_override=True
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_test_evaluation_missing_fields_rejected(self):
        logger = LocalExperimentLogger(self.test_config, runs_root=self.runs_root)
        rec = create_mock_episode_record(1)

        # Missing episode_index
        with self.assertRaises(ValueError):
            logger.log_episode(rec, episode_index=None, protocol_order_index=1, case_id="c1", split="TEST", geometry_generation_seed=11, environment_seed=9101, horizon_steps=1000)

        # Invalid episode_index <= 0
        with self.assertRaises(ValueError):
            logger.log_episode(rec, episode_index=0, protocol_order_index=1, case_id="c1", split="TEST", geometry_generation_seed=11, environment_seed=9101, horizon_steps=1000)

        # Missing protocol_order_index
        with self.assertRaises(ValueError):
            logger.log_episode(rec, episode_index=1, protocol_order_index=None, case_id="c1", split="TEST", geometry_generation_seed=11, environment_seed=9101, horizon_steps=1000)

        # Invalid protocol_order_index <= 0
        with self.assertRaises(ValueError):
            logger.log_episode(rec, episode_index=1, protocol_order_index=-1, case_id="c1", split="TEST", geometry_generation_seed=11, environment_seed=9101, horizon_steps=1000)

        # Missing case_id
        with self.assertRaises(ValueError):
            logger.log_episode(rec, episode_index=1, protocol_order_index=1, case_id="", split="TEST", geometry_generation_seed=11, environment_seed=9101, horizon_steps=1000)

        # Missing split
        with self.assertRaises(ValueError):
            logger.log_episode(rec, episode_index=1, protocol_order_index=1, case_id="c1", split=None, geometry_generation_seed=11, environment_seed=9101, horizon_steps=1000)

        # Wrong split (VALIDATION instead of TEST for TEST_EVALUATION)
        with self.assertRaises(ValueError):
            logger.log_episode(rec, episode_index=1, protocol_order_index=1, case_id="c1", split="VALIDATION", geometry_generation_seed=11, environment_seed=9101, horizon_steps=1000)

        # Missing geometry_generation_seed
        with self.assertRaises(ValueError):
            logger.log_episode(rec, episode_index=1, protocol_order_index=1, case_id="c1", split="TEST", geometry_generation_seed=None, environment_seed=9101, horizon_steps=1000)

        # Missing environment_seed
        with self.assertRaises(ValueError):
            logger.log_episode(rec, episode_index=1, protocol_order_index=1, case_id="c1", split="TEST", geometry_generation_seed=11, environment_seed=None, horizon_steps=1000)

        # Missing horizon_steps
        with self.assertRaises(ValueError):
            logger.log_episode(rec, episode_index=1, protocol_order_index=1, case_id="c1", split="TEST", geometry_generation_seed=11, environment_seed=9101, horizon_steps=None)

        # Invalid horizon_steps <= 0
        with self.assertRaises(ValueError):
            logger.log_episode(rec, episode_index=1, protocol_order_index=1, case_id="c1", split="TEST", geometry_generation_seed=11, environment_seed=9101, horizon_steps=0)

    def test_validation_evaluation_split_strictness(self):
        logger = LocalExperimentLogger(self.val_config, runs_root=self.runs_root)
        rec = create_mock_episode_record(1)

        # Split TEST rejected on VALIDATION_EVALUATION
        with self.assertRaises(ValueError):
            logger.log_episode(rec, episode_index=1, protocol_order_index=1, case_id="c1", split="TEST", geometry_generation_seed=11, environment_seed=5101, horizon_steps=1000)

        # Split VALIDATION accepted on VALIDATION_EVALUATION
        row = logger.log_episode(rec, episode_index=1, protocol_order_index=1, case_id="c1", split="VALIDATION", geometry_generation_seed=11, environment_seed=5101, horizon_steps=1000)
        self.assertEqual(row.split, "VALIDATION")


class TestAgentSeedSemantics(unittest.TestCase):
    """Blocker 3: Preservation of agent_seed = 0."""
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.runs_root = Path(self.temp_dir.name)
        self.config = ExperimentRunConfig(
            run_kind=RunKind.AUDIT,
            benchmark_contract_sha256="dummy",
            agent_contract_sha256="dummy",
            platform_runtime_contract_sha256="dummy",
            logging_contract_sha256="dummy",
            platform_observability_contract_sha256="dummy",
            agent_id="test",
            agent_version="1.0",
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic",
            stateful_within_episode=False,
            allow_dirty_worktree_override=True
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_agent_seed_zero_preserved(self):
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)
        rec = create_mock_episode_record(1, agent_seed=0)

        row = logger.log_episode(
            rec,
            episode_index=1,
            protocol_order_index=1,
            case_id="case_0",
            split="AUDIT",
            environment_seed=9101,
            geometry_generation_seed=11,
            horizon_steps=1000,
            agent_seed=0
        )

        self.assertIsNotNone(row.agent_seed)
        self.assertEqual(row.agent_seed, 0)

        # In CSV dictionary, 0 must be preserved as 0, not empty string
        csv_dict = row.to_csv_dict()
        self.assertEqual(csv_dict["agent_seed"], 0)

    def test_agent_seed_none_handled(self):
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)
        rec = EpisodeRecord(
            tier="Easy", sequence="SCS", scenario_seed=11, terminated=True, truncated=False,
            primary_reason=TerminalReason.SUCCESS, raw_arrival=True, clean_success=True,
            final_route_completion=1.0, max_route_completion=1.0, raw_crash_vehicle=False,
            raw_crash_object=False, raw_crash_building=False, raw_crash_human=False,
            raw_crash_sidewalk=False, raw_out_of_road=False, episode_steps=400,
            simulation_time_s=40.0, mean_speed_kmh=31.5, max_speed_kmh=42.0, episode_return=85.0,
            time_to_clean_success_s=40.0, agent_seed=None
        )

        row = logger.log_episode(
            rec,
            episode_index=1,
            protocol_order_index=1,
            case_id="case_none",
            split="AUDIT",
            environment_seed=9101,
            geometry_generation_seed=11,
            horizon_steps=1000,
            agent_seed=None
        )

        self.assertIsNone(row.agent_seed)
        csv_dict = row.to_csv_dict()
        self.assertEqual(csv_dict["agent_seed"], "")


class TestMetaDriveExactPin(unittest.TestCase):
    """Blocker 4: MetaDrive exact pin verification."""
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.runs_root = Path(self.temp_dir.name)
        self.benchmark_config = ExperimentRunConfig(
            run_kind=RunKind.TEST_EVALUATION,
            benchmark_contract_sha256="dummy",
            agent_contract_sha256="dummy",
            platform_runtime_contract_sha256="dummy",
            logging_contract_sha256="dummy",
            platform_observability_contract_sha256="dummy",
            agent_id="test",
            agent_version="1.0",
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic",
            stateful_within_episode=False,
            allow_dirty_worktree_override=True
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_exact_pin_accepted_as_canonical(self):
        exact_prov = {
            "python_version": "3.10.0",
            "numpy_version": "1.24.0",
            "metadrive_version": PINNED_METADRIVE_VERSION,
            "metadrive_commit": PINNED_METADRIVE_COMMIT,
        }
        logger = LocalExperimentLogger(
            self.benchmark_config,
            runs_root=self.runs_root,
            custom_environment_provenance=exact_prov
        )
        self.assertEqual(logger.environment_verification_status, "VERIFIED")

    def test_unknown_commit_rejected_by_default(self):
        unknown_prov = {
            "python_version": "3.10.0",
            "numpy_version": "1.24.0",
            "metadrive_version": "0.4.3",
            "metadrive_commit": "unknown",
        }
        with self.assertRaises(RuntimeError) as ctx:
            LocalExperimentLogger(
                self.benchmark_config,
                runs_root=self.runs_root,
                custom_environment_provenance=unknown_prov
            )
        self.assertIn("unknown", str(ctx.exception).lower())

    def test_wrong_commit_rejected_by_default(self):
        wrong_commit_prov = {
            "python_version": "3.10.0",
            "numpy_version": "1.24.0",
            "metadrive_version": "0.4.3",
            "metadrive_commit": "badcommit000000000000000000000000000000",
        }
        with self.assertRaises(RuntimeError) as ctx:
            LocalExperimentLogger(
                self.benchmark_config,
                runs_root=self.runs_root,
                custom_environment_provenance=wrong_commit_prov
            )
        self.assertIn("mismatch", str(ctx.exception).lower())

    def test_wrong_version_rejected_by_default(self):
        wrong_ver_prov = {
            "python_version": "3.10.0",
            "numpy_version": "1.24.0",
            "metadrive_version": "0.4.2",
            "metadrive_commit": PINNED_METADRIVE_COMMIT,
        }
        with self.assertRaises(RuntimeError) as ctx:
            LocalExperimentLogger(
                self.benchmark_config,
                runs_root=self.runs_root,
                custom_environment_provenance=wrong_ver_prov
            )
        self.assertIn("version mismatch", str(ctx.exception).lower())

    def test_override_allows_execution_but_marks_noncanonical(self):
        override_config = dataclasses.replace(self.benchmark_config, allow_unverified_env_override=True)
        wrong_prov = {
            "python_version": "3.10.0",
            "numpy_version": "1.24.0",
            "metadrive_version": "0.4.2",
            "metadrive_commit": "wrong_commit",
        }
        logger = LocalExperimentLogger(
            override_config,
            runs_root=self.runs_root,
            custom_environment_provenance=wrong_prov
        )
        self.assertFalse(logger.canonical_run)
        self.assertTrue(logger.unverified_env_override)
        self.assertEqual(logger.environment_verification_status, "MISMATCHED")


class TestTrackingBackendSingleRowContract(unittest.TestCase):
    """Blocker 1: TrackingBackend consumes EpisodeLogRowV1 only; raw EpisodeRecord rejected."""
    def test_fake_backend_accepts_episode_log_row_v1(self):
        backend = FakeTrackingBackend()
        config = ExperimentRunConfig(
            run_kind=RunKind.AUDIT, benchmark_contract_sha256="d", agent_contract_sha256="d",
            platform_runtime_contract_sha256="d", logging_contract_sha256="d", platform_observability_contract_sha256="d",
            agent_id="a", agent_version="1", input_profile_id="STATE_DECISION_V1", action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic", stateful_within_episode=False
        )
        backend.start_run(RunManifestV1("r", "AUDIT", get_utc_now_iso(), "h", {}, {}, {}, {}), config)

        row = EpisodeLogRowV1.from_episode(create_mock_episode_record(1), 1, 1, "c1", "TEST", 11, 9101, 1000)
        backend.log_episode(row)
        self.assertEqual(len(backend.episodes_logged), 1)

    def test_fake_backend_rejects_raw_episode_record(self):
        backend = FakeTrackingBackend()
        config = ExperimentRunConfig(
            run_kind=RunKind.AUDIT, benchmark_contract_sha256="d", agent_contract_sha256="d",
            platform_runtime_contract_sha256="d", logging_contract_sha256="d", platform_observability_contract_sha256="d",
            agent_id="a", agent_version="1", input_profile_id="STATE_DECISION_V1", action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic", stateful_within_episode=False
        )
        backend.start_run(RunManifestV1("r", "AUDIT", get_utc_now_iso(), "h", {}, {}, {}, {}), config)

        raw_rec = create_mock_episode_record(1)
        with self.assertRaises(TypeError):
            backend.log_episode(raw_rec)

    def test_fake_backend_rejects_arbitrary_dict(self):
        backend = FakeTrackingBackend()
        config = ExperimentRunConfig(
            run_kind=RunKind.AUDIT, benchmark_contract_sha256="d", agent_contract_sha256="d",
            platform_runtime_contract_sha256="d", logging_contract_sha256="d", platform_observability_contract_sha256="d",
            agent_id="a", agent_version="1", input_profile_id="STATE_DECISION_V1", action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic", stateful_within_episode=False
        )
        backend.start_run(RunManifestV1("r", "AUDIT", get_utc_now_iso(), "h", {}, {}, {}, {}), config)

        with self.assertRaises(TypeError):
            backend.log_episode({"episode_index": 1, "case_id": "c1"})

    def test_wandb_backend_rejects_raw_episode_record(self):
        backend = WandbBackend(mode=WandbMode.DISABLED)
        raw_rec = create_mock_episode_record(1)
        with self.assertRaises(TypeError):
            backend.log_episode(raw_rec)


class TestFinalizeRunAPIAndAuthoritativeSummary(unittest.TestCase):
    """Blocker 5 & 7: Finalize_run keyword-only API and single authoritative summary."""
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.runs_root = Path(self.temp_dir.name)
        self.config = ExperimentRunConfig(
            run_kind=RunKind.AUDIT, benchmark_contract_sha256="d", agent_contract_sha256="d",
            platform_runtime_contract_sha256="d", logging_contract_sha256="d", platform_observability_contract_sha256="d",
            agent_id="a", agent_version="1", input_profile_id="STATE_DECISION_V1", action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic", stateful_within_episode=False, allow_dirty_worktree_override=True
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_finalize_run_positional_call_rejected(self):
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)
        sync_meta = {"sync_status": "SYNCED"}

        # Positional misuse must raise TypeError immediately
        with self.assertRaises(TypeError):
            logger.finalize_run(sync_meta)

    def test_finalize_run_invalid_payload_rejected(self):
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)
        sync_meta = {"sync_status": "SYNCED"}

        # Passing sync_meta as summary_payload raises ValueError
        with self.assertRaises(ValueError):
            logger.finalize_run(summary_payload=sync_meta)

    def test_same_summary_persisted_locally_and_mirrored(self):
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)
        backend = FakeTrackingBackend()
        backend.start_run(logger.manifest, self.config)

        rec = create_mock_episode_record(1)
        row = logger.log_episode(rec, episode_index=1, protocol_order_index=1, case_id="c1", split="AUDIT", environment_seed=9101, geometry_generation_seed=11, horizon_steps=1000)
        backend.log_episode(row)

        summary = logger.prepare_summary()
        backend.log_summary(summary)
        sync_meta = backend.finish(RunStatus.COMPLETE)
        logger.finalize_run(summary_payload=summary, wandb_sync_info=sync_meta)

        with open(logger.summary_json_path, "r", encoding="utf-8") as f:
            disk_summary = json.load(f)

        self.assertEqual(disk_summary["overall_metrics"], backend.summary_logged["overall_metrics"])


class TestWandbModesAndFailureIsolation(unittest.TestCase):
    """Blocker 6 & 11: W&B failure matrix proving local COMPLETE, and OFFLINE lifecycle."""
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.runs_root = Path(self.temp_dir.name)
        self.config = ExperimentRunConfig(
            run_kind=RunKind.AUDIT, benchmark_contract_sha256="d", agent_contract_sha256="d",
            platform_runtime_contract_sha256="d", logging_contract_sha256="d", platform_observability_contract_sha256="d",
            agent_id="a", agent_version="1", input_profile_id="STATE_DECISION_V1", action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic", stateful_within_episode=False, allow_dirty_worktree_override=True
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_wandb_init_failure_leaves_local_run_complete(self):
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root, custom_run_id="run_fail_init")
        backend = FakeTrackingBackend(simulate_init_failure=True)

        with self.assertRaises(ConnectionError):
            backend.start_run(logger.manifest, self.config)

        self.assertEqual(backend.sync_status, WandbSyncStatus.FAILED)

        # Local experiment proceeds without disruption
        rec = create_mock_episode_record(1)
        row = logger.log_episode(rec, episode_index=1, protocol_order_index=1, case_id="c1", split="AUDIT", environment_seed=9101, geometry_generation_seed=11, horizon_steps=1000)

        summary = logger.prepare_summary()
        sync_meta = backend.fail("Simulated init error")
        logger.finalize_run(summary_payload=summary, wandb_sync_info=sync_meta)

        self.assertEqual(logger.status, RunStatus.COMPLETE)
        self.assertTrue(logger.episodes_csv_path.exists())
        self.assertTrue(logger.summary_json_path.exists())
        self.assertTrue(logger.integrity_json_path.exists())

        with open(logger.wandb_sync_json_path, "r", encoding="utf-8") as f:
            sync_data = json.load(f)
        self.assertEqual(sync_data["sync_status"], WandbSyncStatus.FAILED.value)

    def test_wandb_log_failure_leaves_local_run_complete(self):
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root, custom_run_id="run_fail_log")
        backend = FakeTrackingBackend(simulate_log_failure=True)
        backend.start_run(logger.manifest, self.config)

        rec1 = create_mock_episode_record(1)
        row1 = logger.log_episode(rec1, episode_index=1, protocol_order_index=1, case_id="c1", split="AUDIT", environment_seed=9101, geometry_generation_seed=11, horizon_steps=1000)

        with self.assertRaises(IOError):
            backend.log_episode(row1)

        self.assertEqual(backend.sync_status, WandbSyncStatus.FAILED)

        # Local experiment continues
        rec2 = create_mock_episode_record(2)
        row2 = logger.log_episode(rec2, episode_index=2, protocol_order_index=2, case_id="c2", split="AUDIT", environment_seed=9101, geometry_generation_seed=11, horizon_steps=1000)

        summary = logger.prepare_summary()
        sync_meta = backend.fail("Simulated log error")
        logger.finalize_run(summary_payload=summary, wandb_sync_info=sync_meta)

        self.assertEqual(logger.status, RunStatus.COMPLETE)
        with open(logger.episodes_csv_path, "r", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        self.assertEqual(len(csv_rows), 2)

    def test_wandb_finish_failure_leaves_local_run_complete(self):
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root, custom_run_id="run_fail_finish")
        backend = FakeTrackingBackend(simulate_finish_failure=True)
        backend.start_run(logger.manifest, self.config)

        rec = create_mock_episode_record(1)
        row = logger.log_episode(rec, episode_index=1, protocol_order_index=1, case_id="c1", split="AUDIT", environment_seed=9101, geometry_generation_seed=11, horizon_steps=1000)
        backend.log_episode(row)

        summary = logger.prepare_summary()
        backend.log_summary(summary)

        with self.assertRaises(RuntimeError):
            backend.finish(RunStatus.COMPLETE)

        sync_meta = backend.fail("Simulated finish error")
        logger.finalize_run(summary_payload=summary, wandb_sync_info=sync_meta)

        self.assertEqual(logger.status, RunStatus.COMPLETE)

    @patch("wandb.init")
    def test_offline_mode_restores_wandb_mode_environment(self, mock_wandb_init):
        mock_run = unittest.mock.MagicMock()
        mock_run.id = "mock_offline_run_id"
        mock_run.url = None
        mock_wandb_init.return_value = mock_run

        prior_mode = os.environ.get("WANDB_MODE")
        backend = WandbBackend(mode=WandbMode.OFFLINE)
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)

        backend.start_run(logger.manifest, self.config)
        # While run is active, WANDB_MODE is configured as offline
        self.assertEqual(os.environ.get("WANDB_MODE"), "offline")
        mock_wandb_init.assert_called_once()
        call_kwargs = mock_wandb_init.call_args[1]
        self.assertEqual(call_kwargs["mode"], "offline")
        self.assertNotIn("resume", call_kwargs)
        self.assertEqual(call_kwargs["reinit"], "finish_previous")

        backend.finish(RunStatus.COMPLETE)
        # Environment is cleanly restored to prior state
        self.assertEqual(os.environ.get("WANDB_MODE"), prior_mode)
        self.assertEqual(backend.sync_status, WandbSyncStatus.OFFLINE)


class TestTableParityAndSchema(unittest.TestCase):
    """Blocker 8 & 10: Value parity on shared fields and timing projection check."""
    def test_runtime_table_columns_matches_contract(self):
        core = build_logging_contract_core()
        contract_cols = core["wandb_backend_contract"]["table_columns"]
        self.assertEqual(list(WANDB_TABLE_COLUMNS), contract_cols)
        self.assertEqual(WandbBackend().table_columns, list(WANDB_TABLE_COLUMNS))

    def test_table_parity_shared_fields_and_timing_projection(self):
        rec = create_mock_episode_record(1, clean_success=True, route_completion=1.0)
        timing = EpisodeTimingRecord.from_latencies(1, [2.5, 3.5])
        row = EpisodeLogRowV1.from_episode(rec, 1, 1, "c1", "TEST", 11, 9101, 1000)

        table_row = row.to_wandb_table_row(timing)
        csv_dict = row.to_csv_dict()
        table_dict = dict(zip(WANDB_TABLE_COLUMNS, table_row))

        # 1. Check timing projection
        self.assertAlmostEqual(table_dict["mean_act_ms"], timing.mean_act_ms)

        # 2. Check all shared fields
        for col in WANDB_TABLE_COLUMNS:
            if col == "mean_act_ms":
                continue
            tab_v = table_dict[col]
            csv_v = csv_dict[col]
            if isinstance(tab_v, bool):
                self.assertEqual(tab_v, bool(csv_v))
            elif isinstance(tab_v, (int, float)):
                self.assertAlmostEqual(float(tab_v), float(csv_v), places=5)
            else:
                self.assertEqual(str(tab_v), str(csv_v))


class TestHashSensitivityAndMutation(unittest.TestCase):
    """Blocker 9: Cryptographic hash sensitivity across schema dimensions."""
    def setUp(self):
        self.default_core = build_logging_contract_core()
        self.default_hash = canonical_json_sha256(self.default_core)

    def test_table_column_mutation_changes_hash(self):
        mut_cols = list(WANDB_TABLE_COLUMNS) + ["extra_column"]
        mut_core = build_logging_contract_core(custom_wandb_contract={"table_columns": mut_cols})
        self.assertNotEqual(self.default_hash, canonical_json_sha256(mut_core))

    def test_event_key_mutation_changes_hash(self):
        mut_keys = [k for k in WANDB_STATIC_EPISODE_EVENT_KEYS if k != "eval/clean_success"] + ["eval/success"]
        mut_core = build_logging_contract_core(custom_wandb_contract={"static_episode_event_keys": mut_keys})
        self.assertNotEqual(self.default_hash, canonical_json_sha256(mut_core))

    def test_summary_mapping_rule_mutation_changes_hash(self):
        mut_rules = dict(WANDB_SUMMARY_MAPPING_RULES)
        mut_rules["overall"] = "summary/<metric>"
        mut_core = build_logging_contract_core(custom_wandb_contract={"summary_mapping_rules": mut_rules})
        self.assertNotEqual(self.default_hash, canonical_json_sha256(mut_core))

    def test_dirty_policy_mutation_changes_hash(self):
        mut_core = build_logging_contract_core(custom_policies={"dirty_worktree_policy": "ALLOW_DIRTY_ALWAYS"})
        self.assertNotEqual(self.default_hash, canonical_json_sha256(mut_core))

    def test_lifecycle_mutation_changes_hash(self):
        mut_core = build_logging_contract_core(custom_policies={"run_lifecycle_transitions": ["MUTATED"]})
        self.assertNotEqual(self.default_hash, canonical_json_sha256(mut_core))

    def test_resume_policy_mutation_changes_hash(self):
        mut_core = build_logging_contract_core(custom_policies={"resume_policy": "ENABLED"})
        self.assertNotEqual(self.default_hash, canonical_json_sha256(mut_core))


class TestSecurityAndInterruption(unittest.TestCase):
    """Blocker 12, 13, 14: Sanitization, integrity with wandb_sync_sha256, and interruption."""
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.runs_root = Path(self.temp_dir.name)
        self.config = ExperimentRunConfig(
            run_kind=RunKind.AUDIT, benchmark_contract_sha256="d", agent_contract_sha256="d",
            platform_runtime_contract_sha256="d", logging_contract_sha256="d", platform_observability_contract_sha256="d",
            agent_id="a", agent_version="1", input_profile_id="STATE_DECISION_V1", action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic", stateful_within_episode=False, allow_dirty_worktree_override=True
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sanitize_error_message_redactions(self):
        home = str(Path.home())
        repo = str(Path(__file__).resolve().parent.parent)

        mock_key = "wandb" + "_v1_" + "dummytestkey123"
        raw = f"Error in {home}/secret_file.py at {repo}/sub: {'api_' + 'key='}{mock_key} Bearer token_xyz"
        sanitized = sanitize_error_message(raw)

        self.assertNotIn(home, sanitized)
        self.assertNotIn(repo, sanitized)
        self.assertNotIn(mock_key, sanitized)
        self.assertNotIn("token_xyz", sanitized)

    def test_run_integrity_covers_wandb_sync(self):
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)
        rec = create_mock_episode_record(1)
        logger.log_episode(rec, episode_index=1, protocol_order_index=1, case_id="c1", split="AUDIT", environment_seed=9101, geometry_generation_seed=11, horizon_steps=1000)

        summary = logger.prepare_summary()
        integrity = logger.finalize_run(summary_payload=summary)

        self.assertIsNotNone(integrity.wandb_sync_sha256)
        expected_sync_hash = canonical_json_file_sha256(logger.wandb_sync_json_path)
        self.assertEqual(integrity.wandb_sync_sha256, expected_sync_hash)

    def test_mark_interrupted_persists_interrupted_state(self):
        logger = LocalExperimentLogger(self.config, runs_root=self.runs_root)
        logger.mark_interrupted("SIGINT received by user")

        self.assertEqual(logger.status, RunStatus.INTERRUPTED)
        with open(logger.run_state_path, "r", encoding="utf-8") as f:
            st = json.load(f)
        self.assertEqual(st["status"], "INTERRUPTED")
        self.assertEqual(st["failure_category"], "INTERRUPTED")


if __name__ == "__main__":
    unittest.main()
