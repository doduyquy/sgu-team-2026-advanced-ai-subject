"""
Unit and Integration Tests for Research Workbench Results Browser, Artifact Integrity & Live UX (Gate 7.5B Pass B2 Correction 2).

Verifies:
1. Run repository discovers valid run directories.
2. Random unrelated directories are not treated as valid finalized runs.
3. COMPLETE run loads all required artifacts.
4. COMPLETE untampered run integrity verifies.
5. Tampered episodes.csv produces integrity failure.
6. Tampered summary.json produces integrity failure.
7. Hash failure never rewrites run_integrity.json.
8. RUNNING run with no summary/integrity is NOT_FINAL, not corrupted.
9. FAILED run loads without fabricated summary.
10. INTERRUPTED run loads without fabricated summary.
11. COMPLETE missing run_integrity is UNVERIFIED, never VERIFIED.
12. summary cards use values directly from summary.json.
13. None time-to-success remains N/A/None, not 0.
14. episode_return is labeled diagnostic, not primary.
15. Episodes table is NoEditTriggers.
16. Timing table is NoEditTriggers.
17. Integrity table is NoEditTriggers.
18. No delete/modify run controls exist.
19. Live buffer resets at RUN_STARTED.
20. Live current-episode trace resets at EPISODE_STARTED.
21. EPISODE_PROGRESS appends route/speed points.
22. Free-text event message is not parsed into scientific outcome state.
23. RUN_FINISHED sets completed presentation state.
24. RUN_FAILED and RUN_INTERRUPTED remain distinct.
25. Auto-refresh after execution loads disk artifact state.
26. Existing B1 frozen hashes remain unchanged.
27. New B2 contract hashes are deterministic.
28. Exact real Gate-7 schema parsing with ExperimentRunConfig, RunManifestV1, RunStateV1.
29. Strict run_id parity across dir basename, state, manifest, and integrity.
30. Path traversal in load_run_snapshot is strictly rejected.
31. FAILED integrity trust-gates primary metric display with prominent warning banner.
32. Custom runs_root auto-loads without preconfiguring ResultsWidget.
33. Stored macro_metrics in summary.json is displayed without recomputation.
34. Strict 4-way run identity: COMPLETE run_integrity.run_id mismatch is rejected as malformed.
35. Copied run with different directory basename without rewriting identity is rejected as malformed.
36. Missing run_state.json or missing run_manifest.json is rejected as malformed.
37. Malformed JSON in run_state or run_manifest is rejected cleanly without crashing.
38. UNVERIFIED run displays warning and avoids authoritative claims.
39. NOT_FINAL run with summary displays provisional notice and avoids final authoritative claims.
40. Single-source episode table column names match EPISODE_CSV_COLUMNS exactly (31 columns).
41. Single-source timing table column names match EpisodeTimingRecord dataclass fields exactly (8 fields).
42. Full detailed MetaDrive environment provenance fields are exposed and match manifest.
"""

import dataclasses
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Optional
import unittest

# Ensure offscreen Qt platform before importing PySide6
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QAbstractItemView, QApplication

from src.launcher.cases import get_default_project_root
from src.launcher.contracts import (
    GATE7_LOCKED_OBSERVABILITY_HASH,
    build_launcher_contract_core,
    compute_launcher_contract_sha256,
    compute_platform_execution_contract_sha256,
)
from src.launcher.registry import build_canonical_agent_registry, compute_canonical_registry_sha256
from src.launcher.resolver import (
    GATE5_LOCKED_BENCHMARK_HASH,
    GATE6_LOCKED_AGENT_HASH,
    GATE6_LOCKED_RUNTIME_HASH,
    GATE7_LOCKED_LOGGING_HASH,
)
from src.platform import (
    ExperimentRunConfig,
    RunKind,
    RunManifestV1,
    RunStateV1,
    RunStatus,
    WandbMode,
    canonical_csv_file_sha256,
    canonical_json_file_sha256,
    canonical_json_sha256,
)
from src.platform.experiment_logging import EPISODE_CSV_COLUMNS, EpisodeTimingRecord
from src.workbench.contracts import (
    GATE7_5A_LAUNCHER_CONTRACT_HASH,
    GATE7_5A_PLATFORM_EXECUTION_CONTRACT_HASH,
    build_workbench_contract_core,
    compute_platform_workbench_contract_sha256,
    compute_workbench_contract_sha256,
)
from src.workbench.live_telemetry import LiveTelemetryBufferV1
from src.workbench.main_window import MainWindow
from src.workbench.results_contracts import (
    GATE7_5B_B1_PLATFORM_WORKBENCH_CONTRACT_HASH,
    GATE7_5B_B1_WORKBENCH_CONTRACT_HASH,
    build_workbench_results_contract_core,
    compute_platform_workbench_results_contract_sha256,
    compute_workbench_results_contract_sha256,
)
from src.workbench.results_repository import (
    IntegrityDisplayStatus,
    RunArtifactRepository,
    RunArtifactSnapshotV1,
)
from src.workbench.widgets.results_widget import ResultsWidget


class TestWorkbenchResultsSuite(unittest.TestCase):
    """Comprehensive Gate 7.5B Pass B2 Results & Live UX Test Suite."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.runs_root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_real_schema_run(
        self,
        run_id: str,
        status: str = "COMPLETE",
        tamper_file: Optional[str] = None,
        omit_integrity: bool = False,
        macro_metrics: Optional[Dict[str, Any]] = None,
        custom_integrity_run_id: Optional[str] = None,
    ) -> Path:
        """Helper to create a standard mock run directory matching exact Gate-7 schemas."""
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        config = ExperimentRunConfig(
            run_kind=RunKind.AUDIT,
            benchmark_contract_sha256=GATE5_LOCKED_BENCHMARK_HASH,
            agent_contract_sha256=GATE6_LOCKED_AGENT_HASH,
            platform_runtime_contract_sha256=GATE6_LOCKED_RUNTIME_HASH,
            logging_contract_sha256=GATE7_LOCKED_LOGGING_HASH,
            platform_observability_contract_sha256=GATE7_LOCKED_OBSERVABILITY_HASH,
            agent_id="fixture_constant_continuous",
            agent_version="1.0.0",
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic",
            stateful_within_episode=False,
            expected_episode_count=1,
            manifest_name="geometry_split_manifest.csv",
        )
        cfg_dict = config.to_dict()

        manifest = RunManifestV1(
            run_id=run_id,
            run_kind="AUDIT",
            created_at_utc="2026-10-02T12:00:00Z",
            experiment_config_sha256=config.compute_config_sha256(),
            config=cfg_dict,
            git_provenance={"git_commit_sha": "abc12345", "git_worktree_dirty": False},
            environment_provenance={
                "metadrive_version": "0.4.3",
                "metadrive_commit": "85e5dadc",
                "metadrive_pinned_version": "0.4.3",
                "metadrive_pinned_commit": "85e5dadc",
                "metadrive_verification_status": "EXACT_PIN_VERIFIED",
                "metadrive_verification_reason": "Exact pin match",
            },
            platform_contracts={"gate5": GATE5_LOCKED_BENCHMARK_HASH},
        )

        state = RunStateV1(
            run_id=run_id,
            status=status,
            started_at_utc="2026-10-02T12:00:00Z",
            updated_at_utc="2026-10-02T12:01:00Z",
            finished_at_utc="2026-10-02T12:01:00Z" if status == "COMPLETE" else None,
            recorded_episode_count=1 if status == "COMPLETE" else 0,
            expected_episode_count=1,
            canonical_run=False,
            dirty_override=False,
            unverified_env_override=False,
            environment_verification_status="VERIFIED",
            failure_category="TEST_FAILURE" if status == "FAILED" else None,
            sanitized_failure_message="Simulated failure occurred" if status == "FAILED" else None,
        )

        overall_metrics = {
            "clean_success_rate": 1.0,
            "safety_failure_rate": 0.0,
            "mean_final_route_completion": 1.0,
            "median_final_route_completion": 1.0,
            "mean_time_to_clean_success_s": 24.5,
            "mean_episode_return": 18.2,
            "success_rate": 1.0,
        }
        summary_data: Dict[str, Any] = {
            "run_id": run_id,
            "overall_metrics": overall_metrics,
            "tier_metrics": {
                "Easy": {"clean_success_rate": 1.0, "mean_final_route_completion": 1.0}
            },
        }
        if macro_metrics:
            summary_data["macro_metrics"] = macro_metrics

        # Write core Gate-7 files
        with open(run_dir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        with open(run_dir / "run_state.json", "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2)

        if status == "COMPLETE" or macro_metrics is not None:
            with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2)

            episodes_csv_content = (
                "episode_index,protocol_order_index,case_id,split,tier,sequence,geometry_generation_seed,environment_seed,horizon_steps,agent_seed,primary_terminal_reason,terminal_reason,clean_success,raw_arrival,final_route_completion,max_route_completion,episode_steps,episode_time_s,time_to_clean_success_s,mean_speed_kmh,max_speed_kmh,episode_return,crash_human,crash_vehicle,crash_object,crash_building,crash_sidewalk,out_of_road,timeout,has_technical_failure,technical_failure_reason\n"
                "0,1,sandbox/Easy/SCCS/0/0,TRAIN,Easy,SCCS,0,0,1758,101,SUCCESS,SUCCESS,True,True,1.0,1.0,245,24.5,24.5,32.4,45.2,18.2,False,False,False,False,False,False,False,False,\n"
            )
            with open(run_dir / "episodes.csv", "w", encoding="utf-8", newline="") as f:
                f.write(episodes_csv_content)

            timing_csv_content = (
                "episode_index,act_count,mean_act_ms,median_act_ms,p95_act_ms,max_act_ms,total_act_ms,latency_sync_policy\n"
                "0,245,0.45,0.42,0.61,1.2,110.25,NONE\n"
            )
            with open(run_dir / "timing.csv", "w", encoding="utf-8", newline="") as f:
                f.write(timing_csv_content)

            wandb_sync_data = {"mode": "DISABLED", "sync_status": "NOT_SYNCED"}
            with open(run_dir / "wandb_sync.json", "w", encoding="utf-8") as f:
                json.dump(wandb_sync_data, f, indent=2)

            if not omit_integrity:
                integrity_data = {
                    "run_id": custom_integrity_run_id or run_id,
                    "experiment_config_sha256": config.compute_config_sha256(),
                    "run_manifest_sha256": canonical_json_file_sha256(run_dir / "run_manifest.json"),
                    "episodes_sha256": canonical_csv_file_sha256(run_dir / "episodes.csv"),
                    "summary_sha256": canonical_json_file_sha256(run_dir / "summary.json"),
                    "timing_sha256": canonical_csv_file_sha256(run_dir / "timing.csv"),
                    "run_state_sha256": canonical_json_file_sha256(run_dir / "run_state.json"),
                    "wandb_sync_sha256": canonical_json_file_sha256(run_dir / "wandb_sync.json"),
                    "technical_failures_sha256": None,
                    "finalized_at_utc": "2026-10-02T12:01:00Z",
                }
                with open(run_dir / "run_integrity.json", "w", encoding="utf-8") as f:
                    json.dump(integrity_data, f, indent=2)

        # Apply tamper if requested
        if tamper_file:
            target = run_dir / tamper_file
            if target.exists():
                with open(target, "a", encoding="utf-8") as f:
                    f.write("\n# TAMPERED_DATA\n")

        return run_dir

    def test_01_run_repository_discovers_valid_run_directories(self):
        """Repository discovers valid run directories and sorts them."""
        self._create_real_schema_run("run_b")
        self._create_real_schema_run("run_a")
        repo = RunArtifactRepository(self.runs_root)
        discovered = repo.discover_run_ids()
        self.assertEqual(discovered, ["run_a", "run_b"])

    def test_02_random_unrelated_directories_are_ignored(self):
        """Random subdirectories without run_state or run_manifest are ignored."""
        (self.runs_root / "random_folder").mkdir(parents=True, exist_ok=True)
        (self.runs_root / "empty_dir").mkdir(parents=True, exist_ok=True)
        self._create_real_schema_run("valid_run")
        repo = RunArtifactRepository(self.runs_root)
        discovered = repo.discover_run_ids()
        self.assertEqual(discovered, ["valid_run"])

    def test_03_complete_run_loads_all_required_artifacts(self):
        """Snapshot loads manifest, state, summary, episodes, timing, and wandb_sync."""
        self._create_real_schema_run("complete_run_01")
        repo = RunArtifactRepository(self.runs_root)
        snap = repo.load_run_snapshot("complete_run_01")
        self.assertIsNotNone(snap)
        self.assertEqual(snap.status, "COMPLETE")
        self.assertIsNotNone(snap.summary_payload)
        self.assertEqual(len(snap.episode_rows), 1)
        self.assertEqual(len(snap.timing_rows), 1)

    def test_04_complete_untampered_run_integrity_verifies(self):
        """Untampered COMPLETE run evaluates to VERIFIED integrity status."""
        self._create_real_schema_run("clean_run")
        repo = RunArtifactRepository(self.runs_root)
        snap = repo.load_run_snapshot("clean_run")
        self.assertEqual(snap.integrity_status, IntegrityDisplayStatus.VERIFIED)

    def test_05_tampered_episodes_csv_produces_integrity_failure(self):
        """Tampering with episodes.csv marks integrity as FAILED and identifies mismatch."""
        self._create_real_schema_run("tampered_ep_run", tamper_file="episodes.csv")
        repo = RunArtifactRepository(self.runs_root)
        snap = repo.load_run_snapshot("tampered_ep_run")
        self.assertEqual(snap.integrity_status, IntegrityDisplayStatus.FAILED)
        ep_check = next(c for c in snap.integrity_details if c.artifact_name == "episodes.csv")
        self.assertEqual(ep_check.status, "FAIL")

    def test_06_tampered_summary_json_produces_integrity_failure(self):
        """Tampering with summary.json marks integrity as FAILED."""
        self._create_real_schema_run("tampered_sum_run", tamper_file="summary.json")
        repo = RunArtifactRepository(self.runs_root)
        snap = repo.load_run_snapshot("tampered_sum_run")
        self.assertEqual(snap.integrity_status, IntegrityDisplayStatus.FAILED)
        sum_check = next(c for c in snap.integrity_details if c.artifact_name == "summary.json")
        self.assertEqual(sum_check.status, "FAIL")

    def test_07_hash_failure_never_rewrites_run_integrity(self):
        """Evaluating integrity never mutates run_integrity.json on disk."""
        run_dir = self._create_real_schema_run("tamper_check_run", tamper_file="episodes.csv")
        integ_file = run_dir / "run_integrity.json"
        with open(integ_file, "rb") as f:
            original_bytes = f.read()

        repo = RunArtifactRepository(self.runs_root)
        snap = repo.load_run_snapshot("tamper_check_run")
        self.assertEqual(snap.integrity_status, IntegrityDisplayStatus.FAILED)

        with open(integ_file, "rb") as f:
            after_bytes = f.read()
        self.assertEqual(original_bytes, after_bytes)

    def test_08_running_run_is_not_final_not_corrupted(self):
        """RUNNING run with no summary/integrity evaluates to NOT_FINAL, not FAILED."""
        self._create_real_schema_run("running_run", status="RUNNING")
        repo = RunArtifactRepository(self.runs_root)
        snap = repo.load_run_snapshot("running_run")
        self.assertEqual(snap.status, "RUNNING")
        self.assertEqual(snap.integrity_status, IntegrityDisplayStatus.NOT_FINAL)

    def test_09_failed_run_loads_without_fabricated_summary(self):
        """FAILED run loads failure category and sanitized_failure_message without fabricating summary."""
        self._create_real_schema_run("failed_run", status="FAILED")
        repo = RunArtifactRepository(self.runs_root)
        snap = repo.load_run_snapshot("failed_run")
        self.assertEqual(snap.status, "FAILED")
        self.assertIsNone(snap.summary_payload)
        self.assertEqual(snap.failure_category, "TEST_FAILURE")
        self.assertEqual(snap.sanitized_failure_message, "Simulated failure occurred")
        self.assertEqual(snap.integrity_status, IntegrityDisplayStatus.NOT_FINAL)

    def test_10_interrupted_run_loads_without_fabricated_summary(self):
        """INTERRUPTED run loads truthfully without fabricated summary."""
        self._create_real_schema_run("interrupted_run", status="INTERRUPTED")
        repo = RunArtifactRepository(self.runs_root)
        snap = repo.load_run_snapshot("interrupted_run")
        self.assertEqual(snap.status, "INTERRUPTED")
        self.assertIsNone(snap.summary_payload)
        self.assertEqual(snap.integrity_status, IntegrityDisplayStatus.NOT_FINAL)

    def test_11_complete_missing_integrity_is_unverified(self):
        """COMPLETE run missing run_integrity.json evaluates to UNVERIFIED, never VERIFIED."""
        self._create_real_schema_run("missing_integ_run", status="COMPLETE", omit_integrity=True)
        repo = RunArtifactRepository(self.runs_root)
        snap = repo.load_run_snapshot("missing_integ_run")
        self.assertEqual(snap.status, "COMPLETE")
        self.assertEqual(snap.integrity_status, IntegrityDisplayStatus.UNVERIFIED)

    def test_12_summary_cards_use_values_directly_from_summary_json(self):
        """Overview metric cards display exact values from summary.json['overall_metrics']."""
        self._create_real_schema_run("cards_test_run")
        widget = ResultsWidget(runs_root=self.runs_root)
        widget.select_run_by_id("cards_test_run")

        lbl_success = widget.card_success.findChild(object, "value_label")
        lbl_mean_route = widget.card_mean_route.findChild(object, "value_label")
        self.assertEqual(lbl_success.text(), "100.0%")
        self.assertEqual(lbl_mean_route.text(), "100.0%")
        widget.close()

    def test_13_none_time_to_success_remains_na(self):
        """mean_time_to_clean_success_s = None is displayed as N/A, never zero."""
        run_dir = self._create_real_schema_run("no_clean_success_run")
        sum_path = run_dir / "summary.json"
        with open(sum_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["overall_metrics"]["mean_time_to_clean_success_s"] = None
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        widget = ResultsWidget(runs_root=self.runs_root)
        widget.select_run_by_id("no_clean_success_run")

        lbl_time = widget.card_time.findChild(object, "value_label")
        self.assertIn("N/A", lbl_time.text())
        self.assertNotIn("0.00", lbl_time.text())
        widget.close()

    def test_14_episode_return_is_labeled_diagnostic(self):
        """Overview text explicitly labels return as diagnostic training signal."""
        self._create_real_schema_run("diag_label_run")
        widget = ResultsWidget(runs_root=self.runs_root)
        widget.select_run_by_id("diag_label_run")

        summary_text = widget.text_summary_details.toPlainText()
        self.assertIn("Diagnostic Signal (NOT Ranking Score)", summary_text)
        widget.close()

    def test_15_episodes_table_has_no_edit_triggers(self):
        """Episodes table enforces NoEditTriggers."""
        widget = ResultsWidget(runs_root=self.runs_root)
        self.assertEqual(widget.table_episodes.editTriggers(), QAbstractItemView.EditTrigger.NoEditTriggers)
        widget.close()

    def test_16_timing_table_has_no_edit_triggers(self):
        """Timing table enforces NoEditTriggers."""
        widget = ResultsWidget(runs_root=self.runs_root)
        self.assertEqual(widget.table_timing.editTriggers(), QAbstractItemView.EditTrigger.NoEditTriggers)
        widget.close()

    def test_17_integrity_table_has_no_edit_triggers(self):
        """Integrity table enforces NoEditTriggers."""
        widget = ResultsWidget(runs_root=self.runs_root)
        self.assertEqual(widget.table_integrity.editTriggers(), QAbstractItemView.EditTrigger.NoEditTriggers)
        widget.close()

    def test_18_no_delete_or_modify_run_controls_exist(self):
        """ResultsWidget contains no delete, rename, or file modification buttons."""
        widget = ResultsWidget(runs_root=self.runs_root)
        buttons = widget.findChildren(object)
        for b in buttons:
            if hasattr(b, "text") and callable(b.text):
                t = b.text().lower()
                self.assertNotIn("delete", t)
                self.assertNotIn("remove", t)
                self.assertNotIn("rename", t)
                self.assertNotIn("repair", t)
        widget.close()

    def test_19_live_buffer_resets_at_run_started(self):
        """LiveTelemetryBuffer resets cleanly on RUN_STARTED."""
        buf = LiveTelemetryBufferV1()
        buf.handle_event({"event": {"event_type": "EPISODE_STARTED", "episode_index": 0}})
        buf.handle_event({"event": {"event_type": "EPISODE_PROGRESS", "step_index": 10, "route_completion": 0.1, "speed_kmh": 20.0}})
        self.assertIsNotNone(buf.current_episode_trace)

        buf.handle_event({"event": {"event_type": "RUN_STARTED", "run_id": "new_run"}})
        self.assertEqual(buf.current_run_id, "new_run")
        self.assertEqual(buf.run_status, "RUNNING")
        self.assertIsNone(buf.current_episode_trace)
        self.assertEqual(len(buf.completed_episode_traces), 0)

    def test_20_live_current_episode_trace_resets_at_episode_started(self):
        """EPISODE_STARTED resets the active episode trace."""
        buf = LiveTelemetryBufferV1()
        buf.handle_event({"event": {"event_type": "RUN_STARTED", "run_id": "r1"}})
        buf.handle_event({"event": {"event_type": "EPISODE_STARTED", "episode_index": 0}})
        buf.handle_event({"event": {"event_type": "EPISODE_PROGRESS", "step_index": 10, "route_completion": 0.1, "speed_kmh": 20.0}})
        buf.handle_event({"event": {"event_type": "EPISODE_FINISHED", "status": "SUCCESS"}})

        # Start episode 1
        buf.handle_event({"event": {"event_type": "EPISODE_STARTED", "episode_index": 1}})
        self.assertEqual(buf.current_episode_trace.episode_index, 1)
        self.assertEqual(len(buf.current_episode_trace.step_indices), 0)
        self.assertEqual(len(buf.completed_episode_traces), 1)

    def test_21_episode_progress_appends_route_speed_points(self):
        """EPISODE_PROGRESS appends exact points to active trace."""
        buf = LiveTelemetryBufferV1()
        buf.handle_event({"event": {"event_type": "EPISODE_STARTED", "episode_index": 0}})
        buf.handle_event({"event": {"event_type": "EPISODE_PROGRESS", "step_index": 10, "route_completion": 0.05, "speed_kmh": 15.0}})
        buf.handle_event({"event": {"event_type": "EPISODE_PROGRESS", "step_index": 20, "route_completion": 0.12, "speed_kmh": 22.0}})

        trace = buf.current_episode_trace
        self.assertEqual(trace.step_indices, [10, 20])
        self.assertEqual(trace.route_completions, [0.05, 0.12])
        self.assertEqual(trace.speeds_kmh, [15.0, 22.0])

    def test_22_free_text_event_message_is_not_parsed_into_scientific_outcome(self):
        """Scientific status is read from structured fields, never parsed from free-text message."""
        buf = LiveTelemetryBufferV1()
        buf.handle_event({"event": {"event_type": "EPISODE_STARTED", "episode_index": 0}})
        buf.handle_event({"event": {"event_type": "EPISODE_FINISHED", "status": "FAILURE", "message": "Fabricated SUCCESS in message string"}})
        self.assertEqual(buf.completed_episode_traces[0].final_status, "FAILURE")

    def test_23_run_finished_sets_completed_presentation_state(self):
        """RUN_FINISHED marks run_status as COMPLETE."""
        buf = LiveTelemetryBufferV1()
        buf.handle_event({"event": {"event_type": "RUN_STARTED", "run_id": "r1"}})
        buf.handle_event({"event": {"event_type": "RUN_FINISHED"}})
        self.assertEqual(buf.run_status, "COMPLETE")

    def test_24_run_failed_and_interrupted_remain_distinct(self):
        """RUN_FAILED and RUN_INTERRUPTED produce distinct status values."""
        buf = LiveTelemetryBufferV1()
        buf.handle_event({"event": {"event_type": "RUN_FAILED"}})
        self.assertEqual(buf.run_status, "FAILED")

        buf.handle_event({"event": {"event_type": "RUN_INTERRUPTED"}})
        self.assertEqual(buf.run_status, "INTERRUPTED")

    def test_25_auto_refresh_after_execution_loads_disk_artifact_state(self):
        """MainWindow auto-selects and loads completed run from disk on successful workerDone."""
        self._create_real_schema_run("auto_refresh_run")
        window = MainWindow()
        window.results_widget.set_runs_root(self.runs_root)

        window._on_worker_done({"operation": "RUN", "success": True, "run_id": "auto_refresh_run"})
        self.assertIsNotNone(window.results_widget._current_snapshot)
        self.assertEqual(window.results_widget._current_snapshot.run_id, "auto_refresh_run")
        window.close()

    def test_26_existing_b1_frozen_hashes_remain_unchanged(self):
        """B1 frozen contract hashes remain unchanged."""
        self.assertEqual(GATE7_5B_B1_WORKBENCH_CONTRACT_HASH, "e5711485e6571a04c336739ebc6f285213a1d9631fa29820a89707b37875a82b")
        self.assertEqual(GATE7_5B_B1_PLATFORM_WORKBENCH_CONTRACT_HASH, "36f3a0891a9153df12a8b53b01f2064afbf848eafda462149a768ec989199e47")

    def test_27_new_b2_contract_hashes_are_deterministic(self):
        """New B2 contract hashes are deterministic across multiple calls."""
        c1 = build_workbench_results_contract_core()
        c2 = build_workbench_results_contract_core()
        h1 = compute_workbench_results_contract_sha256(c1)
        h2 = compute_workbench_results_contract_sha256(c2)
        self.assertEqual(h1, h2)
        ph1 = compute_platform_workbench_results_contract_sha256(h1)
        ph2 = compute_platform_workbench_results_contract_sha256(h2)
        self.assertEqual(ph1, ph2)

    def test_28_real_gate7_schema_parity(self):
        """Snapshot view model extracts exact fields from RunManifestV1 and RunStateV1."""
        self._create_real_schema_run("schema_parity_run")
        repo = RunArtifactRepository(self.runs_root)
        snap = repo.load_run_snapshot("schema_parity_run")

        self.assertEqual(snap.agent_id, "fixture_constant_continuous")
        self.assertEqual(snap.agent_version, "1.0.0")
        self.assertEqual(snap.input_profile_id, "STATE_DECISION_V1")
        self.assertEqual(snap.action_adapter_id, "continuous_box2_v1")
        self.assertEqual(snap.inference_stochasticity, "deterministic")
        self.assertFalse(snap.stateful_within_episode)
        self.assertEqual(snap.git_commit_sha, "abc12345")
        self.assertFalse(snap.git_worktree_dirty)
        self.assertEqual(snap.metadrive_version, "0.4.3")
        self.assertEqual(snap.metadrive_commit, "85e5dadc")
        self.assertFalse(snap.canonical_run)

    def test_29_run_identity_parity_enforced(self):
        """Mismatch between dir basename, state run_id, or manifest run_id causes snapshot to be None."""
        run_dir = self._create_real_schema_run("identity_mismatch_run")
        m_path = run_dir / "run_manifest.json"
        with open(m_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["run_id"] = "different_id"
        with open(m_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        repo = RunArtifactRepository(self.runs_root)
        snap = repo.load_run_snapshot("identity_mismatch_run")
        self.assertIsNone(snap, "Run with identity mismatch between manifest and dir must be rejected as malformed")

    def test_30_path_traversal_in_load_run_snapshot_rejected(self):
        """Traversing outside runs_root returns None."""
        repo = RunArtifactRepository(self.runs_root)
        self.assertIsNone(repo.load_run_snapshot("../unauthorized_path"))
        self.assertIsNone(repo.load_run_snapshot(".."))
        self.assertIsNone(repo.load_run_snapshot("sub/nested"))

    def test_31_failed_integrity_trust_gates_primary_metrics(self):
        """When integrity fails, ResultsWidget displays UNTRUSTED warning banner and no Authoritative labels."""
        self._create_real_schema_run("untrusted_metrics_run", tamper_file="summary.json")
        widget = ResultsWidget(runs_root=self.runs_root)
        widget.show()
        widget.select_run_by_id("untrusted_metrics_run")

        self.assertIn("UNTRUSTED", widget.cards_group.title())
        self.assertNotIn("Authoritative Primary", widget.cards_group.title())
        self.assertNotIn("Authoritative", widget.chart_box.title())
        self.assertFalse(widget.lbl_trust_warning.isHidden())
        self.assertIn("CRITICAL WARNING", widget.lbl_trust_warning.text())
        widget.close()

    def test_32_custom_runs_root_autoload_without_preconfiguration(self):
        """Custom runs_root auto-loads into ResultsWidget via execution report without preconfiguring root."""
        custom_temp = tempfile.TemporaryDirectory()
        custom_root = Path(custom_temp.name)
        custom_run_dir = custom_root / "custom_autoload_run"
        custom_run_dir.mkdir(parents=True, exist_ok=True)

        config = ExperimentRunConfig(
            run_kind=RunKind.AUDIT,
            benchmark_contract_sha256=GATE5_LOCKED_BENCHMARK_HASH,
            agent_contract_sha256=GATE6_LOCKED_AGENT_HASH,
            platform_runtime_contract_sha256=GATE6_LOCKED_RUNTIME_HASH,
            logging_contract_sha256=GATE7_LOCKED_LOGGING_HASH,
            platform_observability_contract_sha256=GATE7_LOCKED_OBSERVABILITY_HASH,
            agent_id="fixture_constant_continuous",
            agent_version="1.0.0",
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic",
            stateful_within_episode=False,
            expected_episode_count=1,
            manifest_name="geometry_split_manifest.csv",
        )
        manifest = RunManifestV1(
            run_id="custom_autoload_run",
            run_kind="AUDIT",
            created_at_utc="2026-10-02T12:00:00Z",
            experiment_config_sha256=config.compute_config_sha256(),
            config=config.to_dict(),
            git_provenance={"git_commit_sha": "abc12345", "git_worktree_dirty": False},
            environment_provenance={"metadrive_version": "0.4.3", "metadrive_commit": "85e5dadc"},
            platform_contracts={"gate5": GATE5_LOCKED_BENCHMARK_HASH},
        )
        state = RunStateV1(
            run_id="custom_autoload_run",
            status="COMPLETE",
            started_at_utc="2026-10-02T12:00:00Z",
            updated_at_utc="2026-10-02T12:01:00Z",
            finished_at_utc="2026-10-02T12:01:00Z",
            recorded_episode_count=1,
            expected_episode_count=1,
            canonical_run=False,
            dirty_override=False,
            unverified_env_override=False,
            environment_verification_status="VERIFIED",
        )
        with open(custom_run_dir / "run_manifest.json", "w") as f:
            json.dump(manifest.to_dict(), f)
        with open(custom_run_dir / "run_state.json", "w") as f:
            json.dump(state.to_dict(), f)

        window = MainWindow()
        self.assertNotEqual(window.results_widget.runs_root, custom_root)

        window._on_execution_report({"execution_report": {"run_id": "custom_autoload_run", "run_dir": str(custom_run_dir), "status": "COMPLETE"}})
        window._on_worker_done({"operation": "RUN", "success": True, "run_id": "custom_autoload_run"})

        self.assertEqual(window.results_widget.runs_root, custom_root)
        self.assertIsNotNone(window.results_widget._current_snapshot)
        self.assertEqual(window.results_widget._current_snapshot.run_id, "custom_autoload_run")
        window.close()
        custom_temp.cleanup()

    def test_33_stored_macro_metrics_displayed_without_recomputation(self):
        """Stored macro_metrics from summary.json is displayed directly in summary details."""
        macro = {"clean_success_rate_macro": 0.85, "route_completion_macro": 0.92}
        self._create_real_schema_run("macro_metrics_run", macro_metrics=macro)
        widget = ResultsWidget(runs_root=self.runs_root)
        widget.select_run_by_id("macro_metrics_run")

        text = widget.text_summary_details.toPlainText()
        self.assertIn("Stored Authoritative Macro Metrics (summary.json)", text)
        self.assertIn("clean_success_rate_macro: 0.85", text)
        self.assertIn("route_completion_macro: 0.92", text)
        widget.close()

    def test_34_strict_integrity_run_id_mismatch_rejected_as_malformed(self):
        """COMPLETE run where run_integrity.json has a different run_id is rejected as malformed."""
        self._create_real_schema_run(
            "integrity_id_mismatch_run",
            status="COMPLETE",
            custom_integrity_run_id="imposter_run_id",
        )
        repo = RunArtifactRepository(self.runs_root)
        snap = repo.load_run_snapshot("integrity_id_mismatch_run")
        self.assertIsNone(snap, "Run with run_integrity.run_id mismatch must be rejected as malformed")

    def test_35_copied_run_with_different_basename_rejected_as_malformed(self):
        """Copying a valid run directory to a different folder basename without rewriting identity is malformed."""
        self._create_real_schema_run("original_run_id")
        copied_dir = self.runs_root / "renamed_folder"
        shutil.copytree(self.runs_root / "original_run_id", copied_dir)

        repo = RunArtifactRepository(self.runs_root)
        snap = repo.load_run_snapshot("renamed_folder")
        self.assertIsNone(snap, "Copied run with mismatched directory basename must be rejected as malformed")

    def test_36_missing_root_artifacts_rejected_as_malformed(self):
        """Missing run_state.json or run_manifest.json returns None."""
        run_dir = self.runs_root / "missing_artifacts_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        repo = RunArtifactRepository(self.runs_root)

        # Neither exists
        self.assertIsNone(repo.load_run_snapshot("missing_artifacts_run"))

        # Only manifest exists
        with open(run_dir / "run_manifest.json", "w") as f:
            json.dump({"run_id": "missing_artifacts_run"}, f)
        self.assertIsNone(repo.load_run_snapshot("missing_artifacts_run"))

        # Only state exists
        (run_dir / "run_manifest.json").unlink()
        with open(run_dir / "run_state.json", "w") as f:
            json.dump({"run_id": "missing_artifacts_run"}, f)
        self.assertIsNone(repo.load_run_snapshot("missing_artifacts_run"))

    def test_37_malformed_json_in_root_artifacts_rejected(self):
        """Corrupted JSON in root manifest or state returns None cleanly."""
        run_dir = self._create_real_schema_run("corrupt_json_run")
        repo = RunArtifactRepository(self.runs_root)

        # Corrupt manifest
        with open(run_dir / "run_manifest.json", "w") as f:
            f.write("{ not valid json")
        self.assertIsNone(repo.load_run_snapshot("corrupt_json_run"))

        # Restore manifest, corrupt state
        self._create_real_schema_run("corrupt_json_run")
        with open(run_dir / "run_state.json", "w") as f:
            f.write("[ not a dict ]")
        self.assertIsNone(repo.load_run_snapshot("corrupt_json_run"))

    def test_38_unverified_run_avoids_authoritative_claims(self):
        """COMPLETE run missing run_integrity.json avoids Authoritative wording in UI."""
        self._create_real_schema_run("unverified_ui_run", status="COMPLETE", omit_integrity=True)
        widget = ResultsWidget(runs_root=self.runs_root)
        widget.show()
        widget.select_run_by_id("unverified_ui_run")

        self.assertIn("UNVERIFIED", widget.cards_group.title())
        self.assertNotIn("Authoritative", widget.cards_group.title())
        self.assertFalse(widget.lbl_trust_warning.isHidden())
        self.assertIn("NOTICE", widget.lbl_trust_warning.text())
        widget.close()

    def test_39_not_final_run_with_summary_displays_provisional_notice(self):
        """RUNNING run with transient summary.json is displayed as provisional/not-final."""
        # Create RUNNING run but inject summary.json
        macro = {"clean_success_rate_macro": 1.0}
        self._create_real_schema_run("provisional_run", status="RUNNING", macro_metrics=macro)
        widget = ResultsWidget(runs_root=self.runs_root)
        widget.show()
        widget.select_run_by_id("provisional_run")

        self.assertIn("PROVISIONAL", widget.cards_group.title())
        self.assertNotIn("Authoritative", widget.cards_group.title())
        self.assertFalse(widget.lbl_trust_warning.isHidden())
        self.assertIn("provisional", widget.lbl_trust_warning.text().lower())
        widget.close()

    def test_40_episode_table_columns_single_sourced_from_platform(self):
        """ResultsWidget episode columns match Gate-7 EPISODE_CSV_COLUMNS exactly."""
        widget = ResultsWidget(runs_root=self.runs_root)
        self.assertEqual(widget._episode_columns, list(EPISODE_CSV_COLUMNS))
        self.assertEqual(len(widget._episode_columns), 31)
        self.assertEqual(widget.table_episodes.columnCount(), 31)
        widget.close()

    def test_41_timing_table_columns_single_sourced_from_dataclass(self):
        """ResultsWidget timing columns match EpisodeTimingRecord dataclass fields exactly."""
        expected_fields = [f.name for f in dataclasses.fields(EpisodeTimingRecord)]
        widget = ResultsWidget(runs_root=self.runs_root)
        self.assertEqual(widget._timing_columns, expected_fields)
        self.assertEqual(len(widget._timing_columns), 8)
        self.assertEqual(widget.table_timing.columnCount(), 8)
        widget.close()

    def test_42_detailed_metadrive_provenance_parity(self):
        """Snapshot view model extracts full environment verification fields from manifest."""
        self._create_real_schema_run("env_prov_run")
        repo = RunArtifactRepository(self.runs_root)
        snap = repo.load_run_snapshot("env_prov_run")

        self.assertEqual(snap.metadrive_version, "0.4.3")
        self.assertEqual(snap.metadrive_commit, "85e5dadc")
        self.assertEqual(snap.metadrive_pinned_version, "0.4.3")
        self.assertEqual(snap.metadrive_pinned_commit, "85e5dadc")
        self.assertEqual(snap.metadrive_verification_status, "EXACT_PIN_VERIFIED")
        self.assertEqual(snap.metadrive_verification_reason, "Exact pin match")

    def test_43_malformed_integrity_json_rejected_as_malformed(self):
        """COMPLETE run with malformed JSON in run_integrity.json returns None."""
        run_dir = self._create_real_schema_run("bad_json_integ_run")
        with open(run_dir / "run_integrity.json", "w", encoding="utf-8") as f:
            f.write("{ invalid json")

        repo = RunArtifactRepository(self.runs_root)
        snap = repo.load_run_snapshot("bad_json_integ_run")
        self.assertIsNone(snap, "Malformed run_integrity JSON must be rejected as malformed")

    def test_44_non_dict_integrity_json_rejected_as_malformed(self):
        """COMPLETE run with non-dict run_integrity.json (list or string) returns None."""
        run_dir = self._create_real_schema_run("list_integ_run")
        with open(run_dir / "run_integrity.json", "w", encoding="utf-8") as f:
            f.write("[\"run_id\", 123]")

        repo = RunArtifactRepository(self.runs_root)
        snap = repo.load_run_snapshot("list_integ_run")
        self.assertIsNone(snap, "Non-dict run_integrity must be rejected as malformed")

    def test_45_missing_or_empty_integrity_run_id_rejected_as_malformed(self):
        """COMPLETE run where run_integrity is missing run_id or has empty run_id returns None."""
        run_dir = self._create_real_schema_run("empty_id_integ_run")
        # Missing run_id
        with open(run_dir / "run_integrity.json", "w", encoding="utf-8") as f:
            json.dump({"experiment_config_sha256": "123"}, f)

        repo = RunArtifactRepository(self.runs_root)
        self.assertIsNone(repo.load_run_snapshot("empty_id_integ_run"))

        # Empty string run_id
        with open(run_dir / "run_integrity.json", "w", encoding="utf-8") as f:
            json.dump({"run_id": "   ", "experiment_config_sha256": "123"}, f)
        self.assertIsNone(repo.load_run_snapshot("empty_id_integ_run"))

    def test_46_missing_outcome_rate_not_fabricated_as_zero(self):
        """Missing or None outcome rates are not converted to 0.0 in the chart."""
        run_dir = self._create_real_schema_run("missing_outcome_run")
        sum_path = run_dir / "summary.json"
        with open(sum_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Set success_rate to 1.0, timeout_rate to 0.0, ensure crash_human_rate is absent
        data["overall_metrics"]["success_rate"] = 1.0
        data["overall_metrics"]["timeout_rate"] = 0.0
        data["overall_metrics"].pop("crash_human_rate", None)

        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Re-save integrity to keep it VERIFIED for inspection
        from src.platform import canonical_json_file_sha256
        with open(run_dir / "run_integrity.json", "r", encoding="utf-8") as f:
            integ_d = json.load(f)
        integ_d["summary_sha256"] = canonical_json_file_sha256(sum_path)
        with open(run_dir / "run_integrity.json", "w", encoding="utf-8") as f:
            json.dump(integ_d, f, indent=2)

        widget = ResultsWidget(runs_root=self.runs_root)
        widget.show()
        widget.select_run_by_id("missing_outcome_run")

        # Check plotted labels in ax_outcomes
        plotted_labels = [tick.get_text() for tick in widget.ax_outcomes.get_xticklabels()]
        self.assertNotIn("Crash Hum", plotted_labels, "Missing outcome rate must NOT be fabricated in chart")
        self.assertIn("Timeout", plotted_labels, "Actual 0.0 rate must remain present in chart")
        widget.close()


if __name__ == "__main__":
    unittest.main()
