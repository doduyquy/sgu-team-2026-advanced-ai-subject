"""
Unit and Integration Tests for Research Workbench Results Browser, Artifact Integrity & Live UX (Gate 7.5B Pass B2).

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
"""

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

    def _create_mock_run(
        self,
        run_id: str,
        status: str = "COMPLETE",
        tamper_file: Optional[str] = None,
        omit_integrity: bool = False,
    ) -> Path:
        """Helper to create a standard mock run directory with Gate-7 artifacts."""
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        config_data = {
            "run_id": run_id,
            "run_kind": "AUDIT",
            "agent_descriptor": {"agent_id": "test_agent", "agent_version": "1.0.0"},
            "expected_episode_count": 1,
            "experiment_config_sha256": "mock_cfg_sha256_12345",
        }
        manifest_data = {
            "run_id": run_id,
            "run_kind": "AUDIT",
            "canonical_run": False,
            "config": config_data,
            "provenance": {
                "git": {"git_commit_sha": "abc12345", "git_worktree_dirty": False},
                "environment": {"metadrive_version": "0.4.3", "metadrive_commit": "85e5dadc"},
            },
        }
        state_data = {
            "run_id": run_id,
            "status": status,
            "started_at_utc": "2026-10-02T12:00:00Z",
            "finished_at_utc": "2026-10-02T12:01:00Z" if status == "COMPLETE" else None,
            "recorded_episode_count": 1 if status == "COMPLETE" else 0,
        }
        summary_data = {
            "run_id": run_id,
            "overall_metrics": {
                "clean_success_rate": 1.0,
                "safety_failure_rate": 0.0,
                "mean_final_route_completion": 1.0,
                "median_final_route_completion": 1.0,
                "mean_time_to_clean_success_s": 24.5,
                "mean_episode_return": 18.2,
                "success_rate": 1.0,
            },
            "tier_metrics": {
                "Easy": {"clean_success_rate": 1.0, "mean_final_route_completion": 1.0}
            },
        }

        # Write core files
        with open(run_dir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        with open(run_dir / "run_state.json", "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2)

        if status == "COMPLETE":
            with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2)

            episodes_csv_content = (
                "episode_index,case_id,split,tier,primary_terminal_reason,clean_success,final_route_completion,episode_steps,episode_time_s,episode_return\n"
                "0,case_1,TRAIN,Easy,SUCCESS,True,1.0,245,24.5,18.2\n"
            )
            with open(run_dir / "episodes.csv", "w", encoding="utf-8", newline="") as f:
                f.write(episodes_csv_content)

            timing_csv_content = (
                "episode_index,act_count,mean_act_ms,median_act_ms,p95_act_ms,max_act_ms\n"
                "0,245,0.45,0.42,0.61,1.2\n"
            )
            with open(run_dir / "timing.csv", "w", encoding="utf-8", newline="") as f:
                f.write(timing_csv_content)

            wandb_sync_data = {"mode": "DISABLED", "sync_status": "NOT_SYNCED"}
            with open(run_dir / "wandb_sync.json", "w", encoding="utf-8") as f:
                json.dump(wandb_sync_data, f, indent=2)

            if not omit_integrity:
                from src.platform import canonical_csv_file_sha256, canonical_json_file_sha256, canonical_json_sha256
                integrity_data = {
                    "run_id": run_id,
                    "experiment_config_sha256": "mock_cfg_sha256_12345",
                    "run_manifest_sha256": canonical_json_file_sha256(run_dir / "run_manifest.json"),
                    "episodes_sha256": canonical_csv_file_sha256(run_dir / "episodes.csv"),
                    "summary_sha256": canonical_json_file_sha256(run_dir / "summary.json"),
                    "timing_sha256": canonical_csv_file_sha256(run_dir / "timing.csv"),
                    "run_state_sha256": canonical_json_file_sha256(run_dir / "run_state.json"),
                    "wandb_sync_sha256": canonical_json_file_sha256(run_dir / "wandb_sync.json"),
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
        self._create_mock_run("run_b")
        self._create_mock_run("run_a")
        repo = RunArtifactRepository(self.runs_root)
        discovered = repo.discover_run_ids()
        self.assertEqual(discovered, ["run_a", "run_b"])

    def test_02_random_unrelated_directories_are_ignored(self):
        """Random subdirectories without run_state or run_manifest are ignored."""
        (self.runs_root / "random_folder").mkdir(parents=True, exist_ok=True)
        (self.runs_root / "empty_dir").mkdir(parents=True, exist_ok=True)
        self._create_mock_run("valid_run")
        repo = RunArtifactRepository(self.runs_root)
        discovered = repo.discover_run_ids()
        self.assertEqual(discovered, ["valid_run"])

    def test_03_complete_run_loads_all_required_artifacts(self):
        """Snapshot loads manifest, state, summary, episodes, timing, and wandb_sync."""
        self._create_mock_run("complete_run_01")
        repo = RunArtifactRepository(self.runs_root)
        snap = repo.load_run_snapshot("complete_run_01")
        self.assertIsNotNone(snap)
        self.assertEqual(snap.status, "COMPLETE")
        self.assertIsNotNone(snap.summary_payload)
        self.assertEqual(len(snap.episode_rows), 1)
        self.assertEqual(len(snap.timing_rows), 1)

    def test_04_complete_untampered_run_integrity_verifies(self):
        """Untampered COMPLETE run evaluates to VERIFIED integrity status."""
        self._create_mock_run("clean_run")
        repo = RunArtifactRepository(self.runs_root)
        snap = repo.load_run_snapshot("clean_run")
        self.assertEqual(snap.integrity_status, IntegrityDisplayStatus.VERIFIED)

    def test_05_tampered_episodes_csv_produces_integrity_failure(self):
        """Tampering with episodes.csv marks integrity as FAILED and identifies mismatch."""
        self._create_mock_run("tampered_ep_run", tamper_file="episodes.csv")
        repo = RunArtifactRepository(self.runs_root)
        snap = repo.load_run_snapshot("tampered_ep_run")
        self.assertEqual(snap.integrity_status, IntegrityDisplayStatus.FAILED)
        ep_check = next(c for c in snap.integrity_details if c.artifact_name == "episodes.csv")
        self.assertEqual(ep_check.status, "FAIL")

    def test_06_tampered_summary_json_produces_integrity_failure(self):
        """Tampering with summary.json marks integrity as FAILED."""
        self._create_mock_run("tampered_sum_run", tamper_file="summary.json")
        repo = RunArtifactRepository(self.runs_root)
        snap = repo.load_run_snapshot("tampered_sum_run")
        self.assertEqual(snap.integrity_status, IntegrityDisplayStatus.FAILED)
        sum_check = next(c for c in snap.integrity_details if c.artifact_name == "summary.json")
        self.assertEqual(sum_check.status, "FAIL")

    def test_07_hash_failure_never_rewrites_run_integrity(self):
        """Evaluating integrity never mutates run_integrity.json on disk."""
        run_dir = self._create_mock_run("tamper_check_run", tamper_file="episodes.csv")
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
        self._create_mock_run("running_run", status="RUNNING")
        repo = RunArtifactRepository(self.runs_root)
        snap = repo.load_run_snapshot("running_run")
        self.assertEqual(snap.status, "RUNNING")
        self.assertEqual(snap.integrity_status, IntegrityDisplayStatus.NOT_FINAL)

    def test_09_failed_run_loads_without_fabricated_summary(self):
        """FAILED run loads failure category without fabricating summary payload."""
        run_dir = self._create_mock_run("failed_run", status="FAILED")
        with open(run_dir / "run_state.json", "w", encoding="utf-8") as f:
            json.dump({
                "run_id": "failed_run",
                "status": "FAILED",
                "failure_category": "SIMULATION_CRASH",
                "failure_message": "Segmentation fault in physics engine",
            }, f, indent=2)

        repo = RunArtifactRepository(self.runs_root)
        snap = repo.load_run_snapshot("failed_run")
        self.assertEqual(snap.status, "FAILED")
        self.assertIsNone(snap.summary_payload)
        self.assertEqual(snap.failure_category, "SIMULATION_CRASH")
        self.assertEqual(snap.integrity_status, IntegrityDisplayStatus.NOT_FINAL)

    def test_10_interrupted_run_loads_without_fabricated_summary(self):
        """INTERRUPTED run loads truthfully without fabricated summary."""
        self._create_mock_run("interrupted_run", status="INTERRUPTED")
        repo = RunArtifactRepository(self.runs_root)
        snap = repo.load_run_snapshot("interrupted_run")
        self.assertEqual(snap.status, "INTERRUPTED")
        self.assertIsNone(snap.summary_payload)
        self.assertEqual(snap.integrity_status, IntegrityDisplayStatus.NOT_FINAL)

    def test_11_complete_missing_integrity_is_unverified(self):
        """COMPLETE run missing run_integrity.json evaluates to UNVERIFIED, never VERIFIED."""
        self._create_mock_run("missing_integ_run", status="COMPLETE", omit_integrity=True)
        repo = RunArtifactRepository(self.runs_root)
        snap = repo.load_run_snapshot("missing_integ_run")
        self.assertEqual(snap.status, "COMPLETE")
        self.assertEqual(snap.integrity_status, IntegrityDisplayStatus.UNVERIFIED)

    def test_12_summary_cards_use_values_directly_from_summary_json(self):
        """Overview metric cards display exact values from summary.json['overall_metrics']."""
        self._create_mock_run("cards_test_run")
        widget = ResultsWidget(runs_root=self.runs_root)
        widget.select_run_by_id("cards_test_run")

        lbl_success = widget.card_success.findChild(object, "value_label")
        lbl_mean_route = widget.card_mean_route.findChild(object, "value_label")
        self.assertEqual(lbl_success.text(), "100.0%")
        self.assertEqual(lbl_mean_route.text(), "100.0%")
        widget.close()

    def test_13_none_time_to_success_remains_na(self):
        """mean_time_to_clean_success_s = None is displayed as N/A, never zero."""
        run_dir = self._create_mock_run("no_clean_success_run")
        # Update summary with mean_time_to_clean_success_s = None
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
        """Overview text and tab explicitly label return as diagnostic training signal."""
        self._create_mock_run("diag_label_run")
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
        # Message says SUCCESS but structured status says FAILURE
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
        self._create_mock_run("auto_refresh_run")
        window = MainWindow()
        window.results_widget.set_runs_root(self.runs_root)

        # Emit worker done with run_id
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


if __name__ == "__main__":
    unittest.main()
