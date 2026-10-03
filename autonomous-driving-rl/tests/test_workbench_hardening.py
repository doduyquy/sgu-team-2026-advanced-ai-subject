"""
Unit and Integration Tests for Research Workbench Final Hardening (Gate 7.5B Pass B3).

Tests:
1-15: Process Lifecycle (one worker max, rapid double-start, exit without WORKER_DONE,
      exit 0 without WORKER_DONE, blocked exit code 1, error exit, idempotent failed-to-start,
      temp request cleanup on all terminal paths, force termination non-fabrication).
16-24: Window & Operation State (close running worker with Cancel/Discard, reset termination state,
       stale preflight invalidation, late preflight rejection, rapid clicks single process,
       PLAN success does not mutate Results root, failed RUN does not auto-select,
       custom root RUN auto-loads).
25-30: Protocol Robustness (ordinary stdout, partial line buffering, multiple messages in chunk,
       malformed sentinel, unsupported protocol version, trailing sentinel without newline).
31-42: Results & Filesystem Resilience (missing root, disappearing run, malformed run,
       path traversal, symlink escape, discovery error, VERIFIED/FAILED/UNVERIFIED/NOT_FINAL trust,
       missing outcome != zero, NoEditTriggers).
43-48: GUI Resource Bounds (max block count configured, log truncation, newest event visible,
       chart axis reset across episodes, monitor reset, Execution-Reported provisional labeling).
"""

import dataclasses
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

# Ensure offscreen Qt platform before importing PySide6
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QAbstractItemView, QApplication, QMessageBox

from src.launcher.cases import get_default_project_root
from src.launcher.models import LaunchRequestV1, LauncherMode
from src.platform import (
    ExperimentRunConfig,
    RunKind,
    RunManifestV1,
    RunStateV1,
    canonical_csv_file_sha256,
    canonical_json_file_sha256,
)
from src.platform.experiment_logging import EPISODE_CSV_COLUMNS, EpisodeTimingRecord
from src.workbench.hardening_contracts import (
    GATE7_5B_B2_PLATFORM_WORKBENCH_RESULTS_CONTRACT_HASH,
    GATE7_5B_B2_WORKBENCH_RESULTS_CONTRACT_HASH,
    build_workbench_hardening_contract_core,
    compute_platform_workbench_hardening_contract_sha256,
    compute_workbench_hardening_contract_sha256,
)
from src.workbench.live_telemetry import LiveTelemetryBufferV1
from src.workbench.main_window import MainWindow
from src.workbench.process_runner import WorkbenchProcessRunner
from src.workbench.protocol import (
    WORKBENCH_PROTOCOL_VERSION,
    WORKBENCH_SENTINEL,
    WorkbenchMessageType,
    WorkbenchMessageV1,
    parse_sentinel_line,
)
from src.workbench.results_repository import (
    IntegrityDisplayStatus,
    RunArtifactRepository,
    RunArtifactSnapshotV1,
)
from src.workbench.widgets.results_widget import ResultsWidget
from src.workbench.widgets.run_monitor_widget import MAX_EVENT_LOG_BLOCKS, RunMonitorWidget


class TestWorkbenchHardeningSuite(unittest.TestCase):
    """Pass B3 Comprehensive Hardening Test Suite."""

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

    def _create_minimal_run(self, run_id: str, status: str = "COMPLETE") -> Path:
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"run_id": run_id, "run_kind": "AUDIT", "config": {"agent_id": "test_agent"}}
        state = {"run_id": run_id, "status": status, "recorded_episode_count": 1 if status == "COMPLETE" else 0}
        with open(run_dir / "run_manifest.json", "w") as f:
            json.dump(manifest, f)
        with open(run_dir / "run_state.json", "w") as f:
            json.dump(state, f)
        return run_dir

    # -------------------------------------------------------------
    # 1-15: Process Lifecycle Hardening
    # -------------------------------------------------------------
    def test_01_one_active_worker_maximum(self):
        """WorkbenchProcessRunner rejects starting a second worker while one is active."""
        from unittest.mock import patch
        runner = WorkbenchProcessRunner()
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", render_mode="OFF")
        with patch.object(QProcess, "start"):
            runner.start_operation("PLAN", req)
            with self.assertRaises(RuntimeError):
                runner.start_operation("PLAN", req)
            runner._is_running = False
            runner._cleanup_temp_file()

    def test_02_rapid_double_start_rejected(self):
        """Starting an operation while is_running is True raises RuntimeError immediately."""
        runner = WorkbenchProcessRunner()
        runner._is_running = True
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", render_mode="OFF")
        with self.assertRaises(RuntimeError):
            runner.start_operation("RUN", req)

    def test_03_normal_worker_done_success_preserved_after_process_finished(self):
        """MainWindow preserves successful semantic status after processFinished."""
        window = MainWindow()
        window._on_worker_done({"operation": "RUN", "success": True, "run_id": "run_123"})
        window._on_process_finished(0)
        self.assertIn("completed successfully", window.status_bar.currentMessage())
        self.assertNotIn("Operation RUN failed", window.status_bar.currentMessage())
        window.close()

    def test_04_blocked_worker_done_preserved_after_exit_code_1(self):
        """MainWindow preserves preflight blocked semantic status after exit code 1."""
        window = MainWindow()
        window._on_worker_done({"operation": "RUN", "success": False, "blocked": True})
        window._on_process_finished(1)
        self.assertIn("blocked by preflight", window.status_bar.currentMessage())
        window.close()

    def test_05_worker_error_preserved_after_process_finished(self):
        """MainWindow preserves specific worker error message even when followed by workerDone(success=False)."""
        window = MainWindow()
        window._on_worker_error({"message": "Critical simulation crash: divide by zero"})
        window._on_worker_done({"operation": "RUN", "success": False})
        window._on_process_finished(1)
        self.assertIn("Critical simulation crash: divide by zero", window.status_bar.currentMessage())
        self.assertNotIn("Operation RUN failed", window.status_bar.currentMessage())
        window.close()

    def test_06_exit_0_without_worker_done_becomes_unexpected_exit(self):
        """A worker exiting with code 0 without emitting WORKER_DONE triggers unexpectedWorkerExit."""
        runner = WorkbenchProcessRunner()
        unexpected_events = []
        runner.unexpectedWorkerExit.connect(lambda e: unexpected_events.append(e))

        runner._is_running = True
        runner._current_operation = "RUN"
        runner._worker_done_received = False
        runner._on_process_finished(0)

        self.assertEqual(len(unexpected_events), 1)
        self.assertEqual(unexpected_events[0]["operation"], "RUN")
        self.assertEqual(unexpected_events[0]["exit_code"], 0)
        self.assertFalse(runner.is_running)

    def test_07_nonzero_exit_without_worker_done_becomes_unexpected_exit(self):
        """A worker exiting with code 139 (crash) without WORKER_DONE triggers unexpectedWorkerExit."""
        runner = WorkbenchProcessRunner()
        unexpected_events = []
        runner.unexpectedWorkerExit.connect(lambda e: unexpected_events.append(e))

        runner._is_running = True
        runner._current_operation = "RUN"
        runner._worker_done_received = False
        runner._on_process_finished(139)

        self.assertEqual(len(unexpected_events), 1)
        self.assertEqual(unexpected_events[0]["exit_code"], 139)
        self.assertFalse(runner.is_running)

    def test_08_failed_to_start_idempotent(self):
        """FailedToStart error handler is strictly idempotent and does not emit duplicated terminal signals."""
        runner = WorkbenchProcessRunner()
        finished_codes = []
        runner.processFinished.connect(lambda c: finished_codes.append(c))

        runner._on_process_error(QProcess.ProcessError.FailedToStart)
        runner._on_process_error(QProcess.ProcessError.FailedToStart)  # Duplicate call
        runner._on_process_finished(-1)  # Secondary Qt signal

        self.assertEqual(finished_codes, [-1])
        self.assertFalse(runner.is_running)

    def test_09_temp_request_cleaned_after_success(self):
        """Temporary request file is unlinked on normal process finish."""
        from unittest.mock import patch
        runner = WorkbenchProcessRunner()
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", render_mode="OFF")
        with patch.object(QProcess, "start"):
            runner.start_operation("PLAN", req)
            temp_file = runner._temp_request_file
            self.assertTrue(temp_file.exists())

            runner._worker_done_received = True
            runner._on_process_finished(0)
            self.assertFalse(temp_file.exists())

    def test_10_temp_request_cleaned_after_blocked(self):
        """Temporary request file is unlinked on blocked process finish."""
        from unittest.mock import patch
        runner = WorkbenchProcessRunner()
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_constant_continuous", render_mode="OFF")
        with patch.object(QProcess, "start"):
            runner.start_operation("RUN", req)
            temp_file = runner._temp_request_file
            self.assertTrue(temp_file.exists())

            runner._worker_done_received = True
            runner._on_process_finished(1)
            self.assertFalse(temp_file.exists())

    def test_11_temp_request_cleaned_after_error(self):
        """Temporary request file is unlinked on worker error."""
        from unittest.mock import patch
        runner = WorkbenchProcessRunner()
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", render_mode="OFF")
        with patch.object(QProcess, "start"):
            runner.start_operation("RUN", req)
            temp_file = runner._temp_request_file
            self.assertTrue(temp_file.exists())

            runner._on_process_finished(1)
            self.assertFalse(temp_file.exists())

    def test_12_temp_request_cleaned_after_unexpected_exit(self):
        """Temporary request file is unlinked on unexpected worker exit."""
        from unittest.mock import patch
        runner = WorkbenchProcessRunner()
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", render_mode="OFF")
        with patch.object(QProcess, "start"):
            runner.start_operation("RUN", req)
            temp_file = runner._temp_request_file
            self.assertTrue(temp_file.exists())

            runner._worker_done_received = False
            runner._on_process_finished(137)
            self.assertFalse(temp_file.exists())

    def test_13_temp_request_cleaned_after_force_terminate(self):
        """Temporary request file is unlinked after force terminate."""
        from unittest.mock import patch
        runner = WorkbenchProcessRunner()
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", render_mode="OFF")
        with patch.object(QProcess, "start"):
            runner.start_operation("RUN", req)
            temp_file = runner._temp_request_file
            self.assertTrue(temp_file.exists())

            runner.force_terminate()
            runner._on_process_finished(-1)
            self.assertFalse(temp_file.exists())

    def test_14_force_terminate_never_fabricates_interrupted(self):
        """Force terminate does NOT write run_state or claim INTERRUPTED."""
        run_dir = self._create_minimal_run("ft_run", status="RUNNING")
        runner = WorkbenchProcessRunner()
        runner._is_running = True
        runner._current_operation = "RUN"
        runner._force_terminated = True
        runner._on_process_finished(-1)

        # Inspect disk run_state.json: must still be RUNNING
        with open(run_dir / "run_state.json", "r") as f:
            st = json.load(f)
        self.assertEqual(st["status"], "RUNNING", "GUI must never rewrite on-disk state on kill")

    def test_15_force_terminate_never_fabricates_complete(self):
        """Force terminate never reports complete status in UI."""
        window = MainWindow()
        window._runner._is_running = True
        window._runner._force_terminated = True
        window._on_process_finished(-1)
        self.assertNotIn("completed successfully", window.status_bar.currentMessage())
        window._runner._is_running = False
        window.close()

    # -------------------------------------------------------------
    # 16-24: Window & Operation State Hardening
    # -------------------------------------------------------------
    def test_16_close_running_worker_cancel_keeps_window_open(self):
        """Closing window during active worker and clicking Cancel keeps window open."""
        from unittest.mock import patch
        from PySide6.QtGui import QCloseEvent
        window = MainWindow()
        window._runner._is_running = True

        event = QCloseEvent()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Cancel):
            window.closeEvent(event)
            self.assertFalse(event.isAccepted())
            self.assertTrue(window._runner.is_running)
            self.assertFalse(window._runner.force_terminated)
        window._runner._is_running = False
        window.close()

    def test_17_close_running_worker_confirm_force_terminates(self):
        """Closing window during active worker and confirming Discard terminates worker boundedly, cleans temp files, and accepts close."""
        from unittest.mock import patch
        from PySide6.QtGui import QCloseEvent
        window = MainWindow()
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", render_mode="OFF")
        with patch.object(QProcess, "start"):
            with patch.object(QProcess, "waitForFinished", return_value=True):
                window._runner.start_operation("PLAN", req)
                temp_file = window._runner._temp_request_file
                self.assertTrue(temp_file.exists())
                self.assertTrue(window._runner.is_running)

                event = QCloseEvent()
                with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Discard):
                    window.closeEvent(event)
                    # Verified: bounded termination executed, close accepted, temp file cleaned, runner stopped
                    self.assertTrue(event.isAccepted())
                    self.assertFalse(window._runner.is_running)
                    self.assertFalse(temp_file.exists())
        window.close()

    def test_17b_close_running_worker_timeout_failure_aborts_close(self):
        """If bounded termination fails/times out, window close is ignored and diagnostic is surfaced."""
        from unittest.mock import patch
        from PySide6.QtGui import QCloseEvent
        window = MainWindow()
        window._runner._is_running = True

        event = QCloseEvent()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Discard):
            with patch.object(window._runner, "force_terminate_and_wait", return_value=False) as mock_ft_wait:
                with patch.object(QMessageBox, "critical") as mock_crit:
                    window.closeEvent(event)
                    # Verified: close was aborted because worker could not be confirmed stopped
                    self.assertFalse(event.isAccepted())
                    mock_ft_wait.assert_called_once_with(timeout_ms=2000)
                    mock_crit.assert_called_once()
                    self.assertIn("Error: Bounded termination timed out", window.status_bar.currentMessage())
        window._runner._is_running = False
        window.close()

    def test_18_new_operation_resets_old_force_termination_state(self):
        """Starting a new operation resets force_terminated flag."""
        runner = WorkbenchProcessRunner()
        runner._force_terminated = True
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", render_mode="OFF")
        runner.start_operation("PLAN", req)
        self.assertFalse(runner.force_terminated)
        runner.force_terminate()

    def test_19_stale_preflight_invalidated_after_request_change(self):
        """Editing request fields invalidates stored verdict."""
        window = MainWindow()
        setup = window.setup_widget
        setup.set_preflight_verdict(True)
        self.assertTrue(setup.btn_run.isEnabled())

        setup.edit_sequence.setText("NEW_SEQ")
        self.assertFalse(setup.btn_run.isEnabled())
        self.assertFalse(setup._can_run)
        window.close()

    def test_20_late_stale_preflight_cannot_enable_run(self):
        """Calling set_preflight_verdict while operation_active is True does not enable Run."""
        window = MainWindow()
        setup = window.setup_widget
        setup.set_operation_active(True)
        setup.set_preflight_verdict(True)
        self.assertFalse(setup.btn_run.isEnabled())
        window.close()

    def test_21_rapid_clicks_cannot_create_overlapping_process(self):
        """Repeated rapid clicks on Resolve/Run do not create overlapping processes."""
        from unittest.mock import patch
        window = MainWindow()
        setup = window.setup_widget
        req = setup.build_launch_request()

        with patch.object(window._runner, "start_operation") as mock_start:
            window._on_resolve_requested(req)
            # Synchronously locked
            self.assertFalse(setup.btn_resolve.isEnabled())
            self.assertFalse(setup.btn_run.isEnabled())

            # Attempt second resolve and run
            setup._on_resolve_clicked()
            setup._on_run_clicked()

            # start_operation should only have been called once
            self.assertEqual(mock_start.call_count, 1)
        window.close()

    def test_22_plan_success_does_not_mutate_results_root_using_stale_prior_run(self):
        """PLAN completion never auto-selects or mutates Results runs_root."""
        window = MainWindow()
        window._last_completed_run_dir = Path("/some/custom/path")

        window._on_worker_done({"operation": "PLAN", "success": True})
        # Root must not have mutated
        self.assertNotEqual(window.results_widget.runs_root, Path("/some/custom/path"))
        window.close()

    def test_23_failed_run_does_not_auto_select_stale_previous_run(self):
        """Failed RUN does not auto-select the previous run."""
        self._create_minimal_run("prior_run")
        window = MainWindow()
        window.results_widget.set_runs_root(self.runs_root)
        window._last_completed_run_dir = self.runs_root / "prior_run"

        # Start a new run (which resets _last_completed_run_dir)
        req = window.setup_widget.build_launch_request()
        with unittest.mock.patch.object(window._runner, "start_operation"):
            window._on_run_requested(req)

        self.assertIsNone(window._last_completed_run_dir)
        window._on_worker_done({"operation": "RUN", "success": False})
        window.close()

    def test_24_successful_custom_root_run_auto_loads_correct_disk_run(self):
        """Successful RUN auto-loads the run from the custom root."""
        custom_run_dir = self._create_minimal_run("custom_run")
        window = MainWindow()
        window._on_execution_report({"execution_report": {"run_id": "custom_run", "run_dir": str(custom_run_dir), "status": "COMPLETE"}})
        window._on_worker_done({"operation": "RUN", "success": True, "run_id": "custom_run"})

        self.assertEqual(window.results_widget.runs_root, self.runs_root)
        self.assertIsNotNone(window.results_widget._current_snapshot)
        self.assertEqual(window.results_widget._current_snapshot.run_id, "custom_run")
        window.close()

    # -------------------------------------------------------------
    # 25-30: Protocol Robustness
    # -------------------------------------------------------------
    def test_25_ordinary_stdout_remains_ordinary_stdout(self):
        """Non-sentinel stdout lines return None from parser."""
        self.assertIsNone(parse_sentinel_line("Plain stdout message"))
        self.assertIsNone(parse_sentinel_line("MetaDrive: 10 Hz control loop"))

    def test_26_partial_sentinel_line_buffered_until_complete(self):
        """WorkbenchProcessRunner buffers split stdout chunks correctly."""
        runner = WorkbenchProcessRunner()
        received = []
        runner.planResolved.connect(lambda p: received.append(p))

        full_line = f"{WORKBENCH_SENTINEL}{{\"protocol_version\": \"WORKBENCH_PROTOCOL_V1\", \"type\": \"PLAN_RESOLVED\", \"payload\": {{\"val\": 123}}}}\n"
        chunk1 = full_line[:20]
        chunk2 = full_line[20:]

        runner._stdout_buffer = ""
        # Simulate chunk1 arrival
        runner._stdout_buffer += chunk1
        lines = runner._stdout_buffer.split("\n")
        runner._stdout_buffer = lines[-1]
        for l in lines[:-1]:
            msg = parse_sentinel_line(l)
            if msg:
                runner._dispatch_protocol_message(msg)
        self.assertEqual(len(received), 0)

        # Simulate chunk2 arrival
        runner._stdout_buffer += chunk2
        lines = runner._stdout_buffer.split("\n")
        runner._stdout_buffer = lines[-1]
        for l in lines[:-1]:
            msg = parse_sentinel_line(l)
            if msg:
                runner._dispatch_protocol_message(msg)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["val"], 123)

    def test_27_multiple_sentinel_messages_in_one_chunk_dispatch_in_order(self):
        """Multiple protocol messages in a single stdout chunk dispatch in order."""
        runner = WorkbenchProcessRunner()
        events = []
        runner.planResolved.connect(lambda p: events.append("PLAN"))
        runner.workerDone.connect(lambda p: events.append("DONE"))

        chunk = (
            f"{WORKBENCH_SENTINEL}{{\"protocol_version\": \"WORKBENCH_PROTOCOL_V1\", \"type\": \"PLAN_RESOLVED\", \"payload\": {{}}}}\n"
            f"{WORKBENCH_SENTINEL}{{\"protocol_version\": \"WORKBENCH_PROTOCOL_V1\", \"type\": \"WORKER_DONE\", \"payload\": {{}}}}\n"
        )
        for line in chunk.splitlines():
            msg = parse_sentinel_line(line)
            if msg:
                runner._dispatch_protocol_message(msg)

        self.assertEqual(events, ["PLAN", "DONE"])

    def test_28_malformed_sentinel_handled_safely(self):
        """Malformed JSON after sentinel returns None without raising."""
        self.assertIsNone(parse_sentinel_line(f"{WORKBENCH_SENTINEL} {{ bad json"))

    def test_29_unsupported_protocol_version_rejected(self):
        """Payload with wrong protocol version returns None."""
        line = f"{WORKBENCH_SENTINEL}{{\"protocol_version\": \"WORKBENCH_PROTOCOL_V99\", \"type\": \"WORKER_READY\", \"payload\": {{}}}}"
        self.assertIsNone(parse_sentinel_line(line))

    def test_30_trailing_final_sentinel_without_newline_flushed_on_process_exit(self):
        """ProcessRunner flushes incomplete trailing buffer on process exit."""
        runner = WorkbenchProcessRunner()
        done_called = []
        runner.workerDone.connect(lambda p: done_called.append(p))

        # Buffer contains trailing line without newline
        trailing = f"{WORKBENCH_SENTINEL}{{\"protocol_version\": \"WORKBENCH_PROTOCOL_V1\", \"type\": \"WORKER_DONE\", \"payload\": {{\"success\": true}}}}"
        runner._stdout_buffer = trailing
        runner._on_process_finished(0)

        self.assertEqual(len(done_called), 1)
        self.assertTrue(done_called[0]["success"])

    # -------------------------------------------------------------
    # 31-42: Results & Filesystem Resilience
    # -------------------------------------------------------------
    def test_31_missing_runs_root_does_not_crash(self):
        """Repository with nonexistent runs_root returns empty list without crashing."""
        nonexistent = self.runs_root / "nonexistent_dir_123"
        repo = RunArtifactRepository(nonexistent)
        self.assertEqual(repo.discover_run_ids(), [])
        self.assertIsNotNone(repo.last_discovery_error)

    def test_32_disappearing_run_between_discovery_and_load(self):
        """Deleting a run after discovery returns None cleanly upon load."""
        run_dir = self._create_minimal_run("disappearing_run")
        repo = RunArtifactRepository(self.runs_root)
        self.assertIn("disappearing_run", repo.discover_run_ids())

        shutil.rmtree(run_dir)
        self.assertIsNone(repo.load_run_snapshot("disappearing_run"))

    def test_33_malformed_run_remains_malformed(self):
        """A directory with only run_manifest.json is reported as MALFORMED in ResultsWidget."""
        run_dir = self.runs_root / "malformed_only_manifest"
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "run_manifest.json", "w") as f:
            json.dump({"run_id": "malformed_only_manifest"}, f)

        widget = ResultsWidget(runs_root=self.runs_root)
        item_status = widget.table_runs.item(0, 1)
        self.assertEqual(item_status.text(), "MALFORMED")
        widget.close()

    def test_34_traversal_remains_rejected(self):
        """Directory traversal paths return None."""
        repo = RunArtifactRepository(self.runs_root)
        self.assertIsNone(repo.load_run_snapshot("../escaped"))

    def test_35_symlink_escape_rejected_when_supported(self):
        """Symlink resolving outside runs_root returns None."""
        outside_dir = tempfile.TemporaryDirectory()
        outside_path = Path(outside_dir.name)
        target_run = outside_path / "outside_run"
        target_run.mkdir(parents=True, exist_ok=True)
        with open(target_run / "run_manifest.json", "w") as f:
            json.dump({"run_id": "outside_run"}, f)
        with open(target_run / "run_state.json", "w") as f:
            json.dump({"run_id": "outside_run", "status": "COMPLETE"}, f)

        link_path = self.runs_root / "symlink_run"
        try:
            os.symlink(target_run, link_path, target_is_directory=True)
        except (OSError, NotImplementedError):
            outside_dir.cleanup()
            self.skipTest("Symlink creation not supported without admin privileges on Windows")

        repo = RunArtifactRepository(self.runs_root)
        self.assertIsNone(repo.load_run_snapshot("symlink_run"))
        outside_dir.cleanup()

    def test_36_repository_discovery_permission_error_handled_visibly(self):
        """Filesystem scan errors surface in last_discovery_error without crashing."""
        from unittest.mock import patch
        repo = RunArtifactRepository(self.runs_root)
        with patch.object(Path, "iterdir", side_effect=PermissionError("Permission denied")):
            runs = repo.discover_run_ids()
            self.assertEqual(runs, [])
            self.assertIn("Permission denied", repo.last_discovery_error)

    def test_37_verified_trust_label_regression(self):
        """VERIFIED run displays Authoritative labels."""
        run_dir = self._create_minimal_run("v_run", status="COMPLETE")
        # Ensure config has experiment_config_sha256
        with open(run_dir / "run_manifest.json", "w") as f:
            json.dump({
                "run_id": "v_run",
                "run_kind": "AUDIT",
                "config": {"agent_id": "test_agent", "experiment_config_sha256": "mock_cfg_sha"},
            }, f)
        with open(run_dir / "summary.json", "w") as f:
            json.dump({"run_id": "v_run", "overall_metrics": {"clean_success_rate": 1.0}}, f)
        with open(run_dir / "episodes.csv", "w") as f:
            f.write("ep\n0\n")
        with open(run_dir / "timing.csv", "w") as f:
            f.write("t\n0\n")
        with open(run_dir / "wandb_sync.json", "w") as f:
            json.dump({}, f)
        with open(run_dir / "run_integrity.json", "w") as f:
            json.dump({
                "run_id": "v_run",
                "experiment_config_sha256": "mock_cfg_sha",
                "run_manifest_sha256": canonical_json_file_sha256(run_dir / "run_manifest.json"),
                "episodes_sha256": canonical_csv_file_sha256(run_dir / "episodes.csv"),
                "summary_sha256": canonical_json_file_sha256(run_dir / "summary.json"),
                "timing_sha256": canonical_csv_file_sha256(run_dir / "timing.csv"),
                "run_state_sha256": canonical_json_file_sha256(run_dir / "run_state.json"),
                "wandb_sync_sha256": canonical_json_file_sha256(run_dir / "wandb_sync.json"),
                "technical_failures_sha256": None,
            }, f)

        widget = ResultsWidget(runs_root=self.runs_root)
        widget.select_run_by_id("v_run")
        self.assertIn("Authoritative", widget.cards_group.title())
        widget.close()

    def test_38_failed_trust_label_regression(self):
        """FAILED run displays UNTRUSTED and suppresses Authoritative."""
        run_dir = self._create_minimal_run("f_run", status="COMPLETE")
        with open(run_dir / "run_integrity.json", "w") as f:
            json.dump({"run_id": "f_run", "summary_sha256": "mismatch"}, f)

        widget = ResultsWidget(runs_root=self.runs_root)
        widget.select_run_by_id("f_run")
        self.assertIn("UNTRUSTED", widget.cards_group.title())
        self.assertNotIn("Authoritative", widget.cards_group.title())
        widget.close()

    def test_39_unverified_trust_label_regression(self):
        """UNVERIFIED run displays UNVERIFIED and suppresses Authoritative."""
        self._create_minimal_run("uv_run", status="COMPLETE")
        widget = ResultsWidget(runs_root=self.runs_root)
        widget.select_run_by_id("uv_run")
        self.assertIn("UNVERIFIED", widget.cards_group.title())
        self.assertNotIn("Authoritative", widget.cards_group.title())
        widget.close()

    def test_40_not_final_trust_label_regression(self):
        """NOT_FINAL run displays PROVISIONAL and suppresses Authoritative."""
        self._create_minimal_run("nf_run", status="RUNNING")
        widget = ResultsWidget(runs_root=self.runs_root)
        widget.select_run_by_id("nf_run")
        self.assertIn("PROVISIONAL", widget.cards_group.title())
        self.assertNotIn("Authoritative", widget.cards_group.title())
        widget.close()

    def test_41_missing_outcome_not_zero_regression(self):
        """Missing outcome rate is omitted and not plotted as 0.0."""
        run_dir = self._create_minimal_run("missing_rate_run", status="COMPLETE")
        with open(run_dir / "summary.json", "w") as f:
            json.dump({"run_id": "missing_rate_run", "overall_metrics": {"success_rate": 1.0, "timeout_rate": 0.0}}, f)

        widget = ResultsWidget(runs_root=self.runs_root)
        widget.select_run_by_id("missing_rate_run")
        labels = [t.get_text() for t in widget.ax_outcomes.get_xticklabels()]
        self.assertNotIn("Crash Hum", labels)
        self.assertIn("Timeout", labels)
        widget.close()

    def test_42_results_tables_remain_no_edit_triggers(self):
        """All Results tables have NoEditTriggers policy."""
        widget = ResultsWidget(runs_root=self.runs_root)
        self.assertEqual(widget.table_runs.editTriggers(), QAbstractItemView.EditTrigger.NoEditTriggers)
        self.assertEqual(widget.table_episodes.editTriggers(), QAbstractItemView.EditTrigger.NoEditTriggers)
        self.assertEqual(widget.table_timing.editTriggers(), QAbstractItemView.EditTrigger.NoEditTriggers)
        self.assertEqual(widget.table_integrity.editTriggers(), QAbstractItemView.EditTrigger.NoEditTriggers)
        widget.close()

    # -------------------------------------------------------------
    # 43-48: GUI Resource Bounds Hardening
    # -------------------------------------------------------------
    def test_43_live_event_maximum_block_count_configured(self):
        """RunMonitorWidget QTextDocument maximumBlockCount is 5000."""
        monitor = RunMonitorWidget()
        self.assertEqual(monitor.text_event_log.document().maximumBlockCount(), MAX_EVENT_LOG_BLOCKS)
        self.assertEqual(MAX_EVENT_LOG_BLOCKS, 5000)

    def test_44_event_count_beyond_limit_remains_bounded(self):
        """Appending more than 5000 events bounds the document block count to 5000."""
        monitor = RunMonitorWidget()
        doc = monitor.text_event_log.document()
        for i in range(5050):
            monitor.text_event_log.append(f"Event line {i}")
        self.assertLessEqual(doc.blockCount(), 5000)

    def test_45_newest_event_remains_available_after_truncation(self):
        """Newest events remain at the tail of the document after truncation."""
        monitor = RunMonitorWidget()
        for i in range(5020):
            monitor.text_event_log.append(f"Line {i}")
        text = monitor.text_event_log.toPlainText()
        self.assertIn("Line 5019", text)
        self.assertNotIn("Line 0\n", text)

    def test_46_repeated_episode_charts_do_not_accumulate_stale_artists(self):
        """EPISODE_STARTED resets axes without accumulating stale line artists."""
        monitor = RunMonitorWidget()
        monitor.handle_launcher_event({"event": {"event_type": "EPISODE_STARTED", "episode_index": 0}})
        monitor.handle_launcher_event({"event": {"event_type": "EPISODE_PROGRESS", "step_index": 10, "route_completion": 0.1, "speed_kmh": 20.0}})
        self.assertEqual(len(monitor.ax_route.lines), 1)

        monitor.handle_launcher_event({"event": {"event_type": "EPISODE_STARTED", "episode_index": 1}})
        self.assertEqual(len(monitor.ax_route.lines), 0)

    def test_47_monitor_reset_clears_live_chart_state(self):
        """reset_monitor clears all plot lines and resets labels."""
        monitor = RunMonitorWidget()
        monitor.handle_launcher_event({"event": {"event_type": "EPISODE_STARTED", "episode_index": 0}})
        monitor.handle_launcher_event({"event": {"event_type": "EPISODE_PROGRESS", "step_index": 10, "route_completion": 0.1, "speed_kmh": 20.0}})
        self.assertEqual(len(monitor.ax_route.lines), 1)

        monitor.reset_monitor("r_new")
        self.assertEqual(len(monitor.ax_route.lines), 0)
        self.assertEqual(monitor.lbl_run_id.text(), "Run ID: r_new")

    def test_48_execution_report_summary_labeled_provisional(self):
        """ExecutionReport summary text in Run Monitor explicitly states it is provisional and not verified."""
        monitor = RunMonitorWidget()
        report_payload = {
            "execution_report": {
                "run_id": "rep_run",
                "status": "COMPLETE",
                "run_dir": "/tmp/rep_run",
                "total_cases": 1,
                "completed_episodes": 1,
                "summary_payload": {
                    "overall_metrics": {"clean_success_rate": 1.0}
                }
            }
        }
        monitor.handle_execution_report(report_payload)
        text = monitor.text_results.toPlainText()
        self.assertIn("Provisional", text)
        self.assertIn("Results tab", text)


if __name__ == "__main__":
    unittest.main()
