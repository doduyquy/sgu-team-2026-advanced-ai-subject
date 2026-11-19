"""UI and artifact-discovery tests for the Stage 5 SAC pilot monitor."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QProcess, QTimer
from PySide6.QtWidgets import QApplication

from src.workbench.widgets.stage5_pilot_widget import Stage5PilotWidget, discover_pilot_runs


class TestStage5PilotMonitor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _make_run(self, root: Path, run_id: str = "s5_demo") -> Path:
        run = root / "runs" / "stage5_pilot" / run_id
        (run / "checkpoints").mkdir(parents=True)
        (run / "run_status.json").write_text(json.dumps({"run_id": run_id, "status": "RUNNING"}), encoding="utf-8")
        (run / "progress.json").write_text(json.dumps({
            "status": "RUNNING", "phase": "TRAINING", "training_step": 500,
            "requested_timesteps": 10000, "steps_per_second": 20.0, "completed_episodes": 2,
            "unique_training_geometries": ["geom-a", "geom-b"], "rss_mb": 300.0,
            "actor_loss": -1.2, "critic_loss": 0.3,
            "action_summary": {"throttle_fraction": .2, "brake_fraction": .1,
                               "near_zero_fraction": .7},
        }), encoding="utf-8")
        records = [
            {"training_step": index, "steering": .1, "throttle_brake": .2,
             "speed_km_h": 10.0, "route_completion": .15}
            for index in range(1, 501)
        ]
        (run / "checkpoints" / "training_steps.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
        report_dir = root / "results" / "stage5" / run_id
        report_dir.mkdir(parents=True)
        report = {
            "checkpoints": {
                "0": {"final_route_completion": .01, "clean_success": 0, "safety_failure": 0,
                      "timeout": 1, "stationary_fraction": 1, "episode_return": -.2},
                "5000": {"final_route_completion": .03, "clean_success": 0, "safety_failure": .1,
                         "timeout": .9, "stationary_fraction": .8, "episode_return": -.3},
            },
            "per_tier": {"Hard": {"0": {"final_route_completion": .02, "clean_success": 0,
                                              "safety_failure": 0, "timeout": 1,
                                              "stationary_fraction": 1, "episode_return": -.2}}},
        }
        (report_dir / "scientific_summary.json").write_text(json.dumps(report), encoding="utf-8")
        return run

    def test_discovers_only_stage5_run_folders(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = self._make_run(root)
            (root / "runs" / "stage5_pilot" / "not_a_run").mkdir()
            self.assertEqual(discover_pilot_runs(root / "runs" / "stage5_pilot"), [run])

    def test_dashboard_loads_live_trace_and_checkpoint_comparison_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = self._make_run(root)
            status_before = (run / "run_status.json").read_bytes()
            widget = Stage5PilotWidget(root)
            try:
                self.assertEqual(widget.combo_runs.count(), 1)
                self.assertEqual(widget.cards["steps"].text(), "500 / 10,000")
                self.assertEqual(widget.cards["geometries"].text(), "2")
                self.assertIn("lệnh ga", widget.lbl_explanation.text().lower())
                self.assertEqual(widget.checkpoint_table.rowCount(), 2)
                self.assertGreater(len(widget.ax_action.lines), 0)

                progress = json.loads((run / "progress.json").read_text(encoding="utf-8"))
                progress.update({"phase": "DIAGNOSTICS", "diagnostic_checkpoint_step": 5000,
                                 "diagnostic_cases_completed": 3, "diagnostic_cases_planned": 24,
                                 "diagnostic_current_case": "geom-current"})
                (run / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
                widget.refresh()
                self.assertIn("checkpoint 5000", widget.progress.format())
                self.assertIn("3/24", widget.progress.format())

                tier_index = widget.combo_tier.findData("Hard")
                widget.combo_tier.setCurrentIndex(tier_index)
                self.assertEqual(widget.checkpoint_table.rowCount(), 1)
                self.assertEqual((run / "run_status.json").read_bytes(), status_before)
            finally:
                widget._timer.stop()
                widget.close()

    def test_legacy_pilot_without_step_telemetry_is_explained(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "runs" / "stage5_pilot" / "legacy"
            run.mkdir(parents=True)
            (run / "run_status.json").write_text(json.dumps({"run_id": "legacy", "status": "COMPLETE"}), encoding="utf-8")
            widget = Stage5PilotWidget(root)
            try:
                self.assertIn("chưa có log action từng bước", widget.lbl_explanation.text().lower())
                self.assertIn("chưa có training_steps", widget.ax_action.texts[0].get_text().lower())
            finally:
                widget._timer.stop()
                widget.close()

    def test_close_stops_timer_and_ignores_queued_refresh(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._make_run(root)
            widget = Stage5PilotWidget(root)
            widget.refresh()
            QTimer.singleShot(0, widget.refresh)
            widget.close()
            self.app.processEvents()
            self.assertTrue(widget._closing)
            self.assertFalse(widget._timer.isActive())

    def test_close_after_run_and_tier_selection_is_safe(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._make_run(root)
            widget = Stage5PilotWidget(root)
            widget.combo_tier.setCurrentIndex(1)
            widget.refresh()
            widget.close()
            self.assertTrue(widget._closing)
            self.assertFalse(widget._timer.isActive())

    def test_close_is_safe_across_repeated_widget_lifecycles(self):
        for _ in range(5):
            widget = Stage5PilotWidget(Path(tempfile.gettempdir()) / "stage5-empty-workbench-test")
            widget.refresh()
            widget.close()
            self.app.processEvents()
            self.assertFalse(widget._timer.isActive())

    def test_process_completion_is_safe_during_close_race(self):
        widget = Stage5PilotWidget(Path(tempfile.gettempdir()) / "stage5-empty-workbench-test")
        process = QProcess(widget)
        widget._viewer_process = process
        process.finished.connect(lambda code, _status, p=process: widget._on_process_finished(p, code))
        process.start(sys.executable, ["-c", "import time; time.sleep(0.15)"])
        self.assertTrue(process.waitForStarted(2000))
        # A direct child-widget close cannot bypass MainWindow's process confirmation.
        widget.close()
        self.assertFalse(widget._closing)
        self.assertTrue(widget.is_viewer_running)
        self.assertTrue(process.waitForFinished(5000))
        self.app.processEvents()
        self.assertFalse(widget.is_viewer_running)
        widget.close()
        self.assertTrue(widget._closing)
        self.assertFalse(widget._timer.isActive())

    def test_queued_process_callback_after_shutdown_does_not_touch_qt_children(self):
        widget = Stage5PilotWidget(Path(tempfile.gettempdir()) / "stage5-empty-workbench-test")
        process = QProcess(widget)
        widget._viewer_process = process
        widget.prepare_for_window_close()
        widget._on_process_finished(process, 0)
        self.app.processEvents()
        self.assertIsNone(widget._viewer_process)
        self.assertFalse(widget._timer.isActive())

    def test_main_window_close_stops_monitor_timer(self):
        from src.workbench.main_window import MainWindow

        window = MainWindow()
        timer = window.stage5_pilot_widget._timer
        self.assertTrue(timer.isActive())
        window.close()
        self.assertFalse(timer.isActive())


if __name__ == "__main__":
    unittest.main()
