"""
Research Workbench Main Window (Gate 7.5B).

Main window coordinating:
1. Experiment Setup Widget
2. Plan & Preflight Widget
3. Live Run Monitor Widget
4. Agent Explorer Widget
5. Case Explorer Widget

Handles:
- ProcessRunner signals and state orchestration.
- Tab auto-switching on resolve/run.
- Window close confirmation dialog when worker is actively executing (Specification Section 22).
"""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.launcher.models import LaunchRequestV1
from src.workbench.core_adapter import CoreAdapter
from src.workbench.campaign_controller import CampaignController
from src.workbench.process_runner import WorkbenchProcessRunner
from src.workbench.widgets.agent_explorer_widget import AgentExplorerWidget
from src.workbench.widgets.benchmark_widget import BenchmarkWidget
from src.workbench.widgets.case_explorer_widget import CaseExplorerWidget
from src.workbench.widgets.experiment_setup_widget import ExperimentSetupWidget
from src.workbench.widgets.plan_preflight_widget import PlanAndPreflightWidget
from src.workbench.widgets.results_widget import ResultsWidget
from src.workbench.widgets.run_monitor_widget import RunMonitorWidget
from src.workbench.widgets.training_widget import TrainingWidget


class MainWindow(QMainWindow):
    """Main window for Research Workbench V1."""

    def __init__(self, core_adapter: Optional[CoreAdapter] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Research Workbench V1 — Presentation & Control Layer (Gate 7.5B)")
        self.resize(1100, 750)

        self._core_adapter = core_adapter or CoreAdapter()
        self._runner = WorkbenchProcessRunner(self)
        self.campaign_controller = CampaignController(self._runner, self._core_adapter.project_root, self)
        self._last_completed_run_dir: Optional[Path] = None
        self._terminal_semantic_status: Optional[str] = None
        self._training_active = False
        self._training_done = None

        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.tab_widget = QTabWidget(self)

        # Tab 1: Experiment Setup
        self.setup_widget = ExperimentSetupWidget(self._core_adapter)
        self.benchmark_widget = BenchmarkWidget(self._core_adapter, self.campaign_controller)
        self.setup_pages = QTabWidget()
        guided_scroll = QScrollArea()
        guided_scroll.setWidgetResizable(True)
        guided_scroll.setWidget(self.benchmark_widget)
        self.setup_pages.addTab(guided_scroll, "Bộ đánh giá cố định")
        self.setup_pages.addTab(self.setup_widget, "Nâng cao · chạy thủ công")
        self.training_widget = TrainingWidget(self._core_adapter.project_root)
        training_scroll = QScrollArea()
        training_scroll.setWidgetResizable(True)
        training_scroll.setWidget(self.training_widget)
        self.setup_pages.addTab(training_scroll, "Train · A* + heuristic học được")
        self.tab_widget.addTab(self.setup_pages, "1. Experiment Setup")

        # Tab 2: Plan & Preflight
        self.plan_preflight_widget = PlanAndPreflightWidget()
        self.tab_widget.addTab(self.plan_preflight_widget, "2. Plan & Preflight")

        # Tab 3: Run Monitor
        self.run_monitor_widget = RunMonitorWidget()
        self.tab_widget.addTab(self.run_monitor_widget, "3. Run Monitor")

        # Tab 4: Agent Explorer
        self.agent_explorer_widget = AgentExplorerWidget(self._core_adapter)
        self.tab_widget.addTab(self.agent_explorer_widget, "4. Agents Explorer")

        # Tab 5: Case Explorer
        self.case_explorer_widget = CaseExplorerWidget(self._core_adapter)
        self.tab_widget.addTab(self.case_explorer_widget, "5. Case Explorer")

        # Tab 6: Results Browser (Pass B2)
        self.results_widget = ResultsWidget()
        self.tab_widget.addTab(self.results_widget, "6. Results")

        main_layout.addWidget(self.tab_widget)

        # Status Bar
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Workbench Ready. Scientific authority strictly maintained in Gate 7.5A Core.")

    def _connect_signals(self) -> None:
        self.benchmark_widget.operationRequested.connect(self._on_campaign_requested)
        self.benchmark_widget.selectionChanged.connect(self._on_campaign_selection_changed)
        self.benchmark_widget.terminateRequested.connect(self._on_terminate_requested)
        self.benchmark_widget.openRunRequested.connect(self._open_campaign_run)
        self.benchmark_widget.trainingRequested.connect(self._show_training)
        self.training_widget.startTrainingRequested.connect(self._on_training_requested)
        self.training_widget.terminateRequested.connect(self._on_terminate_requested)
        self.training_widget.trainingComplete.connect(self._on_training_completed)
        self.campaign_controller.busyChanged.connect(self._on_campaign_busy)
        self.campaign_controller.messageChanged.connect(self.status_bar.showMessage)
        self.campaign_controller.changed.connect(self._on_campaign_changed)
        # Setup widget signals
        self.setup_widget.requestChanged.connect(self._on_request_changed)
        self.setup_widget.resolveRequested.connect(self._on_resolve_requested)
        self.setup_widget.runRequested.connect(self._on_run_requested)
        self.setup_widget.terminateRequested.connect(self._on_terminate_requested)

        # Process runner signals
        self._runner.processStarted.connect(self._on_process_started)
        self._runner.processFinished.connect(self._on_process_finished)
        self._runner.planResolved.connect(self._on_plan_resolved)
        self._runner.preflightReport.connect(self._on_preflight_report)
        self._runner.launcherEvent.connect(self._on_launcher_event)
        self._runner.executionReport.connect(self._on_execution_report)
        self._runner.workerError.connect(self._on_worker_error)
        self._runner.workerDone.connect(self._on_worker_done)
        self._runner.unexpectedWorkerExit.connect(self._on_unexpected_worker_exit)
        self._runner.rawOutputReceived.connect(self._on_raw_worker_output)

    def _on_request_changed(self) -> None:
        if self.campaign_controller.active or self._training_active:
            return
        self.plan_preflight_widget.clear()
        self.status_bar.showMessage("Request parameters changed. Plan & Preflight invalidated.")

    def _on_resolve_requested(self, request: LaunchRequestV1) -> None:
        if self.campaign_controller.active or self._training_active:
            return
        self._terminal_semantic_status = None
        self.setup_widget.set_operation_active(True)
        self.tab_widget.setCurrentIndex(1)  # Switch to Plan & Preflight tab
        self.status_bar.showMessage(f"Resolving plan & running preflight for mode {request.mode.value}...")
        try:
            self._runner.start_operation("PLAN", request)
        except Exception as e:
            self.setup_widget.set_operation_active(False)
            self._terminal_semantic_status = f"Failed to start PLAN: {e}"
            self.status_bar.showMessage(self._terminal_semantic_status)

    def _on_run_requested(self, request: LaunchRequestV1) -> None:
        if self.campaign_controller.active or self._training_active:
            return
        self._terminal_semantic_status = None
        self._last_completed_run_dir = None  # Reset pending run-dir on start of every RUN
        self.setup_widget.set_operation_active(True)
        self.tab_widget.setCurrentIndex(2)  # Switch to Run Monitor tab
        self.run_monitor_widget.reset_monitor()
        self.status_bar.showMessage(f"Starting execution run for mode {request.mode.value}...")
        try:
            self._runner.start_operation("RUN", request)
        except Exception as e:
            self.setup_widget.set_operation_active(False)
            self._terminal_semantic_status = f"Failed to start RUN: {e}"
            self.status_bar.showMessage(self._terminal_semantic_status)

    def _on_terminate_requested(self) -> None:
        warning_msg = (
            "FORCE TERMINATE will kill the running worker process immediately.\n\n"
            "Warning: Force termination is non-graceful and may leave the local "
            "run state marked RUNNING. It does NOT rewrite run_state or claim INTERRUPTED.\n\n"
            "Are you sure you want to force terminate?"
        )
        reply = QMessageBox.warning(
            self,
            "Force Terminate Confirmation",
            warning_msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            if self.campaign_controller.active:
                self.campaign_controller.cancel()
            self._terminal_semantic_status = "Worker process terminated forcefully (non-graceful)."
            self._runner.force_terminate()
            self.status_bar.showMessage(self._terminal_semantic_status)

    def _on_process_started(self) -> None:
        self._on_campaign_busy(True)
        if self.campaign_controller.active and self.campaign_controller.operation == "RUN":
            self.run_monitor_widget.reset_monitor()

    def _on_process_finished(self, exit_code: int) -> None:
        if self._training_active:
            self._training_active = False
            self._on_campaign_busy(False)
            self.training_widget.finish(exit_code, self._training_done, self._runner.force_terminated)
            self.status_bar.showMessage(self.training_widget.lbl_status.text())
            return
        if self.campaign_controller.active:
            self.campaign_controller.process_finished(exit_code)
            return
        self._on_campaign_busy(False)
        # Preserve semantic status emitted by WORKER_DONE, WORKER_ERROR, or FORCE TERMINATE
        if self._terminal_semantic_status:
            self.status_bar.showMessage(f"{self._terminal_semantic_status} (exit code {exit_code})")
        else:
            self.status_bar.showMessage(f"Worker process finished with exit code {exit_code}.")

    def _on_plan_resolved(self, payload: dict) -> None:
        self.plan_preflight_widget.display_plan(payload)

    def _on_preflight_report(self, payload: dict) -> None:
        self.plan_preflight_widget.display_preflight(payload)
        if self.campaign_controller.active:
            self.campaign_controller.preflight_received(payload)
            return
        can_execute = payload.get("can_execute", False)
        self.setup_widget.set_preflight_verdict(can_execute)

    def _on_launcher_event(self, payload: dict) -> None:
        self.run_monitor_widget.handle_launcher_event(payload)

    def _on_execution_report(self, payload: dict) -> None:
        self.run_monitor_widget.handle_execution_report(payload)
        if self.campaign_controller.active:
            self.campaign_controller.report_received(payload)
        rep = payload.get("execution_report", payload)
        run_dir_str = rep.get("run_dir")
        if run_dir_str:
            self._last_completed_run_dir = Path(run_dir_str)

    def _on_worker_error(self, payload: dict) -> None:
        if self._training_active:
            self.training_widget.error = payload.get("message", "Training worker failed")
        if self.campaign_controller.active:
            self.campaign_controller.worker_error(payload)
        err_msg = payload.get("message", "Unknown error")
        self._terminal_semantic_status = f"Worker Error: {err_msg}"
        self.status_bar.showMessage(self._terminal_semantic_status)

    def _on_unexpected_worker_exit(self, payload: dict) -> None:
        if self.campaign_controller.active:
            self.campaign_controller.worker_error({"message": payload.get("reason", "Unexpected worker exit")})
        op = payload.get("operation", "UNKNOWN")
        code = payload.get("exit_code", -1)
        self._terminal_semantic_status = f"Abnormal worker exit during {op}: process terminated unexpectedly without WORKER_DONE."
        self.status_bar.showMessage(self._terminal_semantic_status)

    def _on_worker_done(self, payload: dict) -> None:
        if self._training_active:
            self._training_done = payload
            return
        if self.campaign_controller.active:
            self.campaign_controller.worker_done(payload)
            if payload.get("operation") == "RUN" and payload.get("success") and payload.get("run_id"):
                self.results_widget.set_runs_root(self._core_adapter.project_root / "runs")
                self.results_widget.select_run_by_id(payload["run_id"])
            return
        op = payload.get("operation")
        success = payload.get("success", False)
        blocked = payload.get("blocked", False)
        if blocked:
            self._terminal_semantic_status = f"Operation {op} blocked by preflight safety check."
            self.status_bar.showMessage(self._terminal_semantic_status)
        elif success:
            self._terminal_semantic_status = f"Operation {op} completed successfully."
            self.status_bar.showMessage(self._terminal_semantic_status)
            # Pass B2: Auto-refresh and select completed run in Results tab
            # Applied strictly on successful RUN operations (Specification Section 20)
            if op == "RUN":
                run_id = payload.get("run_id")
                if self._last_completed_run_dir and self._last_completed_run_dir.exists():
                    try:
                        parent_root = self._last_completed_run_dir.parent
                        if self.results_widget.runs_root != parent_root:
                            self.results_widget.set_runs_root(parent_root)
                    except Exception:
                        pass

                if run_id:
                    try:
                        self.results_widget.select_run_by_id(run_id)
                    except Exception:
                        pass
        else:
            # Preserve specific error message if already captured (e.g. from workerError or unexpectedWorkerExit)
            if not self._terminal_semantic_status or "completed successfully" in self._terminal_semantic_status:
                self._terminal_semantic_status = f"Operation {op} failed."
            self.status_bar.showMessage(self._terminal_semantic_status)

    def _on_campaign_requested(self, operation, campaign) -> None:
        if self._training_active:
            return
        self._terminal_semantic_status = None
        self.plan_preflight_widget.clear()
        if operation == "RUN":
            self.run_monitor_widget.reset_monitor()
            self.tab_widget.setCurrentIndex(2)
        try:
            self.campaign_controller.start(operation, campaign)
        except (ValueError, RuntimeError) as exc:
            self.status_bar.showMessage(str(exc))

    def _on_campaign_selection_changed(self) -> None:
        if not self._runner.is_running:
            self.plan_preflight_widget.clear()

    def _on_campaign_busy(self, busy: bool) -> None:
        self.benchmark_widget.set_busy(busy)
        self.setup_widget.set_operation_active(busy)
        self.training_widget.set_busy(busy)
        self.setup_pages.setTabEnabled(0, not self._training_active)
        self.setup_pages.setTabEnabled(1, not busy or (not self.campaign_controller.active and not self._training_active))
        self.setup_pages.setTabEnabled(2, not busy or self._training_active)

    def _show_training(self):
        if not self._runner.is_running and not self.campaign_controller.active:
            self.tab_widget.setCurrentIndex(0)
            self.setup_pages.setCurrentIndex(2)

    def _on_training_requested(self, preset):
        if self._runner.is_running or self.campaign_controller.active or self._training_active:
            return
        self._training_active = True
        self._training_done = None
        self.training_widget.begin()
        self._on_campaign_busy(True)
        try:
            self._runner.start_training(preset)
        except (ValueError, RuntimeError, OSError) as exc:
            self.training_widget.error = str(exc)
            self._on_process_finished(-1)

    def _on_raw_worker_output(self, line):
        if self._training_active:
            self.training_widget.consume_line(line)

    def _on_training_completed(self, report):
        if report.get("preset") == "smoke":
            index = self.benchmark_widget.combo_profile.findData("train_demo_v1")
            self.benchmark_widget.combo_profile.setCurrentIndex(index)
        self.benchmark_widget.reload_training_bindings(select_pair=True)
        self.tab_widget.setCurrentIndex(0)
        self.setup_pages.setCurrentIndex(0)

    def _on_campaign_changed(self) -> None:
        if not self.campaign_controller.active:
            self.tab_widget.setCurrentIndex(0)
            self.setup_pages.setCurrentIndex(0)

    def _open_campaign_run(self, run_id: str) -> None:
        self.results_widget.set_runs_root(self._core_adapter.project_root / "runs")
        self.results_widget.select_run_by_id(run_id)
        self.tab_widget.setCurrentIndex(5)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Window close behavior (Specification Section 22)."""
        if self._runner.is_running:
            warning_msg = (
                "An experiment worker is currently running.\n\n"
                "Closing the Workbench will force terminate the active scientific run.\n"
                "Force termination is non-graceful and will NOT claim clean completion or interruption.\n\n"
                "Do you want to force terminate and close, or cancel and return?"
            )
            reply = QMessageBox.question(
                self,
                "Worker Running — Confirm Close",
                warning_msg,
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Discard:
                if self.campaign_controller.active:
                    self.campaign_controller.cancel()
                # Bounded force termination on close (Specification Correction 2)
                terminated = self._runner.force_terminate_and_wait(timeout_ms=2000)
                if terminated:
                    event.accept()
                else:
                    # Bounded termination failed: surface diagnostic, do not silently close while worker may still be alive
                    self.status_bar.showMessage("Error: Bounded termination timed out. Worker process could not be confirmed stopped.")
                    QMessageBox.critical(
                        self,
                        "Termination Timeout",
                        "Failed to terminate worker process within the 2000ms timeout.\n"
                        "To prevent orphaned simulation processes or corrupted state, window close was aborted.\n"
                        "Please retry FORCE TERMINATE or inspect system processes.",
                    )
                    event.ignore()
            else:
                event.ignore()
        else:
            if self.campaign_controller.active:
                self.campaign_controller.cancel()
            event.accept()
