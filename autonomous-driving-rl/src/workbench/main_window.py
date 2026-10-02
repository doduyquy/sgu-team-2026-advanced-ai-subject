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

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.launcher.models import LaunchRequestV1
from src.workbench.core_adapter import CoreAdapter
from src.workbench.process_runner import WorkbenchProcessRunner
from src.workbench.widgets.agent_explorer_widget import AgentExplorerWidget
from src.workbench.widgets.case_explorer_widget import CaseExplorerWidget
from src.workbench.widgets.experiment_setup_widget import ExperimentSetupWidget
from src.workbench.widgets.plan_preflight_widget import PlanAndPreflightWidget
from src.workbench.widgets.run_monitor_widget import RunMonitorWidget


class MainWindow(QMainWindow):
    """Main window for Research Workbench V1."""

    def __init__(self, core_adapter: Optional[CoreAdapter] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Research Workbench V1 — Presentation & Control Layer (Gate 7.5B)")
        self.resize(1100, 750)

        self._core_adapter = core_adapter or CoreAdapter()
        self._runner = WorkbenchProcessRunner(self)

        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.tab_widget = QTabWidget(self)

        # Tab 1: Experiment Setup
        self.setup_widget = ExperimentSetupWidget(self._core_adapter)
        self.tab_widget.addTab(self.setup_widget, "1. Experiment Setup")

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

        main_layout.addWidget(self.tab_widget)

        # Status Bar
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Workbench Ready. Scientific authority strictly maintained in Gate 7.5A Core.")

    def _connect_signals(self) -> None:
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

    def _on_request_changed(self) -> None:
        self.plan_preflight_widget.clear()
        self.status_bar.showMessage("Request parameters changed. Plan & Preflight invalidated.")

    def _on_resolve_requested(self, request: LaunchRequestV1) -> None:
        self.tab_widget.setCurrentIndex(1)  # Switch to Plan & Preflight tab
        self.status_bar.showMessage(f"Resolving plan & running preflight for mode {request.mode.value}...")
        self._runner.start_operation("PLAN", request)

    def _on_run_requested(self, request: LaunchRequestV1) -> None:
        self.tab_widget.setCurrentIndex(2)  # Switch to Run Monitor tab
        self.run_monitor_widget.reset_monitor()
        self.status_bar.showMessage(f"Starting execution run for mode {request.mode.value}...")
        self._runner.start_operation("RUN", request)

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
            self._runner.force_terminate()
            self.status_bar.showMessage("Worker process terminated forcefully.")

    def _on_process_started(self) -> None:
        self.setup_widget.set_worker_running(True)

    def _on_process_finished(self, exit_code: int) -> None:
        self.setup_widget.set_worker_running(False)
        self.status_bar.showMessage(f"Worker process finished with exit code {exit_code}.")

    def _on_plan_resolved(self, payload: dict) -> None:
        self.plan_preflight_widget.display_plan(payload)

    def _on_preflight_report(self, payload: dict) -> None:
        self.plan_preflight_widget.display_preflight(payload)
        can_execute = payload.get("can_execute", False)
        self.setup_widget.set_preflight_verdict(can_execute)

    def _on_launcher_event(self, payload: dict) -> None:
        self.run_monitor_widget.handle_launcher_event(payload)

    def _on_execution_report(self, payload: dict) -> None:
        self.run_monitor_widget.handle_execution_report(payload)

    def _on_worker_error(self, payload: dict) -> None:
        err_msg = payload.get("message", "Unknown error")
        self.status_bar.showMessage(f"Worker Error: {err_msg}")

    def _on_worker_done(self, payload: dict) -> None:
        op = payload.get("operation")
        success = payload.get("success", False)
        blocked = payload.get("blocked", False)
        if blocked:
            self.status_bar.showMessage(f"Operation {op} blocked by preflight safety check.")
        elif success:
            self.status_bar.showMessage(f"Operation {op} completed successfully.")
        else:
            self.status_bar.showMessage(f"Operation {op} failed.")

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
                self._runner.force_terminate()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
