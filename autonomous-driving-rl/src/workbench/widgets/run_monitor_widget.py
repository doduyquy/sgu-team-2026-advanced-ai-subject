"""
Live Run Monitor Widget (Gate 7.5B).

Consumes exclusively LauncherEventV1 messages and ExecutionReport:
- Displays: run ID, run status, current episode, total episodes, current step,
  route completion, speed km/h, latest event message.
- Shows chronological event log and execution results summary.
"""

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class RunMonitorWidget(QWidget):
    """Real-time monitoring panel driven strictly by LauncherEventV1 telemetry."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # Telemetry metrics row
        metrics_group = QGroupBox("Live Experiment Telemetry")
        metrics_layout = QVBoxLayout(metrics_group)

        row1 = QHBoxLayout()
        self.lbl_run_id = QLabel("Run ID: —")
        self.lbl_run_status = QLabel("Status: IDLE")
        self.lbl_episode = QLabel("Episode: 0 / 0")
        self.lbl_step = QLabel("Step: 0")
        row1.addWidget(self.lbl_run_id)
        row1.addWidget(self.lbl_run_status)
        row1.addWidget(self.lbl_episode)
        row1.addWidget(self.lbl_step)
        metrics_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.lbl_route = QLabel("Route Completion: 0.0%")
        self.lbl_speed = QLabel("Speed: 0.0 km/h")
        self.lbl_event_type = QLabel("Latest Event: —")
        row2.addWidget(self.lbl_route)
        row2.addWidget(self.lbl_speed)
        row2.addWidget(self.lbl_event_type)
        metrics_layout.addLayout(row2)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        metrics_layout.addWidget(self.progress_bar)

        main_layout.addWidget(metrics_group)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # Event log
        log_group = QGroupBox("Chronological Telemetry Log (LauncherEventV1)")
        log_layout = QVBoxLayout(log_group)
        self.text_event_log = QTextEdit()
        self.text_event_log.setReadOnly(True)
        log_layout.addWidget(self.text_event_log)
        splitter.addWidget(log_group)

        # Execution results overview
        results_group = QGroupBox("Execution Results Summary")
        results_layout = QVBoxLayout(results_group)
        self.text_results = QTextEdit()
        self.text_results.setReadOnly(True)
        self.text_results.setPlaceholderText("Results will appear here upon experiment completion.")
        results_layout.addWidget(self.text_results)
        splitter.addWidget(results_group)

        main_layout.addWidget(splitter)

    def reset_monitor(self, run_id: Optional[str] = None) -> None:
        self.lbl_run_id.setText(f"Run ID: {run_id or '—'}")
        self.lbl_run_status.setText("Status: STARTING")
        self.lbl_episode.setText("Episode: 0 / 0")
        self.lbl_step.setText("Step: 0")
        self.lbl_route.setText("Route Completion: 0.0%")
        self.lbl_speed.setText("Speed: 0.0 km/h")
        self.lbl_event_type.setText("Latest Event: —")
        self.progress_bar.setValue(0)
        self.text_event_log.clear()
        self.text_results.clear()

    def handle_launcher_event(self, event_data: Dict[str, Any]) -> None:
        """Processes a single LauncherEventV1 payload."""
        event = event_data.get("event", event_data)
        event_type = event.get("event_type", "UNKNOWN")
        run_id = event.get("run_id", "")
        ep_idx = event.get("episode_index")
        total_eps = event.get("total_episodes")
        step_idx = event.get("step_index")
        route = event.get("route_completion")
        speed = event.get("speed_kmh")
        message = event.get("message", "")
        timestamp = event.get("timestamp_utc") or event.get("timestamp", "")

        if run_id:
            self.lbl_run_id.setText(f"Run ID: {run_id}")

        self.lbl_event_type.setText(f"Latest Event: {event_type}")

        if ep_idx is not None and total_eps is not None:
            self.lbl_episode.setText(f"Episode: {ep_idx + 1} / {total_eps}")
            if total_eps > 0:
                pct = int(((ep_idx + (route or 0.0)) / total_eps) * 100)
                self.progress_bar.setValue(min(100, max(0, pct)))

        if step_idx is not None:
            self.lbl_step.setText(f"Step: {step_idx}")

        if route is not None:
            self.lbl_route.setText(f"Route Completion: {route * 100:.1f}%")

        if speed is not None:
            self.lbl_speed.setText(f"Speed: {speed:.1f} km/h")

        if event_type == "RUN_STARTED":
            self.lbl_run_status.setText("Status: RUNNING")
        elif event_type == "RUN_FINISHED":
            self.lbl_run_status.setText("Status: FINISHED")
            self.progress_bar.setValue(100)
        elif event_type == "RUN_FAILED":
            self.lbl_run_status.setText("Status: FAILED")

        log_line = f"[{timestamp}] {event_type}: {message}"
        self.text_event_log.append(log_line)

    def handle_execution_report(self, report_payload: Dict[str, Any]) -> None:
        """Displays execution summary once run completes."""
        report = report_payload.get("execution_report", report_payload)
        run_id = report.get("run_id", "—")
        status = report.get("status", "—")
        run_dir = report.get("run_dir", "—")
        total_cases = report.get("total_cases", 0)
        completed_episodes = report.get("completed_episodes", 0)
        summary = report.get("summary_payload") or {}

        lines = [
            f"=== EXPERIMENT EXECUTION REPORT ===",
            f"Run ID:             {run_id}",
            f"Final Status:       {status}",
            f"Run Directory:      {run_dir}",
            f"Completed Episodes: {completed_episodes} / {total_cases}",
        ]

        if summary:
            lines.append("")
            lines.append("--- Summary Metrics ---")
            mean_route = summary.get("mean_route_completion")
            success_rate = summary.get("success_rate")
            mean_reward = summary.get("mean_total_reward")
            if mean_route is not None:
                lines.append(f"  Mean Route Completion: {mean_route * 100:.1f}%")
            if success_rate is not None:
                lines.append(f"  Success Rate:          {success_rate * 100:.1f}%")
            if mean_reward is not None:
                lines.append(f"  Mean Total Reward:     {mean_reward:.2f}")

        self.text_results.setText("\n".join(lines))
