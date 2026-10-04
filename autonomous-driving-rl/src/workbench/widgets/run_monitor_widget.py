"""
Live Run Monitor Widget (Gate 7.5B Pass B2).

Consumes exclusively LauncherEventV1 messages and ExecutionReport:
- Displays: run ID, run status, current episode, total episodes, current step,
  route completion, speed km/h, latest event message.
- Shows chronological event log and execution results summary.
- Embedded live Matplotlib charts:
  1. Route Completion vs Decision Step (sampled LauncherEventV1 progress)
  2. Speed (km/h) vs Decision Step (sampled LauncherEventV1 progress)
- Strictly decoupled from simulation internals; consumes only passive telemetry.
"""

from typing import Any, Dict, List, Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.workbench.live_telemetry import LiveTelemetryBufferV1


MAX_EVENT_LOG_BLOCKS = 5000


class RunMonitorWidget(QWidget):
    """Real-time monitoring panel driven strictly by LauncherEventV1 telemetry."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._telemetry_buffer = LiveTelemetryBufferV1()
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

        # Matplotlib Live Charts Tab / Area
        charts_group = QGroupBox("Live Sampled Traces (Sampled LauncherEventV1 progress telemetry)")
        charts_layout = QVBoxLayout(charts_group)

        self.figure = Figure(figsize=(7, 3), dpi=90)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax_route = self.figure.add_subplot(1, 2, 1)
        self.ax_speed = self.figure.add_subplot(1, 2, 2)
        self._setup_axes()
        charts_layout.addWidget(self.canvas)
        splitter.addWidget(charts_group)

        # Log & Results Splitter
        lower_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Event log with bounded document block count (Specification Section 30)
        log_group = QGroupBox("Chronological Telemetry Log (LauncherEventV1)")
        log_layout = QVBoxLayout(log_group)
        self.text_event_log = QTextEdit()
        self.text_event_log.setReadOnly(True)
        self.text_event_log.document().setMaximumBlockCount(MAX_EVENT_LOG_BLOCKS)
        log_layout.addWidget(self.text_event_log)
        lower_splitter.addWidget(log_group)

        # Execution results overview
        results_group = QGroupBox("Execution-Reported Overview (Provisional / Non-Authoritative)")
        results_layout = QVBoxLayout(results_group)
        self.text_results = QTextEdit()
        self.text_results.setReadOnly(True)
        self.text_results.setPlaceholderText("Results will appear here upon experiment completion.")
        results_layout.addWidget(self.text_results)
        lower_splitter.addWidget(results_group)

        splitter.addWidget(lower_splitter)
        main_layout.addWidget(splitter)

    def _setup_axes(self) -> None:
        self.ax_route.clear()
        self.ax_route.set_title("Route Completion vs Decision Step\n(Sampled progress)", fontsize=9)
        self.ax_route.set_xlabel("Decision Step", fontsize=8)
        self.ax_route.set_ylabel("Route Completion", fontsize=8)
        self.ax_route.set_ylim(-0.05, 1.05)
        self.ax_route.grid(True, linestyle="--", alpha=0.6)

        self.ax_speed.clear()
        self.ax_speed.set_title("Speed (km/h) vs Decision Step\n(Sampled progress)", fontsize=9)
        self.ax_speed.set_xlabel("Decision Step", fontsize=8)
        self.ax_speed.set_ylabel("Speed (km/h)", fontsize=8)
        self.ax_speed.grid(True, linestyle="--", alpha=0.6)
        self.figure.tight_layout()

    def reset_monitor(self, run_id: Optional[str] = None) -> None:
        self._telemetry_buffer.reset()
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
        self._setup_axes()
        self.canvas.draw_idle()

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

        # Update in-memory telemetry buffer
        self._telemetry_buffer.handle_event(event)

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
            self._setup_axes()
            self.canvas.draw_idle()
        elif event_type == "EPISODE_STARTED":
            self._setup_axes()
            self.canvas.draw_idle()
        elif event_type == "EPISODE_PROGRESS":
            self._update_live_charts()
        elif event_type == "RUN_FINISHED":
            self.lbl_run_status.setText("Status: FINISHED")
            self.progress_bar.setValue(100)
            self._update_live_charts()
        elif event_type == "RUN_FAILED":
            self.lbl_run_status.setText("Status: FAILED")
        elif event_type == "RUN_INTERRUPTED":
            self.lbl_run_status.setText("Status: INTERRUPTED")

        log_line = f"[{timestamp}] {event_type}: {message}"
        self.text_event_log.append(log_line)

    def _update_live_charts(self) -> None:
        """Renders current episode sampled trace on Matplotlib canvas."""
        trace = self._telemetry_buffer.current_episode_trace
        if not trace or not trace.step_indices:
            return

        self._setup_axes()
        self.ax_route.plot(trace.step_indices, trace.route_completions, color="#1f77b4", marker="o", markersize=3, label="Route")
        self.ax_speed.plot(trace.step_indices, trace.speeds_kmh, color="#2ca02c", marker="o", markersize=3, label="Speed")
        self.figure.tight_layout()
        self.canvas.draw_idle()

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
            f"=== EXECUTION-REPORTED SUMMARY (Provisional / Not Integrity-Verified) ===",
            f"Notice: Stored values below are emitted directly from ExperimentExecutor.",
            f"They are provisional until local Gate-7 artifacts are loaded and integrity is VERIFIED in the Results tab.",
            f"",
            f"Run ID:             {run_id}",
            f"Execution Status:   {status}",
            f"Run Directory:      {run_dir}",
            f"Completed Episodes: {completed_episodes} / {total_cases}",
        ]

        if summary:
            lines.append("")
            lines.append("--- Execution-Reported Metrics (Provisional) ---")
            overall = summary.get("overall_metrics", {})
            clean_success = overall.get("clean_success_rate")
            safety_fail = overall.get("safety_failure_rate")
            mean_route = overall.get("mean_final_route_completion")
            median_route = overall.get("median_final_route_completion")
            mean_time = overall.get("mean_time_to_clean_success_s")

            if clean_success is not None:
                lines.append(f"  Clean Success Rate:            {clean_success * 100:.1f}%")
            if safety_fail is not None:
                lines.append(f"  Safety Failure Rate:           {safety_fail * 100:.1f}%")
            if mean_route is not None:
                lines.append(f"  Mean Final Route Completion:   {mean_route * 100:.1f}%")
            if median_route is not None:
                lines.append(f"  Median Final Route Completion: {median_route * 100:.1f}%")
            if mean_time is not None:
                lines.append(f"  Mean Time to Clean Success:    {mean_time:.2f} s")
            else:
                lines.append(f"  Mean Time to Clean Success:    N/A — no clean-success samples")

            lines.append("")
            lines.append("--- Diagnostic / Training Signals (NOT ranking scores) ---")
            mean_ret = summary.get("mean_total_reward") or overall.get("mean_episode_return")
            if mean_ret is not None:
                lines.append(f"  Diagnostic Mean Episode Return: {mean_ret:.2f}")

        self.text_results.setText("\n".join(lines))
