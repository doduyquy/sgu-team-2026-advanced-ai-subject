"""Practical live monitor and launcher for the Stage 5 SAC research pilot."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QProgressBar, QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QTextEdit,
    QVBoxLayout, QWidget,
)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def discover_pilot_runs(pilot_root: Path) -> List[Path]:
    """Discover only Stage 5 pilot directories with their own run status file."""
    if not pilot_root.is_dir():
        return []
    candidates = [path for path in pilot_root.iterdir() if path.is_dir() and (path / "run_status.json").is_file()]
    def modified(path: Path) -> float:
        try:
            return (path / "run_status.json").stat().st_mtime
        except OSError:
            return 0.0
    return sorted(candidates, key=modified, reverse=True)


def read_training_trace(run_dir: Path, limit: int = 1200) -> List[Dict[str, Any]]:
    path = run_dir / "checkpoints" / "training_steps.jsonl"
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    stride = max(1, len(lines) // limit)
    rows: List[Dict[str, Any]] = []
    for index in range(0, len(lines), stride):
        try:
            row = json.loads(lines[index])
            if isinstance(row, dict):
                rows.append(row)
        except json.JSONDecodeError:
            continue  # The last line can be in flight while the training process writes it.
    if lines and (len(lines) - 1) % stride and len(rows) < limit:
        try:
            rows.append(json.loads(lines[-1]))
        except json.JSONDecodeError:
            pass
    return rows[-limit:]


class Stage5PilotWidget(QWidget):
    """A read-only live dashboard around the existing Stage5 pilot run artifacts."""

    busyChanged = Signal(bool)
    runFinished = Signal(str, int)

    def __init__(self, project_root: Path, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.project_root = Path(project_root).resolve()
        self.pilot_root = self.project_root / "runs" / "stage5_pilot"
        self._selected_run: Optional[Path] = None
        self._external_busy = False
        self._process: Optional[QProcess] = None
        self._viewer_process: Optional[QProcess] = None
        self._stop_requested = False
        self._build_ui()

        self._timer = QTimer(self)
        self._timer.setInterval(1500)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    @property
    def is_running(self) -> bool:
        return bool(self._process and self._process.state() != QProcess.ProcessState.NotRunning)

    @property
    def is_viewer_running(self) -> bool:
        return bool(self._viewer_process and self._viewer_process.state() != QProcess.ProcessState.NotRunning)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Theo dõi SAC theo từng bước và xem kết quả các checkpoint. Run mới được ghi riêng; "
            "dữ liệu này là chẩn đoán trên TRAIN nội bộ, không phải VALIDATION/TEST."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("background:#eef4f8; padding:8px; font-size:12px;")
        layout.addWidget(intro)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Run SAC:"))
        self.combo_runs = QComboBox()
        self.combo_runs.setMinimumContentsLength(32)
        self.combo_runs.currentIndexChanged.connect(self._select_run_from_combo)
        controls.addWidget(self.combo_runs, 1)
        self.btn_refresh = QPushButton("Làm mới")
        self.btn_refresh.clicked.connect(self.refresh)
        controls.addWidget(self.btn_refresh)
        self.btn_open_folder = QPushButton("Mở thư mục run")
        self.btn_open_folder.clicked.connect(self.open_run_folder)
        controls.addWidget(self.btn_open_folder)
        self.btn_start = QPushButton("Chạy pilot mới · 10k bước")
        self.btn_start.clicked.connect(self.start_pilot)
        controls.addWidget(self.btn_start)
        self.btn_stop = QPushButton("Yêu cầu dừng an toàn")
        self.btn_stop.setToolTip("Dừng sau tối đa 100 bước kế tiếp, lưu trạng thái và checkpoint từng phần.")
        self.btn_stop.clicked.connect(self.request_stop)
        self.btn_stop.setEnabled(False)
        controls.addWidget(self.btn_stop)
        layout.addLayout(controls)

        status = QHBoxLayout()
        self.lbl_run = QLabel("Chưa chọn run")
        self.lbl_run.setStyleSheet("font-weight:bold; font-size:14px;")
        status.addWidget(self.lbl_run, 1)
        self.lbl_phase = QLabel("—")
        self.lbl_phase.setStyleSheet("font-weight:bold; color:#1f5d87;")
        status.addWidget(self.lbl_phase)
        layout.addLayout(status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        cards = QGridLayout()
        self.cards: Dict[str, QLabel] = {}
        card_specs = [
            ("steps", "Bước train"), ("episodes", "Episode kết thúc"),
            ("geometries", "Geometry đã gặp"), ("throughput", "Tốc độ train"),
            ("actions", "Ga / phanh / gần 0"), ("route", "Route progress gần đây"),
            ("loss", "Loss · chỉ để chẩn đoán"), ("memory", "RAM hiện tại"),
        ]
        for index, (key, title) in enumerate(card_specs):
            box = QGroupBox(title)
            box_layout = QVBoxLayout(box)
            value = QLabel("—")
            value.setWordWrap(True)
            value.setStyleSheet("font-size:15px; font-weight:bold;")
            box_layout.addWidget(value)
            cards.addWidget(box, index // 4, index % 4)
            self.cards[key] = value
        layout.addLayout(cards)

        self.lbl_explanation = QLabel("Chọn run để xem tiến độ và hành vi.")
        self.lbl_explanation.setWordWrap(True)
        self.lbl_explanation.setStyleSheet("background:#fff8e5; padding:8px;")
        layout.addWidget(self.lbl_explanation)

        chart_group = QGroupBox("Hành vi trong lúc train · lấy trực tiếp từ log từng bước")
        chart_layout = QVBoxLayout(chart_group)
        self.figure = Figure(figsize=(9, 3.5), dpi=90)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax_action = self.figure.add_subplot(1, 2, 1)
        self.ax_motion = self.figure.add_subplot(1, 2, 2)
        self.ax_route = self.ax_motion.twinx()
        chart_layout.addWidget(self.canvas)
        layout.addWidget(chart_group, 2)

        lower = QSplitter()
        checkpoint_group = QGroupBox("Đánh giá checkpoint · diagnostic nội bộ")
        checkpoint_layout = QVBoxLayout(checkpoint_group)
        tier_row = QHBoxLayout()
        tier_row.addWidget(QLabel("Nhóm đường:"))
        self.combo_tier = QComboBox()
        self.combo_tier.addItem("Tổng thể", "overall")
        for tier in ("Easy", "Medium", "Hard", "Extreme"):
            self.combo_tier.addItem(tier, tier)
        self.combo_tier.currentIndexChanged.connect(self._load_checkpoint_table)
        tier_row.addWidget(self.combo_tier)
        tier_row.addStretch(1)
        self.checkpoint_combo = QComboBox()
        self.checkpoint_combo.addItems(["0", "5000", "10000"])
        tier_row.addWidget(QLabel("Xem xe tại bước:"))
        tier_row.addWidget(self.checkpoint_combo)
        self.btn_watch = QPushButton("Mở simulator")
        self.btn_watch.clicked.connect(self.watch_checkpoint)
        tier_row.addWidget(self.btn_watch)
        checkpoint_layout.addLayout(tier_row)
        self.checkpoint_table = QTableWidget(0, 7)
        self.checkpoint_table.setHorizontalHeaderLabels(
            ["Bước", "Route TB", "Thành công sạch", "Lỗi an toàn", "Timeout", "Đứng yên", "Return TB"]
        )
        self.checkpoint_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.checkpoint_table.setAlternatingRowColors(True)
        checkpoint_layout.addWidget(self.checkpoint_table)
        self.lbl_checkpoint_note = QLabel("Chưa có báo cáo diagnostic cho run này.")
        self.lbl_checkpoint_note.setWordWrap(True)
        checkpoint_layout.addWidget(self.lbl_checkpoint_note)
        lower.addWidget(checkpoint_group)

        log_group = QGroupBox("Hoạt động gần nhất")
        log_layout = QVBoxLayout(log_group)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(500)
        self.log.setPlaceholderText("Log của tiến trình mới sẽ xuất hiện tại đây.")
        log_layout.addWidget(self.log)
        lower.addWidget(log_group)
        lower.setStretchFactor(0, 3)
        lower.setStretchFactor(1, 2)
        layout.addWidget(lower, 1)

    def set_external_busy(self, busy: bool) -> None:
        self._external_busy = bool(busy)
        self._update_buttons()

    def _update_buttons(self) -> None:
        active = self.is_running
        self.btn_start.setEnabled(not active and not self._external_busy and not self.is_viewer_running)
        progress = _read_json(self._selected_run / "progress.json") if self._selected_run else {}
        self.btn_stop.setEnabled(active and progress.get("phase") == "TRAINING" and not self._stop_requested)
        self.btn_watch.setEnabled(bool(self._selected_run and not active and not self.is_viewer_running
                                       and not self._external_busy and self._selected_checkpoint_exists()))

    def refresh(self) -> None:
        runs = discover_pilot_runs(self.pilot_root)
        selected = self._selected_run
        self.combo_runs.blockSignals(True)
        self.combo_runs.clear()
        for path in runs:
            state = _read_json(path / "run_status.json")
            progress = _read_json(path / "progress.json")
            status = progress.get("phase", progress.get("status", state.get("status", "UNKNOWN")))
            step = progress.get("training_step")
            suffix = f" · {int(step):,}/10,000" if _is_number(step) else ""
            self.combo_runs.addItem(f"{path.name}  ·  {status}{suffix}", str(path))
        index = self.combo_runs.findData(str(selected)) if selected else -1
        if index < 0 and runs:
            index = 0
        self.combo_runs.setCurrentIndex(index)
        self.combo_runs.blockSignals(False)
        self._selected_run = Path(self.combo_runs.currentData()) if self.combo_runs.currentData() else None
        self._render_selected_run()

    def _select_run_from_combo(self, _index: int) -> None:
        value = self.combo_runs.currentData()
        self._selected_run = Path(value) if value else None
        self._render_selected_run()
        self._update_buttons()

    def _render_selected_run(self) -> None:
        run_dir = self._selected_run
        if not run_dir:
            self.lbl_run.setText("Chưa có run SAC. Bấm ‘Chạy pilot mới’ để bắt đầu.")
            self.lbl_phase.setText("—")
            self.progress.setValue(0)
            self.progress.setFormat("Chưa có pilot")
            for label in self.cards.values():
                label.setText("—")
            self.lbl_explanation.setText("Run cũ không bị sửa. Màn hình này sẽ tự làm mới khi có run mới.")
            self._draw_trace([])
            self._load_checkpoint_table()
            self._update_buttons()
            return

        run_status = _read_json(run_dir / "run_status.json")
        progress = _read_json(run_dir / "progress.json")
        phase = progress.get("phase", run_status.get("status", "CHƯA CÓ TIẾN ĐỘ"))
        status = progress.get("status", run_status.get("status", "UNKNOWN"))
        training_summary = run_status.get("training_summary", {})
        self.lbl_run.setText(f"{run_dir.name} · {status}")
        self.lbl_phase.setText(str(phase))
        steps = int(progress.get("training_step", run_status.get("training_summary", {}).get("actual_num_timesteps", 0)) or 0)
        budget = int(progress.get("requested_timesteps", 10000) or 10000)
        self.progress.setValue(max(0, min(100, round(100 * steps / max(1, budget)))))
        if phase == "TRAINING":
            self.progress.setFormat(f"Training {steps:,}/{budget:,} bước · {self.progress.value()}%")
        elif phase in {"DIAGNOSTICS", "TECHNICAL_PROBE"}:
            checkpoint = progress.get("diagnostic_checkpoint_step", "—")
            completed_cases = int(progress.get("diagnostic_cases_completed", 0) or 0)
            planned_cases = int(progress.get("diagnostic_cases_planned", 0) or 0)
            current_case = progress.get("diagnostic_current_case")
            case_detail = f" · {current_case}" if current_case else ""
            self.progress.setFormat(
                f"Train {budget:,}/{budget:,} · checkpoint {checkpoint} · diagnostic {completed_cases}/{planned_cases}{case_detail}"
            )
        elif status == "COMPLETE":
            self.progress.setFormat("Pilot hoàn tất · training và diagnostic checkpoint xong")
        elif status == "INTERRUPTED":
            self.progress.setFormat(f"Đã dừng an toàn tại bước {steps:,} · chưa chạy hết diagnostic")
        elif status == "FAILED_OR_INCOMPLETE":
            self.progress.setFormat(f"Run lỗi/chưa hoàn tất tại bước {steps:,} · xem trạng thái và log bên dưới")
        elif steps >= budget:
            self.progress.setFormat(f"Training {budget:,}/{budget:,} · đang chờ/kết thúc diagnostic")
        else:
            self.progress.setFormat(f"Đang chuẩn bị training · {steps:,}/{budget:,} bước")
        self.cards["steps"].setText(f"{steps:,} / {budget:,}")
        episodes = progress.get("completed_episodes", training_summary.get("completed_episodes"))
        successes = progress.get("clean_success_count", training_summary.get("clean_success_count"))
        self.cards["episodes"].setText(
            f"{episodes} · {successes} thành công sạch" if episodes is not None and successes is not None
            else str(episodes if episodes is not None else "—")
        )
        geometries = progress.get("unique_training_geometries", training_summary.get("training_geometry_ids", []))
        self.cards["geometries"].setText(str(len(geometries)) if isinstance(geometries, list) else str(geometries))
        rate = progress.get("steps_per_second", run_status.get("training_summary", {}).get("training_steps_per_second"))
        self.cards["throughput"].setText(f"{float(rate):.1f} bước/s" if _is_number(rate) else "—")
        action = progress.get("action_summary", run_status.get("training_summary", {}).get("training_action_summary", {}))
        if action:
            self.cards["actions"].setText(
                f"Ga {100 * (action.get('throttle_fraction') or 0):.0f}% · "
                f"Phanh {100 * (action.get('brake_fraction') or 0):.0f}% · "
                f"Gần 0 {100 * (action.get('near_zero_fraction') or 0):.0f}%"
            )
        else:
            self.cards["actions"].setText("Run cũ không ghi action" if status != "RUNNING" else "Đang chờ 100 bước đầu")
        rows = read_training_trace(run_dir)
        if rows:
            route_vals = [row.get("route_completion") for row in rows if _is_number(row.get("route_completion"))]
            self.cards["route"].setText(f"{100 * route_vals[-1]:.1f}% ở dòng mới nhất" if route_vals else "—")
        else:
            self.cards["route"].setText("Run này chưa có log từng bước")
        actor = progress.get("actor_loss", training_summary.get("actor_loss"))
        critic = progress.get("critic_loss", training_summary.get("critic_loss"))
        self.cards["loss"].setText(f"Actor {actor:.4g} · Critic {critic:.4g}" if _is_number(actor) and _is_number(critic) else "Chưa có mẫu mới")
        ram = progress.get("rss_mb", training_summary.get("final_ram_mb"))
        self.cards["memory"].setText(f"{ram:.0f} MB" if _is_number(ram) else "—")
        explanation = self._behavior_explanation(action, rows)
        heartbeat = progress.get("updated_at_epoch_s")
        if status == "RUNNING" and _is_number(heartbeat) and (datetime.now(timezone.utc).timestamp() - heartbeat) > 45:
            explanation = "Không có heartbeat hơn 45 giây. Run có thể đang kẹt hoặc tiến trình đã dừng bất thường; kiểm tra log và run_status.json. " + explanation
        self.lbl_explanation.setText(explanation)
        self._draw_trace(rows)
        self._load_checkpoint_table()
        self._update_buttons()

    @staticmethod
    def _behavior_explanation(action: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
        if not action:
            if not rows:
                return "Run lịch sử này chưa có log action từng bước; telemetry sẽ có ở run mới. Loss không chứng minh xe học lái tốt."
            tb = [row.get("throttle_brake") for row in rows if _is_number(row.get("throttle_brake"))]
            if tb:
                near_zero = sum(abs(value) <= 0.05 for value in tb) / len(tb)
                braking = sum(value < -0.05 for value in tb) / len(tb)
                throttle = sum(value > 0.05 for value in tb) / len(tb)
                action = {"near_zero_fraction": near_zero, "brake_fraction": braking, "throttle_fraction": throttle}
        near_zero = action.get("near_zero_fraction")
        brake = action.get("brake_fraction")
        throttle = action.get("throttle_fraction")
        if not all(_is_number(value) for value in (near_zero, brake, throttle)):
            return "Đang thu thập hành động. Đây là tín hiệu chẩn đoán, chưa phải kết luận chất lượng lái."
        if near_zero >= 0.8:
            message = "Cảnh báo hành vi: phần lớn lệnh ga/phanh đang gần 0; cần kiểm tra xe có đứng yên không."
        elif brake > throttle:
            message = "Tín hiệu quan sát: agent đang phanh thường xuyên hơn đạp ga."
        else:
            message = "Đang thấy lệnh ga trong hành động; cần đối chiếu với tốc độ và route progress để biết xe có tiến không."
        return message + " Các ngưỡng này chỉ giúp quan sát, không tự quyết định dừng hay tăng ngân sách."

    def _draw_trace(self, rows: List[Dict[str, Any]]) -> None:
        self.ax_action.clear()
        self.ax_motion.clear()
        self.ax_route.clear()
        if rows:
            steps = [row.get("training_step") for row in rows]
            tb = [row.get("throttle_brake") for row in rows]
            steering = [row.get("steering") for row in rows]
            speed = [row.get("speed_km_h") for row in rows]
            route = [row.get("route_completion") for row in rows]
            self.ax_action.plot(steps, tb, color="#16865c", alpha=.8, label="Ga (+) / phanh (−)")
            self.ax_action.plot(steps, steering, color="#4455aa", alpha=.6, label="Đánh lái")
            self.ax_action.axhline(0, color="black", linewidth=.6)
            self.ax_action.set_ylim(-1.05, 1.05)
            self.ax_action.legend(fontsize=8, loc="upper right")
            self.ax_motion.plot(steps, speed, color="#e67e22", label="Tốc độ km/h")
            self.ax_route.plot(steps, [100 * value if _is_number(value) else None for value in route],
                            color="#2878b5", alpha=.65, label="Route %")
            self.ax_route.set_ylabel("Route progress (%)", fontsize=8)
            self.ax_route.set_ylim(-5, 105)
            self.ax_motion.legend(fontsize=8, loc="upper left")
            self.ax_route.legend(fontsize=8, loc="upper right")
        else:
            self.ax_action.text(.5, .5, "Chưa có training_steps.jsonl", ha="center", va="center", transform=self.ax_action.transAxes)
            self.ax_motion.text(.5, .5, "Bản run cũ không có step telemetry", ha="center", va="center", transform=self.ax_motion.transAxes)
        self.ax_action.set_title("Hành động của policy")
        self.ax_action.set_xlabel("Bước train")
        self.ax_action.set_ylabel("Lệnh [-1, 1]")
        self.ax_action.grid(True, alpha=.25)
        self.ax_motion.set_title("Xe có phản hồi không?")
        self.ax_motion.set_xlabel("Bước train")
        self.ax_motion.set_ylabel("Tốc độ (km/h)")
        self.ax_motion.grid(True, alpha=.25)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _load_checkpoint_table(self, *_args: Any) -> None:
        self.checkpoint_table.setRowCount(0)
        if not self._selected_run:
            self.lbl_checkpoint_note.setText("Chọn run để xem kết quả checkpoint.")
            return
        report_path = self.project_root / "results" / "stage5" / self._selected_run.name / "scientific_summary.json"
        report = _read_json(report_path)
        tier = self.combo_tier.currentData() or "overall"
        checkpoints = report.get("checkpoints", {}) if tier == "overall" else report.get("per_tier", {}).get(tier, {})
        if not checkpoints:
            self.lbl_checkpoint_note.setText(
                "Chưa có kết quả đánh giá cho nhóm này. Report chỉ xuất hiện sau khi diagnostic hoàn tất."
            )
            return
        self.lbl_checkpoint_note.setText(
            "Các giá trị là metrics đo trên 24 geometry TRAIN nội bộ; không phải benchmark VALIDATION/TEST."
        )
        for step in sorted(checkpoints, key=lambda value: int(value)):
            metrics = checkpoints[step]
            values = [
                step, _percent(metrics.get("final_route_completion")), _percent(metrics.get("clean_success")),
                _percent(metrics.get("safety_failure")), _percent(metrics.get("timeout")),
                _percent(metrics.get("stationary_fraction")), _format_number(metrics.get("episode_return")),
            ]
            row = self.checkpoint_table.rowCount()
            self.checkpoint_table.insertRow(row)
            for column, value in enumerate(values):
                self.checkpoint_table.setItem(row, column, QTableWidgetItem(value))

    def _selected_checkpoint_exists(self) -> bool:
        if not self._selected_run:
            return False
        step = int(self.checkpoint_combo.currentText() or 0)
        return (self._selected_run / "checkpoints" / f"step_{step:05d}" / "model.zip").is_file()

    def start_pilot(self) -> None:
        if self.is_running or self._external_busy:
            return
        answer = QMessageBox.question(
            self, "Bắt đầu pilot SAC 10k",
            "Chạy một pilot mới từ đầu: 10.000 bước train, checkpoint 0/5k/10k và đánh giá 24 geometry TRAIN nội bộ. "
            "Run trước mất khoảng 1 giờ tổng cộng; thời gian thực tế tùy máy. Run mới không ghi đè kết quả cũ. Bắt đầu?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        run_id = datetime.now(timezone.utc).strftime("s5_3a_%Y%m%dT%H%M%SZ")
        script = self.project_root / "scripts" / "run_stage5_sac_pilot.py"
        if not script.is_file():
            QMessageBox.critical(self, "Không tìm thấy runner", str(script))
            return
        process = QProcess(self)
        process.setWorkingDirectory(str(self.project_root))
        process.setProgram(sys.executable)
        process.setArguments([str(script), "--run-id", run_id])
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(lambda p=process: self._consume_output(p, "stdout"))
        process.readyReadStandardError.connect(lambda p=process: self._consume_output(p, "stderr"))
        process.started.connect(lambda rid=run_id: self._on_process_started(rid))
        process.finished.connect(lambda code, _status, p=process: self._on_process_finished(p, code))
        process.errorOccurred.connect(lambda error, p=process: self._on_process_error(p, error))
        self._process = process
        self._stop_requested = False
        self.log.clear()
        process.start()
        self._update_buttons()

    def _on_process_started(self, run_id: str) -> None:
        self.pilot_root.mkdir(parents=True, exist_ok=True)
        expected = self.pilot_root / run_id
        self._selected_run = expected
        self.busyChanged.emit(True)
        self.refresh()
        self.log.append(f"Đã khởi chạy pilot {run_id}. Tiến độ và hành vi xe sẽ tự cập nhật.")

    def _consume_output(self, process: QProcess, channel: str) -> None:
        raw = process.readAllStandardOutput() if channel == "stdout" else process.readAllStandardError()
        for line in bytes(raw).decode("utf-8", errors="replace").splitlines():
            if line.strip():
                self.log.append(line)

    def _on_process_finished(self, process: QProcess, exit_code: int) -> None:
        if process is self._process:
            self.busyChanged.emit(False)
            run_id = self._selected_run.name if self._selected_run else ""
            self.log.append(f"Tiến trình kết thúc với exit code {exit_code}.")
            self._process = None
            self._stop_requested = False
            self.runFinished.emit(run_id, exit_code)
            process.deleteLater()
            self.refresh()
        elif process is self._viewer_process:
            self.busyChanged.emit(False)
            self._viewer_process = None
            process.deleteLater()
            self.refresh()

    def _on_process_error(self, process: QProcess, error: QProcess.ProcessError) -> None:
        if process is self._process and error == QProcess.ProcessError.FailedToStart:
            self.log.append(f"Không mở được tiến trình SAC: {process.errorString()}")
            self.busyChanged.emit(False)
            self._process = None
            self._update_buttons()
            process.deleteLater()
        elif process is self._viewer_process and error == QProcess.ProcessError.FailedToStart:
            self.log.append(f"Không mở được simulator viewer: {process.errorString()}")
            self._viewer_process = None
            self.busyChanged.emit(False)
            self._update_buttons()
            process.deleteLater()

    def request_stop(self) -> None:
        if not self.is_running or not self._selected_run:
            return
        progress = _read_json(self._selected_run / "progress.json")
        if progress.get("phase") != "TRAINING":
            QMessageBox.information(self, "Không thể dừng ở giai đoạn này",
                                    "Dừng an toàn hiện chỉ được hỗ trợ trong giai đoạn cập nhật SAC. "
                                    "Đánh giá checkpoint đang chạy đến hết để không làm mất case đang đo.")
            return
        answer = QMessageBox.question(
            self, "Yêu cầu dừng an toàn",
            "Training sẽ dừng ở điểm kiểm tra gần nhất (tối đa khoảng 100 bước), lưu log và checkpoint từng phần, "
            "đánh dấu run INTERRUPTED và bỏ qua diagnostic chưa chạy. Tiếp tục?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        (self._selected_run / "STOP_REQUESTED").write_text("user requested graceful stop\n", encoding="utf-8")
        self._stop_requested = True
        self.log.append("Đã gửi yêu cầu dừng an toàn; chờ callback đến điểm kiểm tra kế tiếp.")
        self._update_buttons()

    def open_run_folder(self) -> None:
        if self._selected_run and self._selected_run.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._selected_run)))

    def watch_checkpoint(self) -> None:
        if not self._selected_run or self.is_running or self.is_viewer_running or not self._selected_checkpoint_exists():
            return
        script = self.project_root / "scripts" / "watch_stage5_sac_checkpoint.py"
        if not script.is_file():
            QMessageBox.critical(self, "Không tìm thấy checkpoint viewer", str(script))
            return
        step = int(self.checkpoint_combo.currentText())
        process = QProcess(self)
        process.setWorkingDirectory(str(self.project_root))
        process.setProgram(sys.executable)
        process.setArguments([
            str(script), "--checkpoint", str(self._selected_run / "checkpoints" / f"step_{step:05d}"),
            "--split", str(self.project_root / "results" / "stage5" / "pilot_split_v1.json"),
        ])
        process.readyReadStandardOutput.connect(lambda p=process: self._consume_output(p, "stdout"))
        process.readyReadStandardError.connect(lambda p=process: self._consume_output(p, "stderr"))
        process.finished.connect(lambda code, _status, p=process: self._on_process_finished(p, code))
        self._viewer_process = process
        process.start()
        self.busyChanged.emit(True)
        self._update_buttons()

    def force_terminate_and_wait(self, timeout_ms: int = 2000) -> bool:
        """Emergency close only; intentionally leaves the run status untouched for forensic honesty."""
        process = self._process or self._viewer_process
        if not process or process.state() == QProcess.ProcessState.NotRunning:
            return True
        process.kill()
        if not process.waitForFinished(timeout_ms):
            return False
        return process.state() == QProcess.ProcessState.NotRunning


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _format_number(value: Any) -> str:
    return f"{value:.4f}" if _is_number(value) else "—"


def _percent(value: Any) -> str:
    return f"{100 * value:.1f}%" if _is_number(value) else "—"
