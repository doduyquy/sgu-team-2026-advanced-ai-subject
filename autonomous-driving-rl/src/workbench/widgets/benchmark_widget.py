"""Guided fixed-protocol evaluation, with per-run inspection instead of a leaderboard."""

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from src.workbench.benchmark_campaign import build_evaluation_campaign, load_evaluation_profiles
from src.workbench.campaign_controller import CampaignController
from src.workbench.core_adapter import CoreAdapter
from src.launcher.checkpoints import latest_training_bindings


class BenchmarkWidget(QWidget):
    operationRequested = Signal(str, object)
    selectionChanged = Signal()
    openRunRequested = Signal(str)
    terminateRequested = Signal()
    trainingRequested = Signal()

    def __init__(self, adapter: CoreAdapter, controller: CampaignController, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.controller = controller
        self.profiles = load_evaluation_profiles(adapter.project_root)
        self.agents = [a for a in adapter.get_registered_agents() if a.purpose != "AUDIT_FIXTURE"]
        self.preview = None
        self._busy = False
        self._result_rows = []
        self.checkpoint_bindings = ()
        layout = QVBoxLayout(self)
        title = QLabel("<b>CÙNG BỘ ĐÁNH GIÁ · KHÁC THUẬT TOÁN</b>")
        title.setStyleSheet("font-size: 17px; color: #174568; padding: 6px 0;")
        layout.addWidget(title)
        story = QLabel("1. Chọn bộ đánh giá  →  2. Chọn thuật toán  →  3. Kiểm tra và chạy cùng điều kiện")
        story.setWordWrap(True)
        layout.addWidget(story)

        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Bộ đánh giá:"))
        self.combo_profile = QComboBox()
        for profile in self.profiles:
            self.combo_profile.addItem(profile.title, profile.profile_id)
        profile_row.addWidget(self.combo_profile, 1)
        self.check_show_demo = QCheckBox("Hiển thị xe khi chạy thử TRAIN")
        self.check_show_demo.setChecked(True)
        profile_row.addWidget(self.check_show_demo)
        layout.addLayout(profile_row)
        self.lbl_protocol = QLabel()
        self.lbl_protocol.setWordWrap(True)
        self.lbl_protocol.setStyleSheet("background: #eef4f8; padding: 9px; color: #173d56;")
        layout.addWidget(self.lbl_protocol)
        training_row = QHBoxLayout()
        self.lbl_training = QLabel()
        self.lbl_training.setWordWrap(True)
        training_row.addWidget(self.lbl_training, 1)
        self.btn_training = QPushButton("Train heuristic cho A*")
        self.btn_training.clicked.connect(self.trainingRequested.emit)
        training_row.addWidget(self.btn_training)
        layout.addLayout(training_row)
        self.btn_search_group = QPushButton("Chọn nhóm tìm kiếm trên TRAIN · BFS / Dijkstra / A* / Hybrid A*")
        self.btn_search_group.clicked.connect(self._select_search_group)
        layout.addWidget(self.btn_search_group)

        agents_group = QGroupBox("Thuật toán đã triển khai · cấu hình theo phiên bản đăng ký")
        agents_layout = QVBoxLayout(agents_group)
        self.table_agents = QTableWidget(len(self.agents), 5)
        self.table_agents.setHorizontalHeaderLabels(["Chọn", "Thuật toán / phiên bản", "Giai đoạn", "Input / Action", "Điều kiện"])
        self.table_agents.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_agents.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table_agents.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        for row, agent in enumerate(self.agents):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            default_id = "stage2_astar" if any(a.agent_id == "stage2_astar" for a in self.agents) else self.agents[0].agent_id
            check.setCheckState(Qt.CheckState.Checked if agent.agent_id == default_id else Qt.CheckState.Unchecked)
            self.table_agents.setItem(row, 0, check)
            for col, value in enumerate((f"{agent.agent_id} · {agent.agent_version}", agent.stage_label or "—",
                                         f"{agent.input_profile_id}\n{agent.action_adapter_id}",
                                         "Benchmark eligible" if agent.benchmark_eligible else "TRAIN / AUDIT"), 1):
                self.table_agents.setItem(row, col, QTableWidgetItem(value))
        self.table_agents.resizeRowsToContents()
        self.table_agents.setFixedHeight(min(320, self.table_agents.horizontalHeader().height()
                                            + self.table_agents.verticalHeader().length() + 4))
        agents_layout.addWidget(self.table_agents)
        note = QLabel("Stage 2 · Graph search: BFS, Dijkstra, A*, Hybrid A*. BFS tối ưu số cạnh; Dijkstra/A* tối ưu chi phí trên cùng đồ thị. "
                      "Hybrid A* dùng mô hình chuyển động riêng. Stage 6 cần checkpoint đã train.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #58636d;")
        agents_layout.addWidget(note)
        layout.addWidget(agents_group)

        self.lbl_schedule = QLabel()
        self.lbl_schedule.setWordWrap(True)
        layout.addWidget(self.lbl_schedule)
        self.table_jobs = QTableWidget(0, 6)
        self.table_jobs.setHorizontalHeaderLabels(["Thuật toán", "Phiên bản", "Seed (agent/train)", "Case / lượt", "Trạng thái", "Thông tin / Run ID"])
        self.table_jobs.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_jobs.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table_jobs.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.table_jobs.setMaximumHeight(180)
        layout.addWidget(self.table_jobs)
        self.lbl_status = QLabel()
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.lbl_status)

        self.results_group = QGroupBox("Metric đã lưu của từng lượt · bấm đúp để mở bằng chứng trong Results")
        results_layout = QVBoxLayout(self.results_group)
        self.table_results = QTableWidget(0, 8)
        self.table_results.setHorizontalHeaderLabels([
            "Thuật toán", "Seed (agent/train)", "Clean success", "Safety fail", "Mean route", "Median route", "Time to success", "Run ID",
        ])
        self.table_results.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_results.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table_results.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        self.table_results.cellDoubleClicked.connect(self._open_result)
        results_layout.addWidget(self.table_results)
        self.results_group.setMaximumHeight(180)
        self.results_group.hide()
        layout.addWidget(self.results_group)
        layout.addStretch()

        buttons = QHBoxLayout()
        self.btn_plan = QPushButton("1. Kiểm tra toàn bộ kế hoạch")
        self.btn_run = QPushButton("2. Chạy bộ đánh giá")
        self.btn_stop = QPushButton("FORCE TERMINATE")
        self.btn_stop.setStyleSheet("color: #a12222;")
        buttons.addWidget(self.btn_plan)
        buttons.addWidget(self.btn_run)
        buttons.addWidget(self.btn_stop)
        layout.addLayout(buttons)
        self.btn_plan.clicked.connect(lambda: self._request("PLAN"))
        self.btn_run.clicked.connect(lambda: self._request("RUN"))
        self.btn_stop.clicked.connect(self.terminateRequested.emit)
        self.combo_profile.currentIndexChanged.connect(self._selection_changed)
        self.check_show_demo.toggled.connect(self._selection_changed)
        self.table_agents.itemChanged.connect(self._selection_changed)
        controller.changed.connect(self._controller_changed)
        controller.resultsReady.connect(self._show_results)
        self.reload_training_bindings()

    def _select_search_group(self):
        if self._busy:
            return
        self.combo_profile.blockSignals(True)
        self.combo_profile.setCurrentIndex(self.combo_profile.findData("train_demo_v1"))
        self.combo_profile.blockSignals(False)
        self.table_agents.blockSignals(True)
        for row, agent in enumerate(self.agents):
            self.table_agents.item(row, 0).setCheckState(
                Qt.CheckState.Checked if agent.agent_id in (
                    "stage2_bfs", "stage2_dijkstra", "stage2_astar", "stage2_hybrid_astar"
                ) else Qt.CheckState.Unchecked)
        self.table_agents.blockSignals(False)
        self._selection_changed()

    def reload_training_bindings(self, select_pair=False):
        try:
            self.checkpoint_bindings = latest_training_bindings(self.controller.project_root)
            seeds = [b.training_run_seed for b in self.checkpoint_bindings]
            self.lbl_training.setText(f"Stage 6: checkpoint đã xác minh cho training seed {seeds}." if seeds
                                      else "Stage 6 chưa có checkpoint. Mở Train để thu thập dữ liệu và học heuristic.")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.checkpoint_bindings = ()
            self.lbl_training.setText(f"Checkpoint chưa hợp lệ: {exc}")
        if select_pair:
            self.table_agents.blockSignals(True)
            for row, agent in enumerate(self.agents):
                self.table_agents.item(row, 0).setCheckState(
                    Qt.CheckState.Checked if agent.agent_id in ("stage2_astar", "stage6_astar_learned") else Qt.CheckState.Unchecked)
            self.table_agents.blockSignals(False)
        self._selection_changed()

    def _selection_changed(self, *_args):
        if self._busy:
            return
        self.controller.invalidate()
        self.results_group.hide()
        self.preview = None
        profile = self.profiles[self.combo_profile.currentIndex()]
        self.check_show_demo.setVisible(not profile.is_benchmark)
        self.lbl_protocol.setText(profile.description + "\n" + (
            "ĐÃ KHÓA: manifest, đường, giao thông, environment seed, horizon, metric; 10 Hz; render OFF; W&B DISABLED."
            if profile.is_benchmark else "DEMO TRAIN: SCS · geometry 0 · environment 0. Không dùng làm kết quả benchmark."
        ))
        selected = [a for row, a in enumerate(self.agents)
                    if self.table_agents.item(row, 0).checkState() == Qt.CheckState.Checked]
        try:
            self.preview = build_evaluation_campaign(profile, selected, show_demo=self.check_show_demo.isChecked(),
                                                     checkpoint_bindings=self.checkpoint_bindings)
            self.lbl_schedule.setText(
                f"Lịch tự động: {len(self.preview.jobs)} lượt · {sum(j.case_count for j in self.preview.jobs)} episode. "
                "Stochastic: các seed đã định; mỗi checkpoint đã train chạy một lượt deterministic."
            )
            rows = [dict(agent_id=j.request.agent_id, agent_version=j.agent_version,
                         seed=(getattr(j.request, "checkpoint_binding", None).training_run_seed
                               if getattr(j.request, "checkpoint_binding", None) is not None else j.request.agent_seed),
                         seed_kind="train" if getattr(j.request, "checkpoint_binding", None) is not None else "agent",
                         cases=j.case_count, status="CHƯA KIỂM TRA", message="") for j in self.preview.jobs]
            self._fill_jobs(rows)
            reason = "\n".join(self.preview.blocking_reasons)
            self.lbl_status.setText(reason or "Cấu hình thuật toán lấy từ phiên bản đăng ký. Kiểm tra kế hoạch trước khi chạy.")
        except ValueError as exc:
            self.table_jobs.setRowCount(0)
            self.lbl_schedule.setText("")
            self.lbl_status.setText(str(exc))
        self.lbl_status.setStyleSheet("color: #8b4c00; padding: 4px;")
        self.selectionChanged.emit()
        self._update_buttons()

    def _fill_jobs(self, rows):
        self.table_jobs.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column, key in enumerate(("agent_id", "agent_version", "seed", "cases", "status", "message")):
                value = row.get(key) if row.get(key) is not None else "—"
                if key == "seed" and row.get("seed_kind") == "train":
                    value = f"{value} (train)"
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(row.get(key, "")))
                self.table_jobs.setItem(row_index, column, item)
        self.table_jobs.setFixedHeight(min(180, self.table_jobs.horizontalHeader().height()
                                          + self.table_jobs.verticalHeader().length() + 4))

    def _request(self, operation):
        if self.preview is not None and not self._busy:
            self.operationRequested.emit(operation, self.preview)

    def set_busy(self, busy: bool):
        self._busy = busy
        self.combo_profile.setEnabled(not busy)
        self.table_agents.setEnabled(not busy)
        self.check_show_demo.setEnabled(not busy)
        self.btn_training.setEnabled(not busy)
        self.btn_search_group.setEnabled(not busy)
        self._update_buttons()

    def _update_buttons(self):
        ready = bool(self.preview and not self.preview.blocking_reasons
                     and self.controller.validated_fingerprint == self.preview.fingerprint)
        self.btn_plan.setEnabled(not self._busy and self.preview is not None)
        self.btn_run.setEnabled(not self._busy and ready)
        self.btn_stop.setEnabled(self._busy)

    def _controller_changed(self):
        self._fill_jobs(self.controller.rows)
        reasons = self.preview.blocking_reasons if self.preview else ()
        self.lbl_status.setText(f"Trạng thái lịch: {self.controller.status}" + ("\n" + "\n".join(reasons) if reasons else ""))
        self._update_buttons()

    def _show_results(self, rows):
        self._result_rows = rows
        demo = bool(rows and not rows[0]["canonical"])
        self.results_group.setTitle("DEMO TRAIN · dữ liệu VERIFIED, không phải benchmark" if demo
                                    else "Cùng protocol · metric VERIFIED theo từng replicate; không xếp hạng gộp")
        self.table_results.setRowCount(len(rows))
        keys = ("clean_success_rate", "safety_failure_rate", "mean_final_route_completion", "median_final_route_completion")
        for index, row in enumerate(rows):
            seed = (f"{row['training_run_seed']} (train)" if row.get("training_run_seed") is not None
                    else row["agent_seed"] if row["agent_seed"] is not None else "—")
            values = [row["agent_id"], seed]
            values += [f"{row[k] * 100:.2f}%" if row[k] is not None else "N/A" for k in keys]
            value = row["mean_time_to_clean_success_s"]
            values += [f"{value:.2f} s" if value is not None else "N/A", row["run_id"]]
            for column, value in enumerate(values):
                self.table_results.setItem(index, column, QTableWidgetItem(str(value)))
        self.results_group.show()
        self.table_results.setFixedHeight(min(150, self.table_results.horizontalHeader().height()
                                             + self.table_results.verticalHeader().length() + 4))

    def _open_result(self, row, _column):
        if 0 <= row < len(self._result_rows):
            self.openRunRequested.emit(self._result_rows[row]["run_id"])
