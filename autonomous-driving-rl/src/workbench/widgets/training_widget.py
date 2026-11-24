"""Fixed training presets and real loss/checkpoint evidence for Stage 6."""

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QTextEdit, QVBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from src.agents.planning_graph import load_astar_config
from src.launcher.checkpoints import latest_training_bindings
from src.training.astar_heuristic import select_train_geometries
from src.training.protocol import parse_training_line


class TrainingWidget(QWidget):
    startTrainingRequested = Signal(str)
    terminateRequested = Signal()
    trainingComplete = Signal(dict)

    def __init__(self, project_root, parent=None):
        super().__init__(parent)
        self.project_root = Path(project_root)
        self.config = load_astar_config()
        self.report = None
        self.error = ""
        self._busy = False
        self._curves = {}
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>STAGE 6 · A* + HEURISTIC HỌC TỪ DỮ LIỆU</b>"))
        description = QLabel("A* Stage 2 dùng heuristic Euclidean. Stage 6 giữ nguyên đồ thị và điều khiển, học ước lượng chi phí còn lại bằng MLP.\n"
                             "TRAIN → thu thập đồ thị cục bộ → nhãn chi phí chính xác trên đồ thị → cập nhật trọng số → chọn checkpoint.")
        description.setWordWrap(True)
        layout.addWidget(description)
        self.combo_preset = QComboBox()
        self.combo_preset.addItem("Train thử · 8 TRAIN geometry · seed 101 · 40 epoch", "smoke")
        self.combo_preset.addItem("Train đầy đủ · toàn bộ TRAIN · 3 training seed · 80 epoch", "standard")
        layout.addWidget(self.combo_preset)
        self.lbl_settings = QLabel()
        self.lbl_settings.setWordWrap(True)
        layout.addWidget(self.lbl_settings)
        scope = QLabel("Model dự đoán chi phí: đo sai số MSE (càng thấp càng tốt), không dùng accuracy phân loại.\n"
                       "Chọn checkpoint bằng nhóm giữ lại tách theo geometry bên trong TRAIN. Chưa dùng VALIDATION/TEST chuẩn để kết luận chất lượng lái.\n"
                       "Heuristic học được chưa có bảo đảm admissibility; báo cáo riêng sai số, node mở rộng và chi phí đường đi trên đồ thị giữ lại.")
        scope.setWordWrap(True)
        scope.setStyleSheet("background: #eef4f8; padding: 8px;")
        layout.addWidget(scope)
        self.lbl_status = QLabel("Sẵn sàng huấn luyện trên CPU, dùng NumPy trong môi trường của dự án.")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.figure = Figure(figsize=(7, 2.2))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        self._draw()
        layout.addWidget(self.canvas, 1)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(2000)
        self.log.setMaximumHeight(100)
        layout.addWidget(self.log)
        self.lbl_checkpoint = QLabel()
        self.lbl_checkpoint.setWordWrap(True)
        layout.addWidget(self.lbl_checkpoint)
        buttons = QHBoxLayout()
        self.btn_train = QPushButton("Huấn luyện heuristic")
        self.btn_stop = QPushButton("FORCE TERMINATE")
        self.btn_stop.setEnabled(False)
        buttons.addWidget(self.btn_train)
        buttons.addWidget(self.btn_stop)
        layout.addLayout(buttons)
        self.btn_train.clicked.connect(lambda: self.startTrainingRequested.emit(self.combo_preset.currentData()))
        self.btn_stop.clicked.connect(self.terminateRequested.emit)
        self.combo_preset.currentIndexChanged.connect(self._settings_changed)
        self._settings_changed()
        self.refresh_checkpoint()

    def _settings_changed(self):
        preset = self.config["presets"][self.combo_preset.currentData()]
        count = len(select_train_geometries(self.project_root, preset["geometry_limit"]))
        self.lbl_settings.setText(
            f"Cấu hình cố định: {count} geometry × tối đa {preset['steps_per_geometry']} bước/geometry mỗi training seed. "
            f"Lấy mẫu mỗi {preset['sample_every']} bước; {preset['epochs']} epoch; seed {preset['training_seeds']}.\n"
            "Nguồn: configs/experiments/astar_learning_v1.json. Learning rate, kiến trúc và bộ điều khiển được lưu cùng phiên bản cấu hình."
        )

    def set_busy(self, busy):
        self._busy = busy
        self.combo_preset.setEnabled(not busy)
        self.btn_train.setEnabled(not busy)
        self.btn_stop.setEnabled(busy)

    def begin(self):
        self.report = None
        self.error = ""
        self._curves.clear()
        self.log.clear()
        self.progress.setValue(0)
        self.lbl_status.setText("Đang khởi động worker huấn luyện…")
        self.set_busy(True)
        self._draw()

    def consume_line(self, line):
        message = parse_training_line(line)
        if message is None:
            return
        event = message["payload"]
        if message["type"] == "REPORT":
            self.report = event
            return
        if message["type"] == "ERROR":
            self.error = event.get("message", "Training worker failed")
            self.log.append(self.error)
            return
        phase = event.get("phase")
        seed = event.get("training_seed")
        seeds = self.config["presets"][self.combo_preset.currentData()]["training_seeds"]
        seed_index = seeds.index(seed) if seed in seeds else 0
        if phase == "COLLECT":
            fraction = .5 * event["geometry_index"] / event["geometry_count"]
            text = f"TRAIN seed {seed}: thu thập geometry {event['geometry_index']}/{event['geometry_count']} · {event['geometry_id']}"
            self.log.append(text)
        elif phase == "FIT":
            fraction = .5 + .5 * event["epoch"] / event["epochs"]
            text = (f"Seed {seed} · epoch {event['epoch']}/{event['epochs']} · train MSE {event['train_mse']:.6g}"
                    f" · internal holdout MSE {event['internal_holdout_mse']:.6g}")
            self._curves.setdefault(seed, []).append(event)
            self.log.append(text)
            self._draw()
        else:
            fraction, text = 1., f"Checkpoint seed {seed} đã lưu · {event.get('checkpoint_sha256', '')}"
            self.log.append(text)
        self.lbl_status.setText(text)
        self.progress.setValue(min(99, int(100 * (seed_index + fraction) / len(seeds))))

    def _draw(self):
        self.axes.clear()
        for seed, records in self._curves.items():
            epochs = [r["epoch"] for r in records]
            self.axes.plot(epochs, [max(r["train_mse"], 1e-12) for r in records], label=f"TRAIN · {seed}")
            self.axes.plot(epochs, [max(r["internal_holdout_mse"], 1e-12) for r in records], linestyle="--", label=f"TRAIN holdout · {seed}")
        self.axes.set_xlabel("Epoch")
        self.axes.set_ylabel("Residual MSE")
        self.axes.set_yscale("log")
        self.axes.grid(True, alpha=.3)
        if self._curves:
            self.axes.legend(fontsize=8)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def finish(self, exit_code, done, force_terminated=False):
        self.set_busy(False)
        if exit_code == 0 and done and done.get("success") and self.report and self.report.get("status") == "COMPLETE":
            self.progress.setValue(100)
            self.lbl_status.setText(f"Huấn luyện COMPLETE · {len(self.report['models'])} checkpoint · {self.report['run_id']}")
            self.refresh_checkpoint()
            self.trainingComplete.emit(self.report)
        else:
            self.lbl_status.setText("Huấn luyện đã dừng cưỡng bức." if force_terminated else f"Huấn luyện chưa hoàn tất: {self.error or 'worker không kết thúc thành công'}")

    def refresh_checkpoint(self):
        try:
            bindings = latest_training_bindings(self.project_root)
            self.lbl_checkpoint.setText("Checkpoint đã xác minh: " + ", ".join(f"seed {b.training_run_seed} · {b.model_sha256[:12]}" for b in bindings)
                                        if bindings else "Chưa có checkpoint đã train.")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.lbl_checkpoint.setText(f"Checkpoint chưa hợp lệ: {exc}")
