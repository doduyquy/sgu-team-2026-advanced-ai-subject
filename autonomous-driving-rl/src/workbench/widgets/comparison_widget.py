"""Provenance-aware Results Comparison View V1 (presentation-only extension)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from src.workbench.comparison_repository import (
    ComparisonRun, ComparisonResult, compare_runs, discover_comparison_runs,
)


def _fmt(value: Any, percent: bool = False) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        return f"{value * 100:.2f}%" if percent else f"{value:.6g}"
    return str(value)


class ComparisonWidget(QWidget):
    """Read-only comparison view; canonical and diagnostic tracks stay separate."""

    def __init__(self, project_root: Path, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.project_root = Path(project_root).resolve()
        self._runs: List[ComparisonRun] = []
        self._by_key: Dict[str, ComparisonRun] = {}
        self._last_result: Optional[ComparisonResult] = None
        self._source_error: Optional[str] = None
        self._init_ui()
        self.refresh_data()

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.addWidget(QLabel(
            "Canonical evaluations and development diagnostics are separate tracks. "
            "Only verified, protocol-compatible records receive paired differences."
        ))
        selectors = QGridLayout()
        self.combo_left = QComboBox()
        self.combo_right = QComboBox()
        self.btn_refresh = QPushButton("Refresh sources")
        self.btn_refresh.clicked.connect(self.refresh_data)
        selectors.addWidget(QLabel("Run / checkpoint A"), 0, 0)
        selectors.addWidget(self.combo_left, 0, 1)
        selectors.addWidget(QLabel("Run / checkpoint B"), 1, 0)
        selectors.addWidget(self.combo_right, 1, 1)
        selectors.addWidget(self.btn_refresh, 0, 2, 2, 1)
        root.addLayout(selectors)

        self.lbl_status = QLabel("No comparison sources loaded.")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("font-weight: bold; padding: 5px;")
        root.addWidget(self.lbl_status)

        self.lbl_provenance = QLabel("Select two sources to inspect provenance.")
        self.lbl_provenance.setWordWrap(True)
        root.addWidget(self.lbl_provenance)

        buttons = QHBoxLayout()
        self.btn_open_left = QPushButton("Open A result folder")
        self.btn_open_right = QPushButton("Open B result folder")
        self.btn_export = QPushButton("Export comparison JSON…")
        self.btn_open_left.clicked.connect(lambda: self._open_folder("left"))
        self.btn_open_right.clicked.connect(lambda: self._open_folder("right"))
        self.btn_export.clicked.connect(self.export_summary)
        buttons.addWidget(self.btn_open_left)
        buttons.addWidget(self.btn_open_right)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_export)
        root.addLayout(buttons)

        metrics_group = QGroupBox("Summary metrics (source values; delta only when directly comparable)")
        metrics_layout = QVBoxLayout(metrics_group)
        self.table_metrics = QTableWidget(0, 4)
        self.table_metrics.setHorizontalHeaderLabels(["Metric", "A", "B", "B − A"])
        self.table_metrics.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_metrics.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        metrics_layout.addWidget(self.table_metrics)
        root.addWidget(metrics_group)

        tier_group = QGroupBox("Per-tier summary")
        tier_layout = QVBoxLayout(tier_group)
        self.table_tiers = QTableWidget(0, 6)
        self.table_tiers.setHorizontalHeaderLabels(["Tier", "Metric", "A", "B", "B − A", "Cases"])
        self.table_tiers.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        tier_layout.addWidget(self.table_tiers)
        root.addWidget(tier_group)

        case_group = QGroupBox("Per-case records and paired differences")
        case_layout = QVBoxLayout(case_group)
        self.table_cases = QTableWidget(0, 12)
        self.table_cases.setHorizontalHeaderLabels([
            "Case ID", "Tier", "Env seed", "A outcome", "B outcome", "A route", "B route", "B − A",
            "A safety", "B safety", "A censor/timeout", "B censor/timeout",
        ])
        self.table_cases.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_cases.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        case_layout.addWidget(self.table_cases)
        self.figure = Figure(figsize=(8, 2.4), tight_layout=True)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        case_layout.addWidget(self.canvas)
        root.addWidget(case_group, 1)

        self.combo_left.currentIndexChanged.connect(self.render_comparison)
        self.combo_right.currentIndexChanged.connect(self.render_comparison)

    def refresh_data(self) -> None:
        left_key = self.combo_left.currentData()
        right_key = self.combo_right.currentData()
        self._source_error = None
        try:
            runs = discover_comparison_runs(self.project_root)
            self._runs = runs
            self._by_key = {run.key: run for run in runs}
            self.lbl_status.setText(f"Loaded {len(runs)} source(s) from verified local artifacts.")
        except Exception as exc:
            self._runs = []
            self._by_key = {}
            self._source_error = f"SOURCE VALIDATION FAILED — {type(exc).__name__}: {exc}"
        for combo, old_key in ((self.combo_left, left_key), (self.combo_right, right_key)):
            combo.blockSignals(True)
            combo.clear()
            for run in self._runs:
                combo.addItem(run.label, run.key)
            index = combo.findData(old_key) if old_key else -1
            combo.setCurrentIndex(index if index >= 0 else (0 if combo.count() else -1))
            combo.blockSignals(False)
        self.render_comparison()

    def _selected(self, combo: QComboBox) -> Optional[ComparisonRun]:
        return self._by_key.get(combo.currentData()) if combo.currentData() else None

    @staticmethod
    def _route_value(run: ComparisonRun, row: Dict[str, Any]) -> Any:
        if run.track == "DEVELOPMENT_DIAGNOSTICS":
            return row.get("route_progress_delta")
        return row.get("final_route_completion")

    @staticmethod
    def _censor_timeout(run: ComparisonRun, row: Dict[str, Any]) -> str:
        if run.track == "DEVELOPMENT_DIAGNOSTICS":
            if row.get("outcome") == "PROBE_CENSORED":
                return "PROBE_CENSORED"
            return "TIMEOUT" if row.get("timeout") else "—"
        return "TIMEOUT" if str(row.get("timeout", "")).lower() in {"true", "1"} else "—"

    @staticmethod
    def _outcome(run: ComparisonRun, row: Dict[str, Any]) -> str:
        if run.track == "DEVELOPMENT_DIAGNOSTICS":
            return str(row.get("outcome", "UNKNOWN"))
        return str(row.get("primary_terminal_reason") or row.get("terminal_reason") or "UNKNOWN")

    @staticmethod
    def _metric_specs(run: ComparisonRun) -> List[tuple[str, str, bool]]:
        if run.track == "CANONICAL_EVALUATION":
            return [
                ("clean_success_rate", "Clean success rate", True),
                ("safety_failure_rate", "Safety failure rate", True),
                ("mean_final_route_completion", "Mean final route completion", False),
                ("median_final_route_completion", "Median final route completion", False),
                ("mean_time_to_clean_success_s", "Mean time to clean success (s)", False),
            ]
        return [
            ("mean_route_progress_delta", "Mean route-progress delta (diagnostic)", False),
            ("mean_speed_km_h", "Mean speed (km/h)", False),
            ("stationary_fraction", "Stationary fraction (<1 km/h)", True),
            ("mean_throttle", "Mean throttle", False),
            ("mean_brake", "Mean brake", False),
            ("mean_steering", "Mean steering", False),
            ("clean_success_count", "Clean successes (count)", False),
            ("safety_failure_count", "Safety failures (count)", False),
            ("probe_censored", "PROBE_CENSORED (count)", False),
            ("official_timeout_count", "Official timeout (count)", False),
            ("mean_deterministic_min_q", "Mean deterministic min-Q (critic diagnostic)", False),
            ("mean_stochastic_min_q", "Mean stochastic min-Q (critic diagnostic)", False),
            ("mean_target_min_q", "Mean target min-Q (critic diagnostic)", False),
            ("mean_stochastic_positive_throttle_fraction", "Stochastic positive-throttle fraction", True),
            ("mean_entropy_coefficient", "Entropy coefficient (training diagnostic)", False),
        ]

    def render_comparison(self, *_args: Any) -> None:
        left, right = self._selected(self.combo_left), self._selected(self.combo_right)
        self.table_metrics.setRowCount(0)
        self.table_tiers.setRowCount(0)
        self.table_cases.setRowCount(0)
        self.ax.clear()
        if left is None or right is None:
            self._last_result = None
            self.lbl_status.setText(self._source_error or "No data: select two sources with valid artifacts.")
            self.lbl_provenance.setText("No provenance available.")
            self.canvas.draw_idle()
            return

        comparison = compare_runs(left, right)
        self._last_result = comparison
        track_label = "TRAIN ONLY — NOT BENCHMARK" if left.track == "DEVELOPMENT_DIAGNOSTICS" else "CANONICAL EVALUATION"
        self.lbl_status.setText(f"{track_label} · {comparison.status}: {comparison.reason}")
        self.lbl_status.setStyleSheet("font-weight: bold; color: " + ("#185c37" if comparison.directly_comparable else "#8a4b08") + "; padding: 5px;")
        self.lbl_provenance.setText(self._provenance_text(left, "A") + "\n" + self._provenance_text(right, "B"))

        specs = self._metric_specs(left)
        for key, label, percent in specs:
            value_a = left.summary.get(key)
            value_b = right.summary.get(key) if comparison.directly_comparable else right.summary.get(key)
            delta = None
            if comparison.directly_comparable and value_a is not None and value_b is not None:
                delta = float(value_b) - float(value_a)
            self._append_row(self.table_metrics, [label, _fmt(value_a, percent), _fmt(value_b, percent),
                                                  _fmt(delta, percent) if delta is not None else "—"])

        self._render_tiers(left, right, comparison, percent=left.track == "CANONICAL_EVALUATION")
        if comparison.directly_comparable:
            self._render_cases(left, comparison)
        else:
            self._append_row(self.table_cases, [comparison.reason] + ["—"] * 11)
        self.canvas.draw_idle()

    def _provenance_text(self, run: ComparisonRun, side: str) -> str:
        return (
            f"{side}: {run.agent_id} {run.agent_version} · {run.method_family} · "
            f"split={run.dataset_split} · mode={run.inference_mode or 'unknown'} · "
            f"adapter={run.action_adapter_id or 'unknown'} · checkpoint={run.checkpoint_sha256 or 'not recorded'} · "
            f"bundle={run.checkpoint_bundle_status} · source={run.source_revision or 'not recorded'} · "
            f"config={run.provenance.get('config_sha256') or run.provenance.get('experiment_config_sha256') or 'not recorded'} · "
            f"seed={run.provenance.get('training_root_seed', run.provenance.get('agent_seed', 'not recorded'))} · "
            f"eligible={run.benchmark_eligible} · cases={len(run.schedule)}"
        )

    def _render_tiers(self, left: ComparisonRun, right: ComparisonRun,
                      comparison: ComparisonResult, percent: bool) -> None:
        tiers = sorted(set(left.per_tier) | set(right.per_tier))
        primary_key = "mean_final_route_completion" if left.track == "CANONICAL_EVALUATION" else "mean_route_progress_delta"
        label = "Mean final route completion" if percent else "Mean route-progress delta"
        for tier in tiers:
            a, b = left.per_tier.get(tier, {}), right.per_tier.get(tier, {})
            va, vb = a.get(primary_key), b.get(primary_key)
            delta = float(vb) - float(va) if comparison.directly_comparable and va is not None and vb is not None else None
            cases = sum(1 for row in left.records if row.get("tier") == tier)
            self._append_row(self.table_tiers, [tier, label, _fmt(va), _fmt(vb), _fmt(delta), cases])

    def _render_cases(self, left: ComparisonRun, comparison: ComparisonResult) -> None:
        route_differences = []
        labels = []
        for pair in comparison.paired_records:
            a, b = pair["left"], pair["right"]
            route_a, route_b = self._route_value(left, a), self._route_value(self._selected(self.combo_right), b)
            delta = float(route_b) - float(route_a) if route_a is not None and route_b is not None else None
            self._append_row(self.table_cases, [
                pair["case_id"], pair.get("tier") or "—", pair["environment_seed"],
                self._outcome(left, a), self._outcome(self._selected(self.combo_right), b),
                _fmt(route_a), _fmt(route_b), _fmt(delta),
                _fmt(a.get("safety_failure", a.get("primary_terminal_reason"))),
                _fmt(b.get("safety_failure", b.get("primary_terminal_reason"))),
                self._censor_timeout(left, a), self._censor_timeout(self._selected(self.combo_right), b),
            ])
            if delta is not None:
                route_differences.append(delta)
                labels.append(str(pair["case_id"]))
        if route_differences:
            self.ax.bar(range(len(route_differences)), route_differences, color="#3478b8")
            self.ax.axhline(0, color="black", linewidth=.8)
            self.ax.set_title("Per-case route difference (B − A)")
            self.ax.set_ylabel("Route progress delta" if left.track == "DEVELOPMENT_DIAGNOSTICS" else "Final route completion")
            self.ax.set_xticks(range(len(labels)))
            self.ax.set_xticklabels([f"{i + 1}" for i in range(len(labels))])
            self.ax.set_xlabel("Paired cases (see table for case IDs)")
        else:
            self.ax.text(.5, .5, "No route values available for paired visualization.", ha="center", va="center",
                         transform=self.ax.transAxes)

    @staticmethod
    def _append_row(table: QTableWidget, values: List[Any]) -> None:
        row = table.rowCount()
        table.insertRow(row)
        for column, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, column, item)

    def _selected_folder(self, side: str) -> Optional[Path]:
        run = self._selected(self.combo_left if side == "left" else self.combo_right)
        if run is None:
            return None
        path = run.provenance.get("folder")
        return Path(path) if path else None

    def _open_folder(self, side: str) -> None:
        folder = self._selected_folder(side)
        if folder and folder.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        else:
            QMessageBox.information(self, "Artifact folder unavailable", "The local raw artifact folder is missing.")

    def export_summary(self) -> None:
        left, right = self._selected(self.combo_left), self._selected(self.combo_right)
        if left is None or right is None or self._last_result is None:
            QMessageBox.information(self, "No comparison", "Select two sources with valid artifact records first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export comparison", "stage5_comparison.json", "JSON (*.json)")
        if not path:
            return
        payload = {
            "schema": "WorkbenchComparisonViewV1",
            "created_from_authoritative_local_artifacts": True,
            "status": self._last_result.status,
            "directly_comparable": self._last_result.directly_comparable,
            "reason": self._last_result.reason,
            "left": {"key": left.key, "label": left.label, "track": left.track,
                     "checkpoint_sha256": left.checkpoint_sha256, "provenance": left.provenance,
                     "summary": left.summary},
            "right": {"key": right.key, "label": right.label, "track": right.track,
                      "checkpoint_sha256": right.checkpoint_sha256, "provenance": right.provenance,
                      "summary": right.summary},
            "paired_records": list(self._last_result.paired_records) if self._last_result.directly_comparable else [],
            "official_leaderboard": False,
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
