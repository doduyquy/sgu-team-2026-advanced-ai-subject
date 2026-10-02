"""
Workbench Results Tab and Run Inspection Panel (Gate 7.5B Pass B2 Correction 2).

Provides comprehensive read-only browsing of Gate-7 experiment runs:
- Left pane: Discovered run candidate list (Run ID, Status, Kind, Canonical, Agent, Episodes, Integrity, Started time).
- Right pane (Subtabs):
  1. Overview: Authoritative primary metric cards from summary.json['overall_metrics'] (strictly trust-gated upon VERIFIED),
     selected tier breakdown, stored macro_metrics display, complete 9-outcome rates breakdown bar chart, and diagnostic return signal.
  2. Episodes: Complete read-only table displaying all 31 Gate-7 columns directly single-sourced from EPISODE_CSV_COLUMNS.
  3. Timing: Decision latency timing table displaying all 8 fields single-sourced from dataclasses.fields(EpisodeTimingRecord).
  4. Provenance: Experiment configuration, git provenance, environment provenance, and platform contract hashes.
  5. Integrity: Artifact hash check verification against run_integrity.json.
  6. W&B / Tech: W&B sync status and technical failure diagnostics.
- All presentation tables use QAbstractItemView.EditTrigger.NoEditTriggers.
- Zero delete, rename, edit, or metric recomputation controls.
"""

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.launcher.cases import get_default_project_root
from src.platform.experiment_logging import EPISODE_CSV_COLUMNS, EpisodeTimingRecord
from src.workbench.results_repository import (
    IntegrityDisplayStatus,
    RunArtifactRepository,
    RunArtifactSnapshotV1,
)


class ResultsWidget(QWidget):
    """Sixth main tab: Authoritative Gate-7 experiment results browser."""

    def __init__(self, runs_root: Optional[Path] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        project_root = get_default_project_root()
        self._runs_root = Path(runs_root).resolve() if runs_root else (project_root / "runs").resolve()
        self._repo = RunArtifactRepository(self._runs_root)
        self._current_snapshot: Optional[RunArtifactSnapshotV1] = None
        self._init_ui()
        self.refresh_runs_list()

    @property
    def runs_root(self) -> Path:
        return self._runs_root

    def set_runs_root(self, runs_root: Path) -> None:
        """Sets a new runs_root and refreshes the run list."""
        self._runs_root = Path(runs_root).resolve()
        self._repo = RunArtifactRepository(self._runs_root)
        self.lbl_current_root.setText(f"Root: {self._runs_root.name}")
        self.refresh_runs_list()

    def _init_ui(self) -> None:
        main_layout = QHBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # -------------------------------------------------------------
        # Left Pane: Run Browser List (Specification Section 15)
        # Required columns: Run ID, Status, Kind, Canonical, Agent, Episodes, Integrity, Started Time
        # -------------------------------------------------------------
        left_group = QGroupBox("Discovered Runs Repository")
        left_layout = QVBoxLayout(left_group)

        # Top controls: Runs root selector and refresh button
        root_row = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh Runs")
        self.btn_refresh.clicked.connect(self.refresh_runs_list)
        root_row.addWidget(self.btn_refresh)

        self.btn_browse_root = QPushButton("Change Root...")
        self.btn_browse_root.clicked.connect(self._on_browse_root_clicked)
        root_row.addWidget(self.btn_browse_root)
        left_layout.addLayout(root_row)

        self.lbl_current_root = QLabel(f"Root: {self._runs_root.name}")
        self.lbl_current_root.setStyleSheet("color: gray; font-size: 11px;")
        left_layout.addWidget(self.lbl_current_root)

        # Runs table (8 required columns)
        self.table_runs = QTableWidget(0, 8)
        self.table_runs.setHorizontalHeaderLabels([
            "Run ID", "Status", "Kind", "Canonical", "Agent", "Episodes", "Integrity", "Started Time"
        ])
        for col in range(7):
            self.table_runs.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.table_runs.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        self.table_runs.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_runs.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_runs.itemSelectionChanged.connect(self._on_run_selection_changed)
        left_layout.addWidget(self.table_runs)

        splitter.addWidget(left_group)

        # -------------------------------------------------------------
        # Right Pane: Selected Run Details (Subtabs)
        # -------------------------------------------------------------
        right_group = QGroupBox("Selected Run Inspection (Authoritative Local Artifacts)")
        right_layout = QVBoxLayout(right_group)

        self.subtabs = QTabWidget()

        # Subtab 1: Overview
        self.tab_overview = QWidget()
        self._init_overview_tab()
        self.subtabs.addTab(self.tab_overview, "Overview & Metrics")

        # Subtab 2: Episodes
        self.tab_episodes = QWidget()
        self._init_episodes_tab()
        self.subtabs.addTab(self.tab_episodes, "Episodes (episodes.csv)")

        # Subtab 3: Timing
        self.tab_timing = QWidget()
        self._init_timing_tab()
        self.subtabs.addTab(self.tab_timing, "Decision Latency (timing.csv)")

        # Subtab 4: Provenance
        self.tab_provenance = QWidget()
        self._init_provenance_tab()
        self.subtabs.addTab(self.tab_provenance, "Provenance (run_manifest.json)")

        # Subtab 5: Integrity
        self.tab_integrity = QWidget()
        self._init_integrity_tab()
        self.subtabs.addTab(self.tab_integrity, "Artifact Integrity (run_integrity.json)")

        # Subtab 6: W&B / Tech Failures
        self.tab_wandb = QWidget()
        self._init_wandb_tab()
        self.subtabs.addTab(self.tab_wandb, "W&B / Diagnostics")

        right_layout.addWidget(self.subtabs)
        splitter.addWidget(right_group)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter)

    # -------------------------------------------------------------
    # Subtab Inits
    # -------------------------------------------------------------
    def _init_overview_tab(self) -> None:
        layout = QVBoxLayout(self.tab_overview)

        # Top banner with Run ID and Status
        banner_row = QHBoxLayout()
        self.lbl_selected_run_id = QLabel("Select a run from the list")
        self.lbl_selected_run_id.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.lbl_integrity_badge = QLabel("—")
        self.lbl_integrity_badge.setStyleSheet("font-weight: bold; padding: 3px;")
        banner_row.addWidget(self.lbl_selected_run_id)
        banner_row.addStretch()
        banner_row.addWidget(self.lbl_integrity_badge)
        layout.addLayout(banner_row)

        # Primary Benchmark Metric Cards Group (Trust-Gated on Integrity VERIFIED)
        self.cards_group = QGroupBox("Authoritative Primary Benchmark Metrics (summary.json)")
        cards_layout = QHBoxLayout(self.cards_group)

        self.card_success = self._create_metric_card("Clean Success Rate", "—")
        self.card_safety = self._create_metric_card("Safety Failure Rate", "—")
        self.card_mean_route = self._create_metric_card("Mean Route Completion", "—")
        self.card_median_route = self._create_metric_card("Median Route Completion", "—")
        self.card_time = self._create_metric_card("Mean Time to Success", "—")

        cards_layout.addWidget(self.card_success)
        cards_layout.addWidget(self.card_safety)
        cards_layout.addWidget(self.card_mean_route)
        cards_layout.addWidget(self.card_median_route)
        cards_layout.addWidget(self.card_time)
        layout.addWidget(self.cards_group)

        # Trust Warning Banner (Visible on FAILED / UNVERIFIED / NOT_FINAL integrity)
        self.lbl_trust_warning = QLabel("")
        self.lbl_trust_warning.setVisible(False)
        layout.addWidget(self.lbl_trust_warning)

        # Diagnostic Signal Labeling Note
        diag_note = QLabel(
            "Diagnostic Notice: Training metrics such as 'episode_return' and 'mean_episode_return' "
            "are training signals — NOT benchmark ranking scores."
        )
        diag_note.setStyleSheet("color: #b8860b; font-style: italic; font-size: 11px;")
        layout.addWidget(diag_note)

        # Middle splitter: Outcome Breakdown Chart & Text Summary
        mid_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Outcome Chart
        self.chart_box = QGroupBox("Outcome Breakdown (Authoritative Rates)")
        chart_layout = QVBoxLayout(self.chart_box)
        self.overview_figure = Figure(figsize=(5.5, 3.2), dpi=90)
        self.overview_canvas = FigureCanvasQTAgg(self.overview_figure)
        self.ax_outcomes = self.overview_figure.add_subplot(1, 1, 1)
        self._setup_overview_chart()
        chart_layout.addWidget(self.overview_canvas)
        mid_splitter.addWidget(self.chart_box)

        # Text Summary
        text_box = QGroupBox("Summary Details (Tier Breakdown & Macro Metrics)")
        text_layout = QVBoxLayout(text_box)
        self.text_summary_details = QTextEdit()
        self.text_summary_details.setReadOnly(True)
        text_layout.addWidget(self.text_summary_details)
        mid_splitter.addWidget(text_box)

        layout.addWidget(mid_splitter)

    def _create_metric_card(self, title: str, value: str) -> QWidget:
        widget = QWidget()
        w_layout = QVBoxLayout(widget)
        w_layout.setContentsMargins(4, 4, 4, 4)
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("font-size: 10px; color: gray;")
        lbl_val = QLabel(value)
        lbl_val.setStyleSheet("font-size: 15px; font-weight: bold;")
        lbl_val.setObjectName("value_label")
        w_layout.addWidget(lbl_title)
        w_layout.addWidget(lbl_val)
        return widget

    def _setup_overview_chart(self, is_authoritative: bool = True) -> None:
        self.ax_outcomes.clear()
        title = "Outcome Rates Breakdown (Authoritative Rates)" if is_authoritative else "Outcome Rates Breakdown (Untrusted / Non-Verified Rates)"
        self.ax_outcomes.set_title(title, fontsize=8)
        self.ax_outcomes.set_ylabel("Rate", fontsize=8)
        self.ax_outcomes.set_ylim(0, 1.05)
        self.ax_outcomes.grid(True, linestyle="--", alpha=0.5, axis="y")
        self.overview_figure.tight_layout()

    def _init_episodes_tab(self) -> None:
        layout = QVBoxLayout(self.tab_episodes)
        # Single-source directly from Gate-7 EPISODE_CSV_COLUMNS (31 columns)
        self._episode_columns = list(EPISODE_CSV_COLUMNS)
        self.table_episodes = QTableWidget(0, len(self._episode_columns))
        self.table_episodes.setHorizontalHeaderLabels(self._episode_columns)
        for c in range(len(self._episode_columns)):
            self.table_episodes.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table_episodes.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_episodes.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table_episodes)

    def _init_timing_tab(self) -> None:
        layout = QVBoxLayout(self.tab_timing)
        timing_note = QLabel("Decision latency strictly measures agent.act() computation time (timing.csv).")
        timing_note.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(timing_note)

        # Single-source directly from EpisodeTimingRecord dataclass fields (8 fields)
        self._timing_columns = [f.name for f in dataclasses.fields(EpisodeTimingRecord)]
        self.table_timing = QTableWidget(0, len(self._timing_columns))
        self.table_timing.setHorizontalHeaderLabels(self._timing_columns)
        for c in range(len(self._timing_columns)):
            self.table_timing.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table_timing.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_timing.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table_timing)

    def _init_provenance_tab(self) -> None:
        layout = QVBoxLayout(self.tab_provenance)
        self.text_provenance = QTextEdit()
        self.text_provenance.setReadOnly(True)
        layout.addWidget(self.text_provenance)

    def _init_integrity_tab(self) -> None:
        layout = QVBoxLayout(self.tab_integrity)
        self.lbl_integrity_verdict = QLabel("Integrity Status: NOT EVALUATED")
        self.lbl_integrity_verdict.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px;")
        layout.addWidget(self.lbl_integrity_verdict)

        self.table_integrity = QTableWidget(0, 4)
        self.table_integrity.setHorizontalHeaderLabels([
            "Artifact", "Status", "Expected Fingerprint", "Observed Fingerprint"
        ])
        self.table_integrity.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_integrity.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_integrity.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table_integrity.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table_integrity.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_integrity.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table_integrity)

    def _init_wandb_tab(self) -> None:
        layout = QVBoxLayout(self.tab_wandb)
        self.text_wandb = QTextEdit()
        self.text_wandb.setReadOnly(True)
        layout.addWidget(self.text_wandb)

    # -------------------------------------------------------------
    # Logic & Presentation Population
    # -------------------------------------------------------------
    def _on_browse_root_clicked(self) -> None:
        selected_dir = QFileDialog.getExistingDirectory(
            self, "Select Runs Root Directory", str(self._runs_root)
        )
        if selected_dir:
            self.set_runs_root(Path(selected_dir))

    def refresh_runs_list(self) -> None:
        """Discovers run directories and populates the left runs table."""
        run_ids = self._repo.discover_run_ids()
        self.table_runs.setRowCount(len(run_ids))

        for row, rid in enumerate(run_ids):
            snap = self._repo.load_run_snapshot(rid)
            if not snap:
                self.table_runs.setItem(row, 0, QTableWidgetItem(rid))
                self.table_runs.setItem(row, 1, QTableWidgetItem("MALFORMED"))
                for c in range(2, 8):
                    self.table_runs.setItem(row, c, QTableWidgetItem("—"))
                continue

            item_id = QTableWidgetItem(snap.run_id)
            item_status = QTableWidgetItem(snap.status)
            item_kind = QTableWidgetItem(snap.run_kind)
            item_canon = QTableWidgetItem(str(snap.canonical_run))
            item_agent = QTableWidgetItem(snap.agent_id or "—")
            item_eps = QTableWidgetItem(f"{snap.recorded_episode_count}/{snap.expected_episode_count or '—'}")
            item_integ = QTableWidgetItem(snap.integrity_status.value)
            item_start = QTableWidgetItem(snap.started_at_utc or "—")

            if snap.integrity_status == IntegrityDisplayStatus.VERIFIED:
                item_integ.setForeground(Qt.GlobalColor.darkGreen)
            elif snap.integrity_status == IntegrityDisplayStatus.FAILED:
                item_integ.setForeground(Qt.GlobalColor.darkRed)
            elif snap.integrity_status == IntegrityDisplayStatus.NOT_FINAL:
                item_integ.setForeground(Qt.GlobalColor.darkYellow)

            self.table_runs.setItem(row, 0, item_id)
            self.table_runs.setItem(row, 1, item_status)
            self.table_runs.setItem(row, 2, item_kind)
            self.table_runs.setItem(row, 3, item_canon)
            self.table_runs.setItem(row, 4, item_agent)
            self.table_runs.setItem(row, 5, item_eps)
            self.table_runs.setItem(row, 6, item_integ)
            self.table_runs.setItem(row, 7, item_start)

        if run_ids and self.table_runs.currentRow() < 0:
            self.table_runs.selectRow(0)

    def select_run_by_id(self, run_id: str) -> None:
        """Selects a specific run in the table (e.g. on execution completion)."""
        self.refresh_runs_list()
        for row in range(self.table_runs.rowCount()):
            item = self.table_runs.item(row, 0)
            if item and item.text() == run_id:
                self.table_runs.setCurrentCell(row, 0)
                self.table_runs.selectRow(row)
                snap = self._repo.load_run_snapshot(run_id)
                if snap:
                    self.display_snapshot(snap)
                break

    def _on_run_selection_changed(self) -> None:
        selected_rows = self.table_runs.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        item = self.table_runs.item(row, 0)
        if not item:
            return
        run_id = item.text()
        snap = self._repo.load_run_snapshot(run_id)
        if snap:
            self.display_snapshot(snap)

    def display_snapshot(self, snap: RunArtifactSnapshotV1) -> None:
        """Renders RunArtifactSnapshotV1 across all detail tabs."""
        self._current_snapshot = snap
        self.lbl_selected_run_id.setText(f"Run: {snap.run_id} ({snap.status} | {snap.run_kind})")

        # Integrity badge
        self.lbl_integrity_badge.setText(f"Integrity: {snap.integrity_status.value}")
        if snap.integrity_status == IntegrityDisplayStatus.VERIFIED:
            self.lbl_integrity_badge.setStyleSheet("color: white; background: #228b22; font-weight: bold; padding: 4px; border-radius: 3px;")
        elif snap.integrity_status == IntegrityDisplayStatus.FAILED:
            self.lbl_integrity_badge.setStyleSheet("color: white; background: #b22222; font-weight: bold; padding: 4px; border-radius: 3px;")
        elif snap.integrity_status == IntegrityDisplayStatus.NOT_FINAL:
            self.lbl_integrity_badge.setStyleSheet("color: black; background: #ffd700; font-weight: bold; padding: 4px; border-radius: 3px;")
        else:
            self.lbl_integrity_badge.setStyleSheet("color: white; background: #708090; font-weight: bold; padding: 4px; border-radius: 3px;")

        # Populate Overview
        self._populate_overview(snap)
        # Populate Episodes
        self._populate_episodes(snap)
        # Populate Timing
        self._populate_timing(snap)
        # Populate Provenance
        self._populate_provenance(snap)
        # Populate Integrity
        self._populate_integrity(snap)
        # Populate W&B / Tech
        self._populate_wandb(snap)

    def _populate_overview(self, snap: RunArtifactSnapshotV1) -> None:
        def set_card_val(card: QWidget, text: str):
            lbl = card.findChild(QLabel, "value_label")
            if lbl:
                lbl.setText(text)

        summary = snap.summary_payload or {}
        overall = summary.get("overall_metrics", {})

        # Strict Result-Trust Model (Specification Section 5):
        # ONLY IntegrityDisplayStatus.VERIFIED runs may be called Authoritative
        is_verified = (snap.integrity_status == IntegrityDisplayStatus.VERIFIED)
        is_failed = (snap.integrity_status == IntegrityDisplayStatus.FAILED)
        is_unverified = (snap.integrity_status == IntegrityDisplayStatus.UNVERIFIED)
        is_not_final = (snap.integrity_status == IntegrityDisplayStatus.NOT_FINAL)

        if is_verified:
            self.cards_group.setTitle("Authoritative Primary Benchmark Metrics (summary.json)")
            self.cards_group.setStyleSheet("")
            self.chart_box.setTitle("Outcome Breakdown (Authoritative Rates)")
            self.lbl_trust_warning.setVisible(False)
            self.lbl_trust_warning.setText("")
        elif is_failed:
            self.cards_group.setTitle("UNTRUSTED ARTIFACT CONTENT — INTEGRITY FAILED (summary.json)")
            self.cards_group.setStyleSheet("QGroupBox { font-weight: bold; color: #b22222; border: 2px solid #b22222; }")
            self.chart_box.setTitle("Outcome Breakdown (Untrusted Rates — Integrity Failed)")
            self.lbl_trust_warning.setVisible(True)
            self.lbl_trust_warning.setStyleSheet("color: #b22222; font-weight: bold; background: #ffe6e6; padding: 6px; border: 1px solid #b22222;")
            self.lbl_trust_warning.setText(
                "CRITICAL WARNING: Run integrity verification FAILED! Stored metrics mismatch canonical "
                "artifact fingerprints and must NOT be trusted as authoritative benchmark results."
            )
        elif is_unverified:
            self.cards_group.setTitle("UNVERIFIED ARTIFACT CONTENT (run_integrity.json Missing)")
            self.cards_group.setStyleSheet("QGroupBox { font-weight: bold; color: #708090; border: 2px solid #708090; }")
            self.chart_box.setTitle("Outcome Breakdown (Unverified Rates)")
            self.lbl_trust_warning.setVisible(True)
            self.lbl_trust_warning.setStyleSheet("color: #4f4f4f; font-weight: bold; background: #f0f0f0; padding: 6px; border: 1px solid #a0a0a0;")
            self.lbl_trust_warning.setText(
                "NOTICE: Run completed without run_integrity.json. Artifacts cannot be verified as authoritative scientific results."
            )
        else:  # NOT_FINAL
            self.cards_group.setTitle(f"PROVISIONAL / NOT FINAL METRICS ({snap.status})")
            self.cards_group.setStyleSheet("QGroupBox { font-weight: bold; color: #b8860b; border: 2px solid #b8860b; }")
            self.chart_box.setTitle(f"Outcome Breakdown (Provisional Rates — {snap.status})")
            self.lbl_trust_warning.setVisible(True)
            self.lbl_trust_warning.setStyleSheet("color: #8b6508; font-weight: bold; background: #fff8dc; padding: 6px; border: 1px solid #ffd700;")
            self.lbl_trust_warning.setText(
                f"NOTICE: Run lifecycle is {snap.status}. Displayed metrics are provisional/not-final and have not been finalized."
            )

        clean_succ = overall.get("clean_success_rate")
        safety_fail = overall.get("safety_failure_rate")
        mean_route = overall.get("mean_final_route_completion")
        med_route = overall.get("median_final_route_completion")
        mean_time = overall.get("mean_time_to_clean_success_s")

        set_card_val(self.card_success, f"{clean_succ * 100:.1f}%" if clean_succ is not None else "—")
        set_card_val(self.card_safety, f"{safety_fail * 100:.1f}%" if safety_fail is not None else "—")
        set_card_val(self.card_mean_route, f"{mean_route * 100:.1f}%" if mean_route is not None else "—")
        set_card_val(self.card_median_route, f"{med_route * 100:.1f}%" if med_route is not None else "—")
        if mean_time is not None:
            set_card_val(self.card_time, f"{mean_time:.2f} s")
        else:
            set_card_val(self.card_time, "N/A — no clean-success samples" if clean_succ is not None else "—")

        # Summary text details
        trust_header = "=== AUTHORITATIVE EXPERIMENT SUMMARY ===" if is_verified else f"=== EXPERIMENT SUMMARY ({snap.integrity_status.value}) ==="
        lines = [
            trust_header,
            f"Run ID:                          {snap.run_id}",
            f"Lifecycle Status:                {snap.status}",
            f"Integrity Status:                {snap.integrity_status.value}",
            f"Canonical Run:                   {snap.canonical_run}",
            f"Recorded Episodes:               {snap.recorded_episode_count} / {snap.expected_episode_count or '—'}",
            f"Started UTC:                     {snap.started_at_utc or '—'}",
            f"Finished UTC:                    {snap.finished_at_utc or '—'}",
        ]

        if snap.failure_category or snap.sanitized_failure_message:
            lines.append(f"Failure Category:                {snap.failure_category or '—'}")
            lines.append(f"Failure Message:                 {snap.sanitized_failure_message or '—'}")

        lines.extend([
            "",
            f"--- Diagnostic Signal (NOT Ranking Score) ---",
            f"Diagnostic Mean Return:          {overall.get('mean_episode_return', '—')}",
        ])

        # Stored Macro Metrics
        macro_metrics = summary.get("macro_metrics")
        if macro_metrics:
            macro_label = "Stored Authoritative Macro Metrics (summary.json)" if is_verified else f"Stored Macro Metrics ({snap.integrity_status.value})"
            lines.append("")
            lines.append(f"--- {macro_label} ---")
            for mk, mv in macro_metrics.items():
                lines.append(f"  {mk}: {mv}")

        # Tier metrics
        tier_metrics = summary.get("tier_metrics")
        if tier_metrics:
            tier_label = "Tier Summary (summary.json)" if is_verified else f"Tier Summary ({snap.integrity_status.value})"
            lines.append("")
            lines.append(f"--- {tier_label} ---")
            for t_name, t_vals in tier_metrics.items():
                t_succ = t_vals.get("clean_success_rate")
                t_route = t_vals.get("mean_final_route_completion")
                lines.append(f"  Tier {t_name}: Clean Success={f'{t_succ*100:.1f}%' if t_succ is not None else '—'} | Mean Route={f'{t_route*100:.1f}%' if t_route is not None else '—'}")

        self.text_summary_details.setText("\n".join(lines))

        # Outcome Chart (Truthful 9 Mutually-Exclusive Outcome Rates, Specification Section 10-11)
        # Distinguish actual 0.0 from missing/None. Do NOT fabricate missing rates as 0.0.
        self._setup_overview_chart(is_authoritative=is_verified)
        outcome_defs = [
            ("success_rate", "Success", "#2ca02c"),
            ("timeout_rate", "Timeout", "#ff7f0e"),
            ("crash_human_rate", "Crash Hum", "#9467bd"),
            ("crash_vehicle_rate", "Crash Veh", "#d62728"),
            ("crash_object_rate", "Crash Obj", "#8c564b"),
            ("crash_building_rate", "Crash Bld", "#e377c2"),
            ("crash_sidewalk_rate", "Crash Swk", "#7f7f7f"),
            ("out_of_road_rate", "Off Road", "#bcbd22"),
            ("unknown_termination_rate", "Unknown", "#17becf"),
        ]

        present_labels = []
        present_rates = []
        present_colors = []

        for key, lbl, color in outcome_defs:
            val = overall.get(key)
            if val is not None:
                try:
                    num_val = float(val)
                    present_labels.append(lbl)
                    present_rates.append(num_val)
                    present_colors.append(color)
                except (ValueError, TypeError):
                    pass

        if present_labels:
            self.ax_outcomes.bar(present_labels, present_rates, color=present_colors)
            self.ax_outcomes.tick_params(axis="x", rotation=35, labelsize=7.5)
        else:
            self.ax_outcomes.text(0.5, 0.5, "No stored outcome-rate data available", horizontalalignment='center', verticalalignment='center', transform=self.ax_outcomes.transAxes, color='gray', fontsize=8)

        self.overview_figure.tight_layout()
        self.overview_canvas.draw_idle()

    def _populate_episodes(self, snap: RunArtifactSnapshotV1) -> None:
        rows = snap.episode_rows
        self.table_episodes.setRowCount(len(rows))
        for r_idx, r in enumerate(rows):
            for c_idx, col_name in enumerate(self._episode_columns):
                val = r.get(col_name, "")
                self.table_episodes.setItem(r_idx, c_idx, QTableWidgetItem(str(val)))

    def _populate_timing(self, snap: RunArtifactSnapshotV1) -> None:
        rows = snap.timing_rows
        self.table_timing.setRowCount(len(rows))
        for r_idx, r in enumerate(rows):
            for c_idx, col_name in enumerate(self._timing_columns):
                val = r.get(col_name, "")
                self.table_timing.setItem(r_idx, c_idx, QTableWidgetItem(str(val)))

    def _populate_provenance(self, snap: RunArtifactSnapshotV1) -> None:
        lines = [
            f"=== EXPERIMENT PROVENANCE (run_manifest.json & run_state.json) ===",
            f"Run ID:                          {snap.run_id}",
            f"Run Directory:                   {snap.run_dir}",
            f"Run Kind:                        {snap.run_kind}",
            f"Canonical Run:                   {snap.canonical_run}",
            f"Dirty Worktree Override:         {snap.dirty_override}",
            f"Unverified Environment Override: {snap.unverified_env_override}",
            f"Environment Verification Status: {snap.environment_verification_status}",
            "",
            f"--- Agent Specification ---",
            f"Agent ID:                        {snap.agent_id}",
            f"Agent Version:                   {snap.agent_version}",
            f"Input Profile ID:                {snap.input_profile_id}",
            f"Action Adapter ID:               {snap.action_adapter_id}",
            f"Inference Stochasticity:         {snap.inference_stochasticity}",
            f"Stateful in Episode:             {snap.stateful_within_episode}",
            f"Agent Seed:                      {snap.agent_seed}",
            f"Manifest Name:                   {snap.manifest_name}",
            f"Experiment Config SHA-256:       {snap.experiment_config_sha256}",
            "",
            f"--- Provenance ---",
            f"Git Commit SHA:                  {snap.git_commit_sha}",
            f"Git Worktree Dirty:              {snap.git_worktree_dirty}",
            f"MetaDrive Version:               {snap.metadrive_version}",
            f"MetaDrive Commit:                {snap.metadrive_commit}",
            f"MetaDrive Pinned Version:        {snap.metadrive_pinned_version}",
            f"MetaDrive Pinned Commit:         {snap.metadrive_pinned_commit}",
            f"MetaDrive Verification Status:   {snap.metadrive_verification_status}",
            f"MetaDrive Verification Reason:   {snap.metadrive_verification_reason}",
            "",
            f"--- Platform Contract Hashes ---",
        ]
        for ck, cv in snap.platform_contracts.items():
            lines.append(f"  {ck}: {cv}")

        lines.append("")
        lines.append("--- Local Artifact Presence ---")
        for art in snap.artifacts_present:
            lines.append(f"  * {art}")

        self.text_provenance.setText("\n".join(lines))

    def _populate_integrity(self, snap: RunArtifactSnapshotV1) -> None:
        self.lbl_integrity_verdict.setText(f"Overall Integrity Verdict: {snap.integrity_status.value}")
        if snap.integrity_status == IntegrityDisplayStatus.VERIFIED:
            self.lbl_integrity_verdict.setStyleSheet("color: darkgreen; font-weight: bold; padding: 4px;")
        elif snap.integrity_status == IntegrityDisplayStatus.FAILED:
            self.lbl_integrity_verdict.setStyleSheet("color: darkred; font-weight: bold; padding: 4px;")
        elif snap.integrity_status == IntegrityDisplayStatus.NOT_FINAL:
            self.lbl_integrity_verdict.setStyleSheet("color: #b8860b; font-weight: bold; padding: 4px;")
        else:
            self.lbl_integrity_verdict.setStyleSheet("color: gray; font-weight: bold; padding: 4px;")

        checks = snap.integrity_details
        self.table_integrity.setRowCount(len(checks))
        for row, c in enumerate(checks):
            self.table_integrity.setItem(row, 0, QTableWidgetItem(c.artifact_name))
            item_status = QTableWidgetItem(c.status)
            if c.status == "PASS":
                item_status.setForeground(Qt.GlobalColor.darkGreen)
            elif c.status == "FAIL":
                item_status.setForeground(Qt.GlobalColor.darkRed)
            elif c.status == "MISSING":
                item_status.setForeground(Qt.GlobalColor.darkRed)
            self.table_integrity.setItem(row, 1, item_status)
            self.table_integrity.setItem(row, 2, QTableWidgetItem(str(c.expected_sha256 or "—")))
            self.table_integrity.setItem(row, 3, QTableWidgetItem(str(c.observed_sha256 or "—")))

    def _populate_wandb(self, snap: RunArtifactSnapshotV1) -> None:
        lines = ["=== W&B SYNC STATUS (wandb_sync.json) ==="]
        if snap.wandb_sync_payload:
            for k, v in snap.wandb_sync_payload.items():
                lines.append(f"{k}: {v}")
        else:
            lines.append("No wandb_sync.json found.")

        if snap.technical_failures_payload:
            lines.append("")
            lines.append("=== TECHNICAL FAILURES (technical_failures.jsonl) ===")
            for tf in snap.technical_failures_payload:
                lines.append(str(tf))

        self.text_wandb.setText("\n".join(lines))
