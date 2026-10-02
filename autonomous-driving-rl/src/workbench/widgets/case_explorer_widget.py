"""
Cases Explorer Widget (Gate 7.5B).

Read-only browsing for frozen Gate-5 manifests:
- TRAIN, VALIDATION, TEST splits.
- Filter by tier and sequence.
- Displays case properties: Case ID, Split, Tier, Sequence, Geom Seed, Env Seed, Traffic, Horizon.
- SCIENTIFIC CONSTRAINT (Specification Section 20):
  The Case Explorer must NEVER provide a feature to run a partial VALIDATION or TEST selection.
  No 'Run Selected TEST Cases' button. No benchmark subset execution.
"""

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.workbench.core_adapter import CoreAdapter


class CaseExplorerWidget(QWidget):
    """Read-only manifest case browser with filtering."""

    def __init__(self, core_adapter: CoreAdapter, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._core_adapter = core_adapter
        self._all_cases: List[Dict[str, Any]] = []
        self._init_ui()
        self._load_cases()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # Filters
        filter_group = QGroupBox("Manifest & Filters (Read-Only Browsing)")
        filter_layout = QHBoxLayout(filter_group)

        filter_layout.addWidget(QLabel("Manifest Split:"))
        self.combo_split = QComboBox()
        self.combo_split.addItems(["TRAIN", "VALIDATION", "TEST"])
        filter_layout.addWidget(self.combo_split)

        filter_layout.addWidget(QLabel("Tier:"))
        self.combo_tier = QComboBox()
        self.combo_tier.addItems(["ALL", "Easy", "Medium", "Hard", "Extreme"])
        filter_layout.addWidget(self.combo_tier)

        filter_layout.addWidget(QLabel("Sequence Search:"))
        self.edit_seq_filter = QLineEdit()
        self.edit_seq_filter.setPlaceholderText("Filter sequence (e.g. SCCS)")
        filter_layout.addWidget(self.edit_seq_filter)

        main_layout.addWidget(filter_group)

        # Status / error banner
        self.lbl_status = QLabel("")
        self.lbl_status.setVisible(False)
        main_layout.addWidget(self.lbl_status)

        # Scientific constraint note
        lbl_note = QLabel("Scientific Policy: Manifest browsing is strictly read-only. Partial benchmark execution is prohibited.")
        lbl_note.setStyleSheet("color: gray; font-style: italic;")
        main_layout.addWidget(lbl_note)

        # Cases Table
        self.table_cases = QTableWidget(0, 8)
        self.table_cases.setHorizontalHeaderLabels([
            "Case ID", "Split", "Tier", "Sequence", "Geom Seed", "Env Seed", "Traffic", "Horizon"
        ])
        for c in range(7):
            self.table_cases.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table_cases.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        self.table_cases.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_cases.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        main_layout.addWidget(self.table_cases)

        # Signal connections
        self.combo_split.currentIndexChanged.connect(self._load_cases)
        self.combo_tier.currentIndexChanged.connect(self._apply_filter)
        self.edit_seq_filter.textChanged.connect(self._apply_filter)

    def _load_cases(self) -> None:
        split = self.combo_split.currentText()
        try:
            self._all_cases = self._core_adapter.load_case_manifest(split)
            self.lbl_status.setVisible(False)
            self.lbl_status.setText("")
        except Exception as e:
            self._all_cases = []
            self.lbl_status.setVisible(True)
            self.lbl_status.setStyleSheet("color: red; font-weight: bold; background: #ffe6e6; padding: 4px; border: 1px solid red;")
            self.lbl_status.setText(f"ERROR: Failed to load manifest for split '{split}': {e}")
        self._apply_filter()

    def _apply_filter(self) -> None:
        tier_filter = self.combo_tier.currentText()
        seq_filter = self.edit_seq_filter.text().strip().upper()

        filtered = []
        for case in self._all_cases:
            c_tier = str(case.get("tier", ""))
            c_seq = str(case.get("sequence", "")).upper()

            if tier_filter != "ALL" and c_tier != tier_filter:
                continue
            if seq_filter and seq_filter not in c_seq:
                continue
            filtered.append(case)

        self.table_cases.setRowCount(len(filtered))
        for row, c in enumerate(filtered):
            case_id = str(c.get("case_id", f"{c.get('split')}/{c.get('tier')}/{c.get('sequence')}"))
            split = str(c.get("split", ""))
            tier = str(c.get("tier", ""))
            seq = str(c.get("sequence", ""))
            geom = str(c.get("geometry_generation_seed", ""))
            env = str(c.get("environment_seed", ""))
            traffic = str(c.get("traffic_density", ""))
            horizon = str(c.get("horizon_steps", ""))

            self.table_cases.setItem(row, 0, QTableWidgetItem(case_id))
            self.table_cases.setItem(row, 1, QTableWidgetItem(split))
            self.table_cases.setItem(row, 2, QTableWidgetItem(tier))
            self.table_cases.setItem(row, 3, QTableWidgetItem(seq))
            self.table_cases.setItem(row, 4, QTableWidgetItem(geom))
            self.table_cases.setItem(row, 5, QTableWidgetItem(env))
            self.table_cases.setItem(row, 6, QTableWidgetItem(traffic))
            self.table_cases.setItem(row, 7, QTableWidgetItem(horizon))
