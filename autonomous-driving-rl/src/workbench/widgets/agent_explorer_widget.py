"""
Agents Explorer Widget (Gate 7.5B).

Displays read-only registration details for all registered agents in AgentRegistryV1:
- agent_id, agent_version, stage_label, method_family, purpose
- input_profile_id, action_adapter_id, inference_stochasticity, stateful_within_episode
- benchmark_eligible, sandbox_eligible, audit_eligible, requires_checkpoint
- implementation_ref, description
"""

from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGroupBox,
    QHeaderView,
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.launcher.models import AgentRegistrationV1
from src.workbench.core_adapter import CoreAdapter


class AgentExplorerWidget(QWidget):
    """Read-only explorer for canonical and development agent registrations."""

    def __init__(self, core_adapter: CoreAdapter, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._core_adapter = core_adapter
        self._init_ui()
        self._populate_agents()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # Table of agents
        table_group = QGroupBox("Registered Agents (AgentRegistryV1)")
        table_layout = QVBoxLayout(table_group)

        self.table_agents = QTableWidget(0, 6)
        self.table_agents.setHorizontalHeaderLabels([
            "Agent ID", "Version", "Method Family", "Benchmark Eligible", "Sandbox Eligible", "Purpose"
        ])
        self.table_agents.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_agents.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_agents.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table_agents.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table_agents.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table_agents.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.table_agents.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_agents.itemSelectionChanged.connect(self._on_selection_changed)

        table_layout.addWidget(self.table_agents)
        splitter.addWidget(table_group)

        # Detail pane
        detail_group = QGroupBox("Agent Capability Specification")
        detail_layout = QVBoxLayout(detail_group)
        self.text_details = QTextEdit()
        self.text_details.setReadOnly(True)
        detail_layout.addWidget(self.text_details)
        splitter.addWidget(detail_group)

        main_layout.addWidget(splitter)

    def _populate_agents(self) -> None:
        agents = self._core_adapter.get_registered_agents()
        self.table_agents.setRowCount(len(agents))

        for row, reg in enumerate(agents):
            item_id = QTableWidgetItem(reg.agent_id)
            item_ver = QTableWidgetItem(reg.agent_version)
            item_fam = QTableWidgetItem(reg.method_family)
            item_bench = QTableWidgetItem(str(reg.benchmark_eligible))
            item_sand = QTableWidgetItem(str(reg.sandbox_eligible))
            item_purp = QTableWidgetItem(reg.purpose)

            # Highlight benchmark eligibility
            if reg.benchmark_eligible:
                item_bench.setForeground(QColor("#006400"))
            else:
                item_bench.setForeground(QColor("#8b0000"))

            self.table_agents.setItem(row, 0, item_id)
            self.table_agents.setItem(row, 1, item_ver)
            self.table_agents.setItem(row, 2, item_fam)
            self.table_agents.setItem(row, 3, item_bench)
            self.table_agents.setItem(row, 4, item_sand)
            self.table_agents.setItem(row, 5, item_purp)

        if agents:
            self.table_agents.selectRow(0)

    def _on_selection_changed(self) -> None:
        selected_rows = self.table_agents.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        agent_id_item = self.table_agents.item(row, 0)
        if not agent_id_item:
            return
        agent_id = agent_id_item.text()
        reg = self._core_adapter.get_agent_registration(agent_id)
        if not reg:
            return

        lines = [
            f"=== AGENT REGISTRATION: {reg.agent_id} ===",
            f"Agent Version:           {reg.agent_version}",
            f"Stage Label:             {reg.stage_label}",
            f"Method Family:           {reg.method_family}",
            f"Purpose:                 {reg.purpose}",
            f"Implementation Ref:      {reg.implementation_ref}",
            f"Input Profile:           {reg.input_profile_id}",
            f"Action Adapter:          {reg.action_adapter_id}",
            f"Inference Stochasticity: {reg.inference_stochasticity}",
            f"Stateful in Episode:     {reg.stateful_within_episode}",
            f"Benchmark Eligible:      {reg.benchmark_eligible}",
            f"Sandbox Eligible:        {reg.sandbox_eligible}",
            f"Audit Eligible:          {reg.audit_eligible}",
            f"Requires Checkpoint:     {reg.requires_checkpoint}",
            f"",
            f"Description:",
            f"{reg.description}",
        ]
        self.text_details.setText("\n".join(lines))
