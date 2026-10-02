"""
Plan and Preflight View Widget (Gate 7.5B).

Displays:
- Resolved plan details (mode, canonical_run, protocol scope, agent details, resolved plan SHA-256).
- Case count and selected/boundary case summaries.
- Preflight table: Category, Check ID, Status (PASS, WARNING, FAIL), Message.
- Overall preflight summary verdict banner.
"""

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class PlanAndPreflightWidget(QWidget):
    """View showing resolved experiment plan details and preflight verification table."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # Top section: Plan View
        plan_group = QGroupBox("Resolved Experiment Plan (Read-Only Preview)")
        plan_layout = QVBoxLayout(plan_group)
        self.text_plan_details = QTextEdit()
        self.text_plan_details.setReadOnly(True)
        self.text_plan_details.setPlaceholderText("No plan resolved yet. Click 'Resolve & Preflight' to generate plan.")
        plan_layout.addWidget(self.text_plan_details)
        splitter.addWidget(plan_group)

        # Bottom section: Preflight Report
        preflight_group = QGroupBox("Preflight Safety Verification")
        preflight_layout = QVBoxLayout(preflight_group)

        # Summary banner
        self.lbl_verdict = QLabel("Verdict: NOT EVALUATED")
        self.lbl_verdict.setStyleSheet("font-weight: bold; font-size: 14px; padding: 4px;")
        preflight_layout.addWidget(self.lbl_verdict)

        # Checks table
        self.table_checks = QTableWidget(0, 4)
        self.table_checks.setHorizontalHeaderLabels(["Status", "Category", "Check ID", "Message"])
        self.table_checks.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_checks.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_checks.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table_checks.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        preflight_layout.addWidget(self.table_checks)

        splitter.addWidget(preflight_group)
        main_layout.addWidget(splitter)

    def clear(self) -> None:
        self.text_plan_details.clear()
        self.lbl_verdict.setText("Verdict: INVALIDATED")
        self.lbl_verdict.setStyleSheet("font-weight: bold; font-size: 14px; color: gray; padding: 4px;")
        self.table_checks.setRowCount(0)

    def display_plan(self, plan_data: Dict[str, Any]) -> None:
        """Renders plan dictionary into informative text display."""
        plan = plan_data.get("plan", plan_data)
        mode = plan.get("launcher_mode")
        run_kind = plan.get("run_kind")
        canonical = plan.get("canonical_run")
        scope = plan.get("protocol_scope")
        plan_hash = plan.get("resolved_plan_sha256")
        agent_reg = plan.get("agent_registration", {})
        agent_desc = plan.get("agent_descriptor", {})
        cases = plan.get("resolved_cases", [])
        warnings = plan.get("warnings", [])

        lines = [
            f"=== EXPERIMENT EXECUTION PLAN ===",
            f"Launcher Mode:         {mode}",
            f"Run Kind:              {run_kind}",
            f"Canonical Run:         {canonical}",
            f"Protocol Scope:        {scope}",
            f"Resolved Plan SHA-256: {plan_hash}",
            f"",
            f"--- Agent Information ---",
            f"Agent ID:              {agent_reg.get('agent_id')}",
            f"Agent Version:         {agent_reg.get('agent_version')}",
            f"Stage Label:           {agent_reg.get('stage_label')}",
            f"Method Family:         {agent_reg.get('method_family')}",
            f"Purpose:               {agent_reg.get('purpose')}",
            f"Benchmark Eligible:    {agent_reg.get('benchmark_eligible')}",
            f"Input Profile:         {agent_reg.get('input_profile_id')}",
            f"Action Adapter:        {agent_reg.get('action_adapter_id')}",
            f"Inference:             {agent_reg.get('inference_stochasticity')}",
            f"Stateful:              {agent_reg.get('stateful_within_episode')}",
            f"Agent Seed:            {plan.get('agent_seed')}",
            f"",
            f"--- Execution Settings ---",
            f"Render Mode:           {plan.get('render_mode')}",
            f"W&B Mode:              {plan.get('wandb_mode')}",
            f"Control Frequency:     {plan.get('control_frequency_hz')} Hz (dt={plan.get('control_dt_s')} s)",
            f"Resolved Case Count:   {len(cases)}",
        ]

        if cases:
            if len(cases) == 1:
                c = cases[0]
                lines.append(f"\n--- Selected Case: {c.get('case_id')} ---")
                lines.append(f"  Split: {c.get('split')} | Tier: {c.get('tier')} | Seq: {c.get('sequence')}")
                lines.append(f"  Geom Seed: {c.get('geometry_generation_seed')} | Env Seed: {c.get('environment_seed')} | Horizon: {c.get('horizon_steps')}")
            else:
                first = cases[0]
                last = cases[-1]
                lines.append(f"\n--- Suite Case Boundaries ({len(cases)} total cases) ---")
                lines.append(f"  First: {first.get('case_id')} (Split: {first.get('split')}, Tier: {first.get('tier')}, Seq: {first.get('sequence')})")
                lines.append(f"  Last:  {last.get('case_id')} (Split: {last.get('split')}, Tier: {last.get('tier')}, Seq: {last.get('sequence')})")

        if warnings:
            lines.append("\n--- Plan Warnings ---")
            for w in warnings:
                lines.append(f"  * {w}")

        self.text_plan_details.setText("\n".join(lines))

    def display_preflight(self, report_payload: Dict[str, Any]) -> None:
        """Populates preflight report table and verdict banner."""
        report = report_payload.get("report", report_payload)
        can_execute = report.get("can_execute", False)
        summary = report.get("summary_verdict", "UNKNOWN")
        fails = report.get("fail_count", 0)
        warns = report.get("warning_count", 0)
        passes = report.get("pass_count", 0)

        # Banner style
        if can_execute:
            if warns > 0:
                self.lbl_verdict.setText(f"Verdict: {summary} (Pass: {passes}, Warnings: {warns}, Fail: 0) — EXECUTION PERMITTED")
                self.lbl_verdict.setStyleSheet("font-weight: bold; font-size: 13px; color: #b8860b; background-color: #fff8dc; padding: 6px; border: 1px solid #ffd700;")
            else:
                self.lbl_verdict.setText(f"Verdict: {summary} (Pass: {passes}, Warnings: 0, Fail: 0) — READY TO EXECUTE")
                self.lbl_verdict.setStyleSheet("font-weight: bold; font-size: 13px; color: #006400; background-color: #f0fff0; padding: 6px; border: 1px solid #32cd32;")
        else:
            self.lbl_verdict.setText(f"Verdict: {summary} (Pass: {passes}, Warnings: {warns}, Fail: {fails}) — BLOCKED")
            self.lbl_verdict.setStyleSheet("font-weight: bold; font-size: 13px; color: #8b0000; background-color: #ffe4e1; padding: 6px; border: 1px solid #ff4500;")

        checks = report.get("checks", [])
        self.table_checks.setRowCount(len(checks))

        for row, check in enumerate(checks):
            status = check.get("status", "")
            category = check.get("category", "")
            check_id = check.get("check_id", "")
            msg = check.get("message", "")

            item_status = QTableWidgetItem(status)
            item_cat = QTableWidgetItem(category)
            item_id = QTableWidgetItem(check_id)
            item_msg = QTableWidgetItem(msg)

            if status == "PASS":
                item_status.setForeground(QColor("#006400"))
            elif status == "WARNING":
                item_status.setForeground(QColor("#b8860b"))
            elif status == "FAIL":
                item_status.setForeground(QColor("#8b0000"))

            self.table_checks.setItem(row, 0, item_status)
            self.table_checks.setItem(row, 1, item_cat)
            self.table_checks.setItem(row, 2, item_id)
            self.table_checks.setItem(row, 3, item_msg)
