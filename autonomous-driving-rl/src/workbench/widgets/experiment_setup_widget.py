"""
Experiment Setup Widget (Gate 7.5B).

Collects LaunchRequestV1 parameters:
- Mode: SANDBOX, VALIDATION, TEST, AUDIT.
- Agent: registered agents from AgentRegistryV1.
- Tier, Sequence, Geometry Seed, Environment Seed (disabled for VALIDATION/TEST).
- Agent Seed (None or int).
- Render mode (OFF, NATIVE; forced OFF for VALIDATION/TEST).
- W&B mode (AUTO, DISABLED, OFFLINE, ONLINE).
- Runs root, Custom run ID.
- Action buttons: "Resolve & Preflight" and "Run Experiment".
- Emits requestChanged signal whenever any field changes to invalidate previous preflight.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.launcher.models import LaunchRequestV1, LauncherMode
from src.platform import WandbMode
from src.workbench.core_adapter import CoreAdapter


class ExperimentSetupWidget(QWidget):
    """Experiment setup and control panel."""

    requestChanged = Signal()
    resolveRequested = Signal(object)  # LaunchRequestV1
    runRequested = Signal(object)      # LaunchRequestV1
    terminateRequested = Signal()

    def __init__(self, core_adapter: CoreAdapter, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._core_adapter = core_adapter
        self._can_run = False
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # Form group
        form_group = QGroupBox("Experiment Request Parameters")
        form_layout = QFormLayout(form_group)

        # Mode
        self.combo_mode = QComboBox()
        for mode in self._core_adapter.get_supported_modes():
            self.combo_mode.addItem(mode)
        form_layout.addRow(QLabel("Launcher Mode:"), self.combo_mode)

        # Agent
        self.combo_agent = QComboBox()
        for reg in self._core_adapter.get_registered_agents():
            label = f"{reg.agent_id} ({reg.method_family}, benchmark={reg.benchmark_eligible})"
            self.combo_agent.addItem(label, userData=reg.agent_id)
        form_layout.addRow(QLabel("Agent:"), self.combo_agent)

        # Tier
        self.combo_tier = QComboBox()
        self.combo_tier.addItem("Auto / First", userData=None)
        self.combo_tier.addItem("Easy", userData="Easy")
        self.combo_tier.addItem("Medium", userData="Medium")
        self.combo_tier.addItem("Hard", userData="Hard")
        self.combo_tier.addItem("Extreme", userData="Extreme")
        form_layout.addRow(QLabel("Tier:"), self.combo_tier)

        # Sequence
        self.edit_sequence = QLineEdit()
        self.edit_sequence.setPlaceholderText("Optional sequence (e.g. SCCS)")
        form_layout.addRow(QLabel("Sequence:"), self.edit_sequence)

        # Geometry Seed
        self.edit_geom_seed = QLineEdit()
        self.edit_geom_seed.setPlaceholderText("Optional integer (e.g. 0)")
        form_layout.addRow(QLabel("Geometry Seed:"), self.edit_geom_seed)

        # Environment Seed
        self.edit_env_seed = QLineEdit()
        self.edit_env_seed.setPlaceholderText("Optional integer (e.g. 0)")
        form_layout.addRow(QLabel("Environment Seed:"), self.edit_env_seed)

        # Agent Seed
        self.edit_agent_seed = QLineEdit()
        self.edit_agent_seed.setPlaceholderText("Optional integer (101, 202, 303 for stochastic benchmark)")
        form_layout.addRow(QLabel("Agent Seed:"), self.edit_agent_seed)

        # Render Mode
        self.combo_render = QComboBox()
        self.combo_render.addItem("OFF", userData="OFF")
        self.combo_render.addItem("NATIVE", userData="NATIVE")
        form_layout.addRow(QLabel("Render Mode:"), self.combo_render)

        # W&B Mode
        self.combo_wandb = QComboBox()
        self.combo_wandb.addItem("AUTO (Mode Default)", userData=None)
        self.combo_wandb.addItem("DISABLED", userData=WandbMode.DISABLED)
        self.combo_wandb.addItem("OFFLINE", userData=WandbMode.OFFLINE)
        self.combo_wandb.addItem("ONLINE", userData=WandbMode.ONLINE)
        form_layout.addRow(QLabel("W&B Mode:"), self.combo_wandb)

        # Custom Runs Root
        self.edit_runs_root = QLineEdit()
        self.edit_runs_root.setPlaceholderText("Optional custom runs root path")
        form_layout.addRow(QLabel("Runs Root:"), self.edit_runs_root)

        # Custom Run ID
        self.edit_custom_run_id = QLineEdit()
        self.edit_custom_run_id.setPlaceholderText("Optional custom run ID (alphanumeric, -, _, .)")
        form_layout.addRow(QLabel("Custom Run ID:"), self.edit_custom_run_id)

        main_layout.addWidget(form_group)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_resolve = QPushButton("Resolve & Preflight")
        self.btn_run = QPushButton("Run Experiment")
        self.btn_run.setEnabled(False)
        self.btn_terminate = QPushButton("FORCE TERMINATE")
        self.btn_terminate.setEnabled(False)
        self.btn_terminate.setStyleSheet("color: red; font-weight: bold;")

        btn_layout.addWidget(self.btn_resolve)
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_terminate)
        main_layout.addLayout(btn_layout)

        # Signal connections
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.combo_agent.currentIndexChanged.connect(self._on_field_changed)
        self.combo_tier.currentIndexChanged.connect(self._on_field_changed)
        self.edit_sequence.textChanged.connect(self._on_field_changed)
        self.edit_geom_seed.textChanged.connect(self._on_field_changed)
        self.edit_env_seed.textChanged.connect(self._on_field_changed)
        self.edit_agent_seed.textChanged.connect(self._on_field_changed)
        self.combo_render.currentIndexChanged.connect(self._on_field_changed)
        self.combo_wandb.currentIndexChanged.connect(self._on_field_changed)
        self.edit_runs_root.textChanged.connect(self._on_field_changed)
        self.edit_custom_run_id.textChanged.connect(self._on_field_changed)

        self.btn_resolve.clicked.connect(self._on_resolve_clicked)
        self.btn_run.clicked.connect(self._on_run_clicked)
        self.btn_terminate.clicked.connect(self.terminateRequested.emit)

        # Initial mode state adjustment
        self._on_mode_changed()

    def _on_mode_changed(self) -> None:
        """Mode-aware UX adjustment (Specification Section 15)."""
        mode = self.combo_mode.currentText()
        is_benchmark = mode in ("VALIDATION", "TEST")
        # In benchmark modes, case selection is frozen to full manifest
        self.combo_tier.setEnabled(not is_benchmark)
        self.edit_sequence.setEnabled(not is_benchmark)
        self.edit_geom_seed.setEnabled(not is_benchmark)
        self.edit_env_seed.setEnabled(not is_benchmark)

        if is_benchmark:
            # Force render OFF for benchmark UX
            idx = self.combo_render.findText("OFF")
            if idx >= 0:
                self.combo_render.setCurrentIndex(idx)
            self.combo_render.setEnabled(False)
        else:
            self.combo_render.setEnabled(True)

        self._on_field_changed()

    def _on_field_changed(self) -> None:
        """Any field edit invalidates previous plan/preflight and disables Run."""
        self._can_run = False
        self.btn_run.setEnabled(False)
        self.requestChanged.emit()

    def set_preflight_verdict(self, can_execute: bool) -> None:
        """Enables Run button only if preflight passed and no worker is running."""
        self._can_run = can_execute
        self.btn_run.setEnabled(can_execute)

    def set_worker_running(self, running: bool) -> None:
        """Updates controls when worker starts or finishes."""
        self.btn_resolve.setEnabled(not running)
        self.btn_run.setEnabled((not running) and self._can_run)
        self.btn_terminate.setEnabled(running)

    def build_launch_request(self) -> LaunchRequestV1:
        """Constructs LaunchRequestV1 from current UI controls."""
        mode = LauncherMode(self.combo_mode.currentText())
        agent_id = str(self.combo_agent.currentData())

        tier = self.combo_tier.currentData()
        seq_text = self.edit_sequence.text().strip()
        sequence = seq_text if seq_text else None

        geom_text = self.edit_geom_seed.text().strip()
        geom_seed = int(geom_text) if geom_text else None

        env_text = self.edit_env_seed.text().strip()
        env_seed = int(env_text) if env_text else None

        agent_text = self.edit_agent_seed.text().strip()
        agent_seed = int(agent_text) if agent_text else None

        render_mode = str(self.combo_render.currentData())
        wandb_mode = self.combo_wandb.currentData()

        runs_root_text = self.edit_runs_root.text().strip()
        runs_root = Path(runs_root_text) if runs_root_text else None

        custom_id_text = self.edit_custom_run_id.text().strip()
        custom_run_id = custom_id_text if custom_id_text else None

        return LaunchRequestV1(
            mode=mode,
            agent_id=agent_id,
            tier=tier,
            sequence=sequence,
            geometry_generation_seed=geom_seed,
            environment_seed=env_seed,
            agent_seed=agent_seed,
            render_mode=render_mode,
            wandb_mode=wandb_mode,
            runs_root=runs_root,
            custom_run_id=custom_run_id,
        )

    def _on_resolve_clicked(self) -> None:
        try:
            req = self.build_launch_request()
            self.resolveRequested.emit(req)
        except Exception as e:
            # UI display or error handling
            pass

    def _on_run_clicked(self) -> None:
        try:
            req = self.build_launch_request()
            self.runRequested.emit(req)
        except Exception as e:
            pass
