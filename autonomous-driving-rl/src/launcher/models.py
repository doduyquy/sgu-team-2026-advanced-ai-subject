"""
Data Models and Plan Specifications for Research Platform V1 Launcher (Gate 7.5A).

This module defines the declarative layer of the launcher:
- LauncherMode: Orchestration modes (SANDBOX, VALIDATION, TEST, AUDIT).
- LaunchRequestV1: Declarative user intent ("WHAT THE USER ASKED FOR").
- AgentRegistrationV1: Standardized machine-readable agent capability descriptor.
- ResolvedCaseV1: Immutable case specification directly compatible with Gate-5 manifests.
- ResolvedExperimentPlanV1: Concrete execution plan ("WHAT THE PLATFORM WILL EXECUTE").
- Deterministic plan hashing and learning-friendly descriptive output.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.platform import (
    RunKind,
    WandbMode,
    canonical_json_sha256,
)


class LauncherMode(str, Enum):
    """Orchestration execution modes provided by the Research Launcher."""
    SANDBOX = "SANDBOX"
    VALIDATION = "VALIDATION"
    TEST = "TEST"
    AUDIT = "AUDIT"


def mode_to_run_kind(mode: LauncherMode) -> RunKind:
    """Maps launcher-level orchestration mode to lower-level Gate-7 RunKind."""
    if mode == LauncherMode.SANDBOX:
        return RunKind.AUDIT  # Sandbox resolves as non-canonical audit/development run
    elif mode == LauncherMode.AUDIT:
        return RunKind.AUDIT
    elif mode == LauncherMode.VALIDATION:
        return RunKind.VALIDATION_EVALUATION
    elif mode == LauncherMode.TEST:
        return RunKind.TEST_EVALUATION
    else:
        raise ValueError(f"Unknown LauncherMode: {mode}")


@dataclass(frozen=True)
class AgentRegistrationV1:
    """
    Standardized, serializable metadata describing an agent registered in AgentRegistryV1.
    Keeps semantic capabilities segregated from runtime factory instantiation.
    """
    agent_id: str
    agent_version: str = "1.0.0"
    stage_label: Optional[str] = None  # None for test fixtures, "STAGE_0", "STAGE_1", etc.
    method_family: str = "FIXTURE"    # "FIXTURE", "RANDOM", "HEURISTIC", "PLANNER", "RL", etc.
    purpose: str = "AUDIT_FIXTURE"     # "AUDIT_FIXTURE", "RESEARCH_BASELINE", "CANDIDATE"
    implementation_ref: str = ""       # e.g. class name or callable reference
    input_profile_id: str = "STATE_DECISION_V1"
    action_adapter_id: str = "continuous_box2_v1"
    inference_stochasticity: str = "deterministic"  # "deterministic" or "stochastic"
    stateful_within_episode: bool = False
    benchmark_eligible: bool = False   # True only for verified scientific benchmark agents
    sandbox_eligible: bool = True      # Usable in Sandbox mode
    audit_eligible: bool = True        # Usable in platform audit mode
    requires_checkpoint: bool = False
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LaunchRequestV1:
    """
    Declarative specification of user execution intent ("WHAT THE USER ASKED FOR").
    Decoupled from simulator internals and case manifest layout.
    """
    mode: LauncherMode
    agent_id: str
    tier: Optional[str] = None
    sequence: Optional[str] = None
    geometry_generation_seed: Optional[int] = None
    environment_seed: Optional[int] = None
    agent_seed: Optional[int] = None
    render_mode: str = "OFF"           # "OFF" or "NATIVE"
    wandb_mode: WandbMode = WandbMode.DISABLED
    runs_root: Optional[Path] = None
    custom_run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["mode"] = self.mode.value
        d["wandb_mode"] = self.wandb_mode.value
        if self.runs_root is not None:
            d["runs_root"] = str(self.runs_root)
        return d


@dataclass(frozen=True)
class ResolvedCaseV1:
    """
    Single evaluation or sandbox case specification resolved from Gate-5 manifests.
    Guarantees exact parity with Gate-5 EvaluationCase.
    """
    case_id: str
    case_index: int
    protocol_order_index: int
    split: str
    tier: str
    sequence: str
    geometry_generation_seed: int
    geometry_sha256: str
    environment_seed: int
    traffic_density: float
    horizon_steps: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedExperimentPlanV1:
    """
    Immutable, authoritative execution specification ("WHAT THE PLATFORM WILL EXECUTE").
    Provides full reproducibility and enables zero-side-effect plan preview.
    """
    launcher_mode: LauncherMode
    run_kind: RunKind
    canonical_run: bool
    protocol_scope: str                # e.g. "SANDBOX_SINGLE_EPISODE", "VALIDATION_SUITE_96", "TEST_SUITE_60"
    agent_registration: AgentRegistrationV1
    agent_descriptor: Dict[str, Any]
    resolved_cases: List[ResolvedCaseV1]
    render_mode: str
    wandb_mode: WandbMode
    agent_seed: Optional[int]
    control_frequency_hz: float = 10.0
    control_dt_s: float = 0.1
    benchmark_contract_sha256: str = ""
    agent_contract_sha256: str = ""
    platform_runtime_contract_sha256: str = ""
    logging_contract_sha256: str = ""
    platform_observability_contract_sha256: str = ""
    launcher_contract_sha256: str = ""
    platform_execution_contract_sha256: str = ""
    resolved_plan_sha256: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["launcher_mode"] = self.launcher_mode.value
        d["run_kind"] = self.run_kind.value
        d["wandb_mode"] = self.wandb_mode.value
        d["agent_registration"] = self.agent_registration.to_dict()
        d["resolved_cases"] = [c.to_dict() for c in self.resolved_cases]
        return d


def compute_resolved_plan_sha256(plan_dict: Dict[str, Any]) -> str:
    """
    Computes deterministic SHA-256 fingerprint over scientifically relevant plan semantics.
    Excludes human warnings, timestamps, paths, and cosmetic layout.
    """
    hashable_cases = []
    for c in plan_dict.get("resolved_cases", []):
        hashable_cases.append({
            "case_id": c["case_id"],
            "protocol_order_index": c["protocol_order_index"],
            "split": c["split"],
            "tier": c["tier"],
            "sequence": c["sequence"],
            "geometry_generation_seed": c["geometry_generation_seed"],
            "geometry_sha256": c["geometry_sha256"],
            "environment_seed": c["environment_seed"],
            "traffic_density": c["traffic_density"],
            "horizon_steps": c["horizon_steps"]
        })

    reg = plan_dict.get("agent_registration", {})
    hashable_agent = {
        "agent_id": reg.get("agent_id"),
        "agent_version": reg.get("agent_version"),
        "input_profile_id": reg.get("input_profile_id"),
        "action_adapter_id": reg.get("action_adapter_id"),
        "inference_stochasticity": reg.get("inference_stochasticity"),
        "stateful_within_episode": reg.get("stateful_within_episode"),
        "benchmark_eligible": reg.get("benchmark_eligible")
    }

    hashable_core = {
        "launcher_mode": plan_dict.get("launcher_mode"),
        "run_kind": plan_dict.get("run_kind"),
        "canonical_run": plan_dict.get("canonical_run"),
        "protocol_scope": plan_dict.get("protocol_scope"),
        "agent": hashable_agent,
        "resolved_cases": hashable_cases,
        "render_mode": plan_dict.get("render_mode"),
        "wandb_mode": plan_dict.get("wandb_mode"),
        "agent_seed": plan_dict.get("agent_seed"),
        "control_frequency_hz": plan_dict.get("control_frequency_hz"),
        "control_dt_s": plan_dict.get("control_dt_s"),
        "benchmark_contract_sha256": plan_dict.get("benchmark_contract_sha256"),
        "agent_contract_sha256": plan_dict.get("agent_contract_sha256"),
        "platform_runtime_contract_sha256": plan_dict.get("platform_runtime_contract_sha256"),
        "logging_contract_sha256": plan_dict.get("logging_contract_sha256"),
        "platform_observability_contract_sha256": plan_dict.get("platform_observability_contract_sha256"),
        "launcher_contract_sha256": plan_dict.get("launcher_contract_sha256"),
        "platform_execution_contract_sha256": plan_dict.get("platform_execution_contract_sha256")
    }

    return canonical_json_sha256(hashable_core)


def describe_plan(plan: ResolvedExperimentPlanV1) -> str:
    """
    Generates a clear, learning-friendly, text description of a resolved plan.
    Enables students and researchers to inspect exactly what will execute.
    Deterministic output without external LLM dependencies.
    """
    lines = [
        "============================================================",
        f"EXPERIMENT PLAN PREVIEW: {plan.launcher_mode.value}",
        "============================================================",
        f"  Launcher Mode:       {plan.launcher_mode.value}",
        f"  Run Kind:            {plan.run_kind.value}",
        f"  Canonical Run:       {plan.canonical_run}",
        f"  Protocol Scope:      {plan.protocol_scope}",
        f"  Plan SHA-256:        {plan.resolved_plan_sha256}",
        "",
        "--- Agent Specification ---",
        f"  Agent ID:            {plan.agent_registration.agent_id} (v{plan.agent_registration.agent_version})",
        f"  Method Family:       {plan.agent_registration.method_family}",
        f"  Purpose:             {plan.agent_registration.purpose}",
        f"  Stage Label:         {plan.agent_registration.stage_label or 'None (Fixture)'}",
        f"  Benchmark Eligible:  {plan.agent_registration.benchmark_eligible}",
        f"  Input Profile:       {plan.agent_registration.input_profile_id}",
        f"  Action Adapter:      {plan.agent_registration.action_adapter_id}",
        f"  Stochasticity:       {plan.agent_registration.inference_stochasticity}",
        f"  Agent Seed:          {plan.agent_seed if plan.agent_seed is not None else 'None (Deterministic / Unset)'}",
        "",
        "--- Execution Environment & Observability ---",
        f"  Control Frequency:   {plan.control_frequency_hz} Hz (dt = {plan.control_dt_s}s)",
        f"  Render Mode:         {plan.render_mode}",
        f"  W&B Tracking:        {plan.wandb_mode.value}",
        f"  Total Cases:         {len(plan.resolved_cases)}",
        ""
    ]

    if plan.launcher_mode == LauncherMode.SANDBOX and plan.resolved_cases:
        c = plan.resolved_cases[0]
        lines.extend([
            "--- Sandbox Target Case ---",
            f"  Case ID:             {c.case_id}",
            f"  Split:               {c.split} (TRAIN only guaranteed)",
            f"  Tier:                {c.tier}",
            f"  Sequence:            {c.sequence}",
            f"  Geometry Seed:       {c.geometry_generation_seed}",
            f"  Environment Seed:    {c.environment_seed}",
            f"  Traffic Density:     {c.traffic_density}",
            f"  Route Horizon:       {c.horizon_steps} steps ({c.horizon_steps * plan.control_dt_s:.1f}s)",
            ""
        ])
    elif plan.resolved_cases:
        lines.extend([
            "--- Benchmark Case Suite Summary ---",
            f"  Split:               {plan.resolved_cases[0].split}",
            f"  First Case:          {plan.resolved_cases[0].case_id}",
            f"  Last Case:           {plan.resolved_cases[-1].case_id}",
            f"  Horizons Range:      {min(c.horizon_steps for c in plan.resolved_cases)} to {max(c.horizon_steps for c in plan.resolved_cases)} steps",
            ""
        ])

    if plan.warnings:
        lines.extend([
            "--- Warnings & Notes ---",
            *[f"  [!] {w}" for w in plan.warnings],
            ""
        ])

    lines.append("============================================================")
    return "\n".join(lines)
