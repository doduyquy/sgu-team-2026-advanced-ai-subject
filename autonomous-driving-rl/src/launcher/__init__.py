"""
Research Platform V1 Launcher Core (Gate 7.5A).

Public API exports for experiment orchestration, resolution, safety preflight,
and generic simulation execution across Stages 0 through 7.
"""

from src.launcher.cases import (
    load_locked_geometry_block_sequence,
    resolve_audit_cases,
    resolve_sandbox_case,
    resolve_test_cases,
    resolve_validation_cases,
)
from src.launcher.contracts import (
    build_launcher_contract_core,
    compute_launcher_contract_sha256,
    compute_platform_execution_contract_sha256,
)
from src.launcher.events import (
    EventCallback,
    LauncherEventType,
    LauncherEventV1,
)
from src.launcher.executor import (
    ExecutionReportV1,
    ExperimentExecutor,
)
from src.launcher.models import (
    AgentRegistrationV1,
    LaunchRequestV1,
    LauncherMode,
    ResolvedCaseV1,
    ResolvedExperimentPlanV1,
    compute_resolved_plan_sha256,
    describe_plan,
    mode_to_run_kind,
)
from src.launcher.preflight import (
    PreflightCheckV1,
    PreflightReportV1,
    run_preflight,
)
from src.launcher.registry import (
    AgentRegistryV1,
    build_default_agent_registry,
)
from src.launcher.resolver import (
    resolve_experiment_plan,
)

__all__ = [
    "AgentRegistrationV1",
    "AgentRegistryV1",
    "EventCallback",
    "ExecutionReportV1",
    "ExperimentExecutor",
    "LaunchRequestV1",
    "LauncherEventType",
    "LauncherEventV1",
    "LauncherMode",
    "PreflightCheckV1",
    "PreflightReportV1",
    "ResolvedCaseV1",
    "ResolvedExperimentPlanV1",
    "build_default_agent_registry",
    "build_launcher_contract_core",
    "compute_launcher_contract_sha256",
    "compute_platform_execution_contract_sha256",
    "compute_resolved_plan_sha256",
    "describe_plan",
    "load_locked_geometry_block_sequence",
    "mode_to_run_kind",
    "resolve_audit_cases",
    "resolve_experiment_plan",
    "resolve_sandbox_case",
    "resolve_test_cases",
    "resolve_validation_cases",
    "run_preflight",
]
