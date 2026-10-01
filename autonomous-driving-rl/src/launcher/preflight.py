"""
Preflight Validation Engine for Research Platform V1 Launcher (Gate 7.5A).

This module implements deterministic pre-execution scientific safety checks:
- Strict holdout and benchmark agent eligibility enforcement.
- MetaDrive exact pin verification (version 0.4.3, commit 85e5dad...).
- Clean working tree verification for benchmark evaluations (no silent dirty runs).
- Gate-5 seed taxonomy adherence (stochastic benchmark replicates in {101, 202, 303}).
- Headless execution enforcement for benchmark suites (TEST / VALIDATION render must be OFF).
- W&B credential presence verification when ONLINE mode is explicitly requested.
"""

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.launcher.cases import get_default_project_root
from src.launcher.models import (
    LauncherMode,
    ResolvedExperimentPlanV1,
)
from src.platform import (
    CERTIFIED_ACTION_ADAPTERS,
    PINNED_METADRIVE_COMMIT,
    PINNED_METADRIVE_VERSION,
    WandbMode,
    capture_environment_provenance,
    capture_git_provenance,
)


@dataclass(frozen=True)
class PreflightCheckV1:
    """Individual preflight check assertion record."""
    check_id: str
    status: str       # "PASS", "WARNING", "FAIL"
    message: str
    category: str     # "CONTRACT", "ENVIRONMENT", "AGENT", "CASE_PLAN", "SEED", "OBSERVABILITY", "SECURITY"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreflightReportV1:
    """Consolidated preflight validation report evaluated prior to plan execution."""
    can_execute: bool
    checks: List[PreflightCheckV1]
    fail_count: int
    warning_count: int
    pass_count: int
    summary_verdict: str  # "READY", "BLOCKED", "PROCEED_WITH_WARNINGS"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "can_execute": self.can_execute,
            "fail_count": self.fail_count,
            "warning_count": self.warning_count,
            "pass_count": self.pass_count,
            "summary_verdict": self.summary_verdict,
            "checks": [c.to_dict() for c in self.checks]
        }


def run_preflight(
    plan: ResolvedExperimentPlanV1,
    project_root: Optional[Path] = None,
    custom_git_provenance: Optional[Dict[str, Any]] = None,
    custom_environment_provenance: Optional[Dict[str, Any]] = None
) -> PreflightReportV1:
    """
    Executes a comprehensive, pure battery of preflight checks against ResolvedExperimentPlanV1.
    Returns PreflightReportV1. can_execute is True ONLY if fail_count == 0.
    """
    root = project_root or get_default_project_root()
    git_prov = custom_git_provenance or capture_git_provenance(root)
    env_prov = custom_environment_provenance or capture_environment_provenance(root)

    checks: List[PreflightCheckV1] = []

    # 1. MetaDrive exact pin verification
    actual_version = env_prov.get("metadrive_version", "unknown")
    actual_commit = env_prov.get("metadrive_commit", "unknown")
    if actual_version == PINNED_METADRIVE_VERSION and actual_commit == PINNED_METADRIVE_COMMIT:
        checks.append(PreflightCheckV1(
            check_id="metadrive_exact_pin",
            status="PASS",
            message=f"MetaDrive exact pin verified: version {actual_version}, commit {actual_commit[:8]}...",
            category="ENVIRONMENT"
        ))
    else:
        if plan.launcher_mode in (LauncherMode.VALIDATION, LauncherMode.TEST):
            checks.append(PreflightCheckV1(
                check_id="metadrive_exact_pin",
                status="FAIL",
                message=f"Benchmark execution requires exact MetaDrive {PINNED_METADRIVE_VERSION} ({PINNED_METADRIVE_COMMIT[:8]}), but found version '{actual_version}' commit '{actual_commit}'.",
                category="ENVIRONMENT"
            ))
        else:
            checks.append(PreflightCheckV1(
                check_id="metadrive_exact_pin",
                status="WARNING",
                message=f"MetaDrive pin mismatch: expected {PINNED_METADRIVE_VERSION} ({PINNED_METADRIVE_COMMIT[:8]}), found {actual_version} ({actual_commit}). Allowed for non-canonical {plan.launcher_mode.value}.",
                category="ENVIRONMENT"
            ))

    # 2. Git worktree cleanliness for benchmark runs
    is_dirty = bool(git_prov.get("git_worktree_dirty", False))
    if plan.launcher_mode in (LauncherMode.VALIDATION, LauncherMode.TEST):
        if is_dirty:
            checks.append(PreflightCheckV1(
                check_id="git_cleanliness",
                status="FAIL",
                message="Benchmark evaluation requires clean git working tree. Uncommitted changes detected!",
                category="SECURITY"
            ))
        else:
            checks.append(PreflightCheckV1(
                check_id="git_cleanliness",
                status="PASS",
                message=f"Git working tree clean at commit {git_prov.get('git_commit_sha', 'unknown')[:8]}.",
                category="SECURITY"
            ))
    else:
        if is_dirty:
            checks.append(PreflightCheckV1(
                check_id="git_cleanliness",
                status="WARNING",
                message=f"Git working tree is dirty; acceptable for exploratory {plan.launcher_mode.value} mode.",
                category="SECURITY"
            ))
        else:
            checks.append(PreflightCheckV1(
                check_id="git_cleanliness",
                status="PASS",
                message="Git working tree clean.",
                category="SECURITY"
            ))

    # 3. Agent registration and eligibility checks
    reg = plan.agent_registration
    if plan.launcher_mode == LauncherMode.SANDBOX and not reg.sandbox_eligible:
        checks.append(PreflightCheckV1(
            check_id="agent_mode_eligibility",
            status="FAIL",
            message=f"Agent '{reg.agent_id}' is not eligible for Sandbox mode.",
            category="AGENT"
        ))
    elif plan.launcher_mode == LauncherMode.AUDIT and not reg.audit_eligible:
        checks.append(PreflightCheckV1(
            check_id="agent_mode_eligibility",
            status="FAIL",
            message=f"Agent '{reg.agent_id}' is not eligible for Audit mode.",
            category="AGENT"
        ))
    elif plan.launcher_mode in (LauncherMode.VALIDATION, LauncherMode.TEST):
        if not reg.benchmark_eligible:
            checks.append(PreflightCheckV1(
                check_id="agent_benchmark_eligibility",
                status="FAIL",
                message=f"Agent '{reg.agent_id}' is flagged benchmark_eligible=False (e.g. audit fixture). Cannot execute benchmark evaluations on TEST or VALIDATION!",
                category="AGENT"
            ))
        else:
            checks.append(PreflightCheckV1(
                check_id="agent_benchmark_eligibility",
                status="PASS",
                message=f"Agent '{reg.agent_id}' is certified benchmark eligible.",
                category="AGENT"
            ))
    else:
        checks.append(PreflightCheckV1(
            check_id="agent_mode_eligibility",
            status="PASS",
            message=f"Agent '{reg.agent_id}' is eligible for mode {plan.launcher_mode.value}.",
            category="AGENT"
        ))

    # 4. Input profile and Action adapter support
    if reg.input_profile_id != "STATE_DECISION_V1":
        checks.append(PreflightCheckV1(
            check_id="input_profile_support",
            status="FAIL",
            message=f"Unsupported input profile '{reg.input_profile_id}'. Only 'STATE_DECISION_V1' is certified.",
            category="AGENT"
        ))
    else:
        checks.append(PreflightCheckV1(
            check_id="input_profile_support",
            status="PASS",
            message="Input profile 'STATE_DECISION_V1' verified.",
            category="AGENT"
        ))

    if reg.action_adapter_id not in CERTIFIED_ACTION_ADAPTERS:
        checks.append(PreflightCheckV1(
            check_id="action_adapter_support",
            status="FAIL",
            message=f"Action adapter '{reg.action_adapter_id}' is not in Gate-6 certified adapters: {list(CERTIFIED_ACTION_ADAPTERS.keys())}",
            category="AGENT"
        ))
    else:
        checks.append(PreflightCheckV1(
            check_id="action_adapter_support",
            status="PASS",
            message=f"Action adapter '{reg.action_adapter_id}' verified.",
            category="AGENT"
        ))

    # 5. Case plan integrity & split checks
    if plan.launcher_mode == LauncherMode.SANDBOX:
        if len(plan.resolved_cases) != 1:
            checks.append(PreflightCheckV1(
                check_id="sandbox_case_count",
                status="FAIL",
                message=f"Sandbox mode must resolve exactly 1 case, got {len(plan.resolved_cases)}.",
                category="CASE_PLAN"
            ))
        else:
            c = plan.resolved_cases[0]
            if c.split != "TRAIN":
                checks.append(PreflightCheckV1(
                    check_id="sandbox_split_isolation",
                    status="FAIL",
                    message=f"FATAL HOLDOUT VIOLATION: Sandbox resolved non-TRAIN split '{c.split}'!",
                    category="CASE_PLAN"
                ))
            else:
                checks.append(PreflightCheckV1(
                    check_id="sandbox_split_isolation",
                    status="PASS",
                    message="Sandbox case strictly confined to TRAIN split.",
                    category="CASE_PLAN"
                ))

    elif plan.launcher_mode == LauncherMode.VALIDATION:
        if len(plan.resolved_cases) != 96:
            checks.append(PreflightCheckV1(
                check_id="validation_case_count",
                status="FAIL",
                message=f"Validation suite must contain exactly 96 cases, got {len(plan.resolved_cases)}.",
                category="CASE_PLAN"
            ))
        else:
            all_val = all(c.split == "VALIDATION" for c in plan.resolved_cases)
            if not all_val:
                checks.append(PreflightCheckV1(
                    check_id="validation_split_consistency",
                    status="FAIL",
                    message="Validation suite contains cases not in VALIDATION split!",
                    category="CASE_PLAN"
                ))
            else:
                checks.append(PreflightCheckV1(
                    check_id="validation_suite_integrity",
                    status="PASS",
                    message="Validation suite verified: exactly 96 cases in VALIDATION split.",
                    category="CASE_PLAN"
                ))

    elif plan.launcher_mode == LauncherMode.TEST:
        if len(plan.resolved_cases) != 60:
            checks.append(PreflightCheckV1(
                check_id="test_case_count",
                status="FAIL",
                message=f"Test benchmark suite must contain exactly 60 cases, got {len(plan.resolved_cases)}.",
                category="CASE_PLAN"
            ))
        else:
            all_test = all(c.split == "TEST" for c in plan.resolved_cases)
            if not all_test:
                checks.append(PreflightCheckV1(
                    check_id="test_split_consistency",
                    status="FAIL",
                    message="Test suite contains cases not in TEST split!",
                    category="CASE_PLAN"
                ))
            else:
                checks.append(PreflightCheckV1(
                    check_id="test_suite_integrity",
                    status="PASS",
                    message="Test benchmark suite verified: exactly 60 paired cases in TEST split.",
                    category="CASE_PLAN"
                ))

    # 6. Protocol order index monotonicity
    if plan.resolved_cases:
        p_indices = [c.protocol_order_index for c in plan.resolved_cases]
        expected_indices = list(range(1, len(plan.resolved_cases) + 1))
        if p_indices == expected_indices:
            checks.append(PreflightCheckV1(
                check_id="protocol_order_monotonicity",
                status="PASS",
                message=f"Protocol order indices strictly sequential 1 through {len(plan.resolved_cases)}.",
                category="CASE_PLAN"
            ))
        else:
            checks.append(PreflightCheckV1(
                check_id="protocol_order_monotonicity",
                status="FAIL",
                message="Protocol order indices are not strictly sequential!",
                category="CASE_PLAN"
            ))

    # 7. Stochastic agent seed policy
    if plan.launcher_mode in (LauncherMode.VALIDATION, LauncherMode.TEST):
        if reg.inference_stochasticity == "stochastic":
            if plan.agent_seed is None:
                checks.append(PreflightCheckV1(
                    check_id="stochastic_seed_policy",
                    status="FAIL",
                    message=f"Stochastic agent '{reg.agent_id}' on benchmark {plan.launcher_mode.value} requires explicit agent_seed in {{101, 202, 303}}.",
                    category="SEED"
                ))
            elif plan.agent_seed not in (101, 202, 303):
                checks.append(PreflightCheckV1(
                    check_id="stochastic_seed_policy",
                    status="FAIL",
                    message=f"Invalid benchmark agent_seed={plan.agent_seed}. Gate-5 declared replicate seeds are strictly {{101, 202, 303}}.",
                    category="SEED"
                ))
            else:
                checks.append(PreflightCheckV1(
                    check_id="stochastic_seed_policy",
                    status="PASS",
                    message=f"Benchmark stochastic agent replicate seed={plan.agent_seed} verified.",
                    category="SEED"
                ))
        else:
            # Deterministic agent on benchmark should have agent_seed=None
            if plan.agent_seed is not None:
                checks.append(PreflightCheckV1(
                    check_id="deterministic_seed_policy",
                    status="FAIL",
                    message=f"Deterministic agent '{reg.agent_id}' must not receive an agent_seed in benchmark mode, but got agent_seed={plan.agent_seed}.",
                    category="SEED"
                ))
            else:
                checks.append(PreflightCheckV1(
                    check_id="deterministic_seed_policy",
                    status="PASS",
                    message="Deterministic agent seed is None as required.",
                    category="SEED"
                ))
    else:
        # In Sandbox / Audit
        checks.append(PreflightCheckV1(
            check_id="sandbox_seed_policy",
            status="PASS",
            message=f"Agent seed={plan.agent_seed} accepted for {plan.launcher_mode.value}.",
            category="SEED"
        ))

    # 8. Rendering policy
    if plan.launcher_mode in (LauncherMode.VALIDATION, LauncherMode.TEST):
        if plan.render_mode != "OFF":
            checks.append(PreflightCheckV1(
                check_id="benchmark_render_policy",
                status="FAIL",
                message=f"Benchmark mode {plan.launcher_mode.value} prohibits native rendering (got render_mode='{plan.render_mode}'). Benchmarks must run headless (OFF) to avoid timing distortion.",
                category="OBSERVABILITY"
            ))
        else:
            checks.append(PreflightCheckV1(
                check_id="benchmark_render_policy",
                status="PASS",
                message="Headless benchmark rendering verified.",
                category="OBSERVABILITY"
            ))
    else:
        checks.append(PreflightCheckV1(
            check_id="sandbox_render_policy",
            status="PASS",
            message=f"Render mode '{plan.render_mode}' allowed for {plan.launcher_mode.value}.",
            category="OBSERVABILITY"
        ))

    # 9. W&B credential presence policy
    if plan.wandb_mode == WandbMode.ONLINE:
        has_key = bool(os.environ.get("WANDB_API_KEY"))
        if not has_key:
            checks.append(PreflightCheckV1(
                check_id="wandb_online_credential",
                status="FAIL",
                message="WandbMode.ONLINE requested but WANDB_API_KEY is not set in environment. Silent degradation is strictly prohibited.",
                category="OBSERVABILITY"
            ))
        else:
            checks.append(PreflightCheckV1(
                check_id="wandb_online_credential",
                status="PASS",
                message="WANDB_API_KEY detected in environment memory for ONLINE mode.",
                category="OBSERVABILITY"
            ))
    else:
        checks.append(PreflightCheckV1(
            check_id="wandb_mode_policy",
            status="PASS",
            message=f"WandbMode '{plan.wandb_mode.value}' requires no online credentials.",
            category="OBSERVABILITY"
        ))

    # Aggregate counts and verdict
    fail_count = sum(1 for c in checks if c.status == "FAIL")
    warning_count = sum(1 for c in checks if c.status == "WARNING")
    pass_count = sum(1 for c in checks if c.status == "PASS")

    can_execute = (fail_count == 0)
    if fail_count > 0:
        verdict = "BLOCKED"
    elif warning_count > 0:
        verdict = "PROCEED_WITH_WARNINGS"
    else:
        verdict = "READY"

    return PreflightReportV1(
        can_execute=can_execute,
        checks=checks,
        fail_count=fail_count,
        warning_count=warning_count,
        pass_count=pass_count,
        summary_verdict=verdict
    )
