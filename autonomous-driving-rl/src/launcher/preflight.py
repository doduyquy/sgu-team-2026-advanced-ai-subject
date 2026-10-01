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
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.launcher.cases import (
    LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256,
    LOCKED_TEST_CASE_MANIFEST_SHA256,
    LOCKED_VALIDATION_CASE_MANIFEST_SHA256,
    get_default_project_root,
    load_manifest_csv,
)
from src.launcher.contracts import (
    GATE7_LOCKED_OBSERVABILITY_HASH,
    build_launcher_contract_core,
    compute_launcher_contract_sha256,
    compute_platform_execution_contract_sha256,
)
from src.launcher.models import (
    LauncherMode,
    PreflightBlockedError,
    ResolvedExperimentPlanV1,
    compute_resolved_plan_sha256,
    mode_to_run_kind,
)
from src.launcher.resolver import (
    GATE5_LOCKED_BENCHMARK_HASH,
    GATE6_LOCKED_AGENT_HASH,
    GATE6_LOCKED_RUNTIME_HASH,
    GATE7_LOCKED_LOGGING_HASH,
)
from src.platform import (
    CERTIFIED_ACTION_ADAPTERS,
    PINNED_METADRIVE_COMMIT,
    PINNED_METADRIVE_VERSION,
    WandbMode,
    canonical_csv_file_sha256,
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
    custom_environment_provenance: Optional[Dict[str, Any]] = None,
    registry: Optional[Any] = None
) -> PreflightReportV1:
    """
    Executes a comprehensive, pure battery of preflight checks against ResolvedExperimentPlanV1.
    Returns PreflightReportV1. can_execute is True ONLY if fail_count == 0.
    """
    root = project_root or get_default_project_root()
    git_prov = custom_git_provenance or capture_git_provenance(root)
    env_prov = custom_environment_provenance or capture_environment_provenance(root)

    from src.launcher.registry import build_default_agent_registry
    authoritative_registry = registry or build_default_agent_registry()

    checks: List[PreflightCheckV1] = []

    # 0. Authoritative Registry Binding & Anti-Forgery Check (Problem 1)
    if not authoritative_registry.has_agent(plan.agent_registration.agent_id):
        checks.append(PreflightCheckV1(
            check_id="agent_registry_binding",
            status="FAIL",
            message=f"Agent '{plan.agent_registration.agent_id}' is not present in authoritative AgentRegistryV1.",
            category="AGENT"
        ))
    else:
        trusted_reg = authoritative_registry.get(plan.agent_registration.agent_id)
        reg_diffs = []
        for field_name in (
            "agent_id", "agent_version", "stage_label", "method_family", "purpose",
            "implementation_ref", "input_profile_id", "action_adapter_id",
            "inference_stochasticity", "stateful_within_episode", "benchmark_eligible",
            "sandbox_eligible", "audit_eligible", "requires_checkpoint"
        ):
            plan_val = getattr(plan.agent_registration, field_name)
            trusted_val = getattr(trusted_reg, field_name)
            if plan_val != trusted_val:
                reg_diffs.append(f"{field_name} (plan={plan_val} != registry={trusted_val})")

        if reg_diffs:
            checks.append(PreflightCheckV1(
                check_id="agent_registry_binding",
                status="FAIL",
                message=f"Agent registration divergence from authoritative registry: {'; '.join(reg_diffs)}. Unauthorized metadata tampering detected!",
                category="AGENT"
            ))
        else:
            checks.append(PreflightCheckV1(
                check_id="agent_registry_binding",
                status="PASS",
                message=f"Agent registration '{plan.agent_registration.agent_id}' strictly verified against authoritative AgentRegistryV1.",
                category="AGENT"
            ))

    # 0a. Immutable AgentDescriptor projection parity check (Problem 3)
    desc_diffs = []
    if plan.agent_descriptor.agent_id != plan.agent_registration.agent_id:
        desc_diffs.append("agent_id")
    if plan.agent_descriptor.agent_version != plan.agent_registration.agent_version:
        desc_diffs.append("agent_version")
    if plan.agent_descriptor.input_profile_id != plan.agent_registration.input_profile_id:
        desc_diffs.append("input_profile_id")
    if plan.agent_descriptor.action_adapter_id != plan.agent_registration.action_adapter_id:
        desc_diffs.append("action_adapter_id")
    if plan.agent_descriptor.inference_stochasticity != plan.agent_registration.inference_stochasticity:
        desc_diffs.append("inference_stochasticity")
    if plan.agent_descriptor.stateful_within_episode != plan.agent_registration.stateful_within_episode:
        desc_diffs.append("stateful_within_episode")
    if plan.agent_descriptor.method_family != plan.agent_registration.method_family:
        desc_diffs.append("method_family")

    if desc_diffs:
        checks.append(PreflightCheckV1(
            check_id="agent_descriptor_projection_parity",
            status="FAIL",
            message=f"Agent descriptor diverges from registration: {', '.join(desc_diffs)}",
            category="AGENT"
        ))
    else:
        checks.append(PreflightCheckV1(
            check_id="agent_descriptor_projection_parity",
            status="PASS",
            message="Agent descriptor verified identical to registration projection.",
            category="AGENT"
        ))

    # 0. Contract hash chain verification (Section 3)
    core_contract = build_launcher_contract_core()
    expected_launcher_hash = compute_launcher_contract_sha256(core_contract)
    expected_execution_hash = compute_platform_execution_contract_sha256(expected_launcher_hash)

    contract_failures = []
    if plan.benchmark_contract_sha256 != GATE5_LOCKED_BENCHMARK_HASH:
        contract_failures.append(f"Gate 5 benchmark hash mismatch: plan={plan.benchmark_contract_sha256} != expected={GATE5_LOCKED_BENCHMARK_HASH}")
    if plan.agent_contract_sha256 != GATE6_LOCKED_AGENT_HASH:
        contract_failures.append(f"Gate 6 agent hash mismatch: plan={plan.agent_contract_sha256} != expected={GATE6_LOCKED_AGENT_HASH}")
    if plan.platform_runtime_contract_sha256 != GATE6_LOCKED_RUNTIME_HASH:
        contract_failures.append(f"Gate 6 runtime hash mismatch: plan={plan.platform_runtime_contract_sha256} != expected={GATE6_LOCKED_RUNTIME_HASH}")
    if plan.logging_contract_sha256 != GATE7_LOCKED_LOGGING_HASH:
        contract_failures.append(f"Gate 7 logging hash mismatch: plan={plan.logging_contract_sha256} != expected={GATE7_LOCKED_LOGGING_HASH}")
    if plan.platform_observability_contract_sha256 != GATE7_LOCKED_OBSERVABILITY_HASH:
        contract_failures.append(f"Gate 7 observability hash mismatch: plan={plan.platform_observability_contract_sha256} != expected={GATE7_LOCKED_OBSERVABILITY_HASH}")
    if plan.launcher_contract_sha256 != expected_launcher_hash:
        contract_failures.append(f"Gate 7.5A launcher contract hash mismatch: plan={plan.launcher_contract_sha256} != expected={expected_launcher_hash}")
    if plan.platform_execution_contract_sha256 != expected_execution_hash:
        contract_failures.append(f"Gate 7.5A execution contract hash mismatch: plan={plan.platform_execution_contract_sha256} != expected={expected_execution_hash}")

    if contract_failures:
        checks.append(PreflightCheckV1(
            check_id="contract_hash_chain",
            status="FAIL",
            message=f"Contract hash chain broken: {'; '.join(contract_failures)}",
            category="CONTRACT"
        ))
    else:
        checks.append(PreflightCheckV1(
            check_id="contract_hash_chain",
            status="PASS",
            message="Full contract hash chain verified across Gates 5, 6, 7, and 7.5A.",
            category="CONTRACT"
        ))

    # 0b. Gate-5 manifest source hash verification (Section 4)
    manifest_failures = []
    geom_split_p = root / "results" / "audits" / "evaluation_protocol" / "geometry_split_manifest.csv"
    val_manifest_p = root / "results" / "audits" / "evaluation_protocol" / "validation_case_manifest.csv"
    test_manifest_p = root / "results" / "audits" / "evaluation_protocol" / "test_case_manifest.csv"

    if geom_split_p.exists():
        h = canonical_csv_file_sha256(geom_split_p)
        if h != LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256:
            manifest_failures.append(f"geometry_split_manifest.csv hash mismatch ({h[:8]} != {LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256[:8]})")
    else:
        manifest_failures.append("geometry_split_manifest.csv missing")

    if val_manifest_p.exists():
        h = canonical_csv_file_sha256(val_manifest_p)
        if h != LOCKED_VALIDATION_CASE_MANIFEST_SHA256:
            manifest_failures.append(f"validation_case_manifest.csv hash mismatch ({h[:8]} != {LOCKED_VALIDATION_CASE_MANIFEST_SHA256[:8]})")
    else:
        manifest_failures.append("validation_case_manifest.csv missing")

    if test_manifest_p.exists():
        h = canonical_csv_file_sha256(test_manifest_p)
        if h != LOCKED_TEST_CASE_MANIFEST_SHA256:
            manifest_failures.append(f"test_case_manifest.csv hash mismatch ({h[:8]} != {LOCKED_TEST_CASE_MANIFEST_SHA256[:8]})")
    else:
        manifest_failures.append("test_case_manifest.csv missing")

    if manifest_failures:
        checks.append(PreflightCheckV1(
            check_id="gate5_manifest_hashes",
            status="FAIL",
            message=f"Gate-5 source manifest verification failed: {'; '.join(manifest_failures)}",
            category="CONTRACT"
        ))
    else:
        checks.append(PreflightCheckV1(
            check_id="gate5_manifest_hashes",
            status="PASS",
            message="Gate-5 source CSV manifests verified against locked cryptographic hashes.",
            category="CONTRACT"
        ))

    # 0c. Plan hash integrity re-evaluation (Section 1)
    recomputed_plan_hash = compute_resolved_plan_sha256(plan.to_dict())
    if plan.resolved_plan_sha256 != recomputed_plan_hash:
        checks.append(PreflightCheckV1(
            check_id="plan_hash_integrity",
            status="FAIL",
            message=f"Resolved plan hash mismatch! Stored={plan.resolved_plan_sha256} != Recomputed={recomputed_plan_hash}. Plan was tampered with or corrupted after resolution.",
            category="CONTRACT"
        ))
    else:
        checks.append(PreflightCheckV1(
            check_id="plan_hash_integrity",
            status="PASS",
            message="Resolved plan hash strictly verified against actual current plan semantics.",
            category="CONTRACT"
        ))

    # 0d. Structural mode and protocol scope consistency (Section 1)
    expected_run_kind = mode_to_run_kind(plan.launcher_mode)
    expected_canonical = (plan.launcher_mode in (LauncherMode.VALIDATION, LauncherMode.TEST))
    scope_map = {
        LauncherMode.SANDBOX: "SANDBOX_SINGLE_CASE",
        LauncherMode.VALIDATION: "VALIDATION_SUITE_96",
        LauncherMode.TEST: "TEST_SUITE_60",
        LauncherMode.AUDIT: "AUDIT_SUITE"
    }
    expected_scope = scope_map.get(plan.launcher_mode, "")

    structural_errors = []
    if plan.run_kind != expected_run_kind:
        structural_errors.append(f"run_kind mismatch: {plan.run_kind.value} != expected {expected_run_kind.value}")
    if plan.canonical_run != expected_canonical:
        structural_errors.append(f"canonical_run mismatch: {plan.canonical_run} != expected {expected_canonical}")
    if plan.protocol_scope != expected_scope:
        structural_errors.append(f"protocol_scope mismatch: '{plan.protocol_scope}' != expected '{expected_scope}'")
    if abs(plan.control_frequency_hz - 10.0) > 1e-6:
        structural_errors.append(f"control_frequency_hz mismatch: {plan.control_frequency_hz} != 10.0")
    if abs(plan.control_dt_s - 0.1) > 1e-6:
        structural_errors.append(f"control_dt_s mismatch: {plan.control_dt_s} != 0.1")

    if structural_errors:
        checks.append(PreflightCheckV1(
            check_id="plan_structural_consistency",
            status="FAIL",
            message=f"Plan structural consistency violations: {'; '.join(structural_errors)}",
            category="CONTRACT"
        ))
    else:
        checks.append(PreflightCheckV1(
            check_id="plan_structural_consistency",
            status="PASS",
            message="Plan structural semantics, canonical status, and 10 Hz control verified.",
            category="CONTRACT"
        ))

    # 0e. Deep row-by-row, field-by-field manifest parity for benchmark modes (Section 1)
    if plan.launcher_mode == LauncherMode.VALIDATION:
        val_records = load_manifest_csv(val_manifest_p, expected_sha256=LOCKED_VALIDATION_CASE_MANIFEST_SHA256)
        val_field_diffs = []
        if len(plan.resolved_cases) != len(val_records):
            val_field_diffs.append(f"case count mismatch: {len(plan.resolved_cases)} != {len(val_records)}")
        else:
            for idx, (c, m) in enumerate(zip(plan.resolved_cases, val_records)):
                if str(c.case_id) != str(m["case_id"]): val_field_diffs.append(f"row {idx} case_id mismatch")
                if int(c.case_index) != int(m["case_index"]): val_field_diffs.append(f"row {idx} case_index mismatch")
                if int(c.protocol_order_index) != int(m["protocol_order_index"]): val_field_diffs.append(f"row {idx} protocol_order_index mismatch")
                if str(c.split) != "VALIDATION": val_field_diffs.append(f"row {idx} split mismatch")
                if str(c.tier) != str(m["tier"]): val_field_diffs.append(f"row {idx} tier mismatch")
                if str(c.sequence) != str(m["sequence"]): val_field_diffs.append(f"row {idx} sequence mismatch")
                if int(c.geometry_generation_seed) != int(m["geometry_generation_seed"]): val_field_diffs.append(f"row {idx} geom_seed mismatch")
                if str(c.geometry_sha256) != str(m["geometry_sha256"]): val_field_diffs.append(f"row {idx} geom_hash mismatch")
                if int(c.environment_seed) != int(m["environment_seed"]): val_field_diffs.append(f"row {idx} env_seed mismatch")
                if not math.isfinite(float(c.traffic_density)) or float(c.traffic_density) != float(m["traffic_density"]): val_field_diffs.append(f"row {idx} traffic_density mismatch ({c.traffic_density} != {m['traffic_density']})")
                if int(c.horizon_steps) != int(m["horizon_steps"]): val_field_diffs.append(f"row {idx} horizon_steps mismatch")
                if len(val_field_diffs) > 5:
                    val_field_diffs.append("... additional field mismatches omitted")
                    break

        if val_field_diffs:
            checks.append(PreflightCheckV1(
                check_id="validation_manifest_deep_parity",
                status="FAIL",
                message=f"Validation plan cases diverge from Gate-5 manifest: {'; '.join(val_field_diffs)}",
                category="CASE_PLAN"
            ))
        else:
            checks.append(PreflightCheckV1(
                check_id="validation_manifest_deep_parity",
                status="PASS",
                message="Validation plan matches locked Gate-5 manifest row-by-row and field-by-field across all 11 fields.",
                category="CASE_PLAN"
            ))

    elif plan.launcher_mode == LauncherMode.TEST:
        test_records = load_manifest_csv(test_manifest_p, expected_sha256=LOCKED_TEST_CASE_MANIFEST_SHA256)
        test_field_diffs = []
        if len(plan.resolved_cases) != len(test_records):
            test_field_diffs.append(f"case count mismatch: {len(plan.resolved_cases)} != {len(test_records)}")
        else:
            for idx, (c, m) in enumerate(zip(plan.resolved_cases, test_records)):
                if str(c.case_id) != str(m["case_id"]): test_field_diffs.append(f"row {idx} case_id mismatch")
                if int(c.case_index) != int(m["case_index"]): test_field_diffs.append(f"row {idx} case_index mismatch")
                if int(c.protocol_order_index) != int(m["protocol_order_index"]): test_field_diffs.append(f"row {idx} protocol_order_index mismatch")
                if str(c.split) != "TEST": test_field_diffs.append(f"row {idx} split mismatch")
                if str(c.tier) != str(m["tier"]): test_field_diffs.append(f"row {idx} tier mismatch")
                if str(c.sequence) != str(m["sequence"]): test_field_diffs.append(f"row {idx} sequence mismatch")
                if int(c.geometry_generation_seed) != int(m["geometry_generation_seed"]): test_field_diffs.append(f"row {idx} geom_seed mismatch")
                if str(c.geometry_sha256) != str(m["geometry_sha256"]): test_field_diffs.append(f"row {idx} geom_hash mismatch")
                if int(c.environment_seed) != int(m["environment_seed"]): test_field_diffs.append(f"row {idx} env_seed mismatch")
                if not math.isfinite(float(c.traffic_density)) or float(c.traffic_density) != float(m["traffic_density"]): test_field_diffs.append(f"row {idx} traffic_density mismatch ({c.traffic_density} != {m['traffic_density']})")
                if int(c.horizon_steps) != int(m["horizon_steps"]): test_field_diffs.append(f"row {idx} horizon_steps mismatch")
                if len(test_field_diffs) > 5:
                    test_field_diffs.append("... additional field mismatches omitted")
                    break

        if test_field_diffs:
            checks.append(PreflightCheckV1(
                check_id="test_manifest_deep_parity",
                status="FAIL",
                message=f"Test plan cases diverge from Gate-5 manifest: {'; '.join(test_field_diffs)}",
                category="CASE_PLAN"
            ))
        else:
            checks.append(PreflightCheckV1(
                check_id="test_manifest_deep_parity",
                status="PASS",
                message="Test plan matches locked Gate-5 manifest row-by-row and field-by-field across all 11 fields.",
                category="CASE_PLAN"
            ))

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
