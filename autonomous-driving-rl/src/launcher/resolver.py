"""
Experiment Plan Resolver for Research Platform V1 Launcher (Gate 7.5A).

This module implements the deterministic resolution of LaunchRequestV1 into ResolvedExperimentPlanV1:
- Maps LauncherMode to Gate-7 RunKind.
- Resolves cases via Gate-5 manifests (SANDBOX TRAIN-only, 96 VALIDATION, 60 TEST).
- Reconciles agent capabilities, seeds, rendering, and W&B modes.
- Computes deterministic resolved_plan_sha256.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.launcher.cases import (
    get_default_project_root,
    resolve_audit_cases,
    resolve_sandbox_case,
    resolve_test_cases,
    resolve_validation_cases,
)
from src.launcher.models import (
    LaunchRequestV1,
    LauncherMode,
    ResolvedCaseV1,
    ResolvedExperimentPlanV1,
    compute_resolved_plan_sha256,
    mode_to_run_kind,
)
from src.launcher.registry import (
    AgentRegistryV1,
    build_canonical_agent_registry,
    compute_canonical_registry_sha256,
)
from src.platform import AgentDescriptor, WandbMode

GATE5_LOCKED_BENCHMARK_HASH = "9ddd889b84d8705fae618879e5035556c80d0276a3dc2a58a7963937ebb59f77"
GATE6_LOCKED_AGENT_HASH = "53aa37079ff44afa75d9a3f921b0c1f98c4600d882fced51fc8d9197795058eb"
GATE6_LOCKED_RUNTIME_HASH = "c7698768539a769c7b2bc6b90771ff234276a974b6e0353bbb03119faf2f79ad"
GATE7_LOCKED_LOGGING_HASH = "0d17915556d83f49d9519576e95920c0853700f33a91593ef85b3aeef6cbb9f2"
GATE7_LOCKED_OBSERVABILITY_HASH = "f00c27fca8abf8145596576cd0dd2540181ff8ae7eccf7806c864921933d3581"
DEFAULT_LAUNCHER_CONTRACT_HASH = "5e6269b6e32349bd3ee34ca52583c00f832bd0a737330c894e9a8d314653db04"
DEFAULT_PLATFORM_EXECUTION_HASH = "442beafd86212419dc7b3edfa60e53715bf33877d3dd72e61c54091520b0563d"


def resolve_experiment_plan(
    request: LaunchRequestV1,
    registry: Optional[AgentRegistryV1] = None,
    project_root: Optional[Path] = None,
    custom_launcher_contract_sha256: Optional[str] = None,
    custom_platform_execution_contract_sha256: Optional[str] = None
) -> ResolvedExperimentPlanV1:
    """
    Deterministically transforms a user LaunchRequestV1 into a concrete ResolvedExperimentPlanV1.
    Performs zero side-effects on disk, MetaDrive simulator, or remote W&B services.
    Enforces canonical registry authority for benchmark modes (VALIDATION/TEST).
    """
    root = project_root or get_default_project_root()
    canonical_reg = build_canonical_agent_registry()
    canonical_reg_sha = compute_canonical_registry_sha256(canonical_reg)

    # 1. Authority resolution: benchmark modes bind to canonical registry fingerprint
    if request.mode in (LauncherMode.VALIDATION, LauncherMode.TEST):
        effective_registry = registry if registry is not None else canonical_reg
    else:
        effective_registry = registry if registry is not None else canonical_reg

    # Verify agent existence in effective registry
    agent_reg = effective_registry.get(request.agent_id)
    factory_ref = effective_registry.get_factory_ref(request.agent_id)

    # 2. Determine lower-level RunKind and canonical status
    run_kind = mode_to_run_kind(request.mode)
    canonical_run = (request.mode in (LauncherMode.VALIDATION, LauncherMode.TEST))

    # Reject illegal benchmark filtering intent (Section 4)
    if request.mode in (LauncherMode.VALIDATION, LauncherMode.TEST):
        illegal_selectors = []
        if request.tier is not None:
            illegal_selectors.append(f"tier='{request.tier}'")
        if request.sequence is not None:
            illegal_selectors.append(f"sequence='{request.sequence}'")
        if request.geometry_generation_seed is not None:
            illegal_selectors.append(f"geometry_generation_seed={request.geometry_generation_seed}")
        if request.environment_seed is not None:
            illegal_selectors.append(f"environment_seed={request.environment_seed}")

        if illegal_selectors:
            suite_name = "frozen full 96-case suite" if request.mode == LauncherMode.VALIDATION else "frozen full 60-case paired suite"
            raise ValueError(
                f"Illegal case selector(s) {illegal_selectors} supplied to {request.mode.value} mode. "
                f"{request.mode.value} is the {suite_name}; custom filtering, subsetting, and benchmark smoke runs "
                "are strictly prohibited to prevent benchmark gaming and selective reporting."
            )

    # Validate render mode in core (Section 5)
    render_mode = request.render_mode.upper() if isinstance(request.render_mode, str) else ""
    if render_mode not in ("OFF", "NATIVE"):
        raise ValueError(f"Invalid render_mode '{request.render_mode}'. Must be one of ['NATIVE', 'OFF'].")
    if request.mode in (LauncherMode.VALIDATION, LauncherMode.TEST):
        if render_mode != "OFF":
            raise ValueError(
                f"Benchmark mode {request.mode.value} requires render_mode='OFF' (headless execution mandatory). "
                f"Got render_mode='{render_mode}'."
            )

    # 3. Resolve cases and protocol scope according to mode
    warnings: List[str] = []
    if request.mode == LauncherMode.SANDBOX:
        resolved_cases = [resolve_sandbox_case(request, root)]
        protocol_scope = "SANDBOX_SINGLE_CASE"
        if not canonical_run:
            warnings.append("Sandbox run is exploratory/development; results are non-canonical.")
    elif request.mode == LauncherMode.VALIDATION:
        resolved_cases = resolve_validation_cases(root)
        protocol_scope = "VALIDATION_SUITE_96"
    elif request.mode == LauncherMode.TEST:
        resolved_cases = resolve_test_cases(root)
        protocol_scope = "TEST_SUITE_60"
    elif request.mode == LauncherMode.AUDIT:
        resolved_cases = resolve_audit_cases(request, root)
        protocol_scope = "AUDIT_SUITE"
        warnings.append("Audit run is for platform verification; results are non-canonical.")
    else:
        raise ValueError(f"Unsupported LauncherMode: {request.mode}")

    # 4. Resolve agent seed
    resolved_agent_seed: Optional[int] = request.agent_seed
    if agent_reg.inference_stochasticity == "stochastic":
        if request.mode in (LauncherMode.SANDBOX, LauncherMode.AUDIT) and resolved_agent_seed is None:
            resolved_agent_seed = 101  # Documented default for exploratory sandbox
            warnings.append(f"Stochastic agent '{agent_reg.agent_id}' defaulted to agent_seed=101 in {request.mode.value} mode.")
    elif agent_reg.inference_stochasticity == "deterministic":
        if request.mode in (LauncherMode.SANDBOX, LauncherMode.AUDIT) and resolved_agent_seed is not None:
            warnings.append(f"Deterministic agent '{agent_reg.agent_id}' provided with agent_seed={resolved_agent_seed}; seed is ignored by deterministic policy.")

    # 5. Resolve W&B mode:
    # If request.wandb_mode is None (AUTO / unspecified):
    #   SANDBOX / AUDIT -> DISABLED
    #   VALIDATION / TEST -> OFFLINE
    # If user explicitly specifies a WandbMode, respect it strictly!
    if request.wandb_mode is None:
        if request.mode in (LauncherMode.VALIDATION, LauncherMode.TEST):
            wandb_mode = WandbMode.OFFLINE
        else:
            wandb_mode = WandbMode.DISABLED
    else:
        wandb_mode = request.wandb_mode

    # 6. Contract hashes
    launcher_hash = custom_launcher_contract_sha256 or DEFAULT_LAUNCHER_CONTRACT_HASH
    execution_hash = custom_platform_execution_contract_sha256 or DEFAULT_PLATFORM_EXECUTION_HASH

    # Construct immutable AgentDescriptor for execution
    agent_descriptor = AgentDescriptor(
        agent_id=agent_reg.agent_id,
        agent_version=agent_reg.agent_version,
        input_profile_id=agent_reg.input_profile_id,
        action_adapter_id=agent_reg.action_adapter_id,
        inference_stochasticity=agent_reg.inference_stochasticity,
        stateful_within_episode=agent_reg.stateful_within_episode,
        method_family=agent_reg.method_family
    )

    raw_plan_dict = {
        "launcher_mode": request.mode.value,
        "run_kind": run_kind.value,
        "canonical_run": canonical_run,
        "protocol_scope": protocol_scope,
        "agent_registration": agent_reg.to_dict(),
        "agent_descriptor": asdict(agent_descriptor),
        "resolved_cases": [c.to_dict() for c in resolved_cases],
        "render_mode": render_mode,
        "wandb_mode": wandb_mode.value,
        "agent_seed": resolved_agent_seed,
        "control_frequency_hz": 10.0,
        "control_dt_s": 0.1,
        "benchmark_contract_sha256": GATE5_LOCKED_BENCHMARK_HASH,
        "agent_contract_sha256": GATE6_LOCKED_AGENT_HASH,
        "platform_runtime_contract_sha256": GATE6_LOCKED_RUNTIME_HASH,
        "logging_contract_sha256": GATE7_LOCKED_LOGGING_HASH,
        "platform_observability_contract_sha256": GATE7_LOCKED_OBSERVABILITY_HASH,
        "launcher_contract_sha256": launcher_hash,
        "platform_execution_contract_sha256": execution_hash,
        "canonical_agent_registry_sha256": canonical_reg_sha,
        "factory_ref": factory_ref,
    }

    plan_hash = compute_resolved_plan_sha256(raw_plan_dict)

    return ResolvedExperimentPlanV1(
        launcher_mode=request.mode,
        run_kind=run_kind,
        canonical_run=canonical_run,
        protocol_scope=protocol_scope,
        agent_registration=agent_reg,
        agent_descriptor=agent_descriptor,
        resolved_cases=tuple(resolved_cases),
        render_mode=render_mode,
        wandb_mode=wandb_mode,
        agent_seed=resolved_agent_seed,
        control_frequency_hz=10.0,
        control_dt_s=0.1,
        benchmark_contract_sha256=GATE5_LOCKED_BENCHMARK_HASH,
        agent_contract_sha256=GATE6_LOCKED_AGENT_HASH,
        platform_runtime_contract_sha256=GATE6_LOCKED_RUNTIME_HASH,
        logging_contract_sha256=GATE7_LOCKED_LOGGING_HASH,
        platform_observability_contract_sha256=GATE7_LOCKED_OBSERVABILITY_HASH,
        launcher_contract_sha256=launcher_hash,
        platform_execution_contract_sha256=execution_hash,
        canonical_agent_registry_sha256=canonical_reg_sha,
        factory_ref=factory_ref,
        resolved_plan_sha256=plan_hash,
        warnings=warnings
    )
