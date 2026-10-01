"""
Launcher Contract Specification and Cryptographic Hashes for Research Platform V1 (Gate 7.5A).

This module defines the machine-readable LauncherContractV1 and computes additive contract hashes:
- launcher_contract_sha256: Cryptographic fingerprint of all launcher orchestration semantics.
- platform_execution_contract_sha256: Additive hash joining observability (Gate 7) with launcher execution (Gate 7.5A).
"""

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.launcher.events import LauncherEventType, LauncherEventV1
from src.launcher.models import (
    EXCLUDED_DESCRIPTIVE_FIELDS,
    SCIENTIFIC_REGISTRATION_FIELDS,
    AgentRegistrationV1,
    LaunchRequestV1,
    LauncherMode,
    ResolvedCaseV1,
    ResolvedExperimentPlanV1,
)
from src.platform import (
    PINNED_METADRIVE_COMMIT,
    PINNED_METADRIVE_VERSION,
    canonical_json_sha256,
)

GATE7_LOCKED_OBSERVABILITY_HASH = "f00c27fca8abf8145596576cd0dd2540181ff8ae7eccf7806c864921933d3581"


def build_launcher_contract_core(
    status: str = "AUDIT-CANDIDATE",
    custom_modes: Optional[Dict[str, Any]] = None,
    custom_sandbox_policy: Optional[Dict[str, Any]] = None,
    custom_validation_policy: Optional[Dict[str, Any]] = None,
    custom_test_policy: Optional[Dict[str, Any]] = None,
    custom_render_policy: Optional[Dict[str, Any]] = None,
    custom_seed_policy: Optional[Dict[str, Any]] = None,
    custom_preflight_policy: Optional[Dict[str, Any]] = None,
    custom_legacy_policy: Optional[Dict[str, Any]] = None,
    custom_control_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Builds the authoritative machine-readable LauncherContractV1 core dictionary.
    Locks orchestration modes, holdout safety policies, preflight rules, event schemas,
    and legacy component exclusions for canonical cryptographic hashing.
    """
    runtime_dataclasses = {
        "AgentRegistrationV1": [f.name for f in dataclasses.fields(AgentRegistrationV1)],
        "LaunchRequestV1": [f.name for f in dataclasses.fields(LaunchRequestV1)],
        "ResolvedCaseV1": [f.name for f in dataclasses.fields(ResolvedCaseV1)],
        "ResolvedExperimentPlanV1": [f.name for f in dataclasses.fields(ResolvedExperimentPlanV1)],
        "LauncherEventV1": [f.name for f in dataclasses.fields(LauncherEventV1)],
    }

    runtime_enums = {
        "LauncherMode": [e.value for e in LauncherMode],
        "LauncherEventType": [e.value for e in LauncherEventType],
    }

    default_sandbox_policy = {
        "split_restriction": "TRAIN_ONLY",
        "holdout_violation_policy": "FAIL_LOUDLY_NO_SILENT_FALLBACK",
        "ambiguity_policy": "EXPLICIT_SEED_APPLIES_ALL_SELECTORS_MULTIPLE_MATCHES_FAIL",
        "default_geometry_selection": "DETERMINISTIC_SORTED_FIRST_ELIGIBLE",
        "default_environment_seed": 0,
        "default_stochastic_agent_seed": 101,
        "episode_count": 1,
        "description": "Sandbox V1 resolves exactly one episode strictly within TRAIN split for exploratory inspection."
    }
    if custom_sandbox_policy:
        default_sandbox_policy.update(custom_sandbox_policy)

    default_validation_policy = {
        "source_manifest": "validation_case_manifest.csv",
        "expected_case_count": 96,
        "split": "VALIDATION",
        "custom_subset_policy": "PROHIBITED_IN_GATE_7_5A",
        "description": "Validation resolves the full frozen Gate-5 validation suite across 48 geometries and 2 seeds."
    }
    if custom_validation_policy:
        default_validation_policy.update(custom_validation_policy)

    default_test_policy = {
        "source_manifest": "test_case_manifest.csv",
        "expected_case_count": 60,
        "split": "TEST",
        "partial_run_policy": "STRICTLY_PROHIBITED",
        "test_smoke_policy": "STRICTLY_PROHIBITED",
        "description": "Test resolves the immutable 60 paired cases across 12 test geometries and 5 seeds."
    }
    if custom_test_policy:
        default_test_policy.update(custom_test_policy)

    default_render_policy = {
        "SANDBOX": ["OFF", "NATIVE"],
        "AUDIT": ["OFF", "NATIVE"],
        "VALIDATION": ["OFF"],
        "TEST": ["OFF"],
        "benchmark_rule": "VALIDATION and TEST benchmark runs must execute headless (render=OFF) to prevent timing distortion."
    }
    if custom_render_policy:
        default_render_policy.update(custom_render_policy)

    default_seed_policy = {
        "stochastic_benchmark_replicate_seeds": [101, 202, 303],
        "deterministic_benchmark_agent_seed": "MUST_BE_NONE",
        "sandbox_stochastic_default": 101,
        "description": "Stochastic benchmark agents require explicit seed in {101, 202, 303}; deterministic agents require None."
    }
    if custom_seed_policy:
        default_seed_policy.update(custom_seed_policy)

    default_preflight_policy = {
        "rule": "ZERO_FAIL_CHECKS_TO_EXECUTE",
        "check_categories": ["CONTRACT", "ENVIRONMENT", "AGENT", "CASE_PLAN", "SEED", "OBSERVABILITY", "SECURITY"],
        "contract_chain_verification": "PREFLIGHT_VERIFIES_ENTIRE_HASH_CHAIN_ACROSS_GATES_5_6_7_7.5A",
        "manifest_hash_verification": "PREFLIGHT_VERIFIES_LOCKED_GATE5_MANIFEST_HASHES",
        "plan_integrity_recomputation": "PREFLIGHT_RECOMPUTES_RESOLVED_PLAN_SHA256_AND_REJECTS_TAMPERING",
        "deep_manifest_parity_rule": "PREFLIGHT_VERIFIES_ALL_11_FIELDS_ROW_BY_ROW_AGAINST_SOURCE_MANIFEST",
        "custom_run_id_sanitization": "STRICT_CONTAINMENT_GRAMMAR_NO_TRAVERSAL",
        "benchmark_dirty_git_policy": "FAIL_PREFLIGHT",
        "benchmark_unverified_env_policy": "FAIL_PREFLIGHT",
        "fixture_on_benchmark_policy": "FAIL_PREFLIGHT",
        "online_without_key_policy": "FAIL_PREFLIGHT"
    }
    if custom_preflight_policy:
        default_preflight_policy.update(custom_preflight_policy)

    default_run_id_policy = {
        "grammar": "^[a-zA-Z0-9_\\-\\.]+$",
        "max_length": 64,
        "containment": "STRICTLY_WITHIN_RUNS_ROOT_NO_NESTED_DIRS",
        "plan_hash_inclusion": False
    }

    default_control_policy = {
        "physics_world_step_size": 0.02,
        "decision_repeat": 5,
        "decision_dt_s": 0.1,
        "control_frequency_hz": 10.0,
        "traffic_mode": "trigger",
        "truncate_as_terminate": False,
        "reward_passthrough": "SIGNED_DELTA_ROUTE_COMPLETION_PRESERVED_WITHOUT_CLAMPING"
    }
    if custom_control_policy:
        default_control_policy.update(custom_control_policy)

    default_legacy_policy = {
        "excluded_legacy_components": [
            "src/environments/course_env_v1.py",
            "src/evaluation/evaluate_random.py",
            "CourseEnvV1 (35D observation, Discrete(5))"
        ],
        "rule": "Launcher Core must never import, instantiate, or invoke legacy CourseEnvV1 or evaluate_random.py."
    }
    if custom_legacy_policy:
        default_legacy_policy.update(custom_legacy_policy)

    plan_agent_fingerprint_policy = {
        "scientific_registration_fields": list(SCIENTIFIC_REGISTRATION_FIELDS),
        "excluded_descriptive_fields": list(EXCLUDED_DESCRIPTIVE_FIELDS),
        "rule": "ResolvedExperimentPlanV1 must cryptographically fingerprint all 14 scientific registration fields, canonical_agent_registry_sha256, factory_ref, and the projected AgentDescriptor, strictly excluding cosmetic description."
    }

    registry_binding_policy = {
        "canonical_authority_rule": "CANONICAL_BENCHMARK_MODES_MUST_USE_COMMITTED_CANONICAL_REGISTRY",
        "custom_registry_scope": "CUSTOM_REGISTRIES_PERMITTED_ONLY_FOR_NONCANONICAL_SANDBOX_AND_AUDIT",
        "parity_rule": "ALL_14_SCIENTIFIC_FIELDS_MUST_MATCH_CANONICAL_REGISTRY_EXACTLY",
        "factory_identity_rule": "FACTORY_QUALNAME_MUST_MATCH_CANONICAL_REGISTRY_METADATA",
        "implementation_identity_rule": "RUNTIME_CLASS_QUALNAME_MUST_MATCH_CANONICAL_IMPLEMENTATION_REF",
        "descriptor_projection_rule": "AGENT_DESCRIPTOR_MUST_MATCH_REGISTRATION_PROJECTION_EXACTLY"
    }

    core: Dict[str, Any] = {
        "metadata": {
            "contract_name": "LauncherContractV1",
            "spec_version": "1.0.0",
            "status": status,
            "gate": "Gate 7.5A (Research Platform V1)"
        },
        "launcher_modes": [e.value for e in LauncherMode],
        "run_kind_mapping": {
            "SANDBOX": "AUDIT",
            "AUDIT": "AUDIT",
            "VALIDATION": "VALIDATION_EVALUATION",
            "TEST": "TEST_EVALUATION"
        },
        "sandbox_resolution_policy": default_sandbox_policy,
        "validation_resolution_policy": default_validation_policy,
        "test_resolution_policy": default_test_policy,
        "render_policy": default_render_policy,
        "seed_policy": default_seed_policy,
        "preflight_policy": default_preflight_policy,
        "control_policy": default_control_policy,
        "custom_run_id_policy": default_run_id_policy,
        "plan_agent_fingerprint_policy": plan_agent_fingerprint_policy,
        "registry_binding_policy": registry_binding_policy,
        "agent_lifecycle_policy": {
            "instance_lifecycle": "ONE_AGENT_INSTANCE_PER_EXPERIMENT_RUN_RESET_PER_EPISODE",
            "descriptor_verification": "RUNTIME_DESCRIPTOR_MUST_MATCH_REGISTRATION_EXACTLY",
            "cleanup": "TRY_FINALLY_GUARANTEES_ENV_AND_AGENT_CLOSE"
        },
        "wandb_auto_resolution_policy": {
            "unspecified_defaults": {
                "SANDBOX": "DISABLED",
                "AUDIT": "DISABLED",
                "VALIDATION": "OFFLINE",
                "TEST": "OFFLINE"
            },
            "explicit_override_rule": "EXPLICIT_DISABLED_REMAINS_DISABLED_ACROSS_ALL_MODES"
        },
        "manifest_name_mapping": {
            "SANDBOX": "geometry_split_manifest.csv",
            "AUDIT": "geometry_split_manifest.csv",
            "VALIDATION": "validation_case_manifest.csv",
            "TEST": "test_case_manifest.csv"
        },
        "legacy_exclusion_policy": default_legacy_policy,
        "provenance_embedding_policy": {
            "target": "ExperimentRunConfig.algorithm_hyperparameters['launcher']",
            "embedded_keys": [
                "launcher_mode",
                "launcher_contract_sha256",
                "platform_execution_contract_sha256",
                "resolved_plan_sha256",
                "protocol_scope"
            ]
        },
        "event_policy": {
            "types": [e.value for e in LauncherEventType],
            "isolation": "Observer telemetry strictly decoupled from AgentPolicy"
        },
        "runtime_dataclass_schemas": runtime_dataclasses,
        "runtime_enum_schemas": runtime_enums,
    }

    if custom_modes:
        core["launcher_modes"] = custom_modes

    return core


def compute_launcher_contract_sha256(contract_dict: Dict[str, Any]) -> str:
    """Computes canonical SHA-256 fingerprint over LauncherContractV1 core dictionary."""
    return canonical_json_sha256(contract_dict)


def compute_platform_execution_contract_sha256(launcher_contract_sha256: str) -> str:
    """
    Computes additive platform execution contract hash linking Gate 7 observability with Gate 7.5A launcher.
    """
    payload = {
        "platform_observability_contract_sha256": GATE7_LOCKED_OBSERVABILITY_HASH,
        "launcher_contract_sha256": launcher_contract_sha256,
        "metadrive_commit": PINNED_METADRIVE_COMMIT,
        "metadrive_version": PINNED_METADRIVE_VERSION,
        "platform_specification_gate": "Gate 7.5A"
    }
    return canonical_json_sha256(payload)
