"""
Workbench Contract Specification and Cryptographic Hashes for Research Platform V1 (Gate 7.5B).

This module defines the machine-readable WorkbenchContractV1 and computes additive contract hashes:
- workbench_contract_sha256: Cryptographic fingerprint of presentation, protocol, and isolation semantics.
- platform_workbench_contract_sha256: Additive hash joining execution (Gate 7.5A) with workbench (Gate 7.5B).
"""

from pathlib import Path
from typing import Any, Dict, Optional

from src.platform import canonical_json_sha256

GATE7_5A_BASELINE_MERGE_SHA = "0376da8b8bd3e1c3b60d43ed371b6713c0cb31f6"
GATE7_5A_LAUNCHER_CONTRACT_HASH = "5e6269b6e32349bd3ee34ca52583c00f832bd0a737330c894e9a8d314653db04"
GATE7_5A_PLATFORM_EXECUTION_CONTRACT_HASH = "442beafd86212419dc7b3edfa60e53715bf33877d3dd72e61c54091520b0563d"
PINNED_PYSIDE6_VERSION = "6.11.2"
WORKBENCH_PROTOCOL_VERSION = "WORKBENCH_PROTOCOL_V1"


def build_workbench_contract_core(
    status: str = "AUDIT-CANDIDATE",
    custom_protocol_policy: Optional[Dict[str, Any]] = None,
    custom_isolation_policy: Optional[Dict[str, Any]] = None,
    custom_render_policy: Optional[Dict[str, Any]] = None,
    custom_authority_policy: Optional[Dict[str, Any]] = None,
    custom_security_policy: Optional[Dict[str, Any]] = None,
    custom_cancellation_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Builds the authoritative machine-readable WorkbenchContractV1 core dictionary.
    Locks UI boundaries, QProcess isolation, protocol rules, and non-authoritative constraints.
    """
    default_protocol_policy = {
        "protocol_version": WORKBENCH_PROTOCOL_VERSION,
        "sentinel": "@@WORKBENCH@@",
        "message_types": [
            "WORKER_READY",
            "PLAN_RESOLVED",
            "PREFLIGHT_REPORT",
            "LAUNCHER_EVENT",
            "EXECUTION_REPORT",
            "WORKER_ERROR",
            "WORKER_DONE",
        ],
        "serialization": "JSON-safe fields matching LaunchRequestV1 directly",
        "malformed_stdout_policy": "IGNORE_NON_SENTINEL_OUTPUT_SAFELY_CAPTURE_WORKER_LOGS",
    }
    if custom_protocol_policy:
        default_protocol_policy.update(custom_protocol_policy)

    default_isolation_policy = {
        "mechanism": "QProcess subprocess isolation",
        "python_executable": "sys.executable (inherits exact active Conda/Python environment)",
        "gui_thread_policy": "NO_SIMULATOR_OR_EXECUTOR_IN_QT_THREAD",
        "qthread_policy": "PROHIBITED_FOR_METADRIVE_EXECUTOR",
    }
    if custom_isolation_policy:
        default_isolation_policy.update(custom_isolation_policy)

    default_render_policy = {
        "native_render_window": "SEPARATE_NATIVE_WINDOW_FOR_SANDBOX_AUDIT",
        "embedded_viewport": "PROHIBITED_IN_GATE_7_5B_PASS_B1",
        "rgb_streaming": "PROHIBITED_IN_GATE_7_5B_PASS_B1",
        "scientific_observation_profile_modification": "STRICTLY_PROHIBITED",
    }
    if custom_render_policy:
        default_render_policy.update(custom_render_policy)

    default_authority_policy = {
        "scientific_authority_rule": "GUI_IS_THIN_PRESENTATION_CONTROL_ONLY_GATE_7_5A_REMAINS_FINAL_AUTHORITY",
        "benchmark_subset_policy": "NO_PARTIAL_VALIDATION_OR_TEST_EXECUTION_STRICTLY_PROHIBITED",
        "control_disabling_rule": "UX_CONVENIENCE_ONLY_NOT_SCIENTIFIC_ENFORCEMENT",
        "telemetry_source_rule": "DERIVED_EXCLUSIVELY_FROM_LAUNCHER_EVENT_V1",
        "no_internal_inspection_rule": "NO_INSPECTION_OF_AGENT_POLICY_INTERNALS_OR_SIMULATOR_OBJECTS",
    }
    if custom_authority_policy:
        default_authority_policy.update(custom_authority_policy)

    default_security_policy = {
        "secret_handling": "NO_SECRETS_IN_COMMAND_LINE_OR_WORKBENCH_REQUEST_JSON",
        "wandb_api_key_policy": "INHERITED_FROM_ENVIRONMENT_ONLY_NEVER_SERIALIZED_OR_COMMITTED",
        "ephemeral_credentials_scan": "AUDIT_VERIFIES_ZERO_TOKENS_KEYS_COMMITTED",
    }
    if custom_security_policy:
        default_security_policy.update(custom_security_policy)

    default_cancellation_policy = {
        "graceful_cancellation_scope": "OUTSIDE_GATE_7_5B_PASS_B1",
        "telemetry_control_separation": "LAUNCHER_EVENT_IS_OBSERVATIONAL_TELEMETRY_NOT_CONTROL_CHANNEL",
        "force_terminate_policy": "NON_GRACEFUL_LABELED_FORCE_TERMINATE_NO_RUN_STATE_FALSIFICATION",
        "window_close_policy": "CONFIRMATION_DIALOG_IF_WORKER_ACTIVE_NO_SILENT_KILL",
    }
    if custom_cancellation_policy:
        default_cancellation_policy.update(custom_cancellation_policy)

    core: Dict[str, Any] = {
        "metadata": {
            "contract_name": "WorkbenchContractV1",
            "spec_version": "1.0.0",
            "status": status,
            "gate": "Gate 7.5B (Research Workbench GUI)",
        },
        "baseline_provenance": {
            "baseline_main_merge_sha": GATE7_5A_BASELINE_MERGE_SHA,
            "baseline_launcher_contract_sha256": GATE7_5A_LAUNCHER_CONTRACT_HASH,
            "baseline_platform_execution_contract_sha256": GATE7_5A_PLATFORM_EXECUTION_CONTRACT_HASH,
        },
        "gui_technology": {
            "framework": "PySide6",
            "pinned_version": PINNED_PYSIDE6_VERSION,
        },
        "protocol_policy": default_protocol_policy,
        "isolation_policy": default_isolation_policy,
        "rendering_policy": default_render_policy,
        "authority_policy": default_authority_policy,
        "security_policy": default_security_policy,
        "cancellation_policy": default_cancellation_policy,
        "thin_client_invariant": (
            "The Research Workbench can disappear entirely and every scientific experiment remains "
            "fully definable, resolvable, validated, executable, and reproducible through Gate 7.5A alone."
        ),
    }

    return core


def compute_workbench_contract_sha256(contract_dict: Dict[str, Any]) -> str:
    """Computes canonical SHA-256 fingerprint over WorkbenchContractV1 core dictionary."""
    return canonical_json_sha256(contract_dict)


def compute_platform_workbench_contract_sha256(workbench_contract_sha256: str) -> str:
    """
    Computes additive platform workbench contract hash linking Gate 7.5A execution contract,
    Workbench contract, and pinned PySide6 version.
    """
    payload = {
        "platform_execution_contract_sha256": GATE7_5A_PLATFORM_EXECUTION_CONTRACT_HASH,
        "workbench_contract_sha256": workbench_contract_sha256,
        "pyside6_pinned_version": PINNED_PYSIDE6_VERSION,
        "workbench_protocol_version": WORKBENCH_PROTOCOL_VERSION,
        "platform_specification_gate": "Gate 7.5B",
    }
    return canonical_json_sha256(payload)
