"""
Workbench Hardening Contract Specification and Cryptographic Hashes for Research Platform V1 (Gate 7.5B Pass B3).

This module defines the machine-readable WorkbenchHardeningContractV1 and computes additive contract hashes:
- workbench_hardening_contract_sha256: Cryptographic fingerprint of final hardening invariants.
- platform_workbench_hardening_contract_sha256: Additive hash joining B2 platform workbench hash with B3 hardening contract.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from src.platform import canonical_json_sha256
from src.workbench.contracts import PINNED_PYSIDE6_VERSION, WORKBENCH_PROTOCOL_VERSION
from src.workbench.results_contracts import (
    GATE7_5B_B1_PLATFORM_WORKBENCH_CONTRACT_HASH,
    GATE7_5B_B1_WORKBENCH_CONTRACT_HASH,
)

GATE7_5B_B2_BASELINE_MERGE_SHA = "9c5d1fabe07e0c4cc24c28d0b5b2857f5496652d"
GATE7_5B_B2_WORKBENCH_RESULTS_CONTRACT_HASH = "21c2c541a891967d9206e2dc86781f1bfc62b794c13e7e1ee1c3732676e38429"
GATE7_5B_B2_PLATFORM_WORKBENCH_RESULTS_CONTRACT_HASH = "a241e2890b65290e28d717714123625bc2809100fe3287f2fb3e14dba8d6e433"


def build_workbench_hardening_contract_core(
    status: str = "AUDIT-CANDIDATE",
    custom_process_policy: Optional[Dict[str, Any]] = None,
    custom_operation_policy: Optional[Dict[str, Any]] = None,
    custom_resource_policy: Optional[Dict[str, Any]] = None,
    custom_results_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Builds the authoritative machine-readable WorkbenchHardeningContractV1 core dictionary.
    Locks final process isolation, lifecycle recovery, memory bounding, and trust boundaries.
    """
    default_process_policy = {
        "max_active_workers": 1,
        "unexpected_exit_semantics": "NONZERO_OR_ZERO_EXIT_WITHOUT_WORKER_DONE_CLASSIFIED_AS_UNEXPECTED_EXIT",
        "force_termination_rule": "NON_GRACEFUL_PROCESS_KILL_NO_SCIENTIFIC_LIFECYCLE_FABRICATION",
        "temp_request_cleanup_rule": "ALL_TERMINAL_OUTCOMES_CLEAN_TEMP_REQUEST_JSON",
        "close_while_running_policy": "EXPLICIT_CONFIRMATION_DIALOG_DEFAULT_CANCEL_NO_SILENT_KILL",
    }
    if custom_process_policy:
        default_process_policy.update(custom_process_policy)

    default_operation_policy = {
        "single_operation_invariant": "SYNCHRONOUS_UI_LOCK_PREVENTS_OVERLAPPING_PLAN_OR_RUN",
        "stale_preflight_invalidation": "REQUEST_EDITS_INVALIDATE_STORED_VERDICT_LATE_SIGNALS_REJECTED",
        "results_autoload_rule": "AUTOLOAD_ONLY_ON_SUCCESSFUL_RUN_PLAN_AND_FAILED_RUN_DO_NOT_AUTOLOAD_STALE",
    }
    if custom_operation_policy:
        default_operation_policy.update(custom_operation_policy)

    default_resource_policy = {
        "max_event_log_blocks": 5000,
        "chart_memory_policy": "CURRENT_EPISODE_RESET_PREVENTS_UNBOUNDED_ARTIST_GROWTH",
        "no_polling_rule": "NO_CONTINUOUS_BACKGROUND_FILESYSTEM_POLLING_MANUAL_REFRESH_ONLY",
    }
    if custom_resource_policy:
        default_resource_policy.update(custom_resource_policy)

    default_results_policy = {
        "integrity_trust_rule": "ONLY_INTEGRITY_VERIFIED_COMPLETE_RUNS_MAY_BE_LABELED_AUTHORITATIVE",
        "untrusted_states_policy": "FAILED_UNVERIFIED_NOT_FINAL_SUPPRESS_AUTHORITATIVE_CLAIMS",
        "single_source_episode_schema": "DIRECT_REFERENCE_TO_GATE7_EPISODE_CSV_COLUMNS",
        "single_source_timing_schema": "DIRECT_REFERENCE_TO_EPISODE_TIMING_RECORD_FIELDS",
        "execution_report_labeling": "EXECUTION_REPORTED_PROVISIONAL_UNTIL_RESULTS_VERIFIED",
        "no_fabricated_zero_rates": "MISSING_OUTCOME_RATES_OMITTED_NOT_REPRESENTED_AS_ZERO",
    }
    if custom_results_policy:
        default_results_policy.update(custom_results_policy)

    core: Dict[str, Any] = {
        "metadata": {
            "contract_name": "WorkbenchHardeningContractV1",
            "spec_version": "1.0.0",
            "status": status,
            "gate": "Gate 7.5B Pass B3 (Final Workbench Hardening & Platform Handoff)",
        },
        "baseline_provenance": {
            "baseline_merge_sha": GATE7_5B_B2_BASELINE_MERGE_SHA,
            "baseline_workbench_contract_sha256": GATE7_5B_B1_WORKBENCH_CONTRACT_HASH,
            "baseline_platform_workbench_contract_sha256": GATE7_5B_B1_PLATFORM_WORKBENCH_CONTRACT_HASH,
            "baseline_workbench_results_contract_sha256": GATE7_5B_B2_WORKBENCH_RESULTS_CONTRACT_HASH,
            "baseline_platform_workbench_results_contract_sha256": GATE7_5B_B2_PLATFORM_WORKBENCH_RESULTS_CONTRACT_HASH,
        },
        "gui_technology": {
            "framework": "PySide6",
            "pinned_version": PINNED_PYSIDE6_VERSION,
            "protocol_version": WORKBENCH_PROTOCOL_VERSION,
        },
        "process_policy": default_process_policy,
        "operation_policy": default_operation_policy,
        "resource_policy": default_resource_policy,
        "results_policy": default_results_policy,
        "closure_rule": (
            "Gate 7.5B Pass B3 is the final workbench hardening pass. "
            "Upon independent review and explicit merge, Gate 7.5B is CLOSED. "
            "There is no Gate 7.5B Pass B4. The immediate next research milestone is Stage 0 Random Baseline."
        ),
    }
    return core


def compute_workbench_hardening_contract_sha256(contract_dict: Dict[str, Any]) -> str:
    """Computes canonical SHA-256 fingerprint over WorkbenchHardeningContractV1 core dictionary."""
    return canonical_json_sha256(contract_dict)


def compute_platform_workbench_hardening_contract_sha256(workbench_hardening_contract_sha256: str) -> str:
    """
    Computes additive platform workbench hardening contract hash linking Pass B2 platform workbench hash
    with Pass B3 hardening contract hash.
    """
    payload = {
        "platform_workbench_results_contract_sha256": GATE7_5B_B2_PLATFORM_WORKBENCH_RESULTS_CONTRACT_HASH,
        "workbench_hardening_contract_sha256": workbench_hardening_contract_sha256,
        "pyside6_pinned_version": PINNED_PYSIDE6_VERSION,
        "platform_specification_gate": "Gate 7.5B Pass B3",
    }
    return canonical_json_sha256(payload)
