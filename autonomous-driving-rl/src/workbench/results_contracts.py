"""
Workbench Results Contract Specification and Cryptographic Hashes for Research Platform V1 (Gate 7.5B Pass B2).

This module defines the machine-readable WorkbenchResultsContractV1 and computes additive contract hashes:
- workbench_results_contract_sha256: Cryptographic fingerprint of results browsing, artifact integrity, and live UX semantics.
- platform_workbench_results_contract_sha256: Additive hash joining B1 platform workbench hash with B2 results contract.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from src.platform import canonical_json_sha256
from src.workbench.contracts import (
    GATE7_5A_BASELINE_MERGE_SHA,
    PINNED_PYSIDE6_VERSION,
    WORKBENCH_PROTOCOL_VERSION,
)

GATE7_5B_B1_BASELINE_MERGE_SHA = "99f0a66aa156004a5a1ec7993d2368ff83d7c325"
GATE7_5B_B1_WORKBENCH_CONTRACT_HASH = "e5711485e6571a04c336739ebc6f285213a1d9631fa29820a89707b37875a82b"
GATE7_5B_B1_PLATFORM_WORKBENCH_CONTRACT_HASH = "36f3a0891a9153df12a8b53b01f2064afbf848eafda462149a768ec989199e47"


def build_workbench_results_contract_core(
    status: str = "AUDIT-CANDIDATE",
    custom_results_policy: Optional[Dict[str, Any]] = None,
    custom_integrity_policy: Optional[Dict[str, Any]] = None,
    custom_telemetry_policy: Optional[Dict[str, Any]] = None,
    custom_ranking_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Builds the authoritative machine-readable WorkbenchResultsContractV1 core dictionary.
    Locks local artifact authority, primary metric cards, integrity checking, and live telemetry semantics.
    """
    default_results_policy = {
        "authority_source": "Gate 7 local artifacts are authoritative (run_manifest.json, run_state.json, episodes.csv, timing.csv, summary.json, wandb_sync.json, run_integrity.json)",
        "read_only_browsing": "NO_FILE_MUTATION_NO_REPAIR_NO_DELETE_NO_REWRITE",
        "primary_metrics_rule": "Direct pass-through from summary.json['overall_metrics'] without GUI recalculation",
        "primary_metrics": [
            "clean_success_rate",
            "safety_failure_rate",
            "mean_final_route_completion",
            "median_final_route_completion",
            "mean_time_to_clean_success_s",
        ],
        "null_time_to_success_presentation": "N/A — no clean-success samples",
        "diagnostic_metrics_separation": "mean_episode_return and episode_return are labeled diagnostic training signals, NOT benchmark ranking scores",
    }
    if custom_results_policy:
        default_results_policy.update(custom_results_policy)

    default_integrity_policy = {
        "integrity_states": ["VERIFIED", "NOT_FINAL", "UNVERIFIED", "FAILED"],
        "tamper_detection": "GATE7_CANONICAL_SEMANTIC_CONTENT_HASH_MISMATCH_MARKS_FAILED_NO_AUTO_REPAIR",
        "hash_verification_targets": [
            "experiment_config_sha256",
            "run_manifest_sha256",
            "episodes_sha256",
            "summary_sha256",
            "timing_sha256",
            "run_state_sha256",
            "wandb_sync_sha256",
        ],
        "incomplete_run_semantics": "RUNNING / INITIALIZING / INTERRUPTED / FAILED runs without final summary are NOT_FINAL, not corrupted",
    }
    if custom_integrity_policy:
        default_integrity_policy.update(custom_integrity_policy)

    default_telemetry_policy = {
        "telemetry_source": "Exclusively LauncherEventV1 messages received via WORKBENCH_PROTOCOL_V1",
        "cadence_wording": "sampled LauncherEventV1 progress telemetry (sampled every 10 decision steps under current executor)",
        "no_simulator_query": "ZERO_DIRECT_ACCESS_TO_SIMULATOR_OR_AGENT_FROM_QT",
        "reset_policy": "RUN_STARTED resets run buffer; EPISODE_STARTED resets episode trace",
        "free_text_message_rule": "DO_NOT_PARSE_SCIENTIFIC_DATA_FROM_FREE_TEXT_MESSAGE",
    }
    if custom_telemetry_policy:
        default_telemetry_policy.update(custom_telemetry_policy)

    default_ranking_policy = {
        "ranking_rule": "NO_LEADERBOARDS_NO_MULTI_RUN_RANKING_NO_BEST_AGENT_IN_B2",
        "evaluation_unit": "INDIVIDUAL_RUN_INSPECTION_ONLY",
    }
    if custom_ranking_policy:
        default_ranking_policy.update(custom_ranking_policy)

    core: Dict[str, Any] = {
        "metadata": {
            "contract_name": "WorkbenchResultsContractV1",
            "spec_version": "1.0.0",
            "status": status,
            "gate": "Gate 7.5B Pass B2 (Results Browser, Artifact Integrity & Live UX)",
        },
        "baseline_provenance": {
            "baseline_merge_sha": GATE7_5B_B1_BASELINE_MERGE_SHA,
            "baseline_workbench_contract_sha256": GATE7_5B_B1_WORKBENCH_CONTRACT_HASH,
            "baseline_platform_workbench_contract_sha256": GATE7_5B_B1_PLATFORM_WORKBENCH_CONTRACT_HASH,
        },
        "gui_technology": {
            "framework": "PySide6",
            "pinned_version": PINNED_PYSIDE6_VERSION,
            "visualization": "matplotlib (standard project dependency)",
        },
        "results_policy": default_results_policy,
        "integrity_policy": default_integrity_policy,
        "telemetry_policy": default_telemetry_policy,
        "ranking_policy": default_ranking_policy,
        "privacy_policy": {
            "credential_handling": "NO_SECRETS_OR_API_KEYS_IN_VIEW_OR_PERSISTED_ARTIFACTS",
            "path_sanitization": "COMMITTED_AUDIT_EVIDENCE_STRICTLY_EXCLUDES_PERSONAL_HOME_PATHS",
        },
        "scientific_invariant": (
            "summary.json is authoritative aggregate result truth. "
            "episodes.csv is authoritative episode-row truth. "
            "timing.csv is authoritative episode timing truth. "
            "run_state.json is authoritative lifecycle truth. "
            "run_integrity.json is authoritative final fingerprint declaration. "
            "Workbench only reads, verifies, formats, filters, and visualizes; "
            "it never becomes the source of those truths."
        ),
    }
    return core


def compute_workbench_results_contract_sha256(contract_dict: Dict[str, Any]) -> str:
    """Computes canonical SHA-256 fingerprint over WorkbenchResultsContractV1 core dictionary."""
    return canonical_json_sha256(contract_dict)


def compute_platform_workbench_results_contract_sha256(workbench_results_contract_sha256: str) -> str:
    """
    Computes additive platform workbench results contract hash linking Pass B1 platform workbench hash
    with Pass B2 results contract hash.
    """
    payload = {
        "platform_workbench_contract_sha256": GATE7_5B_B1_PLATFORM_WORKBENCH_CONTRACT_HASH,
        "workbench_results_contract_sha256": workbench_results_contract_sha256,
        "pyside6_pinned_version": PINNED_PYSIDE6_VERSION,
        "platform_specification_gate": "Gate 7.5B Pass B2",
    }
    return canonical_json_sha256(payload)
