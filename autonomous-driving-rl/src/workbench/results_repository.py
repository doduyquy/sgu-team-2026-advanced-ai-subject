"""
Workbench Results Repository and Read-Only Artifact Access Layer (Gate 7.5B Pass B2 Correction 2).

This module provides the read-only inspection and integrity verification layer for Gate-7
experiment runs. It strictly consumes authoritative Gate-7 schemas (RunManifestV1, RunStateV1,
ExperimentRunConfig, RunIntegrityRecord) and never mutates, appends, repairs, or recalculates
official metrics.
"""

import csv
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.platform import (
    RunKind,
    RunStatus,
    canonical_csv_file_sha256,
    canonical_json_file_sha256,
    canonical_json_sha256,
)


class IntegrityDisplayStatus(str, Enum):
    """Explicit integrity verification states for Workbench display."""
    VERIFIED = "VERIFIED"        # COMPLETE run, all artifacts present, all hashes match
    NOT_FINAL = "NOT_FINAL"      # Incomplete lifecycle (RUNNING, INITIALIZING, INTERRUPTED, FAILED)
    UNVERIFIED = "UNVERIFIED"    # Appears COMPLETE but lacks run_integrity.json
    FAILED = "FAILED"            # Claimed COMPLETE but one or more hash fingerprints mismatch


@dataclass(frozen=True)
class ArtifactIntegrityCheckResult:
    """Record of an individual artifact hash check."""
    artifact_name: str
    expected_sha256: Optional[str]
    observed_sha256: Optional[str]
    status: str                  # "PASS", "FAIL", "NOT_APPLICABLE", "MISSING"


@dataclass(frozen=True)
class RunArtifactSnapshotV1:
    """
    Immutable view-model of a discovered experiment run directory.
    Zero scientific authority; purely a typed presentation snapshot.
    """
    run_id: str
    run_dir: str
    status: str
    run_kind: str
    canonical_run: bool
    started_at_utc: Optional[str]
    finished_at_utc: Optional[str]
    recorded_episode_count: int
    expected_episode_count: Optional[int]
    agent_id: Optional[str]
    agent_version: Optional[str]
    input_profile_id: Optional[str]
    action_adapter_id: Optional[str]
    inference_stochasticity: Optional[str]
    stateful_within_episode: Optional[bool]
    agent_seed: Optional[int]
    manifest_name: Optional[str]
    experiment_config_sha256: Optional[str]
    git_commit_sha: Optional[str]
    git_worktree_dirty: Optional[bool]
    dirty_override: bool
    unverified_env_override: bool
    environment_verification_status: str
    metadrive_version: Optional[str]
    metadrive_commit: Optional[str]
    metadrive_pinned_version: Optional[str]
    metadrive_pinned_commit: Optional[str]
    metadrive_verification_status: Optional[str]
    metadrive_verification_reason: Optional[str]
    platform_contracts: Dict[str, str]
    integrity_status: IntegrityDisplayStatus
    integrity_details: List[ArtifactIntegrityCheckResult]
    artifacts_present: List[str]
    summary_payload: Optional[Dict[str, Any]]
    episode_rows: List[Dict[str, Any]]
    timing_rows: List[Dict[str, Any]]
    wandb_sync_payload: Optional[Dict[str, Any]]
    technical_failures_payload: Optional[List[Dict[str, Any]]]
    failure_category: Optional[str] = None
    sanitized_failure_message: Optional[str] = None


class RunArtifactRepository:
    """
    Read-only repository discovering and verifying Gate-7 run directories on disk.
    Never creates, modifies, or deletes run artifacts.
    """

    CORE_ARTIFACTS = (
        "run_manifest.json",
        "run_state.json",
        "episodes.csv",
        "timing.csv",
        "summary.json",
        "wandb_sync.json",
        "run_integrity.json",
    )

    def __init__(self, runs_root: Path):
        self._runs_root = Path(runs_root).resolve()

    @property
    def runs_root(self) -> Path:
        return self._runs_root

    def discover_run_ids(self) -> List[str]:
        """Discovers candidate run subdirectories within runs_root."""
        if not self._runs_root.exists() or not self._runs_root.is_dir():
            return []

        candidates = []
        for entry in self._runs_root.iterdir():
            if entry.is_dir():
                # A candidate run directory must have run_state.json or run_manifest.json
                if (entry / "run_state.json").exists() or (entry / "run_manifest.json").exists():
                    candidates.append(entry.name)

        return sorted(candidates)

    def load_run_snapshot(self, run_id: str) -> Optional[RunArtifactSnapshotV1]:
        """
        Loads a single run directory into a RunArtifactSnapshotV1 view model.
        Returns None if directory does not exist, escapes runs_root, or fails run identity parity.
        """
        # Strict path containment check (Specification Section 8)
        try:
            target_path = (self._runs_root / run_id).resolve()
            target_path.relative_to(self._runs_root)
            if target_path.parent != self._runs_root:
                return None
        except Exception:
            return None

        if not target_path.exists() or not target_path.is_dir():
            return None

        run_dir = target_path

        # Both run_state.json and run_manifest.json are required for a valid Gate-7 run snapshot
        run_state_path = run_dir / "run_state.json"
        manifest_path = run_dir / "run_manifest.json"

        if not run_state_path.exists() or not manifest_path.exists():
            return None

        # 1. Parse run_state.json (lifecycle authority)
        try:
            with open(run_state_path, "r", encoding="utf-8") as f:
                state_data = json.load(f)
            if not isinstance(state_data, dict):
                return None
        except Exception:
            return None

        # 2. Parse run_manifest.json (configuration authority)
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            if not isinstance(manifest_data, dict):
                return None
        except Exception:
            return None

        # 3. Strict Run Identity Parity across dir, state, and manifest
        dir_basename = run_dir.name
        state_run_id = state_data.get("run_id")
        manifest_run_id = manifest_data.get("run_id")

        if not (dir_basename == state_run_id == manifest_run_id):
            return None

        actual_run_id = dir_basename
        status_str = state_data.get("status", "UNKNOWN")

        # 4. Strict 4-Way Run Identity Parity (Specification Section 2)
        # If COMPLETE and run_integrity.json exists, integrity run_id MUST equal actual_run_id
        integrity_path = run_dir / "run_integrity.json"
        if status_str == "COMPLETE" and integrity_path.exists():
            try:
                with open(integrity_path, "r", encoding="utf-8") as f:
                    integ_data = json.load(f)
                if isinstance(integ_data, dict):
                    integ_run_id = integ_data.get("run_id")
                    if integ_run_id != actual_run_id:
                        # Cross-run identity corruption: reject run snapshot as malformed
                        return None
            except Exception:
                pass

        run_kind_str = manifest_data.get("run_kind", "UNKNOWN")
        canonical_run = bool(state_data.get("canonical_run", False))
        started_at = state_data.get("started_at_utc")
        finished_at = state_data.get("finished_at_utc")
        recorded_eps = int(state_data.get("recorded_episode_count", 0))
        expected_eps = state_data.get("expected_episode_count")
        dirty_override = bool(state_data.get("dirty_override", False))
        unverified_env_override = bool(state_data.get("unverified_env_override", False))
        env_verif_status = state_data.get("environment_verification_status", "UNKNOWN")
        failure_category = state_data.get("failure_category")
        sanitized_failure_message = state_data.get("sanitized_failure_message")

        # Parse config fields from run_manifest.json["config"]
        config_data = manifest_data.get("config", {})
        agent_id = config_data.get("agent_id")
        agent_version = config_data.get("agent_version")
        input_profile_id = config_data.get("input_profile_id")
        action_adapter_id = config_data.get("action_adapter_id")
        inference_stoch = config_data.get("inference_stochasticity")
        stateful_in_ep = config_data.get("stateful_within_episode")
        agent_seed = config_data.get("agent_seed")
        manifest_name = config_data.get("manifest_name")
        cfg_sha = manifest_data.get("experiment_config_sha256") or config_data.get("experiment_config_sha256")

        # Parse git and environment provenance
        git_prov = manifest_data.get("git_provenance", {})
        git_commit = git_prov.get("git_commit_sha")
        git_dirty = git_prov.get("git_worktree_dirty")

        env_prov = manifest_data.get("environment_provenance", {})
        md_ver = env_prov.get("metadrive_version")
        md_commit = env_prov.get("metadrive_commit")
        md_pin_ver = env_prov.get("metadrive_pinned_version")
        md_pin_commit = env_prov.get("metadrive_pinned_commit")
        md_verif_status = env_prov.get("metadrive_verification_status")
        md_verif_reason = env_prov.get("metadrive_verification_reason")

        platform_contracts = manifest_data.get("platform_contracts", {})

        # 5. Read summary.json if present
        summary_path = run_dir / "summary.json"
        summary_payload = None
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary_payload = json.load(f)
            except Exception:
                summary_payload = None

        # 6. Read episodes.csv rows if present
        episodes_path = run_dir / "episodes.csv"
        episode_rows = []
        if episodes_path.exists():
            try:
                with open(episodes_path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    episode_rows = list(reader)
            except Exception:
                episode_rows = []

        # 7. Read timing.csv rows if present
        timing_path = run_dir / "timing.csv"
        timing_rows = []
        if timing_path.exists():
            try:
                with open(timing_path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    timing_rows = list(reader)
            except Exception:
                timing_rows = []

        # 8. Read wandb_sync.json if present
        wandb_path = run_dir / "wandb_sync.json"
        wandb_payload = None
        if wandb_path.exists():
            try:
                with open(wandb_path, "r", encoding="utf-8") as f:
                    wandb_payload = json.load(f)
            except Exception:
                wandb_payload = None

        # 9. Read technical_failures.jsonl if present
        tf_path = run_dir / "technical_failures.jsonl"
        tf_payload = None
        if tf_path.exists():
            try:
                tf_records = []
                with open(tf_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            tf_records.append(json.loads(line))
                tf_payload = tf_records
            except Exception:
                tf_payload = None

        # Determine artifacts present
        artifacts_present = [f.name for f in run_dir.iterdir() if f.is_file()]

        # 10. Evaluate Integrity
        integrity_status, integrity_checks = self._evaluate_integrity(
            run_dir=run_dir,
            run_id=actual_run_id,
            status_str=status_str,
            manifest_data=manifest_data,
        )

        return RunArtifactSnapshotV1(
            run_id=actual_run_id,
            run_dir=str(run_dir),
            status=status_str,
            run_kind=run_kind_str,
            canonical_run=canonical_run,
            started_at_utc=started_at,
            finished_at_utc=finished_at,
            recorded_episode_count=recorded_eps,
            expected_episode_count=expected_eps,
            agent_id=agent_id,
            agent_version=agent_version,
            input_profile_id=input_profile_id,
            action_adapter_id=action_adapter_id,
            inference_stochasticity=inference_stoch,
            stateful_within_episode=stateful_in_ep,
            agent_seed=agent_seed,
            manifest_name=manifest_name,
            experiment_config_sha256=cfg_sha,
            git_commit_sha=git_commit,
            git_worktree_dirty=git_dirty,
            dirty_override=dirty_override,
            unverified_env_override=unverified_env_override,
            environment_verification_status=env_verif_status,
            metadrive_version=md_ver,
            metadrive_commit=md_commit,
            metadrive_pinned_version=md_pin_ver,
            metadrive_pinned_commit=md_pin_commit,
            metadrive_verification_status=md_verif_status,
            metadrive_verification_reason=md_verif_reason,
            platform_contracts=platform_contracts,
            integrity_status=integrity_status,
            integrity_details=integrity_checks,
            artifacts_present=artifacts_present,
            summary_payload=summary_payload,
            episode_rows=episode_rows,
            timing_rows=timing_rows,
            wandb_sync_payload=wandb_payload,
            technical_failures_payload=tf_payload,
            failure_category=failure_category,
            sanitized_failure_message=sanitized_failure_message,
        )

    def _evaluate_integrity(
        self,
        run_dir: Path,
        run_id: str,
        status_str: str,
        manifest_data: Dict[str, Any],
    ) -> Tuple[IntegrityDisplayStatus, List[ArtifactIntegrityCheckResult]]:
        """
        Evaluates run integrity against run_integrity.json using Gate-7 canonical semantic hash helpers.
        Never rewrites or repairs mismatching hashes.
        """
        checks: List[ArtifactIntegrityCheckResult] = []

        # If run is not COMPLETE, final integrity is NOT_FINAL (not corrupted)
        if status_str != "COMPLETE":
            return IntegrityDisplayStatus.NOT_FINAL, checks

        integrity_path = run_dir / "run_integrity.json"
        if not integrity_path.exists():
            return IntegrityDisplayStatus.UNVERIFIED, checks

        try:
            with open(integrity_path, "r", encoding="utf-8") as f:
                stored = json.load(f)
            if not isinstance(stored, dict):
                return IntegrityDisplayStatus.FAILED, checks
        except Exception:
            return IntegrityDisplayStatus.FAILED, [
                ArtifactIntegrityCheckResult(
                    artifact_name="run_integrity.json",
                    expected_sha256=None,
                    observed_sha256=None,
                    status="FAIL",
                )
            ]

        # Verify run_id parity inside run_integrity.json
        stored_run_id = stored.get("run_id")
        run_id_match = (stored_run_id == run_id)
        checks.append(ArtifactIntegrityCheckResult(
            artifact_name="run_id",
            expected_sha256=stored_run_id,
            observed_sha256=run_id,
            status="PASS" if run_id_match else "FAIL",
        ))

        # Check experiment_config_sha256 parity against run manifest
        expected_cfg_hash = stored.get("experiment_config_sha256")
        cfg_dict = manifest_data.get("config", {})
        observed_cfg_hash = cfg_dict.get("experiment_config_sha256") or manifest_data.get("experiment_config_sha256")
        cfg_status = "PASS" if (expected_cfg_hash and expected_cfg_hash == observed_cfg_hash) else "FAIL"
        checks.append(ArtifactIntegrityCheckResult(
            artifact_name="experiment_config",
            expected_sha256=expected_cfg_hash,
            observed_sha256=observed_cfg_hash,
            status=cfg_status,
        ))

        # Check target artifacts via Gate-7 canonical semantic hash helpers
        target_files = [
            ("run_manifest.json", "run_manifest_sha256", "json"),
            ("episodes.csv", "episodes_sha256", "csv"),
            ("summary.json", "summary_sha256", "json"),
            ("timing.csv", "timing_sha256", "csv"),
            ("run_state.json", "run_state_sha256", "json"),
            ("wandb_sync.json", "wandb_sync_sha256", "json"),
        ]

        for fname, key, ftype in target_files:
            expected = stored.get(key)
            fpath = run_dir / fname
            if not fpath.exists():
                checks.append(ArtifactIntegrityCheckResult(
                    artifact_name=fname,
                    expected_sha256=expected,
                    observed_sha256=None,
                    status="MISSING",
                ))
                continue

            try:
                if ftype == "json":
                    observed = canonical_json_file_sha256(fpath)
                elif ftype == "csv":
                    observed = canonical_csv_file_sha256(fpath)
                else:
                    with open(fpath, "rb") as bf:
                        observed = hashlib.sha256(bf.read()).hexdigest()
            except Exception:
                observed = None

            c_status = "PASS" if (expected and expected == observed) else "FAIL"
            checks.append(ArtifactIntegrityCheckResult(
                artifact_name=fname,
                expected_sha256=expected,
                observed_sha256=observed,
                status=c_status,
            ))

        # Check technical_failures.jsonl if stored in record
        stored_tf_hash = stored.get("technical_failures_sha256")
        tf_path = run_dir / "technical_failures.jsonl"
        if stored_tf_hash is not None or tf_path.exists():
            if tf_path.exists():
                with open(tf_path, "rb") as bf:
                    obs_tf = hashlib.sha256(bf.read()).hexdigest()
            else:
                obs_tf = None
            tf_status = "PASS" if stored_tf_hash == obs_tf else "FAIL"
            checks.append(ArtifactIntegrityCheckResult(
                artifact_name="technical_failures.jsonl",
                expected_sha256=stored_tf_hash,
                observed_sha256=obs_tf,
                status=tf_status,
            ))

        all_passed = all(c.status == "PASS" for c in checks)
        overall_status = IntegrityDisplayStatus.VERIFIED if all_passed else IntegrityDisplayStatus.FAILED
        return overall_status, checks
