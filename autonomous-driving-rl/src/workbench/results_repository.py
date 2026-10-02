"""
Workbench Results Repository and Read-Only Artifact Access Layer (Gate 7.5B Pass B2).

This module provides the read-only inspection and integrity verification layer for Gate-7
experiment runs. It never mutates, appends, repairs, or recalculates official metrics.
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
    git_commit_sha: Optional[str]
    git_worktree_dirty: Optional[bool]
    integrity_status: IntegrityDisplayStatus
    integrity_details: List[ArtifactIntegrityCheckResult]
    artifacts_present: List[str]
    summary_payload: Optional[Dict[str, Any]]
    episode_rows: List[Dict[str, Any]]
    timing_rows: List[Dict[str, Any]]
    wandb_sync_payload: Optional[Dict[str, Any]]
    technical_failures_payload: Optional[List[Dict[str, Any]]]
    failure_category: Optional[str] = None
    failure_message: Optional[str] = None


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
                # A valid candidate run directory must at least have run_state.json or run_manifest.json
                if (entry / "run_state.json").exists() or (entry / "run_manifest.json").exists():
                    candidates.append(entry.name)

        return sorted(candidates)

    def load_run_snapshot(self, run_id: str) -> Optional[RunArtifactSnapshotV1]:
        """
        Loads a single run directory into a RunArtifactSnapshotV1 view model.
        Returns None if directory does not exist or is not a valid run folder.
        """
        run_dir = (self._runs_root / run_id).resolve()
        if not run_dir.exists() or not run_dir.is_dir():
            return None

        # 1. Inspect run_state.json (lifecycle authority)
        run_state_path = run_dir / "run_state.json"
        state_data: Dict[str, Any] = {}
        if run_state_path.exists():
            try:
                with open(run_state_path, "r", encoding="utf-8") as f:
                    state_data = json.load(f)
            except Exception:
                state_data = {}

        # 2. Inspect run_manifest.json (configuration authority)
        manifest_path = run_dir / "run_manifest.json"
        manifest_data: Dict[str, Any] = {}
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
            except Exception:
                manifest_data = {}

        # If neither run_state nor run_manifest could be parsed, not a valid run
        if not state_data and not manifest_data:
            return None

        actual_run_id = state_data.get("run_id") or manifest_data.get("run_id") or run_id
        status_str = state_data.get("status", "UNKNOWN")
        run_kind_str = manifest_data.get("run_kind", "UNKNOWN")
        canonical_run = bool(manifest_data.get("canonical_run", False))
        started_at = state_data.get("started_at_utc") or manifest_data.get("started_at_utc")
        finished_at = state_data.get("finished_at_utc")
        recorded_eps = int(state_data.get("recorded_episode_count", 0))

        config_data = manifest_data.get("config", {})
        expected_eps = config_data.get("expected_episode_count")
        if expected_eps is not None:
            expected_eps = int(expected_eps)

        agent_desc = config_data.get("agent_descriptor", {})
        agent_id = agent_desc.get("agent_id")
        agent_version = agent_desc.get("agent_version")

        prov_data = manifest_data.get("provenance", {})
        git_prov = prov_data.get("git", {})
        git_commit = git_prov.get("git_commit_sha")
        git_dirty = git_prov.get("git_worktree_dirty")

        failure_category = state_data.get("failure_category")
        failure_message = state_data.get("failure_message")

        # 3. Read summary.json if present
        summary_path = run_dir / "summary.json"
        summary_payload = None
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary_payload = json.load(f)
            except Exception:
                summary_payload = None

        # 4. Read episodes.csv rows if present
        episodes_path = run_dir / "episodes.csv"
        episode_rows = []
        if episodes_path.exists():
            try:
                with open(episodes_path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    episode_rows = list(reader)
            except Exception:
                episode_rows = []

        # 5. Read timing.csv rows if present
        timing_path = run_dir / "timing.csv"
        timing_rows = []
        if timing_path.exists():
            try:
                with open(timing_path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    timing_rows = list(reader)
            except Exception:
                timing_rows = []

        # 6. Read wandb_sync.json if present
        wandb_path = run_dir / "wandb_sync.json"
        wandb_payload = None
        if wandb_path.exists():
            try:
                with open(wandb_path, "r", encoding="utf-8") as f:
                    wandb_payload = json.load(f)
            except Exception:
                wandb_payload = None

        # 7. Read technical_failures.jsonl if present
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

        # 8. Evaluate Integrity
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
            git_commit_sha=git_commit,
            git_worktree_dirty=git_dirty,
            integrity_status=integrity_status,
            integrity_details=integrity_checks,
            artifacts_present=artifacts_present,
            summary_payload=summary_payload,
            episode_rows=episode_rows,
            timing_rows=timing_rows,
            wandb_sync_payload=wandb_payload,
            technical_failures_payload=tf_payload,
            failure_category=failure_category,
            failure_message=failure_message,
        )

    def _evaluate_integrity(
        self,
        run_dir: Path,
        run_id: str,
        status_str: str,
        manifest_data: Dict[str, Any],
    ) -> Tuple[IntegrityDisplayStatus, List[ArtifactIntegrityCheckResult]]:
        """
        Evaluates run integrity against run_integrity.json.
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
        except Exception:
            return IntegrityDisplayStatus.FAILED, [
                ArtifactIntegrityCheckResult(
                    artifact_name="run_integrity.json",
                    expected_sha256=None,
                    observed_sha256=None,
                    status="FAIL",
                )
            ]

        # Verify run_id parity
        if stored.get("run_id") != run_id:
            checks.append(ArtifactIntegrityCheckResult(
                artifact_name="run_id",
                expected_sha256=stored.get("run_id"),
                observed_sha256=run_id,
                status="FAIL",
            ))

        # Check experiment_config_sha256 parity against run manifest
        expected_cfg_hash = stored.get("experiment_config_sha256")
        cfg_dict = manifest_data.get("config", {})
        observed_cfg_hash = cfg_dict.get("experiment_config_sha256")
        cfg_status = "PASS" if (expected_cfg_hash and expected_cfg_hash == observed_cfg_hash) else "FAIL"
        checks.append(ArtifactIntegrityCheckResult(
            artifact_name="experiment_config",
            expected_sha256=expected_cfg_hash,
            observed_sha256=observed_cfg_hash,
            status=cfg_status,
        ))

        # Check target artifacts
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
