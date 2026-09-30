"""
Experiment Logging, Provenance, and Local Run Artifacts for Research Platform V1.
Gate 7 of Research Platform V1.

Core Mandates:
- LOCAL SCIENTIFIC RECORDS ARE AUTHORITATIVE.
- Single-source-of-truth: EpisodeLogRowV1 projected identically to CSV, events, and Tables.
- Provenance tracking: Git commit, dirty worktree status, platform contract hashes, seeds.
- Clean technical vs task failure segregation.
- Atomic local writes, durable run states (run_state.json), and duplicate run protection.
- High-resolution agent latency measurement excluding logging overhead.
"""

import copy
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np

from src.platform.metrics import (
    AggregateMetrics,
    EpisodeRecord,
    compute_aggregate_metrics,
)
from src.platform.protocol import (
    canonical_csv_file_sha256,
    canonical_json_file_sha256,
    canonical_json_sha256,
    compute_macro_metrics,
)


class RunKind(str, Enum):
    """Semantic classification of an experiment execution run."""
    TRAINING = "TRAINING"
    VALIDATION_EVALUATION = "VALIDATION_EVALUATION"
    TEST_EVALUATION = "TEST_EVALUATION"
    AUDIT = "AUDIT"


class RunStatus(str, Enum):
    """Lifecycle status of a local scientific experiment run."""
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class WandbMode(str, Enum):
    """Operational mode for Weights & Biases telemetry tracking."""
    DISABLED = "DISABLED"
    OFFLINE = "OFFLINE"
    ONLINE = "ONLINE"


class WandbSyncStatus(str, Enum):
    """Synchronization status with the remote W&B tracking backend."""
    DISABLED = "DISABLED"
    OFFLINE = "OFFLINE"
    ONLINE = "ONLINE"
    SYNCED = "SYNCED"
    FAILED = "FAILED"


def get_utc_now_iso() -> str:
    """Returns ISO-8601 formatted UTC timestamp with timezone designation."""
    return datetime.now(timezone.utc).isoformat()


def sanitize_error_message(err: Any) -> str:
    """
    Centralized error sanitizer. Redacts personal machine paths, home directories,
    project root absolute paths, system temp paths, API keys, and authorization tokens.
    """
    if err is None:
        return ""
    msg = str(err)

    # Redact system temp directory first (before home directory prefix)
    try:
        import tempfile
        tmp_dir = tempfile.gettempdir()
        if tmp_dir and len(tmp_dir) > 3:
            p_tmp = re.compile(re.escape(tmp_dir), re.IGNORECASE)
            msg = p_tmp.sub("<temp_dir>", msg)
            p_tmp_posix = re.compile(re.escape(tmp_dir.replace("\\", "/")), re.IGNORECASE)
            msg = p_tmp_posix.sub("<temp_dir>", msg)
    except Exception:
        pass

    # Redact home directory
    try:
        home_path = str(Path.home())
        if home_path and len(home_path) > 3:
            p_home = re.compile(re.escape(home_path), re.IGNORECASE)
            msg = p_home.sub("~", msg)
            p_home_posix = re.compile(re.escape(home_path.replace("\\", "/")), re.IGNORECASE)
            msg = p_home_posix.sub("~", msg)
    except Exception:
        pass

    # Redact project and workspace root paths
    try:
        root_path = str(Path(__file__).resolve().parent.parent.parent)
        if root_path and len(root_path) > 3:
            p_root = re.compile(re.escape(root_path), re.IGNORECASE)
            msg = p_root.sub("<project_root>", msg)
            p_root_posix = re.compile(re.escape(root_path.replace("\\", "/")), re.IGNORECASE)
            msg = p_root_posix.sub("<project_root>", msg)
    except Exception:
        pass

    # Redact actual in-memory WANDB_API_KEY if present in environment
    env_key = os.environ.get("WANDB_API_KEY")
    if env_key and len(env_key) > 5:
        msg = msg.replace(env_key, "[REDACTED_API_KEY]")

    # Redact secret patterns
    msg = re.sub(r"wandb_v1_[a-zA-Z0-9_\-]+", "[REDACTED_API_KEY]", msg)
    msg = re.sub(r"(api_key=)[a-zA-Z0-9_\-]+", r"\1[REDACTED]", msg, flags=re.IGNORECASE)
    msg = re.sub(r"(Bearer\s+)[a-zA-Z0-9_\.\-]+", r"\1[REDACTED_TOKEN]", msg, flags=re.IGNORECASE)
    msg = re.sub(r"(key\s*[:=]\s*)[a-zA-Z0-9_\-]{20,}", r"\1[REDACTED]", msg, flags=re.IGNORECASE)
    return msg


def capture_git_provenance(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Captures git commit SHA, branch, and dirty working tree status safely.
    Excludes unrelated secrets and personal machine paths.
    """
    root = repo_root or Path(__file__).resolve().parent.parent.parent
    try:
        commit_res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        commit_sha = commit_res.stdout.strip() if commit_res.returncode == 0 else "unknown"

        branch_res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        branch = branch_res.stdout.strip() if branch_res.returncode == 0 else "unknown"

        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        dirty_lines = [l for l in status_res.stdout.strip().splitlines() if l.strip()]
        is_dirty = len(dirty_lines) > 0

        diff_sha = None
        if is_dirty:
            diff_res = subprocess.run(
                ["git", "diff"],
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5
            )
            diff_sha = hashlib.sha256(diff_res.stdout.encode("utf-8")).hexdigest()

        return {
            "git_commit_sha": commit_sha,
            "git_branch": branch,
            "git_worktree_dirty": is_dirty,
            "git_diff_sha256": diff_sha
        }
    except Exception as e:
        return {
            "git_commit_sha": "unknown",
            "git_branch": "unknown",
            "git_worktree_dirty": True,
            "git_diff_sha256": None,
            "error": sanitize_error_message(e)
        }


def capture_environment_provenance(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Captures portable, non-private software dependency versions including MetaDrive commit."""
    import importlib.metadata
    env_info = {
        "python_version": platform.python_version(),
        "os_platform": platform.platform(),
        "os_family": platform.system(),
        "numpy_version": np.__version__,
    }

    try:
        env_info["metadrive_version"] = importlib.metadata.version("metadrive-simulator")
    except Exception:
        try:
            import metadrive.version
            env_info["metadrive_version"] = metadrive.version.VERSION
        except Exception:
            env_info["metadrive_version"] = "unknown"

    commit = "unknown"
    try:
        import metadrive
        f = Path(metadrive.__file__).resolve()
        m_root = f.parent
        while m_root.parent != m_root:
            if (m_root / ".git").exists():
                break
            m_root = m_root.parent

        if (m_root / ".git").exists():
            res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=m_root, stdout=subprocess.PIPE, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                commit = res.stdout.strip()
    except Exception:
        pass

    env_info["metadrive_commit"] = commit

    # wandb SDK version
    try:
        import wandb
        env_info["wandb_version"] = wandb.__version__
    except ImportError:
        env_info["wandb_version"] = "not_installed"

    # Optional torch/cuda (only if installed)
    try:
        import torch
        env_info["torch_version"] = torch.__version__
        env_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env_info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return env_info


PINNED_METADRIVE_VERSION = "0.4.3"
PINNED_METADRIVE_COMMIT = "85e5dadc6c7436d324348f6e3d8f8e680c06b4db"

WANDB_TABLE_COLUMNS: List[str] = [
    "episode_index", "protocol_order_index", "case_id", "split", "tier",
    "sequence", "geometry_generation_seed", "environment_seed",
    "clean_success", "final_route_completion", "max_route_completion",
    "primary_terminal_reason", "episode_steps", "episode_time_s",
    "time_to_clean_success_s", "mean_act_ms", "episode_return"
]

WANDB_STATIC_EPISODE_EVENT_KEYS: List[str] = [
    "eval/episode_index",
    "eval/protocol_order_index",
    "eval/clean_success",
    "eval/raw_arrival",
    "eval/final_route_completion",
    "eval/max_route_completion",
    "eval/episode_steps",
    "eval/episode_time_s",
    "diagnostic/episode_return",
    "eval/mean_speed_kmh",
    "eval/max_speed_kmh",
]

WANDB_TIMING_EVENT_KEYS: List[str] = [
    "timing/agent_act_mean_ms",
    "timing/agent_act_median_ms",
    "timing/agent_act_p95_ms",
    "timing/agent_act_max_ms",
    "timing/agent_act_total_ms",
]

WANDB_DYNAMIC_KEY_TEMPLATES: Dict[str, str] = {
    "primary_terminal_reason": "failure/<primary_terminal_reason>",
    "safety_flags": "flags/<flag>",
    "technical_failure": "technical_failure/<reason>",
}

WANDB_SUMMARY_MAPPING_RULES: Dict[str, str] = {
    "overall": "metrics/overall/<metric>",
    "tier": "metrics/tier/<tier>/<metric>",
    "macro": "metrics/macro/<metric>",
    "diagnostic": "diagnostic/<metric>",
    "timing": "timing/<metric>",
}

TECHNICAL_FAILURE_CATEGORIES = (
    "AGENT_EXCEPTION",
    "INVALID_AGENT_ACTION",
    "ACTION_ADAPTER_ERROR",
    "INVALID_AGENT_INPUT_CONSUMPTION",
    "LOCAL_LOG_WRITE_ERROR",
    "WANDB_INIT_ERROR",
    "WANDB_LOG_ERROR",
    "WANDB_SYNC_ERROR",
)


def sanitize_technical_failure_payload(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Validates and sanitizes technical failure payloads before local persistence.
    Enforces bounded stable failure category ('reason') and redacts free-text messages.
    """
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise TypeError(f"technical_failure must be a dictionary, got {type(payload).__name__}")

    raw_reason = payload.get("reason", "")
    reason_str = raw_reason.value if hasattr(raw_reason, "value") else str(raw_reason)

    if reason_str not in TECHNICAL_FAILURE_CATEGORIES:
        raise ValueError(
            f"Invalid technical failure category '{reason_str}'. "
            f"Must be one of {list(TECHNICAL_FAILURE_CATEGORIES)}"
        )

    msg = payload.get("message") or payload.get("details") or ""
    sanitized_msg = sanitize_error_message(msg)

    sanitized: Dict[str, Any] = {
        "reason": reason_str,
        "message": sanitized_msg
    }

    for k in ("episode_index", "step_index", "timestamp_utc"):
        if k in payload:
            sanitized[k] = payload[k]

    return sanitized


EPISODE_CSV_COLUMNS = [
    "episode_index", "protocol_order_index", "case_id", "split", "tier",
    "sequence", "geometry_generation_seed", "environment_seed", "horizon_steps", "agent_seed",
    "primary_terminal_reason", "terminal_reason",
    "clean_success", "raw_arrival", "final_route_completion",
    "max_route_completion", "episode_steps", "episode_time_s",
    "time_to_clean_success_s", "mean_speed_kmh", "max_speed_kmh",
    "episode_return", "crash_human", "crash_vehicle", "crash_object",
    "crash_building", "crash_sidewalk", "out_of_road", "timeout",
    "has_technical_failure", "technical_failure_reason"
]


@dataclass(frozen=True)
class EpisodeLogRowV1:
    """
    Canonical, single-source-of-truth data row for one evaluation episode.
    Projected identically to:
      1. episodes.csv row
      2. W&B step event stream
      3. W&B evaluation_episodes Table row
    """
    episode_index: int
    protocol_order_index: int
    case_id: str
    split: str
    tier: str
    sequence: str
    geometry_generation_seed: int
    environment_seed: int
    horizon_steps: int
    agent_seed: Optional[int]
    primary_terminal_reason: str
    terminal_reason: str
    clean_success: bool
    raw_arrival: bool
    final_route_completion: float
    max_route_completion: float
    episode_steps: int
    episode_time_s: float
    time_to_clean_success_s: Optional[float]
    mean_speed_kmh: float
    max_speed_kmh: float
    episode_return: float
    crash_human: bool
    crash_vehicle: bool
    crash_object: bool
    crash_building: bool
    crash_sidewalk: bool
    out_of_road: bool
    timeout: bool
    has_technical_failure: bool
    technical_failure_reason: str

    @classmethod
    def from_episode(
        cls,
        record: EpisodeRecord,
        episode_index: int,
        protocol_order_index: int,
        case_id: str,
        split: str,
        geometry_generation_seed: int,
        environment_seed: int,
        horizon_steps: int,
        agent_seed: Optional[int] = None,
        technical_failure: Optional[Dict[str, Any]] = None,
        is_benchmark_eval: bool = True
    ) -> "EpisodeLogRowV1":
        # Strict validation for benchmark evaluation runs
        if is_benchmark_eval:
            if not case_id or not isinstance(case_id, str) or not case_id.strip():
                raise ValueError(f"Benchmark evaluation requires explicit non-empty case_id, got '{case_id}'")
            if split not in ("TRAIN", "VALIDATION", "TEST"):
                raise ValueError(f"Invalid benchmark split '{split}' for episode {episode_index}")
            if not isinstance(geometry_generation_seed, (int, np.integer)):
                raise ValueError(f"geometry_generation_seed must be an integer, got {geometry_generation_seed}")
            if not isinstance(environment_seed, (int, np.integer)):
                raise ValueError(f"environment_seed must be an integer, got {environment_seed}")
            if not isinstance(horizon_steps, (int, np.integer)) or horizon_steps <= 0:
                raise ValueError(f"horizon_steps must be positive integer, got {horizon_steps}")
            if not isinstance(protocol_order_index, (int, np.integer)) or protocol_order_index <= 0:
                raise ValueError(f"protocol_order_index must be positive integer, got {protocol_order_index}")
            if not isinstance(episode_index, (int, np.integer)) or episode_index <= 0:
                raise ValueError(f"episode_index must be positive integer, got {episode_index}")

        # Explicit agent_seed semantics: preserve 0 as valid seed, never treat 0 as None
        assigned_agent_seed = None
        if record.agent_seed is not None:
            assigned_agent_seed = int(record.agent_seed)
        elif agent_seed is not None:
            assigned_agent_seed = int(agent_seed)

        timeout = getattr(record, "truncated", False) or getattr(record, "timeout", False)
        prim_reason = record.primary_reason.value if hasattr(record.primary_reason, "value") else str(record.primary_reason)
        sanitized_tf = sanitize_technical_failure_payload(technical_failure)

        return cls(
            episode_index=int(episode_index),
            protocol_order_index=int(protocol_order_index),
            case_id=str(case_id),
            split=str(split),
            tier=record.tier,
            sequence=record.sequence,
            geometry_generation_seed=int(geometry_generation_seed),
            environment_seed=int(environment_seed),
            horizon_steps=int(horizon_steps),
            agent_seed=assigned_agent_seed,
            primary_terminal_reason=prim_reason,
            terminal_reason=prim_reason,
            clean_success=bool(record.clean_success),
            raw_arrival=bool(record.raw_arrival),
            final_route_completion=float(record.final_route_completion),
            max_route_completion=float(record.max_route_completion),
            episode_steps=int(record.episode_steps),
            episode_time_s=float(record.simulation_time_s),
            time_to_clean_success_s=float(record.time_to_clean_success_s) if record.time_to_clean_success_s is not None else None,
            mean_speed_kmh=float(record.mean_speed_kmh),
            max_speed_kmh=float(record.max_speed_kmh),
            episode_return=float(record.episode_return),
            crash_human=bool(getattr(record, "raw_crash_human", False)),
            crash_vehicle=bool(getattr(record, "raw_crash_vehicle", False)),
            crash_object=bool(getattr(record, "raw_crash_object", False)),
            crash_building=bool(getattr(record, "raw_crash_building", False)),
            crash_sidewalk=bool(getattr(record, "raw_crash_sidewalk", False)),
            out_of_road=bool(getattr(record, "raw_out_of_road", False)),
            timeout=bool(timeout),
            has_technical_failure=sanitized_tf is not None,
            technical_failure_reason=sanitized_tf["reason"] if sanitized_tf else ""
        )

    def to_csv_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["time_to_clean_success_s"] is None:
            d["time_to_clean_success_s"] = ""
        if d["agent_seed"] is None:
            d["agent_seed"] = ""
        return {col: d[col] for col in EPISODE_CSV_COLUMNS}

    def to_wandb_event(self, timing: Optional["EpisodeTimingRecord"] = None) -> Dict[str, Any]:
        event = {
            "eval/episode_index": self.episode_index,
            "eval/protocol_order_index": self.protocol_order_index,
            "eval/clean_success": 1.0 if self.clean_success else 0.0,
            "eval/raw_arrival": 1.0 if self.raw_arrival else 0.0,
            "eval/final_route_completion": self.final_route_completion,
            "eval/max_route_completion": self.max_route_completion,
            "eval/episode_steps": self.episode_steps,
            "eval/episode_time_s": self.episode_time_s,
            "diagnostic/episode_return": self.episode_return,
            "eval/mean_speed_kmh": self.mean_speed_kmh,
            "eval/max_speed_kmh": self.max_speed_kmh,
            f"failure/{self.primary_terminal_reason}": 1.0,
        }
        for flag in ("crash_human", "crash_vehicle", "crash_object", "crash_building", "crash_sidewalk", "out_of_road"):
            if getattr(self, flag):
                event[f"flags/{flag}"] = 1.0
        if self.timeout:
            event["flags/timeout"] = 1.0

        if timing:
            event["timing/agent_act_mean_ms"] = timing.mean_act_ms
            event["timing/agent_act_median_ms"] = timing.median_act_ms
            event["timing/agent_act_p95_ms"] = timing.p95_act_ms
            event["timing/agent_act_max_ms"] = timing.max_act_ms
            event["timing/agent_act_total_ms"] = timing.total_act_ms

        if self.has_technical_failure:
            event["technical_failure/occurred"] = 1.0
            event[f"technical_failure/{self.technical_failure_reason}"] = 1.0

        return event

    def to_wandb_table_row(self, timing: Optional["EpisodeTimingRecord"] = None) -> List[Any]:
        mean_t = timing.mean_act_ms if timing is not None else None
        return [
            self.episode_index,
            self.protocol_order_index,
            self.case_id,
            self.split,
            self.tier,
            self.sequence,
            self.geometry_generation_seed,
            self.environment_seed,
            self.clean_success,
            self.final_route_completion,
            self.max_route_completion,
            self.primary_terminal_reason,
            self.episode_steps,
            self.episode_time_s,
            self.time_to_clean_success_s,
            mean_t,
            self.episode_return
        ]


@dataclass(frozen=True)
class ExperimentRunConfig:
    """
    Semantic, reproducible configuration for an experiment run.
    Contains platform contract hashes across Gates 5, 6, and 7.
    """
    run_kind: RunKind
    benchmark_contract_sha256: str
    agent_contract_sha256: str
    platform_runtime_contract_sha256: str
    logging_contract_sha256: str
    platform_observability_contract_sha256: str
    agent_id: str
    agent_version: str
    input_profile_id: str
    action_adapter_id: str
    inference_stochasticity: str
    stateful_within_episode: bool
    agent_seed: Optional[int] = None
    training_run_seed: Optional[int] = None
    protocol_order_seed: int = 424242
    manifest_name: str = "test_case_manifest.csv"
    expected_episode_count: Optional[int] = None
    allow_dirty_worktree_override: bool = False
    allow_unverified_env_override: bool = False
    wandb_mode: WandbMode = WandbMode.DISABLED
    wandb_project: str = "sgu-autonomous-driving-rl"
    wandb_entity: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    algorithm_hyperparameters: Dict[str, Any] = field(default_factory=dict)

    def compute_config_sha256(self) -> str:
        """Computes canonical SHA-256 fingerprint of semantic configuration."""
        d = {
            "run_kind": self.run_kind.value,
            "benchmark_contract_sha256": self.benchmark_contract_sha256,
            "agent_contract_sha256": self.agent_contract_sha256,
            "platform_runtime_contract_sha256": self.platform_runtime_contract_sha256,
            "logging_contract_sha256": self.logging_contract_sha256,
            "platform_observability_contract_sha256": self.platform_observability_contract_sha256,
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "input_profile_id": self.input_profile_id,
            "action_adapter_id": self.action_adapter_id,
            "inference_stochasticity": self.inference_stochasticity,
            "stateful_within_episode": self.stateful_within_episode,
            "agent_seed": self.agent_seed,
            "training_run_seed": self.training_run_seed,
            "protocol_order_seed": self.protocol_order_seed,
            "manifest_name": self.manifest_name,
            "expected_episode_count": self.expected_episode_count,
            "allow_dirty_worktree_override": self.allow_dirty_worktree_override,
            "allow_unverified_env_override": self.allow_unverified_env_override,
            "tags": sorted(self.tags),
            "algorithm_hyperparameters": self.algorithm_hyperparameters
        }
        return canonical_json_sha256(d)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["run_kind"] = self.run_kind.value
        data["wandb_mode"] = self.wandb_mode.value
        data["experiment_config_sha256"] = self.compute_config_sha256()
        return data


@dataclass(frozen=True)
class RunManifestV1:
    """
    Immutable root provenance manifest created at experiment initialization.
    Saved to runs/<run_id>/run_manifest.json.
    """
    run_id: str
    run_kind: str
    created_at_utc: str
    experiment_config_sha256: str
    config: Dict[str, Any]
    git_provenance: Dict[str, Any]
    environment_provenance: Dict[str, Any]
    platform_contracts: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunStateV1:
    """Authoritative durable lifecycle state persisted to runs/<run_id>/run_state.json."""
    run_id: str
    status: str
    started_at_utc: str
    updated_at_utc: str
    finished_at_utc: Optional[str]
    recorded_episode_count: int
    expected_episode_count: Optional[int]
    canonical_run: bool
    dirty_override: bool
    unverified_env_override: bool = False
    environment_verification_status: str = "VERIFIED"
    failure_category: Optional[str] = None
    sanitized_failure_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeTimingRecord:
    """High-resolution timing statistics preserving raw Python floating-point precision."""
    episode_index: int
    act_count: int
    mean_act_ms: float
    median_act_ms: float
    p95_act_ms: float
    max_act_ms: float
    total_act_ms: float
    latency_sync_policy: str = "NONE"

    @classmethod
    def from_latencies(
        cls,
        episode_index: int,
        latencies_ms: List[float],
        latency_sync_policy: str = "NONE"
    ) -> "EpisodeTimingRecord":
        if not latencies_ms:
            return cls(
                episode_index=episode_index,
                act_count=0,
                mean_act_ms=0.0,
                median_act_ms=0.0,
                p95_act_ms=0.0,
                max_act_ms=0.0,
                total_act_ms=0.0,
                latency_sync_policy=latency_sync_policy
            )
        arr = np.asarray(latencies_ms, dtype=np.float64)
        return cls(
            episode_index=episode_index,
            act_count=len(latencies_ms),
            mean_act_ms=float(np.mean(arr)),
            median_act_ms=float(np.median(arr)),
            p95_act_ms=float(np.percentile(arr, 95)),
            max_act_ms=float(np.max(arr)),
            total_act_ms=float(np.sum(arr)),
            latency_sync_policy=latency_sync_policy
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunIntegrityRecord:
    """Cryptographic content fingerprints for local scientific run artifacts."""
    run_id: str
    experiment_config_sha256: str
    run_manifest_sha256: str
    episodes_sha256: str
    summary_sha256: str
    timing_sha256: str
    run_state_sha256: str
    wandb_sync_sha256: str
    technical_failures_sha256: Optional[str]
    finalized_at_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LocalExperimentLogger:
    """
    Authoritative local-first experiment persistence engine.
    Ensures single canonical EpisodeLogRowV1 projection, atomic writes,
    durable run state tracking, and duplicate run protection.
    """
    def __init__(
        self,
        config: ExperimentRunConfig,
        runs_root: Optional[Path] = None,
        custom_run_id: Optional[str] = None,
        repo_root: Optional[Path] = None,
        custom_environment_provenance: Optional[Dict[str, Any]] = None
    ):
        self.config = config
        self.repo_root = repo_root or Path(__file__).resolve().parent.parent.parent
        self.runs_root = runs_root or (self.repo_root / "autonomous-driving-rl" / "runs")
        self.runs_root.mkdir(parents=True, exist_ok=True)

        self.run_id = custom_run_id or f"run_{uuid.uuid4().hex[:12]}"
        self.run_dir = self.runs_root / self.run_id

        # Duplicate run protection
        if self.run_dir.exists():
            raise FileExistsError(
                f"Duplicate run error: run directory '{self.run_dir}' already exists! "
                "Overwriting or appending to existing scientific runs is strictly forbidden."
            )
        self.run_dir.mkdir(parents=True, exist_ok=False)

        self.status = RunStatus.INITIALIZING
        self.started_at_utc = get_utc_now_iso()
        self.recorded_rows: List[EpisodeLogRowV1] = []
        self.recorded_episodes: List[EpisodeRecord] = []
        self.recorded_timings: List[EpisodeTimingRecord] = []
        self.technical_failures: List[Dict[str, Any]] = []

        # Local file paths
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.run_state_path = self.run_dir / "run_state.json"
        self.episodes_csv_path = self.run_dir / "episodes.csv"
        self.summary_json_path = self.run_dir / "summary.json"
        self.timing_csv_path = self.run_dir / "timing.csv"
        self.failures_jsonl_path = self.run_dir / "technical_failures.jsonl"
        self.integrity_json_path = self.run_dir / "run_integrity.json"
        self.wandb_sync_json_path = self.run_dir / "wandb_sync.json"

        # Dirty worktree and environment verification
        git_prov = capture_git_provenance(self.repo_root)
        env_prov = custom_environment_provenance if custom_environment_provenance is not None else capture_environment_provenance(self.repo_root)
        is_dirty = bool(git_prov.get("git_worktree_dirty"))
        is_benchmark_eval = self.config.run_kind in (RunKind.VALIDATION_EVALUATION, RunKind.TEST_EVALUATION)

        self.canonical_run = not is_dirty
        self.dirty_override = False
        self.unverified_env_override = False
        self.environment_verification_status = "UNKNOWN"
        self.environment_verification_reason = ""

        if is_dirty and is_benchmark_eval:
            if not self.config.allow_dirty_worktree_override:
                self.status = RunStatus.FAILED
                self._persist_run_state(
                    failure_category="DIRTY_WORKTREE_ERROR",
                    failure_message="Cannot execute benchmark evaluation on dirty git working tree without explicit override."
                )
                raise RuntimeError(
                    "Cannot execute benchmark evaluation on dirty git working tree! "
                    "Set allow_dirty_worktree_override=True to override."
                )
            self.canonical_run = False
            self.dirty_override = True

        # MetaDrive exact pin verification
        actual_version = env_prov.get("metadrive_version", "unknown")
        actual_commit = env_prov.get("metadrive_commit", "unknown")

        if actual_commit == "unknown":
            env_status = "UNKNOWN"
            env_reason = "MetaDrive commit unknown / git repository not detected"
        elif actual_version != PINNED_METADRIVE_VERSION:
            env_status = "MISMATCHED"
            env_reason = f"MetaDrive version mismatch: expected {PINNED_METADRIVE_VERSION}, got {actual_version}"
        elif actual_commit != PINNED_METADRIVE_COMMIT:
            env_status = "MISMATCHED"
            env_reason = f"MetaDrive commit mismatch: expected {PINNED_METADRIVE_COMMIT}, got {actual_commit}"
        else:
            env_status = "VERIFIED"
            env_reason = f"MetaDrive exact pin verified: version {PINNED_METADRIVE_VERSION}, commit {PINNED_METADRIVE_COMMIT}"

        env_prov["metadrive_pinned_version"] = PINNED_METADRIVE_VERSION
        env_prov["metadrive_pinned_commit"] = PINNED_METADRIVE_COMMIT
        env_prov["metadrive_verification_status"] = env_status
        env_prov["metadrive_verification_reason"] = env_reason

        self.environment_verification_status = env_status
        self.environment_verification_reason = env_reason
        self.unverified_env_override = False

        if is_benchmark_eval and env_status != "VERIFIED":
            if not self.config.allow_unverified_env_override:
                self.status = RunStatus.FAILED
                self._persist_run_state(
                    failure_category="UNVERIFIED_ENVIRONMENT_ERROR",
                    failure_message=f"Cannot execute canonical benchmark evaluation: {env_reason}."
                )
                raise RuntimeError(
                    f"Cannot execute canonical benchmark evaluation: {env_reason}. "
                    "Set allow_unverified_env_override=True to override."
                )
            self.canonical_run = False
            self.unverified_env_override = True

        # Construct and persist immutable RunManifestV1
        contracts_prov = {
            "benchmark_contract_sha256": self.config.benchmark_contract_sha256,
            "agent_contract_sha256": self.config.agent_contract_sha256,
            "platform_runtime_contract_sha256": self.config.platform_runtime_contract_sha256,
            "logging_contract_sha256": self.config.logging_contract_sha256,
            "platform_observability_contract_sha256": self.config.platform_observability_contract_sha256,
        }

        self.manifest = RunManifestV1(
            run_id=self.run_id,
            run_kind=self.config.run_kind.value,
            created_at_utc=self.started_at_utc,
            experiment_config_sha256=self.config.compute_config_sha256(),
            config=self.config.to_dict(),
            git_provenance=git_prov,
            environment_provenance=env_prov,
            platform_contracts=contracts_prov
        )

        self._atomic_write_json(self.manifest_path, self.manifest.to_dict())
        self._init_csv_files()
        self.status = RunStatus.RUNNING
        self._persist_run_state()

    def _persist_run_state(
        self,
        finished: bool = False,
        failure_category: Optional[str] = None,
        failure_message: Optional[str] = None
    ) -> None:
        """Atomically persists run_state.json ensuring durable post-process visibility."""
        now_utc = get_utc_now_iso()
        state = RunStateV1(
            run_id=self.run_id,
            status=self.status.value,
            started_at_utc=self.started_at_utc,
            updated_at_utc=now_utc,
            finished_at_utc=now_utc if finished else None,
            recorded_episode_count=len(self.recorded_rows),
            expected_episode_count=self.config.expected_episode_count,
            canonical_run=self.canonical_run,
            dirty_override=self.dirty_override,
            unverified_env_override=self.unverified_env_override,
            environment_verification_status=self.environment_verification_status,
            failure_category=failure_category,
            sanitized_failure_message=sanitize_error_message(failure_message) if failure_message else None
        )
        self._atomic_write_json(self.run_state_path, state.to_dict())

    def _atomic_write_json(self, target_path: Path, data: Any) -> None:
        """Atomically writes JSON by flushing to temporary file and renaming."""
        temp_path = target_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, target_path)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            self.status = RunStatus.FAILED
            sanitized_e = sanitize_error_message(e)
            raise IOError(f"Failed to atomically persist scientific record to '{target_path.name}': {sanitized_e}") from e

    def _init_csv_files(self) -> None:
        """Initializes empty CSV headers including protocol_order_index."""
        with open(self.episodes_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=EPISODE_CSV_COLUMNS)
            writer.writeheader()

        timing_headers = [
            "episode_index", "act_count", "mean_act_ms", "median_act_ms",
            "p95_act_ms", "max_act_ms", "total_act_ms", "latency_sync_policy"
        ]
        with open(self.timing_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=timing_headers)
            writer.writeheader()

    def log_episode(
        self,
        record: EpisodeRecord,
        timing: Optional[EpisodeTimingRecord] = None,
        technical_failure: Optional[Dict[str, Any]] = None,
        episode_index: Optional[int] = None,
        protocol_order_index: Optional[int] = None,
        case_id: Optional[str] = None,
        split: Optional[str] = None,
        environment_seed: Optional[Any] = None,
        geometry_generation_seed: Optional[Any] = None,
        horizon_steps: Optional[int] = None,
        agent_seed: Optional[int] = None
    ) -> EpisodeLogRowV1:
        """
        Constructs the single authoritative EpisodeLogRowV1 and appends to local files.
        Fails loudly if local write fails.
        """
        if self.status not in (RunStatus.RUNNING, RunStatus.INITIALIZING):
            raise RuntimeError(f"Cannot log episode to run in status '{self.status.value}'")

        is_benchmark_eval = self.config.run_kind in (RunKind.VALIDATION_EVALUATION, RunKind.TEST_EVALUATION)

        if is_benchmark_eval:
            # Enforce NO fabricated fallbacks for benchmark evaluation
            if episode_index is None or not isinstance(episode_index, (int, np.integer)) or int(episode_index) <= 0:
                raise ValueError(f"{self.config.run_kind.value} requires explicit positive episode_index, got '{episode_index}'!")
            if protocol_order_index is None or not isinstance(protocol_order_index, (int, np.integer)) or int(protocol_order_index) <= 0:
                raise ValueError(f"{self.config.run_kind.value} requires explicit positive protocol_order_index, got '{protocol_order_index}'!")
            if not case_id or not isinstance(case_id, str) or not case_id.strip():
                raise ValueError(f"{self.config.run_kind.value} requires explicit non-empty case_id, got '{case_id}'!")
            if split is None:
                raise ValueError(f"{self.config.run_kind.value} requires explicit split!")
            if self.config.run_kind == RunKind.TEST_EVALUATION and split != "TEST":
                raise ValueError(f"TEST_EVALUATION requires split=='TEST', got '{split}'")
            if self.config.run_kind == RunKind.VALIDATION_EVALUATION and split != "VALIDATION":
                raise ValueError(f"VALIDATION_EVALUATION requires split=='VALIDATION', got '{split}'")
            if geometry_generation_seed is None or not isinstance(geometry_generation_seed, (int, np.integer)):
                raise ValueError(f"{self.config.run_kind.value} requires explicit integer geometry_generation_seed, got '{geometry_generation_seed}'!")
            if environment_seed is None or not isinstance(environment_seed, (int, np.integer)):
                raise ValueError(f"{self.config.run_kind.value} requires explicit integer environment_seed, got '{environment_seed}'!")
            if horizon_steps is None or not isinstance(horizon_steps, (int, np.integer)) or int(horizon_steps) <= 0:
                raise ValueError(f"{self.config.run_kind.value} requires explicit positive horizon_steps, got '{horizon_steps}'!")

            next_ep_idx = int(episode_index)
            p_order_idx = int(protocol_order_index)
            c_id = str(case_id)
            s_split = str(split)
            geom_seed = int(geometry_generation_seed)
            env_seed = int(environment_seed)
            h_steps = int(horizon_steps)
        else:
            # Documented default policy for non-benchmark runs (AUDIT and TRAINING convenience defaults ONLY)
            next_ep_idx = int(episode_index) if episode_index is not None else len(self.recorded_rows) + 1
            p_order_idx = int(protocol_order_index) if protocol_order_index is not None else next_ep_idx
            c_id = str(case_id) if case_id is not None else f"audit_case_{next_ep_idx}"
            s_split = str(split) if split is not None else "AUDIT"
            geom_seed = int(geometry_generation_seed) if geometry_generation_seed is not None else getattr(record, "scenario_seed", 0)
            env_seed = int(environment_seed) if environment_seed is not None else 9101
            h_steps = int(horizon_steps) if horizon_steps is not None else getattr(record, "horizon_steps", 1000)

        # Agent seed with explicit None semantics (agent_seed=0 is preserved)
        effective_agent_seed = None
        if agent_seed is not None:
            effective_agent_seed = int(agent_seed)
        elif record.agent_seed is not None:
            effective_agent_seed = int(record.agent_seed)
        elif self.config.agent_seed is not None:
            effective_agent_seed = int(self.config.agent_seed)

        # Single source-of-truth construction
        log_row = EpisodeLogRowV1.from_episode(
            record=record,
            episode_index=next_ep_idx,
            protocol_order_index=p_order_idx,
            case_id=c_id,
            split=s_split,
            geometry_generation_seed=geom_seed,
            environment_seed=env_seed,
            horizon_steps=h_steps,
            agent_seed=effective_agent_seed,
            technical_failure=technical_failure,
            is_benchmark_eval=is_benchmark_eval
        )

        self.recorded_rows.append(log_row)
        self.recorded_episodes.append(record)
        if timing:
            self.recorded_timings.append(timing)

        # 1. Append episodes.csv
        try:
            with open(self.episodes_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=EPISODE_CSV_COLUMNS)
                writer.writerow(log_row.to_csv_dict())
                f.flush()
        except Exception as e:
            self.status = RunStatus.FAILED
            sanitized_e = sanitize_error_message(e)
            self._persist_run_state(failure_category="LOCAL_LOG_WRITE_ERROR", failure_message=sanitized_e)
            raise IOError(f"Critical local persistence failure in episodes.csv: {sanitized_e}") from e

        # 2. Append timing.csv
        if timing:
            try:
                with open(self.timing_csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(timing.to_dict().keys()))
                    writer.writerow(timing.to_dict())
                    f.flush()
            except Exception as e:
                self.status = RunStatus.FAILED
                sanitized_e = sanitize_error_message(e)
                self._persist_run_state(failure_category="LOCAL_LOG_WRITE_ERROR", failure_message=sanitized_e)
                raise IOError(f"Critical local persistence failure in timing.csv: {sanitized_e}") from e

        # 3. Log technical failure if present
        if technical_failure:
            sanitized_tf = sanitize_technical_failure_payload(technical_failure)
            self.technical_failures.append(sanitized_tf)
            try:
                with open(self.failures_jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(sanitized_tf, sort_keys=True) + "\n")
                    f.flush()
            except Exception as e:
                self.status = RunStatus.FAILED
                sanitized_e = sanitize_error_message(e)
                self._persist_run_state(failure_category="LOCAL_LOG_WRITE_ERROR", failure_message=sanitized_e)
                raise IOError(f"Critical local persistence failure in technical_failures.jsonl: {sanitized_e}") from e

        # Update durable run state
        self._persist_run_state()
        return log_row

    def prepare_summary(self) -> Dict[str, Any]:
        """
        Computes the single authoritative in-memory summary payload.
        Reuses Gate-4 aggregate metrics and Gate-5 macro calculations.
        """
        overall_metrics = compute_aggregate_metrics(self.recorded_episodes)

        tiers = ("Easy", "Medium", "Hard", "Extreme")
        tier_cards = {}
        for t in tiers:
            tier_eps = [e for e in self.recorded_episodes if getattr(e, "tier", None) == t]
            if tier_eps:
                tier_cards[t] = compute_aggregate_metrics(tier_eps)

        macro_metrics = None
        if len(tier_cards) == 4:
            macro_metrics = compute_macro_metrics(tier_cards)

        total_acts = sum(t.act_count for t in self.recorded_timings)
        total_act_time_ms = sum(t.total_act_ms for t in self.recorded_timings)
        weighted_mean_act_ms = float(total_act_time_ms / total_acts) if total_acts > 0 else None
        max_act_ms = float(max([t.max_act_ms for t in self.recorded_timings])) if (self.recorded_timings and total_acts > 0) else None

        return {
            "run_id": self.run_id,
            "run_kind": self.config.run_kind.value,
            "total_episodes": len(self.recorded_rows),
            "finalized_at_utc": get_utc_now_iso(),
            "overall_metrics": overall_metrics.to_dict(),
            "tier_metrics": {t: card.to_dict() for t, card in tier_cards.items()},
            "macro_metrics": macro_metrics,
            "timing_overall": {
                "total_act_count": total_acts,
                "weighted_mean_act_ms": weighted_mean_act_ms,
                "max_act_ms": max_act_ms,
            },
            "technical_failure_count": len(self.technical_failures),
        }

    def finalize_run(
        self,
        *,
        summary_payload: Dict[str, Any],
        wandb_sync_info: Optional[Dict[str, Any]] = None
    ) -> RunIntegrityRecord:
        """
        Finalizes the experiment:
        1. Verifies expected episode count if predeclared.
        2. Atomically persists summary.json and wandb_sync.json.
        3. Computes semantic content hashes (including wandb_sync_sha256) and saves run_integrity.json.
        4. Updates run_state.json to COMPLETE.
        A run must NEVER remain COMPLETE if required integrity finalization fails.
        """
        if self.status == RunStatus.FAILED:
            raise RuntimeError(f"Cannot complete run '{self.run_id}': run status is already FAILED!")

        # Validate summary_payload structure to prevent passing sync_meta or invalid payload
        if not isinstance(summary_payload, dict) or "overall_metrics" not in summary_payload:
            raise ValueError(
                "finalize_run requires a valid summary_payload dictionary containing 'overall_metrics'!"
            )

        # Verify expected episode count
        if self.config.expected_episode_count is not None:
            if len(self.recorded_rows) != self.config.expected_episode_count:
                self.status = RunStatus.FAILED
                msg = f"Episode count mismatch: expected {self.config.expected_episode_count}, but recorded {len(self.recorded_rows)}"
                self._persist_run_state(failure_category="EPISODE_COUNT_MISMATCH", failure_message=msg)
                raise ValueError(msg)

        try:
            # Authoritative summary: strictly consume passed summary_payload
            summary = summary_payload

            # Atomically write summary.json
            self._atomic_write_json(self.summary_json_path, summary)

            # Save wandb_sync.json
            sync_data = wandb_sync_info or {
                "mode": self.config.wandb_mode.value,
                "sync_status": WandbSyncStatus.DISABLED.value if self.config.wandb_mode == WandbMode.DISABLED else WandbSyncStatus.OFFLINE.value,
                "wandb_run_id": None,
                "wandb_run_url": None,
                "configured_project": self.config.wandb_project,
                "configured_entity": self.config.wandb_entity,
                "resolved_project": None,
                "resolved_entity": None
            }
            self._atomic_write_json(self.wandb_sync_json_path, sync_data)

            # Stage status as COMPLETE for state hashing
            self.status = RunStatus.COMPLETE
            self._persist_run_state(finished=True)

            # Compute semantic integrity fingerprints
            manifest_hash = canonical_json_file_sha256(self.manifest_path)
            episodes_hash = canonical_csv_file_sha256(self.episodes_csv_path)
            summary_hash = canonical_json_file_sha256(self.summary_json_path)
            timing_hash = canonical_csv_file_sha256(self.timing_csv_path)
            run_state_hash = canonical_json_file_sha256(self.run_state_path)
            wandb_sync_hash = canonical_json_file_sha256(self.wandb_sync_json_path)
            failures_hash = None
            if self.failures_jsonl_path.exists():
                with open(self.failures_jsonl_path, "rb") as f:
                    failures_hash = hashlib.sha256(f.read()).hexdigest()

            integrity_record = RunIntegrityRecord(
                run_id=self.run_id,
                experiment_config_sha256=self.config.compute_config_sha256(),
                run_manifest_sha256=manifest_hash,
                episodes_sha256=episodes_hash,
                summary_sha256=summary_hash,
                timing_sha256=timing_hash,
                run_state_sha256=run_state_hash,
                wandb_sync_sha256=wandb_sync_hash,
                technical_failures_sha256=failures_hash,
                finalized_at_utc=get_utc_now_iso()
            )

            # Atomically persist run_integrity.json
            self._atomic_write_json(self.integrity_json_path, integrity_record.to_dict())
            return integrity_record

        except Exception as e:
            # Clean up partial / invalid integrity artifact if present
            if self.integrity_json_path.exists():
                try:
                    self.integrity_json_path.unlink(missing_ok=True)
                except Exception:
                    pass

            # A run must NEVER remain COMPLETE if required integrity finalization fails
            self.status = RunStatus.FAILED
            sanitized_err = sanitize_error_message(e)
            self._persist_run_state(
                finished=True,
                failure_category="RUN_INTEGRITY_FINALIZATION_ERROR",
                failure_message=sanitized_err
            )
            raise RuntimeError(
                f"Failed to complete run '{self.run_id}' during integrity finalization: {sanitized_err}"
            ) from e

    def mark_interrupted(self, reason: str = "Process interrupted") -> None:
        """Transitions run status to INTERRUPTED and persists run_state.json durably."""
        self.status = RunStatus.INTERRUPTED
        sanitized_msg = sanitize_error_message(reason)
        self._persist_run_state(
            finished=True,
            failure_category="INTERRUPTED",
            failure_message=sanitized_msg
        )


def build_logging_contract_core(
    status: str = "LOCKED-FOR-PLATFORM-V1",
    custom_rules: Optional[Dict[str, Any]] = None,
    custom_latency_schema: Optional[Dict[str, Any]] = None,
    custom_wandb_contract: Optional[Dict[str, Any]] = None,
    custom_secret_policy: Optional[Dict[str, Any]] = None,
    custom_policies: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Builds the authoritative, complete machine-readable LoggingContractV1 core dictionary.
    Locks local record schemas, metric namespaces, timing boundary, technical failure categories,
    and W&B mirror contracts for canonical hashing.
    """
    import dataclasses

    runtime_dataclasses = {
        "ExperimentRunConfig": [f.name for f in dataclasses.fields(ExperimentRunConfig)],
        "RunManifestV1": [f.name for f in dataclasses.fields(RunManifestV1)],
        "RunStateV1": [f.name for f in dataclasses.fields(RunStateV1)],
        "EpisodeLogRowV1": [f.name for f in dataclasses.fields(EpisodeLogRowV1)],
        "EpisodeTimingRecord": [f.name for f in dataclasses.fields(EpisodeTimingRecord)],
        "RunIntegrityRecord": [f.name for f in dataclasses.fields(RunIntegrityRecord)],
    }

    runtime_enums = {
        "RunKind": [e.value for e in RunKind],
        "RunStatus": [e.value for e in RunStatus],
        "WandbMode": [e.value for e in WandbMode],
        "WandbSyncStatus": [e.value for e in WandbSyncStatus],
    }

    episode_columns = list(EPISODE_CSV_COLUMNS)

    default_timing = {
        "start_event": "Immediately before agent.act(agent_input) is invoked",
        "stop_event": "Immediately after AgentDecision is returned",
        "timer": "high-resolution monotonic timer (perf_counter_ns)",
        "excluded_overhead": [
            "AgentInput assembly",
            "ActionAdapter conversion",
            "environment step (physics)",
            "rendering",
            "logging",
            "W&B I/O"
        ],
        "metrics": ["act_count", "mean_act_ms", "median_act_ms", "p95_act_ms", "max_act_ms", "total_act_ms"],
        "weighted_aggregation": "sum(total_act_ms) / sum(act_count)",
        "latency_sync_policies": ["NONE", "FRAMEWORK_SYNCHRONIZED"]
    }
    timing_schema = dict(default_timing)
    if custom_latency_schema:
        timing_schema.update(custom_latency_schema)

    default_wandb = {
        "modes": [e.value for e in WandbMode],
        "failure_policy": "NON-FATAL to local scientific experiment if local logging is healthy",
        "pinned_wandb_version": "0.30.0",
        "upgrade_policy": "Any subsequent W&B SDK upgrade requires explicit Gate-7 lifecycle and teardown re-audit.",
        "table_name": "evaluation_episodes",
        "table_columns": list(WANDB_TABLE_COLUMNS),
        "step_axis": "episode_index",
        "episode_event_namespaces": ["eval", "flags", "diagnostic", "technical_failure", "timing"],
        "static_episode_event_keys": list(WANDB_STATIC_EPISODE_EVENT_KEYS),
        "timing_event_keys": list(WANDB_TIMING_EVENT_KEYS),
        "dynamic_event_key_templates": dict(WANDB_DYNAMIC_KEY_TEMPLATES),
        "summary_namespaces": ["metrics/overall", "metrics/tier", "metrics/macro", "diagnostic", "timing"],
        "summary_mapping_rules": dict(WANDB_SUMMARY_MAPPING_RULES),
        "privacy_settings": {
            "save_code": False,
            "disable_git": True,
            "notes": "Code and git provenance captured exclusively through local immutable RunManifestV1."
        }
    }
    wandb_contract = dict(default_wandb)
    if custom_wandb_contract:
        wandb_contract.update(custom_wandb_contract)

    default_secrets = {
        "credential_source": "WANDB_API_KEY environment variable",
        "persistence_rule": "STRICTLY PROHIBITED in files, configs, traces, or commits",
        "echo_rule": "STRICTLY PROHIBITED in console or markdown artifacts"
    }
    secret_policy = dict(default_secrets)
    if custom_secret_policy:
        secret_policy.update(custom_secret_policy)

    default_policies = {
        "experiment_run_granularity": "ONE_RUN_PER_EVALUATION_SUITE",
        "one_wandb_run_per_experiment_run": True,
        "run_lifecycle_transitions": [
            "INITIALIZING -> RUNNING -> COMPLETE",
            "INITIALIZING / RUNNING -> FAILED",
            "INITIALIZING / RUNNING -> INTERRUPTED"
        ],
        "dirty_worktree_policy": "FAIL_BY_DEFAULT_FOR_BENCHMARK_OVERRIDE_MARKS_NONCANONICAL",
        "canonical_run_policy": "CANONICAL_REQUIRES_CLEAN_WORKTREE_VERIFIED_PINS_LOCKED_CONTRACTS",
        "resume_policy": "DISABLED",
        "duplicate_run_policy": "FAIL_LOUDLY_NO_OVERWRITE_NO_APPEND",
        "local_failure_policy": "FATAL",
        "wandb_failure_policy": "NON_FATAL_IF_LOCAL_HEALTHY",
        "backend_input_rule": "EPISODE_LOG_ROW_V1_ONLY",
        "interruption_policy": "GRACEFUL_TRANSITION_TO_INTERRUPTED_UNEXPECTED_DEATH_LEAVES_RUNNING"
    }
    policies = dict(default_policies)
    if custom_policies:
        policies.update(custom_policies)

    return {
        "metadata": {
            "contract_name": "LoggingContractV1",
            "spec_version": "1.0.0",
            "status": status,
            "gate": "Gate 7 (Research Platform V1)"
        },
        "operational_policies": policies,
        "local_record_contract": {
            "root_directory_pattern": "runs/<run_id>",
            "duplicate_run_protection": "fail loudly if run_id directory exists",
            "atomic_write_strategy": "write to .tmp file, flush, fsync, atomic replace",
            "authoritative_source": "LOCAL",
            "remote_sync_source": "DOWNSTREAM_MIRROR",
            "required_files": [
                "run_manifest.json",
                "run_state.json",
                "episodes.csv",
                "summary.json",
                "timing.csv",
                "run_integrity.json",
                "wandb_sync.json"
            ],
            "optional_files": [
                "technical_failures.jsonl",
                "training_metrics.jsonl"
            ]
        },
        "runtime_dataclass_schemas": runtime_dataclasses,
        "runtime_enum_schemas": runtime_enums,
        "episode_schema": {
            "file": "episodes.csv",
            "columns": episode_columns,
            "step_axis": "episode_index",
            "null_metric_representation": "null in JSON, empty in CSV, None/omitted in W&B (never 0.0)"
        },
        "summary_metric_schema": {
            "primary_benchmark_metrics": [
                "clean_success_rate",
                "safety_failure_rate",
                "mean_final_route_completion",
                "median_final_route_completion",
                "mean_time_to_clean_success_s"
            ],
            "per_tier_namespaces": "metrics/tier/<tier>/<metric>",
            "overall_namespace": "metrics/overall/<metric>",
            "macro_namespace": "metrics/macro/<metric>",
            "diagnostic_classification": {
                "episode_return": "diagnostic/episode_return (never primary benchmark ranking score)"
            },
            "mega_score_policy": "PROHIBITED"
        },
        "timing_schema": timing_schema,
        "technical_failure_schema": {
            "file": "technical_failures.jsonl",
            "format": "JSON Lines",
            "categories": [
                "AGENT_EXCEPTION",
                "INVALID_AGENT_ACTION",
                "ACTION_ADAPTER_ERROR",
                "INVALID_AGENT_INPUT_CONSUMPTION",
                "LOCAL_LOG_WRITE_ERROR",
                "WANDB_INIT_ERROR",
                "WANDB_LOG_ERROR",
                "WANDB_SYNC_ERROR"
            ]
        },
        "wandb_backend_contract": wandb_contract,
        "secret_policy": secret_policy,
        "privacy_policy": {
            "forbidden_private_identifiers": [
                "os_username",
                "user_home_directory",
                "machine_hostname",
                "ip_address",
                "absolute_local_paths"
            ],
            "allowed_remote_tracking_metadata": [
                "configured_project",
                "configured_entity",
                "resolved_project",
                "resolved_entity",
                "wandb_run_id",
                "wandb_run_url"
            ]
        },
        "integrity_policy": {
            "file": "run_integrity.json",
            "fingerprinted_artifacts": [
                "run_manifest.json",
                "run_state.json",
                "episodes.csv",
                "summary.json",
                "timing.csv",
                "wandb_sync.json",
                "technical_failures.jsonl"
            ]
        }
    }
