"""
Experiment Logging, Provenance, and Local Run Artifacts for Research Platform V1.
Gate 7 of Research Platform V1.

Core Mandates:
- LOCAL SCIENTIFIC RECORDS ARE AUTHORITATIVE.
- Downstream-only telemetry: Agent never reads logger/W&B state.
- Provenance tracking: Git commit, dirty worktree status, platform contract hashes, seeds.
- Clean technical vs task failure segregation.
- Atomic local writes and duplicate run protection.
- High-resolution agent latency measurement excluding logging overhead.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
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
            "error": str(e)
        }


def capture_environment_provenance() -> Dict[str, Any]:
    """Captures portable, non-private software dependency versions."""
    import importlib.metadata
    env_info = {
        "python_version": platform.python_version(),
        "os_platform": platform.platform(),
        "os_family": platform.system(),
        "numpy_version": np.__version__,
    }

    # MetaDrive version
    try:
        env_info["metadrive_version"] = importlib.metadata.version("metadrive-simulator")
    except Exception:
        env_info["metadrive_version"] = "unknown"

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


@dataclass(frozen=True)
class ExperimentRunConfig:
    """
    Semantic, reproducible configuration for an experiment run.
    Contains no ephemeral paths, timestamps, or credentials.
    """
    run_kind: RunKind
    benchmark_contract_sha256: str
    agent_contract_sha256: str
    platform_runtime_contract_sha256: str
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
    wandb_mode: WandbMode = WandbMode.DISABLED
    wandb_project: str = "sgu-autonomous-driving-rl"
    wandb_entity: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    algorithm_hyperparameters: Dict[str, Any] = field(default_factory=dict)

    def compute_config_sha256(self) -> str:
        """
        Computes canonical SHA-256 fingerprint of the semantic experiment configuration.
        Excludes ephemeral paths, timestamps, and credentials.
        """
        d = {
            "run_kind": self.run_kind.value,
            "benchmark_contract_sha256": self.benchmark_contract_sha256,
            "agent_contract_sha256": self.agent_contract_sha256,
            "platform_runtime_contract_sha256": self.platform_runtime_contract_sha256,
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
class EpisodeTimingRecord:
    """High-resolution timing statistics for agent decisions within an episode."""
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
            mean_act_ms=float(round(np.mean(arr), 4)),
            median_act_ms=float(round(np.median(arr), 4)),
            p95_act_ms=float(round(np.percentile(arr, 95), 4)),
            max_act_ms=float(round(np.max(arr), 4)),
            total_act_ms=float(round(np.sum(arr), 4)),
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
    technical_failures_sha256: Optional[str]
    finalized_at_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LocalExperimentLogger:
    """
    Authoritative local-first experiment persistence engine.
    Ensures atomic writes, schema consistency, duplicate run protection,
    and semantic integrity hashing.
    """
    def __init__(
        self,
        config: ExperimentRunConfig,
        runs_root: Optional[Path] = None,
        custom_run_id: Optional[str] = None,
        repo_root: Optional[Path] = None
    ):
        self.config = config
        self.repo_root = repo_root or Path(__file__).resolve().parent.parent.parent
        self.runs_root = runs_root or (self.repo_root / "autonomous-driving-rl" / "runs")
        self.runs_root.mkdir(parents=True, exist_ok=True)

        self.run_id = custom_run_id or f"run_{uuid.uuid4().hex[:12]}"
        self.run_dir = self.runs_root / self.run_id

        # Duplicate run protection: fail loudly if directory exists
        if self.run_dir.exists():
            raise FileExistsError(
                f"Duplicate run error: run directory '{self.run_dir}' already exists! "
                "Overwriting or appending to existing scientific runs is strictly forbidden."
            )
        self.run_dir.mkdir(parents=True, exist_ok=False)

        self.status = RunStatus.INITIALIZING
        self.created_at_utc = get_utc_now_iso()
        self.recorded_episodes: List[EpisodeRecord] = []
        self.recorded_timings: List[EpisodeTimingRecord] = []
        self.technical_failures: List[Dict[str, Any]] = []

        # Local file paths
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.episodes_csv_path = self.run_dir / "episodes.csv"
        self.summary_json_path = self.run_dir / "summary.json"
        self.timing_csv_path = self.run_dir / "timing.csv"
        self.failures_jsonl_path = self.run_dir / "technical_failures.jsonl"
        self.integrity_json_path = self.run_dir / "run_integrity.json"
        self.wandb_sync_json_path = self.run_dir / "wandb_sync.json"

        # Dirty worktree validation for benchmark evaluation runs
        git_prov = capture_git_provenance(self.repo_root)
        if git_prov.get("git_worktree_dirty") and self.config.run_kind in (
            RunKind.VALIDATION_EVALUATION,
            RunKind.TEST_EVALUATION
        ):
            print(f"[WARNING] Running {self.config.run_kind.value} with dirty git worktree (diff_sha: {git_prov.get('git_diff_sha256')})")

        # Construct and atomically persist immutable RunManifestV1
        env_prov = capture_environment_provenance()
        contracts_prov = {
            "benchmark_contract_sha256": self.config.benchmark_contract_sha256,
            "agent_contract_sha256": self.config.agent_contract_sha256,
            "platform_runtime_contract_sha256": self.config.platform_runtime_contract_sha256
        }

        self.manifest = RunManifestV1(
            run_id=self.run_id,
            run_kind=self.config.run_kind.value,
            created_at_utc=self.created_at_utc,
            experiment_config_sha256=self.config.compute_config_sha256(),
            config=self.config.to_dict(),
            git_provenance=git_prov,
            environment_provenance=env_prov,
            platform_contracts=contracts_prov
        )

        self._atomic_write_json(self.manifest_path, self.manifest.to_dict())
        self._init_csv_files()
        self.status = RunStatus.RUNNING

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
            raise IOError(f"Failed to atomically persist scientific record to '{target_path}': {e}") from e

    def _init_csv_files(self) -> None:
        """Initializes empty CSV headers."""
        episode_headers = [
            "episode_index", "case_id", "split", "tier", "sequence",
            "geometry_generation_seed", "environment_seed", "agent_seed",
            "horizon_steps", "primary_terminal_reason", "terminal_reason",
            "clean_success", "raw_arrival", "final_route_completion",
            "max_route_completion", "episode_steps", "episode_time_s",
            "time_to_clean_success_s", "mean_speed_kmh", "max_speed_kmh",
            "episode_return", "crash_human", "crash_vehicle", "crash_object",
            "crash_building", "crash_sidewalk", "out_of_road", "timeout",
            "has_technical_failure", "technical_failure_reason"
        ]
        with open(self.episodes_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=episode_headers)
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
        case_id: Optional[str] = None,
        split: Optional[str] = None,
        environment_seed: Optional[Any] = None,
        geometry_generation_seed: Optional[Any] = None
    ) -> None:
        """
        Appends an evaluation episode record and its timing to the local authoritative logs.
        Fails loudly if local write fails.
        """
        if self.status not in (RunStatus.RUNNING, RunStatus.INITIALIZING):
            raise RuntimeError(f"Cannot log episode to run in status '{self.status.value}'")

        self.recorded_episodes.append(record)
        if timing:
            self.recorded_timings.append(timing)

        ep_idx = episode_index if episode_index is not None else getattr(record, "episode_index", len(self.recorded_episodes))
        c_id = case_id if case_id is not None else getattr(record, "case_id", f"case_{ep_idx}")
        s_split = split if split is not None else getattr(record, "split", "N/A")
        env_seed = environment_seed if environment_seed is not None else getattr(record, "environment_seed", "N/A")
        geom_seed = geometry_generation_seed if geometry_generation_seed is not None else getattr(record, "geometry_generation_seed", getattr(record, "scenario_seed", "N/A"))

        timeout = getattr(record, "truncated", False) or getattr(record, "timeout", False)

        # 1. Format row for episodes.csv
        row = {
            "episode_index": ep_idx,
            "case_id": c_id,
            "split": s_split,
            "tier": record.tier,
            "sequence": record.sequence,
            "geometry_generation_seed": geom_seed,
            "environment_seed": env_seed,
            "agent_seed": record.agent_seed if record.agent_seed is not None else (self.config.agent_seed if self.config.agent_seed is not None else "N/A"),
            "horizon_steps": getattr(record, "horizon_steps", 1000),
            "primary_terminal_reason": record.primary_reason.value,
            "terminal_reason": record.primary_reason.value,
            "clean_success": record.clean_success,
            "raw_arrival": record.raw_arrival,
            "final_route_completion": record.final_route_completion,
            "max_route_completion": record.max_route_completion,
            "episode_steps": record.episode_steps,
            "episode_time_s": record.simulation_time_s,
            "time_to_clean_success_s": record.time_to_clean_success_s if record.time_to_clean_success_s is not None else "",
            "mean_speed_kmh": record.mean_speed_kmh,
            "max_speed_kmh": record.max_speed_kmh,
            "episode_return": record.episode_return,
            "crash_human": getattr(record, "raw_crash_human", False),
            "crash_vehicle": getattr(record, "raw_crash_vehicle", False),
            "crash_object": getattr(record, "raw_crash_object", False),
            "crash_building": getattr(record, "raw_crash_building", False),
            "crash_sidewalk": getattr(record, "raw_crash_sidewalk", False),
            "out_of_road": getattr(record, "raw_out_of_road", False),
            "timeout": timeout,
            "has_technical_failure": technical_failure is not None,
            "technical_failure_reason": technical_failure.get("reason", "") if technical_failure else ""
        }

        try:
            with open(self.episodes_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writerow(row)
                f.flush()
        except Exception as e:
            self.status = RunStatus.FAILED
            raise IOError(f"Critical local persistence failure in episodes.csv: {e}") from e

        # 2. Append timing.csv
        if timing:
            try:
                with open(self.timing_csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(timing.to_dict().keys()))
                    writer.writerow(timing.to_dict())
                    f.flush()
            except Exception as e:
                self.status = RunStatus.FAILED
                raise IOError(f"Critical local persistence failure in timing.csv: {e}") from e

        # 3. Log technical failure if present
        if technical_failure:
            self.technical_failures.append(technical_failure)
            try:
                with open(self.failures_jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(technical_failure, sort_keys=True) + "\n")
                    f.flush()
            except Exception as e:
                self.status = RunStatus.FAILED
                raise IOError(f"Critical local persistence failure in technical_failures.jsonl: {e}") from e

    def finalize_run(self, wandb_sync_info: Optional[Dict[str, Any]] = None) -> RunIntegrityRecord:
        """
        Finalizes the experiment:
        1. Verifies expected episode count if predeclared.
        2. Computes Gate-4 aggregate metrics overall and per tier, plus Gate-5 macro summaries.
        3. Atomically persists summary.json.
        4. Computes semantic content hashes and saves run_integrity.json.
        5. Saves wandb_sync.json.
        6. Marks status COMPLETE.
        """
        if self.status == RunStatus.FAILED:
            raise RuntimeError(f"Cannot complete run '{self.run_id}': run status is already FAILED!")

        # Verify expected episode count
        if self.config.expected_episode_count is not None:
            if len(self.recorded_episodes) != self.config.expected_episode_count:
                self.status = RunStatus.FAILED
                raise ValueError(
                    f"Episode count mismatch: expected {self.config.expected_episode_count}, "
                    f"but recorded {len(self.recorded_episodes)} episodes!"
                )

        # Compute Gate-4 metrics
        overall_metrics = compute_aggregate_metrics(self.recorded_episodes)

        # Compute per-tier metrics if tier metadata is present
        tiers = ("Easy", "Medium", "Hard", "Extreme")
        tier_cards = {}
        for t in tiers:
            tier_eps = [e for e in self.recorded_episodes if getattr(e, "tier", None) == t]
            if tier_eps:
                tier_cards[t] = compute_aggregate_metrics(tier_eps)

        macro_metrics = None
        if len(tier_cards) == 4:
            macro_metrics = compute_macro_metrics(tier_cards)

        summary_payload = {
            "run_id": self.run_id,
            "run_kind": self.config.run_kind.value,
            "total_episodes": len(self.recorded_episodes),
            "finalized_at_utc": get_utc_now_iso(),
            "overall_metrics": overall_metrics.to_dict(),
            "tier_metrics": {t: card.to_dict() for t, card in tier_cards.items()},
            "macro_metrics": macro_metrics,
            "timing_overall": {
                "total_act_count": sum(t.act_count for t in self.recorded_timings),
                "mean_act_ms": float(round(np.mean([t.mean_act_ms for t in self.recorded_timings]), 4)) if self.recorded_timings else 0.0,
                "max_act_ms": float(round(max([t.max_act_ms for t in self.recorded_timings]), 4)) if self.recorded_timings else 0.0,
            },
            "technical_failure_count": len(self.technical_failures),
        }

        # Atomically write summary.json
        self._atomic_write_json(self.summary_json_path, summary_payload)

        # Save wandb_sync.json
        sync_data = wandb_sync_info or {
            "mode": self.config.wandb_mode.value,
            "sync_status": WandbSyncStatus.DISABLED.value if self.config.wandb_mode == WandbMode.DISABLED else WandbSyncStatus.OFFLINE.value,
            "wandb_run_id": None,
            "wandb_run_url": None,
            "project": self.config.wandb_project,
            "entity": self.config.wandb_entity,
        }
        self._atomic_write_json(self.wandb_sync_json_path, sync_data)

        # Compute semantic integrity fingerprints
        manifest_hash = canonical_json_file_sha256(self.manifest_path)
        episodes_hash = canonical_csv_file_sha256(self.episodes_csv_path)
        summary_hash = canonical_json_file_sha256(self.summary_json_path)
        timing_hash = canonical_csv_file_sha256(self.timing_csv_path)
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
            technical_failures_sha256=failures_hash,
            finalized_at_utc=get_utc_now_iso()
        )

        self._atomic_write_json(self.integrity_json_path, integrity_record.to_dict())
        self.status = RunStatus.COMPLETE
        return integrity_record


def build_logging_contract_core(
    status: str = "LOCKED-FOR-PLATFORM-V1",
    custom_rules: Optional[Dict[str, Any]] = None,
    custom_latency_schema: Optional[Dict[str, Any]] = None,
    custom_wandb_contract: Optional[Dict[str, Any]] = None,
    custom_secret_policy: Optional[Dict[str, Any]] = None
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
        "EpisodeTimingRecord": [f.name for f in dataclasses.fields(EpisodeTimingRecord)],
        "RunIntegrityRecord": [f.name for f in dataclasses.fields(RunIntegrityRecord)],
    }

    runtime_enums = {
        "RunKind": [e.value for e in RunKind],
        "RunStatus": [e.value for e in RunStatus],
        "WandbMode": [e.value for e in WandbMode],
        "WandbSyncStatus": [e.value for e in WandbSyncStatus],
    }

    episode_columns = [
        "episode_index", "case_id", "split", "tier", "sequence",
        "geometry_generation_seed", "environment_seed", "agent_seed",
        "horizon_steps", "primary_terminal_reason", "terminal_reason",
        "clean_success", "raw_arrival", "final_route_completion",
        "max_route_completion", "episode_steps", "episode_time_s",
        "time_to_clean_success_s", "mean_speed_kmh", "max_speed_kmh",
        "episode_return", "crash_human", "crash_vehicle", "crash_object",
        "crash_building", "crash_sidewalk", "out_of_road", "timeout",
        "has_technical_failure", "technical_failure_reason"
    ]

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
        "latency_sync_policies": ["NONE", "FRAMEWORK_SYNCHRONIZED"]
    }
    timing_schema = dict(default_timing)
    if custom_latency_schema:
        timing_schema.update(custom_latency_schema)

    default_wandb = {
        "modes": [e.value for e in WandbMode],
        "failure_policy": "NON-FATAL to local scientific experiment if local logging is healthy",
        "table_name": "evaluation_episodes",
        "step_axis": "episode_index"
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

    return {
        "metadata": {
            "contract_name": "LoggingContractV1",
            "spec_version": "1.0.0",
            "status": status,
            "gate": "Gate 7 (Research Platform V1)"
        },
        "local_record_contract": {
            "root_directory_pattern": "runs/<run_id>",
            "duplicate_run_protection": "fail loudly if run_id directory exists",
            "atomic_write_strategy": "write to .tmp file, flush, fsync, atomic replace",
            "authoritative_source": "LOCAL",
            "remote_sync_source": "DOWNSTREAM_MIRROR",
            "required_files": [
                "run_manifest.json",
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
            "excluded_identifiers": [
                "username",
                "home directory",
                "full local repository paths",
                "machine hostname",
                "IP address"
            ]
        },
        "integrity_policy": {
            "file": "run_integrity.json",
            "fingerprinted_artifacts": [
                "run_manifest.json",
                "episodes.csv",
                "summary.json",
                "timing.csv",
                "technical_failures.jsonl"
            ]
        }
    }
