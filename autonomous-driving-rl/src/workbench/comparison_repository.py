"""Read-only data access for the additive Workbench Comparison View V1.

This view never recalculates Platform V1 aggregate metrics. Canonical runs are
eligible for display only after the existing repository verifies their hashes;
Stage 5 development records are separately checked against their own manifest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.launcher.cases import get_default_project_root
from src.launcher.registry import build_canonical_agent_registry
from src.workbench.results_repository import (
    IntegrityDisplayStatus,
    RunArtifactRepository,
    RunArtifactSnapshotV1,
)


PRIMARY_METRICS = (
    "clean_success_rate",
    "safety_failure_rate",
    "mean_final_route_completion",
    "median_final_route_completion",
    "mean_time_to_clean_success_s",
)
STAGE5_DIAGNOSTIC_METRICS = (
    "mean_route_progress_delta",
    "mean_speed_km_h",
    "stationary_fraction",
    "mean_throttle",
    "mean_brake",
    "mean_steering",
    "clean_success_count",
    "safety_failure_count",
    "probe_censored",
    "official_timeout_count",
)


@dataclass(frozen=True)
class ComparisonRun:
    key: str
    label: str
    track: str
    run_id: str
    agent_id: str
    agent_version: str
    method_family: str
    checkpoint_step: Optional[int]
    checkpoint_sha256: Optional[str]
    checkpoint_bundle_status: str
    source_revision: Optional[str]
    dataset_split: str
    schedule: Tuple[Tuple[str, int], ...]
    input_profile_id: Optional[str]
    action_adapter_id: Optional[str]
    inference_mode: Optional[str]
    metric_contracts: Tuple[Tuple[str, str], ...]
    summary: Dict[str, Any] = field(default_factory=dict)
    per_tier: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    records: Tuple[Dict[str, Any], ...] = ()
    provenance: Dict[str, Any] = field(default_factory=dict)
    benchmark_eligible: bool = False


@dataclass(frozen=True)
class ComparisonResult:
    directly_comparable: bool
    status: str
    reason: str
    paired_records: Tuple[Dict[str, Any], ...]
    metrics: Tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _canonical_sha256(value: Any) -> str:
    import hashlib
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid environment seed: {value!r}") from exc


def _case_schedule(records: List[Dict[str, Any]]) -> Tuple[Tuple[str, int], ...]:
    keys = [(str(row["case_id"]), _as_int(row["environment_seed"])) for row in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate case/environment-seed pair in evaluation records")
    return tuple(sorted(keys))


def _checkpoint_bundle_status(project_root: Path, raw_root: str, arm: str, step: int,
                              expected_sha: Optional[str], expected_metadata_sha: Optional[str],
                              expected_checksums_sha: Optional[str]) -> str:
    checkpoint = project_root / raw_root / f"arm_{arm}" / "checkpoints" / f"step_{step:05d}"
    if not checkpoint.is_dir():
        return "MODEL BUNDLE MISSING (digest retained in result manifest)"
    try:
        model_path, metadata_path = checkpoint / "model.zip", checkpoint / "metadata.json"
        checksums_path = checkpoint / "CHECKSUMS.sha256"
        if not all(path.is_file() for path in (model_path, metadata_path, checksums_path)):
            return "FAILED: checkpoint bundle is incomplete"
        if expected_checksums_sha and _sha256(checksums_path) != expected_checksums_sha:
            return "FAILED: checksum manifest differs from result manifest"
        if expected_metadata_sha and _sha256(metadata_path) != expected_metadata_sha:
            return "FAILED: metadata differs from result manifest"
        checksums: Dict[str, str] = {}
        for line in checksums_path.read_text(encoding="utf-8").splitlines():
            digest, separator, filename = line.partition("  ")
            if not separator or len(digest) != 64 or filename not in {"model.zip", "metadata.json", "replay_buffer.pkl"}:
                return "FAILED: malformed checkpoint checksum manifest"
            checksums[filename] = digest
        metadata = _read_json(metadata_path)
        if checksums.get("model.zip") != _sha256(model_path) or checksums.get("metadata.json") != _sha256(metadata_path):
            return "FAILED: checkpoint bundle checksum mismatch"
        if metadata.get("model_sha256") != checksums.get("model.zip"):
            return "FAILED: model digest differs from checkpoint metadata"
        if expected_sha and metadata.get("model_sha256") != expected_sha:
            return "FAILED: model digest differs from result manifest"
        if int(metadata.get("total_timesteps", -1)) != step or not metadata.get("encoder_fingerprint"):
            return "FAILED: checkpoint step/encoder metadata mismatch"
        return "VERIFIED"
    except Exception as exc:
        return f"FAILED: {type(exc).__name__}: {exc}"


def _stage5_runs(project_root: Path) -> List[ComparisonRun]:
    results_root = project_root / "results" / "stage5"
    runs_root = project_root / "runs"
    discovered: List[ComparisonRun] = []
    if not results_root.is_dir():
        return discovered
    for result_dir in sorted(results_root.iterdir()):
        if not result_dir.is_dir():
            continue
        manifest_path = result_dir / "artifact_manifest.json"
        protocol_path = result_dir / "protocol.json"
        analysis_path = result_dir / "analysis.json"
        if not (manifest_path.is_file() and protocol_path.is_file() and analysis_path.is_file()):
            continue
        manifest = _read_json(manifest_path)
        protocol = _read_json(protocol_path)
        analysis = _read_json(analysis_path)
        if manifest.get("protocol") != "stage5_tau_ab_v1":
            continue
        if manifest.get("holdouts_accessed") is not False or analysis.get("holdouts_accessed") is not False:
            raise ValueError(f"Holdout-use declaration is invalid: {result_dir.name}")
        expected_files = {
            "analysis.json": manifest.get("analysis_sha256"),
            "anchor_observations.json": manifest.get("anchors_sha256"),
            "q_anchor_diagnostics.json": manifest.get("q_diagnostics_sha256"),
        }
        for name, expected in expected_files.items():
            path = result_dir / name
            if not path.is_file() or not expected or _sha256(path) != expected:
                raise ValueError(f"Stage 5 diagnostic artifact integrity failed: {path}")
        protocol_sha = str(manifest.get("protocol_sha256", ""))
        if protocol.get("protocol_sha256") != protocol_sha or analysis.get("protocol_sha256") != protocol_sha:
            raise ValueError(f"Protocol identity mismatch: {result_dir.name}")
        protocol_payload = {k: v for k, v in protocol.items() if k != "protocol_sha256"}
        if _canonical_sha256(protocol_payload) != protocol_sha:
            raise ValueError(f"Protocol hash verification failed: {result_dir.name}")
        split_path = project_root / "results" / "stage5" / "pilot_split_v1.json"
        split = _read_json(split_path)
        split_payload = {k: v for k, v in split.items() if k != "split_sha256"}
        if _canonical_sha256(split_payload) != protocol.get("pilot_split_sha256"):
            raise ValueError(f"Internal TRAIN split hash verification failed: {result_dir.name}")
        if manifest.get("pilot_split_sha256") != protocol.get("pilot_split_sha256"):
            raise ValueError(f"Result manifest split identity mismatch: {result_dir.name}")
        registered_config_path = project_root / "configs" / "experiments" / "stage5_tau_ab_v1.json"
        if not registered_config_path.is_file() or _sha256(registered_config_path) != protocol.get("registered_config_sha256"):
            raise ValueError(f"Registered experiment config hash verification failed: {result_dir.name}")
        if protocol.get("status") != "PRE_REGISTERED" or protocol.get("holdouts_accessed") is not False:
            raise ValueError(f"Stage 5 diagnostic protocol is not preregistered/TRAIN-only: {result_dir.name}")
        raw_root = str(manifest.get("raw_artifacts_root", ""))
        if not raw_root or Path(raw_root).is_absolute() or ".." in Path(raw_root).parts:
            raise ValueError(f"Unsafe raw artifact path in manifest: {raw_root!r}")
        schedule = protocol.get("diagnostic_schedule", [])
        schedule_keys = tuple(sorted((str(row["geometry_id"]), _as_int(row["environment_seed"])) for row in schedule))
        if len(schedule_keys) != len(schedule) or not schedule_keys:
            raise ValueError(f"Invalid diagnostic schedule: {result_dir.name}")
        locked_diagnostic_ids = {str(row["geometry_id"]) for row in split.get("pilot_diagnostic", [])}
        if not {geometry_id for geometry_id, _ in schedule_keys}.issubset(locked_diagnostic_ids):
            raise ValueError(f"Diagnostic schedule contains geometry outside locked internal TRAIN split: {result_dir.name}")
        q_artifact = _read_json(result_dir / "q_anchor_diagnostics.json")
        if q_artifact.get("protocol_sha256") != protocol_sha:
            raise ValueError(f"Q diagnostic protocol identity mismatch: {result_dir.name}")
        for arm in ("A", "B"):
            run_status = manifest.get("arm_run_status", {}).get(arm, {})
            if run_status.get("status") != "COMPLETE":
                raise ValueError(f"Stage 5 arm is not COMPLETE: {result_dir.name}/{arm}")
            for step in protocol.get("checkpoint_steps", []):
                key = str(int(step))
                rows = [r for r in analysis.get("diagnostic_records", [])
                        if r.get("arm") == arm and int(r.get("training_step", -1)) == int(step)]
                cp_meta = next((c for c in manifest.get("checkpoints", {}).get(arm, [])
                                if int(c.get("training_step", -1)) == int(step)), None)
                if cp_meta is None or not cp_meta.get("verified"):
                    raise ValueError(f"Checkpoint integrity declaration missing: {result_dir.name}/{arm}/{step}")
                if len(rows) != len(schedule_keys):
                    raise ValueError(f"Diagnostic case count mismatch: {result_dir.name}/{arm}/{step}")
                if any(row.get("split") != "PILOT_DIAGNOSTIC_INTERNAL_TRAIN" for row in rows):
                    raise ValueError(f"Non-TRAIN record found in development diagnostic: {result_dir.name}")
                observed_schedule = tuple(sorted((str(row["geometry_id"]), _as_int(row["environment_seed"])) for row in rows))
                if observed_schedule != schedule_keys:
                    raise ValueError(f"Diagnostic record schedule mismatch: {result_dir.name}/{arm}/{step}")
                model_sha = cp_meta.get("model_sha256")
                if model_sha and any(row.get("checkpoint_sha256") != model_sha for row in rows):
                    raise ValueError(f"Record/checkpoint digest mismatch: {result_dir.name}/{arm}/{step}")
                summary = analysis["arms"][arm][key]
                overall = summary["overall"]
                q_rows = [r for r in q_artifact.get("records", [])
                          if r.get("arm") == arm and int(r.get("training_step", -1)) == int(step)]
                if len(q_rows) != len(schedule_keys) or any(r.get("diagnostic_label") !=
                        "critic evaluations on fixed reset observations; not empirical returns" for r in q_rows):
                    raise ValueError(f"Q diagnostic anchor coverage/label mismatch: {result_dir.name}/{arm}/{step}")
                q_schedule = tuple(sorted((str(r["geometry_id"]), _as_int(r["environment_seed"])) for r in q_rows))
                if q_schedule != schedule_keys or (model_sha and any(r.get("checkpoint_sha256") != model_sha for r in q_rows)):
                    raise ValueError(f"Q diagnostic schedule/checkpoint mismatch: {result_dir.name}/{arm}/{step}")
                q_mean_fields = {
                    "mean_deterministic_min_q": "deterministic_min_q",
                    "mean_stochastic_min_q": "stochastic_min_q_mean",
                    "mean_target_min_q": "stochastic_target_min_q_mean",
                    "mean_stochastic_positive_throttle_fraction": "stochastic_positive_throttle_fraction",
                    "mean_entropy_coefficient": "entropy_coefficient",
                }
                q_summary = {name: sum(float(r[field]) for r in q_rows) / len(q_rows)
                             for name, field in q_mean_fields.items()}
                display_summary = {**overall, **q_summary}
                checkpoint_status = _checkpoint_bundle_status(
                    project_root, raw_root, arm, int(step), cp_meta.get("model_sha256"),
                    cp_meta.get("metadata_sha256"), cp_meta.get("checksums_sha256")
                )
                run_id = f"{result_dir.name}:{arm}:step{step}"
                discovered.append(ComparisonRun(
                    key=f"stage5:{run_id}",
                    label=f"SAC {arm} · {int(step):,} steps · {result_dir.name}",
                    track="DEVELOPMENT_DIAGNOSTICS",
                    run_id=run_id,
                    agent_id="stage5_sac",
                    agent_version="0.1.0",
                    method_family="MODEL_FREE_RL / SAC",
                    checkpoint_step=int(step),
                    checkpoint_sha256=model_sha,
                    checkpoint_bundle_status=(checkpoint_status if model_sha else "MISSING CHECKPOINT SHA"),
                    source_revision=protocol.get("code_commit"),
                    dataset_split="PILOT_DIAGNOSTIC_INTERNAL_TRAIN (TRAIN ONLY)",
                    schedule=tuple(sorted((str(row["case_id"]), _as_int(row["environment_seed"])) for row in rows)),
                    input_profile_id="STATE_DECISION_V1",
                    action_adapter_id="continuous_box2_v1",
                    inference_mode="deterministic",
                    metric_contracts=(("diagnostic", "stage5_pilot_diagnostic_v1"),),
                    summary=display_summary,
                    per_tier=summary.get("per_tier", {}),
                    records=tuple(rows),
                    provenance={
                        "protocol_sha256": protocol_sha,
                        "result_manifest_sha256": _sha256(manifest_path),
                        "analysis_sha256": manifest["analysis_sha256"],
                        "config_sha256": protocol.get("registered_config_sha256"),
                        "split_sha256": protocol.get("pilot_split_sha256"),
                        "training_root_seed": protocol.get("training_root_seed"),
                        "tau": protocol.get("arms", {}).get(arm, {}).get("tau"),
                        "gamma": protocol.get("common_hyperparameters", {}).get("gamma"),
                        "checkpoint_metadata_sha256": cp_meta.get("metadata_sha256"),
                        "checkpoint_checksums_sha256": cp_meta.get("checksums_sha256"),
                        "case_schedule": list(schedule_keys),
                        "q_diagnostics_path": "q_anchor_diagnostics.json",
                        "q_diagnostics": q_summary,
                        "folder": str(result_dir),
                        "benchmark_eligible": False,
                    },
                    benchmark_eligible=False,
                ))
    return discovered


def _canonical_runs(project_root: Path) -> List[ComparisonRun]:
    runs_root = project_root / "runs"
    repository = RunArtifactRepository(runs_root)
    registry = build_canonical_agent_registry()
    found: List[ComparisonRun] = []
    for run_id in repository.discover_run_ids():
        snapshot = repository.load_run_snapshot(run_id)
        if (snapshot is None or snapshot.status != "COMPLETE" or not snapshot.canonical_run
                or snapshot.integrity_status != IntegrityDisplayStatus.VERIFIED):
            continue
        run_dir = Path(snapshot.run_dir)
        manifest = _read_json(run_dir / "run_manifest.json")
        config = manifest.get("config", {})
        algorithm = config.get("algorithm_hyperparameters", {})
        checkpoint = algorithm.get("sac_checkpoint") or algorithm.get("checkpoint") or {}
        summary = (snapshot.summary_payload or {}).get("overall_metrics", {})
        if not all(metric in summary for metric in PRIMARY_METRICS):
            continue
        records = list(snapshot.episode_rows)
        if not records or any(not row.get("case_id") or row.get("environment_seed") in (None, "") for row in records):
            continue
        schedule = _case_schedule(records)
        try:
            registration = registry.get(snapshot.agent_id or "")
            eligible = bool(registration.benchmark_eligible)
            family = registration.method_family
        except (KeyError, AttributeError):
            eligible, family = False, "UNKNOWN"
        # Track A is reserved for runs whose registered agent is benchmark-eligible.
        # Ineligible completed runs remain visible in the existing Results browser.
        if not eligible:
            continue
        splits = sorted({str(row.get("split") or "UNKNOWN") for row in records})
        found.append(ComparisonRun(
            key=f"canonical:{run_id}",
            label=f"{snapshot.agent_id or 'agent'} · {run_id}",
            track="CANONICAL_EVALUATION",
            run_id=run_id,
            agent_id=snapshot.agent_id or "UNKNOWN",
            agent_version=snapshot.agent_version or "UNKNOWN",
            method_family=str(family),
            checkpoint_step=None,
            checkpoint_sha256=checkpoint.get("model_sha256"),
            checkpoint_bundle_status="RECORDED IN RUN MANIFEST" if checkpoint.get("model_sha256") else "NO CHECKPOINT ID",
            source_revision=snapshot.git_commit_sha,
            dataset_split=" / ".join(splits) if len(splits) == 1 else "MIXED: " + ", ".join(splits),
            schedule=schedule,
            input_profile_id=snapshot.input_profile_id,
            action_adapter_id=snapshot.action_adapter_id,
            inference_mode=snapshot.inference_stochasticity,
            metric_contracts=tuple(sorted((str(k), str(v)) for k, v in snapshot.platform_contracts.items())),
            summary=summary,
            per_tier=(snapshot.summary_payload or {}).get("tier_metrics", {}),
            records=tuple(records),
            provenance={
                "integrity": snapshot.integrity_status.value,
                "experiment_config_sha256": snapshot.experiment_config_sha256,
                "git_commit_sha": snapshot.git_commit_sha,
                "platform_contracts": snapshot.platform_contracts,
                "run_manifest": str(run_dir / "run_manifest.json"),
                "benchmark_eligible": eligible,
                "folder": str(run_dir),
            },
            benchmark_eligible=eligible,
        ))
    return found


def discover_comparison_runs(project_root: Optional[Path] = None) -> List[ComparisonRun]:
    root = Path(project_root or get_default_project_root()).resolve()
    return _canonical_runs(root) + _stage5_runs(root)


def compare_runs(left: ComparisonRun, right: ComparisonRun) -> ComparisonResult:
    if left.track != right.track:
        return ComparisonResult(False, "NOT DIRECTLY COMPARABLE",
                                "Canonical evaluation and TRAIN-only development diagnostics are different tracks.", (), ())
    if left.action_adapter_id != right.action_adapter_id:
        return ComparisonResult(False, "NOT DIRECTLY COMPARABLE",
                                "Action adapter IDs differ.", (), ())
    if left.input_profile_id != right.input_profile_id:
        return ComparisonResult(False, "NOT DIRECTLY COMPARABLE",
                                "Input profile IDs differ.", (), ())
    if left.inference_mode != right.inference_mode:
        return ComparisonResult(False, "NOT DIRECTLY COMPARABLE",
                                "Inference modes differ.", (), ())
    if left.metric_contracts != right.metric_contracts:
        return ComparisonResult(False, "NOT DIRECTLY COMPARABLE",
                                "Metric contract identities differ.", (), ())
    if left.dataset_split != right.dataset_split:
        return ComparisonResult(False, "NOT DIRECTLY COMPARABLE",
                                "Dataset split/protocol scope differs.", (), ())
    if left.schedule != right.schedule:
        return ComparisonResult(False, "NOT DIRECTLY COMPARABLE",
                                "Case IDs or environment seeds differ; paired deltas are suppressed.", (), ())
    if left.track == "CANONICAL_EVALUATION":
        if not left.benchmark_eligible or not right.benchmark_eligible:
            return ComparisonResult(False, "NOT DIRECTLY COMPARABLE",
                                    "At least one agent is not benchmark eligible.", (), ())
        metrics = PRIMARY_METRICS
    else:
        if left.provenance.get("protocol_sha256") != right.provenance.get("protocol_sha256"):
            return ComparisonResult(False, "NOT DIRECTLY COMPARABLE",
                                    "Stage 5 protocols differ.", (), ())
        if not left.checkpoint_sha256 or not right.checkpoint_sha256:
            return ComparisonResult(False, "NOT DIRECTLY COMPARABLE",
                                    "At least one checkpoint SHA-256 is missing.", (), ())
        metrics = STAGE5_DIAGNOSTIC_METRICS
    right_rows = {(str(row.get("case_id")), _as_int(row.get("environment_seed"))): row for row in right.records}
    paired = []
    for row_a in left.records:
        key = (str(row_a.get("case_id")), _as_int(row_a.get("environment_seed")))
        row_b = right_rows[key]
        paired.append({"case_id": key[0], "environment_seed": key[1], "tier": row_a.get("tier"),
                       "left": dict(row_a), "right": dict(row_b)})
    return ComparisonResult(True, "PAIRED COMPARISON", "Same metric contract, adapter and case/environment-seed schedule.",
                            tuple(paired), tuple(metrics))
