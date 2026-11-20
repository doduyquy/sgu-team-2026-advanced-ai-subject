"""Deterministic, read-only evaluation on the fixed internal PILOT_DIAGNOSTIC split."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import psutil
import statistics
import sys
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:  # Load torch before Panda3D on Windows.
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.platform import SAFETY_CRITICAL_FLAGS, get_action_adapter
from src.platform.task_context_producer import LEGACY_TASK_PRODUCER, TASK_PRODUCERS, task_producer_identity
from src.training.stage5.checkpoint import load_sac_checkpoint
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.training.stage5.pilot_observation_transform import (
    TRANSFORM_SPECS, WAYPOINT_XY50_V1, GroupwisePhysicalObservation,
    WaypointXY50Observation, transform_fingerprint,
)
from src.training.stage5.pilot_protocol import diagnostic_schedule, load_pilot_split
from src.training.stage5.train_sampler import SamplerStrategy, TrainCaseSampler


def _write_progress(path: Optional[Path], **updates: Any) -> None:
    if path is None:
        return
    payload: Dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload.update(updates)
    payload["updated_at_epoch_s"] = time.time()
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.05 * (attempt + 1))


def policy_state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.policy.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _mean(values: Iterable[float]) -> Optional[float]:
    seq = list(values)
    return float(statistics.mean(seq)) if seq else None


def discounted_return(rewards: Iterable[float], gamma: float) -> float:
    """Return the reward-only discounted episode sum used by SAC's critic target."""
    if not 0.0 < float(gamma) <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    return float(sum((float(gamma) ** step) * float(reward)
                     for step, reward in enumerate(rewards)))


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize completed episodes only; interrupted diagnostic probes are excluded."""
    complete = [row for row in records if not row.get("interrupted", False)]
    n = len(complete)
    summary: Dict[str, Any] = {"episodes_completed": n, "episodes_interrupted": len(records) - n}
    # Probe diagnostics are descriptive only and deliberately kept separate from
    # full-episode outcome rates. Censored records never count as timeout/success.
    for output, field in (
        ("probe_mean_speed_km_h", "mean_speed_km_h"),
        ("probe_stationary_fraction", "stationary_fraction"),
        ("probe_mean_route_progress_delta", "route_progress_delta"),
        ("probe_mean_throttle", "mean_throttle"),
        ("probe_mean_brake", "mean_brake"),
        ("probe_steering_std", "steering_std"),
        ("probe_longest_sustained_moving_steps", "longest_sustained_moving_steps"),
    ):
        summary[output] = _mean(row[field] for row in records if row.get(field) is not None)
    summary["probe_censored_count"] = sum(row.get("outcome") == "PROBE_CENSORED" for row in records)
    summary["probe_safety_failure_count"] = sum(bool(row.get("safety_failure")) for row in records)
    if n == 0:
        return summary
    for field in ("final_route_completion", "max_route_completion", "route_progress_delta", "max_route_progress_delta",
                  "episode_return", "discounted_episode_return", "stationary_fraction", "mean_speed_km_h",
                  "mean_moving_speed_km_h", "mean_throttle", "mean_brake", "mean_steering", "time_to_clean_success_s",
                  "steering_std", "steering_saturation_fraction", "mean_inference_latency_ms", "duration_s"):
        summary[field] = _mean(row[field] for row in complete if row.get(field) is not None)
    summary["median_route_completion"] = float(statistics.median(row["final_route_completion"] for row in complete))
    summary["median_route_progress_delta"] = float(statistics.median(row["route_progress_delta"] for row in complete))
    summary["clean_success_rate"] = sum(bool(row["clean_success"]) for row in complete) / n
    summary["safety_failure_rate"] = sum(bool(row["safety_failure"]) for row in complete) / n
    summary["timeout_rate"] = sum(bool(row["timeout"]) for row in complete) / n
    return summary


def _episode_record(
    env: Stage5TrainingEnv, schedule_item: Dict[str, Any], checkpoint_sha256: str,
    training_step: int, speeds: List[float], controls: List[List[float]], latencies_ms: List[float],
    rewards: List[float], duration_s: float, interrupted: bool, gamma: float,
    initial_route_completion: float,
) -> Dict[str, Any]:
    case = env.current_case
    outcome = env.last_outcome
    raw = outcome.raw_flags if outcome else {}
    route = float(env._last_route_completion)
    longest_moving = 0
    current_moving = 0
    for speed in speeds:
        current_moving = current_moving + 1 if speed >= 1.0 else 0
        longest_moving = max(longest_moving, current_moving)
    return {
        "checkpoint_sha256": checkpoint_sha256, "training_step": int(training_step),
        "geometry_id": schedule_item["geometry_id"], "geometry_sha256": schedule_item["geometry_sha256"],
        "case_id": getattr(case, "case_id", None),
        "tier": getattr(case, "tier", schedule_item["tier"]),
        "sequence": getattr(case, "sequence", schedule_item["sequence"]),
        "geometry_generation_seed": schedule_item["geometry_generation_seed"],
        "environment_seed": schedule_item["environment_seed"], "split": "PILOT_DIAGNOSTIC_INTERNAL_TRAIN",
        "outcome": "PROBE_CENSORED" if interrupted else (
            outcome.primary_reason.value if outcome else "UNDETERMINED"
        ),
        "clean_success": bool(outcome.clean_success) if outcome else False,
        "safety_failure": any(bool(raw.get(flag, False)) for flag in SAFETY_CRITICAL_FLAGS),
        "timeout": bool(outcome and outcome.primary_reason.value == "TIMEOUT"),
        "initial_route_completion": float(initial_route_completion),
        "final_route_completion": route,
        "route_progress_delta": route - float(initial_route_completion),
        "max_route_completion": env.max_route_completion,
        "max_route_progress_delta": float(env.max_route_completion) - float(initial_route_completion),
        "episode_return": float(sum(rewards)), "episode_steps": len(speeds),
        "discount_factor": float(gamma),
        "discounted_episode_return": discounted_return(rewards, gamma),
        "duration_s": duration_s, "mean_speed_km_h": _mean(speeds),
        "time_to_clean_success_s": duration_s if bool(outcome and outcome.clean_success) else None,
        "longest_sustained_moving_steps": longest_moving,
        "stationary_fraction": sum(speed < 1.0 for speed in speeds) / len(speeds) if speeds else None,
        "mean_moving_speed_km_h": _mean(speed for speed in speeds if speed >= 1.0),
        "mean_throttle": _mean(max(0.0, pair[1]) for pair in controls),
        "mean_brake": _mean(max(0.0, -pair[1]) for pair in controls),
        "mean_steering": _mean(pair[0] for pair in controls),
        "steering_std": float(statistics.pstdev(pair[0] for pair in controls)) if controls else None,
        "steering_saturation_fraction": sum(abs(pair[0]) >= 0.95 for pair in controls) / len(controls) if controls else None,
        "mean_inference_latency_ms": _mean(latencies_ms),
        "evaluation_rss_mb": psutil.Process().memory_info().rss / (1024 * 1024),
        "interrupted": bool(interrupted),
        "diagnostic_metrics_label": "development diagnostic; not Platform V1 primary metrics",
    }


def evaluate_checkpoint(
    checkpoint_dir: Path, split_path: Path, output_dir: Path, schedule_seed: int = 5302,
    max_steps_per_episode: Optional[int] = None, max_cases: Optional[int] = None,
    progress_file: Optional[Path] = None, progress_phase: str = "DIAGNOSTICS",
    trace_dir: Optional[Path] = None, geometry_ids: Optional[List[str]] = None,
    *, task_context_producer_id: str = LEGACY_TASK_PRODUCER,
) -> Dict[str, Any]:
    encoder = Stage5ObservationEncoder(core_only=False)
    transform = metadata_transform = None
    metadata_hint = json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata_hint.get("observation_transform"):
        metadata_transform = metadata_hint["observation_transform"]
        transform_id = metadata_transform.get("transform_id") if isinstance(metadata_transform, dict) else None
        expected_spec = TRANSFORM_SPECS.get(transform_id)
        if expected_spec is None or metadata_transform != expected_spec:
            raise ValueError("Unknown checkpoint observation transform; refusing incompatible inference")
        if metadata_hint.get("observation_transform_fingerprint") != transform_fingerprint(expected_spec):
            raise ValueError("Checkpoint observation transform fingerprint mismatch")
        if transform_id == WAYPOINT_XY50_V1["transform_id"]:
            transform = WaypointXY50Observation
        elif transform_id == "pilot_groupwise_physical_v1":
            transform = GroupwisePhysicalObservation
    model, metadata = load_sac_checkpoint(
        checkpoint_dir, device="cpu", expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
        task_context_producer_id=task_context_producer_id
    )
    checkpoint_sha = metadata["model_sha256"]
    producer = task_producer_identity(task_context_producer_id)
    training_step = int(metadata["total_timesteps"])
    gamma = float(getattr(model, "gamma", metadata.get("training_config", {}).get("sac_hyperparameters", {}).get("gamma", 0.99)))
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"Invalid checkpoint discount factor: {gamma}")
    state_before = policy_state_sha256(model)
    update_count_before = int(getattr(model, "_n_updates", 0))
    replay_buffer_before = getattr(model, "replay_buffer", None)
    replay_size_before = replay_buffer_before.size() if replay_buffer_before is not None else None
    sampler = TrainCaseSampler(
        training_root_seed=101, strategy=SamplerStrategy.ROUND_ROBIN,
        allowed_geometry_ids=[x["geometry_id"] for x in load_pilot_split(split_path)["pilot_diagnostic"]],
        project_root=ROOT,
    )
    base_env = Stage5TrainingEnv(sampler=sampler, encoder=encoder, project_root=ROOT,
                                task_context_producer_id=task_context_producer_id)
    env = transform(base_env) if transform is not None else base_env
    adapter = get_action_adapter("continuous_box2_v1")
    schedule = diagnostic_schedule(load_pilot_split(split_path), schedule_seed)
    if geometry_ids:
        allowed = {item["geometry_id"] for item in schedule}
        unknown = sorted(set(geometry_ids) - allowed)
        if unknown:
            raise ValueError(f"Requested trace geometries are not in fixed PILOT_DIAGNOSTIC schedule: {unknown}")
        schedule = [item for item in schedule if item["geometry_id"] in set(geometry_ids)]
    if max_cases is not None:
        schedule = schedule[:max_cases]
    _write_progress(progress_file, status="RUNNING", phase=progress_phase,
                    diagnostic_checkpoint_step=training_step, diagnostic_cases_completed=0,
                    diagnostic_cases_planned=len(schedule), diagnostic_current_case=None,
                    diagnostic_current_episode_step=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"step_{training_step:05d}_episodes.jsonl"
    records: List[Dict[str, Any]] = []
    fixed_policy_observations: List[np.ndarray] = []
    try:
        with records_path.open("w", encoding="utf-8", newline="\n") as sink:
            for case_index, item in enumerate(schedule, start=1):
                case = sampler.get_case_for_geometry_id(item["geometry_id"], item["environment_seed"])
                obs, reset_info = env.reset(options={"case": case})
                fixed_policy_observations.append(np.asarray(obs, dtype=np.float32).copy())
                initial_route_completion = float(
                    reset_info.get("initial_route_completion", base_env._last_route_completion)
                )
                speeds: List[float] = []
                controls: List[List[float]] = []
                latencies: List[float] = []
                rewards: List[float] = []
                trace_rows: List[Dict[str, Any]] = []
                started = time.perf_counter()
                interrupted = False
                done = False
                while not done:
                    policy_observation = np.asarray(obs, dtype=np.float32).copy()
                    before = time.perf_counter_ns()
                    raw_action, _ = model.predict(obs, deterministic=True)
                    latencies.append((time.perf_counter_ns() - before) / 1e6)
                    canonical = adapter.to_canonical(raw_action)
                    action = [float(canonical.steering), float(canonical.throttle_brake)]
                    controls.append(action)
                    obs, reward, terminated, truncated, info = env.step(action)
                    rewards.append(float(reward))
                    speeds.append(float(info.get("speed_km_h", 0.0)))
                    if trace_dir is not None:
                        vehicle = getattr(getattr(base_env, "_env", None), "agent", None)
                        position = getattr(vehicle, "position", None)
                        trace_rows.append({
                            "geometry_id": item["geometry_id"],
                            "environment_seed": item["environment_seed"],
                            "step_index": len(speeds), "policy_observation": policy_observation.tolist(),
                            "action": action, "speed_km_h": float(info.get("speed_km_h", 0.0)),
                            "route_completion": info.get("route_completion"), "delta_route": info.get("delta_route"),
                            "reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated),
                            "primary_reason": info.get("primary_reason"),
                            "raw_safety_flags": dict(getattr(base_env, "_env_flags", {})),
                            "ego_position_xy": ([float(position[0]), float(position[1])] if position is not None else None),
                            "ego_heading_theta": (float(getattr(vehicle, "heading_theta")) if vehicle is not None else None),
                        })
                    done = bool(terminated or truncated)
                    if progress_file is not None and len(speeds) % 100 == 0:
                        _write_progress(
                            progress_file, status="RUNNING", phase=progress_phase,
                            diagnostic_checkpoint_step=training_step,
                            diagnostic_cases_completed=case_index - 1,
                            diagnostic_cases_planned=len(schedule),
                            diagnostic_current_case=item["geometry_id"],
                            diagnostic_current_episode_step=len(speeds),
                            diagnostic_latest_speed_km_h=speeds[-1],
                            diagnostic_latest_route_completion=info.get("route_completion"),
                        )
                    if max_steps_per_episode is not None and len(speeds) >= max_steps_per_episode and not done:
                        interrupted = True
                        break
                record = _episode_record(
                    base_env, item, checkpoint_sha, training_step, speeds, controls, latencies, rewards,
                    time.perf_counter() - started, interrupted, gamma, initial_route_completion,
                )
                record["task_context_producer"] = producer
                records.append(record)
                sink.write(json.dumps(record, sort_keys=True) + "\n")
                sink.flush()
                if trace_dir is not None:
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    trace_path = trace_dir / f"step_{training_step:05d}_{item['geometry_id']}.jsonl"
                    with trace_path.open("w", encoding="utf-8", newline="\n") as trace_sink:
                        for trace_row in trace_rows:
                            trace_sink.write(json.dumps(trace_row, sort_keys=True) + "\n")
                _write_progress(
                    progress_file, status="RUNNING", phase=progress_phase,
                    diagnostic_checkpoint_step=training_step,
                    diagnostic_cases_completed=case_index,
                    diagnostic_cases_planned=len(schedule),
                    diagnostic_current_case=None, diagnostic_current_episode_step=0,
                    diagnostic_last_outcome=record["outcome"],
                )
    finally:
        env.close()
    if sampler.sampled_count != 0:
        raise AssertionError("Diagnostic evaluator mutated the training sampler state")
    if int(getattr(model, "_n_updates", 0)) != update_count_before or policy_state_sha256(model) != state_before:
        raise AssertionError("Evaluation changed policy parameters or ran optimizer updates")
    replay_buffer_after = getattr(model, "replay_buffer", None)
    replay_size_after = replay_buffer_after.size() if replay_buffer_after is not None else None
    if replay_size_after != replay_size_before:
        raise AssertionError("Evaluation changed checkpoint replay-buffer contents")
    policy_observation_array = np.stack(fixed_policy_observations) if fixed_policy_observations else np.empty((0, 409), dtype=np.float32)
    actor_diagnostics: Dict[str, Any] = {"fixed_observation_count": int(len(policy_observation_array))}
    if len(policy_observation_array):
        deterministic_actions, _ = model.predict(policy_observation_array, deterministic=True)
        actor_diagnostics.update({
            "deterministic_action_mean": np.mean(deterministic_actions, axis=0).astype(float).tolist(),
            "deterministic_action_std_across_fixed_observations": np.std(deterministic_actions, axis=0).astype(float).tolist(),
            "policy_input_group_std_mean": {
                "core_259d": float(np.std(policy_observation_array[:, :259], axis=0).mean()),
                "traffic_actor_56d": float(np.std(policy_observation_array[:, 259:315], axis=0).mean()),
                "traffic_mask_metrics_10d": float(np.std(policy_observation_array[:, 315:325], axis=0).mean()),
                "task_waypoints_60d": float(np.std(policy_observation_array[:, 325:385], axis=0).mean()),
                "task_mask_scalars_24d": float(np.std(policy_observation_array[:, 385:409], axis=0).mean()),
            },
        })
        try:
            import torch
            actor = model.policy.actor
            obs_tensor, _ = model.policy.obs_to_tensor(policy_observation_array)
            with torch.no_grad():
                features = actor.extract_features(obs_tensor, actor.features_extractor)
                first_linear = actor.latent_pi[0]
                preactivation = first_linear(features)
                first_activation = actor.latent_pi[1](preactivation)
            actor_diagnostics["first_layer_preactivation_std_mean"] = float(preactivation.std(dim=0, unbiased=False).mean().cpu())
            actor_diagnostics["first_layer_nonpositive_fraction"] = float((preactivation <= 0).float().mean().cpu())
            actor_diagnostics["first_layer_activation_std_mean"] = float(first_activation.std(dim=0, unbiased=False).mean().cpu())
        except (AttributeError, IndexError, TypeError):
            actor_diagnostics["first_layer_activation_diagnostics"] = "UNAVAILABLE_FOR_POLICY_ARCHITECTURE"
    result = {
        "protocol": ("stage5_pilot_waypoint_xy50_ablation_v1" if metadata.get("observation_transform")
                     else "stage5_pilot_diagnostic_v1"),
        "task_context_producer": producer,
        "observation_transform": metadata.get("observation_transform"),
        "observation_transform_fingerprint": metadata.get("observation_transform_fingerprint"),
        "policy_input_fingerprint": metadata.get("policy_input_fingerprint"),
        "split_sha256": load_pilot_split(split_path)["split_sha256"],
        "checkpoint": str(checkpoint_dir), "checkpoint_sha256": checkpoint_sha,
        "training_step": training_step, "schedule_seed": schedule_seed,
        "discount_factor": gamma,
        "planned_cases": len(schedule), "records_file": records_path.name,
        "actor_state_sensitivity": actor_diagnostics,
        "aggregate": summarize(records), "evaluation_state_unchanged": True,
        "optimizer_updates_before": update_count_before,
        "optimizer_updates_after": int(getattr(model, "_n_updates", 0)),
        "replay_buffer_size_before": replay_size_before,
        "replay_buffer_size_after": replay_size_after,
        "training_sampler_sampled_count": sampler.sampled_count,
        "policy_state_sha256_before_after": state_before,
    }
    (output_dir / f"step_{training_step:05d}_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-context-producer", choices=TASK_PRODUCERS, default=LEGACY_TASK_PRODUCER)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=ROOT / "results/stage5/pilot_split_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schedule-seed", type=int, default=5302)
    parser.add_argument("--max-steps-per-episode", type=int, default=None,
                        help="Technical probe only. Interrupted episodes are explicitly excluded from outcome rates.")
    parser.add_argument("--max-cases", type=int, default=None, help="Technical probe only; never a full diagnostic result.")
    parser.add_argument("--progress-file", type=Path, default=None,
                        help="Optional pilot progress file updated during diagnostic episodes.")
    parser.add_argument("--progress-phase", default="DIAGNOSTICS")
    parser.add_argument("--trace-dir", type=Path, default=None,
                        help="Optional development-only per-step trace output directory.")
    parser.add_argument("--geometry-id", action="append", default=None,
                        help="Restrict evaluation to a named geometry from the fixed diagnostic schedule; repeatable.")
    args = parser.parse_args()
    print(json.dumps(evaluate_checkpoint(args.checkpoint, args.split, args.output, args.schedule_seed,
                                         args.max_steps_per_episode, args.max_cases,
                                         args.progress_file, args.progress_phase,
                                         args.trace_dir, args.geometry_id,
                                         task_context_producer_id=args.task_context_producer), indent=2))


if __name__ == "__main__":
    main()
