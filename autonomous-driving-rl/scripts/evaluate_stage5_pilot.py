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
from typing import Any, Dict, Iterable, List, Optional

try:  # Load torch before Panda3D on Windows.
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.platform import SAFETY_CRITICAL_FLAGS, get_action_adapter
from src.training.stage5.checkpoint import load_sac_checkpoint
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
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
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def policy_state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.policy.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _mean(values: Iterable[float]) -> Optional[float]:
    seq = list(values)
    return float(statistics.mean(seq)) if seq else None


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize completed episodes only; interrupted diagnostic probes are excluded."""
    complete = [row for row in records if not row.get("interrupted", False)]
    n = len(complete)
    summary: Dict[str, Any] = {"episodes_completed": n, "episodes_interrupted": len(records) - n}
    if n == 0:
        return summary
    for field in ("final_route_completion", "max_route_completion", "episode_return", "stationary_fraction",
                  "mean_moving_speed_km_h", "mean_throttle", "mean_brake", "mean_steering",
                  "steering_std", "steering_saturation_fraction", "mean_inference_latency_ms", "duration_s"):
        summary[field] = _mean(row[field] for row in complete if row.get(field) is not None)
    summary["median_route_completion"] = float(statistics.median(row["final_route_completion"] for row in complete))
    summary["clean_success_rate"] = sum(bool(row["clean_success"]) for row in complete) / n
    summary["safety_failure_rate"] = sum(bool(row["safety_failure"]) for row in complete) / n
    summary["timeout_rate"] = sum(bool(row["timeout"]) for row in complete) / n
    return summary


def _episode_record(
    env: Stage5TrainingEnv, schedule_item: Dict[str, Any], checkpoint_sha256: str,
    training_step: int, speeds: List[float], controls: List[List[float]], latencies_ms: List[float],
    rewards: List[float], duration_s: float, interrupted: bool,
) -> Dict[str, Any]:
    case = env.current_case
    outcome = env.last_outcome
    raw = outcome.raw_flags if outcome else {}
    route = float(env._last_route_completion)
    return {
        "checkpoint_sha256": checkpoint_sha256, "training_step": int(training_step),
        "geometry_id": schedule_item["geometry_id"], "geometry_sha256": schedule_item["geometry_sha256"],
        "case_id": getattr(case, "case_id", None),
        "tier": getattr(case, "tier", schedule_item["tier"]),
        "sequence": getattr(case, "sequence", schedule_item["sequence"]),
        "geometry_generation_seed": schedule_item["geometry_generation_seed"],
        "environment_seed": schedule_item["environment_seed"], "split": "PILOT_DIAGNOSTIC_INTERNAL_TRAIN",
        "outcome": outcome.primary_reason.value if outcome else "INTERRUPTED_DIAGNOSTIC",
        "clean_success": bool(outcome.clean_success) if outcome else False,
        "safety_failure": any(bool(raw.get(flag, False)) for flag in SAFETY_CRITICAL_FLAGS),
        "timeout": bool(outcome and outcome.primary_reason.value == "TIMEOUT"),
        "final_route_completion": route, "max_route_completion": env.max_route_completion,
        "episode_return": float(sum(rewards)), "episode_steps": len(speeds),
        "duration_s": duration_s, "mean_speed_km_h": _mean(speeds),
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
) -> Dict[str, Any]:
    encoder = Stage5ObservationEncoder(core_only=False)
    model, metadata = load_sac_checkpoint(
        checkpoint_dir, device="cpu", expected_encoder_fingerprint=encoder.get_schema_fingerprint()
    )
    checkpoint_sha = metadata["model_sha256"]
    training_step = int(metadata["total_timesteps"])
    state_before = policy_state_sha256(model)
    update_count_before = int(getattr(model, "_n_updates", 0))
    replay_buffer_before = getattr(model, "replay_buffer", None)
    replay_size_before = replay_buffer_before.size() if replay_buffer_before is not None else None
    sampler = TrainCaseSampler(
        training_root_seed=101, strategy=SamplerStrategy.ROUND_ROBIN,
        allowed_geometry_ids=[x["geometry_id"] for x in load_pilot_split(split_path)["pilot_diagnostic"]],
        project_root=ROOT,
    )
    env = Stage5TrainingEnv(sampler=sampler, encoder=encoder, project_root=ROOT)
    adapter = get_action_adapter("continuous_box2_v1")
    schedule = diagnostic_schedule(load_pilot_split(split_path), schedule_seed)
    if max_cases is not None:
        schedule = schedule[:max_cases]
    _write_progress(progress_file, status="RUNNING", phase=progress_phase,
                    diagnostic_checkpoint_step=training_step, diagnostic_cases_completed=0,
                    diagnostic_cases_planned=len(schedule), diagnostic_current_case=None,
                    diagnostic_current_episode_step=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"step_{training_step:05d}_episodes.jsonl"
    records: List[Dict[str, Any]] = []
    try:
        with records_path.open("w", encoding="utf-8", newline="\n") as sink:
            for case_index, item in enumerate(schedule, start=1):
                case = sampler.get_case_for_geometry_id(item["geometry_id"], item["environment_seed"])
                obs, _ = env.reset(options={"case": case})
                speeds: List[float] = []
                controls: List[List[float]] = []
                latencies: List[float] = []
                rewards: List[float] = []
                started = time.perf_counter()
                interrupted = False
                done = False
                while not done:
                    before = time.perf_counter_ns()
                    raw_action, _ = model.predict(obs, deterministic=True)
                    latencies.append((time.perf_counter_ns() - before) / 1e6)
                    canonical = adapter.to_canonical(raw_action)
                    action = [float(canonical.steering), float(canonical.throttle_brake)]
                    controls.append(action)
                    obs, reward, terminated, truncated, info = env.step(action)
                    rewards.append(float(reward))
                    speeds.append(float(info.get("speed_km_h", 0.0)))
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
                    env, item, checkpoint_sha, training_step, speeds, controls, latencies, rewards,
                    time.perf_counter() - started, interrupted,
                )
                records.append(record)
                sink.write(json.dumps(record, sort_keys=True) + "\n")
                sink.flush()
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
    result = {
        "protocol": "stage5_pilot_diagnostic_v1", "split_sha256": load_pilot_split(split_path)["split_sha256"],
        "checkpoint": str(checkpoint_dir), "checkpoint_sha256": checkpoint_sha,
        "training_step": training_step, "schedule_seed": schedule_seed,
        "planned_cases": len(schedule), "records_file": records_path.name,
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
    args = parser.parse_args()
    print(json.dumps(evaluate_checkpoint(args.checkpoint, args.split, args.output, args.schedule_seed,
                                         args.max_steps_per_episode, args.max_cases,
                                         args.progress_file, args.progress_phase), indent=2))


if __name__ == "__main__":
    main()
