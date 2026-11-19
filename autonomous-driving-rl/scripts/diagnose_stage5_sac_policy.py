#!/usr/bin/env python3
"""Read-only deterministic/stochastic SAC behavior diagnostic on internal TRAIN cases."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from stable_baselines3.common.utils import obs_as_tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.stage5.checkpoint import load_sac_checkpoint
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.training.stage5.pilot_protocol import diagnostic_schedule, load_pilot_split
from src.training.stage5.train_sampler import SamplerStrategy, TrainCaseSampler

DEFAULT_CHECKPOINT = ROOT / "runs/stage5_pilot/s5_3d_tierrr_gamma099_seed101_20261115/checkpoints/step_10000"
DEFAULT_SPLIT = ROOT / "results/stage5/pilot_split_v1.json"
DEFAULT_OUTPUT = ROOT / "runs/stage5_pilot/s5_3f_stochastic_policy_diag_seed101_20261115"
DEFAULT_GEOMETRIES = ("geom_easy_SCCS_seed19", "geom_extreme_CrXROSTR_seed0")
FORCED_THROTTLE = (0.0, 0.7)
CRITIC_DIAGNOSTIC_STRIDE = 250


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_sha(model: Any) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.policy.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _q_values(model: Any, observation: np.ndarray, actions: Sequence[Sequence[float]]) -> List[List[float]]:
    """Read both critic heads for actions in the environment's [-1, 1] Box domain."""
    obs_batch = np.repeat(np.asarray(observation, dtype=np.float32)[None, :], len(actions), axis=0)
    obs_tensor = obs_as_tensor(obs_batch, model.device)
    action_tensor = torch.as_tensor(np.asarray(actions, dtype=np.float32), device=model.device)
    with torch.no_grad():
        values = model.critic(obs_tensor, action_tensor)
    return [[float(head[index].detach().cpu().item()) for head in values]
            for index in range(len(actions))]


def _mean(values: Sequence[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _run_episode(
    env: Stage5TrainingEnv,
    model: Any,
    schedule_item: Dict[str, Any],
    mode: str,
    trial_seed: int,
    sink,
    max_steps: int | None = None,
) -> Dict[str, Any]:
    sampler = env.sampler
    case = sampler.get_case_for_geometry_id(schedule_item["geometry_id"], schedule_item["environment_seed"])
    obs, reset_info = env.reset(options={"case": case})
    initial_route = float(reset_info.get("initial_route_completion", 0.0))
    if mode == "stochastic":
        random.seed(trial_seed)
        np.random.seed(trial_seed)
        torch.manual_seed(trial_seed)

    speeds: List[float] = []
    throttle_brake: List[float] = []
    route_values: List[float] = []
    rewards: List[float] = []
    position_start = position_end = None
    start = time.perf_counter()
    interrupted = False
    done = False
    step_index = 0
    while not done:
        observation = np.asarray(obs, dtype=np.float32).copy()
        deterministic_action, _ = model.predict(observation, deterministic=True)
        if mode == "deterministic":
            action, _ = model.predict(observation, deterministic=True)
        elif mode == "stochastic":
            action, _ = model.predict(observation, deterministic=False)
        else:
            raise ValueError(f"Unsupported policy mode: {mode}")
        action = np.asarray(action, dtype=np.float32).reshape(2)
        q_rows = None
        if step_index % CRITIC_DIAGNOSTIC_STRIDE == 0:
            q_rows = _q_values(model, observation, [deterministic_action, action, FORCED_THROTTLE])

        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        step_index += 1
        speed = float(info.get("speed_km_h", 0.0))
        route = float(info.get("route_completion", 0.0))
        vehicle = getattr(getattr(env, "_env", None), "agent", None)
        position = getattr(vehicle, "position", None)
        position_xy = [float(position[0]), float(position[1])] if position is not None else None
        if position_start is None:
            position_start = position_xy
        position_end = position_xy
        speeds.append(speed)
        throttle_brake.append(float(action[1]))
        route_values.append(route)
        rewards.append(float(reward))
        row = {
            "geometry_id": schedule_item["geometry_id"],
            "tier": schedule_item["tier"],
            "environment_seed": schedule_item["environment_seed"],
            "mode": mode,
            "trial_seed": trial_seed,
            "step": step_index,
            "deterministic_action": np.asarray(deterministic_action, dtype=np.float32).tolist(),
            "executed_action": action.tolist(),
            "q_deterministic": q_rows[0] if q_rows is not None else None,
            "q_executed": q_rows[1] if q_rows is not None else None,
            "q_forced_throttle_07": q_rows[2] if q_rows is not None else None,
            "speed_km_h": speed,
            "route_completion": route,
            "delta_route": float(info.get("delta_route", 0.0)),
            "position_xy": position_xy,
            "reward": float(reward),
            "reward_breakdown": info.get("reward_breakdown"),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "outcome": info.get("primary_reason"),
        }
        sink.write(json.dumps(row, sort_keys=True) + "\n")
        if step_index % 100 == 0:
            sink.flush()
        if max_steps is not None and step_index >= max_steps and not done:
            interrupted = True
            break

    return {
        "geometry_id": schedule_item["geometry_id"], "tier": schedule_item["tier"],
        "geometry_generation_seed": schedule_item["geometry_generation_seed"],
        "geometry_sha256": schedule_item["geometry_sha256"],
        "environment_seed": schedule_item["environment_seed"], "mode": mode,
        "trial_seed": trial_seed, "initial_route_completion": initial_route,
        "final_route_completion": route_values[-1] if route_values else initial_route,
        "route_delta_from_reset": (route_values[-1] - initial_route) if route_values else 0.0,
        "episode_steps": step_index, "interrupted": interrupted,
        "outcome": env.last_outcome.primary_reason.value if env.last_outcome else "INTERRUPTED_DIAGNOSTIC",
        "clean_success": bool(env.last_outcome.clean_success) if env.last_outcome else False,
        "safety_failure": bool(env.last_outcome and env.last_outcome.primary_reason.value in {
            "CRASH_HUMAN", "CRASH_VEHICLE", "CRASH_OBJECT", "CRASH_BUILDING", "CRASH_SIDEWALK", "OUT_OF_ROAD"
        }),
        "mean_speed_km_h": _mean(speeds), "max_speed_km_h": max(speeds) if speeds else None,
        "stationary_fraction_speed_lt_1_km_h": (sum(speed < 1.0 for speed in speeds) / len(speeds)) if speeds else None,
        "mean_throttle": _mean([max(0.0, value) for value in throttle_brake]),
        "positive_throttle_fraction_gt_005": (sum(value > 0.05 for value in throttle_brake) / len(throttle_brake)) if throttle_brake else None,
        "mean_brake": _mean([max(0.0, -value) for value in throttle_brake]),
        "episode_return": sum(rewards), "duration_s": time.perf_counter() - start,
        "position_start_xy": position_start, "position_end_xy": position_end,
    }


def _forced_throttle_probe(env: Stage5TrainingEnv, schedule_item: Dict[str, Any], sink, steps: int = 25) -> Dict[str, Any]:
    case = env.sampler.get_case_for_geometry_id(schedule_item["geometry_id"], schedule_item["environment_seed"])
    _, reset_info = env.reset(options={"case": case})
    initial_route = float(reset_info.get("initial_route_completion", 0.0))
    speeds: List[float] = []
    positions: List[List[float] | None] = []
    routes: List[float] = []
    for step in range(1, steps + 1):
        obs, reward, terminated, truncated, info = env.step(np.asarray(FORCED_THROTTLE, dtype=np.float32))
        speed = float(info.get("speed_km_h", 0.0))
        vehicle = getattr(getattr(env, "_env", None), "agent", None)
        position = getattr(vehicle, "position", None)
        position_xy = [float(position[0]), float(position[1])] if position is not None else None
        speeds.append(speed)
        positions.append(position_xy)
        routes.append(float(info.get("route_completion", 0.0)))
        sink.write(json.dumps({
            "geometry_id": schedule_item["geometry_id"], "mode": "forced_throttle_technical_probe",
            "step": step, "executed_action": list(FORCED_THROTTLE), "speed_km_h": speed,
            "route_completion": routes[-1], "delta_route": float(info.get("delta_route", 0.0)),
            "position_xy": position_xy, "reward": float(reward),
            "reward_breakdown": info.get("reward_breakdown"),
            "terminated": bool(terminated), "truncated": bool(truncated),
        }, sort_keys=True) + "\n")
        if terminated or truncated:
            break
    return {
        "geometry_id": schedule_item["geometry_id"], "tier": schedule_item["tier"],
        "environment_seed": schedule_item["environment_seed"], "mode": "forced_throttle_technical_probe",
        "steps": len(speeds), "interrupted_by_design": len(speeds) < schedule_item["horizon_steps"],
        "initial_route_completion": initial_route, "final_route_completion": routes[-1] if routes else initial_route,
        "route_delta_from_reset": (routes[-1] - initial_route) if routes else 0.0,
        "first_speed_km_h": speeds[0] if speeds else None, "last_speed_km_h": speeds[-1] if speeds else None,
        "max_speed_km_h": max(speeds) if speeds else None,
        "position_start_xy": positions[0] if positions else None,
        "position_end_xy": positions[-1] if positions else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--geometry-ids", nargs="+", default=list(DEFAULT_GEOMETRIES))
    parser.add_argument("--stochastic-rollouts", type=int, default=5)
    parser.add_argument("--trial-seed-base", type=int, default=10100)
    args = parser.parse_args()
    if args.stochastic_rollouts < 1:
        parser.error("--stochastic-rollouts must be >= 1")

    # This is inference-only. A small CPU thread pool avoids large OpenMP
    # scheduling overhead for the short MLP batches used by predict()/critic.
    torch.set_num_threads(1)

    split = load_pilot_split(args.split)
    schedule = diagnostic_schedule(split, schedule_seed=5302)
    schedule_by_id = {item["geometry_id"]: item for item in schedule}
    unknown = sorted(set(args.geometry_ids) - set(schedule_by_id))
    if unknown:
        parser.error(f"Geometry IDs are outside the fixed PILOT_DIAGNOSTIC schedule: {unknown}")
    selected = [dict(schedule_by_id[geometry_id]) for geometry_id in args.geometry_ids]
    for item in selected:
        case = TrainCaseSampler(
            training_root_seed=101, strategy=SamplerStrategy.ROUND_ROBIN,
            allowed_geometry_ids=[x["geometry_id"] for x in split["pilot_diagnostic"]], project_root=ROOT,
        ).get_case_for_geometry_id(item["geometry_id"], item["environment_seed"])
        item["horizon_steps"] = case.horizon_steps

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "episodes.jsonl").touch()
    run_info: Dict[str, Any] = {
        "protocol": "stage5_sac_deterministic_stochastic_diagnostic_v1",
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_model_sha256": _sha256(args.checkpoint / "model.zip"),
        "training_step": 10000, "split_sha256": split["split_sha256"],
        "schedule_seed": 5302, "selected_schedule": selected,
        "stochastic_rollouts_per_geometry": args.stochastic_rollouts,
        "trial_seed_base": args.trial_seed_base, "forced_throttle_action": list(FORCED_THROTTLE),
        "forced_throttle_steps": 25, "training_root_seed": 101,
        "torch_inference_threads": torch.get_num_threads(),
        "official_validation_or_test_used": False,
    }
    model, metadata = load_sac_checkpoint(
        args.checkpoint, device="cpu", expected_encoder_fingerprint=Stage5ObservationEncoder(core_only=False).get_schema_fingerprint()
    )
    run_info["optimizer_updates_before"] = int(getattr(model, "_n_updates", 0))
    run_info["policy_sha256_before"] = _policy_sha(model)
    run_info["checkpoint_training_config"] = metadata.get("training_config")

    sampler = TrainCaseSampler(
        training_root_seed=101, strategy=SamplerStrategy.ROUND_ROBIN,
        allowed_geometry_ids=[x["geometry_id"] for x in split["pilot_diagnostic"]], project_root=ROOT,
    )
    env = Stage5TrainingEnv(sampler=sampler, encoder=Stage5ObservationEncoder(core_only=False), project_root=ROOT)
    summaries: List[Dict[str, Any]] = []
    try:
        trace_path = args.output / "steps.jsonl"
        with trace_path.open("w", encoding="utf-8", newline="\n") as trace_sink:
            with (args.output / "episodes.jsonl").open("w", encoding="utf-8", newline="\n") as summary_sink:
                trial_index = 0
                for item in selected:
                    result = _run_episode(env, model, item, "deterministic", trial_seed=0, sink=trace_sink)
                    summaries.append(result)
                    summary_sink.write(json.dumps(result, sort_keys=True) + "\n")
                    summary_sink.flush()
                    print(f"COMPLETE {result['mode']} {result['geometry_id']} outcome={result['outcome']} steps={result['episode_steps']}", flush=True)
                    for _ in range(args.stochastic_rollouts):
                        trial_seed = args.trial_seed_base + trial_index
                        trial_index += 1
                        result = _run_episode(env, model, item, "stochastic", trial_seed=trial_seed, sink=trace_sink)
                        summaries.append(result)
                        summary_sink.write(json.dumps(result, sort_keys=True) + "\n")
                        summary_sink.flush()
                        print(f"COMPLETE {result['mode']} {result['geometry_id']} seed={trial_seed} outcome={result['outcome']} steps={result['episode_steps']}", flush=True)
                    result = _forced_throttle_probe(env, item, trace_sink)
                    summaries.append(result)
                    summary_sink.write(json.dumps(result, sort_keys=True) + "\n")
                    summary_sink.flush()
                    print(f"COMPLETE forced_throttle {result['geometry_id']} speed={result['last_speed_km_h']} route_delta={result['route_delta_from_reset']}", flush=True)
    finally:
        env.close()

    run_info["optimizer_updates_after"] = int(getattr(model, "_n_updates", 0))
    run_info["policy_sha256_after"] = _policy_sha(model)
    run_info["replay_buffer_size"] = (model.replay_buffer.size() if getattr(model, "replay_buffer", None) is not None else None)
    run_info["policy_unchanged"] = run_info["policy_sha256_before"] == run_info["policy_sha256_after"]
    run_info["optimizer_unchanged"] = run_info["optimizer_updates_before"] == run_info["optimizer_updates_after"]
    run_info["episodes"] = summaries
    (args.output / "run_manifest.json").write_text(json.dumps(run_info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"RESULTS_DIR {args.output.resolve()}", flush=True)
    print(f"POLICY_UNCHANGED {run_info['policy_unchanged']} OPTIMIZER_UNCHANGED {run_info['optimizer_unchanged']}", flush=True)
    return 0 if run_info["policy_unchanged"] and run_info["optimizer_unchanged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
