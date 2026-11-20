"""Evaluate frozen P2 SAC checkpoints on the locked TRAIN diagnostic schedule."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.platform.action_adapter import get_action_adapter
from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER
from src.training.stage5.checkpoint import load_sac_checkpoint, verify_checkpoint_integrity
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.training.stage5.pilot_observation_transform import GROUPWISE_PHYSICAL_V1, GroupwisePhysicalObservation
from src.training.stage5.train_sampler import SamplerStrategy, TrainCaseSampler


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "min": float(np.min(values)), "max": float(np.max(values))}


def make_eval_env(protocol: dict[str, Any]):
    cases = protocol["geometry"]["diagnostic_pool"]
    sampler = TrainCaseSampler(
        training_root_seed=protocol["training"]["root_seed"],
        strategy=SamplerStrategy.ROUND_ROBIN,
        allowed_geometry_ids=[x["geometry_id"] for x in cases], project_root=ROOT,
    )
    base = Stage5TrainingEnv(
        sampler=sampler, training_root_seed=protocol["training"]["root_seed"],
        project_root=ROOT, simulator_render_mode="OFF",
        task_context_producer_id=REFERENCE_TASK_PRODUCER,
    )
    return GroupwisePhysicalObservation(base), base, sampler


def actor_state_diagnostics(model, observations: list[np.ndarray]) -> dict[str, Any]:
    if not observations:
        return {"observation_count": 0, "status": "NO_REAL_OBSERVATIONS"}
    obs = np.asarray(observations, dtype=np.float32)
    actions, _ = model.predict(obs, deterministic=True)
    result: dict[str, Any] = {
        "observation_count": int(len(obs)),
        "deterministic_action_mean": np.mean(actions, axis=0).astype(float).tolist(),
        "deterministic_action_std_across_real_states": np.std(actions, axis=0).astype(float).tolist(),
        "observation_group_std_mean": {
            "ego_core_9d": float(np.std(obs[:, :9], axis=0).mean()),
            "navigation_10d": float(np.std(obs[:, 9:19], axis=0).mean()),
            "lidar_240d": float(np.std(obs[:, 19:259], axis=0).mean()),
            "traffic_actor_56d": float(np.std(obs[:, 259:315], axis=0).mean()),
            "traffic_masks_metrics_10d": float(np.std(obs[:, 315:325], axis=0).mean()),
            "task_waypoints_60d": float(np.std(obs[:, 325:385], axis=0).mean()),
            "task_masks_scalars_24d": float(np.std(obs[:, 385:409], axis=0).mean()),
        },
    }
    try:
        actor = model.policy.actor
        tensor, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            features = actor.extract_features(tensor, actor.features_extractor)
            first_linear = actor.latent_pi[0]
            pre = first_linear(features)
            activation = actor.latent_pi[1](pre)
        result["first_layer_nonpositive_fraction"] = float((pre <= 0).float().mean().cpu())
        result["first_layer_preactivation_std_mean"] = float(pre.std(dim=0, unbiased=False).mean().cpu())
        result["first_layer_activation_std_mean"] = float(activation.std(dim=0, unbiased=False).mean().cpu())
    except (AttributeError, IndexError, TypeError):
        result["first_layer_status"] = "UNAVAILABLE_FOR_POLICY_ARCHITECTURE"
    return result


def run_one(checkpoint: Path, protocol: dict[str, Any], output: Path, cap: int | None) -> dict[str, Any]:
    encoder = Stage5ObservationEncoder(core_only=False)
    metadata = verify_checkpoint_integrity(
        checkpoint, expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
        task_context_producer_id=REFERENCE_TASK_PRODUCER,
    )
    transform = metadata.get("observation_transform") or {}
    if transform != GROUPWISE_PHYSICAL_V1:
        raise ValueError("P2 checkpoint does not use the preregistered groupwise transform")
    if metadata.get("observation_transform_fingerprint") != protocol["observation"]["transform_fingerprint"]:
        raise ValueError("P2 checkpoint transform fingerprint differs from frozen protocol")
    model, metadata = load_sac_checkpoint(
        checkpoint, device="cpu", expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
        task_context_producer_id=REFERENCE_TASK_PRODUCER,
    )
    env, base, sampler = make_eval_env(protocol)
    adapter = get_action_adapter("continuous_box2_v1")
    case_rows: list[dict[str, Any]] = []
    real_obs: list[np.ndarray] = []
    started_all = time.perf_counter()
    try:
        for case in protocol["geometry"]["diagnostic_pool"]:
            resolved = sampler.get_case_for_geometry_id(case["geometry_id"], int(case["environment_seed"]))
            obs, reset_info = env.reset(options={"case": resolved})
            initial_progress = float(reset_info.get("initial_route_completion", base._last_route_completion))
            speeds: list[float] = []
            actions: list[list[float]] = []
            progress: list[float] = []
            rewards: list[float] = []
            observations: list[np.ndarray] = []
            done, terminated, truncated = False, False, False
            terminal_info: dict[str, Any] = {}
            limit = int(cap) if cap is not None else int(resolved.horizon_steps) + 1
            while not done and len(speeds) < limit:
                state = np.asarray(obs, dtype=np.float32).copy()
                if len(speeds) % 5 == 0:
                    observations.append(state)
                    real_obs.append(state)
                raw, _ = model.predict(state, deterministic=True)
                canonical = adapter.to_canonical(raw)
                action = [float(canonical.steering), float(canonical.throttle_brake)]
                actions.append(action)
                obs, reward, terminated, truncated, info = env.step(action)
                speeds.append(float(info.get("speed_km_h", 0.0)))
                progress.append(float(info.get("delta_route", 0.0)))
                rewards.append(float(reward))
                done = bool(terminated or truncated)
                terminal_info = info
            censored = not done
            outcome_obj = terminal_info.get("outcome", {})
            primary = outcome_obj.get("primary_reason", terminal_info.get("primary_reason")) if not censored else "PROBE_CENSORED"
            clean = bool(outcome_obj.get("clean_success", terminal_info.get("clean_success", False))) if not censored else False
            safe_failure = bool(outcome_obj.get("terminated", terminated) and not clean) if not censored else False
            timeout = bool(truncated or terminal_info.get("TimeLimit.truncated", False)) if not censored else False
            throttle = [x[1] for x in actions]
            steering = [x[0] for x in actions]
            moving = [s >= 1.0 for s in speeds]
            max_move, cur = 0, 0
            for flag in moving:
                cur = cur + 1 if flag else 0
                max_move = max(max_move, cur)
            switches = sum(1 for left, right in zip(throttle, throttle[1:]) if left * right < 0)
            row = {
                "geometry_id": case["geometry_id"], "geometry_sha256": case["geometry_sha256"],
                "environment_seed": case["environment_seed"], "tier": resolved.tier,
                "checkpoint_step": int(metadata["total_timesteps"]), "checkpoint_sha256": metadata["model_sha256"],
                "decisions": len(speeds), "probe_censored": censored,
                "outcome": primary, "clean_success": clean, "safety_failure": safe_failure,
                "timeout": timeout, "mean_speed_kmh": float(np.mean(speeds)) if speeds else 0.0,
                "peak_speed_kmh": float(np.max(speeds)) if speeds else 0.0,
                "stationary_fraction_below_1kmh": float(np.mean(np.asarray(speeds) < 1.0)) if speeds else 1.0,
                "max_sustained_moving_decisions": max_move,
                "route_progress_delta": float(sum(progress)),
                "final_route_completion": float(terminal_info.get("route_completion", initial_progress)),
                "initial_route_completion": initial_progress,
                "mean_throttle_brake": float(np.mean(throttle)) if throttle else 0.0,
                "positive_throttle_fraction": float(np.mean(np.asarray(throttle) > 0)) if throttle else 0.0,
                "brake_fraction": float(np.mean(np.asarray(throttle) < 0)) if throttle else 0.0,
                "throttle_sign_switches": int(switches),
                "steering_std": float(statistics.pstdev(steering)) if len(steering) > 1 else 0.0,
                "reward_sum": float(sum(rewards)),
                "action_behavior": {
                    "steering_mean": float(np.mean(steering)) if steering else 0.0,
                    "throttle_mean": float(np.mean(throttle)) if throttle else 0.0,
                },
                "sampled_real_observation_count": len(observations),
            }
            case_rows.append(row)
    finally:
        env.close()
    if sampler.sampled_count != 0:
        raise AssertionError("Evaluation mutated the sampler's training case cursor")
    output.mkdir(parents=True, exist_ok=True)
    obs_path = output / f"actor_observations_step_{int(metadata['total_timesteps']):05d}_{'probe' if cap else 'full'}.npy"
    np.save(obs_path, np.asarray(real_obs, dtype=np.float32))
    diagnostics = actor_state_diagnostics(model, real_obs)
    statuses = Counter(row["outcome"] for row in case_rows)
    result = {
        "protocol": protocol["protocol"], "protocol_sha256": protocol["protocol_sha256"],
        "checkpoint_sha256": metadata["model_sha256"], "checkpoint_identity": metadata.get("checkpoint_identity"),
        "training_step": metadata["total_timesteps"], "producer": metadata["task_context_producer"],
        "observation_transform_fingerprint": metadata["observation_transform_fingerprint"],
        "evaluation_mode": "deterministic", "evaluation_cap": cap,
        "case_count": len(case_rows), "outcome_counts": dict(statuses), "cases": case_rows,
        "aggregate": {
            "mean_speed_kmh": summarize([float(x["mean_speed_kmh"]) for x in case_rows]),
            "route_progress_delta": summarize([float(x["route_progress_delta"]) for x in case_rows]),
            "stationary_fraction": summarize([float(x["stationary_fraction_below_1kmh"]) for x in case_rows]),
            "clean_successes": sum(bool(x["clean_success"]) for x in case_rows),
            "safety_failures": sum(bool(x["safety_failure"]) for x in case_rows),
            "timeouts": sum(bool(x["timeout"]) for x in case_rows),
            "censored": sum(bool(x["probe_censored"]) for x in case_rows),
        },
        "actor_responsiveness": diagnostics, "sampled_observations_sha256": sha256(obs_path),
        "sampler_sampled_count": sampler.sampled_count,
        "wall_time_seconds": time.perf_counter() - started_all,
        "benchmark_eligible": False, "official_validation_test_accessed": False,
    }
    suffix = f"step_{int(metadata['total_timesteps']):05d}_{'probe' if cap else 'full'}"
    path = output / f"{suffix}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {"result_path": path.as_posix(), "result_sha256": sha256(path), **result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "results/stage5/p2_correlated_exploration_seed101_20261120/protocol.json")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/stage5_p2/s5_p2_correlated_exploration_seed101_20261120")
    parser.add_argument("--arm", choices=("A", "B", "BOTH"), default="BOTH")
    parser.add_argument("--checkpoint-step", type=int, choices=(0, 2000, 5000, 10000), action="append")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    digest = protocol.pop("protocol_sha256", None)
    canonical = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    protocol["protocol_sha256"] = digest
    if digest != canonical or protocol.get("status") != "PRE_REGISTERED":
        raise ValueError("P2 protocol hash/status validation failed")
    arms = ("A", "B") if args.arm == "BOTH" else (args.arm,)
    steps = args.checkpoint_step or [0, 2000, 5000, 10000]
    for arm in arms:
        for step in steps:
            checkpoint = args.run_dir / f"arm_{arm}/checkpoints/step_{step:06d}"
            cap = 300
            run_one(checkpoint, protocol, args.run_dir / "evaluation" / f"arm_{arm}", cap)
            print(f"{arm} step={step} fast TRAIN probe complete", flush=True)
            if step == 10000:
                run_one(checkpoint, protocol, args.run_dir / "evaluation" / f"arm_{arm}", None)
                print(f"{arm} step={step} full TRAIN diagnostic complete", flush=True)


if __name__ == "__main__":
    main()
