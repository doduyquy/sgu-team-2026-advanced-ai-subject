"""Frozen TRAIN-only S5.5 probe/full-episode evaluation and observation audit."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.stage5.checkpoint import verify_checkpoint_integrity
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.training.stage5.pilot_observation_transform import apply_observation_transform
from src.training.stage5.train_sampler import SamplerStrategy, TrainCaseSampler
from scripts.run_stage5_s55_reward_preflight import load_actions
from scripts.run_stage5_s55_training import (
    RESULTS, RUNS, TRANSFORM_ID, load_protocol, state_fingerprint,
)


CHECKPOINT_STEPS = [0, 2000, 5000, 10000, 30000]
PROBE_CAP = 300
ANCHOR_DECISIONS = [0, 25, 60, 100, 150, 220, 300, 400]
GROUPS = {
    "ego": (0, 9),
    "navigation": (9, 19),
    "lidar": (19, 259),
    "traffic": (259, 325),
    "task": (325, 409),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_env(protocol: dict[str, Any]) -> tuple[Stage5TrainingEnv, TrainCaseSampler]:
    ids = [row["geometry_id"] for row in protocol["diagnostic_pool"]]
    sampler = TrainCaseSampler(
        training_root_seed=int(protocol["training_root_seed"]),
        strategy=SamplerStrategy.ROUND_ROBIN,
        allowed_geometry_ids=ids,
        project_root=ROOT,
    )
    env = Stage5TrainingEnv(sampler=sampler, project_root=ROOT, simulator_render_mode="OFF")
    return env, sampler


def get_lateral_offset_m(env: Stage5TrainingEnv) -> float | None:
    try:
        vehicle = env._env.agent
        lane = vehicle.navigation.current_lane
        local = lane.local_coordinates(vehicle.position)
        value = float(local[1])
        return value if math.isfinite(value) else None
    except Exception:
        return None


def action_stats(actions: list[list[float]], speeds: list[float]) -> dict[str, Any]:
    if not actions:
        return {}
    throttle = [a[1] for a in actions]
    steering = [a[0] for a in actions]
    signs = [1 if x > 1e-6 else (-1 if x < -1e-6 else 0) for x in throttle]
    switches = sum(1 for a, b in zip(signs, signs[1:]) if a and b and a != b)
    runs: list[int] = []
    run = 0
    for sign in signs:
        if sign > 0:
            run += 1
        else:
            if run:
                runs.append(run)
            run = 0
    if run:
        runs.append(run)
    longest_moving = 0
    current = 0
    for speed in speeds:
        if speed >= 1.0:
            current += 1
            longest_moving = max(longest_moving, current)
        else:
            current = 0
    quantiles = np.quantile(np.asarray(throttle), [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).tolist()
    return {
        "throttle_mean": statistics.fmean(throttle),
        "throttle_std": statistics.pstdev(throttle) if len(throttle) > 1 else 0.0,
        "throttle_quantiles_0_10_25_50_75_90_100": quantiles,
        "positive_throttle_fraction": sum(x > 0 for x in throttle) / len(throttle),
        "negative_throttle_fraction": sum(x < 0 for x in throttle) / len(throttle),
        "steering_mean": statistics.fmean(steering),
        "steering_std": statistics.pstdev(steering) if len(steering) > 1 else 0.0,
        "steering_saturation_fraction_abs_gt_0_95": sum(abs(x) > 0.95 for x in steering) / len(steering),
        "direct_throttle_brake_sign_switches": switches,
        "median_positive_throttle_run_steps": float(statistics.median(runs)) if runs else 0.0,
        "max_positive_throttle_run_steps": max(runs, default=0),
        "max_sustained_moving_steps_speed_ge_1": longest_moving,
    }


def run_episode(model: SAC, env: Stage5TrainingEnv, case: Any, *, arm: str, checkpoint_step: int,
                checkpoint_sha: str, probe: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_obs, reset_info = env.reset(options={"case": case})
    obs = apply_observation_transform(raw_obs, TRANSFORM_ID)
    initial = float(reset_info["initial_route_completion"])
    raw_observations: list[dict[str, Any]] = []
    actions: list[list[float]] = []
    speeds: list[float] = []
    lateral: list[float] = []
    route_values: list[float] = []
    max_route = initial
    cumulative_reward = 0.0
    terminal_info: dict[str, Any] = {}
    for step in range(1, PROBE_CAP + 1 if probe else case.horizon_steps + 1):
        if step - 1 in {0, 25, 60, 100, 150, 220, 299}:
            raw_array = np.asarray(raw_obs, dtype=np.float32).copy()
            task = raw_array[325:385].reshape(20, 3)
            task_mask = raw_array[385:405] > 0.5
            raw_observations.append({
                "anchor_id": f"{case.case_id}_checkpoint_{checkpoint_step}_{'probe' if probe else 'full'}_step_{step - 1}",
                "step": step - 1,
                "arm": arm,
                "checkpoint_step": checkpoint_step,
                "geometry_id": env.sampler.geometry_id_for_case(case),
                "environment_seed": case.environment_seed,
                "speed_km_h": float(env._env.agent.speed_km_h),
                "task_curvature_proxy_max_abs_heading_rad": float(np.max(np.abs(task[task_mask, 2]))) if np.any(task_mask) else 0.0,
                "raw_observation": raw_array.tolist(),
            })
        raw_action, _ = model.predict(obs, deterministic=True)
        raw_action = np.asarray(raw_action, dtype=np.float32).reshape(2)
        canonical = env.action_adapter.to_canonical(raw_action)
        executed = np.asarray(canonical.to_numpy(), dtype=np.float32)
        if not np.allclose(raw_action, executed, rtol=0.0, atol=1e-7):
            raise RuntimeError("S5.5 inference action adapter changed continuous_box2_v1 action")
        raw_next, reward, terminated, truncated, info = env.step(executed)
        actual = [float(info["action_steering"]), float(info["action_throttle_brake"])]
        if not np.allclose(executed, actual, rtol=0.0, atol=2e-7):
            raise RuntimeError("Action passed to MetaDrive differs from the actor output")
        actions.append(actual)
        speed = float(info["speed_km_h"])
        speeds.append(speed)
        offset = get_lateral_offset_m(env)
        if offset is not None:
            lateral.append(offset)
        progress = float(info["route_completion"])
        route_values.append(progress)
        max_route = max(max_route, progress)
        cumulative_reward += float(reward)
        obs = apply_observation_transform(raw_next, TRANSFORM_ID)
        raw_obs = raw_next
        if terminated or truncated:
            terminal_info = dict(info.get("outcome", {}))
            break
    actual_steps = len(actions)
    done = bool(terminal_info)
    status = terminal_info.get("primary_reason") if done else "PROBE_CENSORED"
    final_route = route_values[-1] if route_values else initial
    result = {
        "arm": arm,
        "checkpoint_step": int(checkpoint_step),
        "checkpoint_model_sha256": checkpoint_sha,
        "case_id": case.case_id,
        "geometry_id": env.sampler.geometry_id_for_case(case),
        "geometry_sha256": case.geometry_sha256,
        "geometry_generation_seed": case.geometry_generation_seed,
        "environment_seed": case.environment_seed,
        "tier": case.tier,
        "sequence": case.sequence,
        "split": case.split,
        "probe": bool(probe),
        "outcome": status,
        "clean_success": bool(terminal_info.get("clean_success", False)),
        "safety_failure": status in {
            "OUT_OF_ROAD", "CRASH_HUMAN", "CRASH_VEHICLE", "CRASH_OBJECT", "CRASH_BUILDING", "CRASH_SIDEWALK"
        },
        "timeout": status == "TIMEOUT",
        "censored": status == "PROBE_CENSORED",
        "steps": actual_steps,
        "initial_route_completion": initial,
        "final_route_completion": final_route,
        "max_route_completion": max_route,
        "route_progress_delta": final_route - initial,
        "route_progress_delta_m": float(env._env.agent.navigation.total_length) * (final_route - initial),
        "mean_speed_km_h": statistics.fmean(speeds) if speeds else 0.0,
        "peak_speed_km_h": max(speeds, default=0.0),
        "stationary_fraction_speed_lt_1_km_h": sum(x < 1.0 for x in speeds) / len(speeds) if speeds else 1.0,
        "mean_abs_lateral_offset_m": statistics.fmean(abs(x) for x in lateral) if lateral else None,
        "max_abs_lateral_offset_m": max((abs(x) for x in lateral), default=None),
        "cumulative_environment_reward": cumulative_reward,
        "action_stats": action_stats(actions, speeds),
        "terminal_flags": terminal_info.get("raw_flags", {}),
    }
    return result, raw_observations


def collect_reference_anchors(protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace = (
        ROOT.parent.parent / f"{ROOT.parent.name}-khoa-s55-readonly" / "autonomous-driving-rl" /
        "runs" / "s55_trace_ref_scs0_20261120" / "current" / "episode_000" / "decisions.jsonl.gz"
    )
    actions = load_actions(trace)
    ids = [r["geometry_id"] for r in protocol["training_geometry_pool"]]
    sampler = TrainCaseSampler(training_root_seed=101, strategy=SamplerStrategy.ROUND_ROBIN,
                               allowed_geometry_ids=ids, project_root=ROOT)
    case = sampler.get_case_for_geometry_id("geom_easy_SCS_seed0", environment_seed=0)
    env = Stage5TrainingEnv(sampler=sampler, project_root=ROOT)
    records: list[dict[str, Any]] = []
    try:
        obs, reset = env.reset(options={"case": case})
        for step in range(len(actions)):
            if step in ANCHOR_DECISIONS:
                raw = np.asarray(obs, dtype=np.float32)
                task = raw[325:385].reshape(20, 3)
                valid = raw[385:405] > 0.5
                records.append({
                    "anchor_id": f"stage1_scs0_env0_decision_{step}",
                    "step": step,
                    "case_id": case.case_id,
                    "geometry_id": "geom_easy_SCS_seed0",
                    "environment_seed": 0,
                    "raw_observation": raw.tolist(),
                    "task_curvature_proxy_max_abs_heading_rad": float(np.max(np.abs(task[valid, 2]))) if np.any(valid) else 0.0,
                    "traffic_active_count": float(np.sum(raw[315:323])),
                })
            obs, _, terminated, truncated, _ = env.step(np.asarray(actions[step], dtype=np.float32))
            if terminated or truncated:
                break
        metadata = {
            "source": "Stage 1 v0.6.1 reference trajectory states; analysis-only, never training/replay",
            "source_trace_sha256": sha256(trace),
            "anchor_decision_indices": [r["step"] for r in records],
            "episode_terminal": bool(terminated or truncated),
            "reset_progress": reset["initial_route_completion"],
        }
    finally:
        env.close()
    return records, metadata


def policy_group_sensitivity(model: SAC, raw_anchors: list[dict[str, Any]]) -> dict[str, Any]:
    if not raw_anchors:
        return {"anchors": 0, "groups": {}}
    actions: list[list[float]] = []
    log_stds: list[list[float]] = []
    activations: list[float] = []
    grad_norms: dict[str, list[list[float]]] = {"steering": [], "throttle": []}
    first_linear = next((m for m in model.policy.actor.latent_pi.modules() if isinstance(m, torch.nn.Linear)), None)
    preactivation: list[torch.Tensor] = []
    handle = None
    if first_linear is not None:
        handle = first_linear.register_forward_hook(lambda _module, _inputs, output: preactivation.append(output.detach()))
    try:
        for anchor in raw_anchors:
            transformed = apply_observation_transform(np.asarray(anchor["raw_observation"], dtype=np.float32), TRANSFORM_ID)
            tensor = torch.as_tensor(transformed, dtype=torch.float32, device=model.device).reshape(1, -1)
            tensor.requires_grad_(True)
            preactivation.clear()
            with torch.enable_grad():
                predicted = model.policy._predict(tensor, deterministic=True)
                params = model.policy.actor.get_action_dist_params(tensor)
                mean = params[0]
                log_std = params[1]
                actions.append(predicted.detach().cpu().numpy().reshape(-1).tolist())
                log_stds.append(log_std.detach().cpu().numpy().reshape(-1).tolist())
                if preactivation:
                    activations.append(float((preactivation[0] > 0).float().mean().item()))
                for action_i in range(2):
                    grad = torch.autograd.grad(predicted[0, action_i], tensor, retain_graph=action_i == 0)[0]
                    per_group = []
                    for start, end in GROUPS.values():
                        segment = grad[0, start:end]
                        per_group.append(float(torch.linalg.vector_norm(segment).detach().cpu().item() / math.sqrt(end - start)))
                    grad_norms["steering" if action_i == 0 else "throttle"].append(per_group)
    finally:
        if handle is not None:
            handle.remove()

    action_array = np.asarray(actions, dtype=np.float64)
    log_std_array = np.asarray(log_stds, dtype=np.float64)
    output_groups: dict[str, Any] = {}
    for i, group in enumerate(GROUPS):
        steering_values = [row[i] for row in grad_norms.get("steering", [])]
        throttle_values = [row[i] for row in grad_norms.get("throttle", [])]
        output_groups[group] = {
            "mean_input_rms_gradient_to_steering": statistics.fmean(steering_values) if steering_values else 0.0,
            "mean_input_rms_gradient_to_throttle": statistics.fmean(throttle_values) if throttle_values else 0.0,
        }
    return {
        "anchors": len(raw_anchors),
        "anchor_ids": [x["anchor_id"] for x in raw_anchors],
        "deterministic_steering_mean": float(np.mean(action_array[:, 0])),
        "deterministic_steering_std_across_real_anchors": float(np.std(action_array[:, 0])),
        "deterministic_throttle_mean": float(np.mean(action_array[:, 1])),
        "deterministic_throttle_std_across_real_anchors": float(np.std(action_array[:, 1])),
        "mean_log_std": np.mean(log_std_array, axis=0).tolist(),
        "mean_action_std_pre_tanh": np.mean(np.exp(log_std_array), axis=0).tolist(),
        "first_actor_linear_relu_active_fraction_mean": statistics.fmean(activations) if activations else None,
        "feature_group_local_jacobian_rms": output_groups,
        "interpretation": "local policy sensitivity on fixed real TRAIN observations; not causal feature importance or driving competence",
    }


def feature_group_summary(anchors: list[dict[str, Any]]) -> dict[str, Any]:
    if not anchors:
        return {}
    matrix = np.stack([np.asarray(a["raw_observation"], dtype=np.float32) for a in anchors])
    groups = {}
    for name, (start, end) in GROUPS.items():
        values = matrix[:, start:end]
        groups[name] = {
            "dimension": end - start,
            "finite_fraction": float(np.isfinite(values).mean()),
            "mean_of_feature_means": float(np.mean(values, axis=0).mean()),
            "mean_feature_std_across_anchors": float(np.std(values, axis=0).mean()),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    traffic_mask = matrix[:, 315:323]
    lidar = matrix[:, 19:259]
    task = matrix[:, 325:385].reshape(len(matrix), 20, 3)
    task_masks = matrix[:, 385:405]
    return {
        "feature_groups": groups,
        "traffic_valid_slots_per_anchor": np.sum(traffic_mask > 0.5, axis=1).astype(int).tolist(),
        "traffic_nonzero_fraction": float(np.count_nonzero(matrix[:, 259:315]) / matrix[:, 259:315].size),
        "lidar_anchor_range_m": [float(np.min(lidar)), float(np.max(lidar))],
        "task_valid_waypoints_per_anchor": np.sum(task_masks > 0.5, axis=1).astype(int).tolist(),
        "task_waypoint_xy_range_m": [float(np.min(task[:, :, :2])), float(np.max(task[:, :, :2]))],
        "task_waypoint_heading_range_rad": [float(np.min(task[:, :, 2])), float(np.max(task[:, :, 2]))],
        "input_correctness": {
            "shape_409": matrix.shape[1] == 409,
            "all_finite": bool(np.isfinite(matrix).all()),
            "traffic_easy_context_empty": bool(np.count_nonzero(matrix[:, 259:323]) == 0),
        },
    }


def episode_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"episodes": 0}
    clean = [r for r in records if r["clean_success"]]
    return {
        "episodes": len(records),
        "clean_successes": len(clean),
        "clean_success_rate": len(clean) / len(records),
        "safety_failures": sum(r["safety_failure"] for r in records),
        "safety_failure_rate": sum(r["safety_failure"] for r in records) / len(records),
        "timeouts": sum(r["timeout"] for r in records),
        "censored": sum(r["censored"] for r in records),
        "mean_final_route_completion": statistics.fmean(r["final_route_completion"] for r in records),
        "median_final_route_completion": statistics.median(r["final_route_completion"] for r in records),
        "mean_route_progress_delta": statistics.fmean(r["route_progress_delta"] for r in records),
        "mean_speed_all_episodes_km_h": statistics.fmean(r["mean_speed_km_h"] for r in records),
        "mean_speed_clean_episodes_km_h": statistics.fmean(r["mean_speed_km_h"] for r in clean) if clean else None,
        "peak_speed_all_episodes_km_h": max(r["peak_speed_km_h"] for r in records),
        "mean_stationary_fraction": statistics.fmean(r["stationary_fraction_speed_lt_1_km_h"] for r in records),
        "mean_steering_std": statistics.fmean(r["action_stats"]["steering_std"] for r in records),
        "mean_positive_throttle_fraction": statistics.fmean(r["action_stats"]["positive_throttle_fraction"] for r in records),
        "mean_direct_sign_switches_per_episode": statistics.fmean(
            r["action_stats"]["direct_throttle_brake_sign_switches"] for r in records
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=RESULTS / "protocol.json")
    parser.add_argument("--arm", choices=["A", "B", "C", "ALL"], default="ALL")
    parser.add_argument("--checkpoints", default="0,2000,5000,10000,30000")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    arms = ["A", "B", "C"] if args.arm == "ALL" else [args.arm]
    steps_to_eval = [int(value) for value in args.checkpoints.split(",") if value]
    evaluation_root = RUNS / "evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    anchor_records, anchor_source = collect_reference_anchors(protocol)
    with gzip.open(evaluation_root / "policy_response_anchors.jsonl.gz", "wt", encoding="utf-8") as f:
        for anchor in anchor_records:
            f.write(json.dumps(anchor, separators=(",", ":")) + "\n")
    raw_anchor_info = {"source": anchor_source, "feature_summary": feature_group_summary(anchor_records)}

    all_records: list[dict[str, Any]] = []
    all_observation_anchors: list[dict[str, Any]] = []
    checkpoint_summaries: list[dict[str, Any]] = []
    for arm in arms:
        for checkpoint_step in steps_to_eval:
            checkpoint_dir = RUNS / "arms" / arm / "checkpoints" / f"step_{checkpoint_step:06d}"
            if not checkpoint_dir.is_dir():
                continue
            integrity = verify_checkpoint_integrity(checkpoint_dir)
            metadata = json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))
            if metadata.get("observation_transform") != TRANSFORM_ID:
                raise RuntimeError(f"Checkpoint {checkpoint_dir} has unexpected observation transform")
            model = SAC.load(str(checkpoint_dir / "model.zip"), device="cpu")
            model_hash_before = state_fingerprint(model)
            num_before = int(model.num_timesteps)
            updates_before = int(getattr(model, "_n_updates", 0))
            replay_size_before = int(model.replay_buffer.size())
            sensitivity = policy_group_sensitivity(model, anchor_records)
            checkpoint_observation_anchors: list[dict[str, Any]] = []
            env, sampler = make_env(protocol)
            try:
                checkpoint_records = []
                for case_spec in protocol["diagnostic_pool"]:
                    case = sampler.get_case_for_geometry_id(case_spec["geometry_id"], case_spec["environment_seed"])
                    record, observed = run_episode(model, env, case, arm=arm, checkpoint_step=checkpoint_step,
                                                   checkpoint_sha=integrity["model_sha256"], probe=True)
                    checkpoint_records.append(record)
                    all_records.append(record)
                    checkpoint_observation_anchors.extend(observed)
                    if checkpoint_step in (10000, 30000):
                        full, _ = run_episode(model, env, case, arm=arm, checkpoint_step=checkpoint_step,
                                              checkpoint_sha=integrity["model_sha256"], probe=False)
                        checkpoint_records.append(full)
                        all_records.append(full)
                if state_fingerprint(model) != model_hash_before:
                    raise RuntimeError("Frozen evaluation changed model policy weights")
                if int(model.num_timesteps) != num_before or int(getattr(model, "_n_updates", 0)) != updates_before:
                    raise RuntimeError("Frozen evaluation changed training counters or optimizer updates")
                replay_size_after = int(model.replay_buffer.size())
                if replay_size_after != replay_size_before:
                    raise RuntimeError("Frozen evaluation unexpectedly wrote to its empty replay buffer")
                all_observation_anchors.extend(checkpoint_observation_anchors)
                probes = [r for r in checkpoint_records if r["probe"]]
                full = [r for r in checkpoint_records if not r["probe"]]
                def mean(key: str, records: list[dict[str, Any]]) -> float | None:
                    return statistics.fmean(float(x[key]) for x in records) if records else None
                summary = {
                    "arm": arm,
                    "checkpoint_step": checkpoint_step,
                    "checkpoint_model_sha256": integrity["model_sha256"],
                    "checkpoint_integrity": "PASS",
                    "source_commit": metadata.get("git_commit"),
                    "protocol_sha256": protocol["protocol_sha256"],
                    "probe_cases": len(probes),
                    "probe_censored_cases": sum(r["censored"] for r in probes),
                    "probe_clean_successes": sum(r["clean_success"] for r in probes),
                    "probe_safety_failures": sum(r["safety_failure"] for r in probes),
                    "probe_mean_speed_km_h": mean("mean_speed_km_h", probes),
                    "probe_mean_route_progress_delta": mean("route_progress_delta", probes),
                    "probe_mean_stationary_fraction": mean("stationary_fraction_speed_lt_1_km_h", probes),
                    "full_episode_count": len(full),
                    "full_clean_successes": sum(r["clean_success"] for r in full),
                    "full_safety_failures": sum(r["safety_failure"] for r in full),
                    "full_timeouts": sum(r["timeout"] for r in full),
                    "full_mean_final_route_completion": mean("final_route_completion", full),
                    "full_mean_route_progress_delta": mean("route_progress_delta", full),
                    "full_mean_speed_km_h": mean("mean_speed_km_h", full),
                    "full_mean_stationary_fraction": mean("stationary_fraction_speed_lt_1_km_h", full),
                    "probe_metrics_by_tier": {
                        tier: episode_metrics([r for r in probes if r["tier"] == tier])
                        for tier in sorted({r["tier"] for r in probes})
                    },
                    "probe_metrics_overall": episode_metrics(probes),
                    "full_metrics_by_tier": {
                        tier: episode_metrics([r for r in full if r["tier"] == tier])
                        for tier in sorted({r["tier"] for r in full})
                    },
                    "full_metrics_overall": episode_metrics(full),
                    "actor_responsiveness": sensitivity,
                    "actor_responsiveness_on_own_probe_states": policy_group_sensitivity(
                        model, checkpoint_observation_anchors
                    ),
                    "own_probe_observation_feature_summary": feature_group_summary(checkpoint_observation_anchors),
                    "evaluation_isolation": {
                        "policy_weights_unchanged": True,
                        "num_timesteps_unchanged": True,
                        "optimizer_updates_unchanged": True,
                        "replay_buffer_size_before": replay_size_before,
                        "replay_buffer_size_after": replay_size_after,
                        "no_training_env_or_replay_writes": True,
                        "source_model_hash_before_after": model_hash_before,
                    },
                }
                checkpoint_summaries.append(summary)
                print(json.dumps({"event": "evaluation_checkpoint", **summary}, ensure_ascii=False), flush=True)
            finally:
                env.close()

    records_path = evaluation_root / "per_episode.jsonl.gz"
    with gzip.open(records_path, "wt", encoding="utf-8", compresslevel=6) as f:
        for record in all_records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    observed_anchors_path = evaluation_root / "policy_response_eval_states.jsonl.gz"
    with gzip.open(observed_anchors_path, "wt", encoding="utf-8", compresslevel=6) as f:
        for anchor in all_observation_anchors:
            f.write(json.dumps(anchor, separators=(",", ":")) + "\n")
    summary_out = {
        "protocol_sha256": protocol["protocol_sha256"],
        "source_commit": protocol["source_commit"],
        "benchmark_eligible": False,
        "split": "TRAIN_ONLY_PILOT_DIAGNOSTIC",
        "censored_is_not_timeout": True,
        "feature_anchor_source": raw_anchor_info,
        "checkpoint_summaries": checkpoint_summaries,
        "policy_response_eval_states_path": str(observed_anchors_path.relative_to(ROOT)).replace("\\", "/"),
        "policy_response_eval_states_sha256": sha256(observed_anchors_path),
        "per_episode_path": str(records_path.relative_to(ROOT)).replace("\\", "/"),
        "per_episode_sha256": sha256(records_path),
        "per_episode_size_bytes": records_path.stat().st_size,
        "holdouts_accessed": False,
    }
    out_path = RESULTS / "evaluation_summary.json"
    out_path.write_text(json.dumps(summary_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"result_path": str(out_path), "checkpoint_count": len(checkpoint_summaries),
                      "episode_count": len(all_records)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
