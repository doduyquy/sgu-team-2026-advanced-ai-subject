from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import SAC

from train import FIXED_ENV_CONFIG, NativeTelemetry, geometry_fingerprint, sha256_file
from metadrive.envs.metadrive_env import MetaDriveEnv


def classify(info: dict[str, Any], truncated: bool) -> str:
    out = bool(info.get("out_of_road", False))
    crash = any(bool(info.get(key, False)) for key in (
        "crash", "crash_vehicle", "crash_object", "crash_building", "crash_sidewalk", "crash_human"
    ))
    if bool(info.get("arrive_dest", False)) and not out and not crash:
        return "CLEAN_SUCCESS"
    if out:
        return "OUT_OF_ROAD"
    if crash:
        return "CRASH"
    if truncated or bool(info.get("max_step", False)) or bool(info.get("TimeLimit.truncated", False)):
        return "TIMEOUT"
    return "UNKNOWN_TERMINAL"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--geometry", required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",") if value]
    model = SAC.load(str(checkpoint), device="cpu")
    results = []
    geometry_hashes = []
    geometry = json.loads(Path(args.geometry).read_text(encoding="utf-8"))["block_sequence"]
    eval_config = dict(FIXED_ENV_CONFIG)
    eval_config["map"] = 3
    eval_config["map_config"] = {
        "type": "pg_map_file",
        "config": copy.deepcopy(geometry),
        "lane_width": 3.5,
        "lane_num": 3,
        "exit_length": 50,
        "start_position": [0, 0],
    }
    eval_config["random_spawn_lane_index"] = True
    eval_config["start_seed"] = seeds[0]
    eval_config["num_scenarios"] = len(seeds)
    for episode_index, seed in enumerate(seeds):
        env = NativeTelemetry(MetaDriveEnv(copy.deepcopy(eval_config)))
        rows = []
        total_reward = 0.0
        route_start = None
        outcome = "UNKNOWN_TERMINAL"
        terminated = truncated = False
        obs, reset_info = env.reset(seed=seed)
        geometry_hashes.append(geometry_fingerprint(env))
        reset_telemetry = reset_info.get("native_telemetry", {})
        route_start = reset_telemetry.get("route_completion")
        for decision in range(int(env.unwrapped.config["horizon"])):
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            telem = info.get("native_telemetry", {})
            rows.append({
                "episode_index": episode_index,
                "environment_seed": seed,
                "decision": decision + 1,
                "steering": float(action[0]),
                "throttle_brake": float(action[1]),
                "reward_native": float(reward),
                "x": telem.get("x"), "y": telem.get("y"), "heading": telem.get("heading"),
                "speed_km_h": telem.get("speed_km_h"), "lateral_error_m": telem.get("lateral_error_m"),
                "lane_longitudinal_m": telem.get("lane_longitudinal_m"),
                "route_completion": telem.get("route_completion"),
                "terminated": bool(terminated), "truncated": bool(truncated),
            })
            if terminated or truncated:
                outcome = classify(info, bool(truncated))
                break
        if not (terminated or truncated):
            outcome = "INCOMPLETE_EVAL_ERROR"
        env.close()
        trajectory_path = out.parent / f"{out.stem}_episode_{episode_index + 1:02d}_seed_{seed}.csv"
        with trajectory_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["decision"])
            writer.writeheader()
            writer.writerows(rows)
        speeds = np.array([r["speed_km_h"] for r in rows if r["speed_km_h"] is not None], dtype=np.float64)
        route_end = rows[-1]["route_completion"] if rows else route_start
        progress = None if route_start is None or route_end is None else float(route_end - route_start)
        failures = outcome in {"OUT_OF_ROAD", "CRASH"}
        results.append({
            "episode_index": episode_index,
            "environment_seed": seed,
            "outcome": outcome,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "decisions": len(rows),
            "native_episode_return": total_reward,
            "route_completion_start": route_start,
            "route_completion_end": route_end,
            "route_progress_delta": progress,
            "mean_speed_km_h": float(speeds.mean()) if len(speeds) else None,
            "peak_speed_km_h": float(speeds.max()) if len(speeds) else None,
            "stationary_fraction_speed_lt_1km_h": float(np.mean(speeds < 1.0)) if len(speeds) else None,
            "failure": failures,
            "trajectory_path": str(trajectory_path),
            "trajectory_sha256": sha256_file(trajectory_path),
        })
    outcomes = [item["outcome"] for item in results]
    if len(set(geometry_hashes)) != 1:
        raise RuntimeError(f"Evaluation seeds generated different geometry: {geometry_hashes}")
    def mean_for(field):
        values = [item[field] for item in results if item[field] is not None]
        return float(np.mean(values)) if values else None
    summary = {
        "clean_success": outcomes.count("CLEAN_SUCCESS"),
        "out_of_road": outcomes.count("OUT_OF_ROAD"),
        "crash": outcomes.count("CRASH"),
        "timeout": outcomes.count("TIMEOUT"),
        "other": len(outcomes) - sum(outcomes.count(label) for label in ("CLEAN_SUCCESS", "OUT_OF_ROAD", "CRASH", "TIMEOUT")),
        "outcomes": outcomes,
        "mean_route_progress_delta": mean_for("route_progress_delta"),
        "mean_speed_km_h": mean_for("mean_speed_km_h"),
        "mean_peak_speed_km_h": mean_for("peak_speed_km_h"),
        "mean_stationary_fraction": mean_for("stationary_fraction_speed_lt_1km_h"),
        "mean_clean_episode_speed_km_h": mean(
            item["mean_speed_km_h"] for item in results if item["outcome"] == "CLEAN_SUCCESS" and item["mean_speed_km_h"] is not None
        ),
    }
    payload = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "step": args.step,
        "mode": "deterministic",
        "native_metadrive_version": __import__("metadrive.version", fromlist=["VERSION"]).VERSION,
        "environment_config": eval_config,
        "evaluation_seeds": seeds,
        "geometry_config_sha256": hashlib.sha256(Path(args.geometry).read_bytes()).hexdigest(),
        "geometry_sha256": geometry_hashes[0] if geometry_hashes else None,
        "results": results,
        "summary": summary,
    }
    payload["evaluation_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


def mean(values):
    values = list(values)
    return float(np.mean(values)) if values else None


if __name__ == "__main__":
    main()
