"""Frozen Stage5SacAgent evaluation for the native-reward integration task."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.stage5_sac_agent import Stage5SacAgent
from src.platform.agent import AgentPublicEpisodeContext
from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER, task_producer_identity
from src.training.stage5.checkpoint import verify_checkpoint_integrity
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.native_reward_training import NativeMetaDriveRewardTrainingEnv
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.training.stage5.pilot_observation_transform import (
    GROUPWISE_PHYSICAL_V1,
    GroupwisePhysicalObservation,
    apply_observation_transform,
)
from src.training.stage5.train_sampler import SamplerStrategy, TrainCaseSampler


GEOMETRY_ID = "geom_easy_SCS_seed0"
PRODUCER_ID = REFERENCE_TASK_PRODUCER
TRANSFORM_ID = GROUPWISE_PHYSICAL_V1["transform_id"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_env(seed: int):
    sampler = TrainCaseSampler(
        training_root_seed=seed,
        strategy=SamplerStrategy.FIXED,
        tier_filter="Easy",
        sequence_filter="SCS",
        fixed_geometry_seed=0,
        project_root=ROOT,
    )
    case = sampler.get_case_for_geometry_id(GEOMETRY_ID, environment_seed=seed)
    base = Stage5TrainingEnv(sampler=sampler, project_root=ROOT,
                             task_context_producer_id=PRODUCER_ID, simulator_render_mode="OFF")
    env = GroupwisePhysicalObservation(NativeMetaDriveRewardTrainingEnv(base))
    observation, info = env.reset(seed=seed, options={"case": case})
    if info.get("geometry_id") != GEOMETRY_ID or info.get("tier") != "Easy":
        env.close()
        raise RuntimeError(f"Evaluation reset resolved the wrong TRAIN geometry: {info}")
    if float(info.get("traffic_density", -1)) != 0.0:
        env.close()
        raise RuntimeError("Selected TRAIN geometry is not traffic-free")
    return env, base, case, observation, info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", required=True, help="Comma-separated, pre-registered environment seeds")
    args = parser.parse_args()
    checkpoint_dir = Path(args.checkpoint).resolve()
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    seeds = [int(item) for item in args.seeds.split(",") if item]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("The frozen evaluation requires exactly five unique registered seeds")
    metadata = verify_checkpoint_integrity(
        checkpoint_dir,
        expected_encoder_fingerprint=Stage5ObservationEncoder().get_schema_fingerprint(),
        task_context_producer_id=PRODUCER_ID,
    )
    if metadata.get("training_config", {}).get("training_reward_mode", {}).get("id") != "stage5_metadrive_native_reward_v1":
        raise RuntimeError("Checkpoint is not bound to the registered native Stage 5 reward mode")

    agent = Stage5SacAgent(
        checkpoint_path=checkpoint_dir,
        expected_sha256=metadata["model_sha256"],
        task_context_producer_id=PRODUCER_ID,
    )
    episode_results = []
    all_rows = []
    try:
        for episode_index, seed in enumerate(seeds):
            env, base, case, obs, reset_info = make_env(seed)
            horizon = int(reset_info["horizon_steps"])
            agent.reset(AgentPublicEpisodeContext(horizon_steps=horizon), agent_seed=seed)
            route_start = float(reset_info["initial_route_completion"])
            total_native_reward = 0.0
            total_v1_reward = 0.0
            speeds: list[float] = []
            steering: list[float] = []
            throttle: list[float] = []
            outcome = "INCOMPLETE_EVAL_ERROR"
            terminated = truncated = False
            geometry_sha = case.geometry_sha256
            try:
                for decision in range(horizon):
                    agent_input = base.current_agent_input
                    if agent_input is None:
                        raise RuntimeError("Stage5TrainingEnv did not retain the current public AgentInputV1")
                    decision_obj = agent.act(agent_input, task_context_producer_id=PRODUCER_ID)
                    action = np.asarray(decision_obj.action_payload, dtype=np.float32)
                    if action.shape != (2,) or not np.all(np.isfinite(action)):
                        raise RuntimeError(f"Stage5 actor emitted invalid Box2 action: {action!r}")
                    obs, native_reward, terminated, truncated, info = env.step(action)
                    if not np.allclose(obs, apply_observation_transform(
                        Stage5ObservationEncoder().encode(base.current_agent_input), TRANSFORM_ID
                    ), rtol=0.0, atol=1e-6):
                        raise RuntimeError("Stage5 training/evaluation observation transform parity failed")
                    if abs(float(info["action_steering"]) - float(action[0])) > 1e-6 or abs(float(info["action_throttle_brake"]) - float(action[1])) > 1e-6:
                        raise RuntimeError("Stage5SacAgent action does not match the action executed by MetaDrive")
                    v1_reward = float(info["reward_spec_v1_total"])
                    total_native_reward += float(native_reward)
                    total_v1_reward += v1_reward
                    speed = float(info["speed_km_h"])
                    speeds.append(speed)
                    steering.append(float(action[0]))
                    throttle.append(float(action[1]))
                    ego_position = getattr(base._env.agent, "position", None)
                    row = {
                        "episode_index": episode_index,
                        "environment_seed": seed,
                        "geometry_id": GEOMETRY_ID,
                        "geometry_sha256": geometry_sha,
                        "decision": decision + 1,
                        "steering": float(action[0]),
                        "throttle_brake": float(action[1]),
                        "native_training_reward": float(native_reward),
                        "reward_spec_v1_comparison": v1_reward,
                        "speed_km_h": speed,
                        "route_completion": float(info["route_completion"]),
                        "delta_route": float(info["delta_route"]),
                        "x": float(ego_position[0]) if ego_position is not None else None,
                        "y": float(ego_position[1]) if ego_position is not None else None,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                    all_rows.append(row)
                    if terminated or truncated:
                        raw = info.get("outcome", {})
                        if raw.get("clean_success"):
                            outcome = "CLEAN_SUCCESS"
                        else:
                            reason = str(raw.get("primary_reason", info.get("primary_reason", "UNKNOWN"))).upper()
                            if "OUT_OF_ROAD" in reason:
                                outcome = "OUT_OF_ROAD"
                            elif "CRASH" in reason or "COLLISION" in reason:
                                outcome = "CRASH"
                            elif truncated or "TIMEOUT" in reason or "MAX_STEP" in reason:
                                outcome = "TIMEOUT"
                            else:
                                outcome = reason
                        break
                if not (terminated or truncated):
                    raise RuntimeError(f"Evaluation exceeded official case horizon ({horizon}) without terminal classification")
                final_route = float(info["route_completion"])
                episode_results.append({
                    "episode_index": episode_index,
                    "environment_seed": seed,
                    "case_id": case.case_id,
                    "geometry_id": GEOMETRY_ID,
                    "geometry_sha256": geometry_sha,
                    "outcome": outcome,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "decisions": len(speeds),
                    "route_completion_start": route_start,
                    "route_completion_end": final_route,
                    "route_progress_delta": final_route - route_start,
                    "native_training_return": total_native_reward,
                    "reward_spec_v1_return_comparison": total_v1_reward,
                    "mean_speed_km_h": float(np.mean(speeds)),
                    "peak_speed_km_h": float(np.max(speeds)),
                    "stationary_fraction_lt_1km_h": float(np.mean(np.asarray(speeds) < 1.0)),
                    "mean_steering": float(np.mean(steering)),
                    "steering_std": float(np.std(steering)),
                    "mean_throttle_brake": float(np.mean(throttle)),
                    "positive_throttle_fraction": float(np.mean(np.asarray(throttle) > 0.05)),
                    "brake_fraction": float(np.mean(np.asarray(throttle) < -0.05)),
                })
            finally:
                env.close()
                agent._ready = False
    finally:
        agent.close()

    fields = list(all_rows[0]) if all_rows else []
    trajectory_path = output.with_name(output.stem + "_trajectory.csv")
    with trajectory_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    outcomes = [item["outcome"] for item in episode_results]
    aggregate = {
        "episodes": len(episode_results),
        "clean_success": outcomes.count("CLEAN_SUCCESS"),
        "safety_failure": sum(item["outcome"] in ("OUT_OF_ROAD", "CRASH") for item in episode_results),
        "out_of_road": outcomes.count("OUT_OF_ROAD"),
        "crash": outcomes.count("CRASH"),
        "timeout": outcomes.count("TIMEOUT"),
        "outcomes": outcomes,
        "mean_route_progress_delta": float(np.mean([item["route_progress_delta"] for item in episode_results])),
        "mean_speed_km_h": float(np.mean([item["mean_speed_km_h"] for item in episode_results])),
        "mean_peak_speed_km_h": float(np.mean([item["peak_speed_km_h"] for item in episode_results])),
        "mean_stationary_fraction": float(np.mean([item["stationary_fraction_lt_1km_h"] for item in episode_results])),
        "mean_clean_episode_speed_km_h": (
            float(np.mean([item["mean_speed_km_h"] for item in episode_results if item["outcome"] == "CLEAN_SUCCESS"]))
            if any(item["outcome"] == "CLEAN_SUCCESS" for item in episode_results) else None
        ),
    }
    payload = {
        "protocol_id": "stage5_native_reward_integration_v1",
        "checkpoint": str(checkpoint_dir),
        "model_sha256": metadata["model_sha256"],
        "checkpoint_timesteps": metadata["total_timesteps"],
        "task_context_producer": task_producer_identity(PRODUCER_ID),
        "reward_mode": metadata["training_config"]["training_reward_mode"],
        "evaluation_seeds": seeds,
        "geometry_id": GEOMETRY_ID,
        "geometry_sha256": episode_results[0]["geometry_sha256"],
        "trajectory_path": str(trajectory_path),
        "trajectory_sha256": sha256_file(trajectory_path),
        "episodes": episode_results,
        "summary": aggregate,
    }
    payload["evaluation_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
