"""Evaluate frozen P2-F checkpoints on curriculum tasks and original TRAIN routes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import evaluate_stage5_p2 as p2eval
from scripts import run_stage5_p2f_curriculum as p2f
from scripts import run_stage5_p2d_actor_activation as p2d
from src.platform.action_adapter import get_action_adapter
from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER
from src.training.stage5.actor_activation_ablation import ACTOR_LEAKY_RELU_001_V1, activation_inventory
from src.training.stage5.checkpoint import load_sac_checkpoint, verify_checkpoint_integrity
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.training.stage5.pilot_observation_transform import GROUPWISE_PHYSICAL_V1, GroupwisePhysicalObservation
from src.training.stage5.start_curriculum import CURRICULUM_TASK_ID, CURRICULUM_FINGERPRINT
from src.training.stage5.train_sampler import SamplerStrategy, TrainCaseSampler


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_curriculum_eval(checkpoint: Path, protocol: dict[str, Any], output: Path, model) -> dict[str, Any]:
    cases = protocol["geometry"]["diagnostic_pool"]
    sampler = TrainCaseSampler(
        training_root_seed=101, strategy=SamplerStrategy.ROUND_ROBIN,
        allowed_geometry_ids=[item["geometry_id"] for item in cases], project_root=ROOT,
    )
    base = Stage5TrainingEnv(
        sampler=sampler, training_root_seed=101, project_root=ROOT,
        simulator_render_mode="OFF", task_context_producer_id=REFERENCE_TASK_PRODUCER,
        training_start_curriculum=True,
    )
    env = GroupwisePhysicalObservation(base)
    adapter = get_action_adapter("continuous_box2_v1")
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for episode_stage, target_m in ((0, 60.0), (3, 120.0)):
            for item in cases:
                case = sampler.get_case_for_geometry_id(item["geometry_id"], int(item["environment_seed"]))
                obs, reset_info = env.reset(options={"case": case, "curriculum_episode_index": episode_stage})
                speeds: list[float] = []
                steering: list[float] = []
                throttle: list[float] = []
                progress: list[float] = []
                done = False
                info: dict[str, Any] = {}
                while not done and len(speeds) <= int(case.horizon_steps):
                    raw, _ = model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
                    canonical = adapter.to_canonical(raw)
                    action = [float(canonical.steering), float(canonical.throttle_brake)]
                    obs, reward, terminated, truncated, info = env.step(action)
                    speeds.append(float(info["speed_km_h"]))
                    steering.append(action[0])
                    throttle.append(action[1])
                    progress.append(float(info["delta_route"]))
                    done = bool(terminated or truncated)
                outcome = info.get("outcome", {}) if done else {}
                is_censored = not done
                speed_array = np.asarray(speeds, dtype=np.float64)
                throttle_array = np.asarray(throttle, dtype=np.float64)
                rows.append({
                    "checkpoint_sha256": verify_checkpoint_integrity(
                        checkpoint, task_context_producer_id=REFERENCE_TASK_PRODUCER,
                    )["model_sha256"],
                    "geometry_id": item["geometry_id"], "geometry_sha256": item["geometry_sha256"],
                    "environment_seed": int(item["environment_seed"]), "split": "TRAIN/PILOT_DIAGNOSTIC",
                    "evaluation_track": "training_curriculum_competence_only",
                    "training_task_id": reset_info["training_task_id"],
                    "training_task_fingerprint": reset_info["training_task_fingerprint"],
                    "curriculum_stage_id": reset_info["curriculum_stage_id"],
                    "requested_remaining_route_distance_m": target_m,
                    "effective_remaining_route_distance_m": reset_info["effective_remaining_route_distance_m"],
                    "destination": reset_info["original_destination"],
                    "steps": len(speeds), "outcome": "PROBE_CENSORED" if is_censored else outcome.get("primary_reason"),
                    "clean_success": bool(outcome.get("clean_success", False)) if not is_censored else False,
                    "safety_failure": bool(outcome.get("terminated", False) and not outcome.get("clean_success", False)) if not is_censored else False,
                    "timeout": bool(outcome.get("truncated", False)) if not is_censored else False,
                    "censored": is_censored,
                    "route_progress_delta": float(sum(progress)),
                    "mean_speed_kmh": float(speed_array.mean()) if len(speed_array) else 0.0,
                    "peak_speed_kmh": float(speed_array.max()) if len(speed_array) else 0.0,
                    "stationary_fraction_below_1kmh": float(np.mean(speed_array < 1.0)) if len(speed_array) else 1.0,
                    "longest_moving_run_decisions": _longest_run(speed_array >= 1.0),
                    "mean_steering": float(np.mean(steering)) if steering else 0.0,
                    "steering_std": float(np.std(steering)) if steering else 0.0,
                    "mean_throttle_brake": float(throttle_array.mean()) if len(throttle_array) else 0.0,
                    "positive_throttle_fraction": float(np.mean(throttle_array > 0.0)) if len(throttle_array) else 0.0,
                    "brake_fraction": float(np.mean(throttle_array < 0.0)) if len(throttle_array) else 0.0,
                    "terminal_flags": outcome.get("raw_flags", {}) if not is_censored else {},
                })
    finally:
        env.close()
    payload = {
        "protocol_sha256": protocol["protocol_sha256"], "checkpoint_sha256": sha256(checkpoint / "model.zip"),
        "training_task_id": CURRICULUM_TASK_ID, "training_task_fingerprint": CURRICULUM_FINGERPRINT,
        "evaluation_split": "TRAIN/PILOT_DIAGNOSTIC only", "clean_successes": sum(x["clean_success"] for x in rows),
        "safety_failures": sum(x["safety_failure"] for x in rows), "timeouts": sum(x["timeout"] for x in rows),
        "censored": sum(x["censored"] for x in rows), "case_count": len(rows),
        "wall_time_seconds": time.perf_counter() - started, "cases": rows,
        "interpretation": "curriculum competence is separate from original full-route transfer and is not benchmark performance",
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "curriculum_full_episodes.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    payload["result_path"] = str(path)
    payload["result_sha256"] = sha256(path)
    return payload


def _longest_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=p2f.RUN_ID)
    args = parser.parse_args()
    protocol = p2f.load_protocol()
    run_root = ROOT / "runs/stage5_p2f" / args.run_id
    if not (run_root / "paired_training_summary.json").is_file():
        raise FileNotFoundError("P2-F paired training is incomplete; no evaluation performed")
    anchor_path = ROOT / "results/stage5/p2c_representation_critic_audit_20261121/fixed_train_anchors.npy"
    anchors = np.load(anchor_path, allow_pickle=False)
    if anchors.shape != (201, 409) or sha256(anchor_path) != protocol["anchors"]["sha256"]:
        raise ValueError("P2-F fixed TRAIN anchor artifact mismatch")
    encoder = Stage5ObservationEncoder(core_only=False)
    output_root = run_root / "evaluation"
    for arm in ("A", "B"):
        arm_result = output_root / f"arm_{arm}"
        arm_result.mkdir(parents=True, exist_ok=True)
        for step in (0, 2000, 5000, 10000):
            checkpoint = run_root / f"arm_{arm}/checkpoints/step_{step:06d}"
            meta = verify_checkpoint_integrity(
                checkpoint, expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
                task_context_producer_id=REFERENCE_TASK_PRODUCER,
            )
            if meta.get("training_config", {}).get("training_task_fingerprint") != protocol["training_tasks"][arm]["fingerprint"]:
                raise RuntimeError("Checkpoint training-task provenance differs from registered arm")
            if meta.get("actor_activation") != ACTOR_LEAKY_RELU_001_V1:
                raise RuntimeError("P2-F checkpoint actor activation is not the shared LeakyReLU policy")
            probe = p2eval.run_one(checkpoint, protocol, arm_result / f"step_{step:05d}_transfer", cap=300)
            obs_path = arm_result / f"step_{step:05d}_transfer" / f"actor_observations_step_{step:05d}_probe.npy"
            real_obs = np.load(obs_path, allow_pickle=False)
            fixed_metrics = p2d.audit_observations(__import__(
                "src.training.stage5.checkpoint", fromlist=["load_sac_checkpoint"]
            ).load_sac_checkpoint(
                checkpoint, expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
                task_context_producer_id=REFERENCE_TASK_PRODUCER,
                expected_actor_activation_id=ACTOR_LEAKY_RELU_001_V1,
            )[0], anchors, "P2-C-fixed-TRAIN-anchors")
            real_metrics = None
            if len(real_obs):
                model_for_analysis = __import__(
                    "src.training.stage5.checkpoint", fromlist=["load_sac_checkpoint"]
                ).load_sac_checkpoint(
                    checkpoint, expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
                    task_context_producer_id=REFERENCE_TASK_PRODUCER,
                    expected_actor_activation_id=ACTOR_LEAKY_RELU_001_V1,
                )[0]
                real_metrics = p2d.audit_observations(model_for_analysis, real_obs, "frozen-original-route-probe-states")
                del model_for_analysis
            full = p2eval.run_one(checkpoint, protocol, arm_result / f"step_{step:05d}_transfer", cap=None) if step == 10000 else None
            row = {
                "arm": arm, "checkpoint_step": step, "checkpoint_sha256": meta["model_sha256"],
                "training_task_id": protocol["training_tasks"][arm]["id"],
                "training_task_fingerprint": protocol["training_tasks"][arm]["fingerprint"],
                "original_task_probe": probe["aggregate"], "original_task_probe_cases": probe["cases"],
                "original_task_full": full["aggregate"] if full else None,
                "original_task_full_cases": full["cases"] if full else None,
                "actor_fixed_anchor_sensitivity": {
                    "deterministic_action_std": fixed_metrics["deterministic_action_std_across_observations"],
                    "jacobian_by_group": fixed_metrics["action_input_jacobian_by_observation_group"],
                    "first_layer": fixed_metrics["first_layer_preactivation"],
                },
                "actor_real_probe_sensitivity": real_metrics,
            }
            (arm_result / f"checkpoint_{step:05d}_diagnostics.json").write_text(
                json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            print(f"[{arm}] checkpoint {step}: 6 frozen original-task probes recorded", flush=True)
            if step == 10000:
                model, loaded_meta = load_sac_checkpoint(
                    checkpoint, expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
                    task_context_producer_id=REFERENCE_TASK_PRODUCER,
                    expected_actor_activation_id=ACTOR_LEAKY_RELU_001_V1,
                )
                curriculum = run_curriculum_eval(checkpoint, protocol, arm_result / "step_10000_curriculum", model)
                row["curriculum_evaluation"] = {
                    "clean_successes": curriculum["clean_successes"], "safety_failures": curriculum["safety_failures"],
                    "timeouts": curriculum["timeouts"], "censored": curriculum["censored"],
                    "case_count": curriculum["case_count"], "result_sha256": curriculum["result_sha256"],
                }
                (arm_result / f"checkpoint_{step:05d}_diagnostics.json").write_text(
                    json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
                del model
    print(f"P2-F evaluation complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
