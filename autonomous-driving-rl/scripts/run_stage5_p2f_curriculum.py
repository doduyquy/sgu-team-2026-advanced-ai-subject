"""Run the preregistered TRAIN-only P2-F SAC start-distance curriculum A/B."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from collections import Counter
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_stage5_p2_experiment as p2
from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER
from src.training.stage5.actor_activation_ablation import ACTOR_LEAKY_RELU_001_V1, activation_inventory
from src.training.stage5.checkpoint import save_sac_checkpoint, verify_checkpoint_integrity
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.training.stage5.pilot_observation_transform import GroupwisePhysicalObservation
from src.training.stage5.start_curriculum import (
    CONTROL_FINGERPRINT, CONTROL_TASK_ID, CURRICULUM_FINGERPRINT, CURRICULUM_TASK_ID,
)
from src.training.stage5.train_sampler import SamplerStrategy, TrainCaseSampler

PROTOCOL = ROOT / "results/stage5/p2f_training_curriculum_20261121/protocol.json"
RUN_ID = "s5_p2f_start_distance_seed101_20261121"
TRAIN_STEPS = 10_000
CHECKPOINT_STEPS = (2_000, 5_000, 10_000)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol() -> dict[str, Any]:
    raw = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    expected = raw.pop("protocol_sha256", None)
    actual = p2.canonical_hash(raw)
    raw["protocol_sha256"] = expected
    if expected != actual or raw.get("status") != "PRE_REGISTERED":
        raise ValueError("P2-F protocol hash/status validation failed")
    for relative, digest in raw["source_files_sha256"].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"P2-F source changed after preregistration: {relative}")
    if raw["training"]["max_environment_steps_per_arm"] != TRAIN_STEPS:
        raise ValueError("P2-F step budget differs from frozen protocol")
    return raw


def make_env(protocol: dict[str, Any], arm: str, *, training: bool):
    pool_key = "training_pool" if training else "diagnostic_pool"
    pool = protocol["geometry"][pool_key]
    geometry_ids = [item["geometry_id"] for item in pool]
    sampler = TrainCaseSampler(
        training_root_seed=101, strategy=SamplerStrategy.ROUND_ROBIN,
        allowed_geometry_ids=geometry_ids, project_root=ROOT,
    )
    curriculum = arm == "B" and training
    task_id = CURRICULUM_TASK_ID if curriculum else CONTROL_TASK_ID
    base = Stage5TrainingEnv(
        sampler=sampler, training_root_seed=101, project_root=ROOT,
        simulator_render_mode="OFF", task_context_producer_id=REFERENCE_TASK_PRODUCER,
        training_start_curriculum=curriculum, training_task_id=task_id,
    )
    env = GroupwisePhysicalObservation(base)
    if not training:
        env._p2_diagnostic_schedule = {item["geometry_id"]: item for item in pool}
    return env, base, sampler


class P2FCurriculumSAC(p2.P2InstrumentedSAC):
    """Same P2 SAC collection path with training-task identity in action audit."""

    def _store_transition(self, replay_buffer, buffer_action, new_obs, rewards, dones, infos) -> None:
        sink = self.action_audit_sink
        if sink is not None and len(infos) == 1:
            info = infos[0]

            def enriched(record):
                record.update({key: info[key] for key in (
                    "training_task_id", "training_task_fingerprint", "curriculum_episode_index",
                    "curriculum_stage_id", "requested_remaining_route_distance_m",
                    "effective_remaining_route_distance_m", "curriculum_reference_station_m",
                    "original_destination", "route_completion_reset", "horizon_steps",
                ) if key in info})
                sink(record)

            self.action_audit_sink = enriched
        try:
            super()._store_transition(replay_buffer, buffer_action, new_obs, rewards, dones, infos)
        finally:
            self.action_audit_sink = sink


class P2FTrainingCallback(p2.P2TrainingCallback):
    def __init__(self, *args, task: dict[str, Any], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.task = task
        self._task_episode = {"steps": 0, "speeds": [], "progress": [], "steering": [], "throttle": [], "moving_run": 0, "max_moving_run": 0}
        self.task_episodes: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        row = self._task_episode
        row["steps"] += 1
        row["speeds"].append(float(info.get("speed_km_h", 0.0)))
        row["progress"].append(float(info.get("delta_route", 0.0)))
        row["steering"].append(float(info.get("action_steering", 0.0)))
        row["throttle"].append(float(info.get("action_throttle_brake", 0.0)))
        row["moving_run"] = row["moving_run"] + 1 if float(info.get("speed_km_h", 0.0)) >= 1.0 else 0
        row["max_moving_run"] = max(row["max_moving_run"], row["moving_run"])
        if bool(np.asarray(self.locals["dones"], dtype=bool)[0]):
            outcome = info.get("outcome", {})
            throttle = np.asarray(row["throttle"], dtype=np.float64)
            speed = np.asarray(row["speeds"], dtype=np.float64)
            record = {
                "arm": self.arm, "training_episode_index": int(info.get("curriculum_episode_index") or self.episode_index),
                "geometry_id": info.get("geometry_id"), "case_id": info.get("case_id"),
                "geometry_sha256": self.geometry_lookup.get(str(info.get("geometry_id")), {}).get("geometry_sha256"),
                "environment_seed": info.get("environment_seed"), "training_task_id": info.get("training_task_id"),
                "training_task_fingerprint": info.get("training_task_fingerprint"),
                "curriculum_stage_id": info.get("curriculum_stage_id"),
                "requested_remaining_route_distance_m": info.get("requested_remaining_route_distance_m"),
                "effective_remaining_route_distance_m": info.get("effective_remaining_route_distance_m"),
                "original_destination": info.get("original_destination"),
                "route_completion_reset": info.get("route_completion_reset"),
                "horizon_steps": info.get("horizon_steps"), "actual_steps": row["steps"],
                "outcome": outcome.get("primary_reason", info.get("primary_reason")),
                "clean_success": bool(outcome.get("clean_success", info.get("clean_success", False))),
                "terminated": bool(outcome.get("terminated", not info.get("TimeLimit.truncated", False))),
                "truncated": bool(outcome.get("truncated", info.get("TimeLimit.truncated", False))),
                "route_progress_delta": float(sum(row["progress"])),
                "mean_speed_kmh": float(speed.mean()) if len(speed) else None,
                "peak_speed_kmh": float(speed.max()) if len(speed) else None,
                "stationary_fraction_below_1kmh": float(np.mean(speed < 1.0)) if len(speed) else 1.0,
                "longest_moving_run_decisions": int(row["max_moving_run"]),
                "mean_steering": float(np.mean(row["steering"])) if row["steering"] else None,
                "mean_throttle_brake": float(throttle.mean()) if len(throttle) else None,
                "positive_throttle_fraction": float(np.mean(throttle > 0.0)) if len(throttle) else 0.0,
                "brake_fraction": float(np.mean(throttle < 0.0)) if len(throttle) else 0.0,
                "throttle_sign_switches": int(np.sum(throttle[1:] * throttle[:-1] < 0.0)) if len(throttle) > 1 else 0,
                "scope": "training experience; not frozen-policy evaluation",
            }
            self.task_episodes.append(record)
            with (self.arm_dir / "training_task_episodes.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            self._task_episode = {"steps": 0, "speeds": [], "progress": [], "steering": [], "throttle": [], "moving_run": 0, "max_moving_run": 0}
        return super()._on_step()

    def save_checkpoint(self, step: int) -> None:
        encoder = Stage5ObservationEncoder(core_only=False)
        destination = self.arm_dir / "checkpoints" / f"step_{step:06d}"
        activation = activation_inventory(self.model.policy)
        if activation["actor_id"] != ACTOR_LEAKY_RELU_001_V1 or not activation["critic_is_relu_only"]:
            raise RuntimeError(f"P2-F activation contract mismatch: {activation}")
        config = {
            "protocol": self.protocol["protocol"], "protocol_sha256": self.protocol["protocol_sha256"],
            "arm": self.arm, "training_task": self.task,
            "training_task_id": self.task["id"], "training_task_fingerprint": self.task["fingerprint"],
            "actor_activation_id": ACTOR_LEAKY_RELU_001_V1, "critic_activation": "ReLU",
            "learning_rate": self.protocol["sac"]["learning_rate"],
            "buffer_size": self.protocol["sac"]["buffer_size"], "batch_size": self.protocol["sac"]["batch_size"],
            "gamma": self.protocol["sac"]["gamma"], "tau": self.protocol["sac"]["tau"],
            "learning_starts": self.protocol["sac"]["learning_starts"],
            "gradient_steps": self.protocol["sac"]["gradient_steps"],
            "exploration_mode": "standard_sb3_sac_stochastic",
            "task_context_producer_id": REFERENCE_TASK_PRODUCER,
            "observation_transform": self.protocol["observation"]["post_encoder_transform"],
            "initial_policy_fingerprint": self.initial_fingerprint,
        }
        snapshot = {
            "training_step": step, "optimizer_updates": int(self.model._n_updates),
            "completed_episodes": self.episode_index,
            "partial_episode_at_checkpoint": {
                "episode_index": self.episode_index, "steps_so_far": self.episode_steps,
                "return_so_far": self.episode_return, "route_progress_delta_so_far": self.episode_progress,
                "case": self.episode_geom or None,
            },
            "training_task_episode_count": len(self.task_episodes),
            "alpha": float(self.model.log_ent_coef.detach().exp().cpu()) if getattr(self.model, "log_ent_coef", None) is not None else float(self.model.ent_coef),
        }
        saved = save_sac_checkpoint(
            self.model, destination, encoder, total_timesteps=step, training_root_seed=101,
            training_config=config, training_summary=snapshot, save_replay_buffer=(step == TRAIN_STEPS),
            project_root=ROOT, observation_transform=self.protocol["observation"]["post_encoder_transform"],
            task_context_producer_id=REFERENCE_TASK_PRODUCER,
        )
        metadata = verify_checkpoint_integrity(
            saved, expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
            task_context_producer_id=REFERENCE_TASK_PRODUCER,
        )
        if metadata["training_config"].get("training_task_fingerprint") != self.task["fingerprint"]:
            raise RuntimeError("P2-F checkpoint did not bind the training task identity")
        replay_size = int(self.model.replay_buffer.size())
        indices = np.linspace(0, replay_size - 1, min(1024, replay_size), dtype=np.int64) if replay_size else np.asarray([], dtype=np.int64)
        replay_sample_sha = None
        if replay_size:
            sample_path = self.arm_dir / "replay_samples" / f"observations_step_{step:05d}.npz"
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                sample_path,
                observations=np.asarray(self.model.replay_buffer.observations[indices, 0], dtype=np.float32),
                actions=np.asarray(self.model.replay_buffer.actions[indices, 0], dtype=np.float32),
                task_id=np.asarray([self.action_records[i].get("training_task_id", "") for i in indices]),
                stage_id=np.asarray([self.action_records[i].get("curriculum_stage_id") or "full_route" for i in indices]),
                speed_km_h=np.asarray([self.action_records[i]["speed_km_h"] for i in indices], dtype=np.float32),
                delta_route=np.asarray([self.action_records[i]["delta_route"] for i in indices], dtype=np.float32),
            )
            replay_sample_sha = sha256(sample_path)
        td = p2.td_error_snapshot(self.model)
        row = {
            "step": step, "checkpoint_sha256": metadata["model_sha256"],
            "metadata_sha256": sha256(saved / "metadata.json"),
            "replay_buffer_sha256": metadata.get("replay_buffer_sha256"),
            "replay_sample_sha256": replay_sample_sha, "replay_size": replay_size,
            "optimizer_updates": int(self.model._n_updates), "completed_episodes": self.episode_index,
            "alpha": snapshot["alpha"], "actor_activation": activation["actor_id"],
            "fixed_recent_replay_td_diagnostic": td,
            "task_id": self.task["id"], "task_fingerprint": self.task["fingerprint"],
            "checkpoint_path": str(saved.relative_to(ROOT)).replace("\\", "/"),
        }
        self.checkpoints.append(row)
        (self.arm_dir / "checkpoint_telemetry.json").write_text(json.dumps(self.checkpoints, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"[{self.arm}] checkpoint={step} sha256={metadata['model_sha256']} episodes={self.episode_index} alpha={snapshot['alpha']:.6g}", flush=True)


def create_shared_initial(protocol: dict[str, Any], run_root: Path) -> dict[str, Any]:
    initial_dir = run_root / "shared_initial"
    initial_dir.mkdir(parents=True, exist_ok=False)
    env, _, _ = make_env(protocol, "A", training=True)
    try:
        p2.seed_everything(101)
        kwargs = p2.model_kwargs(protocol, "standard")
        kwargs["policy"] = __import__(
            "src.training.stage5.actor_activation_ablation", fromlist=["ActorOnlyLeakyReLUPolicy"]
        ).ActorOnlyLeakyReLUPolicy
        model = P2FCurriculumSAC(env=env, **kwargs)
        state = {
            "policy": {name: value.detach().cpu().clone() for name, value in model.policy.state_dict().items()},
            "log_ent_coef": model.log_ent_coef.detach().cpu().clone() if getattr(model, "log_ent_coef", None) is not None else None,
        }
        state_path = initial_dir / "initial_state.pt"
        torch.save(state, state_path)
        fingerprint = p2.state_fingerprint(model)
        manifest = {
            "sha256": sha256(state_path), "policy_state_fingerprint": fingerprint,
            "actor_activation": ACTOR_LEAKY_RELU_001_V1, "critic_activation": "ReLU",
            "entropy_coefficient": float(model.log_ent_coef.detach().exp().cpu()),
            "optimizer_state": "fresh and independent per arm",
            "same_initialization_claim": "same state-dict tensors and log entropy coefficient; activation and reset task same actor architecture across arms",
        }
        (initial_dir / "initialization.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"state": state, "manifest": manifest}
    finally:
        env.close()


def run_arm(protocol: dict[str, Any], arm: str, initial: dict[str, Any], run_root: Path) -> dict[str, Any]:
    arm_dir = run_root / f"arm_{arm}"
    arm_dir.mkdir(parents=True, exist_ok=False)
    task = protocol["training_tasks"][arm]
    env, base, _ = make_env(protocol, arm, training=True)
    callback = None
    started = time.perf_counter()
    try:
        p2.seed_everything(101)
        kwargs = p2.model_kwargs(protocol, "standard")
        kwargs["policy"] = __import__(
            "src.training.stage5.actor_activation_ablation", fromlist=["ActorOnlyLeakyReLUPolicy"]
        ).ActorOnlyLeakyReLUPolicy
        model = P2FCurriculumSAC(env=env, **kwargs)
        p2.load_initial_into(model, initial["state"])
        model.install_gradient_hooks()
        fingerprint = p2.state_fingerprint(model)
        if fingerprint != initial["manifest"]["policy_state_fingerprint"]:
            raise RuntimeError("P2-F arm did not receive shared fresh initialization")
        callback = P2FTrainingCallback(arm_dir, arm, protocol, fingerprint, task=task)
        callback.model = model
        p2._AFTER_OPTIMIZER[model] = callback
        callback.save_checkpoint(0)
        model.learn(total_timesteps=TRAIN_STEPS, callback=callback, reset_num_timesteps=True, log_interval=10)
        if int(model.num_timesteps) != TRAIN_STEPS or [x["step"] for x in callback.checkpoints] != [0, *CHECKPOINT_STEPS]:
            raise RuntimeError("P2-F arm did not complete the registered checkpoint schedule")
        if model.action_mismatch_count != 0:
            raise RuntimeError(f"P2-F executed/replay action mismatch count={model.action_mismatch_count}")
        summary = callback.summary()
        summary.update({
            "status": "COMPLETE", "actual_environment_steps": int(model.num_timesteps),
            "optimizer_updates": int(model._n_updates), "wall_time_seconds": time.perf_counter() - started,
            "training_task_id": task["id"], "training_task_fingerprint": task["fingerprint"],
            "training_task_episodes": callback.task_episodes,
            "partial_training_task_episode_at_end": callback._task_episode,
            "initial_policy_fingerprint": fingerprint, "protocol_sha256": protocol["protocol_sha256"],
            "runtime_geometry_count": len(callback.geometry_steps),
        })
        path = arm_dir / "training_summary.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(f"[{arm}] COMPLETE steps={model.num_timesteps} updates={model._n_updates} episodes={summary['completed_episodes']} wall={summary['wall_time_seconds']:.1f}s", flush=True)
        return summary
    finally:
        if callback is not None:
            callback._on_training_end()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--arm", choices=("A", "B", "ALL"), default="ALL")
    args = parser.parse_args()
    if args.arm != "ALL":
        raise ValueError("P2-F is registered as one paired run; individual arms may not be launched")
    protocol = load_protocol()
    run_root = ROOT / "runs/stage5_p2f" / args.run_id
    if run_root.exists():
        raise FileExistsError(f"Refusing to overwrite an existing P2-F run: {run_root}")
    free = shutil.disk_usage(run_root.parent if run_root.parent.exists() else ROOT).free
    if free < 2_000_000_000:
        raise RuntimeError(f"P2-F requires at least 2 GB free for paired checkpoints/replay; available={free}")
    run_root.mkdir(parents=True)
    initial = create_shared_initial(protocol, run_root)
    run_manifest = {
        "experiment": protocol["protocol"], "protocol_sha256": protocol["protocol_sha256"],
        "protocol_path": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
        "source_commit": p2.git_head(), "training_root_seed": 101,
        "initialization": initial["manifest"], "arms": ["A", "B"],
        "task_ids": {arm: protocol["training_tasks"][arm] for arm in ("A", "B")},
        "train_geometry_ids": [x["geometry_id"] for x in protocol["geometry"]["training_pool"]],
        "diagnostic_geometry_ids": [x["geometry_id"] for x in protocol["geometry"]["diagnostic_pool"]],
        "benchmark_eligible": False, "validation_test_accessed": False,
    }
    (run_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summaries = {arm: run_arm(protocol, arm, initial, run_root) for arm in ("A", "B")}
    summary_path = run_root / "paired_training_summary.json"
    summary_path.write_text(json.dumps({"protocol_sha256": protocol["protocol_sha256"], "arms": summaries}, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (run_root / "PAIRED_TRAINING_SHA256").write_text(f"{sha256(summary_path)}  paired_training_summary.json\n", encoding="utf-8")
    print(f"P2-F paired training summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
