"""Run the pre-registered P2-D actor-only activation ablation (TRAIN only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.sac.policies import SACPolicy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_stage5_p2_experiment as p2
from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER
from src.training.stage5.actor_activation_ablation import (
    ACTOR_LEAKY_RELU_001_V1, ACTOR_RELU_V1, activation_inventory,
)
from src.training.stage5.checkpoint import save_sac_checkpoint, verify_checkpoint_integrity
from src.training.stage5.observation_encoder import Stage5ObservationEncoder

PROTOCOL_PATH = ROOT / "results/stage5/p2d_actor_activation_seed101_20261121/protocol_v2.json"
RUN_ID = "s5_p2d_actor_activation_seed101_20261121"
TRAIN_STEPS = 10_000
CHECKPOINTS = (2_000, 5_000, 10_000)


def read_protocol() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    recorded = protocol.pop("protocol_sha256", None)
    if recorded != p2.canonical_hash(protocol):
        raise ValueError("P2-D protocol SHA-256 mismatch")
    protocol["protocol_sha256"] = recorded
    if protocol.get("status") != "PRE_REGISTERED" or protocol["training"]["max_environment_steps_per_arm"] != TRAIN_STEPS:
        raise ValueError("P2-D protocol is not registered for the requested 10k budget")
    for relative, expected in protocol["source_files_sha256"].items():
        if p2.sha256(ROOT / relative) != expected:
            raise ValueError(f"P2-D frozen source changed: {relative}")
    if p2.sha256(ROOT / "results/stage5/p2c_representation_critic_audit_20261121/fixed_train_anchors.npy") != protocol["anchors"]["sha256"]:
        raise ValueError("P2-C fixed anchor identity mismatch")
    base = json.loads((ROOT / "results/stage5/p2_correlated_exploration_seed101_20261120/protocol.json").read_text(encoding="utf-8"))
    if protocol["geometry"] != base["geometry"]:
        # The protocol records this exact P2 TRAIN schedule; exclude any silent split changes.
        raise ValueError("P2-D geometry schedule differs from the P2 registered schedule")
    return protocol


class P2DTrainingCallback(p2.P2TrainingCallback):
    def save_checkpoint(self, step: int) -> None:
        encoder = Stage5ObservationEncoder(core_only=False)
        destination = self.arm_dir / "checkpoints" / f"step_{step:06d}"
        activation = activation_inventory(self.model.policy)
        expected = ACTOR_RELU_V1 if self.arm == "A" else ACTOR_LEAKY_RELU_001_V1
        if activation["actor_id"] != expected or not activation["critic_is_relu_only"]:
            raise RuntimeError(f"Activation contract mismatch before checkpoint: {activation}")
        config = {
            "protocol": self.protocol["protocol"], "protocol_sha256": self.protocol["protocol_sha256"],
            "arm": self.arm, "actor_activation_id": expected, "actor_activation": self.protocol["arms"][self.arm]["actor_activation"],
            "critic_activation": "ReLU", "actor_negative_slope": self.protocol["sac"]["actor_activation_negative_slope"][self.arm],
            "learning_rate": self.protocol["sac"]["learning_rate"], "buffer_size": self.protocol["sac"]["buffer_size"],
            "batch_size": self.protocol["sac"]["batch_size"], "gamma": self.protocol["sac"]["gamma"],
            "tau": self.protocol["sac"]["tau"], "learning_starts": self.protocol["sac"]["learning_starts"],
            "gradient_steps": self.protocol["sac"]["gradient_steps"], "task_context_producer_id": REFERENCE_TASK_PRODUCER,
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
            "alpha": float(self.model.log_ent_coef.detach().exp().cpu()) if getattr(self.model, "log_ent_coef", None) is not None else float(self.model.ent_coef),
        }
        saved = save_sac_checkpoint(
            self.model, destination, encoder, total_timesteps=step, training_root_seed=101,
            training_config=config, training_summary=snapshot, save_replay_buffer=(step == TRAIN_STEPS),
            project_root=ROOT, observation_transform=self.protocol["observation"]["post_encoder_transform"],
            task_context_producer_id=REFERENCE_TASK_PRODUCER,
        )
        metadata = verify_checkpoint_integrity(saved, expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
                                               task_context_producer_id=REFERENCE_TASK_PRODUCER)
        if metadata.get("actor_activation") != expected:
            raise RuntimeError("Saved checkpoint metadata does not bind the registered actor activation")
        replay_size = int(self.model.replay_buffer.size())
        replay_sample_sha = None
        replay_sample_count = 0
        if replay_size:
            if len(self.action_records) != replay_size:
                raise RuntimeError(f"Replay/action telemetry length mismatch at {step}: {replay_size} vs {len(self.action_records)}")
            indices = np.linspace(0, replay_size - 1, min(1024, replay_size), dtype=np.int64)
            sample_dir = self.arm_dir / "replay_samples"
            sample_dir.mkdir(exist_ok=True)
            sample_path = sample_dir / f"observations_step_{step:05d}.npz"
            np.savez_compressed(
                sample_path,
                observations=np.asarray(self.model.replay_buffer.observations[indices, 0], dtype=np.float32),
                actions=np.asarray(self.model.replay_buffer.actions[indices, 0], dtype=np.float32),
                speed_km_h=np.asarray([self.action_records[i]["speed_km_h"] for i in indices], dtype=np.float32),
                delta_route=np.asarray([self.action_records[i]["delta_route"] for i in indices], dtype=np.float32),
            )
            replay_sample_sha = p2.sha256(sample_path)
            replay_sample_count = len(indices)
        td = p2.td_error_snapshot(self.model)
        row = {
            "step": step, "model_sha256": metadata["model_sha256"],
            "activation_fingerprint": metadata["activation_fingerprint"],
            "activation_inventory": metadata["activation_inventory"],
            "metadata_sha256": p2.sha256(saved / "metadata.json"), "alpha": snapshot["alpha"],
            "optimizer_updates": int(self.model._n_updates), "completed_episodes": self.episode_index,
            "replay_size": int(self.model.replay_buffer.size()), "fixed_recent_replay_td_diagnostic": td,
            "replay_buffer_sha256": metadata.get("replay_buffer_sha256"),
            "replay_observation_sample_sha256": replay_sample_sha,
            "replay_observation_sample_count": replay_sample_count,
            "checkpoint_path": str(saved.relative_to(ROOT)).replace("\\", "/"),
        }
        self.checkpoints.append(row)
        (self.arm_dir / "checkpoint_telemetry.json").write_text(json.dumps(self.checkpoints, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"[{self.arm}] checkpoint={step} sha256={metadata['model_sha256']} activation={expected} alpha={snapshot['alpha']:.6g}", flush=True)


def run_arm(protocol: dict, arm: str, initial: dict, run_root: Path) -> dict:
    arm_dir = run_root / f"arm_{arm}"
    if arm_dir.exists():
        raise FileExistsError(f"Refusing to overwrite P2-D arm artifacts: {arm_dir}")
    env, _ = p2.make_env(protocol, arm, training=True)
    callback = None
    try:
        p2.seed_everything(101)
        kwargs = p2.model_kwargs(protocol, "standard")
        kwargs["policy"] = SACPolicy if arm == "A" else __import__(
            "src.training.stage5.actor_activation_ablation", fromlist=["ActorOnlyLeakyReLUPolicy"]
        ).ActorOnlyLeakyReLUPolicy
        model = p2.P2InstrumentedSAC(env=env, **kwargs)
        p2.load_initial_into(model, initial["state"])
        model.install_gradient_hooks()
        fingerprint = p2.state_fingerprint(model)
        if fingerprint != initial["manifest"]["policy_state_fingerprint"]:
            raise RuntimeError(f"Arm {arm} did not receive the shared initial tensors and entropy state")
        callback = P2DTrainingCallback(arm_dir, arm, protocol, fingerprint)
        callback.model = model
        p2._AFTER_OPTIMIZER[model] = callback
        callback.save_checkpoint(0)
        model.learn(total_timesteps=TRAIN_STEPS, callback=callback, reset_num_timesteps=True, log_interval=10)
        if int(model.num_timesteps) != TRAIN_STEPS or [x["step"] for x in callback.checkpoints] != [0, *CHECKPOINTS]:
            raise RuntimeError(f"Arm {arm} did not reach the complete registered checkpoint schedule")
        if model.action_mismatch_count != 0:
            raise RuntimeError(f"Arm {arm} had {model.action_mismatch_count} sampled/executed/replay action mismatches")
        summary = callback.summary()
        summary["actor_activation_id"] = callback.checkpoints[0]["activation_inventory"]["actor_id"]
        summary["critic_activation"] = "ReLU"
        summary["initial_policy_fingerprint"] = fingerprint
        summary["protocol_sha256"] = protocol["protocol_sha256"]
        summary["run_manifest_sha256"] = p2.sha256(PROTOCOL_PATH)
        (arm_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"[{arm}] {summary['status']} steps={summary['actual_environment_steps']} updates={summary['optimizer_updates']} episodes={summary['completed_episodes']}", flush=True)
        return summary
    finally:
        if callback is not None:
            callback._on_training_end()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("A", "B", "ALL"), default="ALL")
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    protocol = read_protocol()
    run_root = ROOT / "runs/stage5_p2d" / args.run_id
    if args.arm == "ALL":
        if run_root.exists():
            raise FileExistsError(f"Refusing to overwrite P2-D run: {run_root}")
        run_root.mkdir(parents=True)
        initial = p2.create_shared_initial(protocol, run_root)
        (run_root / "run_manifest.json").write_text(json.dumps({
            "protocol": protocol["protocol"], "protocol_sha256": protocol["protocol_sha256"],
            "source_commit": p2.git_head(), "initialization": initial["manifest"],
            "arms": ["A", "B"], "benchmark_eligible": False,
            "official_validation_test_accessed": False,
        }, indent=2) + "\n", encoding="utf-8")
        for arm in ("A", "B"):
            run_arm(protocol, arm, initial, run_root)
    else:
        raise RuntimeError("P2-D is registered as a paired run; single-arm continuation is forbidden")


if __name__ == "__main__":
    main()
