"""Fresh, continuous three-arm SAC training for the bounded S5.5 study."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
import weakref
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.stage5.checkpoint import save_sac_checkpoint, verify_checkpoint_integrity
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.training.stage5.pilot_observation_transform import (
    GROUPWISE_PHYSICAL_V1,
    GroupwisePhysicalObservation,
)
from src.training.stage5.reward_experiments import RewardScale, Stage5ExperimentalRewardWrapper
from src.training.stage5.train_sampler import SamplerStrategy, TrainCaseSampler


RUN_ID = "s5_5_meter_reward_seed101_20261120"
RUNS = ROOT / "runs" / "stage5_meter_reward" / RUN_ID
RESULTS = ROOT / "results" / "stage5" / RUN_ID
CHECKPOINT_STEPS = [2000, 5000, 10000, 30000]
TRAIN_STEPS = 30000
TRAIN_ROOT_SEED = 101
TRANSFORM_ID = GROUPWISE_PHYSICAL_V1["transform_id"]
_POST_TRAIN_CALLBACKS: "weakref.WeakKeyDictionary[SAC, Any]" = weakref.WeakKeyDictionary()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol") != "stage5_s5_5_meter_reward_v1":
        raise ValueError("Unexpected protocol identity")
    expected = protocol.get("protocol_sha256")
    unsigned = dict(protocol)
    unsigned.pop("protocol_sha256", None)
    if canonical_hash(unsigned) != expected:
        raise ValueError("Protocol SHA-256 mismatch")
    if protocol["status"] != "PRE_REGISTERED":
        raise ValueError("S5.5 protocol must be pre-registered before training")
    return protocol


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_fingerprint(model: SAC) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    log_alpha = getattr(model, "log_ent_coef", None)
    if log_alpha is not None:
        digest.update(b"log_ent_coef")
        digest.update(log_alpha.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def alpha_value(model: SAC) -> float | None:
    value = getattr(model, "log_ent_coef", None)
    if value is None:
        return float(model.ent_coef) if isinstance(model.ent_coef, (int, float)) else None
    return float(value.detach().exp().cpu().item())


def optimizer_state_summary(model: SAC) -> dict[str, Any]:
    actors = {
        "actor": model.actor.optimizer,
        "critic": model.critic.optimizer,
        "entropy": getattr(model, "ent_coef_optimizer", None),
    }
    return {
        key: None if opt is None else {
            "state_entries": len(opt.state),
            "param_groups": [{
                "lr": float(group["lr"]),
                "betas": list(group.get("betas", ())),
                "eps": float(group.get("eps", 0.0)),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "param_count": len(group["params"]),
            } for group in opt.param_groups],
        }
        for key, opt in actors.items()
    }


def build_env(protocol: dict[str, Any], reward_variant: str, training: bool = True):
    pool = protocol["training_geometry_pool"] if training else protocol["diagnostic_pool"]
    ids = [row["geometry_id"] for row in pool]
    sampler = TrainCaseSampler(
        training_root_seed=TRAIN_ROOT_SEED,
        strategy=SamplerStrategy.ROUND_ROBIN if training else SamplerStrategy.FIXED,
        allowed_geometry_ids=ids,
        project_root=ROOT,
    )
    base = Stage5TrainingEnv(
        sampler=sampler,
        training_root_seed=TRAIN_ROOT_SEED,
        project_root=ROOT,
        simulator_render_mode="OFF",
    )
    if training:
        median = float(protocol["reward_variants"]["median_training_manifest_route_length_m"])
        k = float(protocol["reward_variants"]["k"])
        scale = RewardScale(median, k)
        env = Stage5ExperimentalRewardWrapper(base, reward_variant, scale)
    else:
        env = base
    env = GroupwisePhysicalObservation(env)
    return env, sampler


class S55SAC(SAC):
    """SAC hook that saves a scheduled checkpoint after its matching gradient update."""

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        super().train(gradient_steps=gradient_steps, batch_size=batch_size)
        callback = _POST_TRAIN_CALLBACKS.get(self)
        if callback is not None:
            callback.after_optimizer_update()


def model_kwargs(protocol: dict[str, Any]) -> dict[str, Any]:
    hp = protocol["sac"]
    return {
        "policy": "MlpPolicy",
        "learning_rate": float(hp["learning_rate"]),
        "buffer_size": int(hp["buffer_size"]),
        "learning_starts": int(hp["learning_starts"]),
        "batch_size": int(hp["batch_size"]),
        "tau": float(hp["tau"]),
        "gamma": float(hp["gamma"]),
        "train_freq": (int(hp["train_freq"]), "step"),
        "gradient_steps": int(hp["gradient_steps"]),
        "ent_coef": hp["ent_coef"],
        "policy_kwargs": {"net_arch": list(hp["net_arch"]), "activation_fn": torch.nn.ReLU},
        "seed": int(protocol["training_root_seed"]),
        "device": "cpu",
        "verbose": 0,
    }


def runtime_versions() -> dict[str, Any]:
    import gymnasium
    import metadrive
    import stable_baselines3
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "metadrive": getattr(metadrive, "__version__", "0.4.3"),
        "device": "cpu",
    }


def create_initial_model(protocol: dict[str, Any], target: Path, *, steps: int = 0) -> tuple[Path, str]:
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite fresh initialization: {target}")
    env, _ = build_env(protocol, "V1", training=True)
    try:
        seed_everything(TRAIN_ROOT_SEED)
        model = S55SAC(env=env, **model_kwargs(protocol))
        target.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(target.with_suffix("")))
        saved = target.with_suffix(".zip")
        return saved, state_fingerprint(model)
    finally:
        env.close()


class S55TrainingCallback(BaseCallback):
    def __init__(self, arm_dir: Path, arm: str, reward_variant: str,
                 protocol: dict[str, Any], source_commit: str,
                 initial_weights_fingerprint: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.arm_dir = arm_dir
        self.arm = arm
        self.reward_variant = reward_variant
        self.protocol = protocol
        self.source_commit = source_commit
        self.initial_weights_fingerprint = initial_weights_fingerprint
        self.next_checkpoints = iter(CHECKPOINT_STEPS)
        self.next_checkpoint = next(self.next_checkpoints)
        self.episode_records: list[dict[str, Any]] = []
        self.geometry_steps: Counter[str] = Counter()
        self.geometry_episodes: Counter[str] = Counter()
        self.actions: list[tuple[float, float]] = []
        self.speeds: list[float] = []
        self.rewards: list[float] = []
        self.current_episode_return = 0.0
        self.current_episode_steps = 0
        self.variant_rewards: dict[str, list[float]] = defaultdict(list)
        self.sign_switches = 0
        self.prev_throttle_sign: int | None = None
        self.current_positive_run = 0
        self.max_positive_run = 0
        self.current_negative_run = 0
        self.max_negative_run = 0
        self.current_moving_run = 0
        self.max_moving_run = 0
        self.started = time.perf_counter()
        self.ram_samples: list[dict[str, float]] = []
        self.checkpoint_summaries: list[dict[str, Any]] = []
        self.telemetry_window_start = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for idx, info in enumerate(infos):
            gid = str(info.get("geometry_id", "UNKNOWN"))
            self.geometry_steps[gid] += 1
            steering = float(info.get("action_steering", 0.0))
            throttle = float(info.get("action_throttle_brake", 0.0))
            speed = float(info.get("speed_km_h", 0.0))
            reward = float(self.locals.get("rewards", [0.0])[idx])
            self.current_episode_return += reward
            self.current_episode_steps += 1
            self.actions.append((steering, throttle))
            self.speeds.append(speed)
            self.rewards.append(reward)
            if isinstance(info.get("s5_5_rewards"), dict):
                for name in ("reward_v1", "reward_scale", "reward_meter"):
                    self.variant_rewards[name].append(float(info["s5_5_rewards"][name]))
            sign = 1 if throttle > 1e-6 else (-1 if throttle < -1e-6 else 0)
            if self.prev_throttle_sign is not None and sign != self.prev_throttle_sign and sign != 0 and self.prev_throttle_sign != 0:
                self.sign_switches += 1
            if sign != 0:
                self.prev_throttle_sign = sign
            if sign > 0:
                self.current_positive_run += 1
                self.current_negative_run = 0
            elif sign < 0:
                self.current_negative_run += 1
                self.current_positive_run = 0
            else:
                self.current_positive_run = 0
                self.current_negative_run = 0
            self.max_positive_run = max(self.max_positive_run, self.current_positive_run)
            self.max_negative_run = max(self.max_negative_run, self.current_negative_run)
            if speed >= 1.0:
                self.current_moving_run += 1
            else:
                self.current_moving_run = 0
            self.max_moving_run = max(self.max_moving_run, self.current_moving_run)

            if bool(dones[idx]):
                outcome = info.get("outcome", {})
                self.episode_records.append({
                    "arm": self.arm,
                    "training_step": int(self.num_timesteps),
                    "case_id": info.get("case_id"),
                    "geometry_id": gid,
                    "geometry_generation_seed": info.get("geometry_generation_seed"),
                    "environment_seed": info.get("environment_seed"),
                    "steps": info.get("total_steps", info.get("step_index")),
                    "training_reward_return": self.current_episode_return,
                    "reward_v1_return": info.get("cumulative_reward"),
                    "reward_variant": self.reward_variant,
                    "outcome": outcome.get("primary_reason", info.get("primary_reason")),
                    "clean_success": bool(outcome.get("clean_success", False)),
                })
                self.geometry_episodes[gid] += 1
                self.current_episode_return = 0.0
                self.current_episode_steps = 0
                self.prev_throttle_sign = None
                self.current_positive_run = 0
                self.current_negative_run = 0
                self.current_moving_run = 0
        if self.num_timesteps % 1000 == 0:
            self._sample_ram()
        return True

    def after_optimizer_update(self) -> None:
        if self.next_checkpoint is None or int(self.model.num_timesteps) < self.next_checkpoint:
            return
        if int(self.model.num_timesteps) != self.next_checkpoint:
            raise RuntimeError(
                f"Post-update checkpoint skipped its registered step: got {self.model.num_timesteps}, "
                f"expected {self.next_checkpoint}"
            )
        checkpoint_step = int(self.next_checkpoint)
        self._save_checkpoint(checkpoint_step)
        self.next_checkpoint = next(self.next_checkpoints, None)

    def _sample_ram(self) -> None:
        try:
            import psutil
            ram_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except Exception:
            ram_mb = None
        self.ram_samples.append({"step": int(self.num_timesteps), "rss_mb": ram_mb})

    def _summary(self, step: int) -> dict[str, Any]:
        start = self.telemetry_window_start
        action_window = self.actions[start:]
        speed_window = self.speeds[start:]
        reward_window = self.rewards[start:]
        throttle = [x[1] for x in action_window]
        steering = [x[0] for x in action_window]
        updates = int(getattr(self.model, "_n_updates", 0))
        try:
            logged = {str(k): float(v) for k, v in self.model.logger.name_to_value.items()
                      if isinstance(v, (int, float, np.number)) and math.isfinite(float(v))}
        except Exception:
            logged = {}
        return {
            "arm": self.arm,
            "reward_variant": self.reward_variant,
            "training_steps": int(step),
            "optimizer_updates": updates,
            "completed_episodes": len(self.episode_records),
            "unique_geometries_seen": sorted(g for g, count in self.geometry_steps.items() if count),
            "geometry_steps": dict(self.geometry_steps),
            "geometry_completed_episodes": dict(self.geometry_episodes),
            "window_start_step": start,
            "window_steps": len(action_window),
            "window_mean_speed_km_h": statistics.fmean(speed_window) if speed_window else None,
            "window_stationary_fraction_speed_lt_1_km_h": (
                sum(x < 1.0 for x in speed_window) / len(speed_window) if speed_window else None
            ),
            "window_throttle_mean": statistics.fmean(throttle) if throttle else None,
            "window_throttle_std": statistics.pstdev(throttle) if len(throttle) > 1 else 0.0,
            "window_throttle_positive_fraction": sum(x > 0 for x in throttle) / len(throttle) if throttle else None,
            "window_throttle_negative_fraction": sum(x < 0 for x in throttle) / len(throttle) if throttle else None,
            "window_steering_std": statistics.pstdev(steering) if len(steering) > 1 else 0.0,
            "max_positive_throttle_run_steps": self.max_positive_run,
            "max_negative_throttle_run_steps": self.max_negative_run,
            "max_sustained_moving_run_steps_speed_ge_1": self.max_moving_run,
            "direct_throttle_brake_sign_switches": self.sign_switches,
            "alpha": alpha_value(self.model),
            "sb3_logger_values": logged,
            "ram_last_sample_mb": self.ram_samples[-1]["rss_mb"] if self.ram_samples else None,
            "elapsed_wall_time_s": time.perf_counter() - self.started,
            "replay_buffer_size": int(getattr(self.model.replay_buffer, "size", lambda: 0)()),
            "initial_weights_fingerprint": self.initial_weights_fingerprint,
        }

    def _save_checkpoint(self, step: int) -> None:
        checkpoint_dir = self.arm_dir / "checkpoints" / f"step_{step:06d}"
        if checkpoint_dir.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint_dir}")
        summary = self._summary(step)
        summary.update({
            "source_commit": self.source_commit,
            "protocol_sha256": self.protocol["protocol_sha256"],
            "policy_input_transform": TRANSFORM_ID,
        })
        save_sac_checkpoint(
            model=self.model,
            checkpoint_dir=checkpoint_dir,
            encoder=Stage5ObservationEncoder(),
            total_timesteps=int(step),
            training_root_seed=TRAIN_ROOT_SEED,
            training_config={
                "protocol": "stage5_s5_5_meter_reward_v1",
                "protocol_sha256": self.protocol["protocol_sha256"],
                "arm": self.arm,
                "reward_variant": self.reward_variant,
                "observation_transform": TRANSFORM_ID,
                "only_arm_difference": "training reward signal",
                "sac": self.protocol["sac"],
            },
            training_summary=summary,
            save_replay_buffer=False,
            project_root=ROOT,
            observation_transform=TRANSFORM_ID,
        )
        integrity = verify_checkpoint_integrity(checkpoint_dir)
        record = {
            "training_step": int(step),
            "checkpoint_path": str(checkpoint_dir.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_model_sha256": integrity["model_sha256"],
            "checkpoint_integrity": "PASS",
            "summary": summary,
        }
        self.checkpoint_summaries.append(record)
        self.telemetry_window_start = len(self.actions)
        print(json.dumps({"event": "checkpoint", **record}, ensure_ascii=False), flush=True)

    def _on_training_end(self) -> None:
        # The mandatory 30k callback should have saved the last checkpoint. Preserve a compact run summary.
        summary_path = self.arm_dir / "training_summary.json"
        output = {
            "status": "COMPLETE" if self.num_timesteps == TRAIN_STEPS else "INCOMPLETE",
            "arm": self.arm,
            "reward_variant": self.reward_variant,
            "actual_training_steps": int(self.num_timesteps),
            "optimizer_updates": int(getattr(self.model, "_n_updates", 0)),
            "episodes": self.episode_records,
            "geometry_steps": dict(self.geometry_steps),
            "geometry_completed_episodes": dict(self.geometry_episodes),
            "checkpoint_summaries": self.checkpoint_summaries,
            "telemetry_final": self._summary(int(self.num_timesteps)),
            "ram_samples": self.ram_samples,
            "runtime_versions": runtime_versions(),
            "protocol_sha256": self.protocol["protocol_sha256"],
            "source_commit": self.source_commit,
        }
        summary_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_completed_episodes(self.arm_dir, self.episode_records)


def write_completed_episodes(arm_dir: Path, episodes: list[dict[str, Any]]) -> None:
    """Keep completed episode records independent of the compact run summary."""
    payload = json.dumps(episodes, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with (arm_dir / "completed_episodes.json").open("x", encoding="utf-8") as stream:
        stream.write(payload)


def run_arm(protocol: dict[str, Any], arm: str, initial_model_path: Path,
            initial_weight_fingerprint: str, initial_zip_sha256: str,
            run_root: Path, smoke: bool = False, smoke_steps: int = 1200) -> None:
    variants = {"A": "V1", "B": "SCALE", "C": "METER"}
    variant = variants[arm]
    tag = "technical_smoke" if smoke else "arms"
    arm_dir = run_root / tag / (arm if not smoke else f"{arm}_smoke")
    if arm_dir.exists() and any(arm_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing S5.5 run artifacts: {arm_dir}")
    arm_dir.mkdir(parents=True, exist_ok=True)
    env, sampler = build_env(protocol, variant, training=True)
    try:
        seed_everything(TRAIN_ROOT_SEED)
        model = S55SAC.load(str(initial_model_path), env=env, device="cpu")
        model.set_random_seed(TRAIN_ROOT_SEED)
        initial_fingerprint = state_fingerprint(model)
        if initial_fingerprint != initial_weight_fingerprint:
            raise RuntimeError(f"{arm} did not restore the shared fresh initialization exactly")
        # Ensure load restored the same SAC entropy and optimizer initialization for each arm.
        initial_state = {
            "weights_fingerprint": initial_fingerprint,
            "initial_model_zip_sha256": initial_zip_sha256,
            "alpha": alpha_value(model),
            "optimizer_state": optimizer_state_summary(model),
            "sampler_strategy": sampler.strategy.value,
            "sampler_records_order": [row["geometry_id"] for row in sampler._records],
            "allowed_geometry_ids": sorted(sampler.allowed_geometry_ids or []),
        }
        (arm_dir / "initial_state.json").write_text(json.dumps(initial_state, indent=2) + "\n", encoding="utf-8")
        callback = S55TrainingCallback(
            arm_dir=arm_dir,
            arm=arm,
            reward_variant=variant,
            protocol=protocol,
            source_commit=protocol["source_commit"],
            initial_weights_fingerprint=initial_weight_fingerprint,
        )
        _POST_TRAIN_CALLBACKS[model] = callback
        # The zero-step snapshot uses the same fresh model state as all three arms.
        zero_dir = arm_dir / "checkpoints" / "step_000000"
        save_sac_checkpoint(
            model=model,
            checkpoint_dir=zero_dir,
            encoder=Stage5ObservationEncoder(),
            total_timesteps=0,
            training_root_seed=TRAIN_ROOT_SEED,
            training_config={
                "protocol": "stage5_s5_5_meter_reward_v1",
                "protocol_sha256": protocol["protocol_sha256"],
                "arm": arm,
                "reward_variant": variant,
                "observation_transform": TRANSFORM_ID,
                "only_arm_difference": "training reward signal",
                "sac": protocol["sac"],
            },
            training_summary={**initial_state, "training_steps": 0, "optimizer_updates": 0},
            save_replay_buffer=False,
            project_root=ROOT,
            observation_transform=TRANSFORM_ID,
        )
        integrity = verify_checkpoint_integrity(zero_dir)
        print(json.dumps({"event": "arm_start", "arm": arm, "variant": variant,
                          "initial_weight_fingerprint": initial_fingerprint,
                          "initial_model_zip_sha256": initial_zip_sha256,
                          "checkpoint_0_sha256": integrity["model_sha256"],
                          "seed": TRAIN_ROOT_SEED, "sampler_order": initial_state["sampler_records_order"]}), flush=True)
        timesteps = int(smoke_steps) if smoke else TRAIN_STEPS
        model.learn(total_timesteps=timesteps, callback=callback, reset_num_timesteps=True, log_interval=10,
                    progress_bar=False)
        _POST_TRAIN_CALLBACKS.pop(model, None)
        if int(model.num_timesteps) != timesteps:
            raise RuntimeError(f"Actual steps {model.num_timesteps} differ from requested {timesteps}")
        if not smoke and not (arm_dir / "checkpoints" / "step_030000").is_dir():
            raise RuntimeError("Required final 30k checkpoint was not saved")
        output = {
            "status": "TECHNICAL_SMOKE_ONLY" if smoke else "COMPLETE",
            "arm": arm,
            "reward_variant": variant,
            "actual_training_steps": int(model.num_timesteps),
            "optimizer_updates": int(getattr(model, "_n_updates", 0)),
            "completed_episodes": len(callback.episode_records),
            "episodes": callback.episode_records,
            "geometry_steps": dict(callback.geometry_steps),
            "geometry_completed_episodes": dict(callback.geometry_episodes),
            "checkpoint_summaries": callback.checkpoint_summaries,
            "telemetry_final": callback._summary(int(model.num_timesteps)),
            "ram_samples": callback.ram_samples,
            "wall_time_s": time.perf_counter() - callback.started,
            "runtime_versions": runtime_versions(),
            "protocol_sha256": protocol["protocol_sha256"],
            "source_commit": protocol["source_commit"],
            "initial_weights_fingerprint": initial_weight_fingerprint,
            "initial_model_zip_sha256": initial_zip_sha256,
            "sampler_order": initial_state["sampler_records_order"],
        }
        (arm_dir / "training_summary.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"event": "arm_complete", **output}, ensure_ascii=False), flush=True)
    finally:
        env.close()
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=RESULTS / "protocol.json")
    parser.add_argument("--arm", choices=["A", "B", "C", "ALL"], default="ALL")
    parser.add_argument("--smoke", action="store_true", help="Run a 1.2k-step technical smoke only")
    parser.add_argument("--smoke-name", default="technical_smoke", help="Unique ignored run folder for a technical smoke")
    parser.add_argument("--smoke-steps", type=int, default=1200)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    if not (RESULTS / "reward_preflight.json").is_file():
        raise RuntimeError("S5.5 reward preflight artifact missing; training refused")
    preflight = json.loads((RESULTS / "reward_preflight.json").read_text(encoding="utf-8"))
    if preflight.get("preflight", {}).get("status") != "PASS" or not preflight["preflight"].get("scientific_training_authorized_by_preflight"):
        raise RuntimeError("S5.5 reward preflight did not pass; training refused")
    if preflight.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise RuntimeError("Reward preflight and training protocol identities differ")

    if args.smoke and args.smoke_steps <= 0:
        raise ValueError("--smoke-steps must be positive")
    run_root = (ROOT / "runs" / "stage5_meter_reward" / f"{RUN_ID}_{args.smoke_name}") if args.smoke else RUNS
    initial_rel = Path("common_initial") / "model.zip"
    initial_path = run_root / initial_rel
    if initial_path.exists():
        raise FileExistsError(f"Refusing to reuse an existing fresh init without an explicit run review: {initial_path}")
    initial_path, initial_fingerprint = create_initial_model(protocol, initial_path)
    initial_zip_hash = sha256(initial_path)
    initial_meta = {
        "status": "FRESH_UNTRAINED_INITIALIZATION",
        "training_root_seed": TRAIN_ROOT_SEED,
        "weights_fingerprint": initial_fingerprint,
        "model_zip_sha256": initial_zip_hash,
        "protocol_sha256": protocol["protocol_sha256"],
        "source_commit": protocol["source_commit"],
        "runtime_versions": runtime_versions(),
        "replay_buffer": "empty; each arm receives an independent new buffer",
        "optimizers": "fresh SB3 initial optimizer state; loaded from the same saved initialization for each arm",
    }
    initial_path.parent.mkdir(parents=True, exist_ok=True)
    (initial_path.parent / "initialization.json").write_text(json.dumps(initial_meta, indent=2) + "\n", encoding="utf-8")
    if not args.smoke:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "initialization.json").write_text(json.dumps(initial_meta, indent=2) + "\n", encoding="utf-8")
    arms = ["A", "B", "C"] if args.arm == "ALL" else [args.arm]
    if args.smoke:
        arms = ["A"]
    for arm in arms:
        run_arm(protocol, arm, initial_path, initial_fingerprint, initial_zip_hash,
                run_root=run_root, smoke=args.smoke, smoke_steps=args.smoke_steps)


if __name__ == "__main__":
    main()
