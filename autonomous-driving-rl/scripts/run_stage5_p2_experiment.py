"""Run the frozen P2 two-arm, 10k TRAIN-only correlated-exploration study."""
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
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER, task_producer_identity
from src.training.stage5.checkpoint import save_sac_checkpoint, verify_checkpoint_integrity
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.training.stage5.pilot_observation_transform import GROUPWISE_PHYSICAL_V1, GroupwisePhysicalObservation
from src.training.stage5.temporal_exploration import TemporalExplorationSAC
from src.training.stage5.train_sampler import SamplerStrategy, TrainCaseSampler

RUN_ID = "s5_p2_correlated_exploration_seed101_20261121_retry1"
RESULT_ID = RUN_ID
PROTOCOL_PATH = ROOT / "results/stage5/p2_correlated_exploration_seed101_20261120/protocol.json"
TRAIN_STEPS = 10_000
CHECKPOINT_STEPS = (2_000, 5_000, 10_000)
ROOT_SEED = 101
TRANSFORM_ID = GROUPWISE_PHYSICAL_V1["transform_id"]
_AFTER_OPTIMIZER: "weakref.WeakKeyDictionary[TemporalExplorationSAC, P2TrainingCallback]" = weakref.WeakKeyDictionary()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def git_head() -> str:
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    recorded = protocol.pop("protocol_sha256", None)
    if recorded != canonical_hash(protocol):
        raise ValueError("P2 protocol fingerprint mismatch")
    protocol["protocol_sha256"] = recorded
    if protocol.get("status") != "PRE_REGISTERED":
        raise ValueError("P2 protocol is not frozen as PRE_REGISTERED")
    if protocol["training"]["max_environment_steps_per_arm"] != TRAIN_STEPS:
        raise ValueError("Runner budget does not match registered protocol")
    if protocol["producer"]["id"] != REFERENCE_TASK_PRODUCER:
        raise ValueError("P2 requires the approved corrected F09 producer")
    for relative, expected in protocol.get("source_files_sha256", {}).items():
        source_path = ROOT / relative
        if not source_path.is_file() or sha256(source_path) != expected:
            raise ValueError(f"P2 frozen source hash mismatch: {relative}")
    if sha256(ROOT / protocol["geometry"]["manifest"]) != protocol["geometry"]["manifest_sha256"]:
        raise ValueError("P2 locked MapSuite manifest hash mismatch")
    train = protocol["geometry"]["training_pool"]
    diagnostic = protocol["geometry"]["diagnostic_pool"]
    train_ids = {item["geometry_id"] for item in train}
    diagnostic_ids = {item["geometry_id"] for item in diagnostic}
    if len(train) != 8 or len(diagnostic) != 6 or train_ids & diagnostic_ids:
        raise ValueError("P2 geometry schedule is incomplete or has train/diagnostic overlap")
    verifier = TrainCaseSampler(training_root_seed=ROOT_SEED, strategy=SamplerStrategy.ROUND_ROBIN,
                                allowed_geometry_ids=sorted(train_ids | diagnostic_ids), project_root=ROOT)
    for item in train + diagnostic:
        case = verifier.get_case_for_geometry_id(item["geometry_id"], int(item.get("environment_seed", 0)))
        if case.geometry_sha256 != item["geometry_sha256"] or case.split != "TRAIN":
            raise ValueError(f"P2 geometry identity/split mismatch: {item['geometry_id']}")
    return protocol


def runtime_versions() -> dict[str, str]:
    import gymnasium
    import metadrive
    import stable_baselines3
    return {
        "python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
        "gymnasium": gymnasium.__version__, "stable_baselines3": stable_baselines3.__version__,
        "metadrive": getattr(metadrive, "__version__", "0.4.3"), "device": "cpu",
    }


def make_env(protocol: dict[str, Any], arm: str, *, training: bool):
    pool_key = "training_pool" if training else "diagnostic_pool"
    pool = protocol["geometry"][pool_key]
    geometry_ids = [item["geometry_id"] for item in pool]
    sampler = TrainCaseSampler(
        training_root_seed=ROOT_SEED,
        strategy=SamplerStrategy.ROUND_ROBIN,
        allowed_geometry_ids=geometry_ids,
        project_root=ROOT,
    )
    env = Stage5TrainingEnv(
        sampler=sampler,
        training_root_seed=ROOT_SEED,
        project_root=ROOT,
        simulator_render_mode="OFF",
        task_context_producer_id=REFERENCE_TASK_PRODUCER,
    )
    if not training:
        # The evaluator drives explicit, pre-registered cases; this sampler is
        # still constrained to the diagnostic allowlist by validate_train_case.
        env._p2_diagnostic_schedule = {item["geometry_id"]: item for item in pool}
    return GroupwisePhysicalObservation(env), sampler


def model_kwargs(protocol: dict[str, Any], mode: str) -> dict[str, Any]:
    hp = protocol["sac"]
    exp = protocol["exploration"]
    return {
        "policy": "MlpPolicy", "exploration_mode": mode,
        "ou_rho": float(exp["rho"]), "ou_stationary_std": float(exp["stationary_std"]), "ou_seed": ROOT_SEED,
        "learning_rate": float(hp["learning_rate"]), "buffer_size": int(hp["buffer_size"]),
        "learning_starts": int(hp["learning_starts"]), "batch_size": int(hp["batch_size"]),
        "tau": float(hp["tau"]), "gamma": float(hp["gamma"]),
        "train_freq": (int(hp["train_freq"]), "step"), "gradient_steps": int(hp["gradient_steps"]),
        "ent_coef": hp["ent_coef"], "policy_kwargs": {"net_arch": list(hp["net_arch"]), "activation_fn": torch.nn.ReLU},
        "seed": ROOT_SEED, "device": "cpu", "verbose": 0,
    }


def state_fingerprint(model: TemporalExplorationSAC) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(value.dtype).encode()); digest.update(str(value.shape).encode()); digest.update(value.tobytes())
    entropy = getattr(model, "log_ent_coef", None)
    if entropy is not None:
        digest.update(b"log_ent_coef"); digest.update(entropy.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def create_shared_initial(protocol: dict[str, Any], run_root: Path) -> dict[str, Any]:
    target = run_root / "shared_initial"
    target.mkdir(parents=True, exist_ok=False)
    env, _ = make_env(protocol, "INITIAL", training=True)
    try:
        seed_everything(ROOT_SEED)
        model = TemporalExplorationSAC(env=env, **model_kwargs(protocol, "standard"))
        tensors = {name: value.detach().cpu().clone() for name, value in model.policy.state_dict().items()}
        alpha = getattr(model, "log_ent_coef", None)
        state = {"policy": tensors, "log_ent_coef": alpha.detach().cpu().clone() if alpha is not None else None}
        state_path = target / "initial_state.pt"
        torch.save(state, state_path)
        result = {
            "path": str(state_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(state_path), "policy_state_fingerprint": state_fingerprint(model),
            "alpha_initial": float(alpha.detach().exp().cpu()) if alpha is not None else float(model.ent_coef),
            "optimizer_states": "fresh and independent in each arm; no optimizer moments copied",
            "source_commit": git_head(),
        }
        (target / "initialization.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return {"state": state, "manifest": result}
    finally:
        env.close()


def load_initial_into(model: TemporalExplorationSAC, state: dict[str, Any]) -> None:
    model.policy.load_state_dict(state["policy"], strict=True)
    if state["log_ent_coef"] is not None:
        model.log_ent_coef.data.copy_(state["log_ent_coef"].to(model.device))
    if model.replay_buffer is not None and model.replay_buffer.size() != 0:
        raise RuntimeError("Fresh arm unexpectedly has replay data before learning")


def percentile_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p90": None, "min": None, "max": None}
    return {"mean": float(np.mean(values)), "p50": float(np.quantile(values, .5)),
            "p90": float(np.quantile(values, .9)), "min": float(np.min(values)), "max": float(np.max(values))}


def td_error_snapshot(model: TemporalExplorationSAC, max_samples: int = 256) -> dict[str, Any]:
    rb = model.replay_buffer
    size = int(rb.size())
    if size == 0:
        return {"samples": 0}
    indices = np.arange(max(0, size - max_samples), size, dtype=np.int64)
    obs = torch.as_tensor(rb.observations[indices, 0], device=model.device, dtype=torch.float32)
    next_obs = torch.as_tensor(rb.next_observations[indices, 0], device=model.device, dtype=torch.float32)
    actions = torch.as_tensor(rb.actions[indices, 0], device=model.device, dtype=torch.float32)
    rewards = torch.as_tensor(rb.rewards[indices, 0], device=model.device, dtype=torch.float32).reshape(-1, 1)
    dones = rb.dones[indices, 0].copy()
    if getattr(rb, "handle_timeout_termination", False):
        dones *= 1.0 - rb.timeouts[indices, 0]
    dones_tensor = torch.as_tensor(dones, device=model.device, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        next_actions, next_log_prob = model.actor.action_log_prob(next_obs)
        next_qs = model.critic_target(next_obs, next_actions)
        alpha = model.log_ent_coef.detach().exp() if getattr(model, "log_ent_coef", None) is not None else torch.as_tensor(model.ent_coef, device=model.device)
        next_soft_q = torch.minimum(next_qs[0], next_qs[1]) - alpha * next_log_prob.reshape(-1, 1)
        target = rewards + (1.0 - dones_tensor) * float(model.gamma) * next_soft_q
        q1, q2 = model.critic(obs, actions)
        td1, td2 = target - q1, target - q2
        td_abs = torch.cat((td1.abs().reshape(-1), td2.abs().reshape(-1)))
        q_gap = (q1 - q2).abs().reshape(-1)
    return {
        "samples": int(len(indices)), "timeout_transitions": int(np.sum(rb.timeouts[indices, 0])) if getattr(rb, "handle_timeout_termination", False) else 0,
        "mean_online_q1": float(q1.mean().cpu()), "mean_online_q2": float(q2.mean().cpu()),
        "mean_target_soft_q": float(target.mean().cpu()), "mean_abs_td_error_both_critics": float(td_abs.mean().cpu()),
        "p90_abs_td_error": float(torch.quantile(td_abs, .9).cpu()), "max_abs_td_error": float(td_abs.max().cpu()),
        "mean_abs_twin_q_gap": float(q_gap.mean().cpu()), "alpha": float(alpha.cpu()),
        "definition": "fixed most-recent replay window, SB3 SAC soft target; diagnostic sample, not optimizer batch history",
    }


class P2InstrumentedSAC(TemporalExplorationSAC):
    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_gradient_handles", "_p2_gradient_sq"]

    def _capture_gradient(self, group: str, gradient: torch.Tensor) -> torch.Tensor:
        self._p2_gradient_sq[group] += float(gradient.detach().pow(2).sum().cpu())
        return gradient

    def install_gradient_hooks(self) -> None:
        self._p2_gradient_sq = {"actor": 0.0, "critic": 0.0, "entropy": 0.0}
        self._gradient_handles = []
        for group, module in (("actor", self.actor), ("critic", self.critic)):
            for parameter in module.parameters():
                if parameter.requires_grad:
                    self._gradient_handles.append(parameter.register_hook(lambda grad, key=group: self._capture_gradient(key, grad)))
        if getattr(self, "log_ent_coef", None) is not None and self.log_ent_coef.requires_grad:
            self._gradient_handles.append(self.log_ent_coef.register_hook(lambda grad: self._capture_gradient("entropy", grad)))

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if not hasattr(self, "_p2_gradient_sq"):
            self.install_gradient_hooks()
        self._p2_gradient_sq = {"actor": 0.0, "critic": 0.0, "entropy": 0.0}
        super().train(gradient_steps=gradient_steps, batch_size=batch_size)
        callback = _AFTER_OPTIMIZER.get(self)
        if callback is not None:
            callback.after_optimizer_update()


class P2TrainingCallback(BaseCallback):
    def __init__(self, arm_dir: Path, arm: str, protocol: dict[str, Any], initial_fingerprint: str) -> None:
        super().__init__(verbose=0)
        self.arm_dir, self.arm, self.protocol = arm_dir, arm, protocol
        self.geometry_lookup = {item["geometry_id"]: item for item in protocol["geometry"]["training_pool"]}
        self.initial_fingerprint = initial_fingerprint
        self.training_started = time.perf_counter()
        self.last_sample_time = self.training_started
        self.episode_index = 0
        self.episode_steps = 0
        self.episode_return = 0.0
        self.episode_progress = 0.0
        self.episode_geom: dict[str, Any] = {}
        self.geometry_steps: Counter[str] = Counter()
        self.episode_records: list[dict[str, Any]] = []
        self.ram_samples: list[dict[str, Any]] = []
        self.optimizer_file = None
        self.action_file = None
        self.episode_file = None
        self.pending_checkpoint_step: int | None = None
        self.checkpoints: list[dict[str, Any]] = []
        self.updates_recorded = 0
        self.action_records: list[dict[str, Any]] = []

    def _on_training_start(self) -> None:
        # The step-0 bundle is written before learn() starts.  Reuse that
        # already-created directory, while x-mode below still prevents any
        # telemetry file from being overwritten on a rerun.
        self.arm_dir.mkdir(parents=True, exist_ok=True)
        (self.arm_dir / "checkpoints").mkdir(exist_ok=True)
        self.action_file = (self.arm_dir / "replay_action_trace.jsonl").open("x", encoding="utf-8", newline="\n")
        self.episode_file = (self.arm_dir / "training_episodes.jsonl").open("x", encoding="utf-8", newline="\n")
        self.optimizer_file = (self.arm_dir / "optimizer_telemetry.jsonl").open("x", encoding="utf-8", newline="\n")
        self.model.action_audit_sink = self.write_action

    def write_action(self, record: dict[str, Any]) -> None:
        assert self.action_file is not None
        self.action_file.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        self.action_records.append(record)
        # Flush by observed rows, not the pre-step SB3 counter, so live
        # monitoring remains reliable even if that counter is zero-based.
        if len(self.action_records) % 100 == 0:
            self.action_file.flush()

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self.episode_steps += 1
        self.episode_return += float(self.locals["rewards"][0])
        self.episode_progress += float(info.get("delta_route", 0.0))
        geom = info.get("geometry_id", "UNKNOWN")
        geometry_identity = self.geometry_lookup.get(str(geom), {})
        self.episode_geom = {
            "case_id": info.get("case_id"), "geometry_id": geom,
            "geometry_generation_seed": info.get("geometry_generation_seed"),
            "geometry_sha256": geometry_identity.get("geometry_sha256"), "environment_seed": info.get("environment_seed"),
            "tier": info.get("tier"), "sequence": geometry_identity.get("sequence"),
        }
        self.geometry_steps[str(geom)] += 1
        step = int(self.model.num_timesteps)
        if step > 0 and step % 1000 == 0:
            try:
                import psutil
                self.ram_samples.append({"step": step, "rss_mb": psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)})
            except Exception:
                self.ram_samples.append({"step": step, "rss_mb": None})
        dones = np.asarray(self.locals["dones"], dtype=bool)
        if bool(dones[0]):
            outcome = info.get("outcome", {})
            record = {
                "arm": self.arm, "episode_index": self.episode_index,
                **self.episode_geom, "steps": self.episode_steps, "episode_return": self.episode_return,
                "route_progress_delta": self.episode_progress, "outcome": outcome.get("primary_reason", info.get("primary_reason")),
                "clean_success": bool(outcome.get("clean_success", info.get("clean_success", False))),
                "terminated": bool(outcome.get("terminated", not info.get("TimeLimit.truncated", False))),
                "truncated": bool(outcome.get("truncated", info.get("TimeLimit.truncated", False))),
            }
            self.episode_records.append(record)
            assert self.episode_file is not None
            self.episode_file.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            self.episode_file.flush()
            self.episode_index += 1
            self.episode_steps, self.episode_return, self.episode_progress = 0, 0.0, 0.0
            self.episode_geom = {}
        if step in CHECKPOINT_STEPS:
            self.pending_checkpoint_step = step
        return True

    def after_optimizer_update(self) -> None:
        model = self.model
        step = int(model.num_timesteps)
        self.updates_recorded += 1
        if step % 100 == 0:
            metrics = model.logger.name_to_value
            row = {
                "training_step": step, "optimizer_updates": int(model._n_updates),
                "alpha": float(model.log_ent_coef.detach().exp().cpu()) if getattr(model, "log_ent_coef", None) is not None else float(model.ent_coef),
                "actor_loss": _safe_metric(metrics.get("train/actor_loss")),
                "critic_loss": _safe_metric(metrics.get("train/critic_loss")),
                "ent_coef_loss": _safe_metric(metrics.get("train/ent_coef_loss")),
                "actor_gradient_global_norm": math.sqrt(model._p2_gradient_sq["actor"]),
                "critic_gradient_global_norm": math.sqrt(model._p2_gradient_sq["critic"]),
                "entropy_gradient_l2": math.sqrt(model._p2_gradient_sq["entropy"]),
            }
            assert self.optimizer_file is not None
            self.optimizer_file.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            self.optimizer_file.flush()
            self.updates_recorded += 0
        if self.pending_checkpoint_step is not None:
            checkpoint_step = self.pending_checkpoint_step
            self.pending_checkpoint_step = None
            self.save_checkpoint(checkpoint_step)

    def save_checkpoint(self, step: int) -> None:
        encoder = Stage5ObservationEncoder(core_only=False)
        destination = self.arm_dir / "checkpoints" / f"step_{step:06d}"
        config = {
            "protocol": self.protocol["protocol"], "protocol_sha256": self.protocol["protocol_sha256"],
            "arm": self.arm, "learning_rate": self.protocol["sac"]["learning_rate"],
            "buffer_size": self.protocol["sac"]["buffer_size"], "batch_size": self.protocol["sac"]["batch_size"],
            "gamma": self.protocol["sac"]["gamma"], "tau": self.protocol["sac"]["tau"],
            "learning_starts": self.protocol["sac"]["learning_starts"], "gradient_steps": self.protocol["sac"]["gradient_steps"],
            "task_context_producer_id": REFERENCE_TASK_PRODUCER,
            "exploration": self.protocol["exploration"] if self.arm == "B" else {"mode": "standard_sampling"},
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
            self.model, destination, encoder, total_timesteps=step, training_root_seed=ROOT_SEED,
            training_config=config, training_summary=snapshot, save_replay_buffer=(step == TRAIN_STEPS),
            project_root=ROOT, observation_transform=TRANSFORM_ID,
            task_context_producer_id=REFERENCE_TASK_PRODUCER,
        )
        metadata = verify_checkpoint_integrity(
            saved, expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
            task_context_producer_id=REFERENCE_TASK_PRODUCER,
        )
        td = td_error_snapshot(self.model)
        row = {
            "step": step, "model_sha256": metadata["model_sha256"],
            "checkpoint_bundle_identity": metadata.get("checkpoint_identity"),
            "metadata_sha256": sha256(saved / "metadata.json"), "alpha": snapshot["alpha"],
            "optimizer_updates": int(self.model._n_updates), "completed_episodes": self.episode_index,
            "replay_size": int(self.model.replay_buffer.size()), "fixed_recent_replay_td_diagnostic": td,
            "replay_buffer_sha256": metadata.get("replay_buffer_sha256"),
            "checkpoint_path": str(saved.relative_to(ROOT)).replace("\\", "/"),
        }
        self.checkpoints.append(row)
        checkpoint_summary_path = self.arm_dir / "checkpoint_telemetry.json"
        checkpoint_summary_path.write_text(json.dumps(self.checkpoints, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"[{self.arm}] checkpoint {step}: {metadata['model_sha256']} alpha={snapshot['alpha']:.6g} updates={self.model._n_updates}", flush=True)

    def _on_training_end(self) -> None:
        for handle in (self.action_file, self.episode_file, self.optimizer_file):
            if handle is not None and not handle.closed:
                handle.flush(); handle.close()

    def summary(self) -> dict[str, Any]:
        elapsed = time.perf_counter() - self.training_started
        outcomes = Counter(str(item["outcome"]) for item in self.episode_records)
        executed = np.asarray([row["executed_action"] for row in self.action_records], dtype=np.float64)
        throttle = executed[:, 1] if len(executed) else np.asarray([], dtype=np.float64)
        noise = np.asarray([row["ou_noise"] for row in self.action_records], dtype=np.float64)
        per_episode_throttle: dict[int, list[float]] = {}
        per_episode_noise: dict[int, list[float]] = {}
        for row in self.action_records:
            episode_key = int(row["episode_index"])
            per_episode_throttle.setdefault(episode_key, []).append(float(row["executed_action"][1]))
            per_episode_noise.setdefault(episode_key, []).append(float(row["ou_noise"][1]))
        switches = 0
        positive_runs: list[int] = []
        for sequence in per_episode_throttle.values():
            signs = np.sign(np.asarray(sequence))
            switches += int(np.sum((signs[1:] * signs[:-1]) < 0))
            run = 0
            for value in sequence:
                if value > 0:
                    run += 1
                elif run:
                    positive_runs.append(run); run = 0
            if run:
                positive_runs.append(run)
        def lag_one(values: np.ndarray) -> float | None:
            if len(values) < 3 or float(np.std(values[:-1])) == 0 or float(np.std(values[1:])) == 0:
                return None
            return float(np.corrcoef(values[:-1], values[1:])[0, 1])
        def mean_episode_lag_one(sequences: dict[int, list[float]]) -> float | None:
            values = [lag_one(np.asarray(sequence, dtype=np.float64)) for sequence in sequences.values()]
            finite = [value for value in values if value is not None and math.isfinite(value)]
            return float(np.mean(finite)) if finite else None
        return {
            "arm": self.arm, "status": "COMPLETE" if int(self.model.num_timesteps) == TRAIN_STEPS else "INCOMPLETE",
            "actual_environment_steps": int(self.model.num_timesteps), "optimizer_updates": int(self.model._n_updates),
            "optimizer_updates_telemetry_rows": self.updates_recorded // 100,
            "completed_episodes": len(self.episode_records), "episode_outcomes": dict(outcomes),
            "partial_episode_at_end": {
                "episode_index": self.episode_index, "steps_so_far": self.episode_steps,
                "return_so_far": self.episode_return, "route_progress_delta_so_far": self.episode_progress,
                "case": self.episode_geom or None,
            },
            "geometry_step_counts": dict(self.geometry_steps), "unique_geometries_visited": len(self.geometry_steps),
            "wall_time_seconds": elapsed, "throughput_steps_per_second": int(self.model.num_timesteps) / max(elapsed, 1e-9),
            "ram_samples": self.ram_samples, "action_mismatch_count": int(self.model.action_mismatch_count),
            "ou_action_steps": int(self.model.action_noise_count), "ou_clipped_components": int(self.model.action_clip_count),
            "ou_episode_resets": int(self.model.ou_episode_resets), "checkpoints": self.checkpoints,
            "replay_action_distribution": {
                "transition_count": int(len(executed)),
                "steering_mean_std": [float(np.mean(executed[:, 0])), float(np.std(executed[:, 0]))] if len(executed) else None,
                "throttle_mean_std": [float(np.mean(throttle)), float(np.std(throttle))] if len(throttle) else None,
                "throttle_positive_fraction": float(np.mean(throttle > 0)) if len(throttle) else None,
                "throttle_near_zero_fraction_abs_lt_005": float(np.mean(np.abs(throttle) < .05)) if len(throttle) else None,
                "throttle_brake_sign_switches": switches,
                "throttle_positive_run_length_mean": float(np.mean(positive_runs)) if positive_runs else 0.0,
                "throttle_positive_run_length_median": float(np.median(positive_runs)) if positive_runs else 0.0,
                "executed_throttle_lag1_autocorrelation_global": lag_one(throttle),
                "ou_noise_throttle_lag1_autocorrelation_global": lag_one(noise[:, 1]) if len(noise) else None,
                "executed_throttle_lag1_autocorrelation_mean_within_episode": mean_episode_lag_one(per_episode_throttle),
                "ou_noise_throttle_lag1_autocorrelation_mean_within_episode": mean_episode_lag_one(per_episode_noise),
                "episode_sequences_with_estimable_lag1": sum(len(sequence) >= 3 and float(np.std(sequence[:-1])) > 0 and float(np.std(sequence[1:])) > 0 for sequence in per_episode_noise.values()),
                "warning": "Global lag-1 includes episode boundaries; run-length and switch statistics reset per episode.",
            },
            "initial_policy_fingerprint": self.initial_fingerprint,
            "training_source_commit": git_head(), "runtime_versions": runtime_versions(),
            "run_manifest_sha256": sha256(PROTOCOL_PATH),
            "benchmark_eligible": False,
        }


def _safe_metric(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def run_arm(protocol: dict[str, Any], arm: str, initial: dict[str, Any], run_root: Path) -> dict[str, Any]:
    mode = "standard" if arm == "A" else "ou_ar1_v1"
    arm_dir = run_root / f"arm_{arm}"
    if arm_dir.exists():
        raise FileExistsError(f"Refusing to overwrite P2 arm artifacts: {arm_dir}")
    env, _ = make_env(protocol, arm, training=True)
    callback: P2TrainingCallback | None = None
    try:
        seed_everything(ROOT_SEED)
        model = P2InstrumentedSAC(env=env, **model_kwargs(protocol, mode))
        load_initial_into(model, initial["state"])
        model.install_gradient_hooks()
        fingerprint = state_fingerprint(model)
        if fingerprint != initial["manifest"]["policy_state_fingerprint"]:
            raise RuntimeError(f"Arm {arm} initial policy differs from registered shared initialization")
        callback = P2TrainingCallback(arm_dir, arm, protocol, fingerprint)
        # SB3 normally attaches this reference inside learn().  We need it for
        # the required pre-training step-0 checkpoint.
        callback.model = model
        _AFTER_OPTIMIZER[model] = callback
        callback.save_checkpoint(0)
        model.learn(total_timesteps=TRAIN_STEPS, callback=callback, reset_num_timesteps=True, log_interval=10)
        if int(model.num_timesteps) != TRAIN_STEPS:
            raise RuntimeError(f"Arm {arm} ended at {model.num_timesteps}, expected {TRAIN_STEPS}")
        if callback.checkpoints[-1]["step"] != TRAIN_STEPS:
            raise RuntimeError(f"Arm {arm} lacks final scheduled checkpoint")
        if model.action_mismatch_count != 0:
            raise RuntimeError(f"Arm {arm} action/replay mismatches: {model.action_mismatch_count}")
        summary = callback.summary()
        (arm_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"[{arm}] COMPLETE steps={summary['actual_environment_steps']} updates={summary['optimizer_updates']} episodes={summary['completed_episodes']} geometries={summary['unique_geometries_visited']} wall={summary['wall_time_seconds']:.1f}s", flush=True)
        return summary
    finally:
        if callback is not None:
            callback._on_training_end()
        env.close()
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("A", "B", "ALL"), default="ALL")
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    protocol = read_protocol()
    run_root = ROOT / "runs/stage5_p2" / args.run_id
    if args.arm == "ALL":
        if run_root.exists():
            raise FileExistsError(f"Refusing to overwrite existing P2 run: {run_root}")
        run_root.mkdir(parents=True)
        initial = create_shared_initial(protocol, run_root)
    else:
        state_path = run_root / "shared_initial/initial_state.pt"
        if not state_path.is_file():
            raise FileNotFoundError("Single-arm execution requires a previously frozen shared initial state")
        initial_json = json.loads((run_root / "shared_initial/initialization.json").read_text(encoding="utf-8"))
        if sha256(state_path) != initial_json["sha256"]:
            raise ValueError("Shared initial state SHA mismatch")
        initial = {"state": torch.load(state_path, map_location="cpu", weights_only=False), "manifest": initial_json}
    selected_arms = ["A", "B"] if args.arm == "ALL" else [args.arm]
    completed: list[dict[str, Any]] = []
    manifest_path = run_root / "run_manifest.json"
    manifest_base = {
        "run_id": args.run_id, "protocol_sha256": protocol["protocol_sha256"],
        "training_source_commit": git_head(), "arms_requested": selected_arms,
        "training_performed": True, "official_validation_test_accessed": False,
        "benchmark_eligible": False,
    }
    try:
        for arm in selected_arms:
            completed.append(run_arm(protocol, arm, initial, run_root))
            manifest_path.write_text(json.dumps({**manifest_base, "completed_arms": completed,
                                                 "status": "RUNNING"}, indent=2) + "\n", encoding="utf-8")
    except BaseException as exc:
        manifest_path.write_text(json.dumps({**manifest_base, "completed_arms": completed,
                                             "status": "FAILED_OR_INTERRUPTED",
                                             "exception_type": type(exc).__name__,
                                             "exception": str(exc)}, indent=2) + "\n", encoding="utf-8")
        raise
    else:
        status = "COMPLETE" if selected_arms == ["A", "B"] else "PARTIAL_SINGLE_ARM_ONLY"
        manifest_path.write_text(json.dumps({**manifest_base, "completed_arms": completed,
                                             "status": status}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
