"""S5.3A callbacks for exact step checkpoints and resource/update diagnostics."""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time
from typing import Any, Dict, Iterable, List

import numpy as np

import psutil
from stable_baselines3.common.callbacks import BaseCallback


class PilotCheckpointCallback(BaseCallback):
    def __init__(self, trainer: Any, output_root: Path, checkpoints: Iterable[int],
                 requested_timesteps: int = 10000, stop_file: Path | None = None, verbose: int = 0):
        super().__init__(verbose)
        self.trainer = trainer
        self.output_root = Path(output_root)
        self.run_root = self.output_root.parent
        self.requested_timesteps = int(requested_timesteps)
        self.stop_file = Path(stop_file) if stop_file else self.run_root / "STOP_REQUESTED"
        self.checkpoints = sorted({int(step) for step in checkpoints if int(step) > 0})
        self.saved_steps: List[int] = []
        self.resource_samples: List[Dict[str, Any]] = []
        self.started = 0.0
        self.process = psutil.Process()
        self._step_sink = None
        self.action_counts = {"throttle": 0, "brake": 0, "near_zero": 0, "steps": 0}
        self.action_sums = {"steering": 0.0, "throttle_brake": 0.0}

    def _on_training_start(self) -> None:
        self.started = time.perf_counter()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._step_sink = (self.output_root / "training_steps.jsonl").open("w", encoding="utf-8", newline="\n")

    def _on_step(self) -> bool:
        step = int(self.model.num_timesteps)
        infos = self.locals.get("infos", [])
        actions = np.asarray(self.locals.get("actions", []))
        rewards = np.asarray(self.locals.get("rewards", []))
        dones = np.asarray(self.locals.get("dones", []))
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        for idx, info in enumerate(infos):
            action = actions[idx].reshape(-1).tolist() if idx < len(actions) else []
            steering = float(action[0]) if len(action) > 0 else info.get("action_steering")
            throttle_brake = float(action[1]) if len(action) > 1 else info.get("action_throttle_brake")
            reward_parts = info.get("reward_breakdown", {})
            row = {
                "training_step": step,
                "case_id": info.get("case_id"), "geometry_id": info.get("geometry_id"),
                "tier": info.get("tier"), "geometry_generation_seed": info.get("geometry_generation_seed"),
                "environment_seed": info.get("environment_seed"), "episode_step": info.get("step_index"),
                "steering": steering, "throttle_brake": throttle_brake,
                "speed_km_h": info.get("speed_km_h"), "route_completion": info.get("route_completion"),
                "delta_route": info.get("delta_route"),
                "progress_reward": reward_parts.get("progress_reward"),
                "time_cost": reward_parts.get("time_cost"),
                "terminal_reward": reward_parts.get("terminal_reward"),
                "total_reward": float(rewards[idx]) if idx < len(rewards) else None,
                "terminated_or_truncated": bool(dones[idx]) if idx < len(dones) else False,
                "clean_success": info.get("clean_success"), "primary_reason": info.get("primary_reason"),
            }
            if self._step_sink is not None:
                self._step_sink.write(json.dumps(row, sort_keys=True) + "\n")
            if steering is not None and throttle_brake is not None:
                self.action_counts["steps"] += 1
                self.action_counts["throttle"] += int(float(throttle_brake) > 0.05)
                self.action_counts["brake"] += int(float(throttle_brake) < -0.05)
                self.action_counts["near_zero"] += int(abs(float(throttle_brake)) <= 0.05)
                self.action_sums["steering"] += float(steering)
                self.action_sums["throttle_brake"] += float(throttle_brake)
        if self._step_sink is not None and step % 100 == 0:
            self._step_sink.flush()
        if step in self.checkpoints and step not in self.saved_steps:
            self._save(step)
        if step % 100 == 0:
            sample = {
                "training_step": step,
                "elapsed_s": time.perf_counter() - self.started,
                "rss_mb": self.process.memory_info().rss / (1024 * 1024),
                "optimizer_updates": int(getattr(self.model, "_n_updates", 0)),
            }
            self.resource_samples.append(sample)
            self._write_progress(step, sample)
        if self.stop_file.exists():
            return False
        return True

    def _write_progress(self, step: int, resource_sample: Dict[str, Any]) -> None:
        values = getattr(getattr(self.model, "logger", None), "name_to_value", {}) or {}
        count = self.action_counts["steps"]
        episodes = getattr(self.trainer.callback, "episode_records", [])
        unique_geometries = sorted({row.get("geometry_id") for row in episodes if row.get("geometry_id")})
        episode_routes = [float(row["route_completion"]) for row in episodes if row.get("route_completion") is not None]
        episode_returns = [float(row["cumulative_reward"]) for row in episodes if row.get("cumulative_reward") is not None]
        payload = {
            "status": "RUNNING", "phase": "TRAINING", "training_step": step,
            "requested_timesteps": self.requested_timesteps,
            "progress_fraction": min(1.0, step / max(1, self.requested_timesteps)),
            "elapsed_s": resource_sample["elapsed_s"],
            "steps_per_second": step / max(0.001, resource_sample["elapsed_s"]),
            "optimizer_updates": resource_sample["optimizer_updates"],
            "actor_loss": _json_number(values.get("train/actor_loss")),
            "critic_loss": _json_number(values.get("train/critic_loss")),
            "entropy_coefficient": _json_number(values.get("train/ent_coef")),
            "rss_mb": resource_sample["rss_mb"],
            "completed_episodes": len(episodes),
            "clean_success_count": sum(bool(row.get("clean_success")) for row in episodes),
            "mean_completed_episode_route": statistics.mean(episode_routes) if episode_routes else None,
            "mean_completed_episode_return": statistics.mean(episode_returns) if episode_returns else None,
            "unique_training_geometries": unique_geometries,
            "action_summary": {
                "steps": count,
                "throttle_fraction": self.action_counts["throttle"] / count if count else None,
                "brake_fraction": self.action_counts["brake"] / count if count else None,
                "near_zero_fraction": self.action_counts["near_zero"] / count if count else None,
                "mean_steering": self.action_sums["steering"] / count if count else None,
                "mean_throttle_brake": self.action_sums["throttle_brake"] / count if count else None,
            },
            "updated_at_epoch_s": time.time(),
        }
        path = self.run_root / "progress.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def _save(self, step: int) -> None:
        target = self.output_root / f"step_{step:05d}"
        self.trainer.save_checkpoint(target, save_replay_buffer=False)
        self.saved_steps.append(step)
        if self.verbose:
            print(f"[PILOT CHECKPOINT] step={step} path={target}")

    def _on_training_end(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        if self._step_sink is not None:
            self._step_sink.flush()
            self._step_sink.close()
            self._step_sink = None
        count = self.action_counts["steps"]
        action_summary = {
            "steps": count,
            "throttle_fraction": self.action_counts["throttle"] / count if count else None,
            "brake_fraction": self.action_counts["brake"] / count if count else None,
            "near_zero_fraction": self.action_counts["near_zero"] / count if count else None,
            "mean_steering": self.action_sums["steering"] / count if count else None,
            "mean_throttle_brake": self.action_sums["throttle_brake"] / count if count else None,
        }
        (self.output_root / "training_action_summary.json").write_text(
            json.dumps(action_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (self.output_root / "resource_samples.json").write_text(
            json.dumps(self.resource_samples, indent=2) + "\n", encoding="utf-8"
        )


def _json_number(value: Any) -> Any:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def checkpoint_runtime_stats(trainer: Any, started: float) -> Dict[str, Any]:
    values = getattr(getattr(trainer.model, "logger", None), "name_to_value", {}) or {}
    process = psutil.Process()
    return {
        "training_step": int(trainer.model.num_timesteps),
        "optimizer_updates": int(getattr(trainer.model, "_n_updates", 0)),
        "actor_loss": values.get("train/actor_loss"),
        "critic_loss": values.get("train/critic_loss"),
        "entropy_coefficient": values.get("train/ent_coef"),
        "elapsed_s": time.perf_counter() - started,
        "rss_mb": process.memory_info().rss / (1024 * 1024),
    }
