"""S5.3A callbacks for exact step checkpoints and resource/update diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List

import psutil
from stable_baselines3.common.callbacks import BaseCallback


class PilotCheckpointCallback(BaseCallback):
    def __init__(self, trainer: Any, output_root: Path, checkpoints: Iterable[int], verbose: int = 0):
        super().__init__(verbose)
        self.trainer = trainer
        self.output_root = Path(output_root)
        self.checkpoints = sorted({int(step) for step in checkpoints if int(step) > 0})
        self.saved_steps: List[int] = []
        self.resource_samples: List[Dict[str, Any]] = []
        self.started = 0.0
        self.process = psutil.Process()

    def _on_training_start(self) -> None:
        self.started = time.perf_counter()

    def _on_step(self) -> bool:
        step = int(self.model.num_timesteps)
        if step in self.checkpoints and step not in self.saved_steps:
            self._save(step)
        if step % 100 == 0:
            self.resource_samples.append({
                "training_step": step,
                "elapsed_s": time.perf_counter() - self.started,
                "rss_mb": self.process.memory_info().rss / (1024 * 1024),
                "optimizer_updates": int(getattr(self.model, "_n_updates", 0)),
            })
        return True

    def _save(self, step: int) -> None:
        target = self.output_root / f"step_{step:05d}"
        self.trainer.save_checkpoint(target, save_replay_buffer=False)
        self.saved_steps.append(step)
        if self.verbose:
            print(f"[PILOT CHECKPOINT] step={step} path={target}")

    def _on_training_end(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "resource_samples.json").write_text(
            json.dumps(self.resource_samples, indent=2) + "\n", encoding="utf-8"
        )


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
