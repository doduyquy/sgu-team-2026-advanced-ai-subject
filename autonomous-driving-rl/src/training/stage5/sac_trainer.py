"""
SAC Trainer and Diagnostic Pipeline for Stage 5 Reinforcement Learning (Gate S5.2).

Orchestrates model training using Stable-Baselines3 SAC wrapped around
Stage5TrainingEnv and Stage5ObservationEncoder:
- Strict single-worker MetaDrive environment lifecycle.
- Authentic Platform V1 telemetry and reward logging per episode.
- Tracks actor loss, critic loss, entropy coefficient, and simulation throughput.
- Checkpoint persistence and resume integration.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import psutil
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from src.launcher.cases import get_default_project_root
from src.platform.reward import RewardSpecV1
from src.training.stage5.checkpoint import save_sac_checkpoint
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.observation_encoder import (
    STAGE5_FULL_OBSERVATION_DIM,
    Stage5ObservationEncoder,
)
from src.training.stage5.train_sampler import (
    SamplerStrategy,
    TrainCaseSampler,
)


class SacDiagnosticsCallback(BaseCallback):
    """
    Real-time training diagnostics callback collecting episodic performance,
    gradient updates, losses, and resource utilization.
    """

    def __init__(self, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.episode_records: List[Dict[str, Any]] = []
        self.step_latencies_ms: List[float] = []
        self.start_wall_time = time.time()
        self._last_step_time = time.perf_counter_ns()
        self._proc = psutil.Process()

    def _on_step(self) -> bool:
        now_ns = time.perf_counter_ns()
        dt_ms = (now_ns - self._last_step_time) / 1e6
        self.step_latencies_ms.append(dt_ms)
        self._last_step_time = now_ns

        # Inspect environment info dicts on step completion
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for idx, done in enumerate(dones):
            if done and idx < len(infos):
                info = infos[idx]
                record = {
                    "total_timesteps": int(self.num_timesteps),
                    "step_index": int(info.get("step_index", 0)),
                    "case_id": str(info.get("case_id", "unknown")),
                    "tier": str(info.get("tier", "unknown")),
                    "geometry_generation_seed": info.get("geometry_generation_seed"),
                    "environment_seed": info.get("environment_seed"),
                    "speed_km_h": float(info.get("speed_km_h", 0.0)),
                    "route_completion": float(info.get("route_completion", 0.0)),
                    "cumulative_reward": float(info.get("cumulative_reward", 0.0)),
                    "clean_success": bool(info.get("clean_success", False)),
                    "primary_reason": str(info.get("primary_reason", "UNDETERMINED")),
                    "wall_time_s": float(time.time() - self.start_wall_time),
                    "ram_mb": float(self._proc.memory_info().rss / (1024 * 1024)),
                }
                self.episode_records.append(record)

                if self.verbose > 0:
                    print(
                        f"  [EPISODE {len(self.episode_records):3d}] "
                        f"Timestep={self.num_timesteps:6d} | "
                        f"Case={record['case_id']} | "
                        f"Steps={record['step_index']:3d} | "
                        f"Return={record['cumulative_reward']:+7.3f} | "
                        f"Route={record['route_completion']:5.1%} | "
                        f"Reason={record['primary_reason']}"
                    )
        return True

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Calculates aggregate training performance and diagnostic statistics."""
        wall_time = time.time() - self.start_wall_time
        total_steps = self.num_timesteps
        throughput = total_steps / max(0.001, wall_time)

        completed_episodes = len(self.episode_records)
        clean_success_count = sum(1 for ep in self.episode_records if ep["clean_success"])
        clean_success_rate = (clean_success_count / completed_episodes) if completed_episodes else 0.0

        route_completions = [ep["route_completion"] for ep in self.episode_records]
        mean_route_comp = float(np.mean(route_completions)) if route_completions else 0.0

        episode_returns = [ep["cumulative_reward"] for ep in self.episode_records]
        mean_return = float(np.mean(episode_returns)) if episode_returns else 0.0

        # Retrieve training losses from SB3 logger if available
        model = getattr(self, "model", None)
        logger = getattr(model, "logger", None) if model else None
        name_to_value = getattr(logger, "name_to_value", {}) if logger else {}
        actor_loss = name_to_value.get("train/actor_loss", None)
        critic_loss = name_to_value.get("train/critic_loss", None)
        ent_coef = name_to_value.get("train/ent_coef", None)

        return {
            "total_timesteps": int(total_steps),
            "completed_episodes": completed_episodes,
            "clean_success_count": clean_success_count,
            "clean_success_rate": clean_success_rate,
            "mean_route_completion": mean_route_comp,
            "mean_episode_return": mean_return,
            "wall_time_s": round(wall_time, 2),
            "throughput_steps_per_sec": round(throughput, 1),
            "actor_loss": float(actor_loss) if actor_loss is not None else None,
            "critic_loss": float(critic_loss) if critic_loss is not None else None,
            "ent_coef": float(ent_coef) if ent_coef is not None else None,
            "final_ram_mb": round(self._proc.memory_info().rss / (1024 * 1024), 1),
        }


class SacBaselineTrainer:
    """
    Configurable trainer managing the execution, logging, and persistence of SAC policies.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        training_root_seed: int = 101,
        project_root: Optional[Path] = None,
        device: str = "cpu",
    ) -> None:
        """
        Initializes the SAC Baseline Trainer.

        Args:
            config_path: Path to baseline experiment JSON configuration.
            config_overrides: Optional dictionary of hyperparameter overrides.
            training_root_seed: Master root seed.
            project_root: Subproject root directory path.
            device: Execution device ('cpu' or 'cuda').
        """
        self.project_root = project_root or get_default_project_root()
        self.training_root_seed = int(training_root_seed)
        self.device = device

        # 1. Load baseline experiment configuration
        cfg_path = config_path or (self.project_root / "configs" / "experiments" / "stage5_rl_baseline_v1.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = json.load(f)

        if config_overrides:
            self._apply_overrides(self.config, config_overrides)

        # 2. Extract hyperparameters
        sac_cfg = self.config.get("sac_hyperparameters", {})
        self.learning_rate = float(sac_cfg.get("learning_rate", 3e-4))
        self.buffer_size = int(sac_cfg.get("buffer_size", 100000))
        self.batch_size = int(sac_cfg.get("batch_size", 256))
        self.gamma = float(sac_cfg.get("gamma", 0.99))
        self.tau = float(sac_cfg.get("tau", 0.005))
        self.learning_starts = int(sac_cfg.get("learning_starts", 1000))
        self.ent_coef = sac_cfg.get("ent_coef", "auto")
        self.train_freq = int(sac_cfg.get("train_freq", 1))
        self.gradient_steps = int(sac_cfg.get("gradient_steps", 1))

        # 3. Instantiate components
        self.encoder = Stage5ObservationEncoder(core_only=False)
        sampler_cfg = self.config.get("dataset_sampler", {})
        strategy = sampler_cfg.get("default_strategy", "ROUND_ROBIN")

        self.sampler = TrainCaseSampler(
            training_root_seed=self.training_root_seed,
            strategy=SamplerStrategy(strategy),
            project_root=self.project_root,
        )

        reward_cfg = self.config.get("reward_contract", {})
        self.reward_spec = RewardSpecV1(
            progress_weight=float(reward_cfg.get("progress_weight", 1.0)),
            time_penalty_budget=float(reward_cfg.get("time_penalty_budget", 0.25)),
            success_bonus=float(reward_cfg.get("success_bonus", 1.0)),
            safety_penalty=float(reward_cfg.get("safety_penalty", 1.0)),
            unknown_termination_penalty=float(reward_cfg.get("unknown_termination_penalty", 1.0)),
            timeout_penalty=float(reward_cfg.get("timeout_penalty", 0.0)),
        )

        # 4. Instantiate Gymnasium training environment
        self.env = Stage5TrainingEnv(
            sampler=self.sampler,
            encoder=self.encoder,
            reward_spec=self.reward_spec,
            project_root=self.project_root,
        )

        # 5. Instantiate Stable-Baselines3 SAC
        self.model = SAC(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            gamma=self.gamma,
            tau=self.tau,
            learning_starts=self.learning_starts,
            ent_coef=self.ent_coef,
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            seed=self.training_root_seed,
            device=self.device,
            verbose=0,
        )

        self.callback = SacDiagnosticsCallback(verbose=1)

    def _apply_overrides(self, target: Dict[str, Any], overrides: Dict[str, Any]) -> None:
        """Recursively merges dictionary overrides into target configuration."""
        for k, v in overrides.items():
            if isinstance(v, dict) and k in target and isinstance(target[k], dict):
                self._apply_overrides(target[k], v)
            else:
                target[k] = v

    def train(self, total_timesteps: int) -> Dict[str, Any]:
        """
        Executes SAC model training for designated number of environment timesteps.

        Args:
            total_timesteps: Target environment steps to step.

        Returns:
            Dictionary containing summary metrics and diagnostic results.
        """
        print(f"\n==================================================================")
        print(f"STARTING SAC TRAINING: {total_timesteps} STEPS on MAPSUITE V1 TRAIN")
        print(f"==================================================================")
        print(f"  Learning Rate:   {self.learning_rate}")
        print(f"  Buffer Size:     {self.buffer_size}")
        print(f"  Batch Size:      {self.batch_size}")
        print(f"  Learning Starts: {self.learning_starts}")
        print(f"  Device:          {self.device}")
        print(f"  Root Seed:       {self.training_root_seed}")

        self.model.learn(total_timesteps=total_timesteps, callback=self.callback)

        summary = self.callback.get_summary_metrics()
        print(f"\n==================================================================")
        print(f"TRAINING COMPLETE: {summary['total_timesteps']} steps stepped")
        print(f"  Episodes Finished: {summary['completed_episodes']}")
        print(f"  Mean Return:       {summary['mean_episode_return']:+.3f}")
        print(f"  Mean Route:        {summary['mean_route_completion']:.1%}")
        print(f"  Clean Success:     {summary['clean_success_rate']:.1%}")
        print(f"  Throughput:        {summary['throughput_steps_per_sec']} steps/sec")
        print(f"  Wall Time:         {summary['wall_time_s']} s")
        print(f"==================================================================")
        return summary

    def save_checkpoint(
        self,
        checkpoint_dir: Path,
        save_replay_buffer: bool = False,
    ) -> Path:
        """Persists the trained model and metadata."""
        summary = self.callback.get_summary_metrics()
        return save_sac_checkpoint(
            model=self.model,
            checkpoint_dir=checkpoint_dir,
            encoder=self.encoder,
            total_timesteps=int(self.model.num_timesteps),
            training_root_seed=self.training_root_seed,
            training_config=self.config,
            training_summary=summary,
            save_replay_buffer=save_replay_buffer,
            project_root=self.project_root,
        )

    def close(self) -> None:
        """Releases simulator and environment resources."""
        if self.env is not None:
            self.env.close()
