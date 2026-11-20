"""Explicit pilot-only temporally correlated action exploration for SAC."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.noise import ActionNoise


class OrnsteinUhlenbeckActionNoise:
    """Stationary AR(1) action perturbation with resettable episode state."""

    def __init__(self, rho: float, stationary_std: float, seed: int, action_dim: int = 2) -> None:
        if not 0.0 <= rho < 1.0:
            raise ValueError("rho must be in [0, 1)")
        if stationary_std <= 0.0:
            raise ValueError("stationary_std must be positive")
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        self.rho = float(rho)
        self.stationary_std = float(stationary_std)
        self.innovation_std = self.stationary_std * float(np.sqrt(1.0 - self.rho**2))
        self.action_dim = int(action_dim)
        self.rng = np.random.default_rng(int(seed))
        self.state = np.zeros(self.action_dim, dtype=np.float32)

    def reset_episode(self) -> None:
        self.state.fill(0.0)

    def sample(self) -> np.ndarray:
        innovation = self.rng.normal(0.0, self.innovation_std, size=self.action_dim).astype(np.float32)
        self.state = (self.rho * self.state + innovation).astype(np.float32)
        return self.state.copy()


class TemporalExplorationSAC(SAC):
    """SAC with an optional additive OU sampler applied before env/replay diverge.

    The policy sample, perturbed action sent to the environment, and action
    inserted into replay are recorded and compared. The standard control mode
    uses SB3's sampler unchanged. ``ou_ar1_v1`` adds noise only after random
    warm-up, clips to the declared Box bounds, and resets process state on every
    termination or truncation.
    """

    def __init__(
        self,
        *args: Any,
        exploration_mode: str = "standard",
        ou_rho: float = 0.85,
        ou_stationary_std: float = 0.15,
        ou_seed: int = 101,
        **kwargs: Any,
    ) -> None:
        if exploration_mode not in ("standard", "ou_ar1_v1"):
            raise ValueError(f"Unsupported exploration mode: {exploration_mode}")
        self.exploration_mode = exploration_mode
        self.ou_rho = float(ou_rho)
        self.ou_stationary_std = float(ou_stationary_std)
        self.ou_seed = int(ou_seed)
        self.ou_process = OrnsteinUhlenbeckActionNoise(self.ou_rho, self.ou_stationary_std, self.ou_seed)
        self.last_sample_audit: dict[str, Any] | None = None
        self.action_audit_sink: Callable[[dict[str, Any]], None] | None = None
        self.action_mismatch_count = 0
        self.episode_index = 0
        self.ou_episode_resets = 0
        self.action_clip_count = 0
        self.action_noise_count = 0
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self) -> list[str]:
        # The OU generator and live audit callback are process-local collection
        # state. They are represented in the run protocol/logs, not model.zip.
        return super()._excluded_save_params() + [
            "ou_process", "last_sample_audit", "action_audit_sink", "action_mismatch_count",
            "episode_index", "action_clip_count", "action_noise_count",
            "ou_episode_resets",
            "exploration_mode", "ou_rho", "ou_stationary_std", "ou_seed",
        ]

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: ActionNoise | None = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        env_actions, buffer_actions = super()._sample_action(
            learning_starts=learning_starts, action_noise=action_noise, n_envs=n_envs,
        )
        if n_envs != 1:
            raise RuntimeError("P2 temporal exploration is registered only for one training environment")
        if not isinstance(self.action_space, spaces.Box) or buffer_actions.shape != (1, 2):
            raise RuntimeError("P2 temporal exploration requires continuous Box(2)")
        low = np.asarray(self.action_space.low, dtype=np.float32)
        high = np.asarray(self.action_space.high, dtype=np.float32)
        if not (np.array_equal(low, [-1.0, -1.0]) and np.array_equal(high, [1.0, 1.0])):
            raise RuntimeError("P2 action sampler requires continuous_box2_v1 bounds [-1, 1]")

        policy_sample = np.asarray(buffer_actions, dtype=np.float32).copy()
        noise = np.zeros_like(policy_sample)
        clip_mask = np.zeros_like(policy_sample, dtype=bool)
        executed = policy_sample.copy()
        if self.exploration_mode == "ou_ar1_v1" and int(self.num_timesteps) >= int(learning_starts):
            noise[0] = self.ou_process.sample()
            unbounded = policy_sample + noise
            executed = np.clip(unbounded, low, high).astype(np.float32)
            clip_mask = unbounded != executed
            self.action_noise_count += 1
            self.action_clip_count += int(np.count_nonzero(clip_mask))
            env_actions = self.policy.unscale_action(executed)
            buffer_actions = executed

        self.last_sample_audit = {
            "training_step": int(self.num_timesteps) + 1,
            "episode_index": int(self.episode_index),
            "policy_sample": policy_sample[0].astype(float).tolist(),
            "ou_noise": noise[0].astype(float).tolist(),
            "executed_action_candidate": executed[0].astype(float).tolist(),
            "clip_mask": clip_mask[0].tolist(),
            "mode": self.exploration_mode,
        }
        return env_actions, buffer_actions

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos) -> None:
        if len(infos) != 1 or len(dones) != 1:
            raise RuntimeError("P2 action-integrity audit requires one environment")
        audit = dict(self.last_sample_audit or {})
        info = infos[0]
        env_action = np.asarray([info.get("action_steering"), info.get("action_throttle_brake")], dtype=np.float32)
        buffer_action_row = np.asarray(buffer_action[0], dtype=np.float32)
        if not np.isfinite(env_action).all() or not np.isfinite(buffer_action_row).all():
            self.action_mismatch_count += 1
            raise RuntimeError("Non-finite executed/replay action in P2 collection")
        if not np.allclose(env_action, buffer_action_row, rtol=0.0, atol=1e-6):
            self.action_mismatch_count += 1
            raise RuntimeError(f"Executed action differs from replay action: env={env_action}, replay={buffer_action_row}")

        slot = int(replay_buffer.pos)
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
        replay_row = np.asarray(replay_buffer.actions[slot, 0], dtype=np.float32)
        if not np.array_equal(replay_row, buffer_action_row):
            self.action_mismatch_count += 1
            raise RuntimeError(f"Stored replay action differs from sampled buffer action: stored={replay_row}, input={buffer_action_row}")

        audit.update({
            "geometry_id": info.get("geometry_id"),
            "environment_seed": info.get("environment_seed"),
            "executed_action": env_action.astype(float).tolist(),
            "replay_action": replay_row.astype(float).tolist(),
            "speed_km_h": float(info.get("speed_km_h", 0.0)),
            "route_completion": float(info.get("route_completion", 0.0)),
            "delta_route": float(info.get("delta_route", 0.0)),
            "reward": float(reward[0]),
            "reward_breakdown": info.get("reward_breakdown"),
            "terminated": bool(dones[0] and not info.get("TimeLimit.truncated", False)),
            "truncated": bool(info.get("TimeLimit.truncated", False)),
            "outcome": info.get("outcome"),
        })
        if self.action_audit_sink is not None:
            self.action_audit_sink(audit)
        if bool(dones[0]):
            if self.exploration_mode == "ou_ar1_v1":
                self.ou_process.reset_episode()
                self.ou_episode_resets += 1
            self.episode_index += 1

