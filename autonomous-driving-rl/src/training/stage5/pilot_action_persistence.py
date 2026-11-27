"""Pilot-only SAC action persistence intervention.

This module keeps the throttle component of each sampled action for a fixed
number of environment steps while allowing steering to be sampled every step.
Both the action sent to VecEnv and the normalized action returned for replay
storage are changed together, so replay transitions describe the executed
action rather than the discarded actor sample.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise


class ThrottlePersistenceSAC(SAC):
    """SAC collector variant for a fixed-duration throttle-hold pilot."""

    def __init__(self, *args, throttle_hold_steps: int, **kwargs):
        if int(throttle_hold_steps) < 2:
            raise ValueError("throttle_hold_steps must be at least 2")
        self.throttle_hold_steps = int(throttle_hold_steps)
        self._held_throttle: Optional[np.ndarray] = None
        self._last_hold_block: Optional[int] = None
        super().__init__(*args, **kwargs)

    def reset_throttle_persistence(self) -> None:
        """Start a fresh persistence block after an episode boundary."""
        self._held_throttle = None
        self._last_hold_block = None

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: ActionNoise | None = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        actions, buffer_actions = super()._sample_action(
            learning_starts=learning_starts,
            action_noise=action_noise,
            n_envs=n_envs,
        )
        if n_envs != 1:
            raise RuntimeError("Throttle persistence pilot is restricted to one environment")
        if not isinstance(self.action_space, spaces.Box) or buffer_actions.shape[-1] != 2:
            raise RuntimeError("Throttle persistence requires the continuous Box(2) action space")
        low = np.asarray(self.action_space.low, dtype=np.float32)
        high = np.asarray(self.action_space.high, dtype=np.float32)
        if not (np.array_equal(low, np.asarray([-1.0, -1.0], dtype=np.float32)) and
                np.array_equal(high, np.asarray([1.0, 1.0], dtype=np.float32))):
            raise RuntimeError("Throttle persistence expects the normalized continuous_box2_v1 bounds")

        hold_block = int(self.num_timesteps) // self.throttle_hold_steps
        if self._held_throttle is None or self._last_hold_block != hold_block:
            self._held_throttle = buffer_actions[:, 1].copy()
            self._last_hold_block = hold_block

        # Box(2) is already normalized to [-1, 1], so the buffer action and
        # environment action have identical values. Update both before env.step.
        buffer_actions[:, 1] = self._held_throttle
        actions = self.policy.unscale_action(buffer_actions)
        return actions, buffer_actions


class ThrottlePersistenceResetCallback(BaseCallback):
    """Reset the held throttle at each completed or truncated episode."""

    def _on_step(self) -> bool:
        dones = np.asarray(self.locals.get("dones", []), dtype=bool)
        if np.any(dones):
            reset = getattr(self.model, "reset_throttle_persistence", None)
            if reset is None:
                raise RuntimeError("Persistence reset callback is paired with the wrong SAC model")
            reset()
        return True
