"""Versioned Stage 5-only pass-through of MetaDrive's native training reward.

RewardSpecV1 remains computed by Stage5TrainingEnv and is retained in info for
comparison. This wrapper changes only the reward returned to the SAC learner.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import gymnasium as gym

from src.training.stage5.env_adapter import Stage5TrainingEnv


NATIVE_REWARD_TRAINING_MODE_ID = "stage5_metadrive_native_reward_v1"
NATIVE_REWARD_KEYS = (
    "success_reward",
    "out_of_road_penalty",
    "crash_vehicle_penalty",
    "crash_object_penalty",
    "crash_sidewalk_penalty",
    "driving_reward",
    "speed_reward",
    "use_lateral_reward",
)


def _fingerprint(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class NativeMetaDriveRewardTrainingEnv(gym.Wrapper):
    """Use native MetaDrive reward for learning while retaining RewardSpecV1 telemetry."""

    def __init__(self, env: Stage5TrainingEnv) -> None:
        if not isinstance(env.unwrapped, Stage5TrainingEnv):
            raise TypeError("Native reward mode requires Stage5TrainingEnv")
        super().__init__(env)
        from metadrive.envs.metadrive_env import MetaDriveEnv
        from metadrive.version import VERSION

        defaults = MetaDriveEnv.default_config()
        config = {key: defaults[key] for key in NATIVE_REWARD_KEYS}
        self.reward_mode_identity = {
            "id": NATIVE_REWARD_TRAINING_MODE_ID,
            "runtime_version": str(VERSION),
            "native_reward_configuration": config,
            "mapping": "return raw MetaDriveEnv.step reward without scaling or clipping",
            "comparison_signal": "RewardSpecV1 remains independently computed and logged in info",
        }
        self.reward_mode_identity["fingerprint"] = _fingerprint(self.reward_mode_identity)

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        base = self.env.unwrapped
        active = {key: base._env.config[key] for key in NATIVE_REWARD_KEYS}
        if active != self.reward_mode_identity["native_reward_configuration"]:
            raise RuntimeError(
                "Stage 5 MetaDrive reward config differs from the registered native reward mode: "
                f"active={active!r} registered={self.reward_mode_identity['native_reward_configuration']!r}"
            )
        enriched = dict(info)
        enriched["training_reward_mode_id"] = NATIVE_REWARD_TRAINING_MODE_ID
        enriched["training_reward_mode_fingerprint"] = self.reward_mode_identity["fingerprint"]
        return observation, enriched

    def step(self, action: Any):
        observation, reward_spec_v1, terminated, truncated, info = self.env.step(action)
        native_reward = info.get("native_metadrive_reward")
        recorded_v1 = info.get("reward_spec_v1_total")
        if native_reward is None or recorded_v1 is None:
            raise RuntimeError("Stage5TrainingEnv omitted parallel native/RewardSpecV1 reward telemetry")
        native_reward = float(native_reward)
        if not math.isfinite(native_reward) or not math.isfinite(float(reward_spec_v1)):
            raise RuntimeError("Non-finite native or RewardSpecV1 reward")
        if not math.isclose(float(recorded_v1), float(reward_spec_v1), rel_tol=0.0, abs_tol=1e-8):
            raise RuntimeError("RewardSpecV1 info does not match Stage5TrainingEnv return value")
        enriched = dict(info)
        enriched["training_reward_mode_id"] = NATIVE_REWARD_TRAINING_MODE_ID
        enriched["training_reward_mode_fingerprint"] = self.reward_mode_identity["fingerprint"]
        enriched["native_training_reward"] = native_reward
        enriched["reward_spec_v1_comparison"] = float(reward_spec_v1)
        return observation, native_reward, terminated, truncated, enriched
