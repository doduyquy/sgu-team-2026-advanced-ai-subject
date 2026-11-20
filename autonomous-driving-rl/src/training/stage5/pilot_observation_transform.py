"""Explicit, versioned observation transforms for Stage 5 development pilots.

These transforms run after the frozen Platform V1 encoder and are never enabled
by default. They are intended for controlled internal ablations only.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import gymnasium as gym
import numpy as np


WAYPOINT_XY50_V1 = {
    "transform_id": "pilot_waypoint_xy_div50m_v1",
    "observation_dim": 409,
    "waypoint_slice": [325, 385],
    "waypoints": 20,
    "features_per_waypoint": ["relative_x_m", "relative_y_m", "relative_heading_rad"],
    "xy_divisor_m": 50.0,
    "heading_transform": "identity",
    "all_other_features": "identity",
}


def transform_fingerprint(spec: dict[str, Any] = WAYPOINT_XY50_V1) -> str:
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class WaypointXY50Observation(gym.ObservationWrapper):
    """Scale only the waypoint relative x/y coordinates by a fixed 50 metres."""

    transform_id = WAYPOINT_XY50_V1["transform_id"]

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        low = np.array(env.observation_space.low, dtype=np.float32, copy=True)
        high = np.array(env.observation_space.high, dtype=np.float32, copy=True)
        indices = np.array([325 + 3 * i + j for i in range(20) for j in (0, 1)])
        low[indices] /= 50.0
        high[indices] /= 50.0
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        result = np.array(observation, dtype=np.float32, copy=True)
        result[325:385].reshape(20, 3)[:, :2] /= 50.0
        return result
