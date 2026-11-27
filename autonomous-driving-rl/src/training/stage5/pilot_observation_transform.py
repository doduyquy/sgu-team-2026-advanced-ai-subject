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

# Exploratory S5.4 treatment. This is a post-encoder policy-input transform;
# Stage5ObservationEncoder and its frozen 409D schema remain unchanged.
GROUPWISE_PHYSICAL_V1 = {
    "transform_id": "pilot_groupwise_physical_v1",
    "observation_dim": 409,
    "core": {"slice": [0, 259], "transform": "identity"},
    "traffic_actors": {
        "slice": [259, 315], "shape": [8, 7],
        "fields": ["relative_x_m", "relative_y_m", "relative_vx_m_s", "relative_vy_m_s",
                   "relative_heading_rad", "length_m", "width_m"],
        "divisors": [50.0, 50.0, 30.0, 30.0, 3.141592653589793, 5.0, 5.0],
        "rationale": "50m is the declared traffic context radius; 30m/s and 5m are fixed engineering reference scales, not clipping bounds.",
    },
    "traffic_mask_metrics": {"slice": [315, 325], "transform": "identity"},
    "task_waypoints": {
        "slice": [325, 385], "shape": [20, 3],
        "fields": ["relative_x_m", "relative_y_m", "relative_heading_rad"],
        "divisors": [50.0, 50.0, 3.141592653589793],
        "rationale": "50m is the declared waypoint lookahead scale; heading is bounded by pi radians.",
    },
    "task_mask_scalars": {"slice": [385, 409], "transform": "identity"},
    "clipping": "none",
    "fitted_statistics": "none",
    "mask_semantics": "preserved exactly",
    "input_rights": "public AgentInputV1 fields only",
}

TRANSFORM_SPECS = {
    WAYPOINT_XY50_V1["transform_id"]: WAYPOINT_XY50_V1,
    GROUPWISE_PHYSICAL_V1["transform_id"]: GROUPWISE_PHYSICAL_V1,
}


def transform_spec(transform_id: str | None) -> dict[str, Any] | None:
    """Return the immutable versioned spec for a transform ID, if enabled."""
    if transform_id is None:
        return None
    try:
        return TRANSFORM_SPECS[transform_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported Stage 5 observation transform: {transform_id}") from exc


def transform_fingerprint(spec: dict[str, Any] = WAYPOINT_XY50_V1) -> str:
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def apply_observation_transform(observation: np.ndarray, transform_id: str | None) -> np.ndarray:
    """Apply a versioned policy-input transform without changing masks/padding."""
    result = np.array(observation, dtype=np.float32, copy=True)
    if result.shape[-1:] != (409,):
        raise ValueError(f"Expected final observation dimension 409, got {result.shape}")
    if transform_id is None:
        return result
    if transform_id in (WAYPOINT_XY50_V1["transform_id"], "waypoint_xy_div50m_v1"):
        result[..., 325:385].reshape(*result.shape[:-1], 20, 3)[..., :2] /= 50.0
        return result
    if transform_id != GROUPWISE_PHYSICAL_V1["transform_id"]:
        transform_spec(transform_id)  # raise a useful unsupported-ID error
    actor = result[..., 259:315].reshape(*result.shape[:-1], 8, 7)
    actor /= np.asarray(GROUPWISE_PHYSICAL_V1["traffic_actors"]["divisors"], dtype=np.float32)
    waypoints = result[..., 325:385].reshape(*result.shape[:-1], 20, 3)
    waypoints /= np.asarray(GROUPWISE_PHYSICAL_V1["task_waypoints"]["divisors"], dtype=np.float32)
    return result


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
        return apply_observation_transform(observation, "waypoint_xy_div50m_v1")


class GroupwisePhysicalObservation(gym.ObservationWrapper):
    """Fixed unit scaling of physical traffic and waypoint fields only."""

    transform_id = GROUPWISE_PHYSICAL_V1["transform_id"]

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        low = np.array(env.observation_space.low, dtype=np.float32, copy=True)
        high = np.array(env.observation_space.high, dtype=np.float32, copy=True)
        actor = np.arange(259, 315).reshape(8, 7)
        actor_divisors = np.asarray(GROUPWISE_PHYSICAL_V1["traffic_actors"]["divisors"], dtype=np.float32)
        waypoint = np.arange(325, 385).reshape(20, 3)
        waypoint_divisors = np.asarray(GROUPWISE_PHYSICAL_V1["task_waypoints"]["divisors"], dtype=np.float32)
        low[actor] /= actor_divisors
        high[actor] /= actor_divisors
        low[waypoint] /= waypoint_divisors
        high[waypoint] /= waypoint_divisors
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return apply_observation_transform(observation, self.transform_id)
