"""
Authoritative Observation Encoder for Stage 5 Reinforcement Learning (Gate S5.1).

Transforms public AgentInputV1 (STATE_DECISION_V1) into a deterministic,
fixed-size 1D np.float32 vector for policy networks (SAC / PPO).

Strict Information Parity & Sandboxing:
- Consumes strictly public AgentInputV1 fields.
- Excludes all 34 evaluator-private fields (route_completion, clean_success,
  reward, episode return, future outcome, environment_seed, case ID, etc.).
- Shared bit-for-bit between RL training environment and Launcher inference.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.platform.agent_input import (
    AgentInputV1,
    CoreObservationV1,
    InputProfileId,
    TaskContextV1,
    TrafficContextV1,
)

# ------------------------------------------------------------------------------
# Authoritative Dimensions according to Platform V1 Contract
# ------------------------------------------------------------------------------
# 1. Core Observation: 259D in [0.0, 1.0]
CORE_OBSERVATION_DIM = 259

# 2. Traffic Context: 8 actors x 7 features = 56D
#    + 8D validity mask + 1D active count + 1D overflow count = 66D
TRAFFIC_ACTORS_DIM = 8 * 7  # 56
TRAFFIC_MASK_DIM = 8
TRAFFIC_METRICS_DIM = 2     # active_count_ratio, overflow_count_ratio
TRAFFIC_TOTAL_DIM = TRAFFIC_ACTORS_DIM + TRAFFIC_MASK_DIM + TRAFFIC_METRICS_DIM  # 66

# 3. Task Context: 20 waypoints x 3 features = 60D
#    + 20D validity mask + 1D lane width + 2D goal direction + 1D route end flag = 84D
TASK_WAYPOINTS_DIM = 20 * 3  # 60
TASK_MASK_DIM = 20
TASK_METRICS_DIM = 4         # lane_width_ratio, goal_dir_x, goal_dir_y, route_end_flag
TASK_TOTAL_DIM = TASK_WAYPOINTS_DIM + TASK_MASK_DIM + TASK_METRICS_DIM  # 84

# Full structured flat representation: 259 + 66 + 84 = 409D
STAGE5_FULL_OBSERVATION_DIM = CORE_OBSERVATION_DIM + TRAFFIC_TOTAL_DIM + TASK_TOTAL_DIM
STAGE5_CORE_ONLY_OBSERVATION_DIM = CORE_OBSERVATION_DIM

STAGE5_ENCODER_ID = "stage5_flat409_v1"
STAGE5_ENCODER_VERSION = "1.0.0"


class Stage5ObservationEncoder:
    """
    Stateless observation encoder transforming AgentInputV1 into a flat float32 vector.
    Compatible with Gymnasium spaces.Box for standard RL algorithms (SAC, PPO).
    """

    def __init__(self, core_only: bool = False) -> None:
        """
        Initializes the encoder.

        Args:
            core_only: If True, encodes strictly the 259D core observation.
                       If False (default), encodes the full 409D structured representation.
        """
        self._core_only = bool(core_only)
        self._dimension = STAGE5_CORE_ONLY_OBSERVATION_DIM if self._core_only else STAGE5_FULL_OBSERVATION_DIM

        # Define Gymnasium observation space based strictly on Platform V1 schema:
        # - Core features are normalized in [0.0, 1.0].
        # - Traffic actor features and task waypoints are continuous physical coordinates
        #   in (-inf, inf), avoiding arbitrary clipping or rejection of distant entities.
        # - Validity masks and ratios are strictly in [0.0, 1.0].
        # - Direction vector is in [-1.0, 1.0].
        low = np.full(self._dimension, -np.inf, dtype=np.float32)
        high = np.full(self._dimension, np.inf, dtype=np.float32)

        if self._core_only:
            low[0:CORE_OBSERVATION_DIM] = 0.0
            high[0:CORE_OBSERVATION_DIM] = 1.0
        else:
            # 1. Core features [0:259]: strictly [0.0, 1.0]
            low[0:CORE_OBSERVATION_DIM] = 0.0
            high[0:CORE_OBSERVATION_DIM] = 1.0

            # 2. Traffic actors [259:315]: continuous physical features in (-inf, inf)
            # 3. Traffic validity mask [315:323]: strictly [0.0, 1.0]
            low[315:323] = 0.0
            high[315:323] = 1.0

            # 4. Traffic metrics [323:325]: active_count_ratio and overflow_ratio in [0.0, 1.0]
            low[323:325] = 0.0
            high[323:325] = 1.0

            # 5. Task waypoints [325:385]: continuous coordinates in (-inf, inf)
            # 6. Task validity mask [385:405]: strictly [0.0, 1.0]
            low[385:405] = 0.0
            high[385:405] = 1.0

            # 7. Task metrics [405:409]:
            # lane_width_ratio >= 0.0
            low[405] = 0.0
            high[405] = np.inf
            # 2D goal direction unit vector in [-1.0, 1.0]
            low[406:408] = -1.0
            high[406:408] = 1.0
            # route_end_within_lookahead flag in [0.0, 1.0]
            low[408] = 0.0
            high[408] = 1.0

        self._observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(self._dimension,),
            dtype=np.float32,
        )

    @property
    def encoder_id(self) -> str:
        """Identifier of this encoder architecture."""
        return "stage5_core259_v1" if self._core_only else STAGE5_ENCODER_ID

    @property
    def encoder_version(self) -> str:
        """Semantic version of this encoder."""
        return STAGE5_ENCODER_VERSION

    @property
    def dimension(self) -> int:
        """Exact 1D vector dimension."""
        return self._dimension

    @property
    def observation_space(self) -> spaces.Box:
        """Gymnasium Box observation space."""
        return self._observation_space

    @property
    def is_core_only(self) -> bool:
        """Whether this encoder is in core-only ablation mode."""
        return self._core_only

    def encode(self, agent_input: AgentInputV1) -> np.ndarray:
        """
        Encodes an authoritative AgentInputV1 instance into a 1D float32 array.

        Args:
            agent_input: The decision-cycle input provided by Platform V1.

        Returns:
            np.ndarray of shape (dimension,) and dtype float32.
        """
        if not isinstance(agent_input, AgentInputV1):
            raise TypeError(f"Expected AgentInputV1 instance, got {type(agent_input).__name__}")

        # 1. Core Observation (259D)
        core_features = agent_input.core_observation.features
        if core_features.shape != (CORE_OBSERVATION_DIM,):
            raise ValueError(f"Core features shape mismatch: expected ({CORE_OBSERVATION_DIM},), got {core_features.shape}")

        if self._core_only:
            out = np.array(core_features, dtype=np.float32, copy=True)
            return out

        # 2. Traffic Context (66D)
        traffic_ctx = agent_input.traffic_context
        # actors_array: (8, 7) -> (56,)
        actors_flat = traffic_ctx.actors_array.reshape(-1)
        # validity_mask: (8,) bool -> float32 (1.0 or 0.0)
        traffic_mask = traffic_ctx.validity_mask.astype(np.float32)
        # active count ratio in [0.0, 1.0]
        active_ratio = float(traffic_ctx.active_count) / float(max(1, traffic_ctx.capacity))
        # overflow count normalized (capped at 10 for normalization)
        overflow_ratio = float(min(traffic_ctx.overflow_count, 10)) / 10.0
        traffic_scalars = np.array([active_ratio, overflow_ratio], dtype=np.float32)

        # 3. Task Context (84D)
        task_ctx = agent_input.task_context
        # waypoints_array: (20, 3) -> (60,)
        wps_flat = task_ctx.waypoints_array.reshape(-1)
        # validity_mask: (20,) bool -> float32 (1.0 or 0.0)
        task_mask = task_ctx.validity_mask.astype(np.float32)
        # current lane width normalized by standard 5.0m
        lane_width_ratio = float(task_ctx.current_lane_width) / 5.0
        # navigation goal direction: 2D unit vector
        goal_dir = np.array(task_ctx.navigation_goal_direction, dtype=np.float32)
        # route end within lookahead flag: 1.0 or 0.0
        route_end_flag = 1.0 if task_ctx.route_end_within_lookahead else 0.0
        task_scalars = np.array([lane_width_ratio, goal_dir[0], goal_dir[1], route_end_flag], dtype=np.float32)

        # 4. Concatenate into a single flat vector
        encoded = np.concatenate([
            core_features,      # 259
            actors_flat,        # 56
            traffic_mask,       # 8
            traffic_scalars,    # 2
            wps_flat,           # 60
            task_mask,          # 20
            task_scalars,       # 4
        ]).astype(np.float32)

        assert encoded.shape == (self._dimension,), f"Encoded shape {encoded.shape} != ({self._dimension},)"
        assert np.all(np.isfinite(encoded)), "Encoded observation contains non-finite values."

        return encoded

    def get_feature_manifest(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns an exact, interpretable feature slice manifest for policy architectures.
        """
        if self._core_only:
            return {
                "core_observation": {"slice": [0, 259], "dim": 259, "description": "Core 259D observation features"}
            }

        return {
            "core_observation": {
                "slice": [0, 259],
                "dim": 259,
                "description": "Core 259D observation (ego state, checkpoints, 240D lidar)",
            },
            "traffic_actors": {
                "slice": [259, 315],
                "dim": 56,
                "description": "Flattened 8 x 7 surrounding traffic actor states (rel_x, rel_y, vx, vy, yaw, length, width)",
            },
            "traffic_mask": {
                "slice": [315, 323],
                "dim": 8,
                "description": "Traffic actor validity mask (1.0 = valid actor, 0.0 = padded)",
            },
            "traffic_metrics": {
                "slice": [323, 325],
                "dim": 2,
                "description": "Normalized active actor count and overflow count",
            },
            "task_waypoints": {
                "slice": [325, 385],
                "dim": 60,
                "description": "Flattened 20 x 3 route lookahead waypoints (rel_x, rel_y, rel_heading)",
            },
            "task_mask": {
                "slice": [385, 405],
                "dim": 20,
                "description": "Waypoint validity mask (1.0 = valid waypoint, 0.0 = padded past route end)",
            },
            "task_metrics": {
                "slice": [405, 409],
                "dim": 4,
                "description": "Lane width ratio, 2D goal direction unit vector, and route end flag",
            },
        }

    def get_schema_fingerprint(self) -> str:
        """Returns the cryptographic SHA-256 digest of the feature manifest schema."""
        manifest = self.get_feature_manifest()
        canonical_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


# Module-level default encoder singleton for convenience
_DEFAULT_ENCODER = Stage5ObservationEncoder(core_only=False)


def encode_agent_input(agent_input: AgentInputV1) -> np.ndarray:
    """Pure functional helper encoding an AgentInputV1 to the default 409D vector."""
    return _DEFAULT_ENCODER.encode(agent_input)
