"""
Canonical Actuator Contract and Action Adapters for Research Platform V1.
Gate 6 of Research Platform V1.

Physical actuator contract: continuous Box(-1.0, 1.0, shape=(2,))
Representation: [steering, throttle_brake]
- steering: -1.0 (full left) to +1.0 (full right)
- throttle_brake: -1.0 (full brake) to +1.0 (full throttle)
Control cycle: nominal 10 Hz (physics dt=0.02s, decision_repeat=5).
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class InvalidActionError(ValueError):
    """Raised when an agent produces an invalid, non-finite, out-of-bounds, or unmappable action."""
    pass


@dataclass(frozen=True)
class CanonicalActionV1:
    """
    Validated canonical physical action passed to MetaDrive simulator.
    Strictly 2-dimensional continuous float32 values in [-1.0, 1.0].
    """
    steering: float
    throttle_brake: float

    def __post_init__(self):
        # Validate finite values
        if not math.isfinite(self.steering):
            raise InvalidActionError(f"Steering must be a finite float, got {self.steering}")
        if not math.isfinite(self.throttle_brake):
            raise InvalidActionError(f"Throttle/brake must be a finite float, got {self.throttle_brake}")

        # Validate bounds [-1.0, 1.0] without silent clipping
        if not (-1.0 <= self.steering <= 1.0):
            raise InvalidActionError(
                f"Steering value {self.steering} violates canonical bounds [-1.0, 1.0]. "
                "Silent clipping is strictly forbidden in benchmark mode."
            )
        if not (-1.0 <= self.throttle_brake <= 1.0):
            raise InvalidActionError(
                f"Throttle/brake value {self.throttle_brake} violates canonical bounds [-1.0, 1.0]. "
                "Silent clipping is strictly forbidden in benchmark mode."
            )

    def to_numpy(self) -> np.ndarray:
        """Returns read-only float32 array for MetaDrive env.step()."""
        arr = np.array([self.steering, self.throttle_brake], dtype=np.float32)
        arr.flags.writeable = False
        return arr

    def to_dict(self) -> Dict[str, float]:
        return {"steering": float(self.steering), "throttle_brake": float(self.throttle_brake)}


class ActionAdapter(ABC):
    """Abstract interface for deterministic action space adapters."""
    @property
    @abstractmethod
    def adapter_id(self) -> str:
        """Globally unique adapter identifier."""
        pass

    @property
    @abstractmethod
    def action_space_description(self) -> Dict[str, Any]:
        """Machine-readable description of the agent-facing action space."""
        pass

    @abstractmethod
    def to_canonical(self, agent_action: Any) -> CanonicalActionV1:
        """Converts raw agent action to CanonicalActionV1 or raises InvalidActionError."""
        pass


class ContinuousBox2Adapter(ActionAdapter):
    """
    Certified continuous identity adapter.
    Accepts 2-dimensional array-like continuous actions [steering, throttle_brake].
    """
    @property
    def adapter_id(self) -> str:
        return "continuous_box2_v1"

    @property
    def action_space_description(self) -> Dict[str, Any]:
        return {
            "type": "Box",
            "shape": [2],
            "low": [-1.0, -1.0],
            "high": [1.0, 1.0],
            "dtype": "float32",
            "semantics": ["steering (-1=left, +1=right)", "throttle_brake (-1=brake, +1=throttle)"]
        }

    def to_canonical(self, agent_action: Any) -> CanonicalActionV1:
        if agent_action is None:
            raise InvalidActionError("Agent action is None.")

        # Convert to flat sequence
        if isinstance(agent_action, (list, tuple)):
            seq = agent_action
        elif isinstance(agent_action, np.ndarray):
            if agent_action.shape != (2,):
                raise InvalidActionError(f"Expected action shape (2,), got shape {agent_action.shape}")
            seq = agent_action.tolist()
        else:
            raise InvalidActionError(f"Unsupported action type: {type(agent_action)}")

        if len(seq) != 2:
            raise InvalidActionError(f"Expected action dimension 2, got length {len(seq)}")

        try:
            st = float(seq[0])
            tb = float(seq[1])
        except (ValueError, TypeError) as e:
            raise InvalidActionError(f"Could not convert action components to float: {e}")

        return CanonicalActionV1(steering=st, throttle_brake=tb)


class Discrete25Adapter(ActionAdapter):
    """
    Certified native MetaDrive 25-discrete-action adapter.
    Directly reflects MetaDrive EnvInputPolicy with 5 steering dims and 5 throttle dims:
      steering = (index % 5) * 0.5 - 1.0  => [-1.0, -0.5, 0.0, 0.5, 1.0]
      throttle = (index // 5) * 0.5 - 1.0 => [-1.0, -0.5, 0.0, 0.5, 1.0]
    """
    STEERING_VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0]
    THROTTLE_VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0]

    @property
    def adapter_id(self) -> str:
        return "discrete25_native_v1"

    @property
    def action_space_description(self) -> Dict[str, Any]:
        return {
            "type": "Discrete",
            "n": 25,
            "steering_dim": 5,
            "throttle_dim": 5,
            "steering_grid": self.STEERING_VALUES,
            "throttle_grid": self.THROTTLE_VALUES,
            "mapping_rule": "steering = (action % 5) * 0.5 - 1.0; throttle = (action // 5) * 0.5 - 1.0"
        }

    def to_canonical(self, agent_action: Any) -> CanonicalActionV1:
        if agent_action is None:
            raise InvalidActionError("Agent discrete action is None.")

        # Accept integer or 0D numpy integer
        if isinstance(agent_action, np.ndarray) and agent_action.ndim == 0:
            agent_action = agent_action.item()

        if not isinstance(agent_action, (int, np.integer)) or isinstance(agent_action, bool):
            raise InvalidActionError(f"Expected integer discrete index in 0..24, got {type(agent_action)} ({agent_action})")

        idx = int(agent_action)
        if not (0 <= idx < 25):
            raise InvalidActionError(f"Discrete action index {idx} out of valid range [0, 24]")

        steer_idx = idx % 5
        throttle_idx = idx // 5

        steering = self.STEERING_VALUES[steer_idx]
        throttle = self.THROTTLE_VALUES[throttle_idx]

        return CanonicalActionV1(steering=steering, throttle_brake=throttle)


class Discrete9Adapter(ActionAdapter):
    """
    Certified low-branching 9-discrete-action adapter for planning/MCTS/search algorithms.
    Combines 3 steering levels [-0.6, 0.0, 0.6] and 3 throttle levels [-0.8, 0.0, 0.6].
    Index = 3 * throttle_idx + steer_idx:
      0: [-0.6, -0.8]  (Left + Brake)
      1: [ 0.0, -0.8]  (Straight + Brake)
      2: [ 0.6, -0.8]  (Right + Brake)
      3: [-0.6,  0.0]  (Left + Coast)
      4: [ 0.0,  0.0]  (Straight + Coast)
      5: [ 0.6,  0.0]  (Right + Coast)
      6: [-0.6,  0.6]  (Left + Throttle)
      7: [ 0.0,  0.6]  (Straight + Throttle)
      8: [ 0.6,  0.6]  (Right + Throttle)
    """
    STEERING_LEVELS = [-0.6, 0.0, 0.6]
    THROTTLE_LEVELS = [-0.8, 0.0, 0.6]

    GRID = [
        (-0.6, -0.8),  # 0
        ( 0.0, -0.8),  # 1
        ( 0.6, -0.8),  # 2
        (-0.6,  0.0),  # 3
        ( 0.0,  0.0),  # 4
        ( 0.6,  0.0),  # 5
        (-0.6,  0.6),  # 6
        ( 0.0,  0.6),  # 7
        ( 0.6,  0.6),  # 8
    ]

    @property
    def adapter_id(self) -> str:
        return "discrete9_lowbranch_v1"

    @property
    def action_space_description(self) -> Dict[str, Any]:
        return {
            "type": "Discrete",
            "n": 9,
            "steering_levels": self.STEERING_LEVELS,
            "throttle_levels": self.THROTTLE_LEVELS,
            "actions": [list(g) for g in self.GRID],
            "intended_usage": "Low-branching search, MCTS, and discrete planning"
        }

    def to_canonical(self, agent_action: Any) -> CanonicalActionV1:
        if agent_action is None:
            raise InvalidActionError("Agent discrete action is None.")

        if isinstance(agent_action, np.ndarray) and agent_action.ndim == 0:
            agent_action = agent_action.item()

        if not isinstance(agent_action, (int, np.integer)) or isinstance(agent_action, bool):
            raise InvalidActionError(f"Expected integer discrete index in 0..8, got {type(agent_action)} ({agent_action})")

        idx = int(agent_action)
        if not (0 <= idx < 9):
            raise InvalidActionError(f"Discrete action index {idx} out of valid range [0, 8]")

        st, tb = self.GRID[idx]
        return CanonicalActionV1(steering=st, throttle_brake=tb)


CERTIFIED_ACTION_ADAPTERS: Dict[str, ActionAdapter] = {
    "continuous_box2_v1": ContinuousBox2Adapter(),
    "discrete25_native_v1": Discrete25Adapter(),
    "discrete9_lowbranch_v1": Discrete9Adapter(),
}


def get_action_adapter(adapter_id: str) -> ActionAdapter:
    """Returns certified action adapter by identifier."""
    if adapter_id not in CERTIFIED_ACTION_ADAPTERS:
        raise ValueError(
            f"Unknown action adapter '{adapter_id}'. Certified adapters: {list(CERTIFIED_ACTION_ADAPTERS.keys())}"
        )
    return CERTIFIED_ACTION_ADAPTERS[adapter_id]
