"""
Agent Policy Lifecycle, Descriptors, and Audit Fixtures for Research Platform V1.
Gate 6 of Research Platform V1.

Defines the lightweight, composition-friendly runtime boundary between
Research Platform V1 and all agent families (Stages 0 to 7).
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np

from src.platform.action_adapter import (
    ActionAdapter,
    CanonicalActionV1,
    ContinuousBox2Adapter,
    Discrete9Adapter,
    Discrete25Adapter,
    InvalidActionError,
    get_action_adapter,
)
from src.platform.agent_input import (
    AgentInputV1,
    FORBIDDEN_EVALUATOR_FIELDS,
    InputProfileId,
)


class TechnicalFailureReason(str, Enum):
    """Standardized technical error categories distinguished from task-level driving failures."""
    AGENT_EXCEPTION = "AGENT_EXCEPTION"
    INVALID_AGENT_ACTION = "INVALID_AGENT_ACTION"
    ACTION_ADAPTER_ERROR = "ACTION_ADAPTER_ERROR"
    INVALID_AGENT_INPUT_CONSUMPTION = "INVALID_AGENT_INPUT_CONSUMPTION"


@dataclass(frozen=True)
class AgentDescriptor:
    """
    Machine-readable metadata declaring an agent's identity and operational capabilities.
    Belongs to the test harness; NEVER alters the environment information profile.
    """
    agent_id: str
    agent_version: str = "1.0.0"
    input_profile_id: str = "STATE_DECISION_V1"
    action_adapter_id: str = "continuous_box2_v1"
    inference_stochasticity: str = "deterministic"  # "deterministic" or "stochastic"
    stateful_within_episode: bool = False
    method_family: Optional[str] = None

    def __post_init__(self):
        if self.inference_stochasticity not in ("deterministic", "stochastic"):
            raise ValueError(f"inference_stochasticity must be 'deterministic' or 'stochastic', got '{self.inference_stochasticity}'")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentPublicEpisodeContext:
    """
    Minimal public episode context provided to agent upon reset().
    Strictly sanitised to eliminate test-case metadata and prevent benchmark gaming.
    """
    control_frequency_hz: int = 10
    control_dt_s: float = 0.1
    horizon_steps: int = 1000
    input_profile_id: str = "STATE_DECISION_V1"
    action_adapter_id: str = "continuous_box2_v1"
    mode: str = "INFERENCE"  # "INFERENCE" (frozen evaluation) or "TRAINING"

    def __post_init__(self):
        if self.mode not in ("INFERENCE", "TRAINING"):
            raise ValueError(f"mode must be 'INFERENCE' or 'TRAINING', got '{self.mode}'")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentDecision:
    """
    Immutable container returned by an agent policy at each decision cycle.
    """
    action_payload: Any
    diagnostics: Optional[Dict[str, Any]] = None  # Logging/telemetry only; never alters simulation

    def to_dict(self) -> Dict[str, Any]:
        diag = dict(self.diagnostics) if self.diagnostics else None
        return {"action_payload": self.action_payload, "diagnostics": diag}


@runtime_checkable
class AgentPolicy(Protocol):
    """
    Protocol defining the runtime policy contract for all agent families.
    Composition-friendly; does not require heavy inheritance.
    """
    @property
    def descriptor(self) -> AgentDescriptor:
        """Machine-readable metadata declaring identity and capabilities."""
        ...

    def reset(self, public_context: AgentPublicEpisodeContext, agent_seed: Optional[int] = None) -> None:
        """
        Resets episodic internal state before a new episode begins.
        Must NOT wipe learned model weights or static configuration.
        """
        ...

    def act(self, agent_input: AgentInputV1) -> AgentDecision:
        """
        Produces a decision for the current 10 Hz decision cycle.
        During evaluation (mode=INFERENCE), policy must remain strictly frozen.
        """
        ...

    def close(self) -> None:
        """Releases any local resources or worker processes."""
        ...


# ==============================================================================
# Reference Audit Fixtures (Gate-6 Verification Only; NOT Stage implementations)
# ==============================================================================

class DeterministicConstantFixtureAgent:
    """Audit fixture: returns constant continuous action [0.0, 0.4]."""
    def __init__(self, action: Optional[List[float]] = None):
        self._action = list(action) if action is not None else [0.0, 0.4]
        self._descriptor = AgentDescriptor(
            agent_id="fixture_constant_continuous",
            agent_version="1.0.0",
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic",
            stateful_within_episode=False,
            method_family="fixture"
        )

    @property
    def descriptor(self) -> AgentDescriptor:
        return self._descriptor

    def reset(self, public_context: AgentPublicEpisodeContext, agent_seed: Optional[int] = None) -> None:
        pass

    def act(self, agent_input: AgentInputV1) -> AgentDecision:
        return AgentDecision(action_payload=list(self._action))

    def close(self) -> None:
        pass


class SeededRandomFixtureAgent:
    """Audit fixture: stochastic policy seeded strictly via agent_seed."""
    def __init__(self):
        self._rng: Optional[np.random.RandomState] = None
        self._descriptor = AgentDescriptor(
            agent_id="fixture_seeded_random",
            agent_version="1.0.0",
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="continuous_box2_v1",
            inference_stochasticity="stochastic",
            stateful_within_episode=False,
            method_family="fixture"
        )

    @property
    def descriptor(self) -> AgentDescriptor:
        return self._descriptor

    def reset(self, public_context: AgentPublicEpisodeContext, agent_seed: Optional[int] = None) -> None:
        seed = agent_seed if agent_seed is not None else 42
        self._rng = np.random.RandomState(seed)

    def act(self, agent_input: AgentInputV1) -> AgentDecision:
        if self._rng is None:
            self._rng = np.random.RandomState(42)
        steer = float(self._rng.uniform(-1.0, 1.0))
        throttle = float(self._rng.uniform(-1.0, 1.0))
        return AgentDecision(action_payload=[steer, throttle])

    def close(self) -> None:
        pass


class StatefulCounterFixtureAgent:
    """Audit fixture: stateful within episode; counter resets on reset()."""
    def __init__(self, initial_static_weight: float = 1.0):
        self.static_weight = initial_static_weight
        self.step_counter = 0
        self._descriptor = AgentDescriptor(
            agent_id="fixture_stateful_counter",
            agent_version="1.0.0",
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic",
            stateful_within_episode=True,
            method_family="fixture"
        )

    @property
    def descriptor(self) -> AgentDescriptor:
        return self._descriptor

    def reset(self, public_context: AgentPublicEpisodeContext, agent_seed: Optional[int] = None) -> None:
        # Reset episodic counter while preserving static_weight
        self.step_counter = 0

    def act(self, agent_input: AgentInputV1) -> AgentDecision:
        self.step_counter += 1
        # Throttle modulates with step counter
        val = 0.2 + 0.01 * min(self.step_counter, 40)
        return AgentDecision(
            action_payload=[0.0, float(val)],
            diagnostics={"step_counter": self.step_counter, "static_weight": self.static_weight}
        )

    def close(self) -> None:
        pass


class DiscreteFixtureAgent:
    """Audit fixture: produces discrete actions for Discrete25 or Discrete9 adapters."""
    def __init__(self, discrete_index: int = 12, adapter_id: str = "discrete25_native_v1"):
        self.discrete_index = discrete_index
        self._descriptor = AgentDescriptor(
            agent_id="fixture_discrete",
            agent_version="1.0.0",
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id=adapter_id,
            inference_stochasticity="deterministic",
            stateful_within_episode=False,
            method_family="fixture"
        )

    @property
    def descriptor(self) -> AgentDescriptor:
        return self._descriptor

    def reset(self, public_context: AgentPublicEpisodeContext, agent_seed: Optional[int] = None) -> None:
        pass

    def act(self, agent_input: AgentInputV1) -> AgentDecision:
        return AgentDecision(action_payload=self.discrete_index)

    def close(self) -> None:
        pass


class InvalidOutputFixtureAgent:
    """Audit fixture: intentionally produces invalid actions to verify technical failure handling."""
    def __init__(self, invalid_mode: str = "nan"):
        self.invalid_mode = invalid_mode
        self._descriptor = AgentDescriptor(
            agent_id="fixture_invalid_output",
            agent_version="1.0.0",
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic",
            stateful_within_episode=False,
            method_family="fixture"
        )

    @property
    def descriptor(self) -> AgentDescriptor:
        return self._descriptor

    def reset(self, public_context: AgentPublicEpisodeContext, agent_seed: Optional[int] = None) -> None:
        pass

    def act(self, agent_input: AgentInputV1) -> AgentDecision:
        if self.invalid_mode == "nan":
            return AgentDecision(action_payload=[float("nan"), 0.5])
        elif self.invalid_mode == "inf":
            return AgentDecision(action_payload=[float("inf"), 0.0])
        elif self.invalid_mode == "out_of_bounds":
            return AgentDecision(action_payload=[1.2, 0.0])
        elif self.invalid_mode == "wrong_shape":
            return AgentDecision(action_payload=[0.0])
        elif self.invalid_mode == "exception":
            raise RuntimeError("Deliberate agent exception for testing technical failure handling.")
        else:
            return AgentDecision(action_payload=None)

    def close(self) -> None:
        pass
