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


def validate_diagnostics_payload(data: Any, max_depth: int = 4, current_depth: int = 0) -> None:
    """Recursively validates that diagnostics contain only JSON-safe bounded primitives."""
    if current_depth > max_depth:
        raise ValueError(f"Diagnostics payload exceeds maximum nesting depth {max_depth}")

    if data is None or isinstance(data, (bool, str, int)):
        return
    if isinstance(data, (float, np.floating)):
        if not math.isfinite(data):
            raise ValueError(f"Diagnostics contains non-finite float: {data}")
        return
    if isinstance(data, (list, tuple)):
        if len(data) > 100:
            raise ValueError(f"Diagnostics sequence exceeds maximum item limit 100: {len(data)}")
        for item in data:
            validate_diagnostics_payload(item, max_depth, current_depth + 1)
        return
    if isinstance(data, dict):
        if len(data) > 50:
            raise ValueError(f"Diagnostics dictionary exceeds maximum key limit 50: {len(data)}")
        for k, v in data.items():
            if not isinstance(k, str):
                raise TypeError(f"Diagnostics dictionary keys must be strings, got {type(k)}")
            validate_diagnostics_payload(v, max_depth, current_depth + 1)
        return

    raise TypeError(
        f"Illegal object in AgentDecision diagnostics: {type(data)} ({data}). "
        "Only JSON-safe bounded primitives (str, int, float, bool, None, list, dict) are permitted."
    )


@dataclass(frozen=True)
class AgentDecision:
    """
    Immutable container returned by an agent policy at each decision cycle.
    Diagnostics are logging/telemetry only and must be JSON-safe bounded primitives.
    """
    action_payload: Any
    diagnostics: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.diagnostics is not None:
            if not isinstance(self.diagnostics, dict):
                raise TypeError(f"diagnostics must be a dict or None, got {type(self.diagnostics)}")
            validate_diagnostics_payload(self.diagnostics)

    def to_dict(self) -> Dict[str, Any]:
        diag = copy.deepcopy(self.diagnostics) if self.diagnostics else None
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


def build_agent_contract_core(
    pinned_commit: str = "85e5dadc6c7436d324348f6e3d8f8e680c06b4db",
    pinned_version: str = "0.4.3",
    status: str = "LOCKED-FOR-PLATFORM-V1",
    custom_rules: Optional[Dict[str, Any]] = None,
    custom_adapter_grids: Optional[Dict[str, Any]] = None,
    custom_latency_boundary: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Builds the authoritative, complete machine-readable AgentContractV1 core dictionary.
    Locks public AgentInput schema, actuator contract, certified adapter mappings,
    evaluation rules, and latency measurement boundary for canonical hashing.
    """
    d25_adapter = Discrete25Adapter()
    d25_mappings = []
    for idx in range(25):
        act = d25_adapter.to_canonical(idx)
        d25_mappings.append({
            "action_index": idx,
            "steering": round(act.steering, 4),
            "throttle_brake": round(act.throttle_brake, 4)
        })

    d9_adapter = Discrete9Adapter()
    d9_mappings = []
    for idx in range(9):
        act = d9_adapter.to_canonical(idx)
        d9_mappings.append({
            "action_index": idx,
            "steering": round(act.steering, 4),
            "throttle_brake": round(act.throttle_brake, 4)
        })

    if custom_adapter_grids:
        if "discrete25" in custom_adapter_grids:
            d25_mappings = custom_adapter_grids["discrete25"]
        if "discrete9" in custom_adapter_grids:
            d9_mappings = custom_adapter_grids["discrete9"]

    default_rules = {
        "no_online_learning_during_evaluation": (
            "AgentPolicy.act() receives zero rewards, returns, or evaluation outcomes during INFERENCE mode. "
            "Cross-episode online adaptation is strictly prohibited."
        ),
        "stateful_policy_reset": (
            "reset() must clear episodic recurrent hidden states while preserving static learned parameters."
        ),
        "information_parity_principle": (
            "Every benchmark-compliant agent receives the exact same AgentInputV1 under a declared profile. "
            "No hidden platform privileges based on algorithm family or stage label."
        )
    }
    rules = dict(default_rules)
    if custom_rules:
        rules.update(custom_rules)

    default_latency = {
        "start_event": "Immediately before agent.act(agent_input) is invoked",
        "stop_event": "Immediately after AgentDecision is returned",
        "excluded_components": [
            "AgentInput assembly",
            "ActionAdapter conversion",
            "environment step (physics)",
            "rendering",
            "logging"
        ],
        "nominal_realtime_budget_ms": 100.0,
        "enforcement_policy": "Diagnostic measurement boundary; hard real-time cutoff is not enforced in platform simulation."
    }
    latency_boundary = dict(default_latency)
    if custom_latency_boundary:
        latency_boundary.update(custom_latency_boundary)

    return {
        "metadata": {
            "contract_name": "AgentContractV1",
            "spec_version": "1.0.0",
            "status": status,
            "gate": "Gate 6 (Research Platform V1)",
            "pinned_metadrive_commit": pinned_commit,
            "pinned_metadrive_version": pinned_version,
        },
        "information_profiles": {
            "main_profile_id": "STATE_DECISION_V1",
            "certified_profiles": ["STATE_DECISION_V1", "CORE_ONLY_V1"],
            "parity_principle": rules["information_parity_principle"]
        },
        "agent_input_schema": {
            "core_observation": {
                "dimensions": 259,
                "dtype": "float32",
                "normalized_range": [0.0, 1.0],
                "bounds_tolerance": 1e-4,
                "subvectors": {
                    "ego_state": [0, 9],
                    "navigation_checkpoints": [9, 19],
                    "lidar_rays": [19, 259]
                },
                "immutability": "read-only defensive array copy (writeable=False)"
            },
            "traffic_context": {
                "capacity": 8,
                "radius_m": 50.0,
                "actor_features_dim": 7,
                "actor_features": [
                    "relative_position_x",
                    "relative_position_y",
                    "relative_velocity_x",
                    "relative_velocity_y",
                    "relative_heading",
                    "length",
                    "width"
                ],
                "coordinate_frame": "ego-centric (forward +x, left +y)",
                "ordering": "Euclidean distance ascending with deterministic tie-breaking (distance, x_rel, y_rel)"
            },
            "task_context": {
                "lookahead_count": 20,
                "lookahead_spacing_m": 2.5,
                "lookahead_range_m": 50.0,
                "current_lane_width_m": 3.5,
                "navigation_goal_direction": "2D normalized direction unit vector",
                "route_end_within_lookahead": "Boolean flag indicating whether reference route terminates within lookahead",
                "coordinate_frame": "ego-centric relative waypoints (lookahead_distance_m, relative_x, relative_y, relative_heading)",
                "evaluator_progress_metrics_excluded": True
            }
        },
        "forbidden_evaluator_fields": sorted(list(FORBIDDEN_EVALUATOR_FIELDS)),
        "public_episode_context": {
            "allowed_fields": [
                "control_frequency_hz",
                "control_dt_s",
                "horizon_steps",
                "input_profile_id",
                "action_adapter_id",
                "mode (INFERENCE | TRAINING)"
            ],
            "excluded_fields": [
                "tier", "split", "case_id", "environment_seed", "geometry_sha256"
            ]
        },
        "agent_policy_lifecycle": {
            "interface_type": "AgentPolicy (Python Protocol)",
            "methods": [
                "descriptor -> AgentDescriptor",
                "reset(public_context, agent_seed) -> None",
                "act(agent_input) -> AgentDecision",
                "close() -> None"
            ]
        },
        "agent_decision_schema": {
            "action_payload": "Raw action to be consumed by declared action adapter",
            "diagnostics": "Optional JSON-safe bounded primitive dictionary (logging only, max depth 4, max 50 keys)"
        },
        "technical_failure_reasons": [e.value for e in TechnicalFailureReason],
        "actuator_contract": {
            "canonical_physical_action": "Continuous Box(-1.0, 1.0, shape=(2,)) [steering, throttle_brake]",
            "control_frequency_hz": 10,
            "nominal_decision_dt_s": 0.10,
            "invalid_action_policy": "Loud rejection via InvalidActionError (Technical Failure); silent clipping strictly forbidden",
            "certified_adapters": {
                "continuous_box2_v1": {
                    "type": "Box(2)",
                    "range": [-1.0, 1.0],
                    "identity": True
                },
                "discrete25_native_v1": {
                    "type": "Discrete(25)",
                    "steering_grid": Discrete25Adapter.STEERING_VALUES,
                    "throttle_grid": Discrete25Adapter.THROTTLE_VALUES,
                    "mappings": d25_mappings
                },
                "discrete9_lowbranch_v1": {
                    "type": "Discrete(9)",
                    "steering_grid": Discrete9Adapter.STEERING_LEVELS,
                    "throttle_grid": Discrete9Adapter.THROTTLE_LEVELS,
                    "mappings": d9_mappings
                }
            }
        },
        "evaluation_rules": rules,
        "latency_measurement_boundary": latency_boundary
    }
