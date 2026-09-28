"""
Platform V1 Episode Specification and Route-Aware Horizon Policies.
Gate 3 of Research Platform V1.

Pure, unit-testable module independent of Panda3D / simulator engine state.
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def compute_route_aware_horizon(
    route_length_m: float,
    reference_floor_speed_kmh: float = 18.0,
    safety_margin: float = 1.5,
    control_frequency_hz: int = 10,
    min_horizon_steps: int = 1000,
    max_horizon_steps: int = 4000,
) -> int:
    """
    Computes a deterministic, route-length-aware episode horizon budget.

    Formula:
        floor_speed_mps = reference_floor_speed_kmh / 3.6
        nominal_steps = ceil((route_length_m / floor_speed_mps) * safety_margin * control_frequency_hz)
        horizon = max(min_horizon_steps, min(max_horizon_steps, nominal_steps))

    Design Rules:
        1. Fixed prior to episode reset based solely on scenario metadata.
        2. Never adapts based on agent identity, algorithm Stage, or current episode progress.
        3. Treats horizon as an emergency safety cutoff, not an aggressive speed target.
    """
    if route_length_m <= 0:
        return min_horizon_steps

    floor_speed_mps = max(reference_floor_speed_kmh / 3.6, 0.1)
    nominal_seconds = route_length_m / floor_speed_mps
    budget_seconds = nominal_seconds * safety_margin
    steps = math.ceil(budget_seconds * control_frequency_hz)

    clamped_steps = max(min_horizon_steps, min(max_horizon_steps, steps))
    return int(clamped_steps)


@dataclass(frozen=True)
class EpisodeSpecV1:
    """
    Versioned contract describing platform episode lifecycle policy.
    Does not instantiate or own simulator resources.
    """
    spec_version: str = "1.0.0"
    control_frequency_hz: int = 10
    truncate_as_terminate: bool = False
    traffic_mode: str = "trigger"
    horizon_policy_type: str = "route_aware"
    reference_floor_speed_kmh: float = 18.0
    safety_margin: float = 1.5
    min_horizon_steps: int = 1000
    max_horizon_steps: int = 4000
    terminal_failure_flags: List[str] = field(default_factory=lambda: [
        "crash_human",
        "crash_vehicle",
        "crash_object",
        "crash_building",
        "crash_sidewalk",
        "out_of_road",
    ])
    outcome_precedence: List[str] = field(default_factory=lambda: [
        "CRASH_HUMAN",
        "CRASH_VEHICLE",
        "CRASH_OBJECT",
        "CRASH_BUILDING",
        "CRASH_SIDEWALK",
        "OUT_OF_ROAD",
        "SUCCESS",
        "TIMEOUT",
    ])

    def compute_horizon(self, route_length_m: float) -> int:
        """Compute horizon for a given scenario route length."""
        if self.horizon_policy_type == "fixed":
            return self.min_horizon_steps
        return compute_route_aware_horizon(
            route_length_m=route_length_m,
            reference_floor_speed_kmh=self.reference_floor_speed_kmh,
            safety_margin=self.safety_margin,
            control_frequency_hz=self.control_frequency_hz,
            min_horizon_steps=self.min_horizon_steps,
            max_horizon_steps=self.max_horizon_steps,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert specification to a plain serializable dictionary."""
        return asdict(self)
