"""
Immutable Telemetry Snapshot Model for HUD Visualization (Gate HUD-2).

Provides HudTelemetrySnapshotV1, an immutable per-step telemetry representation
consumed strictly in read-only mode by the visualizer components.
Enforces complete decoupling from simulator state and scientific benchmark runs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple


def _sanitize_finite_float(val: Any) -> Optional[float]:
    """Helper to convert to float and verify finiteness; returns None if invalid."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class HudTelemetrySnapshotV1:
    """
    Immutable per-step telemetry snapshot consumed strictly by the HUD renderer.

    Attributes:
        step_index: Current simulation step (0..horizon).
        simulation_time_s: Elapsed simulation time in seconds.
        world_position: Ego vehicle (x, y) coordinates in world meters (None if unavailable).
        heading_theta_rad: Ego vehicle yaw angle in radians [-pi, pi] (None if unavailable).
        speed_kmh: Authoritative current speed in km/h (None if unavailable).
        route_completion: Current route completion ratio [0.0, 1.0] (None if unavailable).
        canonical_steering: Canonical steering command [-1.0, 1.0] (None if unavailable).
        canonical_throttle_brake: Canonical throttle/brake command [-1.0, 1.0] (None if unavailable).
        terminal_outcome_status: Episode terminal status ('UNDETERMINED' during episode, or TerminalReason).
        is_clean_success: Clean success indicator (None during episode, bool upon terminal evaluation).
    """

    step_index: int
    simulation_time_s: float
    world_position: Optional[Tuple[float, float]]
    heading_theta_rad: Optional[float]
    speed_kmh: Optional[float]
    route_completion: Optional[float]
    canonical_steering: Optional[float]
    canonical_throttle_brake: Optional[float]
    terminal_outcome_status: str = "UNDETERMINED"
    is_clean_success: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes snapshot to a pure dictionary."""
        return {
            "step_index": self.step_index,
            "simulation_time_s": self.simulation_time_s,
            "world_position": self.world_position,
            "heading_theta_rad": self.heading_theta_rad,
            "speed_kmh": self.speed_kmh,
            "route_completion": self.route_completion,
            "canonical_steering": self.canonical_steering,
            "canonical_throttle_brake": self.canonical_throttle_brake,
            "terminal_outcome_status": self.terminal_outcome_status,
            "is_clean_success": self.is_clean_success,
        }


def create_telemetry_snapshot(
    step_index: Any = 0,
    simulation_time_s: Any = 0.0,
    world_position: Any = None,
    heading_theta_rad: Any = None,
    speed_kmh: Any = None,
    route_completion: Any = None,
    canonical_steering: Any = None,
    canonical_throttle_brake: Any = None,
    terminal_outcome_status: str = "UNDETERMINED",
    is_clean_success: Optional[bool] = None,
) -> HudTelemetrySnapshotV1:
    """
    Factory function that sanitizes and validates inputs to construct a frozen HudTelemetrySnapshotV1.
    Replaces NaN, Inf, and non-numeric values with None for optional fields.
    """
    # 1. Step index
    try:
        idx = max(0, int(step_index))
    except (ValueError, TypeError):
        idx = 0

    # 2. Simulation time
    sim_t = _sanitize_finite_float(simulation_time_s)
    if sim_t is None or sim_t < 0.0:
        sim_t = 0.0

    # 3. World position
    sanitized_pos: Optional[Tuple[float, float]] = None
    if world_position is not None:
        try:
            px = _sanitize_finite_float(world_position[0])
            py = _sanitize_finite_float(world_position[1])
            if px is not None and py is not None:
                sanitized_pos = (px, py)
        except (IndexError, TypeError, KeyError):
            sanitized_pos = None

    # 4. Heading
    sanitized_heading = _sanitize_finite_float(heading_theta_rad)

    # 5. Speed: non-negative finite float; negative speeds are invalid in pinned MetaDrive API
    raw_spd = _sanitize_finite_float(speed_kmh)
    sanitized_spd = raw_spd if (raw_spd is not None and raw_spd >= 0.0) else None

    # 6. Route completion
    sanitized_route = _sanitize_finite_float(route_completion)

    # 7. Steering
    sanitized_steering = _sanitize_finite_float(canonical_steering)

    # 8. Throttle/brake
    sanitized_tb = _sanitize_finite_float(canonical_throttle_brake)

    # 9. Clean success: strictly boolean (True/False); do not coerce non-boolean types
    clean_succ: Optional[bool] = None
    if type(is_clean_success) is bool:
        clean_succ = is_clean_success

    return HudTelemetrySnapshotV1(
        step_index=idx,
        simulation_time_s=sim_t,
        world_position=sanitized_pos,
        heading_theta_rad=sanitized_heading,
        speed_kmh=sanitized_spd,
        route_completion=sanitized_route,
        canonical_steering=sanitized_steering,
        canonical_throttle_brake=sanitized_tb,
        terminal_outcome_status=str(terminal_outcome_status),
        is_clean_success=clean_succ,
    )
