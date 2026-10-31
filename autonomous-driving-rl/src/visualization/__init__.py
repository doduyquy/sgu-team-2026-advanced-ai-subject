"""
Research Platform V1 Visualization & HUD Subsystem (Gate HUD-1, HUD-2 & HUD-3).

Provides hardware-accelerated minimap, transparent racing HUD overlay, and
the unified RacingHudManager facade for MetaDrive NATIVE mode.
"""

from src.visualization.hud_manager import RacingHudManager
from src.visualization.minimap_math import (
    TrajectoryTrailBuffer,
    calculate_film_size,
    calculate_heading_rotation_deg,
    calculate_map_bounds,
)
from src.visualization.minimap_overlay import MiniMapOverlay
from src.visualization.racing_hud import RacingHud
from src.visualization.telemetry_model import (
    HudTelemetrySnapshotV1,
    create_telemetry_snapshot,
)

__all__ = [
    "MiniMapOverlay",
    "RacingHud",
    "RacingHudManager",
    "HudTelemetrySnapshotV1",
    "create_telemetry_snapshot",
    "calculate_map_bounds",
    "calculate_film_size",
    "calculate_heading_rotation_deg",
    "TrajectoryTrailBuffer",
]
