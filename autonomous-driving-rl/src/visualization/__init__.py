"""
Research Platform V1 Visualization & HUD Subsystem (Gate HUD-1 & HUD-2).

Provides hardware-accelerated minimap and transparent racing HUD overlay components
for MetaDrive NATIVE mode.
"""

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
    "HudTelemetrySnapshotV1",
    "create_telemetry_snapshot",
    "calculate_map_bounds",
    "calculate_film_size",
    "calculate_heading_rotation_deg",
    "TrajectoryTrailBuffer",
]
