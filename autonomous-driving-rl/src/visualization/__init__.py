"""
Research Platform V1 Visualization & HUD Subsystem (Gate HUD-1).

Provides hardware-accelerated minimap and HUD overlay components for MetaDrive NATIVE mode.
"""

from src.visualization.minimap_math import (
    TrajectoryTrailBuffer,
    calculate_film_size,
    calculate_heading_rotation_deg,
    calculate_map_bounds,
)
from src.visualization.minimap_overlay import MiniMapOverlay

__all__ = [
    "MiniMapOverlay",
    "calculate_map_bounds",
    "calculate_film_size",
    "calculate_heading_rotation_deg",
    "TrajectoryTrailBuffer",
]
