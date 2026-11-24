"""
Research Platform V1 Visualization & HUD Subsystem (Gate HUD-1, HUD-2 & HUD-3).

Provides hardware-accelerated minimap, transparent racing HUD overlay, and
the unified RacingHudManager facade for MetaDrive NATIVE mode.
"""

from __future__ import annotations

from typing import Any, Optional

from src.visualization.finish_floating_hologram import FinishFloatingHologram
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

FinishLineVisual = FinishFloatingHologram


def create_finish_visual(
    env: Any,
    style: str = "hologram",
    **kwargs: Any,
) -> Optional[FinishFloatingHologram]:
    """
    Factory khởi tạo đối tượng vạch đích 3D Hologram cuối bản đồ.

    Parameters:
    - env: MetaDrive environment đã qua env.reset()
    - style:
        + 'hologram' | 'neon' | 'cyber' | 'on': Hiển thị cổng vạch đích Hologram 3D Cyberpunk
        + 'none' | 'off' | 'disabled': Tắt không hiển thị
    """
    normalized_style = style.lower().strip()

    if normalized_style in ("none", "off", "disabled"):
        return None
    elif normalized_style in ("hologram", "neon", "cyber", "on", "default", ""):
        return FinishFloatingHologram(env, **kwargs)
    else:
        # Dự phòng chấp nhận bất kỳ cấu hình nào bật visual
        return FinishFloatingHologram(env, **kwargs)

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
    "FinishFloatingHologram",
    "FinishLineVisual",
    "create_finish_visual",
]
