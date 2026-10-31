from __future__ import annotations

from typing import Any, Optional

from src.visualization.finish_floating_hologram import FinishFloatingHologram
from src.visualization.minimap_overlay import MiniMapOverlay

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
    "FinishFloatingHologram",
    "FinishLineVisual",
    "create_finish_visual",
]
