"""
Pure Mathematical and Geometric Utilities for MetaDrive Live Minimap (HUD-1).

Pure, unit-testable module independent of Panda3D / graphics display state.
Provides coordinate transformations, aspect-ratio film calculations,
heading rotation derivations, and bounded trajectory sampling.
"""

from collections import deque
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def calculate_map_bounds(
    map_features: Optional[Dict[Any, Dict[str, Any]]] = None,
    default_center: Tuple[float, float] = (0.0, 0.0),
    padding_m: float = 12.0
) -> Tuple[float, float, float, float]:
    """
    Computes (min_x, max_x, min_y, max_y) extents of a MetaDrive map from feature polylines.
    Filters out non-finite (NaN, Inf) vertices and degenerate polylines.
    Falls back to a safe 100m x 100m bounding box around default_center if features are empty.
    """
    valid_points: List[Tuple[float, float]] = []

    if map_features:
        for feature in map_features.values():
            if not isinstance(feature, dict):
                continue
            polyline = feature.get("polyline")
            if not isinstance(polyline, (list, tuple, np.ndarray)):
                continue

            for pt in polyline:
                if isinstance(pt, (list, tuple, np.ndarray)) and len(pt) >= 2:
                    try:
                        x = float(pt[0])
                        y = float(pt[1])
                        if math.isfinite(x) and math.isfinite(y):
                            valid_points.append((x, y))
                    except (ValueError, TypeError, IndexError):
                        continue

    if not valid_points:
        cx, cy = default_center
        return (cx - 50.0, cx + 50.0, cy - 50.0, cy + 50.0)

    xs = [p[0] for p in valid_points]
    ys = [p[1] for p in valid_points]
    return (min(xs), max(xs), min(ys), max(ys))


def calculate_film_size(
    bounds: Tuple[float, float, float, float],
    region_aspect: float,
    padding_m: float = 12.0
) -> Tuple[float, float]:
    """
    Calculates orthographic film size (film_width, film_height) preserving map geometric aspect ratio.
    Ensures that road geometry is never distorted or stretched inside the minimap display region.
    """
    if region_aspect <= 0:
        raise ValueError(f"region_aspect must be positive, got {region_aspect}")

    min_x, max_x, min_y, max_y = bounds
    map_width = max(max_x - min_x, 20.0) + 2.0 * padding_m
    map_height = max(max_y - min_y, 20.0) + 2.0 * padding_m

    map_aspect = map_width / map_height

    if map_aspect > region_aspect:
        film_width = map_width
        film_height = map_width / region_aspect
    else:
        film_height = map_height
        film_width = map_height * region_aspect

    return (film_width, film_height)


def calculate_heading_rotation_deg(heading_rad: float) -> float:
    """
    Derives Panda3D 2D roll rotation (degrees) for a heading angle in radians.
    In MetaDrive, heading theta=0 is along +X (East/Right), theta=pi/2 is along +Y (North/Up).
    In Panda3D render2d, rotation around the screen normal (-Z rotation / +Y roll) maps counter-clockwise
    rotations to -degrees(heading_rad).
    """
    if not math.isfinite(heading_rad):
        return 0.0
    return -math.degrees(heading_rad)


class TrajectoryTrailBuffer:
    """
    Bounded FIFO ring buffer storing valid (x, y) vehicle coordinates for the minimap trajectory trail.
    Enforces a minimum spatial distance threshold (0.5m) to filter out stationary vehicle noise.
    """
    def __init__(self, max_capacity: int = 500):
        if max_capacity <= 0:
            raise ValueError(f"max_capacity must be positive integer, got {max_capacity}")
        self.max_capacity = int(max_capacity)
        self._buffer: deque = deque(maxlen=self.max_capacity)

    def add_sample(
        self,
        x: Optional[float],
        y: Optional[float],
        min_distance: float = 0.5
    ) -> bool:
        """
        Appends a new point if coordinates are valid, finite, and Euclidean distance from
        the previous retained point is >= min_distance (or if buffer is empty).
        Returns True if the point was retained, False otherwise.
        """
        if x is None or y is None:
            return False

        try:
            fx = float(x)
            fy = float(y)
        except (ValueError, TypeError):
            return False

        if not math.isfinite(fx) or not math.isfinite(fy):
            return False

        if len(self._buffer) == 0:
            self._buffer.append((fx, fy))
            return True

        last_x, last_y = self._buffer[-1]
        dist = math.hypot(fx - last_x, fy - last_y)

        if dist >= min_distance:
            self._buffer.append((fx, fy))
            return True

        return False

    def get_points(self) -> List[Tuple[float, float]]:
        """Returns ordered list of retained trajectory coordinates (oldest to newest)."""
        return list(self._buffer)

    def reset(self) -> None:
        """Empties the trajectory buffer. Ensures zero cross-episode leakage."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
