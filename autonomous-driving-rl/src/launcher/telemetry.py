"""Đọc số đo vật lý cho nhật ký; lỗi cảm biến không được biến thành số 0."""

import math


def read_vehicle_speed_kmh(vehicle):
    """MetaDrive 0.4.3 exposes speed_km_h in km/h; log keys retain speed_kmh."""
    speed = float(vehicle.speed_km_h)
    if not math.isfinite(speed) or speed < 0:
        raise ValueError(f"Invalid MetaDrive speed_km_h: {speed}")
    return speed
