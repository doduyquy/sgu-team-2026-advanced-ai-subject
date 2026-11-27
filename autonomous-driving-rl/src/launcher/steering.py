"""Optional Stage 5 SANDBOX steering profiles and decision-cycle telemetry."""
from dataclasses import dataclass
from math import atan2, cos, sin

from src.platform.action_adapter import CanonicalActionV1


def wrap(angle):
    return atan2(sin(angle), cos(angle))


@dataclass
class SteeringFilter:
    profile: str
    previous: float | None = None

    def apply(self, action, speed_kmh, task):
        # Relative tangent changes cancel the ego heading. Use the next 10 m.
        headings = [point.relative_heading for point, valid in
                    zip(task.waypoints[:4], task.validity_mask[:4]) if valid]
        bend = sum(abs(wrap(b - a)) for a, b in zip(headings, headings[1:]))
        if self.profile == "adaptive":
            speed_weight = min(1.0, max(0.0, (speed_kmh - 15.0) / 25.0))
            bend_weight = min(1.0, bend / 0.20) if len(headings) >= 2 else 1.0
            alpha = 0.85 - 0.50 * speed_weight * (1.0 - bend_weight)
        else:
            alpha = 0.55 if self.profile == "fixed" else 1.0
        steering = action.steering
        if self.previous is not None and alpha < 1.0:
            steering = alpha * steering + (1.0 - alpha) * self.previous
        self.previous = steering
        return CanonicalActionV1(steering, action.throttle_brake), alpha, bend


def vehicle_motion(vehicle):
    return float(vehicle.heading_theta), tuple(float(v) for v in vehicle.velocity[:2])


def steering_sample(vehicle, before, after, dt, raw, executed, alpha, bend, previous_accel):
    delta_yaw = wrap(after[0] - before[0])
    heading = before[0] + delta_yaw / 2.0
    dvx = after[1][0] - before[1][0]
    dvy = after[1][1] - before[1][1]
    lateral_accel = (-sin(heading) * dvx + cos(heading) * dvy) / dt
    lane = vehicle.lane
    return {
        "steering_raw": raw.steering,
        "steering_executed": executed.steering,
        "steering_simulator": float(vehicle.steering),
        "throttle_brake_raw": raw.throttle_brake,
        "throttle_brake_executed": executed.throttle_brake,
        "speed_kmh": float(vehicle.speed_km_h),
        "alpha": alpha,
        "preview_bend_rad": bend,
        "cross_track_m": float(lane.local_coordinates(vehicle.position)[1]) if lane is not None else None,
        "yaw_rate_rad_s": delta_yaw / dt,
        "lateral_accel_m_s2": lateral_accel,
        "lateral_jerk_m_s3": (lateral_accel - previous_accel) / dt if previous_accel is not None else None,
    }
