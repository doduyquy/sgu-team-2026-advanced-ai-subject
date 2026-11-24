"""Shared 96-value encoder and public MetaDrive observation adapter for Agent 4."""

import math
import struct


SCHEMA = "PDD_COMMON_V2"
BOUNDARIES = ("BROKEN_WHITE", "SOLID_WHITE", "SOLID_YELLOW", "EDGE", "UNKNOWN")
ACTOR_FIELDS = ("relative_position_x", "relative_position_y", "relative_velocity_x",
                "relative_velocity_y", "relative_heading", "length", "width")
FEATURE_NAMES = (
    "speed_mps", "velocity_forward_mps", "velocity_left_mps", "ego_valid",
    "lane_width_m", "lane_width_valid",
    *(f"boundary_{side}_{name}" for side in ("left", "right") for name in BOUNDARIES),
    *(f"actor_{slot}_{name}" for slot in range(8) for name in (
        "x_m", "y_left_m", "relative_vx_mps", "relative_vy_mps", "heading_sin",
        "heading_cos", "length_m", "width_m", "valid")),
    "active_count_over_8", "log1p_overflow_count",
    *(f"route_hint_{i}_{name}" for i in (1, 2) for name in ("x_m", "y_left_m", "valid")),
)


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Expected a finite number, got {value!r}")
    return float(value)


def pair(value):
    x, y = value
    return number(x), number(y)


def float32(values):
    return list(struct.unpack(f"<{len(values)}f", struct.pack(f"<{len(values)}f", *values)))


def project_carla(dx, dy, yaw_rad):
    """Transform a world delta into the common forward/left frame."""
    cosine, sine = math.cos(yaw_rad), math.sin(yaw_rad)
    return dx * cosine + dy * sine, dx * sine - dy * cosine


def encode(ego_velocity, lane_width, boundaries, actors, overflow, hints):
    if len(actors) > 8 or type(overflow) is not int or overflow < 0:
        raise ValueError("Invalid traffic capacity or overflow count")
    if overflow and len(actors) != 8:
        raise ValueError("Traffic overflow requires all eight slots occupied")
    if ego_velocity is None:
        result = [0.0] * 4
    else:
        vx, vy = pair(ego_velocity)
        result = [math.hypot(vx, vy), vx, vy, 1.0]
    if lane_width is None:
        result.extend((0.0, 0.0))
    else:
        width = number(lane_width)
        if not 0 < width <= 6:
            raise ValueError("Lane width outside public range (0, 6] m")
        result.extend((width, 1.0))
    left, right = boundaries
    for boundary in (left, right):
        if boundary not in BOUNDARIES:
            raise ValueError(f"Invalid boundary category {boundary!r}")
        result.extend(float(boundary == category) for category in BOUNDARIES)
    for actor in actors:
        x, y, vx, vy, yaw, length, width = map(number, actor)
        # Public positions are rounded to four decimal places after filtering.
        if math.hypot(x, y) > 50.0001 or min(length, width) <= 0:
            raise ValueError("Actor outside public radius or invalid dimensions")
        result.extend((x, y, vx, vy, math.sin(yaw), math.cos(yaw), length, width, 1.0))
    result.extend([0.0] * (9 * (8 - len(actors))))
    result.extend((len(actors) / 8, math.log1p(overflow)))
    first, second = hints
    for hint in (first, second):
        if hint is None:
            result.extend((0.0, 0.0, 0.0))
        else:
            x, y = pair(hint)
            scale = 50.0 / max(50.0, math.hypot(x, y))
            result.extend((x * scale, y * scale, 1.0))
    return float32(result)


def encode_public(snapshot):
    """Accept AgentInputV1 or its serialized public observation."""
    if hasattr(snapshot, "to_dict"):
        snapshot = snapshot.to_dict()
    if snapshot["profile_id"] != "STATE_DECISION_V1":
        raise ValueError("Expected STATE_DECISION_V1")
    motion = snapshot["ego_motion"]
    velocity = None if motion is None else (motion["velocity_x_mps"], motion["velocity_y_mps"])

    traffic = snapshot["traffic_context"]
    mask = traffic["validity_mask"]
    active = traffic["active_count"]
    if (traffic["capacity"] != 8 or traffic["radius_m"] != 50
            or len(traffic["actors"]) != 8 or type(active) is not int or not 0 <= active <= 8
            or any(type(value) is not bool for value in mask)
            or mask != [True] * active + [False] * (8 - active)):
        raise ValueError("Inconsistent public traffic mask, count or capacity")
    actors = [tuple(number(row[key]) for key in ACTOR_FIELDS) for row in traffic["actors"]]
    if any(any(actor) for actor in actors[active:]):
        raise ValueError("Public traffic padding must be zero")

    task = snapshot["task_context"]
    lane_context = task["lane_context"]
    current = None
    if lane_context is not None:
        currents = [lane for lane in lane_context["lanes"] if lane["side"] == "CURRENT"]
        if len(currents) != 1:
            raise ValueError("Public lane context must identify exactly one CURRENT lane")
        current = currents[0]
    lane_width = None if current is None else number(current["width_m"])
    boundaries = (("UNKNOWN", "UNKNOWN") if current is None
                  else (current["left_boundary"], current["right_boundary"]))

    route = task["waypoints"]
    mask = task["validity_mask"]
    if (len(route) != 20 or len(mask) != 20 or any(type(value) is not bool for value in mask)
            or task["lookahead_count"] != 20 or task["lookahead_spacing_m"] != 2.5
            or any(number(row["lookahead_distance_m"]) != 2.5 * i
                   for i, row in enumerate(route, start=1))):
        raise ValueError("Expected twenty public waypoints at 2.5 m increments")
    hints = [(route[i]["relative_x"], route[i]["relative_y"]) if mask[i] else None
             for i in (9, 17)]  # 25 m and 45 m
    return encode(velocity, lane_width, boundaries, actors[:active], traffic["overflow_count"], hints)
