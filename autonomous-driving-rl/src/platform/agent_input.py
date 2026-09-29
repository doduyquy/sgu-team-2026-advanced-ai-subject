"""
AgentInput Specification and Context Representations for Research Platform V1.
Gate 6 of Research Platform V1.

Implements the common runtime information boundary:
  AgentInputV1
  ├── CoreObservationV1 (259D defensive read-only vector)
  ├── TrafficContextV1   (Structured state-based local traffic actors)
  └── TaskContextV1      (Read-only ego-relative reference route lookahead)

Security / Scientific Mandates:
- ZERO live simulator handles (no MetaDriveEnv, engine, Panda3D, vehicle, traffic manager).
- ZERO evaluator-private telemetry (no route_completion, terminal reason, collision flags, rewards, case/tier/split identity, env seed).
- Defensive immutability (exported NumPy arrays marked writeable=False).
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class InputProfileId(str, Enum):
    """Certified information profiles for Research Platform V1."""
    STATE_DECISION_V1 = "STATE_DECISION_V1"  # Main benchmark profile: CoreObservation + TrafficContext + TaskContext
    CORE_ONLY_V1 = "CORE_ONLY_V1"            # Ablation profile: CoreObservation only (Traffic & Task contexts masked)


FORBIDDEN_EVALUATOR_FIELDS = frozenset({
    "route_completion",
    "max_route_completion",
    "distance_along_route",
    "percent_complete",
    "arrive_dest",
    "clean_success",
    "crash_human",
    "crash_vehicle",
    "crash_object",
    "crash_building",
    "crash_sidewalk",
    "out_of_road",
    "terminated",
    "truncated",
    "terminal_reason",
    "primary_terminal_reason",
    "episode_return",
    "reward",
    "reward_breakdown",
    "future_outcome",
    "ground_truth_trajectory",
    "test_score",
    "case_result",
    "tier",
    "difficulty_tier",
    "split",
    "candidate_role",
    "case_id",
    "protocol_order_index",
    "geometry_sha256",
    "geometry_generation_seed",
    "environment_seed",
    "env_seed",
    "test_manifest_position",
})


@dataclass(frozen=True)
class CoreObservationV1:
    """
    Defensive, immutable wrapper around Gate-1 259-dimensional local observation vector.
    Features breakdown:
      0..8   (9D)  : Ego vehicle state (heading theta diff, steering, velocity x/y, gyro, angular vel)
      9..18  (10D) : Navigation checkpoints vector (5 relative checkpoint directions)
      19..258(240D): 360-degree LiDAR cloud points distance rays in [0.0, 1.0]
    Normalized range contract: strictly [0.0, 1.0].
    """
    features: np.ndarray

    def __post_init__(self):
        # Convert floating input safely to float32
        if not isinstance(self.features, np.ndarray):
            object.__setattr__(self, "features", np.asarray(self.features, dtype=np.float32))

        if self.features.shape != (259,):
            raise ValueError(f"CoreObservation features must have shape (259,), got {self.features.shape}")

        if not np.issubdtype(self.features.dtype, np.floating):
            raise TypeError(f"CoreObservation features must be floating type, got {self.features.dtype}")

        if not np.all(np.isfinite(self.features)):
            raise ValueError("CoreObservation contains non-finite values (NaN or Inf).")

        # Explicit bounds validation: normalized range [0.0, 1.0] with small numerical tolerance (1e-4)
        TOLERANCE = 1e-4
        if np.any(self.features < -TOLERANCE) or np.any(self.features > 1.0 + TOLERANCE):
            min_val = float(np.min(self.features))
            max_val = float(np.max(self.features))
            raise ValueError(
                f"CoreObservation features violate declared normalized range [0.0, 1.0]: "
                f"min={min_val:.4f}, max={max_val:.4f}"
            )

        # Defensive copy and read-only enforcement
        copied = np.array(self.features, dtype=np.float32, copy=True)
        copied.flags.writeable = False
        object.__setattr__(self, "features", copied)

    @property
    def ego_state(self) -> np.ndarray:
        return self.features[0:9]

    @property
    def navigation_vector(self) -> np.ndarray:
        return self.features[9:19]

    @property
    def lidar_points(self) -> np.ndarray:
        return self.features[19:259]

    def to_dict(self) -> Dict[str, Any]:
        return {"shape": list(self.features.shape), "dtype": str(self.features.dtype), "features": self.features.tolist()}


@dataclass(frozen=True)
class TrafficActorV1:
    """
    Ego-centric structured state of an observed surrounding traffic vehicle.
    Coordinate frame: Ego vehicle local frame.
      +x: Forward (along ego longitudinal axis)
      -x: Backward (behind ego)
      +y: Left (ego lateral left)
      -y: Right (ego lateral right)
    Contains exactly 7 numeric actor features matching the declared schema.
    """
    relative_position_x: float  # meters
    relative_position_y: float  # meters
    relative_velocity_x: float  # m/s
    relative_velocity_y: float  # m/s
    relative_heading: float     # radians in [-pi, pi]
    length: float               # meters
    width: float                # meters

    def __post_init__(self):
        for field_name in (
            "relative_position_x", "relative_position_y",
            "relative_velocity_x", "relative_velocity_y",
            "relative_heading", "length", "width"
        ):
            val = getattr(self, field_name)
            if not isinstance(val, (int, float, np.floating)):
                raise TypeError(f"TrafficActor field '{field_name}' must be numeric, got {type(val)}")
            if not math.isfinite(val):
                raise ValueError(f"TrafficActor field '{field_name}' must be finite, got {val}")

        if self.length < 0.0 or self.width < 0.0:
            raise ValueError(f"TrafficActor dimensions must be non-negative, got length={self.length}, width={self.width}")

    def to_tuple(self) -> Tuple[float, float, float, float, float, float, float]:
        """Returns 7D numeric tuple [rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y, rel_heading, length, width]."""
        return (
            float(self.relative_position_x),
            float(self.relative_position_y),
            float(self.relative_velocity_x),
            float(self.relative_velocity_y),
            float(self.relative_heading),
            float(self.length),
            float(self.width),
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "relative_position_x": float(self.relative_position_x),
            "relative_position_y": float(self.relative_position_y),
            "relative_velocity_x": float(self.relative_velocity_x),
            "relative_velocity_y": float(self.relative_velocity_y),
            "relative_heading": float(self.relative_heading),
            "length": float(self.length),
            "width": float(self.width),
        }


@dataclass(frozen=True)
class TrafficContextV1:
    """
    State-based structured local traffic context for decision and planning.
    Fixed capacity N=8 actors (calibrated empirically on TRAIN/VAL).
    Sorted deterministically by Euclidean distance to ego vehicle.
    """
    capacity: int = 8
    radius_m: float = 50.0
    actors: Tuple[TrafficActorV1, ...] = field(default_factory=tuple)
    actors_array: np.ndarray = field(default_factory=lambda: np.zeros((8, 7), dtype=np.float32))
    validity_mask: np.ndarray = field(default_factory=lambda: np.zeros((8,), dtype=bool))
    active_count: int = 0
    overflow_count: int = 0

    def __post_init__(self):
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity}")
        if self.radius_m <= 0.0:
            raise ValueError(f"radius_m must be positive, got {self.radius_m}")
        if not (0 <= self.active_count <= self.capacity):
            raise ValueError(f"active_count must be in [0, {self.capacity}], got {self.active_count}")
        if self.overflow_count < 0:
            raise ValueError(f"overflow_count must be non-negative, got {self.overflow_count}")

        if len(self.actors) != self.capacity:
            raise ValueError(f"TrafficContext actors tuple length must equal capacity {self.capacity}, got {len(self.actors)}")
        if self.actors_array.shape != (self.capacity, 7):
            raise ValueError(f"TrafficContext actors_array must have shape ({self.capacity}, 7), got {self.actors_array.shape}")
        if self.validity_mask.shape != (self.capacity,):
            raise ValueError(f"TrafficContext validity_mask must have shape ({self.capacity},), got {self.validity_mask.shape}")

        if not np.all(np.isfinite(self.actors_array)):
            raise ValueError("actors_array contains non-finite values.")

        true_mask_count = int(np.sum(self.validity_mask))
        if true_mask_count != self.active_count:
            raise ValueError(f"validity_mask True count ({true_mask_count}) does not match active_count ({self.active_count})")

        # Verify active slots consistency between actors tuple and actors_array
        for idx in range(self.active_count):
            if not self.validity_mask[idx]:
                raise ValueError(f"Active slot {idx} must have validity_mask=True")
            if not np.allclose(self.actors_array[idx], self.actors[idx].to_tuple(), atol=1e-4):
                raise ValueError(f"actors_array[{idx}] does not match actors[{idx}].to_tuple()")

        # Verify padded invalid slots are zeroed out
        for idx in range(self.active_count, self.capacity):
            if self.validity_mask[idx]:
                raise ValueError(f"Slot {idx} beyond active_count must have validity_mask=False")
            if not np.all(self.actors_array[idx] == 0.0):
                raise ValueError(f"Padded slot {idx} must be zeroed out in actors_array")
            if self.actors[idx].to_tuple() != (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0):
                raise ValueError(f"Padded actor {idx} must be zero-valued")

        # Always defensive copy to prevent retaining aliases to caller-owned memory
        arr_copy = np.array(self.actors_array, dtype=np.float32, copy=True)
        arr_copy.flags.writeable = False
        object.__setattr__(self, "actors_array", arr_copy)

        mask_copy = np.array(self.validity_mask, dtype=bool, copy=True)
        mask_copy.flags.writeable = False
        object.__setattr__(self, "validity_mask", mask_copy)

    @classmethod
    def empty(cls, capacity: int = 8, radius_m: float = 50.0) -> "TrafficContextV1":
        """Constructs an empty/masked traffic context."""
        empty_actor = TrafficActorV1(
            relative_position_x=0.0,
            relative_position_y=0.0,
            relative_velocity_x=0.0,
            relative_velocity_y=0.0,
            relative_heading=0.0,
            length=0.0,
            width=0.0
        )
        actors = tuple(empty_actor for _ in range(capacity))
        actors_arr = np.zeros((capacity, 7), dtype=np.float32)
        valid_mask = np.zeros((capacity,), dtype=bool)
        return cls(
            capacity=capacity,
            radius_m=radius_m,
            actors=actors,
            actors_array=actors_arr,
            validity_mask=valid_mask,
            active_count=0,
            overflow_count=0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "radius_m": self.radius_m,
            "active_count": self.active_count,
            "overflow_count": self.overflow_count,
            "actors": [a.to_dict() for a in self.actors],
            "validity_mask": self.validity_mask.tolist(),
        }


@dataclass(frozen=True)
class RouteWaypointV1:
    """Sampled route lookahead waypoint in ego-relative coordinates."""
    lookahead_distance_m: float  # Nominal distance along route ahead of ego (e.g. 2.5, 5.0, ..., 50.0)
    relative_x: float            # Forward coordinate relative to ego (meters)
    relative_y: float            # Lateral coordinate relative to ego (meters)
    relative_heading: float      # Heading deviation relative to ego heading (radians in [-pi, pi])

    def __post_init__(self):
        for field_name in ("lookahead_distance_m", "relative_x", "relative_y", "relative_heading"):
            val = getattr(self, field_name)
            if not isinstance(val, (int, float, np.floating)):
                raise TypeError(f"RouteWaypoint field '{field_name}' must be numeric, got {type(val)}")
            if not math.isfinite(val):
                raise ValueError(f"RouteWaypoint field '{field_name}' must be finite, got {val}")

    def to_tuple(self) -> Tuple[float, float, float]:
        return (float(self.relative_x), float(self.relative_y), float(self.relative_heading))

    def to_dict(self) -> Dict[str, float]:
        return {
            "lookahead_distance_m": float(self.lookahead_distance_m),
            "relative_x": float(self.relative_x),
            "relative_y": float(self.relative_y),
            "relative_heading": float(self.relative_heading),
        }


@dataclass(frozen=True)
class TaskContextV1:
    """
    Read-only public planning context providing ego-relative route lookahead.
    Contains NO evaluator private progress scalars (no route_completion, no total distance).
    route_end_within_lookahead indicates only that the designated reference route ends within
    the sampled lookahead distance; it does NOT indicate episode termination.
    """
    lookahead_count: int = 20
    lookahead_spacing_m: float = 2.5
    lookahead_range_m: float = 50.0
    current_lane_width: float = 3.5
    navigation_goal_direction: Tuple[float, float] = (1.0, 0.0)  # (cos_angle, sin_angle) to next checkpoint
    waypoints: Tuple[RouteWaypointV1, ...] = field(default_factory=tuple)
    waypoints_array: np.ndarray = field(default_factory=lambda: np.zeros((20, 3), dtype=np.float32))
    validity_mask: np.ndarray = field(default_factory=lambda: np.zeros((20,), dtype=bool))
    route_end_within_lookahead: bool = False

    def __post_init__(self):
        if self.lookahead_count <= 0:
            raise ValueError(f"lookahead_count must be positive, got {self.lookahead_count}")
        if self.lookahead_spacing_m <= 0.0:
            raise ValueError(f"lookahead_spacing_m must be positive, got {self.lookahead_spacing_m}")
        expected_range = self.lookahead_count * self.lookahead_spacing_m
        if abs(self.lookahead_range_m - expected_range) > 1e-4:
            raise ValueError(f"lookahead_range_m ({self.lookahead_range_m}) must equal lookahead_count * spacing ({expected_range})")
        if not math.isfinite(self.current_lane_width) or self.current_lane_width <= 0.0:
            raise ValueError(f"current_lane_width must be positive finite float, got {self.current_lane_width}")
        if len(self.navigation_goal_direction) != 2 or not all(math.isfinite(x) for x in self.navigation_goal_direction):
            raise ValueError(f"navigation_goal_direction must be 2 finite floats, got {self.navigation_goal_direction}")

        if len(self.waypoints) != self.lookahead_count:
            raise ValueError(f"TaskContext waypoints length must equal lookahead_count {self.lookahead_count}, got {len(self.waypoints)}")
        if self.waypoints_array.shape != (self.lookahead_count, 3):
            raise ValueError(f"TaskContext waypoints_array shape must be ({self.lookahead_count}, 3), got {self.waypoints_array.shape}")
        if self.validity_mask.shape != (self.lookahead_count,):
            raise ValueError(f"TaskContext validity_mask shape must be ({self.lookahead_count},), got {self.validity_mask.shape}")

        if not np.all(np.isfinite(self.waypoints_array)):
            raise ValueError("waypoints_array contains non-finite values.")

        # Consistency check between waypoints and waypoints_array
        for idx in range(self.lookahead_count):
            if self.validity_mask[idx]:
                if not np.allclose(self.waypoints_array[idx], self.waypoints[idx].to_tuple(), atol=1e-4):
                    raise ValueError(f"waypoints_array[{idx}] does not match waypoints[{idx}].to_tuple()")
            else:
                if not np.all(self.waypoints_array[idx] == 0.0):
                    raise ValueError(f"Invalid waypoint {idx} must be zeroed out in waypoints_array")
                if self.waypoints[idx].to_tuple() != (0.0, 0.0, 0.0):
                    raise ValueError(f"Invalid waypoint {idx} must be zero-valued in waypoints tuple")

        # Always defensive copy to prevent retaining aliases to caller-owned memory
        wp_copy = np.array(self.waypoints_array, dtype=np.float32, copy=True)
        wp_copy.flags.writeable = False
        object.__setattr__(self, "waypoints_array", wp_copy)

        m_copy = np.array(self.validity_mask, dtype=bool, copy=True)
        m_copy.flags.writeable = False
        object.__setattr__(self, "validity_mask", m_copy)

    @classmethod
    def empty(cls, lookahead_count: int = 20, lookahead_spacing_m: float = 2.5) -> "TaskContextV1":
        """Constructs an empty/masked task context."""
        empty_wp = RouteWaypointV1(
            lookahead_distance_m=0.0,
            relative_x=0.0,
            relative_y=0.0,
            relative_heading=0.0
        )
        waypoints = tuple(empty_wp for _ in range(lookahead_count))
        wps_arr = np.zeros((lookahead_count, 3), dtype=np.float32)
        valid_mask = np.zeros((lookahead_count,), dtype=bool)
        return cls(
            lookahead_count=lookahead_count,
            lookahead_spacing_m=lookahead_spacing_m,
            lookahead_range_m=lookahead_count * lookahead_spacing_m,
            current_lane_width=3.5,
            navigation_goal_direction=(1.0, 0.0),
            waypoints=waypoints,
            waypoints_array=wps_arr,
            validity_mask=valid_mask,
            route_end_within_lookahead=False
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lookahead_count": self.lookahead_count,
            "lookahead_spacing_m": self.lookahead_spacing_m,
            "lookahead_range_m": self.lookahead_range_m,
            "current_lane_width": self.current_lane_width,
            "navigation_goal_direction": list(self.navigation_goal_direction),
            "route_end_within_lookahead": self.route_end_within_lookahead,
            "waypoints": [w.to_dict() for w in self.waypoints],
            "validity_mask": self.validity_mask.tolist(),
        }


@dataclass(frozen=True)
class AgentInputV1:
    """
    Authoritative decision-cycle input provided to all benchmark-compliant agents.
    Same information rights for all stages: Stage 0 through Stage 7.
    """
    profile_id: InputProfileId
    core_observation: CoreObservationV1
    traffic_context: TrafficContextV1
    task_context: TaskContextV1
    step_index: int  # 0-indexed decision step count within episode (0, 1, 2, ...)

    def __post_init__(self):
        if self.step_index < 0:
            raise ValueError(f"step_index must be non-negative, got {self.step_index}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id.value,
            "step_index": self.step_index,
            "core_observation": self.core_observation.to_dict(),
            "traffic_context": self.traffic_context.to_dict(),
            "task_context": self.task_context.to_dict(),
        }


# ==============================================================================
# Pure extraction helpers from simulator objects (Simulator Isolation Boundary)
# ==============================================================================

def extract_core_observation(raw_obs: Any) -> CoreObservationV1:
    """Extracts, validates, and creates a defensive read-only CoreObservationV1."""
    return CoreObservationV1(features=raw_obs)


def extract_traffic_context(
    ego_vehicle: Any,
    traffic_vehicles: List[Any],
    capacity: int = 8,
    radius_m: float = 50.0
) -> TrafficContextV1:
    """
    Extracts ego-centric structured traffic context from simulator ground-truth states.
    Orders detected actors deterministically by Euclidean distance with deterministic tie-breaking.
    Pads to fixed capacity N.
    """
    ego_pos = (float(ego_vehicle.position[0]), float(ego_vehicle.position[1]))
    ego_theta = float(ego_vehicle.heading_theta)
    ego_vel = ego_vehicle.velocity
    ego_vx = float(ego_vel[0]) if hasattr(ego_vel, "__getitem__") else 0.0
    ego_vy = float(ego_vel[1]) if hasattr(ego_vel, "__getitem__") else 0.0

    cos_th = math.cos(ego_theta)
    sin_th = math.sin(ego_theta)

    detected = []
    for v in traffic_vehicles:
        # Ignore ego vehicle itself if present in container
        if v is ego_vehicle:
            continue

        v_pos = (float(v.position[0]), float(v.position[1]))
        dx = v_pos[0] - ego_pos[0]
        dy = v_pos[1] - ego_pos[1]
        dist = math.hypot(dx, dy)

        if dist > radius_m:
            continue

        # Ego-centric relative position:
        # x_rel: forward, y_rel: left
        x_rel = dx * cos_th + dy * sin_th
        y_rel = -dx * sin_th + dy * cos_th

        # Velocity in global frame
        v_vel = v.velocity
        v_vx = float(v_vel[0]) if hasattr(v_vel, "__getitem__") else 0.0
        v_vy = float(v_vel[1]) if hasattr(v_vel, "__getitem__") else 0.0

        # Relative velocity rotated into ego frame
        dvx = v_vx - ego_vx
        dvy = v_vy - ego_vy
        vx_rel = dvx * cos_th + dvy * sin_th
        vy_rel = -dvx * sin_th + dvy * cos_th

        # Relative heading in [-pi, pi]
        v_theta = float(v.heading_theta)
        hd_diff = (v_theta - ego_theta + math.pi) % (2.0 * math.pi) - math.pi

        length = float(getattr(v, "LENGTH", 4.5))
        width = float(getattr(v, "WIDTH", 1.8))

        actor = TrafficActorV1(
            relative_position_x=float(round(x_rel, 4)),
            relative_position_y=float(round(y_rel, 4)),
            relative_velocity_x=float(round(vx_rel, 4)),
            relative_velocity_y=float(round(vy_rel, 4)),
            relative_heading=float(round(hd_diff, 4)),
            length=float(round(length, 2)),
            width=float(round(width, 2))
        )
        detected.append((dist, actor))

    # Deterministic sorting: distance ascending, tie-breaker: (x_rel, y_rel)
    detected_sorted = sorted(detected, key=lambda item: (item[0], item[1].relative_position_x, item[1].relative_position_y))

    active_count = len(detected_sorted)
    overflow_count = max(0, active_count - capacity)
    selected_items = detected_sorted[:capacity]

    # Create padded slots to fixed capacity
    padded_actors = [item[1] for item in selected_items]
    valid_flags = [True] * len(padded_actors)

    empty_actor = TrafficActorV1(
        relative_position_x=0.0,
        relative_position_y=0.0,
        relative_velocity_x=0.0,
        relative_velocity_y=0.0,
        relative_heading=0.0,
        length=0.0,
        width=0.0
    )
    while len(padded_actors) < capacity:
        padded_actors.append(empty_actor)
        valid_flags.append(False)

    actors_tuple = tuple(padded_actors)
    actors_arr = np.array([a.to_tuple() for a in actors_tuple], dtype=np.float32)
    valid_mask = np.array(valid_flags, dtype=bool)

    return TrafficContextV1(
        capacity=capacity,
        radius_m=radius_m,
        actors=actors_tuple,
        actors_array=actors_arr,
        validity_mask=valid_mask,
        active_count=min(active_count, capacity),
        overflow_count=overflow_count
    )


def extract_task_context(
    navigation: Any,
    ego_vehicle: Any,
    road_network: Any,
    lookahead_count: int = 20,
    lookahead_spacing_m: float = 2.5
) -> TaskContextV1:
    """
    Extracts ego-relative reference route lookahead without exposing evaluator progress metrics.
    Samples waypoints along the designated reference path ahead of current ego position.
    """
    ego_pos = (float(ego_vehicle.position[0]), float(ego_vehicle.position[1]))
    ego_theta = float(ego_vehicle.heading_theta)
    cos_th = math.cos(ego_theta)
    sin_th = math.sin(ego_theta)

    current_lane = navigation.current_lane
    lane_width = float(navigation.get_current_lane_width())

    # Direction to next navigation checkpoint
    navi_info = navigation.get_navi_info()
    if len(navi_info) >= 2:
        # Checkpoint vector in ego frame
        cp_dx = float(navi_info[0]) - 0.5
        cp_dy = float(navi_info[1]) - 0.5
        norm = math.hypot(cp_dx, cp_dy)
        if norm > 1e-4:
            goal_dir = (float(round(cp_dx / norm, 4)), float(round(cp_dy / norm, 4)))
        else:
            goal_dir = (1.0, 0.0)
    else:
        goal_dir = (1.0, 0.0)

    # Collect route lanes from current checkpoint index onwards
    ckpt_idx = navigation._target_checkpoints_index[0]
    checkpoints = navigation.checkpoints
    route_lanes = []
    if checkpoints and len(checkpoints) >= 2:
        for i in range(ckpt_idx, len(checkpoints) - 1):
            u, v = checkpoints[i], checkpoints[i + 1]
            if u in road_network.graph and v in road_network.graph[u]:
                lanes = road_network.graph[u][v]
                if lanes:
                    route_lanes.append(lanes[0])

    if not route_lanes and current_lane is not None:
        route_lanes = [current_lane]

    # Sample waypoints along route
    long_s, _ = current_lane.local_coordinates(ego_pos) if current_lane else (0.0, 0.0)
    sampled_waypoints = []
    valid_flags = []
    route_ended = False

    for k in range(1, lookahead_count + 1):
        target_s = long_s + k * lookahead_spacing_m
        temp_s = target_s
        found = False

        for l in route_lanes:
            if temp_s <= l.length:
                pt = l.position(temp_s, 0.0)
                hd = l.heading_theta_at(temp_s)
                dx = float(pt[0]) - ego_pos[0]
                dy = float(pt[1]) - ego_pos[1]
                x_rel = dx * cos_th + dy * sin_th
                y_rel = -dx * sin_th + dy * cos_th
                th_rel = (hd - ego_theta + math.pi) % (2.0 * math.pi) - math.pi

                sampled_waypoints.append(RouteWaypointV1(
                    lookahead_distance_m=float(round(k * lookahead_spacing_m, 2)),
                    relative_x=float(round(x_rel, 4)),
                    relative_y=float(round(y_rel, 4)),
                    relative_heading=float(round(th_rel, 4))
                ))
                valid_flags.append(True)
                found = True
                break
            else:
                temp_s -= l.length

        if not found:
            route_ended = True
            sampled_waypoints.append(RouteWaypointV1(
                lookahead_distance_m=float(round(k * lookahead_spacing_m, 2)),
                relative_x=0.0,
                relative_y=0.0,
                relative_heading=0.0
            ))
            valid_flags.append(False)

    waypoints_tuple = tuple(sampled_waypoints)
    wps_arr = np.array([w.to_tuple() for w in waypoints_tuple], dtype=np.float32)
    valid_mask = np.array(valid_flags, dtype=bool)

    return TaskContextV1(
        lookahead_count=lookahead_count,
        lookahead_spacing_m=lookahead_spacing_m,
        lookahead_range_m=lookahead_count * lookahead_spacing_m,
        current_lane_width=float(round(lane_width, 2)),
        navigation_goal_direction=goal_dir,
        waypoints=waypoints_tuple,
        waypoints_array=wps_arr,
        validity_mask=valid_mask,
        route_end_within_lookahead=route_ended
    )


def build_agent_input(
    raw_obs: Any,
    ego_vehicle: Any,
    traffic_vehicles: List[Any],
    navigation: Any,
    road_network: Any,
    step_index: int,
    profile_id: InputProfileId = InputProfileId.STATE_DECISION_V1,
    traffic_capacity: int = 8,
    traffic_radius_m: float = 50.0,
    task_lookahead_count: int = 20,
    task_lookahead_spacing_m: float = 2.5
) -> AgentInputV1:
    """
    Constructs an authoritative AgentInputV1 adhering strictly to information profile.
    Main profile STATE_DECISION_V1 provides CoreObservation + TrafficContext + TaskContext.
    CORE_ONLY_V1 provides CoreObservation with masked empty traffic and task contexts.
    """
    core_obs = extract_core_observation(raw_obs)

    if profile_id == InputProfileId.STATE_DECISION_V1:
        traffic_ctx = extract_traffic_context(
            ego_vehicle=ego_vehicle,
            traffic_vehicles=traffic_vehicles,
            capacity=traffic_capacity,
            radius_m=traffic_radius_m
        )
        task_ctx = extract_task_context(
            navigation=navigation,
            ego_vehicle=ego_vehicle,
            road_network=road_network,
            lookahead_count=task_lookahead_count,
            lookahead_spacing_m=task_lookahead_spacing_m
        )
    elif profile_id == InputProfileId.CORE_ONLY_V1:
        traffic_ctx = TrafficContextV1.empty(capacity=traffic_capacity, radius_m=traffic_radius_m)
        task_ctx = TaskContextV1.empty(lookahead_count=task_lookahead_count, lookahead_spacing_m=task_lookahead_spacing_m)
    else:
        raise ValueError(f"Unsupported input profile ID: {profile_id}")

    return AgentInputV1(
        profile_id=profile_id,
        core_observation=core_obs,
        traffic_context=traffic_ctx,
        task_context=task_ctx,
        step_index=step_index
    )
