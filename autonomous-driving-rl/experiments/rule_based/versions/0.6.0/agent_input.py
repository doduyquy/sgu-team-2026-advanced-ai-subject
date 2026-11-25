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

from dataclasses import asdict, dataclass, field, replace
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
class ActorPathV1:
    """Possible local road path, never another actor's private navigation.

    Points are (distance from observed actor, ego x, ego y, ego heading).
    Slot associates a path with this snapshot only, not across frames.
    """
    actor_slot: int
    branch_id: str
    points: Tuple[Tuple[float, float, float, float], ...]

    def __post_init__(self):
        if self.actor_slot < 0 or not self.branch_id or not 2 <= len(self.points) <= 22:
            raise ValueError("Invalid actor path identity or point count.")
        normalized = tuple(tuple(float(value) for value in point) for point in self.points)
        if any(len(point) != 4 or not all(math.isfinite(value) for value in point) for point in normalized):
            raise ValueError("Actor path points must be finite distance/x/y/heading tuples.")
        if normalized[0][0] != 0 or normalized[-1][0] > 50.0001 or any(
                first[0] >= second[0] for first, second in zip(normalized, normalized[1:])):
            raise ValueError("Actor path distances must increase from zero, within 50 m.")
        object.__setattr__(self, "points", normalized)

    def to_dict(self):
        return {"actor_slot": self.actor_slot, "branch_id": self.branch_id,
                "points": [list(point) for point in self.points]}


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
    motion_paths: Tuple[ActorPathV1, ...] = ()
    motion_paths_overflow_count: int = 0

    def __post_init__(self):
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity}")
        if self.radius_m <= 0.0:
            raise ValueError(f"radius_m must be positive, got {self.radius_m}")
        if not (0 <= self.active_count <= self.capacity):
            raise ValueError(f"active_count must be in [0, {self.capacity}], got {self.active_count}")
        if self.overflow_count < 0:
            raise ValueError(f"overflow_count must be non-negative, got {self.overflow_count}")
        if self.motion_paths_overflow_count < 0 or len(self.motion_paths) > self.capacity * 8:
            raise ValueError("Actor road-path capacity exceeded or invalid overflow.")
        object.__setattr__(self, "motion_paths", tuple(self.motion_paths))
        if any(not isinstance(path, ActorPathV1) or path.actor_slot >= self.active_count for path in self.motion_paths):
            raise ValueError("Actor road paths must refer to active snapshot slots.")

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
            "motion_paths": [path.to_dict() for path in self.motion_paths],
            "motion_paths_overflow_count": self.motion_paths_overflow_count,
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
class RoadDecisionPointV1:
    """Local road fork, independent of the evaluator's designated route.

    Exits are deduplicated by direction, not by parallel lane count. No live
    simulator objects or global map are retained in this public snapshot.
    """
    point_id: str
    distance_m: float
    available_turns: Tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.point_id, str) or not self.point_id:
            raise ValueError("Decision point ID must be a non-empty string.")
        if not math.isfinite(self.distance_m) or self.distance_m < 0.0:
            raise ValueError("Decision point distance must be finite and non-negative.")
        turns = tuple(self.available_turns)
        if len(turns) < 2 or len(set(turns)) != len(turns):
            raise ValueError("A decision point must have at least two distinct turns.")
        if any(turn not in ("left", "straight", "right") for turn in turns):
            raise ValueError("Unknown decision point turn.")
        object.__setattr__(self, "available_turns", turns)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "distance_m": float(self.distance_m),
            "available_turns": list(self.available_turns),
        }


@dataclass(frozen=True)
class EgoMotionV1:
    """Local motion since the previous control step, expressed in current ego axes.

    No global pose is exposed. Signed heading and displacement allow policies
    to compensate their own local memories when the ego vehicle turns.
    """
    displacement_x_m: float
    displacement_y_m: float
    heading_delta_rad: float
    velocity_x_mps: float
    velocity_y_mps: float
    lane_heading_error_rad: float

    def __post_init__(self):
        if not all(math.isfinite(value) for value in vars(self).values()):
            raise ValueError("Ego motion must contain finite local measurements.")
        if abs(self.heading_delta_rad) > math.pi or abs(self.lane_heading_error_rad) > math.pi:
            raise ValueError("Ego motion angles must be wrapped to [-pi, pi].")

    def to_dict(self):
        return {name: float(value) for name, value in vars(self).items()}


@dataclass(frozen=True)
class RoutePassageV1:
    """Local route connector/conflict section, not a legal right-of-way label.

    Entry is signed (negative when inside). Exit is None until it is within
    the declared lookahead. No stop-line, traffic-light or priority is invented.
    """
    passage_id: str
    branch_id: str
    kind: str
    entry_distance_m: float
    exit_distance_m: Optional[float]
    maneuver: str
    yield_distance_m: Optional[float] = None
    conflict_exit_distance_m: Optional[float] = None

    def __post_init__(self):
        if not self.passage_id or not self.branch_id:
            raise ValueError("Passage and branch IDs must be non-empty.")
        if self.kind not in ("intersection", "roundabout_merge", "fork"):
            raise ValueError("Unknown route passage kind.")
        if self.maneuver not in ("left", "straight", "right"):
            raise ValueError("Unknown route maneuver.")
        if not math.isfinite(self.entry_distance_m):
            raise ValueError("Passage entry must be finite.")
        if self.exit_distance_m is not None and (
                not math.isfinite(self.exit_distance_m) or self.exit_distance_m <= self.entry_distance_m):
            raise ValueError("Passage exit must be finite and after entry.")
        if self.yield_distance_m is not None and (
                not math.isfinite(self.yield_distance_m) or self.yield_distance_m > self.entry_distance_m):
            raise ValueError("Yield boundary must be finite and no later than connector entry.")
        if self.conflict_exit_distance_m is not None and (
                not math.isfinite(self.conflict_exit_distance_m) or self.conflict_exit_distance_m <= self.entry_distance_m):
            raise ValueError("Conflict exit must be after connector entry.")

    def to_dict(self):
        return dict(vars(self))


@dataclass(frozen=True)
class LaneGeometryV1:
    """Local same-direction lane, points=(signed distance, ego x/y/heading).

    No map handles or global coordinates. Boundary styles describe the lines
    actually constructed by the pinned PG renderer, not unused nominal sides.
    UNKNOWN is deliberately not permission to cross.
    """
    lane_id: str
    side: str
    width_m: float
    remaining_m: float
    left_boundary: str
    right_boundary: str
    route_compatible: Optional[bool]
    points: Tuple[Tuple[float, float, float, float], ...]

    def __post_init__(self):
        styles = {"BROKEN_WHITE", "SOLID_WHITE", "SOLID_YELLOW", "EDGE", "UNKNOWN"}
        if not self.lane_id or self.side not in {"CURRENT", "LEFT", "RIGHT"}:
            raise ValueError("Invalid lane identity/side")
        if self.left_boundary not in styles or self.right_boundary not in styles:
            raise ValueError("Invalid rendered boundary style")
        if self.route_compatible is not None and type(self.route_compatible) is not bool:
            raise TypeError("route_compatible must be bool or None")
        if not math.isfinite(self.width_m) or not 0 < self.width_m <= 6 or not math.isfinite(self.remaining_m) or not 0 <= self.remaining_m <= 50:
            raise ValueError("Invalid bounded lane width/remaining distance")
        points = tuple(tuple(float(v) for v in p) for p in self.points)
        if not 2 <= len(points) <= 28 or any(len(p) != 4 or not all(math.isfinite(v) for v in p) for p in points):
            raise ValueError("Invalid local lane points")
        if points[0][0] < -15.0001 or points[-1][0] > 50.0001 or any(a[0] >= b[0] for a, b in zip(points, points[1:])):
            raise ValueError("Lane points must increase in [-15,50] m")
        object.__setattr__(self, "points", points)

    def to_dict(self):
        return asdict(self) | {"points": [list(p) for p in self.points]}


@dataclass(frozen=True)
class LaneContextV1:
    lanes: Tuple[LaneGeometryV1, ...]
    required_side: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "lanes", tuple(self.lanes))
        if not 1 <= len(self.lanes) <= 3 or any(not isinstance(lane, LaneGeometryV1) for lane in self.lanes):
            raise ValueError("Lane context contains current and at most two neighbors")
        if len({lane.side for lane in self.lanes}) != len(self.lanes) or sum(lane.side == "CURRENT" for lane in self.lanes) != 1 or len({lane.lane_id for lane in self.lanes}) != len(self.lanes):
            raise ValueError("Lane identities and sides must be unique")
        if self.required_side not in {None, "LEFT", "RIGHT"}:
            raise ValueError("Invalid mandatory lane-change side")

    def to_dict(self):
        return {"lanes": [lane.to_dict() for lane in self.lanes], "required_side": self.required_side}


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
    decision_point: Optional[RoadDecisionPointV1] = None
    route_passage: Optional[RoutePassageV1] = None
    lane_context: Optional[LaneContextV1] = None

    def __post_init__(self):
        if self.lane_context is not None and not isinstance(self.lane_context, LaneContextV1):
            raise TypeError("lane_context must be LaneContextV1 or None")
        if self.decision_point is not None and not isinstance(self.decision_point, RoadDecisionPointV1):
            raise TypeError("decision_point must be RoadDecisionPointV1 or None.")
        if self.route_passage is not None and not isinstance(self.route_passage, RoutePassageV1):
            raise TypeError("route_passage must be RoutePassageV1 or None.")
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
        goal_norm = math.hypot(float(self.navigation_goal_direction[0]), float(self.navigation_goal_direction[1]))
        if abs(goal_norm - 1.0) > 1e-2:
            raise ValueError(f"navigation_goal_direction must be an approximately normalized unit vector, got norm={goal_norm:.4f}")

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
            "decision_point": self.decision_point.to_dict() if self.decision_point else None,
            "route_passage": self.route_passage.to_dict() if self.route_passage else None,
            "lane_context": self.lane_context.to_dict() if self.lane_context else None,
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
    ego_motion: Optional[EgoMotionV1] = None

    def __post_init__(self):
        if self.step_index < 0:
            raise ValueError(f"step_index must be non-negative, got {self.step_index}")
        if self.ego_motion is not None and not isinstance(self.ego_motion, EgoMotionV1):
            raise TypeError("ego_motion must be EgoMotionV1 or None.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id.value,
            "step_index": self.step_index,
            "core_observation": self.core_observation.to_dict(),
            "traffic_context": self.traffic_context.to_dict(),
            "task_context": self.task_context.to_dict(),
            "ego_motion": self.ego_motion.to_dict() if self.ego_motion else None,
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


def extract_actor_motion_paths(ego_vehicle, traffic, road_network):
    """Map-match observed poses and enumerate connected local road alternatives.

    Does not read any traffic vehicle's lane/navigation/policy. Every branch is
    a hypothesis from public road geometry, limited to the 50 m ego window.
    """
    graph = road_network.graph
    lanes = [lane for successors in graph.values() for edge in successors.values()
             for lane in edge if getattr(lane, "index", None) is not None]
    theta = float(ego_vehicle.heading_theta)
    c, s = math.cos(theta), math.sin(theta)
    origin = np.asarray(ego_vehicle.position)[:2]
    paths, overflow = [], 0
    for slot, actor in enumerate(traffic.actors[:traffic.active_count]):
        actor_position = origin + np.array((c * actor.relative_position_x - s * actor.relative_position_y,
                                            s * actor.relative_position_x + c * actor.relative_position_y))
        actor_heading = theta + actor.relative_heading
        matches = []
        for lane in lanes:
            along, lateral = lane.local_coordinates(actor_position)
            if along < -0.5 or along > lane.length + 0.5 or abs(lateral) > 1.5:
                continue
            along = float(np.clip(along, 0, lane.length))
            difference = float(lane.heading_theta_at(along)) - actor_heading
            angle = abs(math.atan2(math.sin(difference), math.cos(difference)))
            if angle < 0.65:
                matches.append((abs(float(lateral)) + angle, str(lane.index), along, lane))
        if not matches:
            continue
        matches.sort(key=lambda item: item[:2])
        plausible = [item for item in matches if item[0] <= matches[0][0] + 0.3]
        seeds = plausible[:8]
        overflow += max(0, len(plausible) - len(seeds))
        frontier = [([item[3]], item[2]) for item in reversed(seeds)]
        finished = []
        expansions = 0
        while frontier and expansions < 96:
            chain, start_s = frontier.pop()
            expansions += 1
            last = chain[-1]
            available = sum(float(lane.length) for lane in chain) - start_s
            successors = []
            if available < 50 and len(chain) < 5:
                endpoint = np.asarray(last.position(last.length, 0.0))[:2]
                heading = float(last.heading_theta_at(last.length))
                for node, choices in sorted(graph.get(last.index[1], {}).items(), key=lambda item: str(item[0])):
                    viable = []
                    for lane in choices:
                        gap = float(np.linalg.norm(np.asarray(lane.position(0, 0))[:2] - endpoint))
                        delta = float(lane.heading_theta_at(0)) - heading
                        angle = abs(math.atan2(math.sin(delta), math.cos(delta)))
                        if gap <= 4 and angle < 0.8 and lane not in chain:
                            viable.append((gap + angle * 2, str(lane.index), lane))
                    if viable:
                        successors.append(min(viable, key=lambda item: item[:2])[2])
            if successors:
                frontier.extend((chain + [lane], start_s) for lane in reversed(successors))
            else:
                finished.append((chain, start_s, available))
                if available < 50 and len(chain) >= 5 and graph.get(last.index[1], {}):
                    overflow += 1
        overflow += len(frontier)
        unique_paths = set()
        exported = 0
        for chain, start_s, available in finished:
            points = [(0.0, actor.relative_position_x, actor.relative_position_y, actor.relative_heading)]
            distances = list(np.arange(2.5, min(50.0, available) + 1e-5, 2.5))
            if available < 50 and (not distances or available - distances[-1] > 0.5):
                distances.append(available)
            for distance in distances:
                if distance <= 0.01:
                    continue
                along = start_s + float(distance)
                for lane in chain:
                    if along <= lane.length + 1e-6:
                        position = np.asarray(lane.position(min(along, lane.length), 0))[:2]
                        dx, dy = position - origin
                        x, y = float(dx * c + dy * s), float(-dx * s + dy * c)
                        heading = float(lane.heading_theta_at(min(along, lane.length))) - theta
                        heading = math.atan2(math.sin(heading), math.cos(heading))
                        break
                    along -= lane.length
                else:
                    break
                if math.hypot(x, y) > traffic.radius_m + 1e-4:
                    break
                points.append((round(float(distance), 4), round(x, 4), round(y, 4), round(heading, 4)))
            if len(points) >= 2:
                key = tuple(points)
                if key in unique_paths:
                    continue
                unique_paths.add(key)
                if exported >= 8:
                    overflow += 1
                    continue
                branch = "|".join(str(tuple(lane.index)) for lane in chain)
                paths.append(ActorPathV1(slot, branch, tuple(points)))
                exported += 1
    return tuple(paths), overflow


def extract_road_decision_point(
    ego_vehicle: Any,
    road_network: Any,
    lookahead_range_m: float = 50.0,
) -> Optional[RoadDecisionPointV1]:
    """Find the nearest actual fork on the current road within local lookahead.

    Traverse only single-successor connectors; stop at a fork or dead end.
    Curves and parallel lanes alone are not decision points. Exit headings at
    the end of each outgoing connector distinguish turns whose entry tangents
    are all parallel in MetaDrive intersections. U-turns are excluded.
    """
    lane = getattr(ego_vehicle, "lane", None)
    index = getattr(lane, "index", None)
    if lane is None or index is None:
        return None
    graph = road_network.graph
    longitudinal, _ = lane.local_coordinates(ego_vehicle.position)
    distance = max(0.0, float(lane.length) - float(longitudinal))
    node = index[1]
    visited = set()
    while distance <= lookahead_range_m and node not in visited:
        visited.add(node)
        incoming_heading = float(lane.heading_theta_at(lane.length))
        successors = []
        turns = set()
        for next_node, lanes in graph.get(node, {}).items():
            if not lanes:
                continue
            successor = lanes[0]
            heading = float(successor.heading_theta_at(successor.length))
            delta = (heading - incoming_heading + math.pi) % (2 * math.pi) - math.pi
            if abs(delta) >= math.radians(150):
                continue
            turn = "straight" if abs(delta) <= math.radians(20) else (
                "left" if delta > 0 else "right"
            )
            turns.add(turn)
            successors.append((next_node, successor))
        if len(turns) >= 2:
            return RoadDecisionPointV1(
                point_id=str(node),
                distance_m=float(round(distance, 4)),
                available_turns=tuple(turn for turn in ("left", "straight", "right") if turn in turns),
            )
        if len(successors) != 1:
            return None
        node, lane = successors[0]
        distance += float(lane.length)
    return None


def _roundabout_yield_offset(approach, route_lanes, graph):
    """Earliest crossing of the approach by other local lane envelopes.

    A road node is a centerline connector boundary, not a physical stop line.
    In PG roundabouts, circulating/exit curves may cross the inlet approach
    upstream of that boundary. Sample the last 20 m to find a geometric yield
    boundary, excluding same-route edges and nearly parallel traffic.
    """
    excluded = {tuple(lane.index[:2]) for lane in route_lanes}
    samples = np.arange(max(0.0, float(approach.length) - 20.0), float(approach.length) + 1e-6, 0.5)
    for along in samples:
        position = approach.position(float(along), 0.0)
        heading = float(approach.heading_theta_at(float(along)))
        for start, successors in graph.items():
            for end, lanes in successors.items():
                if (start, end) in excluded:
                    continue
                for lane in lanes:
                    longitudinal, lateral = lane.local_coordinates(position)
                    if not 0 <= longitudinal <= lane.length:
                        continue
                    angle = float(lane.heading_theta_at(longitudinal)) - heading
                    if abs(math.atan2(math.sin(angle), math.cos(angle))) < 0.35:
                        continue
                    width = float(lane.width_at(longitudinal)) if hasattr(lane, "width_at") else float(lane.width)
                    approach_width = float(approach.width_at(along)) if hasattr(approach, "width_at") else float(approach.width)
                    if abs(lateral) <= (width + approach_width) / 2:
                        return min(0.0, float(along) - float(approach.length))
    return 0.0


def extract_route_passage(route_lanes, longitudinal, graph, lookahead_range_m):
    """Bounded local topology adapter for intersections and roundabout merges.

    Uses lane surface types and graph connectivity, never map names/seeds.
    Roundabouts are handled as consecutive merge/fork sections so a distant
    exit does not authorize driving beyond the 50 m observation window.
    """
    incoming = {}
    for successors in graph.values():
        for node, lanes in successors.items():
            if lanes:
                incoming[node] = incoming.get(node, 0) + 1

    def curved(lane):
        delta = float(lane.heading_theta_at(lane.length)) - float(lane.heading_theta_at(0))
        return abs(math.atan2(math.sin(delta), math.cos(delta))) > 0.1

    starts = []
    distance = -float(longitudinal)
    for lane in route_lanes:
        starts.append(distance)
        distance += float(lane.length)
    for i, lane in enumerate(route_lanes):
        if starts[i] > lookahead_range_m:
            break
        kind, end = None, i
        if getattr(lane, "metadrive_type", None) == "LANE_SURFACE_UNSTRUCTURE":
            kind = "intersection"
            while end + 1 < len(route_lanes) and getattr(
                    route_lanes[end + 1], "metadrive_type", None) == "LANE_SURFACE_UNSTRUCTURE":
                end += 1
        elif curved(lane) and incoming.get(lane.index[1], 0) > 1:
            # Inlet curve joins circulating traffic, followed by the local
            # circulation section up to the next fork. Do not reserve a whole
            # roundabout whose route can extend beyond sensor range.
            kind = "roundabout_merge"
            while end + 1 < len(route_lanes) and curved(route_lanes[end + 1]):
                end += 1
                if len(graph.get(route_lanes[end].index[1], {})) > 1:
                    break
        elif curved(lane) and len(graph.get(lane.index[0], {})) > 1:
            # Ordinary straight ramp/diverging roads are not intersections.
            # Reserving their entire (possibly >50 m) road creates a false
            # permanent stop. Lane changes/merge maneuvers remain P4; P2 still
            # evaluates their actual traffic. Curved circulation forks are
            # local roundabout sections that do require branch commitment.
            kind = "fork"
        if kind is None:
            continue
        exit_distance = starts[end] + float(route_lanes[end].length)
        if exit_distance <= 0:
            continue
        delta = float(route_lanes[end].heading_theta_at(route_lanes[end].length)) - float(lane.heading_theta_at(0))
        delta = math.atan2(math.sin(delta), math.cos(delta))
        maneuver = "straight" if abs(delta) <= math.radians(20) else ("left" if delta > 0 else "right")
        yield_distance, conflict_exit = starts[i], exit_distance
        if kind == "roundabout_merge":
            approach = route_lanes[i - 1] if i > 0 else None
            if approach is not None:
                yield_distance += _roundabout_yield_offset(approach, route_lanes, graph)
            # Admit into the local merge and a car-length of circulation space;
            # the whole roundabout is not an intersection box to reserve at once.
            # Branch commitment still lasts through the circulation fork.
            conflict_exit = starts[i] + float(lane.length)
        return RoutePassageV1(
            passage_id=str(lane.index[0]),
            branch_id="|".join(str(tuple(item.index)) for item in route_lanes[i:end + 1]),
            kind=kind, entry_distance_m=round(starts[i], 4),
            exit_distance_m=round(exit_distance, 4) if exit_distance <= lookahead_range_m else None,
            maneuver=maneuver,
            yield_distance_m=round(yield_distance, 4),
            conflict_exit_distance_m=round(conflict_exit, 4) if conflict_exit <= lookahead_range_m else None,
        )
    return None


def extract_ego_motion(ego_vehicle, navigation):
    if not all(hasattr(ego_vehicle, name) for name in ("last_position", "last_heading_dir", "velocity")):
        return None
    theta = float(ego_vehicle.heading_theta)
    c, s = math.cos(theta), math.sin(theta)
    dx, dy = np.asarray(ego_vehicle.position)[:2] - np.asarray(ego_vehicle.last_position)[:2]
    vx, vy = np.asarray(ego_vehicle.velocity)[:2]
    previous = ego_vehicle.last_heading_dir
    delta = theta - math.atan2(float(previous[1]), float(previous[0]))
    lane = navigation.current_lane
    longitudinal, _ = lane.local_coordinates(ego_vehicle.position)
    error = float(lane.heading_theta_at(longitudinal)) - theta
    return EgoMotionV1(
        float(dx * c + dy * s), float(-dx * s + dy * c),
        math.atan2(math.sin(delta), math.cos(delta)),
        float(vx * c + vy * s), float(-vx * s + vy * c),
        math.atan2(math.sin(error), math.cos(error)),
    )


def extract_lane_context(navigation, ego_vehicle, road_network):
    """PG same-edge lane geometry within [-15,50] m of ego's projection.

    Shared boundary ownership follows PGBlock.create_in_world (left sides plus
    the outer right side). Unknown/missing geometry cannot grant permission.
    Route viability checks only the next checkpoint edge within 50 m.
    """
    current = navigation.current_lane
    if current is None or not hasattr(current, "index"):
        return None
    edge = tuple(current.index[:2])
    lanes = road_network.graph.get(edge[0], {}).get(edge[1], ())
    index = next((i for i, lane in enumerate(lanes) if lane is current), None)
    if index is None:
        return None
    theta = float(ego_vehicle.heading_theta)
    c, s = math.cos(theta), math.sin(theta)
    position = np.asarray(ego_vehicle.position)[:2]
    checkpoints = navigation.checkpoints
    ckpt = navigation._target_checkpoints_index[0]
    route_index = next((i for i in range(max(0, ckpt - 1), len(checkpoints) - 1)
                        if tuple(checkpoints[i:i + 2]) == edge), None)
    successors = None
    if route_index is not None and route_index + 2 < len(checkpoints):
        successors = road_network.graph.get(edge[1], {}).get(checkpoints[route_index + 2], ())

    def boundary(owner, side):
        types = getattr(owner, "line_types", ())
        colors = getattr(owner, "line_colors", ())
        if len(types) != 2 or len(colors) != 2:
            return "UNKNOWN"
        color = tuple(colors[side])
        white = len(color) >= 3 and all(abs(v - 1.0) < 1e-4 for v in color[:3])
        if types[side] in {"ROAD_EDGE_BOUNDARY", "GUARDRAIL"}:
            return "EDGE"
        if types[side] == "ROAD_LINE_BROKEN_SINGLE_WHITE" and white:
            return "BROKEN_WHITE"
        if types[side] == "ROAD_LINE_SOLID_SINGLE_WHITE":
            return "SOLID_WHITE" if white else "SOLID_YELLOW"
        return "UNKNOWN"

    exported = []
    for i in range(max(0, index - 1), min(len(lanes), index + 2)):
        lane = lanes[i]
        long_s, _ = lane.local_coordinates(position)
        width = float(lane.width_at(long_s)) if hasattr(lane, "width_at") else float(getattr(lane, "width", 3.5))
        if not 0 < width <= 6:
            return None
        start, end = max(-15.0, -long_s), min(50.0, float(lane.length) - long_s)
        if end - start < 0.1:
            return None
        distances = sorted(set([start, end] + [float(k) for k in np.arange(-15, 50.001, 2.5) if start < k < end]))
        points = []
        for distance in distances:
            pt = np.asarray(lane.position(long_s + distance, 0.0))[:2] - position
            heading = float(lane.heading_theta_at(long_s + distance)) - theta
            points.append((round(distance, 4), round(float(pt[0] * c + pt[1] * s), 4),
                           round(float(-pt[0] * s + pt[1] * c), 4), round(math.atan2(math.sin(heading), math.cos(heading)), 5)))
            # A lane endpoint can lie a few micrometers from the regular grid.
            # Deduplicate AFTER serialization precision, preserving its endpoint.
            if len(points) >= 2 and points[-1][0] == points[-2][0]:
                points[-2:] = [points[-1]]
        compatible = None
        if route_index is not None:
            compatible = True
            if float(lane.length) - long_s <= 50 and successors is not None:
                endpoint = np.asarray(lane.position(lane.length, 0.0))[:2]
                heading = float(lane.heading_theta_at(lane.length))
                compatible = any(np.linalg.norm(np.asarray(n.position(0, 0))[:2] - endpoint) <= 0.3
                                 and abs(math.atan2(math.sin(n.heading_theta_at(0) - heading),
                                                    math.cos(n.heading_theta_at(0) - heading))) < 0.35
                                 for n in successors)
        exported.append(LaneGeometryV1(
            lane_id="|".join(map(str, lane.index)), side="CURRENT" if i == index else ("LEFT" if i < index else "RIGHT"),
            width_m=round(width, 4), remaining_m=round(max(0.0, end), 4),
            left_boundary=boundary(lane, 0),
            right_boundary=boundary(lanes[i + 1], 0) if i + 1 < len(lanes) else boundary(lane, 1),
            route_compatible=compatible, points=tuple(points)))
    current_public = next(lane for lane in exported if lane.side == "CURRENT")
    required = None
    if current_public.route_compatible is False:
        viable = [lane for lane in exported if lane.side != "CURRENT" and lane.route_compatible is True]
        if viable:
            required = viable[0].side
    return LaneContextV1(tuple(exported), required)


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

    # Anchor the reference to the actual current lane, not lanes[0]. Keep
    # geometric continuity at each route connector (indices may change when
    # the number of lanes changes). This is shared input for every agent.
    ckpt_idx = navigation._target_checkpoints_index[0]
    checkpoints = navigation.checkpoints
    route_lanes = [current_lane] if current_lane is not None else []
    reference_end_reachable = False
    if current_lane is not None and checkpoints and len(checkpoints) >= 2:
        current_edge = tuple(current_lane.index[:2])
        current_route_index = next((i for i in range(max(0, ckpt_idx - 1), len(checkpoints) - 1)
                                    if tuple(checkpoints[i:i + 2]) == current_edge), None)
        if current_route_index is not None:
            for i in range(current_route_index + 1, len(checkpoints) - 1):
                u, v = checkpoints[i], checkpoints[i + 1]
                lanes = road_network.graph.get(u, {}).get(v, ())
                if not lanes:
                    break
                previous = route_lanes[-1]
                endpoint = np.asarray(previous.position(previous.length, 0.0))[:2]
                previous_heading = float(previous.heading_theta_at(previous.length))

                def continuity_cost(candidate):
                    start = np.asarray(candidate.position(0.0, 0.0))[:2]
                    angle = float(candidate.heading_theta_at(0.0)) - previous_heading
                    angle = math.atan2(math.sin(angle), math.cos(angle))
                    return float(np.linalg.norm(start - endpoint)) + abs(angle) * 2.0

                successor = min(lanes, key=continuity_cost)
                gap = float(np.linalg.norm(np.asarray(successor.position(0.0, 0.0))[:2] - endpoint))
                # A disconnected lane does not authorize an instantaneous lane
                # change. Stop exporting before the gap rather than teleport.
                if gap > 0.3:
                    break
                route_lanes.append(successor)
            reference_end_reachable = tuple(route_lanes[-1].index[:2]) == tuple(checkpoints[-2:])

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
            route_ended = reference_end_reachable
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
        route_end_within_lookahead=route_ended,
        decision_point=extract_road_decision_point(
            ego_vehicle, road_network, lookahead_count * lookahead_spacing_m
        ),
        route_passage=extract_route_passage(
            route_lanes, long_s, road_network.graph, lookahead_count * lookahead_spacing_m
        ),
        lane_context=extract_lane_context(navigation, ego_vehicle, road_network),
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
        actor_paths, paths_overflow = extract_actor_motion_paths(ego_vehicle, traffic_ctx, road_network)
        traffic_ctx = replace(traffic_ctx, motion_paths=actor_paths, motion_paths_overflow_count=paths_overflow)
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
        step_index=step_index,
        ego_motion=extract_ego_motion(ego_vehicle, navigation) if profile_id == InputProfileId.STATE_DECISION_V1 else None,
    )
