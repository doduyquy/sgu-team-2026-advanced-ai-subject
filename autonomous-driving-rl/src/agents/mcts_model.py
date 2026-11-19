"""Local forward model for MCTS; consumes public observations, never live MetaDrive.

The frame is frozen at the start of each act(): +x forward, +y left. Actors
have constant velocity in this frame. Vehicle dynamics, corridor and LiDAR
occupancy are approximations, not a clone of the simulator or a safety proof.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

from src.platform.action_adapter import get_action_adapter


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/experiments/mcts_v1.json"
ACTION_GRID = tuple(get_action_adapter("discrete9_lowbranch_v1").GRID)
FAILURE_REASONS = frozenset({"COLLISION", "OUTSIDE_CORRIDOR"})


@dataclass(frozen=True)
class MCTSConfig:
    num_iterations: int = 64
    max_depth: int = 20
    gamma: float = 0.98
    exploration_constant: float = 1.5
    rollout_random_probability: float = 0.1
    integration_dt_s: float = 0.02
    max_speed_kmh: float = 80.0
    target_speed_m_s: float = 5.0
    wheelbase_m: float = 2.5
    max_steering_deg: float = 40.0
    steering_sign: float = 1.0
    acceleration_m_s2: float = 2.8
    braking_m_s2: float = 5.0
    drag_per_s: float = 0.05
    vehicle_length_m: float = 4.6
    vehicle_width_m: float = 1.9
    lane_margin_m: float = 0.1
    obstacle_margin_m: float = 0.25
    lidar_range_m: float = 50.0
    control_lookahead_m: float = 5.0
    progress_weight: float = 1.0
    lateral_weight: float = 0.5
    heading_weight: float = 0.2
    speed_weight: float = 0.1
    steering_change_weight: float = 0.04
    time_weight: float = 0.02
    collision_penalty: float = 40.0
    outside_penalty: float = 30.0

    def __post_init__(self):
        for name in ("num_iterations", "max_depth"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_iterations < len(ACTION_GRID):
            raise ValueError("num_iterations must allow all nine root actions to be tried")
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid MCTS setting: {name}")
            if name != "steering_sign" and value < 0:
                raise ValueError(f"Negative MCTS setting: {name}")
        for name in ("integration_dt_s", "max_speed_kmh", "target_speed_m_s", "wheelbase_m",
                     "vehicle_length_m", "vehicle_width_m", "lidar_range_m", "control_lookahead_m"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.gamma <= 1 or not 0 <= self.rollout_random_probability <= 1:
            raise ValueError("Invalid discount or rollout probability")
        if self.steering_sign not in (-1., 1.) or not 0 < self.max_steering_deg < 90:
            raise ValueError("Invalid steering calibration")
        if self.target_speed_m_s > self.max_speed_kmh / 3.6:
            raise ValueError("Target speed exceeds model maximum speed")


def load_mcts_config():
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if payload["version"] != "mcts_planner_v1":
        raise ValueError("Unsupported MCTS configuration")
    return MCTSConfig(**payload["parameters"])


@dataclass(frozen=True)
class PlanningState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0
    time_s: float = 0.0
    last_steering: float = 0.0


@dataclass(frozen=True)
class ModelTransition:
    state: PlanningState
    reward: float
    terminal: bool
    reason: str = "RUNNING"


class PlanningScene:
    """Immutable snapshot: route polyline, moving actor boxes, residual LiDAR hits."""

    def __init__(self, planning_input, config):
        self.config = config
        route = np.asarray(planning_input["waypoints"], dtype=float)
        if not len(route):
            raise ValueError("No public route")
        centers = np.vstack((np.zeros((1, 2)), route[:, :2]))
        keep = np.r_[True, np.linalg.norm(np.diff(centers, axis=0), axis=1) > 1e-7]
        centers = centers[keep]
        
        if len(centers) < 2:
            raise ValueError("No nonzero public route segment")
        
        # Extend the corridor for the body overhang; this does not extend the
        # observed planning horizon or declare a destination beyond observations.
        self.route_length = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
        extension = config.vehicle_length_m
        first, last = centers[1] - centers[0], centers[-1] - centers[-2]
        centers = np.vstack((centers[0] - extension * first / np.linalg.norm(first), centers,
                              centers[-1] + extension * last / np.linalg.norm(last)))
        self.starts = centers[:-1]
        self.vectors = np.diff(centers, axis=0)
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        self.cumulative = np.r_[0., np.cumsum(self.lengths)]
        self.progress_offset = extension
        self.half_lane = float(planning_input["lane_width"]) / 2 - config.lane_margin_m
        if self.half_lane <= 0:
            raise ValueError("No usable public lane corridor")
        self.actors = np.asarray(planning_input["actors"], dtype=float).reshape((-1, 7)).copy()
        
        # Input velocities are relative: add the initial ego velocity once.
        # Neglecting ego lateral slip is an explicit model approximation.
        
        self.actor_velocities = self.actors[:, 2:4].copy()
        self.actor_velocities[:, 0] += planning_input["speed_m_s"]
        self.actor_cos = np.cos(self.actors[:, 4])
        self.actor_sin = np.sin(self.actors[:, 4])

        rays = np.asarray(planning_input["lidar"], dtype=float)
        valid = rays < 1.0 - 1e-6  # 1 = no hit, not an obstacle at the range limit.
        angles = np.arange(len(rays)) * (2 * math.pi / len(rays))
        hits = config.lidar_range_m * rays[valid, None] * np.column_stack((np.cos(angles[valid]), np.sin(angles[valid])))
        
        # Remove hits explained by tracked cars; otherwise a moving car would
        # also leave a stationary 'ghost' in this snapshot's obstacle cloud.
        
        explained = np.zeros(len(hits), dtype=bool)
        for actor, c, s in zip(self.actors, self.actor_cos, self.actor_sin):
            delta = hits - actor[:2]
            along = delta[:, 0] * c + delta[:, 1] * s
            across = -delta[:, 0] * s + delta[:, 1] * c
            explained |= ((np.abs(along) <= actor[5] / 2 + .5)
                          & (np.abs(across) <= actor[6] / 2 + .5))
        self.static_hits = hits[~explained].copy()
        for array in (self.starts, self.vectors, self.lengths, self.cumulative, self.actors,
                      self.actor_velocities, self.actor_cos, self.actor_sin, self.static_hits):
            array.flags.writeable = False

    def project(self, points):
        """Return polyline progress (m), distance (m), tangent heading (rad)."""
        points = np.asarray(points, dtype=float).reshape((-1, 2))
        delta = points[:, None, :] - self.starts[None, :, :]
        fraction = np.clip(np.sum(delta * self.vectors, axis=2) / self.lengths ** 2, 0., 1.)
        residual = delta - fraction[:, :, None] * self.vectors
        squared_distance = np.sum(residual ** 2, axis=2)
        index = np.argmin(squared_distance, axis=1)
        rows = np.arange(len(points))
        progress = self.cumulative[index] + fraction[rows, index] * self.lengths[index] - self.progress_offset
        heading = np.arctan2(self.vectors[index, 1], self.vectors[index, 0])
        return progress, np.sqrt(squared_distance[rows, index]), heading

    def first_failure(self, states):
        """Check sampled body footprints; return earliest failure index and reason."""
        positions = np.array([(state.x, state.y) for state in states])
        yaw = np.array([state.yaw for state in states])
        times = np.array([state.time_s for state in states])
        c, s = np.cos(yaw), np.sin(yaw)
        half_l = self.config.vehicle_length_m / 2
        half_w = self.config.vehicle_width_m / 2
        along = np.column_stack((c, s)) * half_l
        across = np.column_stack((-s, c)) * half_w
        corners = np.stack([positions + a * along + b * across
                            for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1))], axis=1)
        distances = self.project(corners.reshape((-1, 2)))[1].reshape((-1, 4))
        outside = np.any(distances > self.half_lane, axis=1)
        collision = np.zeros(len(states), dtype=bool)
        if len(self.actors):
            actor_positions = self.actors[None, :, :2] + times[:, None, None] * self.actor_velocities
            delta = actor_positions - positions[:, None, :]
            dx, dy = delta[:, :, 0], delta[:, :, 1]
            relative_c = np.abs(c[:, None] * self.actor_cos + s[:, None] * self.actor_sin)
            relative_s = np.abs(s[:, None] * self.actor_cos - c[:, None] * self.actor_sin)
            actor_l = self.actors[:, 5] / 2 + self.config.obstacle_margin_m
            actor_w = self.actors[:, 6] / 2 + self.config.obstacle_margin_m
            overlap = ((np.abs(dx * c[:, None] + dy * s[:, None]) <= half_l + actor_l * relative_c + actor_w * relative_s)
                       & (np.abs(-dx * s[:, None] + dy * c[:, None]) <= half_w + actor_l * relative_s + actor_w * relative_c)
                       & (np.abs(dx * self.actor_cos + dy * self.actor_sin) <= actor_l + half_l * relative_c + half_w * relative_s)
                       & (np.abs(-dx * self.actor_sin + dy * self.actor_cos) <= actor_w + half_l * relative_s + half_w * relative_c))
            collision |= np.any(overlap, axis=1)
        if len(self.static_hits):
            delta = self.static_hits[None, :, :] - positions[:, None, :]
            local_x = delta[:, :, 0] * c[:, None] + delta[:, :, 1] * s[:, None]
            local_y = -delta[:, :, 0] * s[:, None] + delta[:, :, 1] * c[:, None]
            collision |= np.any((np.abs(local_x) <= half_l + self.config.obstacle_margin_m)
                                & (np.abs(local_y) <= half_w + self.config.obstacle_margin_m), axis=1)
        invalid = np.flatnonzero(collision | outside)
        if not len(invalid):
            return None, "RUNNING"
        index = int(invalid[0])
        return index, "COLLISION" if collision[index] else "OUTSIDE_CORRIDOR"

    def guided_action(self, state):
        """A small pure-pursuit/speed policy for rollout continuation, not the root decision."""
        config = self.config
        progress = float(self.project([[state.x, state.y]])[0][0])
        target_s = min(self.progress_offset + self.route_length,
                       max(self.progress_offset, progress + self.progress_offset + config.control_lookahead_m))
        index = min(int(np.searchsorted(self.cumulative, target_s, side="right") - 1), len(self.lengths) - 1)
        target = self.starts[index] + (target_s - self.cumulative[index]) / self.lengths[index] * self.vectors[index]
        dx, dy = target - np.array([state.x, state.y])
        angle = (math.atan2(float(dy), float(dx)) - state.yaw + math.pi) % (2 * math.pi) - math.pi
        steering = math.atan2(2 * config.wheelbase_m * math.sin(angle), max(math.hypot(dx, dy), 1.))
        command = steering / (math.radians(config.max_steering_deg) * config.steering_sign)
        steer_index = 2 if command > .15 else 0 if command < -.15 else 1
        target_speed = config.target_speed_m_s * (0.6 if abs(angle) > .3 else 1.)
        brake_for_actor = False
        for actor, velocity in zip(self.actors, self.actor_velocities):
            delta = actor[:2] + state.time_s * velocity - np.array([state.x, state.y])
            forward = delta[0] * math.cos(state.yaw) + delta[1] * math.sin(state.yaw)
            lateral = -delta[0] * math.sin(state.yaw) + delta[1] * math.cos(state.yaw)
            if forward <= 0 or abs(lateral) > (actor[6] + config.vehicle_width_m) / 2 + .3:
                continue
            gap = forward - (actor[5] + config.vehicle_length_m) / 2
            closing = max(0., state.speed - float(velocity @ [math.cos(state.yaw), math.sin(state.yaw)]))
            brake_for_actor |= gap < 2. + closing ** 2 / (2 * max(.1, .8 * config.braking_m_s2))
        throttle_index = 0 if brake_for_actor or state.speed > target_speed + .3 else 1 if state.speed > target_speed else 2
        return 3 * throttle_index + steer_index


def predict_step(scene, state, action_id, dt):
    """Time-based bicycle integration + sampled collisions + local planning score."""
    if type(action_id) is not int or not 0 <= action_id < len(ACTION_GRID):
        raise ValueError("Expected action index 0..8")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    config = scene.config
    steering, throttle = ACTION_GRID[action_id]
    curvature = math.tan(config.steering_sign * steering * math.radians(config.max_steering_deg)) / config.wheelbase_m
    count = max(1, math.ceil(dt / config.integration_dt_s))
    h = dt / count
    x, y, yaw, speed = state.x, state.y, state.yaw, state.speed
    samples = []
    for index in range(count):
        acceleration = (config.acceleration_m_s2 * max(throttle, 0.)
                        - config.braking_m_s2 * max(-throttle, 0.) - config.drag_per_s * speed)
        active_h = min(h, speed / -acceleration) if acceleration < 0 else h
        distance = max(0., speed * active_h + .5 * acceleration * active_h ** 2)
        speed = min(config.max_speed_kmh / 3.6, max(0., speed + acceleration * h))
        next_yaw = yaw + distance * curvature
        if abs(curvature) < 1e-10:
            x += distance * math.cos(yaw)
            y += distance * math.sin(yaw)
        else:
            x += (math.sin(next_yaw) - math.sin(yaw)) / curvature
            y += (math.cos(yaw) - math.cos(next_yaw)) / curvature
        yaw = (next_yaw + math.pi) % (2 * math.pi) - math.pi
        samples.append(PlanningState(x, y, yaw, speed, state.time_s + (index + 1) * h, steering))
    failure_index, reason = scene.first_failure(samples)
    next_state = samples[-1] if failure_index is None else samples[failure_index]
    progress, lateral, heading = scene.project([[state.x, state.y], [next_state.x, next_state.y]])
    
    # Nearest-segment projection can jump at crossings; never reward more than
    # the distance the model could actually have travelled during this step.
    
    elapsed = next_state.time_s - state.time_s
    travelled_bound = max(state.speed, next_state.speed) * elapsed + .5 * config.acceleration_m_s2 * elapsed ** 2
    delta_progress = float(np.clip(progress[1] - progress[0], -travelled_bound, travelled_bound))
    heading_error = (next_state.yaw - float(heading[1]) + math.pi) % (2 * math.pi) - math.pi
    reward = (config.progress_weight * delta_progress
              - elapsed * config.lateral_weight * (float(lateral[1]) / scene.half_lane) ** 2
              - elapsed * config.heading_weight * (1 - math.cos(heading_error))
              - elapsed * config.speed_weight * (next_state.speed - config.target_speed_m_s) ** 2
              - config.steering_change_weight * (steering - state.last_steering) ** 2
              - elapsed * config.time_weight)
    if reason == "COLLISION":
        reward -= config.collision_penalty
    elif reason == "OUTSIDE_CORRIDOR":
        reward -= config.outside_penalty
    terminal = failure_index is not None
    if not terminal and progress[1] >= scene.route_length:
        terminal, reason = True, "PUBLIC_LOOKAHEAD_END"  # Not real destination success.
    return ModelTransition(next_state, float(reward), terminal, reason)
