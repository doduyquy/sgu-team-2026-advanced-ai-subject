"""Forward Hybrid A*: continuous bicycle poses, discretized duplicate detection.

The scene comes only from public route waypoints and traffic actor rectangles.
This local approximation has no simulator handle, reverse gear or map oracle.
"""

from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path

import numpy as np

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/experiments/search_planners_v1.json"


def validate_hybrid_config(config):
    positive = ("primitive_length_m", "collision_sample_spacing_m", "xy_resolution_m",
                "goal_position_tolerance_m", "goal_heading_tolerance_deg", "wheelbase_m",
                "max_steering_deg", "vehicle_length_m", "vehicle_width_m")
    for key in positive + ("lane_margin_m", "obstacle_margin_m", "steering_cost_per_m"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid Hybrid A* setting: {key}")
        if value < 0 or (key in positive and value == 0):
            raise ValueError(f"Hybrid A* setting must be positive: {key}")
    if not 0 < config["max_steering_deg"] < 90 or config["goal_heading_tolerance_deg"] > 180:
        raise ValueError("Invalid Hybrid A* steering or goal angle")
    if type(config["heading_bins"]) is not int or not 8 <= config["heading_bins"] <= 720:
        raise ValueError("Invalid Hybrid A* heading discretization")
    levels = config["steering_levels"]
    if not isinstance(levels, list) or not levels or len(levels) > 9 or 0.0 not in levels:
        raise ValueError("Hybrid A* needs a straight primitive and a bounded steering set")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > .6 for v in levels):
        raise ValueError("Hybrid A* steering levels must be finite and within [-0.6, 0.6]")
    if config["collision_sample_spacing_m"] > config["primitive_length_m"]:
        raise ValueError("Collision sampling must cover the motion primitive")
    return config


def load_hybrid_config():
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if data["version"] != "search_planners_v1":
        raise ValueError("Unsupported search-planner configuration")
    return validate_hybrid_config(data["hybrid_astar"])


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def pose_key(pose, config):
    resolution = config["xy_resolution_m"]
    bins = config["heading_bins"]
    return (math.floor(float(pose[0]) / resolution + .5), math.floor(float(pose[1]) / resolution + .5),
            math.floor((float(wrap_angle(pose[2])) + math.pi) * bins / (2 * math.pi) + .5) % bins)


def rollout_primitive(pose, steering, config):
    """Integrate a constant-steering bicycle arc analytically, sampled along its length."""
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (3,) or not np.all(np.isfinite(pose)) or not math.isfinite(steering) or abs(steering) > .6:
        raise ValueError("A motion primitive requires a finite pose and bounded steering")
    length = config["primitive_length_m"]
    distances = np.linspace(0., length, int(math.ceil(length / config["collision_sample_spacing_m"])) + 1)
    curvature = math.tan(steering * math.radians(config["max_steering_deg"])) / config["wheelbase_m"]
    headings = pose[2] + curvature * distances
    if abs(curvature) < 1e-12:
        x = pose[0] + distances * math.cos(pose[2])
        y = pose[1] + distances * math.sin(pose[2])
    else:
        x = pose[0] + (np.sin(headings) - math.sin(pose[2])) / curvature
        y = pose[1] - (np.cos(headings) - math.cos(pose[2])) / curvature
    samples = np.column_stack((x, y, wrap_angle(headings)))
    samples.flags.writeable = False
    return samples


@dataclass(frozen=True)
class HybridSearchResult:
    poses: np.ndarray
    cost: float
    expansions: int
    found: bool
    steering_sequence: tuple = ()

    def __post_init__(self):
        poses = np.array(self.poses, dtype=float, copy=True).reshape((-1, 3))
        poses.flags.writeable = False
        object.__setattr__(self, "poses", poses)


class PublicPlanningScene:
    """A fixed-width route corridor and oriented rectangles, built without private state."""

    def __init__(self, agent_input, planner_config, hybrid_config):
        self.config = hybrid_config
        task = agent_input.task_context
        reference = task.waypoints_array[task.validity_mask][:planner_config["lookahead_points"]]
        if not len(reference):
            raise ValueError("No public route for Hybrid A*")
        self.goal = reference[-1].astype(float)
        self.half_lane = task.current_lane_width / 2 - hybrid_config["lane_margin_m"]
        centers = np.vstack((np.zeros((1, 2)), reference[:, :2]))
        keep = np.r_[True, np.linalg.norm(np.diff(centers, axis=0), axis=1) > 1e-8]
        centers = centers[keep]
        if len(centers) < 2:
            raise ValueError("Public route has no nonzero segment")
        first, last = centers[1] - centers[0], centers[-1] - centers[-2]
        extension = hybrid_config["vehicle_length_m"]
        centers = np.vstack((centers[0] - extension * first / np.linalg.norm(first), centers,
                              centers[-1] + extension * last / np.linalg.norm(last)))
        self.starts = centers[:-1]
        self.vectors = np.diff(centers, axis=0)
        self.squared_lengths = np.sum(self.vectors ** 2, axis=1)
        actors = agent_input.traffic_context.actors_array[agent_input.traffic_context.validity_mask]
        self.actors = actors.astype(float)
        self.actor_cos = np.cos(self.actors[:, 4])
        self.actor_sin = np.sin(self.actors[:, 4])

    def distance_to_route(self, points):
        delta = np.asarray(points)[:, None, :] - self.starts[None, :, :]
        fraction = np.clip(np.sum(delta * self.vectors[None, :, :], axis=2) / self.squared_lengths, 0., 1.)
        offset = delta - fraction[:, :, None] * self.vectors[None, :, :]
        return np.sqrt(np.min(np.sum(offset ** 2, axis=2), axis=1))

    def is_clear(self, poses):
        poses = np.asarray(poses)
        c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        along = np.column_stack((c, s)) * (self.config["vehicle_length_m"] / 2)
        across = np.column_stack((-s, c)) * (self.config["vehicle_width_m"] / 2)
        corners = np.stack([poses[:, :2] + a * along + b * across for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1))], axis=1)
        if np.any(self.distance_to_route(corners.reshape((-1, 2))) > self.half_lane):
            return False
        if not len(self.actors):
            return True
        # SAT: rectangles overlap only if no separating axis exists on either box.
        dx = self.actors[None, :, 0] - poses[:, None, 0]
        dy = self.actors[None, :, 1] - poses[:, None, 1]
        relative_c = np.abs(c[:, None] * self.actor_cos + s[:, None] * self.actor_sin)
        relative_s = np.abs(s[:, None] * self.actor_cos - c[:, None] * self.actor_sin)
        ego_l, ego_w = self.config["vehicle_length_m"] / 2, self.config["vehicle_width_m"] / 2
        actor_l = self.actors[:, 5] / 2 + self.config["obstacle_margin_m"]
        actor_w = self.actors[:, 6] / 2 + self.config["obstacle_margin_m"]
        overlap = ((np.abs(dx * c[:, None] + dy * s[:, None]) <= ego_l + actor_l * relative_c + actor_w * relative_s)
                   & (np.abs(-dx * s[:, None] + dy * c[:, None]) <= ego_w + actor_l * relative_s + actor_w * relative_c)
                   & (np.abs(dx * self.actor_cos + dy * self.actor_sin) <= actor_l + ego_l * relative_c + ego_w * relative_s)
                   & (np.abs(-dx * self.actor_sin + dy * self.actor_cos) <= actor_w + ego_l * relative_s + ego_w * relative_c))
        return not bool(np.any(overlap))

    def at_goal(self, pose):
        return (np.linalg.norm(pose[:2] - self.goal[:2]) <= self.config["goal_position_tolerance_m"]
                and abs(wrap_angle(pose[2] - self.goal[2])) <= math.radians(self.config["goal_heading_tolerance_deg"]))

    def heuristic(self, pose):
        # Distance to the goal region is a lower bound on nonnegative forward arc cost.
        return max(0., float(np.linalg.norm(pose[:2] - self.goal[:2])) - self.config["goal_position_tolerance_m"])


def hybrid_astar_search(agent_input, planner_config, hybrid_config=None, max_expansions=256):
    config = validate_hybrid_config(hybrid_config) if hybrid_config is not None else load_hybrid_config()
    if type(max_expansions) is not int or max_expansions < 0:
        raise ValueError("max_expansions must be a nonnegative integer")
    empty = lambda n: HybridSearchResult(np.empty((0, 3)), float("inf"), n, False)
    task = agent_input.task_context
    if not np.any(task.validity_mask):
        return empty(0)
    reference = task.waypoints_array[task.validity_mask][:planner_config["lookahead_points"]]
    if not len(reference) or np.all(np.linalg.norm(reference[:, :2], axis=1) < 1e-8):
        return empty(0)
    scene = PublicPlanningScene(agent_input, planner_config, config)
    start = np.zeros(3)
    if not scene.is_clear(start[None, :]):
        return empty(0)
    # Retain immutable continuous poses and parent IDs even when a cheaper pose replaces a bin.
    nodes = [(start, 0.0, -1, 0.0)]
    best = {pose_key(start, config): (0.0, 0)}
    frontier = [(scene.heuristic(start), 0.0, 0)]
    expansions = 0
    while frontier and expansions < max_expansions:
        _, cost, index = heapq.heappop(frontier)
        pose, _, parent, _ = nodes[index]
        if best.get(pose_key(pose, config), (None, None))[1] != index:
            continue
        expansions += 1
        if scene.at_goal(pose):
            chain = []
            while nodes[index][2] != -1:
                chain.append(index)
                index = nodes[index][2]
            samples, steering_sequence = [start[None, :]], []
            for index in reversed(chain):
                _, _, parent, steering = nodes[index]
                samples.append(rollout_primitive(nodes[parent][0], steering, config)[1:])
                steering_sequence.append(steering)
            return HybridSearchResult(np.vstack(samples), cost, expansions, True, tuple(steering_sequence))
        for steering in config["steering_levels"]:
            arc = rollout_primitive(pose, steering, config)
            if not scene.is_clear(arc[1:]):
                continue
            candidate = arc[-1]
            key = pose_key(candidate, config)
            next_cost = cost + config["primitive_length_m"] * (1 + config["steering_cost_per_m"] * abs(steering))
            if next_cost + 1e-10 < best.get(key, (float("inf"), None))[0]:
                child = len(nodes)
                nodes.append((candidate, next_cost, index, steering))
                best[key] = (next_cost, child)
                heapq.heappush(frontier, (next_cost + scene.heuristic(candidate), next_cost, child))
    return empty(expansions)
