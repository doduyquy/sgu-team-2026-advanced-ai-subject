"""Ego-relative route lattice and A*: only public AgentInput is used.

This is a local geometric planning approximation, not access to the evaluator's
map graph, live simulator or future ground-truth trajectory.
"""

from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

from src.platform.agent_input import AgentInputV1
from src.platform import canonical_json_sha256

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/experiments/astar_learning_v1.json"
FEATURE_NAMES = (
    "euclidean_to_goal", "remaining_reference_length", "goal_dx", "goal_dy",
    "depth_fraction", "lateral_offset", "heading_cos", "heading_sin",
    "remaining_curvature", "future_blocked_fraction", "nearest_actor_distance", "out_degree",
)


def load_astar_config():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["version"] != "astar_heuristic_v1":
        raise ValueError("Unsupported A* configuration version")
    return config


def planner_fingerprint(config):
    return canonical_json_sha256(dict(version="local_route_lattice_v1", planner=config,
                                      features=list(FEATURE_NAMES)))


@dataclass(frozen=True)
class PlanningGraph:
    points: np.ndarray
    edges: Tuple[Tuple[Tuple[int, float], ...], ...]
    features: np.ndarray
    goal: int
    scale_m: float

    def euclidean_heuristic(self):
        return np.linalg.norm(self.points - self.points[self.goal], axis=1)


@dataclass(frozen=True)
class SearchResult:
    path: Tuple[int, ...]
    cost: float
    expansions: int
    found: bool


def build_planning_graph(agent_input: AgentInputV1, config=None) -> Optional[PlanningGraph]:
    config = config or load_astar_config()["planner"]
    task = agent_input.task_context
    reference = task.waypoints_array[task.validity_mask][:config["lookahead_points"]].astype(np.float64)
    if not len(reference):
        return None
    actors = agent_input.traffic_context.actors_array[agent_input.traffic_context.validity_mask]
    # Keep all samples inside the public reference lane, reserving vehicle clearance.
    lateral_limit = max(0.0, task.current_lane_width / 2 - 0.85)
    offsets = np.clip(np.asarray(config["lateral_samples"], dtype=float), -lateral_limit, lateral_limit)
    if len(offsets) != 3 or abs(offsets[1]) > 1e-9:
        raise ValueError("The shared lattice requires three lateral samples with a central sample")
    points = [np.zeros(2)]
    depths, sides, headings = [0], [1], [0.0]
    for depth, (x, y, heading) in enumerate(reference, 1):
        normal = np.array([-math.sin(heading), math.cos(heading)])
        for side, offset in enumerate(offsets):
            points.append(np.array([x, y]) + offset * normal)
            depths.append(depth)
            sides.append(side)
            headings.append(heading)
    points = np.asarray(points)
    depths = np.asarray(depths)
    sides = np.asarray(sides)
    goal = 1 + (len(reference) - 1) * 3 + 1
    margin = config["obstacle_margin_m"]

    def blocked(sample_points):
        for actor in actors:
            dx = sample_points[:, 0] - actor[0]
            dy = sample_points[:, 1] - actor[1]
            c, s = math.cos(actor[4]), math.sin(actor[4])
            along, across = c * dx + s * dy, -s * dx + c * dy
            if np.any((np.abs(along) <= actor[5] / 2 + margin)
                      & (np.abs(across) <= actor[6] / 2 + margin)):
                return True
        return False

    occupancy = np.array([blocked(point[None, :]) for point in points])
    nearest = np.min(np.linalg.norm(points[:, None, :] - actors[None, :, :2], axis=2), axis=1) if len(actors) else np.full(len(points), 50.)
    edges = []
    for index, point in enumerate(points):
        successors = []
        if depths[index] < len(reference) and not occupancy[index]:
            next_depth = depths[index] + 1
            for side in range(3):
                if index != 0 and abs(side - sides[index]) > 1:
                    continue
                neighbor = 1 + (next_depth - 1) * 3 + side
                if occupancy[neighbor] or blocked(np.linspace(point, points[neighbor], 5)):
                    continue
                distance = float(np.linalg.norm(points[neighbor] - point))
                cost = (max(distance, 1e-3)
                        + config["lateral_change_cost"] * abs(float(offsets[side] - offsets[sides[index]]))
                        + config["clearance_cost"] / (1 + nearest[neighbor]))
                successors.append((neighbor, cost))
        edges.append(tuple(successors))
    centers = np.vstack((np.zeros((1, 2)), reference[:, :2]))
    segment_lengths = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    remaining_lengths = np.r_[np.cumsum(segment_lengths[::-1])[::-1], 0.]
    curvature = np.abs(np.angle(np.exp(1j * np.diff(np.r_[0., reference[:, 2]]))))
    remaining_curvature = np.r_[np.cumsum(curvature[::-1])[::-1], 0.]
    scale = float(config["feature_scale_m"])
    features = []
    for index, point in enumerate(points):
        delta = points[goal] - point
        future = occupancy[depths >= depths[index]]
        features.append((np.linalg.norm(delta) / scale, remaining_lengths[depths[index]] / scale,
                         delta[0] / scale, delta[1] / scale, depths[index] / len(reference),
                         offsets[sides[index]] / task.current_lane_width,
                         math.cos(headings[index]), math.sin(headings[index]),
                         remaining_curvature[depths[index]] / math.pi,
                         float(np.mean(future)), min(nearest[index], scale) / scale, len(edges[index]) / 3))
    features = np.asarray(features, dtype=np.float64)
    points.flags.writeable = features.flags.writeable = False
    return PlanningGraph(points, tuple(edges), features, goal, scale)


def astar_search(graph: PlanningGraph, heuristic=None, max_expansions=256) -> SearchResult:
    h = graph.euclidean_heuristic() if heuristic is None else np.asarray(heuristic, dtype=float).copy()
    if h.shape != (len(graph.points),) or not np.all(np.isfinite(h)) or np.any(h < 0):
        raise ValueError("Heuristics must be finite nonnegative estimates for all nodes")
    h[graph.goal] = 0.0
    distances = np.full(len(h), np.inf)
    distances[0] = 0.0
    parents = {}
    frontier = [(float(h[0]), 0.0, 0)]
    expansions = 0
    while frontier and expansions < max_expansions:
        _, cost, node = heapq.heappop(frontier)
        if cost > distances[node] + 1e-10:
            continue
        expansions += 1
        if node == graph.goal:
            path = [node]
            while node != 0:
                node = parents[node]
                path.append(node)
            return SearchResult(tuple(reversed(path)), cost, expansions, True)
        for neighbor, edge_cost in graph.edges[node]:
            candidate = cost + edge_cost
            if candidate + 1e-10 < distances[neighbor]:
                distances[neighbor] = candidate
                parents[neighbor] = node
                heapq.heappush(frontier, (candidate + h[neighbor], candidate, neighbor))
    return SearchResult((), float("inf"), expansions, False)


def exact_cost_to_goal(graph: PlanningGraph):
    """Supervised labels on the local graph, never called by inference agents."""
    predecessors = [[] for _ in graph.edges]
    for node, edges in enumerate(graph.edges):
        for neighbor, cost in edges:
            predecessors[neighbor].append((node, cost))
    distances = np.full(len(graph.edges), np.inf)
    distances[graph.goal] = 0.0
    frontier = [(0.0, graph.goal)]
    while frontier:
        cost, node = heapq.heappop(frontier)
        if cost > distances[node] + 1e-10:
            continue
        for previous, edge_cost in predecessors[node]:
            candidate = cost + edge_cost
            if candidate + 1e-10 < distances[previous]:
                distances[previous] = candidate
                heapq.heappush(frontier, (candidate, previous))
    return distances
