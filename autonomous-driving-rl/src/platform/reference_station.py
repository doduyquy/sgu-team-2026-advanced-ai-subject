"""F09 producer proposal. Kept outside the canonical AgentInputV1 pipeline."""
from dataclasses import replace
import math

import numpy as np

from src.platform.agent_input import RouteWaypointV1, TaskContextV1


PROPOSAL_ID = "reference_route_station_proposal_v1"
SEAM_TOLERANCE_M = 0.001


def _locate(navigation, graph, ego_position):
    """Use lane identity near the checkpoint, so nearby crossing roads cannot win."""
    checkpoints = getattr(navigation, "checkpoints", None)
    target = getattr(navigation, "_target_checkpoints_index", None)
    current = getattr(navigation, "current_lane", None)
    if checkpoints is None or len(checkpoints) < 2 or target is None or len(target) < 1 or current is None:
        return None
    edges = list(zip(checkpoints, checkpoints[1:]))
    checkpoint = int(target[0])
    if not 0 <= checkpoint < len(edges):
        return None
    groups = [graph.get(start, {}).get(end, ()) for start, end in edges]
    nearby = range(max(0, checkpoint - 1), min(len(groups), checkpoint + 2))
    anchors = [i for i in nearby if any(lane is current for lane in groups[i])]
    if len(anchors) != 1:
        return None
    route = []
    for group in groups:
        if not group:
            return None
        ref = group[0]
        if not math.isfinite(ref.length) or ref.length <= 0:
            return None
        if route:
            tip = np.asarray(route[-1].position(route[-1].length, 0))
            start = np.asarray(ref.position(0, 0))
            if not np.linalg.norm(tip - start) <= SEAM_TOLERANCE_M:
                return None
        route.append(ref)
    anchor = anchors[0]
    station, _ = route[anchor].local_coordinates(ego_position)
    if not math.isfinite(station):
        return None
    return route, sum(ref.length for ref in route[:anchor]) + float(station)


def corrected_task_context(navigation, ego_vehicle, road_network, task: TaskContextV1) -> TaskContextV1:
    """Rebuild only the existing waypoint triples, masks and lookahead-end flag.

    Called before policy encoding in the proposal audit. World pose and lane
    geometry are transient producer inputs and are not returned.
    """
    pos = np.asarray(ego_vehicle.position[:2], dtype=float)
    yaw = float(ego_vehicle.heading_theta)
    cs, sn = math.cos(yaw), math.sin(yaw)
    located = _locate(navigation, road_network.graph, pos)
    waypoints, mask = [], []
    for i in range(1, task.lookahead_count + 1):
        distance = i * task.lookahead_spacing_m
        relative = (0.0, 0.0, 0.0)
        valid = False
        if located is not None:
            refs, station = located
            remaining = station + distance
            for ref in refs:
                if remaining < 0:
                    break
                if remaining <= ref.length:
                    dx, dy = np.asarray(ref.position(remaining, 0.0)) - pos
                    heading = (ref.heading_theta_at(remaining) - yaw + math.pi) % (2 * math.pi) - math.pi
                    relative = (round(float(dx * cs + dy * sn), 4),
                                round(float(-dx * sn + dy * cs), 4), round(float(heading), 4))
                    valid = True
                    break
                remaining -= ref.length
        waypoints.append(RouteWaypointV1(round(distance, 2), *relative))
        mask.append(valid)
    return replace(task, waypoints=tuple(waypoints),
                   waypoints_array=np.asarray([wp.to_tuple() for wp in waypoints], dtype=np.float32),
                   validity_mask=np.asarray(mask, dtype=bool),
                   route_end_within_lookahead=(located is not None and
                                               located[1] + task.lookahead_range_m >
                                               sum(ref.length for ref in located[0])))
