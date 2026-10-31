"""Local corridor projection and deterministic traffic/obstacle rules.

Only public snapshots are consumed. Constant-velocity prediction is a bounded
safety heuristic, not an oracle for other vehicles' future decisions.
"""

import math
from dataclasses import dataclass

import numpy as np


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class RouteCorridor:
    def __init__(self, waypoints, heading_error, lateral_right_m):
        anchor = (-math.sin(heading_error) * lateral_right_m,
                  math.cos(heading_error) * lateral_right_m)
        points = [anchor]
        for waypoint in waypoints:
            candidate = (waypoint.relative_x, waypoint.relative_y)
            if np.linalg.norm(np.asarray(candidate) - points[-1]) > 1e-4:
                points.append(candidate)
        if len(points) < 2:
            points.append((anchor[0] + math.cos(heading_error),
                           anchor[1] + math.sin(heading_error)))
        self.points = np.asarray(points, dtype=float)
        self.vectors = np.diff(self.points, axis=0)
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        self.arc = np.r_[0.0, np.cumsum(self.lengths)]
        self.angles = np.arctan2(self.vectors[:, 1], self.vectors[:, 0])

    def project(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        delta = points[:, None, :] - self.points[None, :-1, :]
        fraction = np.clip(np.sum(delta * self.vectors, axis=2) / self.lengths ** 2, 0, 1)
        residual = delta - fraction[:, :, None] * self.vectors
        squared = np.sum(residual ** 2, axis=2)
        index = np.argmin(squared, axis=1)
        rows = np.arange(len(points))
        arc = self.arc[index] + fraction[rows, index] * self.lengths[index]
        return arc, np.sqrt(squared[rows, index]), self.angles[index]

    def point_at(self, distance):
        distance = float(np.clip(distance, 0, self.arc[-1]))
        index = min(len(self.lengths) - 1, max(0, int(np.searchsorted(self.arc, distance) - 1)))
        fraction = (distance - self.arc[index]) / self.lengths[index]
        return self.points[index] + fraction * self.vectors[index], float(self.angles[index])

    def project_with_rear(self, points):
        """Signed approach extension for body conflicts behind/beside ego."""
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        arcs, distance, heading = self.project(points)
        tangent = self.vectors[0] / self.lengths[0]
        delta = points - self.points[0]
        along = delta @ tangent
        behind = (arcs < 1e-6) & (along < 0)
        arcs[behind] = along[behind]
        distance[behind] = np.abs(delta[behind, 0] * tangent[1] - delta[behind, 1] * tangent[0])
        return arcs, distance, heading


@dataclass
class TrafficAssessment:
    speed_limit_kmh: float | None = None
    front_gap_m: float | None = None
    min_ttc_s: float | None = None
    emergency: bool = False
    conflict_distance_m: float | None = None
    conflict_time_s: float | None = None
    obstacle_gap_m: float | None = None
    junction_conflict: bool = False
    front_hard_speed_limit_kmh: float | None = None


def assess_traffic(agent_input, corridor, speed_mps, config, tracks=None, junction_mode=False):
    result = TrafficAssessment()
    actors = [actor for actor, valid in zip(agent_input.traffic_context.actors,
                                           agent_input.traffic_context.validity_mask) if valid]
    if tracks is not None:
        actors = [track.actor for track in tracks]
    ego_half_length = config.vehicle_length_m / 2
    ego_half_width = config.vehicle_width_m / 2
    if actors:
        positions = np.array([(a.relative_position_x, a.relative_position_y) for a in actors])
        actor_s, distances, headings = corridor.project(positions)
        for actor_index, (actor, arc, distance, tangent) in enumerate(zip(actors, actor_s, distances, headings)):
            angle = wrap(actor.relative_heading - tangent)
            half_length = (actor.length * abs(math.cos(angle)) + actor.width * abs(math.sin(angle))) / 2
            half_width = (actor.width * abs(math.cos(angle)) + actor.length * abs(math.sin(angle))) / 2
            velocity = np.array([speed_mps + actor.relative_velocity_x, actor.relative_velocity_y])
            track = tracks[actor_index] if tracks is not None else None
            uncertainty = track.uncertainty_m if track else 0.0
            if track is not None:
                velocity = track.velocity
            tangent_velocity = float(velocity @ np.array([math.cos(tangent), math.sin(tangent)]))
            # Front vehicles are matched to the actual curved path, not an
            # infinite straight strip through the ego car.
            if arc > 0.05 and distance <= ego_half_width + half_width + config.traffic_margin_m + uncertainty:
                gap = max(0.0, float(arc) - ego_half_length - half_length)
                lead_speed = max(0.0, tangent_velocity)
                closing = max(0.0, speed_mps - tangent_velocity)
                desired = config.minimum_following_gap_m + config.following_time_s * speed_mps
                following = max(0.0, lead_speed + (gap - desired) / config.following_time_s)
                a = config.comfortable_deceleration_mps2
                reaction = a * config.response_time_s
                braking = max(0.0, math.sqrt(reaction ** 2 + lead_speed ** 2
                              + 2 * a * max(0.0, gap - config.minimum_following_gap_m)) - reaction)
                limit = 3.6 * min(following, braking)
                hard = 3.6 * min(braking, max(0.0, (gap - config.minimum_following_gap_m) / config.following_time_s))
                result.front_hard_speed_limit_kmh = hard if result.front_hard_speed_limit_kmh is None else min(
                    result.front_hard_speed_limit_kmh, hard)
                result.speed_limit_kmh = limit if result.speed_limit_kmh is None else min(result.speed_limit_kmh, limit)
                result.front_gap_m = gap if result.front_gap_m is None else min(result.front_gap_m, gap)
                ttc = gap / closing if closing > 1e-3 else None
                if ttc is not None:
                    result.min_ttc_s = ttc if result.min_ttc_s is None else min(result.min_ttc_s, ttc)
                if gap <= config.minimum_following_gap_m or (ttc is not None and ttc <= config.emergency_ttc_s):
                    result.emergency = True

            lateral_speed = abs(float(velocity @ np.array([-math.sin(tangent), math.cos(tangent)])))
            if abs(angle) < 0.5 and lateral_speed < 0.7 and (track is None or abs(track.yaw_rate) < 0.04):
                continue
            # Predict cut-ins and crossing vehicles in the fixed current ego
            # frame. Expanding by half a time sample covers between-sample motion.
            horizon = min(3.0, config.prediction_horizon_s) if junction_mode else config.prediction_horizon_s
            times = np.arange(config.prediction_dt_s, horizon + 1e-6,
                              config.prediction_dt_s)
            # Cap longitudinal prediction at the known route; do not extrapolate
            # into an unobserved exit or through a missing connector.
            ego_s = speed_mps * times
            padding = (float(np.linalg.norm(velocity)) + speed_mps) * config.prediction_dt_s / 2
            models = track.predictions(times, nearby_turn_horizon_s=2.0 if junction_mode else 4.0) if track else [(
                np.array([actor.relative_position_x, actor.relative_position_y]) + times[:, None] * velocity,
                np.full(len(times), actor.relative_heading), np.ones(len(times), dtype=bool))]
            for predicted, predicted_heading, valid_prediction in models:
                projected_s, future_distance, future_heading = corridor.project(predicted)
                relative_heading = predicted_heading - future_heading
                future_half_length = (actor.length * np.abs(np.cos(relative_heading))
                                      + actor.width * np.abs(np.sin(relative_heading))) / 2
                future_half_width = (actor.width * np.abs(np.cos(relative_heading))
                                     + actor.length * np.abs(np.sin(relative_heading))) / 2
                overlap = (valid_prediction & (ego_s <= corridor.arc[-1]) & (projected_s > ego_half_length)
                           & (future_distance <= ego_half_width + future_half_width + config.traffic_margin_m
                              + uncertainty + (0.0 if junction_mode else config.prediction_lateral_uncertainty_mps * times))
                           & (np.abs(projected_s - ego_s) <= ego_half_length + future_half_length + padding))
                collisions = np.flatnonzero(overlap)
                if len(collisions):
                    first = int(collisions[0])
                    conflict_distance = max(0.0, float(projected_s[first]) - ego_half_length - float(future_half_length[first]))
                    conflict_time = float(times[first])
                    if result.conflict_distance_m is None or conflict_distance < result.conflict_distance_m:
                        result.conflict_distance_m = conflict_distance
                        result.conflict_time_s = conflict_time
                    if conflict_time <= config.emergency_ttc_s:
                        result.emergency = True

        point = agent_input.task_context.decision_point
        if point is not None and agent_input.task_context.route_passage is None and point.distance_m <= config.junction_scan_m:
            entry, entry_heading = corridor.point_at(point.distance_m)
            ego_eta = point.distance_m / max(3.0, min(speed_mps, config.junction_speed_kmh / 3.6))
            for actor in actors:
                # An approaching crossing/oncoming vehicle may turn into the
                # intersection. Constant straight-line prediction alone misses
                # that intention, so reserve the entry before reaching it.
                if abs(wrap(actor.relative_heading - entry_heading)) < 0.5:
                    continue
                position = np.array([actor.relative_position_x, actor.relative_position_y])
                velocity = np.array([speed_mps + actor.relative_velocity_x, actor.relative_velocity_y])
                delta = position - entry
                velocity_squared = float(velocity @ velocity)
                closest_time = float(np.clip(-float(delta @ velocity) / max(velocity_squared, 1e-6),
                                             0, config.prediction_horizon_s))
                closest_distance = float(np.linalg.norm(delta + velocity * closest_time))
                occupation_time = (config.junction_conflict_radius_m + actor.length / 2) / max(
                    1.0, math.sqrt(velocity_squared))
                if (closest_distance <= config.junction_conflict_radius_m
                        and abs(ego_eta - closest_time) <= occupation_time + 0.8):
                    stop_distance = max(0.0, point.distance_m - ego_half_length)
                    result.conflict_distance_m = stop_distance if result.conflict_distance_m is None else min(
                        stop_distance, result.conflict_distance_m)
                    result.junction_conflict = True

    # Unclassified LiDAR hits in the planned corridor can represent objects not
    # present in the fixed-capacity traffic context. Known vehicle surfaces are
    # excluded here so a moving lead car is not treated as a stationary wall.
    rays = agent_input.core_observation.lidar_points
    mask = rays < 0.9999
    if np.any(mask):
        angles = np.arange(len(rays)) * (2 * math.pi / len(rays))
        ranges = rays * config.lidar_range_m
        hits = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))[mask]
        unknown = np.ones(len(hits), dtype=bool)
        for actor in actors:
            delta = hits - np.array([actor.relative_position_x, actor.relative_position_y])
            c, s = math.cos(actor.relative_heading), math.sin(actor.relative_heading)
            along = delta[:, 0] * c + delta[:, 1] * s
            across = -delta[:, 0] * s + delta[:, 1] * c
            inside = ((np.abs(along) <= actor.length / 2 + 0.5)
                      & (np.abs(across) <= actor.width / 2 + 0.5))
            unknown &= ~inside
        if np.any(unknown):
            arc, distance, _ = corridor.project(hits[unknown])
            obstructing = (arc > 0.05) & (distance <= ego_half_width + config.obstacle_margin_m)
            if np.any(obstructing):
                result.obstacle_gap_m = max(0.0, float(np.min(arc[obstructing])) - ego_half_length)
                gap = result.obstacle_gap_m
                if gap <= config.minimum_following_gap_m or (speed_mps > 0.01 and gap / speed_mps <= config.emergency_ttc_s):
                    result.emergency = True
    return result


def assess_rear_side_escape(corridor, tracks, speed_mps, cap_kmh, config):
    """Compare holding speed with a bounded acceleration for a rear-side cut-in.

    Braking blindly for a vehicle already moving into ego's rear quarter can
    prolong body overlap. Acceleration is allowed only if every predicted actor
    remains clear and front stopping/headway plus all other speed caps permit it.
    No lateral maneuver or override of an emergency stop is authorized here.
    """
    result = {"rear_side_risk_ttc_s": None, "rear_escape_safe": False,
              "rear_escape_cap_kmh": float(cap_kmh)}
    side_ids = set()
    for track in tracks:
        actor = track.actor
        approaching = (actor.relative_position_y * track.velocity[1] < -0.05
                       or actor.relative_position_y * track.yaw_rate < -0.05)
        if (-(config.vehicle_length_m + actor.length) <= actor.relative_position_x
                <= (config.vehicle_length_m + actor.length) / 2
                and abs(actor.relative_heading) < 1.0 and approaching):
            side_ids.add(track.track_id)
    if not side_ids or speed_mps < 1:
        return result
    times = np.arange(config.prediction_dt_s, config.prediction_horizon_s + 1e-6, config.prediction_dt_s)
    coast = speed_mps * times
    cap = max(speed_mps, cap_kmh / 3.6)
    a = config.rear_escape_acceleration_mps2
    accelerating_time = min(config.prediction_horizon_s, max(0, (cap - speed_mps) / a))
    active_time = np.minimum(times, accelerating_time)
    accelerating = speed_mps * active_time + a * active_time ** 2 / 2 + cap * np.maximum(0, times - accelerating_time)
    escape_collision = False
    for track in tracks:
        actor = track.actor
        padding = (float(np.linalg.norm(track.velocity)) + speed_mps) * config.prediction_dt_s / 2
        for positions, actor_headings, valid in track.predictions(times):
            arcs, distance, headings = corridor.project_with_rear(positions)
            angle = actor_headings - headings
            half_length = (actor.length * np.abs(np.cos(angle)) + actor.width * np.abs(np.sin(angle))) / 2
            half_width = (actor.width * np.abs(np.cos(angle)) + actor.length * np.abs(np.sin(angle))) / 2
            lateral = valid & (distance <= config.vehicle_width_m / 2 + half_width + config.traffic_margin_m + track.uncertainty_m)
            envelope = config.vehicle_length_m / 2 + half_length + padding
            coasting_overlap = lateral & (coast <= corridor.arc[-1]) & (np.abs(arcs - coast) <= envelope)
            if track.track_id in side_ids and np.any(coasting_overlap):
                ttc = float(times[np.flatnonzero(coasting_overlap)[0]])
                result["rear_side_risk_ttc_s"] = ttc if result["rear_side_risk_ttc_s"] is None else min(
                    result["rear_side_risk_ttc_s"], ttc)
            if np.any(lateral & (accelerating <= corridor.arc[-1]) & (np.abs(arcs - accelerating) <= envelope)):
                escape_collision = True
    result["rear_escape_safe"] = (result["rear_side_risk_ttc_s"] is not None and not escape_collision
                                  and cap_kmh > speed_mps * 3.6 + 1.0)
    return result
