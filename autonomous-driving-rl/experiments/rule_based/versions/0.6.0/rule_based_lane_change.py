"""P4: deterministic adjacent-lane maneuvers using public local snapshots only.

No permission is inferred from missing/unknown lines. Targets and paths persist
across navigation's nearest-lane switch. Safety is a bounded prediction guard,
not a claim that unobserved actors or future intent are known.
"""

from dataclasses import dataclass, field
import math

import numpy as np

from src.platform.agent_input import RouteWaypointV1
from .rule_based_traffic import RouteCorridor, assess_traffic, wrap


@dataclass
class LanePlan:
    waypoints: object = None
    heading: float = 0.0
    lateral: float = 0.0
    active: bool = False
    speed_cap: float | None = None
    diagnostics: dict = field(default_factory=dict)


def lane_corridor(lane):
    samples = [p for p in lane.points if p[0] >= 0]
    points = np.asarray(lane.points)
    heading = float(np.interp(0, points[:, 0], np.unwrap(points[:, 3])))
    xy = np.array([np.interp(0, points[:, 0], points[:, i]) for i in (1, 2)])
    lateral = float(xy @ np.array([-math.sin(heading), math.cos(heading)]))
    waypoints = [RouteWaypointV1(p[0], p[1], p[2], p[3]) for p in samples if p[0] > 0]
    return RouteCorridor(waypoints, heading, lateral)


class LaneChangeController:
    def __init__(self):
        self.phase = "KEEP_LANE"
        self.source_id = self.target_id = None
        self.world_path = None
        self.reason = None
        self.cooldown = 0.0
        self.slow_time = 0.0
        self.elapsed = 0.0
        self.settle = 0.0
        self.committed = False
        self.last_progress = 0.0
        self.return_lane_id = None
        self.overtaken_track_id = None

    def _trajectory(self, target, length):
        samples = np.asarray(target.points)
        distance = np.arange(0, min(50.0, target.remaining_m) + 1e-6, 1.0)
        if len(distance) < 3:
            return None
        target_xy = np.column_stack([np.interp(distance, samples[:, 0], samples[:, i]) for i in (1, 2)])
        headings = np.interp(distance, samples[:, 0], np.unwrap(samples[:, 3]))
        # Start at the actual ego center; blend its lateral offset to the
        # target centerline by spatial quintic (zero endpoint slope/curvature).
        tangent = np.column_stack((np.cos(headings), np.sin(headings)))
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        offset = float(target_xy[0] @ normal[0])
        u = np.clip(distance / length, 0, 1)
        blend = 10*u**3 - 15*u**4 + 6*u**5
        xy = target_xy - (1 - blend)[:, None] * offset * normal
        xy -= (1 - blend)[:, None] * float(xy[0] @ tangent[0]) * tangent
        return xy

    def _corridor_from_path(self, path):
        return RouteCorridor.from_points(path)

    def _gap(self, lane, tracks, speed, cfg):
        corridor = lane_corridor(lane)
        reasons, front, rear = [], None, None
        for track in tracks:
            a = track.actor
            arc, lateral, headings = corridor.project_with_rear([(a.relative_position_x, a.relative_position_y)])
            h = float(headings[0])
            angle = wrap(a.relative_heading - h)
            half_w = (a.width * abs(math.cos(angle)) + a.length * abs(math.sin(angle))) / 2
            if lateral[0] > lane.width_m / 2 + half_w + track.uncertainty_m:
                continue
            half_l = (a.length * abs(math.cos(angle)) + a.width * abs(math.sin(angle))) / 2
            gap = abs(float(arc[0])) - half_l - cfg.vehicle_length_m / 2
            velocity = float(track.velocity @ np.array([math.cos(h), math.sin(h)]))
            closing = max(0.0, speed - velocity) if arc[0] >= 0 else max(0.0, velocity - speed)
            follower = speed if arc[0] >= 0 else max(0.0, velocity)
            lead = max(0.0, velocity) if arc[0] >= 0 else speed
            required = cfg.lane_change_min_gap_m + cfg.response_time_s * follower + max(
                cfg.lane_change_gap_time_s * follower,
                (follower**2 - lead**2) / (2 * cfg.comfortable_deceleration_mps2))
            ttc = gap / closing if closing > 1e-3 else None
            entry = {"track_id": track.track_id, "gap_m": gap, "required_m": required, "ttc_s": ttc}
            if arc[0] >= 0:
                front = entry if front is None or gap < front["gap_m"] else front
            else:
                rear = entry if rear is None or gap < rear["gap_m"] else rear
            if gap < required or (ttc is not None and ttc < cfg.lane_change_ttc_s):
                reasons.append("front_gap" if arc[0] >= 0 else "rear_gap")
        return list(dict.fromkeys(reasons)), {"front": front, "rear": rear}

    def _swept(self, inp, path, tracks, speed, cfg):
        corridor = self._corridor_from_path(path)
        # Include the actual center/heading while the controller catches up to
        # the sampled path. A clear ideal centerline alone is insufficient.
        tracking_offset = float(np.linalg.norm(path[0]))
        start_angle = float(corridor.angles[0])
        current_half_width = max(cfg.vehicle_width_m / 2, (
            cfg.vehicle_width_m * abs(math.cos(start_angle)) +
            cfg.vehicle_length_m * abs(math.sin(start_angle))) / 2)
        times = np.arange(0, cfg.lane_change_prediction_s + 1e-6, cfg.prediction_dt_s)
        # Future body orientation follows the planned tangent. Retain the
        # measured initial heading envelope through the controller response,
        # rather than freezing that transient angle for the entire maneuver.
        ego_half_width = cfg.vehicle_width_m / 2 + (current_half_width - cfg.vehicle_width_m / 2) * np.maximum(
            0, 1 - times / cfg.response_time_s)
        for track in tracks:
            a = track.actor
            half_l = (a.length * abs(math.cos(a.relative_heading)) + a.width * abs(math.sin(a.relative_heading))) / 2
            half_w = (a.width * abs(math.cos(a.relative_heading)) + a.length * abs(math.sin(a.relative_heading))) / 2
            if abs(a.relative_position_x) < cfg.vehicle_length_m / 2 + half_l + .5 and abs(a.relative_position_y) < cfg.vehicle_width_m / 2 + half_w + track.uncertainty_m + .2:
                return "current_body_actor", track.track_id
        # The same spatial path under coasting, comfortable braking, and a
        # bounded launch. A change must not rely on exactly one throttle trace.
        acceleration_distance = np.minimum(speed * times + .5 * cfg.lane_change_accel_mps2 * times**2,
                                           cfg.lane_change_speed_kmh / 3.6 * times)
        brake_time = np.minimum(times, speed / cfg.comfortable_deceleration_mps2)
        profiles = [speed * times, acceleration_distance,
                    speed * brake_time - .5 * cfg.comfortable_deceleration_mps2 * brake_time**2]
        for distances in profiles:
            samples = [corridor.point_at(d) for d in distances]
            ego_xy = np.array([p for p, _ in samples])
            headings = np.array([h for _, h in samples])
            for track in tracks:
                for positions, actor_headings, valid in track.predictions(times):
                    relative = positions - ego_xy
                    longitudinal = abs(relative[:, 0]*np.cos(headings) + relative[:, 1]*np.sin(headings))
                    lateral = abs(-relative[:, 0]*np.sin(headings) + relative[:, 1]*np.cos(headings))
                    angles = actor_headings - headings
                    a = track.actor
                    longitudinal_bound = tracking_offset + cfg.vehicle_length_m / 2 + (
                        a.length * abs(np.cos(angles)) + a.width * abs(np.sin(angles))) / 2 + .5
                    lateral_bound = tracking_offset + ego_half_width + (
                        a.width * abs(np.cos(angles)) + a.length * abs(np.sin(angles))) / 2 + track.uncertainty_m + .2
                    # Expand across half a sample interval, including relative
                    # travel. Fast crossing actors cannot jump between samples.
                    velocity = track.velocity - speed * np.column_stack((np.cos(headings), np.sin(headings)))
                    longitudinal_bound += cfg.prediction_dt_s / 2 * abs(
                        velocity[:, 0]*np.cos(headings) + velocity[:, 1]*np.sin(headings))
                    lateral_bound += cfg.prediction_dt_s / 2 * abs(
                        -velocity[:, 0]*np.sin(headings) + velocity[:, 1]*np.cos(headings))
                    lateral_bound += cfg.prediction_lateral_uncertainty_mps * times
                    if np.any(valid & (longitudinal < longitudinal_bound) & (lateral < lateral_bound)):
                        return "swept_actor", track.track_id
        # Reuse the unknown LiDAR return filter (known actor surfaces excluded).
        traffic = assess_traffic(inp, corridor, speed, cfg, tracks, False)
        if traffic.obstacle_gap_m is not None and traffic.obstacle_gap_m < min(corridor.arc[-1], 35):
            return "swept_lidar", None
        return None, None

    def _permission(self, source, target):
        # Both views describe the same actual rendered line. Disagreement
        # indicates bad/missing geometry, never an implicit permission.
        source_points, target_points = np.asarray(source.points), np.asarray(target.points)
        sh = float(np.interp(0, source_points[:, 0], np.unwrap(source_points[:, 3])))
        delta = np.array([np.interp(0, target_points[:, 0], target_points[:, i]) -
                          np.interp(0, source_points[:, 0], source_points[:, i]) for i in (1, 2)])
        left = float(delta @ np.array([-math.sin(sh), math.cos(sh)])) > 0
        boundaries = (source.left_boundary, target.right_boundary) if left else (source.right_boundary, target.left_boundary)
        return boundaries == ("BROKEN_WHITE", "BROKEN_WHITE")

    def update(self, inp, route, heading, lateral, tracks, frame, speed, cfg, junction_locked=False):
        dt = cfg._control_dt_s
        self.cooldown = max(0.0, self.cooldown - dt)
        result = LanePlan(heading=heading, lateral=lateral)
        context = inp.task_context.lane_context
        diag = {"phase": self.phase, "reason": self.reason, "source_id": self.source_id,
                "target_id": self.target_id, "committed": self.committed,
                "progress": self.last_progress, "cooldown_s": self.cooldown, "veto": []}
        result.diagnostics = diag
        if context is None or inp.ego_motion is None:
            diag["veto"] = ["missing_lane_geometry_or_odometry"]
            if self.world_path is not None:
                result.active, result.speed_cap = True, 0.0
                self.phase = diag["phase"] = "BLOCKED_CHANGE"
                cached = self._corridor_from_path(frame.to_ego(self.world_path))
                near = float(cached.project([(0, 0)])[0][0])
                p, h = cached.point_at(near)
                result.heading = h
                result.lateral = float(p @ np.array([-math.sin(h), math.cos(h)]))
                result.waypoints = [RouteWaypointV1(float(d), float(q[0]), float(q[1]), float(angle))
                                    for d in np.arange(2.5, 50.001, 2.5) if near + d < cached.arc[-1]
                                    for q, angle in [cached.point_at(near + d)]]
            return result
        lanes = {lane.lane_id: lane for lane in context.lanes}
        current = next(lane for lane in context.lanes if lane.side == "CURRENT")
        passage = inp.task_context.route_passage
        decision = inp.task_context.decision_point
        # A discretionary pass must leave the entire observed junction approach
        # free. Mandatory positioning may still use the upstream 35 m window.
        junction_buffer = 35.0 if context.required_side is not None else cfg.lane_change_overtake_junction_buffer_m
        restricted = junction_locked or (passage is not None and passage.entry_distance_m <= junction_buffer) or (
            decision is not None and decision.distance_m <= junction_buffer)
        unavailable = inp.traffic_context.overflow_count or inp.traffic_context.motion_paths_overflow_count or (
            np.mean(inp.core_observation.lidar_points <= 1e-4) > .5)
        if self.world_path is None:
            if current.route_compatible is False:
                result.speed_cap = cfg._approach_speed_kmh(0, current.remaining_m - cfg.lane_change_end_buffer_m, speed)
            corridor = RouteCorridor(route, heading, lateral)
            traffic = assess_traffic(inp, corridor, speed, cfg, tracks, False)
            slow = traffic.speed_limit_kmh is not None and traffic.speed_limit_kmh < cfg.cruise_speed_kmh * .65
            self.slow_time = self.slow_time + dt if slow else 0.0
            mandatory = context.required_side is not None
            return_target = lanes.get(self.return_lane_id)
            passed = False
            if return_target is not None and self.overtaken_track_id is not None:
                observed = next((t for t in tracks if t.track_id == self.overtaken_track_id and t.missed_s == 0), None)
                if observed is not None:
                    arc, _, _ = corridor.project_with_rear([(observed.actor.relative_position_x, observed.actor.relative_position_y)])
                    passed = float(arc[0]) < -(cfg.vehicle_length_m + observed.actor.length) / 2 - cfg.lane_change_min_gap_m
            returning = passed and self.cooldown <= 0 and return_target.side != "CURRENT"
            if not mandatory and not returning and self.slow_time < cfg.lane_change_slow_wait_s:
                return result
            if unavailable or restricted or self.cooldown > 0:
                diag["veto"] = ["sensor_overflow" if unavailable else "junction_or_cooldown"]
            else:
                length = min(30.0, max(cfg.lane_change_min_length_m, speed * cfg.lane_change_duration_s))
                candidates = [lane for lane in context.lanes if lane.side != "CURRENT" and
                              (lane.side == context.required_side if mandatory else
                               (lane.lane_id == self.return_lane_id if returning else True))]
                for target in candidates:
                    veto = []
                    if speed * 3.6 > cfg.lane_change_start_speed_kmh or abs(heading) > .2 or abs(lateral) > .4:
                        veto.append("prepare_speed_or_alignment")
                        result.speed_cap = min(result.speed_cap if result.speed_cap is not None else cfg.cruise_speed_kmh,
                                               cfg.lane_change_speed_kmh)
                    if not self._permission(current, target):
                        veto.append("boundary_prohibited_or_unknown")
                    if target.route_compatible is not True:
                        veto.append("target_route_unavailable")
                    if min(current.remaining_m, target.remaining_m) < length + cfg.lane_change_end_buffer_m:
                        veto.append("insufficient_lane_length")
                    angles = np.unwrap(np.array(target.points)[:, 3])
                    if np.max(abs(angles - heading)) > .35:
                        veto.append("curved_or_diverging_lane")
                    gaps, gap_diag = self._gap(target, tracks, speed, cfg)
                    veto += gaps
                    path = self._trajectory(target, length)
                    if path is not None and not veto:
                        swept, actor_id = self._swept(inp, path, tracks, speed, cfg)
                        if swept:
                            veto.append(swept)
                            diag["conflict_track_id"] = actor_id
                    diag["candidate"] = {"lane_id": target.lane_id, **gap_diag}
                    diag["veto"] = veto
                    if veto or path is None:
                        continue
                    self.source_id, self.target_id = current.lane_id, target.lane_id
                    self.world_path = frame.to_world(path)
                    self.phase = "PREPARE_CHANGE"
                    self.reason = "MANDATORY_ROUTE" if mandatory else ("RETURN_AFTER_PASS" if returning else "OVERTAKE_SLOW_LEADER")
                    if self.reason == "OVERTAKE_SLOW_LEADER":
                        front = [t for t in tracks if t.actor.relative_position_x > 0 and abs(t.actor.relative_position_y) < current.width_m / 2]
                        self.overtaken_track_id = min(front, key=lambda t: t.actor.relative_position_x).track_id if front else None
                        self.return_lane_id = current.lane_id
                    elif self.reason != "RETURN_AFTER_PASS":
                        self.return_lane_id = self.overtaken_track_id = None
                    self.elapsed = self.settle = 0.0
                    self.committed = False
                    break
            if self.world_path is None:
                if mandatory or current.route_compatible is False:
                    result.speed_cap = min(result.speed_cap if result.speed_cap is not None else cfg.cruise_speed_kmh,
                                           cfg._approach_speed_kmh(0, current.remaining_m - cfg.lane_change_end_buffer_m, speed))
                return result

        result.active = True
        result.speed_cap = cfg.lane_change_speed_kmh
        self.elapsed += dt
        source, target = lanes.get(self.source_id), lanes.get(self.target_id)
        path = frame.to_ego(self.world_path)
        corridor = self._corridor_from_path(path)
        arc, _, _ = corridor.project([(0, 0)])
        near = float(arc[0])
        # Use original path stationing, rather than re-planning toward a new
        # nearest lane on every step. This prevents target oscillation.
        samples = [corridor.point_at(min(corridor.arc[-1], near + d)) for d in np.arange(0, 50.001, 2.5)
                   if near + d < corridor.arc[-1]]
        if len(samples) < 2 or source is None or target is None:
            self.phase = "BLOCKED_CHANGE"
            result.speed_cap = 0.0
            diag["veto"] = ["maneuver_geometry_lost"]
        else:
            sc, tc = lane_corridor(source), lane_corridor(target)
            _, sd, source_headings = sc.project_with_rear([(0, 0)])
            _, td, _ = tc.project_with_rear([(0, 0)])
            self.last_progress = float(sd[0] / max(.1, sd[0] + td[0]))
            body_half_width = (cfg.vehicle_width_m * abs(math.cos(source_headings[0])) +
                               cfg.vehicle_length_m * abs(math.sin(source_headings[0]))) / 2
            body_reaches_divider = bool(sd[0] + body_half_width >= source.width_m / 2 - .05)
            self.committed = self.committed or body_reaches_divider or self.last_progress >= cfg.lane_change_commit_fraction
            gaps, gap_diag = self._gap(target, tracks, speed, cfg)
            # Check only the remaining maneuver in current ego coordinates.
            remaining = np.array([p for p, _ in samples])
            swept, actor_id = self._swept(inp, remaining, tracks, speed, cfg)
            unsafe = gaps + ([swept] if swept else [])
            # The approach threshold is an initiation veto. Once a legal
            # maneuver has begun upstream, finish it within the same edge;
            # newly observing a junction 35 m ahead must not immobilize a car
            # astride a broken divider. Actual junction locking still vetoes.
            if unavailable or junction_locked or not self._permission(source, target):
                unsafe.append("sensor_boundary_or_junction_lock_changed")
            diag["candidate"] = {"lane_id": target.lane_id, **gap_diag}
            diag["veto"], diag["conflict_track_id"] = unsafe, actor_id
            if unsafe and not self.committed and self.phase != "ABORT_CHANGE":
                abort = self._trajectory(source, cfg.lane_change_min_length_m)
                source_gap, _ = self._gap(source, tracks, speed, cfg)
                abort_risk = self._swept(inp, abort, tracks, speed, cfg)[0] if abort is not None else "missing_path"
                if not source_gap and not abort_risk and not unavailable and self._permission(source, target):
                    self.world_path = frame.to_world(abort)
                    self.source_id, self.target_id = self.target_id, self.source_id
                    self.phase = "ABORT_CHANGE"
                    self.committed = True  # Do not reverse an abort on another tick.
                    result.speed_cap = 0.0  # Re-read the new path on the next tick.
                else:
                    self.phase = "BLOCKED_CHANGE"
                    result.speed_cap = 0.0
            elif unsafe:
                self.phase = "BLOCKED_CHANGE" if self.phase != "ABORT_CHANGE" else self.phase
                result.speed_cap = 0.0
            elif self.phase != "ABORT_CHANGE":
                self.phase = "CHANGE_LANE" if self.committed else "PREPARE_CHANGE"
            self.settle = self.settle + dt if td[0] < .25 and abs(samples[0][1]) < .12 else 0.0
            if self.settle >= cfg.lane_change_settle_s:
                was_abort = self.phase == "ABORT_CHANGE"
                self.world_path = None
                self.phase = "KEEP_LANE"
                self.cooldown = cfg.lane_change_cooldown_s
                self.slow_time = 0.0
                if self.reason == "RETURN_AFTER_PASS" or was_abort:
                    self.return_lane_id = self.overtaken_track_id = None
                diag["completed"] = True
                result.active = False
        if samples:
            p, h = samples[0]
            result.heading = h
            result.lateral = float(p @ np.array([-math.sin(h), math.cos(h)]))
            result.waypoints = [RouteWaypointV1(float(d), float(p[0]), float(p[1]), float(h))
                                for d, (p, h) in zip(np.arange(0, 50.001, 2.5), samples) if d > 0]
        diag.update(phase=self.phase, reason=self.reason, source_id=self.source_id,
                    target_id=self.target_id, committed=self.committed, progress=self.last_progress)
        return result
