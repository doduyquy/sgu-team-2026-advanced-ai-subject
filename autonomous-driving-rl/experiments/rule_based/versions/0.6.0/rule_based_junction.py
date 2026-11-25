"""Explicit junction lifecycle, route commitment and space-time gap checks.

No legal priority is inferred from absent signs/lights. Geometry-based yielding
is used for the simulator's unsignalized connectors and roundabout sections.
"""

import math

import numpy as np

from src.platform.agent_input import RouteWaypointV1
from .rule_based_traffic import RouteCorridor, wrap


def arrival_time(distance, speed, cap, acceleration):
    distance = np.maximum(0.0, np.asarray(distance))
    initial = max(0.0, min(speed, cap))
    accelerating_distance = max(0.0, (cap * cap - initial * initial) / (2 * acceleration))
    accelerating_time = (np.sqrt(initial * initial + 2 * acceleration * distance) - initial) / acceleration
    return np.where(distance <= accelerating_distance, accelerating_time,
                    (cap - initial) / acceleration + (distance - accelerating_distance) / cap)


class JunctionController:
    def __init__(self):
        self.phase = "CRUISE"
        self.passage_id = None
        self.branch_id = None
        self.maneuver = None
        self.kind = None
        self.entry = 0.0
        self.exit = None
        self.yield_entry = 0.0
        self.conflict_exit = None
        self.entered = False
        self.clear_s = 0.0
        self.wait_s = 0.0
        self.queue_timer_s = 0.0
        self.locked = False
        self.path_world = None
        self.path_origin_s = 0.0
        self.entry_origin_s = 0.0
        self.exit_origin_s = None
        self.yield_origin_s = 0.0
        self.conflict_exit_origin_s = None
        self.last_reason = None
        self.transition = None
        self.last_finished_id = None
        self.path_mismatch_s = 0.0

    def _change(self, phase, reason):
        self.transition = f"{self.phase}->{phase}" if self.phase != phase else None
        self.phase, self.last_reason = phase, reason

    def select_waypoints(self, candidates, frame, dt, config):
        """Keep the committed connector if navigation changes branch mid-node.

        Local immutable map geometry can be kept through traversal. Missing or
        incompatible fresh route input is tolerated only for the configured TTL.
        """
        self.transition = None
        if not self.locked or self.path_world is None:
            return candidates
        local = frame.to_ego(self.path_world.points)
        progress, _, _ = self.path_world.project([frame.position])
        progress = float(progress[0])
        self.entry = self.entry_origin_s - progress
        self.exit = None if self.exit_origin_s is None else self.exit_origin_s - progress
        self.yield_entry = self.yield_origin_s - progress
        self.conflict_exit = None if self.conflict_exit_origin_s is None else self.conflict_exit_origin_s - progress
        consistent = bool(candidates)
        for point in candidates[:4]:
            if progress + point.lookahead_distance_m > self.path_world.arc[-1] - 0.25:
                # Fresh lookahead naturally extends beyond the finite cache as
                # ego advances. Only compare the shared, previously known part.
                continue
            world = frame.to_world((point.relative_x, point.relative_y))
            _, distance, heading = self.path_world.project([world])
            if distance[0] > 0.75 or abs(wrap(heading[0] - frame.heading - point.relative_heading)) > 0.35:
                consistent = False
                break
        if consistent:
            self.path_mismatch_s = 0.0
            # Extend only beyond the frozen prefix, in roughly waypoint-sized
            # increments. This supports a branch end becoming visible later
            # without turning the finite cache into a false road end.
            for point in candidates:
                if progress + point.lookahead_distance_m <= self.path_world.arc[-1] + 1.5:
                    continue
                world = frame.to_world((point.relative_x, point.relative_y))
                if np.linalg.norm(world - self.path_world.points[-1]) > 1.5:
                    self.path_world.points = np.vstack((self.path_world.points, world))
                    self._refresh_path_arrays()
            return candidates
        self.path_mismatch_s += dt
        # Sample the original branch in current ego coordinates; never steer
        # towards a newly selected conflicting branch after commitment.
        kept = []
        for distance in np.arange(2.5, min(50.0, self.path_world.arc[-1] - progress) + 1e-5, 2.5):
            point, heading = self.path_world.point_at(progress + distance)
            relative = frame.to_ego(point)
            kept.append(RouteWaypointV1(float(distance), float(relative[0]), float(relative[1]),
                                        wrap(heading - frame.heading)))
        return kept

    def _snapshot_route(self, corridor, frame):
        # RouteCorridor supports arbitrary 2D frames; freeze the same polyline
        # in the episode-local odometry frame, not simulator global coordinates.
        self.path_world = RouteCorridor.__new__(RouteCorridor)
        self.path_world.points = frame.to_world(corridor.points)
        self._refresh_path_arrays()
        self.entry_origin_s, self.exit_origin_s = self.entry, self.exit
        self.yield_origin_s, self.conflict_exit_origin_s = self.yield_entry, self.conflict_exit

    def _refresh_path_arrays(self):
        self.path_world.vectors = np.diff(self.path_world.points, axis=0)
        self.path_world.lengths = np.linalg.norm(self.path_world.vectors, axis=1)
        self.path_world.arc = np.r_[0.0, np.cumsum(self.path_world.lengths)]
        self.path_world.angles = np.arctan2(self.path_world.vectors[:, 1], self.path_world.vectors[:, 0])

    def update(self, task, corridor, tracks, frame, speed, dt, config, paths_overflow=0):
        passage = task.route_passage
        if self.locked and self.exit is None and passage is not None and passage.branch_id == self.branch_id:
            if passage.exit_distance_m is not None:
                progress, _, _ = self.path_world.project([frame.position])
                self.exit = passage.exit_distance_m
                self.exit_origin_s = float(progress[0]) + self.exit
        if self.locked and self.exit is not None and self.exit < -config.vehicle_length_m / 2 - config.junction_exit_buffer_m:
            self.last_finished_id = self.passage_id
            self.locked = False
            self.entered = False
            self.passage_id = None
            self.path_world = None
            self._change("CRUISE", "rear_cleared_exit")
        if not self.locked:
            if passage is None or (passage.passage_id == self.last_finished_id and passage.entry_distance_m <= 0):
                self.passage_id = None
                self.clear_s = self.wait_s = 0.0
                self._change("CRUISE", "no_local_passage")
                return None, self.diagnostics()
            if self.passage_id != passage.passage_id or self.branch_id != passage.branch_id:
                self.clear_s = self.wait_s = 0.0
                self.queue_timer_s = 0.0
            self.passage_id, self.branch_id = passage.passage_id, passage.branch_id
            self.entry, self.exit = passage.entry_distance_m, passage.exit_distance_m
            self.yield_entry = passage.yield_distance_m if passage.yield_distance_m is not None else self.entry
            self.conflict_exit = passage.conflict_exit_distance_m if passage.conflict_exit_distance_m is not None else self.exit
            self.maneuver, self.kind = passage.maneuver, passage.kind

        cap = config.junction_speed_kmh
        if self.yield_entry > config.junction_scan_m:
            self._change("APPROACH_JUNCTION", "distant_passage")
            return None, self.diagnostics()
        # A connector already occupied at reset cannot be approached again.
        if self.entry < 0 and not self.locked:
            self.locked = True
            self.entered = True
            self._snapshot_route(corridor, frame)
            self._change("TRAVERSE_JUNCTION", "initialized_inside_connector")
        if self.locked:
            # Recheck admission through the approach, including any upstream
            # geometric conflict boundary. Commitment to a branch is separate
            # from permission to proceed into the actual inlet/connector.
            self.entered = self.entered or self.entry <= config.vehicle_length_m / 2
            phase = "EXIT_JUNCTION" if self.exit is not None and self.exit <= 0 else (
                "TRAVERSE_JUNCTION" if self.entered else "COMMIT_JUNCTION")
            if self.path_mismatch_s > config.route_memory_ttl_s:
                self._change("DEGRADED_STOP", "committed_route_input_mismatch")
                return 0.0, self.diagnostics()
            if self.entered:
                self._change(phase, "branch_locked")
                return cap, self.diagnostics()

        blocked, conflicts, exit_known = self._gap_check(corridor, tracks, speed, config)
        if self.kind == "intersection":
            if blocked:
                self.queue_timer_s = config.queue_release_s
            else:
                self.queue_timer_s = max(0.0, self.queue_timer_s - dt)
            blocked = blocked or self.queue_timer_s > 0
        if task.route_end_within_lookahead and self.exit is not None:
            # End-of-route is not an exit blockage; still require space for ego
            # before the actual route tail and preserve ordinary visibility cap.
            exit_known = self.exit + config.vehicle_length_m / 2 <= corridor.arc[-1]
        hazard = blocked or bool(conflicts) or not exit_known or paths_overflow > 0
        self.clear_s = 0.0 if hazard else self.clear_s + dt
        extra_buffer = config.junction_blocked_extra_buffer_m if blocked and self.kind == "intersection" else 0.0
        stop_distance = max(0.0, self.yield_entry - config.vehicle_length_m / 2 - config.junction_entry_buffer_m - extra_buffer)
        if hazard or self.clear_s + 1e-6 < config.yield_release_s:
            reason = "exit_blocked" if blocked else ("crossing_occupancy" if conflicts else (
                "exit_not_visible" if not exit_known else ("actor_path_overflow" if paths_overflow else "confirming_clear_gap")))
            self.wait_s = self.wait_s + dt if speed < 0.3 else 0.0
            state = "BLOCKED" if self.wait_s >= config.blocked_after_s else "YIELD"
            self._change(state, reason)
            # stop_distance already includes the explicit entry buffer. Cancel
            # the generic braking buffer here rather than counting it twice;
            # otherwise ego stops too far back to ever observe the exit.
            limit = config._approach_speed_kmh(0.0, stop_distance + config.braking_buffer_m, speed)
            diagnostics = self.diagnostics()
            diagnostics.update(junction_conflict_track_ids=conflicts,
                               junction_exit_blocked=blocked, junction_admission={"exit_visible": exit_known,
                               "queue_latch_s": self.queue_timer_s, "extra_stop_buffer_m": extra_buffer})
            return min(cap, limit), diagnostics
        self.wait_s = 0.0
        if self.yield_entry <= config.junction_commit_distance_m:
            if not self.locked:
                self.locked = True
                self._snapshot_route(corridor, frame)
            self._change("COMMIT_JUNCTION", "whole_passage_gap_confirmed")
        else:
            self._change("APPROACH_JUNCTION", "whole_passage_gap_clear")
        diagnostics = self.diagnostics()
        diagnostics.update(junction_conflict_track_ids=[], junction_exit_blocked=False,
                           junction_admission={"exit_visible": exit_known, "queue_latch_s": self.queue_timer_s,
                                               "extra_stop_buffer_m": 0.0})
        return cap, diagnostics

    def _gap_check(self, corridor, tracks, speed, config):
        exit_known = self.conflict_exit is not None and self.conflict_exit + config.junction_exit_clearance_m <= corridor.arc[-1]
        end = self.conflict_exit if self.conflict_exit is not None else corridor.arc[-1]
        blocked, conflicts = False, []
        times = np.arange(0, config.junction_prediction_s + 1e-6, config.prediction_dt_s)
        for track in tracks:
            actor = track.actor
            current_s, current_distance, tangent = corridor.project([(actor.relative_position_x, actor.relative_position_y)])
            s, distance, tangent = float(current_s[0]), float(current_distance[0]), float(tangent[0])
            aligned = abs(wrap(actor.relative_heading - tangent)) < 0.5
            actor_speed = float(np.linalg.norm(track.velocity))
            footprint = actor.length / 2 + config.vehicle_length_m / 2 + config.minimum_following_gap_m
            # Do not enter behind a queue unless our entire body can clear.
            if aligned and distance < config.vehicle_width_m / 2 + actor.width / 2 + config.traffic_margin_m:
                along_speed = float(track.velocity @ np.array((math.cos(tangent), math.sin(tangent))))
                if s > max(0, self.yield_entry) and s - footprint < end + config.junction_exit_clearance_m and along_speed < config.junction_queue_speed_mps:
                    blocked = True
                # Aligned moving traffic is handled by ordinary headway rules;
                # a full speed-matched leader is not cross traffic.
                if along_speed > config.junction_queue_speed_mps and abs(track.yaw_rate) < 0.1:
                    continue
            models = track.predictions(times, nearby_turn_horizon_s=2.0)
            if actor_speed < 0.5 and not aligned:
                # A stationary perpendicular actor is not assumed harmless.
                # Include a bounded possible launch, never classify it as static.
                launch_distance = 0.5 * config.stationary_actor_acceleration_mps2 * np.minimum(times, 2) ** 2
                launch_distance += np.maximum(0, times - 2) * 2 * config.stationary_actor_acceleration_mps2
                direction = np.array((math.cos(actor.relative_heading), math.sin(actor.relative_heading)))
                start = np.array((actor.relative_position_x, actor.relative_position_y))
                models.append((start + launch_distance[:, None] * direction, np.full(len(times), actor.relative_heading),
                               np.ones(len(times), dtype=bool)))
            for predicted, actor_headings, valid_prediction in models:
                arcs, distances, headings = corridor.project(predicted)
                angle = actor_headings - headings
                half_width = (actor.width * np.abs(np.cos(angle)) + actor.length * np.abs(np.sin(angle))) / 2
                half_length = (actor.length * np.abs(np.cos(angle)) + actor.width * np.abs(np.sin(angle))) / 2
                ego_time = arrival_time(arcs, speed, config.junction_speed_kmh / 3.6, config.junction_launch_acceleration_mps2)
                occupancy = (half_length + config.vehicle_length_m / 2) / max(2.0, min(speed, cap_mps(config)))
                intersects = (valid_prediction & (arcs >= max(0, self.yield_entry - config.vehicle_length_m / 2))
                              & (arcs <= min(corridor.arc[-1], end + config.vehicle_length_m / 2))
                              & (distances <= config.vehicle_width_m / 2 + half_width + config.traffic_margin_m + track.uncertainty_m)
                              & (np.abs(ego_time - times) <= occupancy + config.junction_gap_s / 2))
                if np.any(intersects):
                    conflicts.append(track.track_id)
                    break
        return blocked, sorted(set(conflicts)), bool(exit_known)

    def diagnostics(self):
        return {"junction_phase": self.phase, "junction_transition": self.transition,
                "junction_reason": self.last_reason, "junction_passage_id": self.passage_id,
                "junction_branch_id": self.branch_id if self.passage_id is not None else None,
                "junction_maneuver": self.maneuver, "junction_kind": self.kind,
                "junction_branch_locked": self.locked, "junction_entry_distance_m": self.entry,
                "junction_yield_distance_m": self.yield_entry, "junction_front_entered": self.entered,
                "junction_conflict_exit_distance_m": self.conflict_exit,
                "junction_exit_distance_m": self.exit, "junction_clear_s": self.clear_s,
                "junction_wait_s": self.wait_s, "route_mismatch_s": self.path_mismatch_s,
                "right_of_way_source": "GEOMETRY_ONLY_NO_SIGN_OR_SIGNAL_INPUT"}


def cap_mps(config):
    return config.junction_speed_kmh / 3.6
