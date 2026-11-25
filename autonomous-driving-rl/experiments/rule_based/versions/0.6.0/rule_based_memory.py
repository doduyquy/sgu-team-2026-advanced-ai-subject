"""Episode-local odometry and actor association using public snapshots only."""

from dataclasses import dataclass
import math

import numpy as np

from src.platform.agent_input import TrafficActorV1
from .rule_based_traffic import wrap


def rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, -s), (s, c)))


class LocalFrame:
    def __init__(self):
        self.position = np.zeros(2)
        self.heading = 0.0
        self.measured = False

    def update(self, motion, speed_mps, dt, first=False):
        self.measured = motion is not None
        if motion is not None:
            self.heading = wrap(self.heading + motion.heading_delta_rad)
            self.position += rotation(self.heading) @ np.array((motion.displacement_x_m, motion.displacement_y_m))
        elif not first:
            self.position += rotation(self.heading) @ np.array((speed_mps * dt, 0))

    def to_world(self, points):
        return np.asarray(points) @ rotation(self.heading).T + self.position

    def to_ego(self, points):
        return (np.asarray(points) - self.position) @ rotation(self.heading)


@dataclass
class Track:
    track_id: int
    position: np.ndarray
    velocity: np.ndarray
    heading: float
    length: float
    width: float
    yaw_rate: float = 0.0
    age_s: float = 0.0
    missed_s: float = 0.0
    hits: int = 1


@dataclass
class ObservedTrack:
    track_id: int
    actor: TrafficActorV1
    velocity: np.ndarray  # absolute velocity expressed in current ego frame
    yaw_rate: float
    missed_s: float
    hits: int
    paths: tuple = ()

    @property
    def uncertainty_m(self):
        return 0.1 + self.missed_s * (1.0 + 0.2 * float(np.linalg.norm(self.velocity)))

    def predictions(self, times, nearby_turn_horizon_s=4.0):
        position = np.array((self.actor.relative_position_x, self.actor.relative_position_y))
        straight = position + times[:, None] * self.velocity
        free_valid = times <= 2.0 if self.paths else np.ones(len(times), dtype=bool)
        models = [(straight, np.full(len(times), self.actor.relative_heading), free_valid)]
        if self.hits >= 3 and abs(self.yaw_rate) > 0.04 and np.linalg.norm(self.velocity) > 0.3:
            w = self.yaw_rate
            angle = math.atan2(self.velocity[1], self.velocity[0])
            speed = float(np.linalg.norm(self.velocity))
            turning = position + speed / w * np.column_stack((
                np.sin(angle + w * times) - math.sin(angle),
                -np.cos(angle + w * times) + math.cos(angle)))
            turn_valid = free_valid
            if self.paths and abs(self.actor.relative_heading) < 0.5 and abs(self.actor.relative_position_y) < 4.5:
                # A nearby roughly parallel actor can change lane even when its
                # map-matched road is straight. Retain the observed turn model
                # through the cut-in horizon rather than suppressing it at 2 s.
                turn_valid = times <= nearby_turn_horizon_s
            models.append((turning, self.actor.relative_heading + w * times, turn_valid))
        for path in self.paths:
            points = np.asarray(path.points)
            distances = times * float(np.linalg.norm(self.velocity))
            predicted = np.column_stack((np.interp(distances, points[:, 0], points[:, 1]),
                                         np.interp(distances, points[:, 0], points[:, 2])))
            headings = np.interp(distances, points[:, 0], np.unwrap(points[:, 3]))
            models.append((predicted, headings, distances <= points[-1, 0]))
        return models


class ActorTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1
        self.ambiguous_matches = 0

    def update(self, agent_input, frame, speed_mps, dt, config):
        observed = [a for a, valid in zip(agent_input.traffic_context.actors,
                                         agent_input.traffic_context.validity_mask) if valid]
        motion = agent_input.ego_motion
        ego_velocity = np.array((motion.velocity_x_mps, motion.velocity_y_mps)) if motion else np.array((speed_mps, 0))
        rot = rotation(frame.heading)
        paths_by_slot = {i: tuple(path for path in agent_input.traffic_context.motion_paths if path.actor_slot == i)
                         for i in range(len(observed))}
        paths_for_id = {}
        positions = [frame.to_world((a.relative_position_x, a.relative_position_y)) for a in observed]
        velocities = [rot @ (ego_velocity + np.array((a.relative_velocity_x, a.relative_velocity_y))) for a in observed]
        # Global greedy matching with deterministic tie-breaking. Slot indices
        # are never identities; position, velocity, dimensions and heading gate.
        costs = []
        for old_index, track in enumerate(self.tracks):
            prediction = track.position + track.velocity * dt
            for index, actor in enumerate(observed):
                gap = float(np.linalg.norm(positions[index] - prediction))
                angle = abs(wrap(frame.heading + actor.relative_heading - track.heading))
                if (gap <= config.track_match_distance_m + track.missed_s * 3
                        and angle < 1.2 and abs(actor.length - track.length) < 1.0
                        and abs(actor.width - track.width) < 0.7):
                    cost = gap + 0.15 * float(np.linalg.norm(velocities[index] - track.velocity)) + angle * 0.5
                    key = (actor.relative_position_x, actor.relative_position_y, actor.relative_heading)
                    costs.append((cost, track.track_id, key, old_index, index))
        used_old, used_new = set(), set()
        self.ambiguous_matches = 0
        for cost, _, _, old_index, index in sorted(costs):
            if old_index in used_old or index in used_new:
                continue
            alternatives = [c[0] for c in costs if c[3] == old_index and c[4] != index and c[4] not in used_new]
            ambiguous = bool(alternatives and min(alternatives) - cost < 0.3)
            self.ambiguous_matches += int(ambiguous)
            track = self.tracks[old_index]
            actor = observed[index]
            heading = wrap(frame.heading + actor.relative_heading)
            measured_yaw = wrap(heading - track.heading) / max(dt + track.missed_s, 1e-4)
            # Without measured ego rotation, apparent actor turning is ambiguous.
            track.yaw_rate = (0.5 * track.yaw_rate + 0.5 * float(np.clip(measured_yaw, -1.2, 1.2))) if frame.measured and not ambiguous else 0.0
            track.position, track.velocity = positions[index], velocities[index]
            track.heading, track.length, track.width = heading, actor.length, actor.width
            track.age_s += dt
            track.missed_s = 0.0
            track.hits += 1
            paths_for_id[track.track_id] = paths_by_slot[index]
            used_old.add(old_index)
            used_new.add(index)
        remaining = []
        for index, track in enumerate(self.tracks):
            if index not in used_old:
                track.position = track.position + track.velocity * dt
                track.heading = wrap(track.heading + track.yaw_rate * dt)
                track.missed_s += dt
                track.age_s += dt
            if track.missed_s <= config.track_ttl_s + 1e-6:
                remaining.append(track)
        # Sort new detections by geometry, independent of input slot order.
        for index in sorted(set(range(len(observed))) - used_new,
                            key=lambda i: (observed[i].relative_position_x, observed[i].relative_position_y)):
            actor = observed[index]
            remaining.append(Track(self.next_id, positions[index], velocities[index],
                                   wrap(frame.heading + actor.relative_heading), actor.length, actor.width))
            paths_for_id[self.next_id] = paths_by_slot[index]
            self.next_id += 1
        self.tracks = remaining
        output = []
        for track in sorted(self.tracks, key=lambda t: t.track_id):
            position = frame.to_ego(track.position)
            if np.linalg.norm(position) > agent_input.traffic_context.radius_m + track.length:
                continue
            velocity = rot.T @ track.velocity
            relative = velocity - ego_velocity
            actor = TrafficActorV1(float(position[0]), float(position[1]), float(relative[0]), float(relative[1]),
                                   wrap(track.heading - frame.heading), track.length, track.width)
            output.append(ObservedTrack(track.track_id, actor, velocity, track.yaw_rate,
                                        track.missed_s, track.hits, paths_for_id.get(track.track_id, ())))
        return output
