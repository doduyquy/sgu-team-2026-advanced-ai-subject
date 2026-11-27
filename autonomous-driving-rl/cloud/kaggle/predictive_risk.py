"""Short-horizon occupancy and restart controller for MetaDrive simulation."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from cloud.kaggle.risk_guidance import _path, _project

DT = .25
TIME = np.arange(1, 13) * DT
EGO_HALF = np.array([2.25, .95])
ANGLES = np.arange(240) * (2 * np.pi / 240)
RAY_DIRECTIONS = np.column_stack((np.cos(ANGLES), np.sin(ANGLES)))
PG_ROAD_DISTANCE_SCALE = 18.


@dataclass
class Prediction:
    status: str
    earliest_s: float | None
    min_clearance_m: float | None
    progress_m: float
    closing_mps: float | None = None
    headway_s: float | None = None
    stopping_m: float = 0.
    lead_ttc_s: float | None = None
    lead_gap_m: float | None = None


def trajectory(speed, steering, pedal):
    """Approximate bicycle motion, with the input steering receding after 1 s."""
    acceleration = (2.1 if pedal >= 0 else 4.5) * pedal
    end_speed = np.maximum(0., speed + acceleration * TIME)
    distance = DT * (np.r_[speed, end_speed[:-1]] + end_speed) / 2
    steer_angle = -steering * .48 * np.maximum(0., 1. - (TIME - DT) / 1.5)
    yaw_change = distance * np.tan(steer_angle) / 2.7
    yaw = np.cumsum(yaw_change)
    midpoint_yaw = yaw - .5 * yaw_change
    points = np.cumsum(distance[:, None] * np.column_stack((np.cos(midpoint_yaw), np.sin(midpoint_yaw))), axis=0)
    return points, yaw


def occupancy_margin(centers, yaw, actors, ego_speed):
    """Signed separating-axis clearance proxy for oriented, uncertain occupancies."""
    initial = actors[:, :2] * 50.
    velocity = actors[:, 2:4] * 30. + [ego_speed, 0.]
    heading = actors[:, 4] * np.pi
    npc_x = np.column_stack((np.cos(heading), np.sin(heading)))
    npc_y = np.column_stack((-np.sin(heading), np.cos(heading)))
    ego_x = np.column_stack((np.cos(yaw), np.sin(yaw)))
    ego_y = np.column_stack((-np.sin(yaw), np.cos(yaw)))
    dot = np.abs(ego_x @ npc_x.T)
    cross = np.abs(ego_x @ npc_y.T)
    offset = initial[None] + TIME[:, None, None] * velocity - centers[:, None, :]
    # Along-heading uncertainty grows for braking / accelerating motion.
    # Smaller lateral uncertainty covers modest yaw/side motion.
    half_l = actors[:, 5] * 2.5 + .15 + .20 * TIME[:, None] ** 2
    half_w = actors[:, 6] * 2.5 + .12 + .11 * TIME[:, None]
    axes = (
        np.abs(np.einsum("tni,ti->tn", offset, ego_x)) - EGO_HALF[0] - half_l * dot - half_w * cross,
        np.abs(np.einsum("tni,ti->tn", offset, ego_y)) - EGO_HALF[1] - half_l * cross - half_w * dot,
        np.abs(np.einsum("tni,ni->tn", offset, npc_x)) - half_l - EGO_HALF[0] * dot - EGO_HALF[1] * cross,
        np.abs(np.einsum("tni,ni->tn", offset, npc_y)) - half_w - EGO_HALF[0] * cross - EGO_HALF[1] * dot,
    )
    return np.maximum.reduce(axes)


def lidar_clearance(centers, yaw, rays):
    hits = rays < .995
    if not hits.any():
        return np.full(len(centers), np.inf)
    points = RAY_DIRECTIONS[hits] * (rays[hits] * 50.)[:, None]
    delta = points[None] - centers[:, None]
    along = np.einsum("tni,ti->tn", delta, np.column_stack((np.cos(yaw), np.sin(yaw))))
    transverse = np.einsum("tni,ti->tn", delta, np.column_stack((-np.sin(yaw), np.cos(yaw))))
    return np.max(np.abs(np.stack((along, transverse), axis=-1)) - (EGO_HALF + .15), axis=-1).min(axis=1)


class Scene:
    def __init__(self, observation):
        obs = np.asarray(observation)
        self.speed = float(np.clip((obs[3] * 81 - 1) / 3.6, 0, 80 / 3.6))
        self.width = float(obs[405] * 5.)
        self.road_edges = obs[:2] * PG_ROAD_DISTANCE_SCALE
        self.overflow = bool(obs[324] > 0)
        self.end = bool(obs[408] > .5)
        self.rays = obs[19:259]
        self.actors = obs[259:315].reshape(8, 7)[obs[315:323] > .5]
        points = obs[325:385].reshape(20, 3)[obs[385:405] > .5, :2] * 50.
        self.route = _path(points) if len(points) >= 3 else None
        self.lead_speed = None
        self.gap = None
        if self.route is not None and len(self.actors):
            station, offset, tangent = _project(self.actors[:, :2] * 50., self.route)
            velocity = self.actors[:, 2:4] * 30. + [self.speed, 0.]
            heading = self.actors[:, 4] * np.pi
            alignment = (np.column_stack((np.cos(heading), np.sin(heading))) * tangent).sum(axis=-1)
            valid = (station > 1.) & (offset < 2.) & ((alignment > .6) | (np.linalg.norm(velocity, axis=1) < .8))
            if valid.any():
                index = int(np.argmin(np.where(valid, station, np.inf)))
                self.gap = max(0., float(station[index] - EGO_HALF[0] - self.actors[index, 5] * 2.5))
                self.lead_speed = max(0., float(velocity[index] @ tangent[index]))

    def predict(self, action):
        centers, yaw = trajectory(self.speed, *action)
        stopping = .4 * self.speed + self.speed**2 / 6.
        closing = None if self.lead_speed is None else self.speed - self.lead_speed
        headway = self.gap / self.speed if self.gap is not None and self.speed > .25 else None
        lead_ttc = self.gap / closing if self.gap is not None and closing is not None and closing > .2 else None
        occupancy = np.full(len(TIME), np.inf)
        if len(self.actors):
            occupancy = occupancy_margin(centers, yaw, self.actors, self.speed).min(axis=1)
        # Treat unexplained returns as stationary for conservative release.
        # Moving tracked actors use their predicted footprints on their own.
        rays = self.rays.copy()
        if len(self.actors):
            hit_idx = np.flatnonzero(rays < .995)
            hit_pos = RAY_DIRECTIONS[hit_idx] * (50. * rays[hit_idx, None])
            delta = hit_pos[:, None] - self.actors[None, :, :2] * 50.
            heading = self.actors[:, 4] * np.pi
            axis_x = np.column_stack((np.cos(heading), np.sin(heading)))
            axis_y = np.column_stack((-np.sin(heading), np.cos(heading)))
            x = np.abs(np.einsum("hni,ni->hn", delta, axis_x))
            y = np.abs(np.einsum("hni,ni->hn", delta, axis_y))
            accounted = ((x <= self.actors[:, 5] * 2.5 + .35)
                         & (y <= self.actors[:, 6] * 2.5 + .35)).any(axis=1)
            rays[hit_idx[accounted]] = 1.
        clearance = np.minimum(occupancy, lidar_clearance(centers, yaw, rays))
        hit = np.flatnonzero(clearance <= 0.)
        margin = float(clearance.min()) if np.isfinite(clearance.min()) else None
        if len(hit):
            status, earliest = "conflict", float(TIME[hit[0]])
        else:
            status, earliest = "uncertain", None
            if (self.route is not None and not self.end and 2.5 <= self.width <= 5 and not self.overflow
                    and self.width - .25 <= self.road_edges.sum() <= 3 * self.width + .5):
                station, lateral, _ = _project(centers, self.route)
                first = self.route[3][0]
                measured = station >= first
                body_margin = self.width / 2 - EGO_HALF[1] - .16
                tangent = self.route[1][0]
                normal = np.array([-tangent[1], tangent[0]])
                near = (centers @ tangent) < 5.
                offset = (centers @ normal)[near]
                local_extent = EGO_HALF[1] + EGO_HALF[0] * np.abs(
                    np.sin(yaw[near] - math.atan2(tangent[1], tangent[0])))
                if np.any(offset + local_extent > self.road_edges[0]) or np.any(
                        -offset + local_extent > self.road_edges[1]):
                    status = "local_road_exit"
                elif np.any(measured & (lateral > body_margin)):
                    status = "corridor_exit"
                elif station[-1] <= self.route[3][-1] + self.route[2][-1]:
                    status = "clear"
        return Prediction(
            status, earliest, margin, float(centers[-1, 0]), closing, headway, stopping, lead_ttc, self.gap,
        )


class PredictiveController:
    """Frozen actor with a screened restart and short-horizon collision response."""

    def __init__(self, *, guarded=False):
        self.guarded = guarded
        self.stopped_frames = 0
        self.clear_frames = 0
        self.releasing = False

    def act(self, obs, raw):
        raw = np.asarray(raw, dtype=np.float32)
        scene = Scene(obs)
        self.stopped_frames = self.stopped_frames + 1 if scene.speed < .25 and raw[1] < .12 else 0
        decision = scene.predict(raw)
        output = raw.copy()
        reason = "actor"
        if decision.status == "conflict" and decision.earliest_s <= 1.0 and scene.speed > .9:
            brake = np.array([raw[0], min(raw[1], -.65)], dtype=np.float32)
            response = scene.predict(brake)
            if response.earliest_s is None or response.earliest_s > decision.earliest_s:
                output, decision, reason = brake, response, "predicted_conflict"
            self.clear_frames, self.releasing = 0, False
        elif (scene.route is not None and raw[1] < .12
              and scene.speed < (3.8 if self.releasing else 1.6)
              and (not self.guarded or self.releasing or self.stopped_frames >= 40)):
            tangent = scene.route[1][0]
            heading_error = math.atan2(tangent[1], tangent[0])
            lane_steering = float(np.clip(-1.1 * heading_error, -.24, .24))
            # At rest the actor may steer sharply despite a nearly aligned
            # corridor. Check both retained-steer and locally aligned options.
            pedal = .32 if scene.speed < 2. else .21
            steer_options = (raw[0],) if self.guarded else (raw[0], lane_steering)
            choices = [np.array([steer, pedal], dtype=np.float32) for steer in steer_options]
            candidates = [(action, scene.predict(action)) for action in choices]
            feasible = [(action, test) for action, test in candidates if test.status == "clear"
                        and (test.min_clearance_m is None or test.min_clearance_m > .50)
                        and abs(float(action[0])) < .42 and test.progress_m > 1.]
            self.clear_frames = self.clear_frames + 1 if feasible else 0
            if feasible and (self.releasing or self.clear_frames >= 5):
                output, decision = feasible[0]
                self.releasing, reason = True, "predictive_release"
            elif not feasible:
                self.releasing = False
        else:
            self.clear_frames, self.releasing = 0, False
        return output, {"reason": reason, **vars(decision)}
