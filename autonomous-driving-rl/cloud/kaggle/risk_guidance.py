"""Longitudinal collection guidance using only the existing Stage 5 observation."""
from __future__ import annotations

import numpy as np

PROFILE = "route_projected_risk_v1"
RAY_INDEX = np.r_[0:8, 233:240]
LIDAR_RANGE_M = 50.0


def _path(points):
    # The public route follows a reference lane. Shift the local corridor to
    # the ego's current lateral offset rather than pulling it across lanes.
    delta = np.diff(points, axis=0)
    valid = np.linalg.norm(delta, axis=1) > 0.3
    starts, delta = points[:-1][valid], delta[valid]
    if len(delta) < 2:
        return None
    lengths = np.linalg.norm(delta, axis=1)
    tangent = delta / lengths[:, None]
    if tangent[0, 0] < 0.7 or np.linalg.norm(starts[0]) > 9.0:
        return None
    normal = np.array([-tangent[0, 1], tangent[0, 0]])
    offset = float(starts[0] @ normal)
    if abs(offset) > 5.5:
        return None
    starts = starts - offset * normal
    origin = float(starts[0] @ tangent[0])
    station = np.r_[0.0, lengths[:-1].cumsum()] + origin
    return starts, tangent, lengths, station


def _project(p, route):
    start, tangent, length, station = route
    rel = p[:, None, :] - start
    longitudinal = np.sum(rel * tangent, axis=-1)
    clipped = np.clip(longitudinal, 0.0, length)
    residual = rel - clipped[..., None] * tangent
    selected = (residual ** 2).sum(axis=-1).argmin(axis=1)
    index = np.arange(len(p))
    return (station[selected] + longitudinal[index, selected],
            np.linalg.norm(residual[index, selected], axis=-1),
            tangent[selected])


def risk_action(observation, action):
    """Return executed intent and observable evidence, retaining raw steering."""
    obs = np.asarray(observation)
    intent = np.array(action, copy=True)
    speed = max(0.0, float(obs[3] * 81 - 1) / 3.6)
    rays = obs[19:259][RAY_INDEX]
    lidar = float(rays.min() * LIDAR_RANGE_M)
    result = {"reason": "none", "gap_m": None, "ttc_s": None, "closing_mps": None,
              "hazard": "unresolved_route", "curvature_1pm": None, "forward_lidar_m": lidar,
              "speed_mps": speed, "target_mps": None}
    points = obs[325:385].reshape(20, 3)[obs[385:405] > 0.5, :2] * 50.0
    route = _path(points) if len(points) >= 3 else None
    if route is None:
        return intent, result
    _, tangent, lengths, station = route
    bend = np.abs(np.arctan2(
        tangent[:-1, 0] * tangent[1:, 1] - tangent[:-1, 1] * tangent[1:, 0],
        (tangent[:-1] * tangent[1:]).sum(axis=1))) / ((lengths[:-1] + lengths[1:]) / 2.0)
    near = (station[1:] < max(11., speed * 1.8)) & (station[1:] > 1.0)
    curvature = float(np.median(bend[near])) if near.any() else 0.0
    result["curvature_1pm"] = curvature
    target, cause = np.inf, "none"
    if curvature > 0.028:
        target, cause = float(np.sqrt(2.5 / curvature)), "curvature"

    actors = obs[259:315].reshape(8, 7)[obs[315:323] > .5]
    leads = False
    if len(actors):
        position = actors[:, :2] * 50.0
        relative = actors[:, 2:4] * 30.0
        velocity = relative + np.array([speed, 0.0])
        along, offset, forward = _project(position, route)
        gap = along - (actors[:, 5] * 5.0 + 4.5) / 2.0
        projected_speed = (velocity * forward).sum(axis=1)
        heading = actors[:, 4] * np.pi
        alignment = (np.c_[np.cos(heading), np.sin(heading)] * forward).sum(axis=1)
        corridor = (along > 1.0) & (along < 37.) & (offset < (actors[:, 6] * 5.0 + 1.9) / 2.0 + .2)
        lead = corridor & ((alignment > .4) | (np.linalg.norm(velocity, axis=1) < .8))
        leads = bool(lead.any())
        if leads:
            j = int(np.argmin(np.where(lead, gap, np.inf)))
            front_speed = max(0., float(projected_speed[j]))
            distance = max(0., float(gap[j]))
            closing = speed - front_speed
            ttc = distance / closing if closing > .2 else None
            stop_margin = 3.5 + speed * .75 + max(0.0, speed**2 - front_speed**2) / 6.
            follow_speed = max(0., min(front_speed + (distance - 3.5) / 1.5,
                                       np.sqrt(front_speed**2 + 5. * max(0., distance - 3.5))))
            result.update(gap_m=distance, ttc_s=ttc, closing_mps=closing)
            if distance < stop_margin or (ttc is not None and ttc < 2.6):
                target, cause = min(target, follow_speed, max(0.0, speed - .25)), "closing_risk"
            elif distance < 3.5 + speed * 1.4 and follow_speed < target:
                target, cause = follow_speed, "safe_following"
        # Crossing requires an actual moving lateral actor and predicted time
        # overlap in the route corridor. A nearby adjacent lead is not a crossing.
        cross = ~lead & (position[:, 0] > 0.) & (alignment < .70) & (np.linalg.norm(velocity, axis=1) > 1.0)
        if cross.any() and speed > 2.0:
            crossing_positions = position[cross]
            crossing_velocities = velocity[cross]
            for eta in (0.75, 1.25, 2.0, 2.75):
                s, lateral, _ = _project(crossing_positions + eta * crossing_velocities, route)
                if np.any((s > 0.) & (np.abs(s - speed * eta) < 3.5) & (lateral < 2.1)):
                    request = max(0.0, speed - (2.75 - eta) * 1.3 - 1.0)
                    if request < target:
                        target, cause = request, "crossing_risk"
                    break
    if lidar < max(3.5, .40 * speed + speed**2 / 11.) and speed > 1.5:
        target, cause = min(target, max(0., speed - 2.3)), "lidar_obstacle"
    if np.isfinite(target):
        result["target_mps"] = target
        if target < speed + .1:
            intent[1] = min(intent[1], np.clip((target - speed) / 2.8, -.85, 0.05))
    result["hazard"] = cause
    # Restore low-speed movement only with valid, nearly straight and visibly
    # clear forward space. The lidar check also covers actors missing from slots.
    if (not leads and cause == "none" and len(tangent) >= 4 and lidar > 13.0 and curvature < .022
            and tangent[0, 0] > .93 and speed < 3.2 and abs(float(intent[0])) < .32 and intent[1] < .08
            and not bool(obs[408] > .5)):
        intent[1] = min(.26, .1 + (3.2 - speed) * .07)
        cause = "clear_crawl"
    if abs(float(intent[1]) - float(action[1])) > 1e-6:
        result["reason"] = cause
    return intent, result
