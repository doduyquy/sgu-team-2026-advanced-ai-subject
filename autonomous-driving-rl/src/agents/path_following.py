"""Discrete9 path following for Stage 2 and the original learned A* policy."""

import math
import numpy as np

from src.platform.agent import AgentDecision

WHEELBASE_M = 2.5
MAX_STEERING_RAD = math.radians(60)
SPEED_SCALE = 81.0  # MetaDrive observation is (km/h + 1) / 81.
STAGE2_CONTROL = dict(lookahead_m=4.5, steering_deadband=0.065,
                      straight_speed_multiplier=1.10, straight_heading_change_rad=0.08,
                      straight_target_angle_rad=0.15, braking_m_s2=2.5,
                      following_buffer_m=2.0, reaction_time_s=0.5,
                      traffic_prediction_s=[0., .4, .8, 1.2])


def follow_local_path(candidates, agent_input, config, diagnostics):
    """Convert ego-relative path points to a certified discrete9 action."""
    if not len(candidates):
        return AgentDecision(1, {**diagnostics, "planner_status": "AT_LOCAL_GOAL"})
    target = candidates[-1]
    for candidate in candidates:
        if np.linalg.norm(candidate) >= config["control_lookahead_m"]:
            target = candidate
            break
    angle = math.atan2(float(target[1]), float(target[0]))
    # Ego-relative +y is left; positive steering was calibrated on the pinned simulator.
    steering = 2 * WHEELBASE_M * math.sin(angle) / (max(float(np.linalg.norm(target)), 1.0) * MAX_STEERING_RAD)
    steer_index = 2 if steering > config["steering_deadband"] else 0 if steering < -config["steering_deadband"] else 1
    speed = float(agent_input.core_observation.ego_state[3])
    target_speed = config["target_normalized_speed"] * (0.65 if abs(angle) > .25 else 1.)
    throttle_index = 0 if speed > target_speed + .025 else 1 if speed > target_speed else 2
    return AgentDecision(throttle_index * 3 + steer_index, {
        **diagnostics, "target_x": float(target[0]), "target_y": float(target[1]),
    })


def follow_stage2_path(candidates, agent_input, config, diagnostics):
    """Track an interpolated target and cap speed by the space ahead."""
    if not len(candidates):
        return AgentDecision(1, {**diagnostics, "planner_status": "AT_LOCAL_GOAL"})

    settings = STAGE2_CONTROL
    path = np.vstack(([0., 0.], candidates))
    vectors = np.diff(path, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    along = np.r_[0., np.cumsum(lengths)]
    lookahead = min(settings["lookahead_m"], float(along[-1]))
    target = np.array([np.interp(lookahead, along, path[:, axis]) for axis in (0, 1)])
    angle = math.atan2(float(target[1]), float(target[0]))
    steering = math.atan2(2 * WHEELBASE_M * math.sin(angle), max(float(np.linalg.norm(target)), 1.))
    steering /= MAX_STEERING_RAD
    deadband = settings["steering_deadband"]
    steer_index = 2 if steering > deadband else 0 if steering < -deadband else 1

    observation_speed = float(agent_input.core_observation.ego_state[3])
    speed = max(0., (observation_speed * SPEED_SCALE - 1.) / 3.6)
    target_speed = (config["target_normalized_speed"] * SPEED_SCALE - 1.) / 3.6
    reference = agent_input.task_context
    headings = reference.waypoints_array[reference.validity_mask][:4, 2]
    straight = (len(headings) >= 2 and abs(angle) <= settings["straight_target_angle_rad"]
                and max(abs(math.remainder(float(h - headings[0]), math.tau)) for h in headings)
                <= settings["straight_heading_change_rad"])
    if straight:
        target_speed *= settings["straight_speed_multiplier"]
    elif abs(angle) > .25:
        target_speed *= .65

    # Project short traffic motion into the fixed ego frame so a vehicle
    # outside the path now can still trigger braking before crossing it.
    clearance_speed = target_speed
    traffic = agent_input.traffic_context
    prediction_times = np.asarray(settings["traffic_prediction_s"])[:, None]
    squared_lengths = np.maximum(lengths**2, 1e-8)
    deceleration = settings["braking_m_s2"]
    reaction = settings["reaction_time_s"]
    for actor in traffic.actors_array[traffic.validity_mask]:
        actor_velocity = actor[2:4] + np.array([speed, 0.])
        positions = actor[:2] + prediction_times * actor_velocity
        offsets = positions[:, None, :] - path[None, :-1, :]
        fractions = np.clip(np.sum(offsets * vectors, axis=2) / squared_lengths, 0., 1.)
        distances = np.linalg.norm(offsets - fractions[:, :, None] * vectors, axis=2)
        samples = np.arange(len(positions))
        segments = np.argmin(distances, axis=1)
        in_path = (positions[:, 0] > 0.) & (distances[samples, segments] <= (actor[6] + 1.9) / 2 + .35)
        if not np.any(in_path):
            continue
        projected_distances = along[segments] + fractions[samples, segments] * lengths[segments]
        gap = float(np.min(projected_distances[in_path])) - (actor[5] + 4.6) / 2
        available = max(0., gap - settings["following_buffer_m"])
        actor_speed = max(0., float(actor_velocity[0]))
        # Solve closing_speed * reaction + closing_speed**2 / (2*a) <= available.
        closing = math.sqrt((deceleration * reaction)**2 + 2 * deceleration * available) - deceleration * reaction
        cap = 0. if available == 0. else actor_speed + closing
        clearance_speed = min(clearance_speed, cap)

    gap_limited = clearance_speed < target_speed
    brake = speed > clearance_speed + (.10 if gap_limited else .55)
    throttle_index = 0 if brake or clearance_speed == 0. else 1 if speed > clearance_speed else 2
    return AgentDecision(throttle_index * 3 + steer_index, {
        **diagnostics, "target_x": float(target[0]), "target_y": float(target[1]),
        "steering_before_quantization": steering, "target_speed_kmh": clearance_speed * 3.6,
        "traffic_speed_limited": gap_limited,
    })
