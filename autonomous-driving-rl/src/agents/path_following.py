"""The same discrete9 path follower for all Stage 2 planners and learned A*."""

import math
import numpy as np

from src.platform.agent import AgentDecision

WHEELBASE_M = 2.5
MAX_STEERING_RAD = math.radians(60)


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
