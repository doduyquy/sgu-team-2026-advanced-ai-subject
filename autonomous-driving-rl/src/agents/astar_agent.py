"""Stage 2 A* and Stage 6 A* + learned heuristic share graph and controller."""

import math
import numpy as np

from src.agents.heuristic_model import HeuristicMLP
from src.agents.planning_graph import astar_search, build_planning_graph, load_astar_config, planner_fingerprint
from src.platform.agent import AgentDecision, AgentDescriptor


class AStarAgent:
    agent_id = "stage2_astar"
    method_family = "PLANNER"

    def __init__(self):
        self.config = load_astar_config()["planner"]
        self._ready = False
        self._model = None

    @property
    def descriptor(self):
        return AgentDescriptor(agent_id=self.agent_id, agent_version="0.1.0",
                               input_profile_id="STATE_DECISION_V1", action_adapter_id="discrete9_lowbranch_v1",
                               inference_stochasticity="deterministic", stateful_within_episode=False,
                               method_family=self.method_family)

    def reset(self, public_context, agent_seed=None):
        if self.agent_id == "stage6_astar_learned" and self._model is None:
            raise RuntimeError("Load a trained heuristic checkpoint before Stage 6 inference")
        self._ready = True

    def act(self, agent_input):
        if not self._ready:
            raise RuntimeError("Reset the A* agent before inference")
        graph = build_planning_graph(agent_input, self.config)
        if graph is None:
            return AgentDecision(1, {"planner_status": "NO_PUBLIC_ROUTE"})
        h = self._model.heuristic(graph) if self._model is not None else None
        result = astar_search(graph, h, self.config["max_expansions"])
        if not result.found:
            return AgentDecision(1, {"planner_status": "NO_PATH", "expansions": result.expansions})
        candidates = graph.points[list(result.path)[1:]]
        target = candidates[-1]
        for candidate in candidates:
            if np.linalg.norm(candidate) >= self.config["control_lookahead_m"]:
                target = candidate
                break
        angle = math.atan2(float(target[1]), float(target[0]))
        # Ego-relative +y is left; steering sign is calibrated against the pinned simulator.
        steering = 2 * 2.5 * math.sin(angle) / (max(float(np.linalg.norm(target)), 1.0) * math.radians(60))
        steer_index = 2 if steering > self.config["steering_deadband"] else 0 if steering < -self.config["steering_deadband"] else 1
        speed = float(agent_input.core_observation.ego_state[3])
        target_speed = self.config["target_normalized_speed"] * (0.65 if abs(angle) > .25 else 1.)
        throttle_index = 0 if speed > target_speed + .025 else 1 if speed > target_speed else 2
        return AgentDecision(throttle_index * 3 + steer_index, {
            "planner_status": "PATH_FOUND", "expansions": result.expansions,
            "local_path_cost": result.cost, "heuristic": "learned" if self._model is not None else "euclidean",
            "target_x": float(target[0]), "target_y": float(target[1]),
        })

    def close(self):
        self._ready = False


class LearnedAStarAgent(AStarAgent):
    agent_id = "stage6_astar_learned"
    method_family = "SEARCH_LEARNING"

    def load_checkpoint(self, path, sha256):
        self._model = HeuristicMLP.load(path, sha256, planner_fingerprint(self.config))
