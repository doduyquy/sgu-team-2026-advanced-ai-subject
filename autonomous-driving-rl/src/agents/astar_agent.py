"""Stage 2 A* and Stage 6 A* + learned heuristic share graph and controller."""

from src.agents.heuristic_model import HeuristicMLP
from src.agents.path_following import MAX_STEERING_RAD, WHEELBASE_M, follow_local_path
from src.agents.planning_graph import astar_search, build_planning_graph, load_astar_config, planner_fingerprint
from src.platform.agent import AgentDecision, AgentDescriptor
from src.platform import canonical_json_sha256

import math


class AStarAgent:
    agent_id = "stage2_astar"
    method_family = "PLANNER"
    heuristic_name = "euclidean"
    search_method = "astar"
    objective = "sum_edge_cost"

    @classmethod
    def experiment_parameters(cls):
        """Pure factory metadata: record fixed settings without creating a policy or simulator."""
        parameters = dict(schema_version="search_planner_parameters_v1", search_method=cls.search_method,
                          state_space="local_route_lattice_v1", objective=cls.objective,
                          heuristic=cls.heuristic_name, shared_planner=load_astar_config()["planner"],
                          controller=dict(wheelbase_m=WHEELBASE_M, max_steering_deg=math.degrees(MAX_STEERING_RAD)))
        return dict(parameters, config_sha256=canonical_json_sha256(parameters))

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
        result = self.search_graph(graph)
        if not result.found:
            return AgentDecision(1, {"planner_status": "NO_PATH", "expansions": result.expansions})
        return follow_local_path(graph.points[list(result.path)[1:]], agent_input, self.config, {
            "planner_status": "PATH_FOUND", "expansions": result.expansions,
            "local_path_cost": result.cost, "heuristic": self.heuristic_name,
        })

    def search_graph(self, graph):
        h = self._model.heuristic(graph) if self._model is not None else None
        return astar_search(graph, h, self.config["max_expansions"])

    def close(self):
        self._ready = False


class LearnedAStarAgent(AStarAgent):
    agent_id = "stage6_astar_learned"
    method_family = "SEARCH_LEARNING"
    heuristic_name = "learned"
    search_method = "astar_learned"

    def load_checkpoint(self, path, sha256):
        self._model = HeuristicMLP.load(path, sha256, planner_fingerprint(self.config))
