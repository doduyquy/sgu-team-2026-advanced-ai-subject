"""Hybrid A* uses the same public input and discrete path controller as graph A*."""

from src.agents.astar_agent import AStarAgent
from src.agents.hybrid_astar import hybrid_astar_search, load_hybrid_config
from src.agents.path_following import follow_local_path
from src.platform.agent import AgentDecision
from src.platform import canonical_json_sha256


class HybridAStarAgent(AStarAgent):
    agent_id = "stage2_hybrid_astar"
    search_method = "hybrid_astar"
    heuristic_name = "hybrid_euclidean_goal_region"
    objective = "sum_bicycle_arc_cost"

    @classmethod
    def experiment_parameters(cls):
        parameters = super().experiment_parameters()
        parameters.pop("config_sha256")
        parameters.update(state_space="continuous_bicycle_poses_v1", hybrid_astar=load_hybrid_config())
        return dict(parameters, config_sha256=canonical_json_sha256(parameters))

    def __init__(self):
        super().__init__()
        self.hybrid_config = load_hybrid_config()

    def act(self, agent_input):
        if not self._ready:
            raise RuntimeError("Reset the Hybrid A* agent before inference")
        if not agent_input.task_context.validity_mask.any():
            return AgentDecision(1, {"planner_status": "NO_PUBLIC_ROUTE"})
        result = hybrid_astar_search(agent_input, self.config, self.hybrid_config, self.config["max_expansions"])
        if not result.found:
            return AgentDecision(1, {"planner_status": "NO_PATH", "expansions": result.expansions})
        return follow_local_path(result.poses[1:, :2], agent_input, self.config, {
            "planner_status": "PATH_FOUND", "expansions": result.expansions,
            "local_path_cost": result.cost, "heuristic": self.heuristic_name,
            "motion_primitives": len(result.steering_sequence),
        })
