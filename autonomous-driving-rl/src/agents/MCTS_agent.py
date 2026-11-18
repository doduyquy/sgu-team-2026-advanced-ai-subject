"""Stage 3 MCTS:  
    BUT IT NOT SUITABLE FOR CONTINUOUS DOMAIN 

- Phase 1: Selection
- Phase 2: Expansion 
- Phase 3: Simulation(Rollout)
- Phase 4: Backpropagation

---4 Phases---
act(agent_input):
    -> đọc input (tình trạng xe)
    -> thử nhiều trường hợp (mô phỏng tương lai)
    -> chọn act tốt
    -> return AgentDecision

Based on the student's Stage0-style lifecycle. Each act() builds a fresh tree
from AgentInputV1; all trial actions use mcts_model, never env.step().
This is a SANDBOX/AUDIT candidate, not a certified benchmark driving policy.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Optional

import numpy as np

from src.agents.mcts_model import (
    ACTION_GRID, FAILURE_REASONS, MCTSConfig, PlanningScene, PlanningState,
    load_mcts_config, predict_step,
)
from src.platform.agent import AgentDecision, AgentDescriptor, AgentPublicEpisodeContext
from src.platform.agent_input import AgentInputV1
from src.platform.protocol import canonical_json_sha256


@dataclass
class MCTSNode:
    """Predicted state; non-root statistics measure its INCOMING action's return.

    total_return/visits includes reward_from_parent. UCT compares these values
    between siblings. Root statistics instead measure the complete return.
    """
    state: PlanningState
    depth: int = 0
    action_from_parent: Optional[int] = None
    reward_from_parent: float = 0.0
    terminal: bool = False
    reason: str = "RUNNING"
    visits: int = 0
    total_return: float = 0.0
    children: dict = field(default_factory=dict)
    untried_actions: list = field(default_factory=lambda: list(range(len(ACTION_GRID))))

    @property
    def mean_return(self):
        return self.total_return / self.visits if self.visits else 0.0


class MCTSAgent:
    agent_id = "stage3_mcts"
    agent_version = "3.0.0"  
    method_family = "SIMULATION_PLANNING"

    def __init__(self, config: Optional[MCTSConfig] = None):
        self.config = load_mcts_config() if config is None else config
        if not isinstance(self.config, MCTSConfig):
            raise TypeError("config must be MCTSConfig")
        
        self._rng = None
        self._step_counter = 0
        self._dt = 0.1
        self._num_iterations = self.config.num_iterations
        self._max_depth = self.config.max_depth
        self._gamma = self.config.gamma
        self._last_search_diagnostics = {}
        self._model_steps = 0
        self._descriptor = AgentDescriptor(
            agent_id=self.agent_id,
            agent_version=self.agent_version,
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="discrete9_lowbranch_v1",
            inference_stochasticity="stochastic",
            stateful_within_episode=True,
            method_family=self.method_family,
        )

    @classmethod
    def experiment_parameters(cls):
        """Pure hook: executor records configuration and hash in the run manifest."""
        parameters = {
            "schema_version": "mcts_parameters_v1",
            "search_method": "uct_mcts",
            "model": "local_bicycle_constant_actor_velocity_v1",
            "parameters": asdict(load_mcts_config()),
        }
        return dict(parameters, config_sha256=canonical_json_sha256(parameters))

    @property
    def descriptor(self) -> AgentDescriptor:
        return self._descriptor

    def reset(self, public_context: AgentPublicEpisodeContext, agent_seed: Optional[int] = None) -> None:
        if public_context.input_profile_id != self.descriptor.input_profile_id:
            raise ValueError("MCTS requires STATE_DECISION_V1")
        if public_context.action_adapter_id != self.descriptor.action_adapter_id:
            raise ValueError("MCTS requires discrete9_lowbranch_v1")
        self._rng = np.random.RandomState(42 if agent_seed is None else int(agent_seed))
        self._step_counter = 0
        self._dt = float(public_context.control_dt_s)
        self._last_search_diagnostics = {}
        self._model_steps = 0

    def close(self) -> None:
        self._rng = None
        self._last_search_diagnostics = {}

    def act(self, agent_input: AgentInputV1) -> AgentDecision:
        if self._rng is None:
            raise RuntimeError("Call reset() before act().")
        if agent_input.profile_id != self.descriptor.input_profile_id:
            raise ValueError("MCTS requires the declared public input profile")
        
        self._step_counter += 1
        action_id = self._search(agent_input)
        
        return AgentDecision(
            action_payload=int(action_id),
            diagnostics={
                "step_counter": self._step_counter,
                "selected_action": int(action_id),
                **self._last_search_diagnostics,
            },
        )

    def _read_planning_input(self, agent_input: AgentInputV1):
        """Copy public data; decode speed using the declared static model constant."""
        traffic, task = agent_input.traffic_context, agent_input.task_context
        z = float(agent_input.core_observation.ego_state[3])
        speed_kmh = max(0., z * (self.config.max_speed_kmh + 1.) - 1.)
        return {
            "speed_normalized": z,
            "speed_m_s": speed_kmh / 3.6,
            "last_steering": 2 * float(agent_input.core_observation.ego_state[5]) - 1.,
            "actors": traffic.actors_array[traffic.validity_mask].copy(),
            "waypoints": task.waypoints_array[task.validity_mask].copy(),
            "lane_width": float(task.current_lane_width),
            "lidar": agent_input.core_observation.lidar_points.copy(),
        }

    def _predict_step(self, scene, state, action_id):
        """Count model steps, not real simulator steps."""
        self._model_steps += 1
        return predict_step(scene, state, action_id, self._dt)

    def _select_child(self, node):
        """Phase 1 - UCT: mean return + exploration bonus."""
        
        actions = sorted(node.children)
        scores = []
        for action in actions:
            child = node.children[action]
            
            # Trien khai cong thuc UCB1 (UCT)
            score = (child.mean_return + self.config.exploration_constant
                     * math.sqrt(math.log(max(1, node.visits)) / child.visits)) if child.visits else float("inf")
            scores.append(score)
        best = max(scores)
        ties = [action for action, score in zip(actions, scores) if score == best]
        
        return node.children[int(self._rng.choice(ties))]

    def _expand(self, scene, node):
        """Phase 2 - try one unexpanded action using the local model."""
       
        index = int(self._rng.randint(len(node.untried_actions)))
        action = node.untried_actions.pop(index)
        transition = self._predict_step(scene, node.state, action)
        child = MCTSNode(
            state=transition.state, depth=node.depth + 1, action_from_parent=action,
            reward_from_parent=transition.reward, terminal=transition.terminal, reason=transition.reason,
        )
        node.children[action] = child
        return child

    def _rollout(self, scene, node):
        """Phase 3 - epsilon-guided continuation over the REMAINING horizon."""
        
        state, terminal = node.state, node.terminal
        total, discount = 0., 1.
        for _ in range(node.depth, self._max_depth):
            if terminal:
                break
            if self._rng.random_sample() < self.config.rollout_random_probability:
                action = int(self._rng.randint(len(ACTION_GRID)))
            else:
                action = scene.guided_action(state)
            transition = self._predict_step(scene, state, action)
            total += discount * transition.reward
            discount *= self._gamma
            state, terminal = transition.state, transition.terminal
        
        return float(total)

    def _backpropagate(self, path, leaf_return):
        """Phase 4 - add each edge reward once, with discount; no player sign flip."""
        
        value = leaf_return
        for child in reversed(path[1:]):
            value = child.reward_from_parent + self._gamma * value
            child.visits += 1
            child.total_return += value
       
        path[0].visits += 1
        path[0].total_return += value

    def _search(self, agent_input: AgentInputV1) -> int:
        planning_input = self._read_planning_input(agent_input)
        
        self._model_steps = 0
        self._last_search_diagnostics = {
            "search_iterations": 0, "model_steps": 0, "planner_status": "NO_PUBLIC_ROUTE",
        }

        if not len(planning_input["waypoints"]):
            return 1  # Brake straight; a fallback, not a safety guarantee.
        try:
            scene = PlanningScene(planning_input, self.config)
        except ValueError:
            return 1
        
        state = PlanningState(speed=planning_input["speed_m_s"], last_steering=planning_input["last_steering"])
        failure_index, reason = scene.first_failure([state])
        
        if failure_index is not None:
            self._last_search_diagnostics["planner_status"] = "ROOT_MODEL_" + reason
            return 1
        root = MCTSNode(state)

        # Search the tree
        for _ in range(self._num_iterations):
            node, path = root, [root]
            while not node.terminal and node.depth < self._max_depth:
                if node.untried_actions:
                    node = self._expand(scene, node)
                    path.append(node)
                    break
                node = self._select_child(node)
                path.append(node)

            self._backpropagate(path, self._rollout(scene, node))

        # Avoid root edges predicted to fail in their FIRST control interval
        # whenever another root edge does not predict an immediate failure.
        candidates = [child for child in root.children.values() if child.reason not in FAILURE_REASONS]
        if candidates:
            best_key = max((child.visits, child.mean_return) for child in candidates)
            ties = [child for child in candidates if (child.visits, child.mean_return) == best_key]
            chosen = ties[int(self._rng.randint(len(ties)))].action_from_parent
            status = "ACTION_SELECTED"
        else:
            chosen, status = 1, "NO_SAFE_MODEL_ACTION"
       
        self._last_search_diagnostics = {
            "planner_status": status,
            "search_iterations": root.visits,
            "model_steps": self._model_steps,
            "lookahead_s": self._max_depth * self._dt,
            "root_action_visits": [root.children[a].visits if a in root.children else 0 for a in range(9)],
            "root_action_values": [float(root.children[a].mean_return) if a in root.children else None for a in range(9)],
        }
        
        return int(chosen)
