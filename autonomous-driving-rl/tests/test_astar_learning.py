"""Search optimality on the local model, actual MLP gradients, and TRAIN isolation."""

import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

import numpy as np

from src.agents.heuristic_model import HeuristicMLP, file_sha256
from src.agents.astar_agent import AStarAgent, LearnedAStarAgent
from src.agents.planning_graph import PlanningGraph, FEATURE_NAMES, astar_search, build_planning_graph, exact_cost_to_goal, load_astar_config
from src.training.astar_heuristic import grouped_training_split, select_train_geometries
from src.launcher.cases import get_default_project_root
from src.platform import AgentPublicEpisodeContext
from src.platform.agent_input import AgentInputV1, CoreObservationV1, InputProfileId, RouteWaypointV1, TaskContextV1, TrafficContextV1


class TestAStarLearning(unittest.TestCase):
    def public_input(self):
        waypoints = tuple(RouteWaypointV1(2.5 * i, 2.5 * i, 0., 0.) for i in range(1, 21))
        task = TaskContextV1(waypoints=waypoints, waypoints_array=np.array([p.to_tuple() for p in waypoints]),
                             validity_mask=np.ones(20, dtype=bool))
        return AgentInputV1(InputProfileId.STATE_DECISION_V1, CoreObservationV1(np.zeros(259)),
                            TrafficContextV1.empty(), task, 0)

    def graph(self):
        return PlanningGraph(np.array([[0., 0.], [1., 0.], [0., 1.], [2., 0.]]),
                             (((1, 4.), (2, 1.)), ((3, 1.),), ((1, 1.), (3, 9.)), ()),
                             np.zeros((4, len(FEATURE_NAMES))), 3, 50.)

    def test_astar_reopens_improved_states_and_matches_exact_costs(self):
        graph = self.graph()
        result = astar_search(graph, np.zeros(4))
        self.assertEqual(result.path, (0, 2, 1, 3))
        self.assertEqual(result.cost, 3.)
        np.testing.assert_allclose(exact_cost_to_goal(graph), [3, 1, 2, 0])

    def test_search_handles_no_path_and_rejects_invalid_heuristics(self):
        graph = self.graph()
        result = astar_search(graph, max_expansions=1)
        self.assertFalse(result.found)
        with self.assertRaises(ValueError):
            astar_search(graph, [float('nan')] * 4)

    def test_public_route_graph_is_immutable_and_euclidean_is_a_lower_bound(self):
        inputs = self.public_input()
        graph = build_planning_graph(inputs)
        exact = exact_cost_to_goal(graph)
        self.assertTrue(np.all(graph.euclidean_heuristic() <= exact + 1e-9))
        self.assertAlmostEqual(astar_search(graph).cost, exact[0])
        with self.assertRaises(ValueError):
            graph.points[0, 0] = 5
        baseline = AStarAgent()
        with self.assertRaises(RuntimeError):
            baseline.act(inputs)
        baseline.reset(AgentPublicEpisodeContext(action_adapter_id="discrete9_lowbranch_v1"))
        self.assertEqual(baseline.act(inputs).action_payload, 7)  # Straight acceleration.
        self.assertEqual(baseline.act(replace(inputs, task_context=TaskContextV1.empty())).action_payload, 1)
        learned = LearnedAStarAgent()
        with self.assertRaises(RuntimeError):
            learned.reset(AgentPublicEpisodeContext())

    def test_backpropagation_matches_finite_difference(self):
        model = HeuristicMLP(hidden_units=4, seed=5)
        x = np.random.default_rng(3).normal(size=(5, len(FEATURE_NAMES)))
        y = np.arange(5) / 5
        _, gradients = model.loss_and_grad(x, y)
        for key, parameter in model.params.items():
            for index in list(np.ndindex(parameter.shape))[:3]:
                original = parameter[index]
                parameter[index] = original + 1e-6
                plus = model.loss_and_grad(x, y)[0]
                parameter[index] = original - 1e-6
                minus = model.loss_and_grad(x, y)[0]
                parameter[index] = original
                self.assertAlmostEqual((plus - minus) / 2e-6, gradients[key][index], places=5)

    def test_real_optimizer_updates_weights_reduces_loss_and_roundtrips_checkpoint(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(160, len(FEATURE_NAMES)))
        y = .4 * np.tanh(x[:, 0]) + .1 * x[:, 1]
        model = HeuristicMLP(16, 101)
        before = model.params['w1'].copy()
        model.fit(x[:120], y[:120], x[120:], y[120:], epochs=50, batch_size=32, learning_rate=.01)
        self.assertFalse(np.array_equal(before, model.params['w1']))
        self.assertLess(model.metadata['best_holdout_mse'], model.metadata['initial_holdout_mse'] * .2)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model.npz'
            digest = model.save(path, {'planner_fingerprint': 'test-planner'})
            restored = HeuristicMLP.load(path, digest, 'test-planner')
            np.testing.assert_array_equal(model.predict_residual(x), restored.predict_residual(x))
            with self.assertRaises(ValueError):
                HeuristicMLP.load(path, 'wrong-digest', 'test-planner')
            with self.assertRaises(ValueError):
                HeuristicMLP.load(path, digest, 'different-planner')

    def test_train_geometry_selection_and_internal_holdout_are_disjoint(self):
        root = get_default_project_root()
        rows = select_train_geometries(root, 8)
        self.assertEqual(len(rows), 8)
        self.assertEqual({r['tier'] for r in rows}, {'Easy', 'Medium', 'Hard', 'Extreme'})
        self.assertTrue(all(r['split'] == 'TRAIN' for r in rows))
        groups = np.repeat([r['geometry_id'] for r in rows], 5)
        train, holdout, _ = grouped_training_split(groups, 101)
        self.assertFalse(set(groups[train]) & set(groups[holdout]))
        self.assertTrue(np.all(train ^ holdout))


if __name__ == '__main__':
    unittest.main()
