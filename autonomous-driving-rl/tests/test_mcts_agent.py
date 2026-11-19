"""MCTS mechanics, physical model and public-agent boundary; no renderer required."""

from dataclasses import replace
import importlib.util
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from src.agents.MCTS_agent import MCTSAgent, MCTSNode
from src.agents.mcts_model import (
    MCTSConfig, ModelTransition, PlanningScene, PlanningState, load_mcts_config, predict_step,
)
from src.platform.agent import AgentPublicEpisodeContext
from src.platform.agent_input import (
    AgentInputV1, CoreObservationV1, InputProfileId, RouteWaypointV1, TaskContextV1,
    TrafficActorV1, TrafficContextV1,
)
from src.platform.action_adapter import get_action_adapter


def public_input(speed=0., radius=None, actors=(), lidar=None, empty_route=False):
    features = np.full(259, .5, dtype=np.float32)
    features[3] = (speed * 3.6 + 1) / 81
    features[19:] = 1. if lidar is None else lidar
    distances = np.arange(1, 21) * 2.5
    route = np.column_stack((distances, np.zeros(20), np.zeros(20)))
    if radius is not None:
        route = np.column_stack((radius * np.sin(distances / radius),
                                 radius * (1 - np.cos(distances / radius)), distances / radius))
    task = TaskContextV1.empty() if empty_route else TaskContextV1(
        waypoints=tuple(RouteWaypointV1(float(d), *map(float, point)) for d, point in zip(distances, route)),
        waypoints_array=route, validity_mask=np.ones(20, dtype=bool),
    )
    traffic = TrafficContextV1.empty()
    if actors:
        array = np.zeros((8, 7), dtype=np.float32)
        for index, actor in enumerate(actors):
            array[index] = actor.to_tuple()
        traffic = replace(traffic, actors=tuple(actors) + traffic.actors[len(actors):],
                          actors_array=array, active_count=len(actors), validity_mask=np.arange(8) < len(actors))
    return AgentInputV1(InputProfileId.STATE_DECISION_V1, CoreObservationV1(features), traffic, task, 0)


def make_scene(inputs=None, config=None):
    agent = MCTSAgent(config)
    return PlanningScene(agent._read_planning_input(inputs or public_input()), agent.config)


class TestMCTSModel(unittest.TestCase):
    def test_configuration_rejects_invalid_budgets_units_and_nonfinite_values(self):
        for kwargs in ({"num_iterations": 8}, {"max_depth": 0}, {"gamma": 1.1},
                       {"wheelbase_m": 0}, {"steering_sign": 0}, {"max_steering_deg": 90},
                       {"rollout_random_probability": -1}, {"progress_weight": float("nan")},
                       {"num_iterations": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MCTSConfig(**kwargs)

    def test_observation_speed_is_decoded_to_m_s_and_input_is_copied(self):
        inputs = public_input(speed=10.)
        planning_input = MCTSAgent()._read_planning_input(inputs)
        self.assertAlmostEqual(planning_input["speed_m_s"], 10., places=5)
        planning_input["waypoints"][0, 0] = -100
        self.assertEqual(inputs.task_context.waypoints_array[0, 0], 2.5)

    def test_actor_relative_velocity_is_converted_once_in_the_fixed_root_frame(self):
        actor = TrafficActorV1(20., 0., -2., 0., 0., 4.5, 1.8)
        scene = make_scene(public_input(speed=10., actors=(actor,)))
        self.assertAlmostEqual(scene.actor_velocities[0, 0], 8., places=5)
        np.testing.assert_allclose(scene.actors[0, :2] + 2 * scene.actor_velocities[0], [36., 0.], atol=1e-5)

    def test_straight_motion_is_time_based_and_does_not_modify_parent(self):
        config = replace(load_mcts_config(), acceleration_m_s2=0., drag_per_s=0.)
        scene = make_scene(config=config)
        parent = PlanningState(speed=5.)
        result = predict_step(scene, parent, 4, .1)
        self.assertAlmostEqual(result.state.x, .5)
        self.assertAlmostEqual(result.state.y, 0.)
        self.assertAlmostEqual(result.state.time_s, .1)
        self.assertEqual(parent, PlanningState(speed=5.))
        repeat = predict_step(scene, parent, 4, .1)
        self.assertEqual(result, repeat)

    def test_braking_never_reverses_and_standstill_does_not_turn(self):
        scene = make_scene()
        result = predict_step(scene, PlanningState(speed=.1), 1, .5)
        self.assertEqual(result.state.speed, 0.)
        self.assertGreaterEqual(result.state.x, 0.)
        self.assertLess(result.state.x, .01)
        for action in (0, 1, 2, 3, 4, 5):
            result = predict_step(scene, PlanningState(), action, .1)
            self.assertEqual((result.state.x, result.state.y, result.state.yaw), (0., 0., 0.))

    def test_bicycle_arcs_have_expected_heading_and_mirror_symmetry(self):
        config = replace(load_mcts_config(), drag_per_s=0.)
        scene = make_scene(config=config)
        left = predict_step(scene, PlanningState(speed=5.), 5, .1)
        right = predict_step(scene, PlanningState(speed=5.), 3, .1)
        expected = .5 * math.tan(.6 * math.radians(config.max_steering_deg)) / config.wheelbase_m
        self.assertAlmostEqual(left.state.yaw, expected)
        self.assertAlmostEqual(left.state.x, right.state.x)
        self.assertAlmostEqual(left.state.y, -right.state.y)
        self.assertGreater(left.state.y, 0.)

    def test_collision_is_checked_between_endpoints_and_uses_actor_orientation(self):
        config = replace(load_mcts_config(), vehicle_length_m=.2, vehicle_width_m=.2, drag_per_s=0.)
        actor = TrafficActorV1(2., 0., -10., 0., 0., .1, .1)
        scene = make_scene(public_input(speed=10., actors=(actor,)), config)
        self.assertIsNone(scene.first_failure([PlanningState(speed=10.)])[0])
        # A half-second jump would pass the obstacle; sampled checking catches it.
        result = predict_step(scene, PlanningState(speed=10.), 4, .5)
        self.assertEqual(result.reason, "COLLISION")
        self.assertLess(result.state.time_s, .5)
        actor = TrafficActorV1(0., 2.5, 0., 0., 0., 6., 1.)
        self.assertIsNone(make_scene(public_input(actors=(actor,))).first_failure([PlanningState()])[0])
        rotated = replace(actor, relative_heading=math.pi / 2)
        self.assertEqual(make_scene(public_input(actors=(rotated,))).first_failure([PlanningState()])[1], "COLLISION")

    def test_lidar_no_hits_are_not_obstacles_but_static_hits_are(self):
        self.assertEqual(len(make_scene().static_hits), 0)
        lidar = np.ones(240)
        lidar[0] = 1. / 50  # Hit 1m ahead, inside the nominal vehicle footprint.
        scene = make_scene(public_input(lidar=lidar))
        self.assertEqual(scene.first_failure([PlanningState()])[1], "COLLISION")

    def test_lidar_hits_on_tracked_actor_are_not_kept_as_static_ghosts(self):
        lidar = np.ones(240)
        lidar[0] = 8. / 50
        actor = TrafficActorV1(10., 0., 0., 0., 0., 4.5, 1.8)
        self.assertEqual(len(make_scene(public_input(actors=(actor,), lidar=lidar)).static_hits), 0)

    def test_guide_follows_curvature_and_brakes_for_close_stationary_actor(self):
        scene = make_scene(public_input(speed=5., radius=20.))
        self.assertEqual(scene.guided_action(PlanningState(speed=2.)) % 3, 2)
        actor = TrafficActorV1(9., 0., -5., 0., 0., 4.5, 1.8)
        scene = make_scene(public_input(speed=5., actors=(actor,)))
        self.assertEqual(scene.guided_action(PlanningState(speed=5.)) // 3, 0)

    def test_lookahead_end_is_a_model_boundary_not_success(self):
        scene = make_scene(config=replace(load_mcts_config(), drag_per_s=0.))
        result = predict_step(scene, PlanningState(x=49.9, speed=2.), 4, .1)
        self.assertTrue(result.terminal)
        self.assertEqual(result.reason, "PUBLIC_LOOKAHEAD_END")
        self.assertNotIn("SUCCESS", result.reason)

    def test_body_outside_route_corridor_is_a_terminal_model_failure(self):
        scene = make_scene()
        result = predict_step(scene, PlanningState(y=2., speed=1.), 4, .1)
        self.assertTrue(result.terminal)
        self.assertEqual(result.reason, "OUTSIDE_CORRIDOR")
        self.assertLess(result.reward, -scene.config.outside_penalty)


class TestMCTSSearch(unittest.TestCase):
    def setUp(self):
        self.config = replace(load_mcts_config(), num_iterations=18, max_depth=5)
        self.agent = MCTSAgent(self.config)
        self.context = AgentPublicEpisodeContext(action_adapter_id="discrete9_lowbranch_v1")
        self.agent.reset(self.context, agent_seed=101)

    def test_backup_discounts_each_edge_reward_once_and_increments_visits_once(self):
        agent = MCTSAgent(replace(self.config, gamma=.5))
        root = MCTSNode(PlanningState())
        child = MCTSNode(PlanningState(), reward_from_parent=1.)
        leaf = MCTSNode(PlanningState(), reward_from_parent=2.)
        agent._backpropagate([root, child, leaf], 4.)
        self.assertEqual(leaf.total_return, 4.)  # 2 + .5*4
        self.assertEqual(child.total_return, 3.)  # 1 + .5*4
        self.assertEqual(root.total_return, 3.)
        self.assertEqual([n.visits for n in (root, child, leaf)], [1, 1, 1])

    def test_rollout_uses_remaining_depth_and_terminal_nodes_do_not_step(self):
        scene = make_scene(config=self.config)
        node = MCTSNode(PlanningState(), depth=3)
        with patch.object(self.agent, "_predict_step", return_value=ModelTransition(node.state, 1., False)) as step:
            self.assertAlmostEqual(self.agent._rollout(scene, node), 1. + self.config.gamma)
            self.assertEqual(step.call_count, 2)
            step.reset_mock()
            self.assertEqual(self.agent._rollout(scene, replace(node, terminal=True)), 0.)
            step.assert_not_called()

    def test_real_tree_avoids_a_delayed_loss_in_a_small_known_model(self):
        agent = MCTSAgent(replace(self.config, num_iterations=128, max_depth=2, gamma=.95))
        agent.reset(self.context, agent_seed=101)

        def toy_step(scene, state, action):
            agent._model_steps += 1
            if state.time_s == 0:
                if action == 7:
                    return ModelTransition(PlanningState(x=7., time_s=.1), 1., False)
                if action == 1:
                    return ModelTransition(PlanningState(x=1., time_s=.1), 0., False)
                return ModelTransition(PlanningState(time_s=.1), -4., True)
            return ModelTransition(replace(state, time_s=.2), -10. if state.x == 7 else 2., True)

        with patch.object(agent, "_predict_step", side_effect=toy_step):
            decision = agent.act(public_input())
        self.assertEqual(decision.action_payload, 1)
        self.assertEqual(sum(decision.diagnostics["root_action_visits"]), 128)
        self.assertTrue(all(n > 0 for n in decision.diagnostics["root_action_visits"]))
        self.assertLessEqual(decision.diagnostics["model_steps"], 128 * 2)

    def test_same_seed_restarts_actions_diagnostics_and_episode_counter(self):
        first = [self.agent.act(public_input()).to_dict() for _ in range(2)]
        self.agent.reset(self.context, agent_seed=202)
        self.agent.act(public_input())
        self.agent.reset(self.context, agent_seed=101)
        self.assertEqual(first, [self.agent.act(public_input()).to_dict() for _ in range(2)])
        self.assertEqual(first[0]["diagnostics"]["step_counter"], 1)

    def test_action_diagnostics_budget_and_public_input_remain_valid(self):
        inputs = public_input()
        before = inputs.to_dict()
        decision = self.agent.act(inputs)
        self.assertEqual(decision.diagnostics["planner_status"], "ACTION_SELECTED")
        self.assertEqual(decision.diagnostics["search_iterations"], self.config.num_iterations)
        self.assertLessEqual(decision.diagnostics["model_steps"], self.config.num_iterations * self.config.max_depth)
        get_action_adapter("discrete9_lowbranch_v1").to_canonical(decision.action_payload)
        json.dumps(decision.to_dict(), allow_nan=False)
        self.assertEqual(inputs.to_dict(), before)

    def test_missing_route_contact_and_invalid_lifecycle_have_explicit_fallbacks(self):
        self.assertEqual(self.agent.act(public_input(empty_route=True)).action_payload, 1)
        self.assertEqual(self.agent._last_search_diagnostics["planner_status"], "NO_PUBLIC_ROUTE")
        decision = self.agent.act(public_input(lidar=np.zeros(240)))
        self.assertEqual(decision.action_payload, 1)
        self.assertEqual(decision.diagnostics["planner_status"], "ROOT_MODEL_COLLISION")
        self.agent.close()
        with self.assertRaises(RuntimeError):
            self.agent.act(public_input())
        with self.assertRaises(ValueError):
            self.agent.reset(AgentPublicEpisodeContext())

    def test_all_immediately_failing_root_actions_return_explicit_brake_fallback(self):
        transition = ModelTransition(PlanningState(), -40., True, "COLLISION")
        with patch.object(self.agent, "_predict_step", return_value=transition):
            decision = self.agent.act(public_input())
        self.assertEqual(decision.action_payload, 1)
        self.assertEqual(decision.diagnostics["planner_status"], "NO_SAFE_MODEL_ACTION")

    @unittest.skipUnless(importlib.util.find_spec("metadrive"), "MetaDrive needed to import launcher")
    def test_registry_descriptor_parity_and_parameter_provenance(self):
        from src.launcher.registry import build_canonical_agent_registry
        registry = build_canonical_agent_registry()
        registration = registry.get("stage3_mcts")
        agent = registry.get_factory("stage3_mcts")()
        for name in ("agent_id", "agent_version", "input_profile_id", "action_adapter_id",
                     "inference_stochasticity", "stateful_within_episode", "method_family"):
            self.assertEqual(getattr(registration, name), getattr(agent.descriptor, name))
        self.assertFalse(registration.benchmark_eligible)
        self.assertTrue(registration.sandbox_eligible)
        parameters = registry.get_factory("stage3_mcts").experiment_parameters()
        self.assertEqual(parameters["parameters"]["num_iterations"], 64)
        self.assertEqual(len(parameters["config_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
