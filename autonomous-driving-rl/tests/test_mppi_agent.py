"""MPPI mechanics, dynamics, perception geometry and the public policy boundary."""

from dataclasses import replace
import importlib.util
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from src.agents.MPPI import BicycleModel, MPPI, MPPIAgent, MPPIConfig, MPPIScene, cost_fn, load_mppi_config
from src.platform.action_adapter import get_action_adapter
from src.platform.agent import AgentPublicEpisodeContext
from src.platform.agent_input import (
    AgentInputV1, CoreObservationV1, InputProfileId, RouteWaypointV1,
    TaskContextV1, TrafficActorV1, TrafficContextV1,
)
from src.platform.protocol import canonical_json_sha256


def public_input(speed=0., radius=None, offset=0., actors=(), lidar=None, empty_route=False):
    features = np.full(259, .5, dtype=np.float32)
    features[3] = (speed * 3.6 + 1) / 81
    features[19:] = 1. if lidar is None else lidar
    distances = np.arange(1, 21) * 2.5
    route = np.column_stack((distances, np.full(20, offset), np.zeros(20)))
    if radius is not None:
        route = np.column_stack((radius * np.sin(distances / radius),
                                 radius * (1 - np.cos(distances / radius)), distances / radius))
    task = TaskContextV1.empty() if empty_route else TaskContextV1(
        waypoints=tuple(RouteWaypointV1(float(d), *map(float, p)) for d, p in zip(distances, route)),
        waypoints_array=route, validity_mask=np.ones(20, dtype=bool),
    )
    traffic = TrafficContextV1.empty()
    if actors:
        array = np.zeros((8, 7), dtype=np.float32)
        for index, actor in enumerate(actors):
            array[index] = actor.to_tuple()
        traffic = replace(traffic, actors=tuple(actors) + traffic.actors[len(actors):], actors_array=array,
                          active_count=len(actors), validity_mask=np.arange(8) < len(actors))
    return AgentInputV1(InputProfileId.STATE_DECISION_V1, CoreObservationV1(features), traffic, task, 0)


class TestBicycleModel(unittest.TestCase):
    def test_config_validation_and_versioned_defaults(self):
        self.assertEqual(load_mppi_config(), MPPIConfig())
        for kwargs in ({"horizon": 0}, {"horizon": True}, {"num_samples": 2}, {"temperature": 0},
                       {"sigma_steering": 0}, {"wheelbase_m": 0}, {"steering_sign": 0},
                       {"max_steering_deg": 90}, {"noise_correlation": 1},
                       {"speed_weight": float("nan")}, {"target_speed_m_s": 100}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MPPIConfig(**kwargs)

    def test_straight_motion_uses_dt_and_preserves_inputs(self):
        cfg = MPPIConfig(drag_per_s=0.)
        model = BicycleModel(cfg, dt=.2)
        x0, U = np.array([1., 2., .0, 5.]), np.zeros((3, 4, 2))
        result = model.rollout(x0, U)
        np.testing.assert_allclose(result[:, -1], [[5., 2., 0., 5.]] * 3)
        np.testing.assert_array_equal(x0, [1., 2., 0., 5.])
        self.assertFalse(U.any())

    def test_acceleration_braking_stop_distance_and_speed_limit(self):
        model = BicycleModel(MPPIConfig(drag_per_s=0.), dt=.5)
        result = model.rollout([0., 0., 0., 0.], np.tile([0., 1.], (1, 1, 1)))
        np.testing.assert_allclose(result[0, 0], [.5, 0., 0., 2.], atol=1e-10)
        brake = model.rollout([0., 0., 0., .1], np.tile([1., -1.], (1, 3, 1)))[0]
        self.assertTrue(np.all(brake[:, 3] == 0))
        self.assertAlmostEqual(np.hypot(*brake[0, :2]), .1 ** 2 / (2 * 8), places=8)
        np.testing.assert_allclose(brake[1:], [brake[0]] * 2)
        fast = model.rollout([0., 0., 0., 22.2], np.tile([0., 1.], (1, 2, 1)))
        self.assertLessEqual(fast[:, :, 3].max(), 80 / 3.6)

    def test_bicycle_arcs_and_steering_sign_are_mirror_symmetric(self):
        cfg = MPPIConfig(drag_per_s=0.)
        U = np.array([[[.4, 0.]], [[-.4, 0.]]])
        result = BicycleModel(cfg).rollout([0., 0., 0., 5.], U)[:, 0]
        yaw = .5 * math.tan(math.radians(40) * .4) / cfg.wheelbase_m
        self.assertAlmostEqual(result[0, 2], yaw)
        self.assertAlmostEqual(result[0, 0], result[1, 0])
        self.assertAlmostEqual(result[0, 1], -result[1, 1])
        self.assertGreater(result[0, 1], 0.)
        reversed_sign = BicycleModel(replace(cfg, steering_sign=-1)).rollout([0., 0., 0., 5.], U)
        np.testing.assert_allclose(reversed_sign[0], result[1:2])

    def test_invalid_dt_states_and_controls_are_rejected(self):
        for dt in (0., -1., float("nan")):
            with self.assertRaises(ValueError):
                BicycleModel(dt=dt)
        for x0, U in (([0., 0., 0., -1.], np.zeros((1, 2, 2))),
                      ([0., 0., 0., 1.], np.ones((1, 2, 2)) * 2),
                      ([0., 0., 0., 1.], np.zeros((2, 2)))):
            with self.assertRaises(ValueError):
                BicycleModel().rollout(x0, U)


class TestMPPISceneAndCost(unittest.TestCase):
    def setUp(self):
        self.cfg = MPPIConfig(horizon=5, num_samples=128)

    def scene(self, **kwargs):
        speed = kwargs.get("speed", 0.)
        return MPPIScene(public_input(**kwargs), speed, self.cfg)

    def test_route_segment_projection_preserves_lateral_offset(self):
        scene = self.scene(offset=1.2)
        progress, distance, heading = scene.project([[0., 0.], [1.25, 1.2], [8., 2.2]])
        np.testing.assert_allclose(distance, [1.2, 0., 1.], atol=1e-6)
        np.testing.assert_allclose(heading, 0.)
        self.assertAlmostEqual(progress[1] - progress[0], 1.25)

    def test_relative_actor_velocity_is_converted_once_and_masked(self):
        actor = TrafficActorV1(20., 0., -2., 0., 0., 4.5, 1.8)
        scene = self.scene(speed=10., actors=(actor,))
        np.testing.assert_allclose(scene.velocities, [[8., 0.]])
        np.testing.assert_allclose(scene.actors[0, :2] + 2 * scene.velocities[0], [36., 0.])
        self.assertEqual(len(scene.actors), 1)

    def test_nohit_lidar_and_tracked_actor_hits_do_not_make_static_ghosts(self):
        self.assertEqual(len(self.scene().static_hits), 0)
        lidar = np.ones(240)
        lidar[0] = 8 / 50
        actor = TrafficActorV1(10., 0., 0., 0., 0., 4.5, 1.8)
        self.assertEqual(len(self.scene(actors=(actor,), lidar=lidar).static_hits), 0)
        scene = self.scene(lidar=lidar)
        np.testing.assert_allclose(scene.static_hits, [[8., 0.]], atol=1e-6)
        self.assertTrue(scene.footprint_flags(np.array([[7., 0., 0., 0.]]), np.zeros(1))[0][0])

    def test_oriented_actor_box_and_vehicle_body_determine_collisions_and_outside(self):
        actor = TrafficActorV1(0., 3., 0., 0., 0., 6., 1.)
        state, times = np.zeros((1, 4)), np.zeros(1)
        self.assertFalse(self.scene(actors=(actor,)).footprint_flags(state, times)[0][0])
        rotated = replace(actor, relative_heading=math.pi / 2)
        self.assertTrue(self.scene(actors=(rotated,)).footprint_flags(state, times)[0][0])
        self.assertFalse(self.scene().footprint_flags(state, times)[1][0])
        state[0, 1] = 1.
        self.assertTrue(self.scene().footprint_flags(state, times)[1][0])

    def test_cost_prefers_lane_tracking_target_speed_and_smooth_controls(self):
        scene = self.scene()
        traj = np.tile([2., 0., 0., 8.], (4, 5, 1))
        traj[1, :, 1], traj[2, :, 3] = 1.5, 0.
        U = np.zeros((4, 5, 2))
        U[3, :, 0] = [1., -1., 1., -1., 1.]
        costs, collisions = cost_fn(traj, U, scene, np.zeros(2), .1)
        self.assertTrue(np.all(costs[1:] > costs[0]))
        self.assertFalse(collisions.any())
        initial_change, _ = cost_fn(traj[:1], U[:1], scene, np.ones(2), .1)
        self.assertAlmostEqual(initial_change[0] - costs[0], 2 * self.cfg.control_change_weight)

    def test_horizon_one_and_end_of_lookahead_remain_finite_without_success(self):
        scene = self.scene()
        traj = np.array([[[60., 0., 0., 8.]]])
        cost, _ = cost_fn(traj, np.zeros((1, 1, 2)), scene, np.zeros(2), .1)
        self.assertTrue(np.isfinite(cost).all())
        self.assertGreater(cost[0], 0.)


class TestMPPIControllerAndAgent(unittest.TestCase):
    def setUp(self):
        self.cfg = MPPIConfig(horizon=10, num_samples=128)
        self.agent = MPPIAgent(self.cfg)
        self.context = AgentPublicEpisodeContext()
        self.agent.reset(self.context, 101)

    def test_seed_repeatability_reset_and_different_seeds(self):
        first = [self.agent.act(public_input()).to_dict() for _ in range(2)]
        self.agent.reset(self.context, 202)
        different = self.agent.act(public_input()).to_dict()
        self.assertNotEqual(first[0]["action_payload"], different["action_payload"])
        self.agent.reset(self.context, 101)
        self.assertFalse(self.agent._controller.U.any())
        self.assertEqual(first, [self.agent.act(public_input()).to_dict() for _ in range(2)])
        self.assertEqual(first[0]["diagnostics"]["step_counter"], 1)

    def test_actions_diagnostics_and_input_immutability(self):
        inputs = public_input(speed=10.)
        before = inputs.to_dict()
        decision = self.agent.act(inputs)
        get_action_adapter("continuous_box2_v1").to_canonical(decision.action_payload)
        json.dumps(decision.to_dict(), allow_nan=False)
        self.assertAlmostEqual(decision.diagnostics["speed_m_s"], 10., places=5)
        self.assertEqual(inputs.to_dict(), before)
        self.assertTrue(1 <= decision.diagnostics["effective_sample_size"] <= self.cfg.num_samples + 1e-6)

    def test_closed_loop_model_drives_forward_and_turns_toward_curve(self):
        cfg = MPPIConfig()
        agent = MPPIAgent(cfg)
        agent.reset(self.context, 101)
        self.assertGreater(agent.act(public_input()).action_payload[1], 0.)
        agent.reset(self.context, 101)
        self.assertGreater(agent.act(public_input(speed=5., radius=20.)).action_payload[0], 0.)
        agent.reset(self.context, 101)
        self.assertLess(agent.act(public_input(speed=5., radius=-20.)).action_payload[0], 0.)

    def test_close_stationary_actor_changes_acceleration_to_braking(self):
        agent = MPPIAgent()
        actor = TrafficActorV1(10., 0., -5., 0., 0., 4.5, 1.8)
        agent.reset(self.context, 101)
        free = agent.act(public_input(speed=5.)).action_payload[1]
        agent.reset(self.context, 101)
        blocked = agent.act(public_input(speed=5., actors=(actor,))).action_payload[1]
        self.assertLess(blocked, 0.)
        self.assertLess(blocked, free)

    def test_softmax_update_uses_clipped_noise_and_shifts_the_optimized_plan(self):
        cfg = replace(self.cfg, num_samples=3, horizon=2, noise_correlation=0.)
        controller = MPPI(BicycleModel(cfg), 101)
        controller.U[:] = [.9, -.9]
        noise = np.array([[[4., 4.], [4., 4.]]] * 3)
        expected = np.array([[[.9, -.9], [.9, -.9]], [[0., -1.], [0., -1.]], [[1., 1.], [1., 1.]]])
        with patch.object(controller, "rng") as rng, patch("src.agents.MPPI.cost_fn") as score:
            rng.normal.return_value = noise
            score.return_value = (np.zeros(3), np.zeros(3, dtype=bool))
            action = controller.act([0., 0., 0., 0.], MPPIScene(public_input(), 0., cfg))
        np.testing.assert_allclose(action, expected.mean(axis=0)[0])
        np.testing.assert_allclose(controller.U, expected.mean(axis=0))
        self.assertAlmostEqual(controller.diagnostics["effective_sample_size"], 3.)

    def test_colliding_weighted_plan_uses_best_sample_or_explicit_brake(self):
        cfg = replace(self.cfg, num_samples=3, horizon=2)
        scene = MPPIScene(public_input(), 0., cfg)
        controller = MPPI(BicycleModel(cfg), 101)
        with patch.object(controller, "_validated_plan", side_effect=[False, True]), patch("src.agents.MPPI.cost_fn") as score:
            score.return_value = (np.array([1., 0., 2.]), np.zeros(3, dtype=bool))
            np.testing.assert_allclose(controller.act([0., 0., 0., 0.], scene), [0., -1.])
        self.assertEqual(controller.diagnostics["planner_status"], "BEST_SAMPLE_FALLBACK")
        with patch.object(controller, "_validated_plan", return_value=False), patch("src.agents.MPPI.cost_fn") as score:
            score.return_value = (np.zeros(3), np.ones(3, dtype=bool))
            np.testing.assert_allclose(controller.act([0., 0., 0., 0.], scene), [0., -1.])
        self.assertEqual(controller.diagnostics["planner_status"], "NO_COLLISION_FREE_PLAN")

    def test_fine_validation_detects_collision_between_control_endpoints(self):
        cfg = replace(self.cfg, vehicle_length_m=.2, vehicle_width_m=.2, horizon=1, drag_per_s=0.)
        actor = TrafficActorV1(1., 0., -20., 0., 0., .1, .1)
        scene = MPPIScene(public_input(speed=20., actors=(actor,)), 20., cfg)
        controller = MPPI(BicycleModel(cfg), 101)
        plan = np.zeros((1, 2))
        endpoints = controller.m.rollout([0., 0., 0., 20.], plan[None])[0]
        self.assertFalse(scene.footprint_flags(endpoints, np.array([.1]))[0][0])
        self.assertFalse(controller._validated_plan([0., 0., 0., 20.], plan, scene))

    def test_lifecycle_missing_route_collision_and_control_dt(self):
        self.agent._controller.U.fill(.5)
        decision = self.agent.act(public_input(empty_route=True))
        self.assertEqual(decision.action_payload, [0., -1.])
        self.assertEqual(decision.diagnostics["planner_status"], "NO_PUBLIC_ROUTE")
        self.assertFalse(self.agent._controller.U.any())
        decision = self.agent.act(public_input(lidar=np.zeros(240)))
        self.assertEqual(decision.diagnostics["planner_status"], "ROOT_MODEL_COLLISION")
        self.agent.reset(replace(self.context, control_dt_s=.2), 101)
        self.assertAlmostEqual(self.agent.act(public_input()).diagnostics["lookahead_s"], 2.)
        self.agent.close()
        with self.assertRaises(RuntimeError):
            self.agent.act(public_input())
        with self.assertRaises(ValueError):
            self.agent.reset(replace(self.context, action_adapter_id="discrete9_lowbranch_v1"))

    @unittest.skipUnless(importlib.util.find_spec("metadrive"), "MetaDrive needed to import launcher")
    def test_registry_provenance_and_sandbox_vs_benchmark_preflight(self):
        from src.launcher.models import LaunchRequestV1, LauncherMode
        from src.launcher.preflight import run_preflight
        from src.launcher.registry import build_canonical_agent_registry
        from src.launcher.resolver import resolve_experiment_plan
        from src.platform import PINNED_METADRIVE_COMMIT, PINNED_METADRIVE_VERSION
        registry = build_canonical_agent_registry()
        registration = registry.get("stage3_mppi")
        factory = registry.get_factory("stage3_mppi")
        for name, value in factory().descriptor.to_dict().items():
            self.assertEqual(getattr(registration, name), value)
        params = factory.experiment_parameters()
        digest = params.pop("config_sha256")
        self.assertEqual(digest, canonical_json_sha256(params))
        for mode in (LauncherMode.SANDBOX, LauncherMode.VALIDATION, LauncherMode.TEST):
            plan = resolve_experiment_plan(LaunchRequestV1(mode, "stage3_mppi", agent_seed=101))
            report = run_preflight(plan, custom_git_provenance={"git_commit_sha": "test", "git_worktree_dirty": False},
                                   custom_environment_provenance={"metadrive_version": PINNED_METADRIVE_VERSION,
                                                                  "metadrive_commit": PINNED_METADRIVE_COMMIT})
            self.assertEqual(report.can_execute, mode == LauncherMode.SANDBOX)
            if mode == LauncherMode.SANDBOX:
                self.assertEqual(plan.resolved_cases[0].split, "TRAIN")


if __name__ == "__main__":
    unittest.main()
