"""Search objectives, genuine Hybrid A* kinematics, and shared launcher/GUI execution."""

from dataclasses import replace
import math
import json
import os
import time
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtWidgets import QApplication
import numpy as np

from src.agents.astar_agent import AStarAgent
from src.agents.graph_search import breadth_first_search, dijkstra_search
from src.agents.hybrid_astar import (
    PublicPlanningScene, hybrid_astar_search, load_hybrid_config, pose_key, rollout_primitive, wrap_angle,
)
from src.agents.planning_graph import FEATURE_NAMES, PlanningGraph, astar_search, exact_cost_to_goal, load_astar_config
from src.launcher.cases import get_default_project_root
from src.launcher.models import LaunchRequestV1, LauncherMode
from src.launcher.preflight import run_preflight
from src.launcher.registry import (
    HISTORICAL_FIXTURE_REGISTRY_SHA256, build_canonical_agent_registry, build_historical_fixture_registry,
    compute_canonical_registry_sha256,
)
from src.launcher.resolver import resolve_experiment_plan
from src.platform import AgentPublicEpisodeContext, PINNED_METADRIVE_COMMIT, PINNED_METADRIVE_VERSION, get_action_adapter
from src.platform.agent_input import (
    AgentInputV1, CoreObservationV1, InputProfileId, RouteWaypointV1, TaskContextV1, TrafficActorV1, TrafficContextV1,
)
from src.workbench.main_window import MainWindow
from src.workbench.results_repository import RunArtifactRepository

SEARCH_IDS = {"stage2_bfs", "stage2_dijkstra", "stage2_astar", "stage2_hybrid_astar"}


def public_input(radius=None, actors=(), lane_width=3.5):
    distances = np.arange(1, 21) * 2.5
    if radius is None:
        route = np.column_stack((distances, np.zeros(20), np.zeros(20)))
    else:
        route = np.column_stack((radius * np.sin(distances / radius), radius * (1 - np.cos(distances / radius)), distances / radius))
    task = TaskContextV1(current_lane_width=lane_width,
                         waypoints=tuple(RouteWaypointV1(float(d), *map(float, point)) for d, point in zip(distances, route)),
                         waypoints_array=route, validity_mask=np.ones(20, dtype=bool))
    traffic = TrafficContextV1.empty()
    if actors:
        array = np.zeros((8, 7))
        for index, actor in enumerate(actors):
            array[index] = actor.to_tuple()
        traffic = replace(traffic, actors=tuple(actors) + traffic.actors[len(actors):], actors_array=array,
                          active_count=len(actors), validity_mask=np.arange(8) < len(actors))
    return AgentInputV1(InputProfileId.STATE_DECISION_V1, CoreObservationV1(np.zeros(259)), traffic, task, 0)


class TestGraphSearchAlgorithms(unittest.TestCase):
    def graph(self):
        # BFS's two-edge route costs 18; the three-edge route costs 3.
        return PlanningGraph(np.zeros((5, 2)), (((1, 9.), (2, 1.)), ((4, 9.),), ((3, 1.),), ((4, 1.),), ()),
                             np.zeros((5, len(FEATURE_NAMES))), 4, 50.)

    def test_bfs_minimizes_hops_while_dijkstra_and_astar_minimize_cost(self):
        graph = self.graph()
        bfs = breadth_first_search(graph)
        self.assertEqual(bfs.path, (0, 1, 4))
        self.assertEqual(bfs.cost, 18.)
        for search in (dijkstra_search, astar_search):
            result = search(graph)
            self.assertEqual(result.path, (0, 2, 3, 4))
            self.assertEqual(result.cost, exact_cost_to_goal(graph)[0])

    def test_cycles_disconnected_goals_and_expansion_budget(self):
        graph = self.graph()
        graph = replace(graph, edges=(((1, 1.),), ((0, 1.),), (), (), ()))
        for search in (breadth_first_search, dijkstra_search):
            self.assertFalse(search(graph).found)
            self.assertEqual(search(graph).expansions, 2)
            self.assertEqual(search(graph, max_expansions=0).expansions, 0)
            self.assertEqual(search(self.graph(), max_expansions=1).expansions, 1)
            same_node = search(replace(graph, goal=0))
            self.assertEqual((same_node.path, same_node.cost), ((0,), 0.))
            for cost in (-1., float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    search(replace(graph, edges=(((1, cost),), (), (), (), ())))

    def test_dijkstra_and_astar_agree_on_many_nonnegative_weighted_graphs(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            edges = tuple(tuple((j, float(rng.uniform(.1, 10))) for j in range(i + 1, 12)
                                if j == i + 1 or rng.random() < .2) for i in range(12))
            graph = PlanningGraph(np.zeros((12, 2)), edges, np.zeros((12, len(FEATURE_NAMES))), 11, 50.)
            self.assertAlmostEqual(dijkstra_search(graph).cost, exact_cost_to_goal(graph)[0])
            self.assertAlmostEqual(astar_search(graph).cost, dijkstra_search(graph).cost)


class TestHybridAStar(unittest.TestCase):
    def setUp(self):
        self.config = load_hybrid_config()
        self.planner = load_astar_config()["planner"]

    def test_motion_primitives_follow_bicycle_arcs_and_keep_continuous_poses(self):
        pose = np.array([.13, -.17, .21])
        for steering in self.config["steering_levels"]:
            arc = rollout_primitive(pose, steering, self.config)
            curvature = math.tan(steering * math.radians(self.config["max_steering_deg"])) / self.config["wheelbase_m"]
            self.assertAlmostEqual(wrap_angle(arc[-1, 2] - pose[2]), curvature * self.config["primitive_length_m"])
            self.assertLessEqual(float(np.max(np.linalg.norm(np.diff(arc[:, :2], axis=0), axis=1))),
                                 self.config["collision_sample_spacing_m"] + 1e-9)
            # Every chord points along the arc's midpoint heading.
            headings = np.arctan2(np.diff(arc[:, 1]), np.diff(arc[:, 0]))
            np.testing.assert_allclose(wrap_angle(headings - (arc[:-1, 2] + arc[1:, 2]) / 2), 0., atol=1e-10)
        self.assertNotEqual(pose_key([0, 0, 0], self.config), pose_key([0, 0, .3], self.config))
        self.assertEqual(pose_key([0, 0, math.pi], self.config), pose_key([0, 0, -math.pi], self.config))

    def test_straight_and_curved_routes_reach_position_and_heading_goal_regions(self):
        for radius in (None, 30.):
            with self.subTest(radius=radius):
                inputs = public_input(radius)
                result = hybrid_astar_search(inputs, self.planner, self.config)
                self.assertTrue(result.found, (radius, result.expansions))
                self.assertLessEqual(result.expansions, self.planner["max_expansions"])
                self.assertTrue(PublicPlanningScene(inputs, self.planner, self.config).at_goal(result.poses[-1]))
                self.assertTrue(PublicPlanningScene(inputs, self.planner, self.config).is_clear(result.poses))
                self.assertGreater(len(result.steering_sequence), 0)
                self.assertTrue(all(abs(s) <= .6 for s in result.steering_sequence))
                if radius:
                    self.assertGreater(abs(result.poses[-1, 2]), .3)
                    self.assertTrue(any(s != 0 for s in result.steering_sequence))
                with self.assertRaises(ValueError):
                    result.poses[0, 0] = 3
                repeat = hybrid_astar_search(inputs, self.planner, self.config)
                np.testing.assert_array_equal(result.poses, repeat.poses)

    def test_footprint_checks_actor_orientation_and_intermediate_arc_samples(self):
        actor = TrafficActorV1(0., 2.5, 0., 0., 0., 6., 1.)
        straight_actor = PublicPlanningScene(public_input(actors=(actor,)), self.planner, self.config)
        rotated_actor = PublicPlanningScene(public_input(actors=(replace(actor, relative_heading=math.pi / 2),)), self.planner, self.config)
        self.assertTrue(straight_actor.is_clear(np.zeros((1, 3))))
        self.assertFalse(rotated_actor.is_clear(np.zeros((1, 3))))
        toy = dict(self.config, vehicle_length_m=.2, vehicle_width_m=.2, obstacle_margin_m=0.)
        obstacle = TrafficActorV1(.75, 0., 0., 0., 0., .1, .1)
        scene = PublicPlanningScene(public_input(actors=(obstacle,)), self.planner, toy)
        arc = rollout_primitive(np.zeros(3), 0., toy)
        self.assertTrue(scene.is_clear(arc[[0, -1]]))
        self.assertFalse(scene.is_clear(arc))

    def test_blocked_routes_empty_routes_and_goal_heading_do_not_claim_a_path(self):
        obstacle = TrafficActorV1(8., 0., 0., 0., 0., 2., 5.)
        result = hybrid_astar_search(public_input(actors=(obstacle,)), self.planner, self.config)
        self.assertFalse(result.found)
        self.assertLessEqual(result.expansions, 256)
        empty = replace(public_input(), task_context=TaskContextV1.empty())
        self.assertFalse(hybrid_astar_search(empty, self.planner).found)
        self.assertEqual(hybrid_astar_search(public_input(), self.planner, max_expansions=0).expansions, 0)
        inputs = public_input()
        route = inputs.task_context.waypoints_array.copy()
        route[:, 2] = math.pi
        waypoints = tuple(replace(w, relative_heading=math.pi) for w in inputs.task_context.waypoints)
        inputs = replace(inputs, task_context=replace(inputs.task_context, waypoints_array=route, waypoints=waypoints))
        self.assertFalse(hybrid_astar_search(inputs, self.planner, max_expansions=64).found)


class TestSearchPlannerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_registry_parity_public_inputs_deterministic_actions_and_checkpoint_free_preflight(self):
        registry = build_canonical_agent_registry()
        self.assertEqual(compute_canonical_registry_sha256(build_historical_fixture_registry()), HISTORICAL_FIXTURE_REGISTRY_SHA256)
        for agent_id in sorted(SEARCH_IDS):
            with self.subTest(agent=agent_id):
                reg = registry.get(agent_id)
                agent = registry.get_factory(agent_id)()
                for key, value in agent.descriptor.to_dict().items():
                    self.assertEqual(value, getattr(reg, key))
                self.assertFalse(reg.requires_checkpoint)
                parameters = registry.get_factory(agent_id).experiment_parameters()
                self.assertEqual(parameters["search_method"], agent.search_method)
                self.assertEqual(parameters["shared_planner"]["max_expansions"], 256)
                with self.assertRaises(RuntimeError):
                    agent.act(public_input())
                agent.reset(AgentPublicEpisodeContext(action_adapter_id="discrete9_lowbranch_v1"))
                first = agent.act(public_input()).to_dict()
                self.assertEqual(first, agent.act(public_input()).to_dict())
                get_action_adapter(reg.action_adapter_id).to_canonical(first["action_payload"])
                self.assertEqual(agent.act(replace(public_input(), task_context=TaskContextV1.empty())).action_payload, 1)
                agent.close()
                with self.assertRaises(RuntimeError):
                    agent.act(public_input())
                for mode in (LauncherMode.SANDBOX, LauncherMode.TEST):
                    plan = resolve_experiment_plan(LaunchRequestV1(mode, agent_id))
                    report = run_preflight(plan, custom_git_provenance={"git_commit_sha": "test", "git_worktree_dirty": False},
                                           custom_environment_provenance={"metadrive_version": PINNED_METADRIVE_VERSION,
                                                                          "metadrive_commit": PINNED_METADRIVE_COMMIT})
                    self.assertEqual(report.can_execute, mode == LauncherMode.SANDBOX)

    def test_group_button_selects_four_fixed_train_jobs_and_invalidates_previous_plan(self):
        window = MainWindow()
        self.addCleanup(window.close)
        widget = window.benchmark_widget
        window.campaign_controller.validated_fingerprint = widget.preview.fingerprint
        widget.btn_search_group.click()
        self.assertEqual(widget.preview.profile.profile_id, "train_demo_v1")
        self.assertEqual({j.request.agent_id for j in widget.preview.jobs}, SEARCH_IDS)
        self.assertFalse(widget.preview.blocking_reasons)
        self.assertTrue(all(j.request.agent_seed is None for j in widget.preview.jobs))
        self.assertFalse(widget.btn_run.isEnabled())
        widget.set_busy(True)
        self.assertFalse(widget.btn_search_group.isEnabled())

    def test_real_gui_runs_all_four_on_the_same_train_case_and_verifies_saved_results(self):
        window = MainWindow()
        def cleanup():
            if window._runner.is_running:
                window._runner.force_terminate_and_wait()
            window.close()
        self.addCleanup(cleanup)
        widget = window.benchmark_widget
        widget.btn_search_group.click()
        widget.check_show_demo.setChecked(False)
        controller = window.campaign_controller
        def wait_until_idle():
            deadline = time.monotonic() + 360
            while controller.active and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(.01)
            self.assertFalse(controller.active, controller.rows)
        widget.btn_plan.click()
        wait_until_idle()
        self.assertEqual(controller.status, "READY", controller.rows)
        widget.btn_run.click()
        wait_until_idle()
        self.assertEqual(controller.status, "COMPLETE", controller.rows)
        self.assertEqual({r["agent_id"] for r in widget._result_rows}, SEARCH_IDS)
        repository = RunArtifactRepository(get_default_project_root() / "runs")
        for row in widget._result_rows:
            snapshot = repository.load_run_snapshot(row["run_id"])
            self.assertEqual(snapshot.integrity_status.value, "VERIFIED")
            self.assertFalse(snapshot.canonical_run)
            self.assertEqual(snapshot.episode_rows[0]["case_id"], "sandbox/Easy/SCS/0/0")
            manifest = json.loads((get_default_project_root() / "runs" / row["run_id"] / "run_manifest.json").read_text(encoding="utf-8"))
            planner = manifest["config"]["algorithm_hyperparameters"]["planner"]
            self.assertEqual(planner, build_canonical_agent_registry().get_factory(row["agent_id"]).experiment_parameters())


if __name__ == "__main__":
    unittest.main()
