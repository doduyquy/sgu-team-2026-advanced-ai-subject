"""Scenario and memory regressions for public-data P2/P3 rules."""

from dataclasses import replace
import math
from types import SimpleNamespace
import unittest

import numpy as np

from src.agents.rule_based_agent import Stage1RuleBasedAgent
from src.agents.rule_based_memory import ActorTracker, LocalFrame, rotation, ObservedTrack
from src.agents.rule_based_traffic import RouteCorridor, assess_rear_side_escape
from src.platform.agent import AgentPublicEpisodeContext, build_agent_contract_core
from src.platform.agent_input import (EgoMotionV1, RoutePassageV1, extract_ego_motion,
                                      extract_route_passage, TaskContextV1,
                                      _roundabout_yield_offset, build_agent_input, InputProfileId,
                                      ActorPathV1, extract_actor_motion_paths, TrafficActorV1)
from tests.test_rule_based_agent import make_input
from tests.test_rule_based_corridor import Lane


def passage_input(entry=4, exit=30, speed=0, actors=(), step=0, motion=None,
                  branch="left_connector", kind="intersection"):
    inp = make_input(speed=speed, actors=actors)
    passage = RoutePassageV1("entry", branch, kind, entry, exit, "left")
    return replace(inp, step_index=step, task_context=replace(inp.task_context, route_passage=passage),
                   ego_motion=motion or EgoMotionV1(0, 0, 0, speed / 3.6, 0, 0))


class TestLocalMeasurements(unittest.TestCase):
    def test_signed_ego_motion_in_current_rotated_frame(self):
        lane = Lane((0, 0), (100, 0), ("a", "b", 0))
        vehicle = SimpleNamespace(position=(1, 0), last_position=(0, 0),
                                  heading_theta=math.pi / 2, last_heading_dir=(1, 0), velocity=(2, 0))
        result = extract_ego_motion(vehicle, SimpleNamespace(current_lane=lane))
        self.assertAlmostEqual(result.displacement_x_m, 0, places=6)
        self.assertAlmostEqual(result.displacement_y_m, -1)
        self.assertAlmostEqual(result.heading_delta_rad, math.pi / 2)
        self.assertAlmostEqual(result.lane_heading_error_rad, -math.pi / 2)
        self.assertAlmostEqual(result.velocity_y_mps, -2)

    def test_nonfinite_motion_and_invalid_passage_are_rejected(self):
        with self.assertRaises(ValueError):
            EgoMotionV1(0, 0, float("nan"), 0, 0, 0)
        with self.assertRaises(ValueError):
            RoutePassageV1("a", "b", "intersection", 10, 5, "left")

    def test_public_schema_declares_new_fields_and_masking(self):
        schema = build_agent_contract_core()
        self.assertEqual(schema["metadata"]["spec_version"], "1.3.0")
        self.assertIn("ego_motion", schema["runtime_dataclass_schemas"]["AgentInputV1"])
        self.assertTrue(schema["agent_input_schema"]["ego_motion"]["masked_in_core_only"])
        self.assertIsNone(TaskContextV1.empty().route_passage)
        masked = build_agent_input(np.ones(259), None, [], None, None, 0, InputProfileId.CORE_ONLY_V1)
        self.assertIsNone(masked.ego_motion)
        self.assertIsNone(masked.task_context.route_passage)

    def test_intersection_has_signed_entry_and_bounded_exit(self):
        first = Lane((0, 0), (20, 0), ("a", "b", 0))
        connector = Lane((20, 0), (57, 0), ("b", "c", 0))
        connector.metadrive_type = "LANE_SURFACE_UNSTRUCTURE"
        exit_lane = Lane((57, 0), (100, 0), ("c", "d", 0))
        graph = {"a": {"b": [first]}, "b": {"c": [connector]}, "c": {"d": [exit_lane]}}
        far = extract_route_passage([first, connector, exit_lane], 0, graph, 50)
        self.assertEqual(far.entry_distance_m, 20)
        self.assertIsNone(far.exit_distance_m)
        near = extract_route_passage([first, connector, exit_lane], 17, graph, 50)
        inside = extract_route_passage([connector, exit_lane], 5, graph, 50)
        self.assertEqual(near.exit_distance_m, 40)
        self.assertEqual(inside.entry_distance_m, -5)
        self.assertEqual(inside.passage_id, near.passage_id)
        self.assertEqual(inside.branch_id, near.branch_id)

    def test_straight_ramp_branch_is_not_an_unbounded_intersection(self):
        road = Lane((0, 0), (100, 0), ("b", "c", 0))
        alternative = Lane((0, 0), (40, 5), ("b", "d", 0))
        graph = {"b": {"c": [road], "d": [alternative]}}
        self.assertIsNone(extract_route_passage([road], 0, graph, 50))

    def test_roundabout_inlet_and_circulation_form_bounded_merge_section(self):
        inlet = Lane((0, 0), (15, 0), ("a", "merge", 0))
        circulation = Lane((15, 0), (30, 0), ("merge", "fork", 0))
        for lane in (inlet, circulation):
            lane.heading_theta_at = lambda s: -s / 20
        graph = {"a": {"merge": [inlet]}, "other": {"merge": [inlet]},
                 "merge": {"fork": [circulation]}, "fork": {"continue": [inlet], "exit": [inlet]}}
        result = extract_route_passage([inlet, circulation], 0, graph, 50)
        self.assertEqual(result.kind, "roundabout_merge")
        self.assertEqual(result.exit_distance_m, 30)

    def test_geometric_yield_boundary_precedes_connector_when_other_lane_crosses_approach(self):
        approach = Lane((0, 0), (30, 0), ("a", "entry", 0))
        crossing = Lane((25, -10), (25, 10), ("other", "exit", 0))
        graph = {"a": {"entry": [approach]}, "other": {"exit": [crossing]}}
        offset = _roundabout_yield_offset(approach, [approach], graph)
        self.assertLessEqual(offset, -5)
        self.assertGreaterEqual(offset, -20)


class TestActorMemory(unittest.TestCase):
    def setUp(self):
        self.config = Stage1RuleBasedAgent()
        self.tracker, self.frame = ActorTracker(), LocalFrame()

    def update(self, actors, step, heading=0, previous_heading=0, position=(0, 0), previous_position=(0, 0)):
        speed = 3.0
        rot = rotation(heading)
        local_actors = []
        for x, y, vx, vy, actor_heading in actors:
            p = rot.T @ (np.array((x, y)) - position)
            v = rot.T @ np.array((vx, vy))
            local_actors.append((p[0], p[1], v[0], v[1], actor_heading - heading))
        displacement = rot.T @ (np.asarray(position) - previous_position)
        motion = EgoMotionV1(float(displacement[0]), float(displacement[1]), heading - previous_heading,
                             speed, 0, -heading)
        inp = replace(make_input(speed=speed * 3.6, actors=local_actors), ego_motion=motion, step_index=step)
        self.frame.update(motion, speed, 0.1, first=step == 0)
        return self.tracker.update(inp, self.frame, speed, 0.1, self.config)

    def test_distance_order_swap_does_not_swap_actor_identity(self):
        first = self.update([(10, 3, 8, 0, 0), (11, -3, 0, 0, 0)], 0)
        second = self.update([(11.6, 3, 8, 0, 0), (11, -3, 0, 0, 0)], 1)
        ids = {round(t.actor.relative_position_y): t.track_id for t in first}
        self.assertEqual(ids, {round(t.actor.relative_position_y): t.track_id for t in second})

    def test_ego_turn_does_not_make_stationary_actor_turn(self):
        self.update([(20, 3, 0, 0, 0)], 0)
        tracks = self.update([(20, 3, 0, 0, 0)], 1, heading=0.2, position=(0.3, 0))
        self.assertEqual(tracks[0].track_id, 1)
        self.assertAlmostEqual(tracks[0].yaw_rate, 0, places=6)

    def test_missed_actor_is_retained_then_expires(self):
        self.update([(20, 3, 0, 0, 0)], 0)
        tracks = self.update([], 1)
        self.assertEqual(len(tracks), 1)
        self.assertAlmostEqual(tracks[0].missed_s, 0.1)
        for step in range(2, 8):
            tracks = self.update([], step)
        self.assertEqual(tracks, [])

    def test_turn_rate_requires_observation_history_and_has_two_models(self):
        for step in range(5):
            tracks = self.update([(20 + 0.3 * step, 3, 3, 0, 0.04 * step)], step)
        self.assertGreater(tracks[0].yaw_rate, 0.2)
        models = tracks[0].predictions(np.array([1.0, 2.0]))
        self.assertEqual(len(models), 2)
        self.assertGreater(float(np.linalg.norm(models[0][0] - models[1][0])), 0.5)

    def test_one_detection_cannot_update_two_tracks(self):
        self.update([(20, 0, 0, 0, 0), (20, 3, 0, 0, 0)], 0)
        tracks = self.update([(20, 1, 0, 0, 0)], 1)
        self.assertEqual(sum(t.missed_s == 0 for t in tracks), 1)
        self.assertEqual(len(tracks), 2)


class ArcLane:
    index = ("b", "turn", 0)
    length = math.pi * 5
    width = 3.5

    def position(self, along, lateral):
        angle = along / 10
        return np.array((20 + (10 + lateral) * math.sin(angle),
                         10 - (10 + lateral) * math.cos(angle)))

    def heading_theta_at(self, along):
        return along / 10

    def local_coordinates(self, position):
        x, y = np.asarray(position) - (20, 10)
        return math.atan2(x, -y) * 10, math.hypot(x, y) - 10


class TestRoadMotionHypotheses(unittest.TestCase):
    def network(self):
        road = Lane((0, 0), (20, 0), ("a", "b", 0))
        straight = Lane((20, 0), (100, 0), ("b", "straight", 0))
        turn = ArcLane()
        exit_lane = Lane((30, 10), (30, 80), ("turn", "exit", 0))
        return SimpleNamespace(graph={"a": {"b": [road]}, "b": {"straight": [straight], "turn": [turn]},
                                      "turn": {"exit": [exit_lane]}})

    def test_all_connected_turns_are_hypotheses_without_actor_navigation(self):
        traffic = make_input(actors=((10, 0, 5),)).traffic_context
        ego = SimpleNamespace(position=(0, 0), heading_theta=0)
        paths, overflow = extract_actor_motion_paths(ego, traffic, self.network())
        self.assertEqual(overflow, 0)
        self.assertEqual(len(paths), 2)
        self.assertTrue(any(path.points[-1][2] > 10 for path in paths))
        self.assertTrue(any(abs(path.points[-1][2]) < 0.01 for path in paths))
        for path in paths:
            self.assertEqual(path.actor_slot, 0)
            self.assertEqual(path.points[0][1:3], (10, 0))
            self.assertTrue(all(math.hypot(p[1], p[2]) <= 50.0001 for p in path.points))

    def test_path_points_are_defensively_immutable_and_require_active_slot(self):
        points = [[0, 10, 0, 0], [5, 15, 0, 0]]
        path = ActorPathV1(0, "road", points)
        points[0][1] = 999
        self.assertEqual(path.points[0][1], 10)
        with self.assertRaises(ValueError):
            replace(make_input().traffic_context, motion_paths=(path,))

    def test_three_coincident_entry_tangents_do_not_drop_the_third_turn(self):
        network = self.network()
        right = ArcLane()
        right.index = ("b", "right", 0)
        right.position = lambda along, lateral: np.array((20 + (10 + lateral) * math.sin(along / 10),
                                                          -10 + (10 + lateral) * math.cos(along / 10)))
        right.heading_theta_at = lambda along: -along / 10
        right.local_coordinates = lambda position: (
            math.atan2(position[0] - 20, position[1] + 10) * 10,
            math.hypot(position[0] - 20, position[1] + 10) - 10)
        network.graph["b"]["right"] = [right]
        network.graph["right"] = {"right_exit": [Lane((30, -10), (30, -80), ("right", "right_exit", 0))]}
        traffic = make_input(actors=((20, 0, 5),)).traffic_context
        paths, overflow = extract_actor_motion_paths(SimpleNamespace(position=(0, 0), heading_theta=0), traffic, network)
        self.assertEqual(overflow, 0)
        self.assertEqual(len(paths), 3)
        self.assertTrue(any(path.points[-1][2] < -10 for path in paths))
        self.assertTrue(any(path.points[-1][2] > 10 for path in paths))

    def test_predictions_do_not_extend_road_geometry_beyond_its_known_end(self):
        inp = make_input(actors=((10, 0, 5),))
        path = ActorPathV1(0, "turn", ((0, 10, 0, 0), (5, 15, 0, 0), (10, 17, 4, math.pi / 2)))
        inp = replace(inp, traffic_context=replace(inp.traffic_context, motion_paths=(path,)))
        frame = LocalFrame()
        tracker = ActorTracker()
        tracks = tracker.update(inp, frame, 15 / 3.6, 0.1, Stage1RuleBasedAgent())
        models = tracks[0].predictions(np.array([1.0, 2.0, 4.0]))
        self.assertEqual(len(models), 2)
        self.assertFalse(models[0][2][-1])  # Free-motion fallback is short when map-matched.
        self.assertFalse(models[1][2][-1])  # Road hypothesis ends with known geometry.
        self.assertAlmostEqual(models[1][0][1, 1], 4, places=3)

    def test_possible_turn_can_reserve_gap_when_straight_velocity_misses_it(self):
        agent = Stage1RuleBasedAgent()
        inp = passage_input(speed=15, exit=40, actors=((20, 10, 5, 0, 0),))
        path = ActorPathV1(0, "possible_right", ((0, 20, 10, 0), (5, 25, 10, 0),
                           (10, 28, 6, -1), (15, 27, 1, -2), (20, 23, -2, -2.5)))
        inp = replace(inp, traffic_context=replace(inp.traffic_context, motion_paths=(path,)))
        decision = agent.act(inp)
        self.assertEqual(decision.diagnostics["junction_reason"], "crossing_occupancy")
        self.assertTrue(decision.diagnostics["junction_conflict_track_ids"])

    def test_hypothesis_overflow_never_authorizes_entry(self):
        agent = Stage1RuleBasedAgent()
        inp = passage_input()
        inp = replace(inp, traffic_context=replace(inp.traffic_context, motion_paths_overflow_count=1))
        for step in range(10):
            decision = agent.act(replace(inp, step_index=step))
        self.assertEqual(decision.diagnostics["junction_reason"], "actor_path_overflow")
        self.assertFalse(decision.diagnostics["junction_branch_locked"])


class TestRearSideConflict(unittest.TestCase):
    def setUp(self):
        self.agent = Stage1RuleBasedAgent()
        self.corridor = RouteCorridor(make_input().task_context.waypoints, 0, 0)
        # Observed public state from the failing TRAIN CrXROSTR/env0 trace,
        # step 712; no simulator ID/navigation/future action is used.
        actor = TrafficActorV1(-1.2804, -3.4081, -4.3257, 0.4439, 0.0525, 5.74, 2.3)
        self.track = ObservedTrack(9, actor, np.array((1.419084, 0.443902)), 0.158007, 0, 220)
        self.speed = 20.681222 / 3.6

    def test_signed_projection_keeps_rear_body_geometry(self):
        arcs, distance, _ = self.corridor.project_with_rear([(-3, -2)])
        self.assertAlmostEqual(arcs[0], -3)
        self.assertAlmostEqual(distance[0], 2)

    def test_bounded_acceleration_can_clear_an_avoidable_rear_cut_in(self):
        result = assess_rear_side_escape(self.corridor, [self.track], self.speed, 29.9, self.agent)
        self.assertIsNotNone(result["rear_side_risk_ttc_s"])
        self.assertTrue(result["rear_escape_safe"])

    def test_no_speed_headroom_never_authorizes_escape(self):
        result = assess_rear_side_escape(self.corridor, [self.track], self.speed, self.speed * 3.6, self.agent)
        self.assertFalse(result["rear_escape_safe"])

    def test_stationary_front_actor_prevents_unsafe_acceleration(self):
        front = ObservedTrack(10, TrafficActorV1(10, 0, -self.speed, 0, 0, 4.5, 1.85),
                              np.zeros(2), 0, 0, 4)
        result = assess_rear_side_escape(self.corridor, [self.track, front], self.speed, 29.9, self.agent)
        self.assertFalse(result["rear_escape_safe"])

    def test_future_cut_in_reserves_time_instead_of_accelerating_towards_it(self):
        # A straight map-matched lane does not suppress the observed turn of a
        # nearby actor: possible lane changes must remain in the forecast.
        actor = TrafficActorV1(12.2, -3.47, -1.6333, -0.0918, -0.0313, 5.74, 2.3)
        path = ActorPathV1(0, "adjacent", ((0, 12.2, -3.47, -0.0313), (35, 47.2, -3.47, 0)))
        track = ObservedTrack(9, actor, np.array((4.2, -0.0918)), 0.06, 0, 200, (path,))
        self.agent._tracker = SimpleNamespace(update=lambda *args: [track], ambiguous_matches=0)
        decision = self.agent.act(make_input(speed=21, actors=((12.2, -3.47, 4.2, -0.0918, -0.0313),)))
        self.assertGreater(decision.diagnostics["conflict_time_s"], 3)
        self.assertLess(decision.diagnostics["target_speed_kmh"], 21)
        self.assertLess(decision.action_payload[1], 0)

    def test_junction_keeps_road_model_scope_without_far_cut_in_freeze(self):
        actor = TrafficActorV1(12.2, -3.47, -1.6333, -0.0918, -0.0313, 5.74, 2.3)
        path = ActorPathV1(0, "adjacent", ((0, 12.2, -3.47, -0.0313), (35, 47.2, -3.47, 0)))
        track = ObservedTrack(9, actor, np.array((4.2, -0.0918)), 0.06, 0, 200, (path,))
        self.agent._tracker = SimpleNamespace(update=lambda *args: [track], ambiguous_matches=0)
        decision = self.agent.act(passage_input(entry=-2, exit=20, speed=21))
        self.assertEqual(decision.diagnostics["junction_phase"], "TRAVERSE_JUNCTION")
        self.assertEqual(decision.diagnostics["rear_side"]["prediction_scope"], "JUNCTION_ROAD_PATHS")
        self.assertIsNone(decision.diagnostics["conflict_distance_m"])
        self.assertGreater(decision.action_payload[1], 0)

    def test_immediate_body_conflict_still_brakes_inside_committed_junction(self):
        decision = self.agent.act(passage_input(entry=-2, exit=20, speed=15,
                                               actors=((4, 3, 0, -5, -math.pi / 2),)))
        self.assertEqual(decision.diagnostics["behavior_state"], "EMERGENCY_BRAKE")
        self.assertEqual(decision.action_payload[1], -1)


class TestJunctionLifecycle(unittest.TestCase):
    def setUp(self):
        self.agent = Stage1RuleBasedAgent()
        self.agent.reset(AgentPublicEpisodeContext())

    def test_clear_gap_is_confirmed_before_commit(self):
        first = self.agent.act(passage_input())
        self.assertEqual(first.diagnostics["junction_phase"], "YIELD")
        self.assertEqual(first.diagnostics["target_speed_kmh"], 0)
        for step in range(1, 6):
            decision = self.agent.act(passage_input(step=step))
        self.assertEqual(decision.diagnostics["junction_phase"], "COMMIT_JUNCTION")
        self.assertTrue(decision.diagnostics["junction_branch_locked"])
        self.assertGreater(decision.action_payload[1], 0)

    def test_crossing_actor_blocks_start_even_when_ego_stopped(self):
        decision = self.agent.act(passage_input(actors=((20, -10, 0, 4, math.pi / 2),)))
        self.assertEqual(decision.diagnostics["junction_reason"], "crossing_occupancy")
        self.assertEqual(decision.diagnostics["target_speed_kmh"], 0)
        self.assertTrue(decision.diagnostics["junction_conflict_track_ids"])

    def test_stationary_cross_actor_is_not_assumed_harmless(self):
        decision = self.agent.act(passage_input(actors=((15, 5, 0, 0, -math.pi / 2),)))
        self.assertEqual(decision.diagnostics["junction_phase"], "YIELD")
        self.assertTrue(decision.diagnostics["junction_conflict_track_ids"])

    def test_vehicle_moving_away_from_path_allows_eventual_commit(self):
        for step in range(6):
            decision = self.agent.act(passage_input(actors=((20, 10 + step, 0, 5, math.pi / 2),), step=step))
        self.assertEqual(decision.diagnostics["junction_phase"], "COMMIT_JUNCTION")

    def test_blocked_exit_never_authorizes_forced_launch_after_timeout(self):
        for step in range(120):
            decision = self.agent.act(passage_input(actors=((33, 0, 0),), step=step))
        self.assertEqual(decision.diagnostics["junction_phase"], "BLOCKED")
        self.assertEqual(decision.diagnostics["junction_reason"], "exit_blocked")
        self.assertEqual(decision.diagnostics["target_speed_kmh"], 0)
        self.assertFalse(decision.diagnostics["junction_branch_locked"])

    def test_blocked_exit_waits_farther_upstream_than_crossing_yield_line(self):
        decision = self.agent.act(passage_input(entry=14, exit=42, actors=((40, 0, 0),)))
        self.assertEqual(decision.diagnostics["junction_reason"], "exit_blocked")
        self.assertEqual(decision.diagnostics["target_speed_kmh"], 0)
        self.assertEqual(decision.diagnostics["junction_admission"]["extra_stop_buffer_m"], 8)

    def test_brief_queue_motion_does_not_release_waiting_buffer(self):
        self.agent.act(passage_input(entry=14, exit=42, actors=((40, 0, 0),)))
        for step in range(1, 10):
            decision = self.agent.act(passage_input(entry=14, exit=42, actors=((40 + step * 0.2, 0, 2),), step=step))
        self.assertEqual(decision.diagnostics["junction_reason"], "exit_blocked")
        self.assertFalse(decision.diagnostics["junction_branch_locked"])

    def test_unobserved_exit_does_not_authorize_entry(self):
        for step in range(7):
            decision = self.agent.act(passage_input(exit=None, step=step))
        self.assertEqual(decision.diagnostics["junction_reason"], "exit_not_visible")
        self.assertFalse(decision.diagnostics["junction_branch_locked"])

    def test_visibility_wait_can_advance_to_entry_buffer_without_double_counting(self):
        decision = self.agent.act(passage_input(entry=7.26, exit=44.76))
        self.assertEqual(decision.diagnostics["junction_reason"], "exit_not_visible")
        self.assertFalse(decision.diagnostics["junction_branch_locked"])
        self.assertGreater(decision.diagnostics["target_speed_kmh"], 0)
        for step in range(1, 7):
            decision = self.agent.act(passage_input(entry=4.26, exit=41.76, step=step))
        self.assertTrue(decision.diagnostics["junction_branch_locked"])

    def test_yield_buffer_keeps_ego_clear_of_actor_overhanging_entry(self):
        decision = self.agent.act(passage_input(entry=6, actors=((6, 0, 0, 0, math.pi),)))
        # Entry denotes centerline connector start, not a physical stop line.
        # An opposing actor's body can extend backwards over that boundary.
        self.assertEqual(decision.diagnostics["target_speed_kmh"], 0)
        ego_front = -self.agent.junction_entry_buffer_m
        actor_rear = -4.5 / 2
        self.assertGreater(actor_rear - ego_front, 1.0)

    def test_branch_change_after_commit_preserves_original_branch(self):
        for step in range(6):
            self.agent.act(passage_input(step=step))
        inp = passage_input(entry=-1, exit=22, step=6, branch="different_connector")
        inp = replace(inp, task_context=replace(inp.task_context, decision_point=None))
        decision = self.agent.act(inp)
        self.assertEqual(decision.diagnostics["junction_branch_id"], "left_connector")
        self.assertTrue(decision.diagnostics["junction_branch_locked"])

    def test_gap_permission_is_revoked_before_front_enters_without_losing_branch(self):
        for step in range(6):
            self.agent.act(passage_input(step=step))
        decision = self.agent.act(passage_input(step=6, actors=((20, -10, 0, 4, math.pi / 2),)))
        self.assertEqual(decision.diagnostics["junction_phase"], "YIELD")
        self.assertTrue(decision.diagnostics["junction_branch_locked"])
        self.assertFalse(decision.diagnostics["junction_front_entered"])
        self.assertEqual(decision.diagnostics["target_speed_kmh"], 0)

    def test_roundabout_admission_and_branch_exit_are_distinct(self):
        inp = passage_input(entry=10, exit=None, kind="roundabout_merge")
        passage = replace(inp.task_context.route_passage, yield_distance_m=4, conflict_exit_distance_m=25)
        inp = replace(inp, task_context=replace(inp.task_context, route_passage=passage))
        for step in range(6):
            decision = self.agent.act(replace(inp, step_index=step))
        self.assertTrue(decision.diagnostics["junction_branch_locked"])
        self.assertIsNone(decision.diagnostics["junction_exit_distance_m"])
        visible = replace(passage, entry_distance_m=8, yield_distance_m=2,
                          conflict_exit_distance_m=23, exit_distance_m=48)
        inp = replace(inp, task_context=replace(inp.task_context, route_passage=visible),
                      ego_motion=EgoMotionV1(2, 0, 0, 2, 0, 0))
        decision = self.agent.act(inp)
        self.assertEqual(decision.diagnostics["junction_exit_distance_m"], 48)

    def test_actual_route_divergence_uses_cached_branch_then_degrades(self):
        for step in range(6):
            self.agent.act(passage_input(step=step))
        inp = passage_input(step=6)
        waypoints = tuple(replace(wp, relative_y=10, relative_heading=0.8) for wp in inp.task_context.waypoints)
        inp = replace(inp, task_context=replace(inp.task_context, waypoints=waypoints,
                      waypoints_array=np.array([wp.to_tuple() for wp in waypoints], dtype=np.float32)))
        first = self.agent.act(inp)
        self.assertAlmostEqual(first.action_payload[0], 0, places=5)
        for step in range(12):
            last = self.agent.act(replace(inp, step_index=7 + step))
        self.assertEqual(last.diagnostics["junction_phase"], "DEGRADED_STOP")
        self.assertEqual(last.diagnostics["target_speed_kmh"], 0)

    def test_fresh_route_beyond_finite_cache_is_not_false_branch_divergence(self):
        for step in range(6):
            self.agent.act(passage_input(entry=7, exit=43, step=step))
        inp = passage_input(entry=-35, exit=1, step=6, motion=EgoMotionV1(42, 0, 0, 3, 0, 0))
        for step in range(12):
            decision = self.agent.act(replace(inp, step_index=6 + step,
                  ego_motion=inp.ego_motion if step == 0 else EgoMotionV1(0, 0, 0, 3, 0, 0)))
        self.assertEqual(decision.diagnostics["route_mismatch_s"], 0)
        self.assertEqual(decision.diagnostics["junction_phase"], "TRAVERSE_JUNCTION")

    def test_connector_memory_lasts_until_rear_clears_exit(self):
        self.agent.act(passage_input(entry=-2, exit=5))
        inp = replace(passage_input(step=1, motion=EgoMotionV1(3, 0, 0, 3, 0, 0)),
                      task_context=replace(make_input().task_context, route_passage=None))
        still_inside = self.agent.act(inp)
        self.assertTrue(still_inside.diagnostics["junction_branch_locked"])
        cleared = self.agent.act(replace(inp, step_index=2, ego_motion=EgoMotionV1(8, 0, 0, 3, 0, 0)))
        self.assertFalse(cleared.diagnostics["junction_branch_locked"])
        self.assertEqual(cleared.diagnostics["junction_phase"], "CRUISE")

    def test_roundabout_uses_same_gap_guard_and_curved_route_control(self):
        decision = self.agent.act(passage_input(kind="roundabout_merge", actors=((20, -10, 0, 4, math.pi / 2),)))
        self.assertEqual(decision.diagnostics["junction_kind"], "roundabout_merge")
        self.assertFalse(decision.diagnostics["junction_branch_locked"])

    def test_reset_clears_tracks_route_and_timers(self):
        self.agent.act(passage_input(actors=((33, 0, 0),)))
        self.agent.reset(AgentPublicEpisodeContext())
        decision = self.agent.act(make_input())
        self.assertEqual(decision.diagnostics["tracked_actor_count"], 0)
        self.assertFalse(decision.diagnostics["junction_branch_locked"])
        self.assertEqual(decision.diagnostics["junction_phase"], "CRUISE")


if __name__ == "__main__":
    unittest.main()
