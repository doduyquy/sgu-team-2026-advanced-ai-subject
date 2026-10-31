"""Connected-lane extraction, curved traffic and stateful safety regressions."""

from dataclasses import replace
import math
from types import SimpleNamespace
import unittest

import numpy as np

from src.agents.rule_based_agent import Stage1RuleBasedAgent
from src.platform.agent import AgentPublicEpisodeContext
from src.platform.agent_input import RoadDecisionPointV1, RouteWaypointV1, extract_task_context
from tests.test_rule_based_agent import make_input


class Lane:
    def __init__(self, start, end, index, width=3.5):
        self.start, self.end = np.array(start, dtype=float), np.array(end, dtype=float)
        self.index, self.width = index, width
        delta = self.end - self.start
        self.length = float(np.linalg.norm(delta))
        self.forward = delta / self.length
        self.right = np.array([self.forward[1], -self.forward[0]])

    def position(self, longitudinal, lateral):
        return self.start + longitudinal * self.forward + lateral * self.right

    def heading_theta_at(self, longitudinal):
        return math.atan2(self.forward[1], self.forward[0])

    def local_coordinates(self, position):
        delta = np.asarray(position) - self.start
        return float(delta @ self.forward), float(delta @ self.right)


class TestConnectedRoute(unittest.TestCase):
    def context(self, gap=False, checkpoint=0):
        first = [Lane((0, y), (20, y), ("a", "b", i)) for i, y in enumerate((0, 3.5))]
        # Same lane index would pick the wrong continuation; geometry wins.
        second = [Lane((20 + (10 if gap else 0), y), (60, y), ("b", "c", i))
                  for i, y in enumerate((3.5, 0))]
        vehicle = SimpleNamespace(position=(17, 3.5), heading_theta=0)
        nav = SimpleNamespace(current_lane=first[1], get_current_lane_width=lambda: 3.5,
                              get_navi_info=lambda: [1.0, 0.5],
                              _target_checkpoints_index=[checkpoint, 1], checkpoints=["a", "b", "c"])
        network = SimpleNamespace(graph={"a": {"b": first}, "b": {"c": second}})
        return nav, vehicle, network

    def test_current_lane_and_changed_lane_indices_remain_continuous(self):
        task = extract_task_context(*self.context())
        self.assertTrue(all(abs(wp.relative_y) < 1e-6 for wp, valid in zip(task.waypoints, task.validity_mask) if valid))
        self.assertAlmostEqual(task.waypoints[0].relative_x, 2.5)
        self.assertAlmostEqual(task.waypoints[1].relative_x, 5)

    def test_checkpoint_drift_does_not_start_from_a_different_lane(self):
        nav, vehicle, network = self.context()
        vehicle.position = (25, 3.5)
        nav.current_lane = network.graph["b"]["c"][0]
        task = extract_task_context(nav, vehicle, network)
        self.assertAlmostEqual(task.waypoints[0].relative_x, 2.5)
        self.assertAlmostEqual(task.waypoints[0].relative_y, 0)

    def test_disconnected_connector_masks_future_without_claiming_route_end(self):
        task = extract_task_context(*self.context(gap=True))
        self.assertTrue(task.validity_mask[0])
        self.assertFalse(task.validity_mask[1])
        self.assertFalse(task.route_end_within_lookahead)

    def test_actual_route_tail_is_distinct_from_missing_connector(self):
        nav, vehicle, network = self.context()
        nav.current_lane = network.graph["b"]["c"][0]
        vehicle.position = (57, 3.5)
        nav._target_checkpoints_index = [1, 1]
        task = extract_task_context(nav, vehicle, network)
        self.assertTrue(task.route_end_within_lookahead)
        self.assertEqual(int(task.validity_mask.sum()), 1)


class TestCorridorSafety(unittest.TestCase):
    def setUp(self):
        self.agent = Stage1RuleBasedAgent()
        self.agent.reset(AgentPublicEpisodeContext())

    def curved_input(self, actor_offset=0):
        angle = 15 / 20
        x, y = 20 * math.sin(angle), 20 * (1 - math.cos(angle))
        inp = make_input(speed=30, actors=((x - actor_offset * math.sin(angle),
                                          y + actor_offset * math.cos(angle),
                                          math.cos(angle), math.sin(angle), angle),))
        waypoints = tuple(RouteWaypointV1(s, 20 * math.sin(s / 20),
                                         20 * (1 - math.cos(s / 20)), s / 20)
                          for s in np.arange(2.5, 50.1, 2.5))
        task = replace(inp.task_context, waypoints=waypoints,
                       waypoints_array=np.array([wp.to_tuple() for wp in waypoints], dtype=np.float32))
        return replace(inp, task_context=task)

    def test_front_vehicle_on_curve_is_detected_outside_straight_strip(self):
        decision = self.agent.act(self.curved_input())
        self.assertIsNotNone(decision.diagnostics["front_gap_m"])
        self.assertLess(decision.diagnostics["target_speed_kmh"], 25)

    def test_adjacent_vehicle_on_curve_is_not_treated_as_leader(self):
        decision = self.agent.act(self.curved_input(actor_offset=3.5))
        self.assertIsNone(decision.diagnostics["front_gap_m"])

    def test_crossing_vehicle_is_predicted_before_it_enters_ego_lane(self):
        decision = self.agent.act(make_input(speed=36, actors=((20, 10, 0, -5, -math.pi / 2),)))
        self.assertIsNotNone(decision.diagnostics["conflict_distance_m"])
        self.assertEqual(decision.diagnostics["behavior_state"], "YIELD")
        self.assertLess(decision.action_payload[1], 0)

    def test_vehicle_moving_away_does_not_create_crossing_conflict(self):
        decision = self.agent.act(make_input(speed=36, actors=((20, 10, 0, 5, math.pi / 2),)))
        self.assertIsNone(decision.diagnostics["conflict_distance_m"])

    def test_cross_traffic_reserves_junction_entry_before_turning_vehicle_reaches_path(self):
        point = RoadDecisionPointV1("junction", 18, ("left", "straight"))
        decision = self.agent.act(make_input(speed=30, point=point,
                                             actors=((22, -12, 0, 6, math.pi / 2),)))
        self.assertTrue(decision.diagnostics["junction_entry_conflict"])
        self.assertLessEqual(decision.diagnostics["conflict_distance_m"], 18 - self.agent.vehicle_length_m / 2)
        self.assertLess(decision.diagnostics["target_speed_kmh"], 30)

    def test_unavailable_lidar_scan_does_not_authorize_acceleration(self):
        inp = make_input(speed=30)
        obs = inp.core_observation.features.copy()
        obs[19:] = 0
        decision = self.agent.act(replace(inp, core_observation=type(inp.core_observation)(obs)))
        self.assertEqual(decision.diagnostics["behavior_state"], "DEGRADED_STOP")
        self.assertEqual(decision.diagnostics["target_speed_kmh"], 0)

    def test_yield_release_has_hysteresis_and_reset_clears_it(self):
        self.agent.act(make_input(speed=36, actors=((20, 10, 0, -5, -math.pi / 2),)))
        decision = self.agent.act(make_input(speed=25))
        self.assertEqual(decision.diagnostics["behavior_state"], "YIELD")
        self.assertEqual(decision.diagnostics["target_speed_kmh"], 0)
        self.agent.reset(AgentPublicEpisodeContext())
        self.assertEqual(self.agent.act(make_input(speed=25)).diagnostics["target_speed_kmh"], 45)

    def test_lidar_obstacle_in_corridor_triggers_braking(self):
        inp = make_input(speed=30)
        obs = inp.core_observation.features.copy()
        obs[19] = 6 / 50
        decision = self.agent.act(replace(inp, core_observation=type(inp.core_observation)(obs)))
        self.assertIsNotNone(decision.diagnostics["lidar_obstacle_gap_m"])
        self.assertEqual(decision.action_payload[1], -1)

    def test_side_lidar_hit_does_not_block_route(self):
        inp = make_input(speed=30)
        obs = inp.core_observation.features.copy()
        obs[19 + 60] = 6 / 50
        decision = self.agent.act(replace(inp, core_observation=type(inp.core_observation)(obs)))
        self.assertIsNone(decision.diagnostics["lidar_obstacle_gap_m"])

    def test_known_moving_vehicle_is_not_duplicated_as_stationary_lidar_obstacle(self):
        inp = make_input(speed=30, actors=((20, 0, 30 / 3.6),))
        obs = inp.core_observation.features.copy()
        obs[19] = 17.75 / 50
        decision = self.agent.act(replace(inp, core_observation=type(inp.core_observation)(obs)))
        self.assertIsNone(decision.diagnostics["lidar_obstacle_gap_m"])

    def test_actor_overflow_reduces_cruise_speed(self):
        inp = make_input(speed=30, actors=tuple((i + 5, 20, 5) for i in range(12)))
        decision = self.agent.act(inp)
        self.assertEqual(decision.diagnostics["speed_limit_reason"], "traffic_overflow")
        self.assertEqual(decision.diagnostics["target_speed_kmh"], 27)


if __name__ == "__main__":
    unittest.main()
