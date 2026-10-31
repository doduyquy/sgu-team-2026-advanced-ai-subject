"""Behavior checks for lane keeping using the existing public V1 input only."""

from dataclasses import replace
import unittest

import numpy as np

from src.agents.rule_based_agent import Stage1RuleBasedAgent
from src.platform.agent import AgentPublicEpisodeContext
from src.platform.agent_input import (
    AgentInputV1, CoreObservationV1, InputProfileId, RouteWaypointV1,
    TaskContextV1, TrafficContextV1, RoadDecisionPointV1, extract_traffic_context,
)
from types import SimpleNamespace


def make_input(speed=15.0, offset=0.0, heading=0.5, bend=0.0,
               waypoint_shift=0.0, valid=True, count=20, point=None, actors=()):
    obs = np.full(259, 0.5, dtype=np.float32)
    obs[19:] = 1.0  # A clear LiDAR scan; 0.5 would mean surfaces at 25 m.
    obs[0:2] = 0.3
    obs[2] = heading
    obs[3] = (speed + 1.0) / 81.0
    obs[8] = 0.5 + offset / 4.5
    waypoints = tuple(
        RouteWaypointV1(2.5 * (i + 1), 2.5 * (i + 1), waypoint_shift, bend * i)
        if valid else RouteWaypointV1(0.0, 0.0, 0.0, 0.0)
        for i in range(count)
    )
    task = TaskContextV1(
        lookahead_count=count, lookahead_spacing_m=2.5, lookahead_range_m=count * 2.5,
        waypoints=waypoints,
        waypoints_array=np.array([wp.to_tuple() for wp in waypoints], dtype=np.float32),
        validity_mask=np.full(count, valid, dtype=bool),
        decision_point=point,
    )
    ego = SimpleNamespace(position=(0.0, 0.0), heading_theta=0.0,
                          velocity=(speed / 3.6, 0.0))
    vehicles = []
    for actor in actors:
        x, y, velocity = actor[:3]
        vy, actor_heading = actor[3:] if len(actor) == 5 else (0.0, 0.0)
        vehicles.append(SimpleNamespace(position=(x, y), velocity=(velocity, vy),
                                       heading_theta=actor_heading, LENGTH=4.5, WIDTH=1.85))
    traffic = extract_traffic_context(ego, vehicles)
    return AgentInputV1(InputProfileId.STATE_DECISION_V1, CoreObservationV1(obs),
                        traffic, task, 0)


class TestRuleBasedAgent(unittest.TestCase):
    def setUp(self):
        self.agent = Stage1RuleBasedAgent()

    def test_centered_straight_has_neutral_steering(self):
        self.assertAlmostEqual(self.agent.act(make_input()).action_payload[0], 0.0)

    def test_right_offset_corrects_left_and_left_offset_corrects_right(self):
        right = self.agent.act(make_input(offset=0.4)).action_payload[0]
        left = self.agent.act(make_input(offset=-0.4)).action_payload[0]
        self.assertGreater(right, 0.0)  # Runtime probe: positive steering turns left.
        self.assertLess(left, 0.0)
        self.assertAlmostEqual(right, -left, places=5)

    def test_heading_error_corrects_in_runtime_direction(self):
        self.assertGreater(self.agent.act(make_input(heading=0.6)).action_payload[0], 0.0)
        self.assertLess(self.agent.act(make_input(heading=0.4)).action_payload[0], 0.0)

    def test_reference_waypoint_in_other_lane_does_not_force_lane_change(self):
        ordinary = self.agent.act(make_input()).action_payload
        shifted = self.agent.act(make_input(waypoint_shift=7.0)).action_payload
        self.assertEqual(ordinary, shifted)

    def test_overspeed_brakes_and_underspeed_accelerates(self):
        self.assertLess(self.agent.act(make_input(speed=65.0)).action_payload[1], 0.0)
        self.assertGreater(self.agent.act(make_input(speed=5.0)).action_payload[1], 0.0)

    def test_upcoming_curve_reduces_speed_before_lateral_error(self):
        straight = self.agent.act(make_input(speed=20.0))
        bend = self.agent.act(make_input(speed=20.0, bend=0.2))
        self.assertLess(bend.diagnostics["target_speed_kmh"], straight.diagnostics["target_speed_kmh"])
        self.assertLess(bend.action_payload[1], 0.0)

    def test_near_boundary_slows_without_abandoning_steering(self):
        decision = self.agent.act(make_input(speed=20.0, offset=0.9))
        self.assertEqual(decision.diagnostics["rule"], "recover_lane")
        self.assertGreater(decision.action_payload[0], 0.0)
        self.assertLess(decision.action_payload[1], 0.0)

    def test_missing_waypoints_brakes(self):
        decision = self.agent.act(make_input(valid=False))
        self.assertLess(decision.action_payload[1], 0.0)

    def test_actions_are_finite_bounded_and_input_unchanged(self):
        for speed in (0.0, 80.0):
            for offset in (-2.0, 0.0, 2.0):
                inp = make_input(speed=speed, offset=offset, heading=0.9, bend=-0.4)
                original = inp.core_observation.features.copy()
                action = self.agent.act(inp).action_payload
                self.assertTrue(np.isfinite(action).all())
                self.assertTrue(all(-1.0 <= value <= 1.0 for value in action))
                np.testing.assert_array_equal(inp.core_observation.features, original)

    def test_repeatability_across_resets(self):
        inp = make_input(offset=0.3, bend=0.1)
        before = self.agent.act(inp)
        self.agent.reset(AgentPublicEpisodeContext(), agent_seed=999)
        self.assertEqual(before, self.agent.act(inp))

    def test_clear_straight_can_accelerate_above_old_twenty_two_cap(self):
        decision = self.agent.act(make_input(speed=30))
        self.assertEqual(decision.diagnostics["target_speed_kmh"], 45)
        self.assertGreater(decision.action_payload[1], 0)

    def test_far_curve_allows_more_speed_than_near_curve(self):
        straight = make_input(speed=30)
        def with_curve(start_m):
            waypoints = tuple(replace(wp, relative_heading=max(
                0, wp.lookahead_distance_m - start_m) * 0.08)
                for wp in straight.task_context.waypoints)
            task = replace(straight.task_context, waypoints=waypoints,
                           waypoints_array=np.array([wp.to_tuple() for wp in waypoints], dtype=np.float32))
            return replace(straight, task_context=task)
        near = self.agent.act(with_curve(5)).diagnostics["target_speed_kmh"]
        far = self.agent.act(with_curve(25)).diagnostics["target_speed_kmh"]
        self.assertLess(near, far)
        self.assertLess(far, 45)

    def test_short_visibility_reduces_speed_even_on_straight(self):
        decision = self.agent.act(make_input(speed=25, count=3))
        self.assertEqual(decision.diagnostics["speed_limit_reason"], "visibility")
        self.assertLess(decision.diagnostics["target_speed_kmh"], 25)
        self.assertLess(decision.action_payload[1], 0)

    def test_junction_has_braking_approach_then_lower_entry_speed(self):
        def decision(distance):
            point = RoadDecisionPointV1("fork", distance, ("left", "straight"))
            return self.agent.act(make_input(speed=30, point=point))
        far, near = decision(40), decision(5)
        self.assertGreater(far.diagnostics["target_speed_kmh"], near.diagnostics["target_speed_kmh"])
        self.assertLessEqual(near.diagnostics["target_speed_kmh"], 24)
        self.assertEqual(near.diagnostics["speed_limit_reason"], "junction")

    def test_slow_front_vehicle_limits_speed_and_brakes(self):
        decision = self.agent.act(make_input(speed=35, actors=((18, 0, 3),)))
        self.assertEqual(decision.diagnostics["speed_limit_reason"], "front_vehicle")
        self.assertLess(decision.diagnostics["target_speed_kmh"], 35)
        self.assertLess(decision.action_payload[1], 0)

    def test_close_stopped_vehicle_triggers_full_emergency_braking(self):
        decision = self.agent.act(make_input(speed=35, actors=((8, 0, 0),)))
        self.assertEqual(decision.diagnostics["rule"], "emergency_brake")
        self.assertEqual(decision.action_payload[1], -1)

    def test_adjacent_or_rear_vehicle_does_not_block_clear_lane(self):
        for actors in (((8, 3.5, 0),), ((-5, 0, 0),)):
            decision = self.agent.act(make_input(speed=30, actors=actors))
            self.assertEqual(decision.diagnostics["target_speed_kmh"], 45)
            self.assertIsNone(decision.diagnostics["front_gap_m"])

    def test_front_vehicle_with_sufficient_gap_does_not_trigger_emergency(self):
        decision = self.agent.act(make_input(speed=30, actors=((40, 0, 30 / 3.6),)))
        self.assertFalse(decision.diagnostics["emergency_brake"])
        self.assertGreater(decision.diagnostics["target_speed_kmh"], 30)

    def test_nearer_leader_and_emergency_override_other_constraints(self):
        decision = self.agent.act(make_input(speed=30, offset=0.9,
                                             actors=((40, 0, 10), (7, 0, 0))))
        self.assertEqual(decision.diagnostics["rule"], "emergency_brake")
        self.assertEqual(decision.action_payload[1], -1)

    def test_speed_configuration_is_validated(self):
        self.assertEqual(Stage1RuleBasedAgent(35).cruise_speed_kmh, 35)
        for speed in (0, -1, 81, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                Stage1RuleBasedAgent(speed)


if __name__ == "__main__":
    unittest.main()
