"""P4 safety decisions, shared geometry, lifecycle and exact serialization."""
from dataclasses import FrozenInstanceError, replace
import math
from types import SimpleNamespace
import unittest

import numpy as np

from src.agents.rule_based_agent import Stage1RuleBasedAgent
from src.platform.agent import AgentPublicEpisodeContext, build_agent_contract_core
from src.platform.agent_input import (LaneGeometryV1, LaneContextV1, EgoMotionV1,
                                     extract_lane_context, build_agent_input, InputProfileId,
                                     RoadDecisionPointV1)
from scripts.replay_rule_based_trace import restore_input
import src.platform.agent_input as input_module
from tests.test_rule_based_agent import make_input
from tests.test_rule_based_corridor import Lane


def lane(side="CURRENT", y=0, style="BROKEN_WHITE", compatible=True, remaining=50, lane_id=None):
    return LaneGeometryV1(lane_id or side, side, 3.5, remaining, style, style, compatible,
                          tuple((float(x), float(x), float(y), 0.) for x in np.arange(-15, remaining + .01, 2.5)))


def p4_input(speed=0, actors=(), mandatory=True, style="BROKEN_WHITE", step=0, offset=0):
    inp = make_input(speed=speed, actors=actors)
    ctx = LaneContextV1((lane(y=-offset, style=style, compatible=not mandatory),
                         lane("LEFT", 3.5-offset, style=style)), "LEFT" if mandatory else None)
    return replace(inp, step_index=step, ego_motion=EgoMotionV1(0, 0, 0, speed/3.6, 0, 0),
                   task_context=replace(inp.task_context, lane_context=ctx))


class TestP4PublicGeometry(unittest.TestCase):
    def test_rendered_right_boundary_is_owned_by_right_neighbor(self):
        first = Lane((0, 3.5), (100, 3.5), ("a", "b", 0))
        second = Lane((0, 0), (100, 0), ("a", "b", 1))
        for item in (first, second):
            item.line_types = ["ROAD_LINE_BROKEN_SINGLE_WHITE"]*2
            item.line_colors = [(1, 1, 1, 1)]*2
        second.line_types[0] = "ROAD_LINE_SOLID_SINGLE_WHITE"
        nav = SimpleNamespace(current_lane=first, checkpoints=["a", "b"], _target_checkpoints_index=[0, 1])
        network = SimpleNamespace(graph={"a": {"b": [first, second]}})
        ctx = extract_lane_context(nav, SimpleNamespace(position=(30, 3.5), heading_theta=0), network)
        self.assertEqual(ctx.lanes[0].right_boundary, "SOLID_WHITE")
        self.assertEqual(ctx.lanes[1].left_boundary, "SOLID_WHITE")
        self.assertEqual(ctx.lanes[1].side, "RIGHT")
        self.assertAlmostEqual(ctx.lanes[1].points[6][2], -3.5)

    def test_mandatory_route_detects_disconnected_current_lane(self):
        first = Lane((0, 3.5), (40, 3.5), ("a", "b", 0))
        second = Lane((0, 0), (40, 0), ("a", "b", 1))
        successor = Lane((40, 3.5), (80, 3.5), ("b", "c", 0))
        nav = SimpleNamespace(current_lane=second, checkpoints=["a", "b", "c"], _target_checkpoints_index=[0, 1])
        network = SimpleNamespace(graph={"a": {"b": [first, second]}, "b": {"c": [successor]}})
        ctx = extract_lane_context(nav, SimpleNamespace(position=(2, 0), heading_theta=0), network)
        self.assertEqual(ctx.required_side, "LEFT")
        self.assertIs(ctx.lanes[-1].route_compatible, False)
        self.assertIs(ctx.lanes[0].route_compatible, True)

    def test_unknown_line_and_core_only_do_not_authorize_change(self):
        masked = build_agent_input(np.ones(259), None, [], None, None, 0, InputProfileId.CORE_ONLY_V1)
        self.assertIsNone(masked.task_context.lane_context)
        decision = Stage1RuleBasedAgent().act(p4_input(style="UNKNOWN"))
        self.assertIn("boundary_prohibited_or_unknown", decision.diagnostics["lane_change"]["veto"])

    def test_schema_immutable_and_roundtrip(self):
        inp = p4_input()
        restored = restore_input(inp.to_dict(), input_module)
        self.assertEqual(restored.to_dict(), inp.to_dict())
        with self.assertRaises(FrozenInstanceError):
            restored.task_context.lane_context.lanes[0].width_m = 5
        self.assertEqual(build_agent_contract_core()["metadata"]["spec_version"], "1.3.0")
        with self.assertRaises(ValueError):
            replace(lane(), points=((0, 0, 0, float("nan")), (1, 1, 0, 0)))
        with self.assertRaises(ValueError):
            LaneContextV1((lane(), lane()))

    def test_endpoint_near_sample_grid_does_not_duplicate_signed_distance(self):
        for delta in (-.00001, .00001):
            road = Lane((0, 0), (40, 0), ("a", "b", 0))
            nav = SimpleNamespace(current_lane=road, checkpoints=["a", "b"], _target_checkpoints_index=[0, 1])
            ctx = extract_lane_context(nav, SimpleNamespace(position=(17.5+delta, 0), heading_theta=0),
                                       SimpleNamespace(graph={"a": {"b": [road]}}))
            distances = [p[0] for p in ctx.lanes[0].points]
            self.assertEqual(distances, sorted(set(distances)))


class TestP4Policy(unittest.TestCase):
    def test_safe_mandatory_left_change_starts_and_steers_left(self):
        decision = Stage1RuleBasedAgent().act(p4_input())
        self.assertEqual(decision.diagnostics["lane_change"]["phase"], "PREPARE_CHANGE")
        self.assertEqual(decision.diagnostics["lane_change"]["reason"], "MANDATORY_ROUTE")
        self.assertGreater(decision.action_payload[0], 0)
        self.assertLessEqual(len(decision.diagnostics), 50)

    def test_right_change_steers_right(self):
        inp = p4_input()
        ctx = LaneContextV1((lane(compatible=False), lane("RIGHT", -3.5)), "RIGHT")
        decision = Stage1RuleBasedAgent().act(replace(inp, task_context=replace(inp.task_context, lane_context=ctx)))
        self.assertLess(decision.action_payload[0], 0)

    def test_fast_rear_and_front_target_gap_veto(self):
        for actors, reason in (((( -20, 3.5, 20),), "rear_gap"), (((8, 3.5, 0),), "front_gap")):
            decision = Stage1RuleBasedAgent().act(p4_input(speed=18, actors=actors))
            self.assertIn(reason, decision.diagnostics["lane_change"]["veto"])
            self.assertEqual(decision.diagnostics["lane_change"]["phase"], "KEEP_LANE")

    def test_parallel_actor_beside_ego_is_rejected(self):
        decision = Stage1RuleBasedAgent().act(p4_input(actors=((0, 3.5, 0),)))
        self.assertIn("front_gap", decision.diagnostics["lane_change"]["veto"])

    def test_source_leader_in_swept_path_vetoes_launch(self):
        decision = Stage1RuleBasedAgent().act(p4_input(speed=18, actors=((9, 0, 0),)))
        self.assertIn("swept_actor", decision.diagnostics["lane_change"]["veto"])

    def test_solid_white_yellow_edge_and_unknown_veto(self):
        for style in ("SOLID_WHITE", "SOLID_YELLOW", "EDGE", "UNKNOWN"):
            decision = Stage1RuleBasedAgent().act(p4_input(style=style))
            self.assertIn("boundary_prohibited_or_unknown", decision.diagnostics["lane_change"]["veto"])

    def test_sensor_overflow_junction_and_short_lane_veto(self):
        inp = p4_input()
        overflow = replace(inp, traffic_context=replace(inp.traffic_context, overflow_count=1))
        self.assertIn("sensor_overflow", Stage1RuleBasedAgent().act(overflow).diagnostics["lane_change"]["veto"])
        junction = replace(inp, task_context=replace(inp.task_context, decision_point=RoadDecisionPointV1("x", 20, ("left", "straight"))))
        self.assertIn("junction_or_cooldown", Stage1RuleBasedAgent().act(junction).diagnostics["lane_change"]["veto"])
        ctx = LaneContextV1((lane(remaining=20, compatible=False), lane("LEFT", 3.5, remaining=20)), "LEFT")
        short = replace(inp, task_context=replace(inp.task_context, lane_context=ctx))
        self.assertIn("insufficient_lane_length", Stage1RuleBasedAgent().act(short).diagnostics["lane_change"]["veto"])

    def test_overtake_requires_persistent_slow_leader(self):
        agent = Stage1RuleBasedAgent()
        inp = p4_input(speed=0, mandatory=False, actors=((20, 0, 0),))
        for step in range(12):
            decision = agent.act(replace(inp, step_index=step))
            if step == 0:
                self.assertEqual(decision.diagnostics["lane_change"]["phase"], "KEEP_LANE")
        self.assertEqual(decision.diagnostics["lane_change"]["reason"], "OVERTAKE_SLOW_LEADER")

    def test_reset_clears_maneuver_and_cooldown(self):
        agent = Stage1RuleBasedAgent()
        agent.act(p4_input())
        agent.reset(AgentPublicEpisodeContext())
        self.assertIsNone(agent._lane_change.world_path)
        self.assertEqual(agent._lane_change.cooldown, 0)
        self.assertEqual(agent.act(make_input()).diagnostics["lane_change"]["phase"], "KEEP_LANE")

    def test_precommit_rear_intrusion_aborts_if_source_safe(self):
        agent = Stage1RuleBasedAgent()
        agent.act(p4_input())
        decision = agent.act(p4_input(step=1, actors=((-15, 3.5, 15),)))
        self.assertEqual(decision.diagnostics["lane_change"]["phase"], "ABORT_CHANGE")
        self.assertEqual(decision.diagnostics["lane_change"]["target_id"], "CURRENT")

    def test_postcommit_threat_stops_without_target_flip(self):
        agent = Stage1RuleBasedAgent()
        agent.act(p4_input())
        agent._frame.position[1] = 1.5
        decision = agent.act(p4_input(step=1, offset=1.5, actors=((-15, 2, 15),)))
        self.assertTrue(decision.diagnostics["lane_change"]["committed"])
        self.assertEqual(decision.diagnostics["lane_change"]["target_id"], "LEFT")
        self.assertEqual(decision.diagnostics["speed_limits_kmh"]["lane_change"], 0)

    def test_lost_geometry_mid_change_stops(self):
        agent = Stage1RuleBasedAgent()
        agent.act(p4_input())
        decision = agent.act(make_input(speed=10))
        self.assertEqual(decision.diagnostics["speed_limits_kmh"]["lane_change"], 0)

    def test_target_persists_after_nearest_lane_switch_and_completes(self):
        agent = Stage1RuleBasedAgent()
        agent.act(p4_input())
        agent._frame.position[:] = (25, 3.5)
        inp = p4_input(mandatory=False)
        ctx = LaneContextV1((lane("RIGHT", -3.5, lane_id="CURRENT"),
                             lane("CURRENT", 0, lane_id="LEFT")))
        inp = replace(inp, task_context=replace(inp.task_context, lane_context=ctx))
        for step in range(1, 7):
            decision = agent.act(replace(inp, step_index=step))
            self.assertEqual(decision.diagnostics["lane_change"]["target_id"], "LEFT")
        self.assertIsNone(agent._lane_change.world_path)
        self.assertGreater(agent._lane_change.cooldown, 0)

    def test_unknown_target_route_and_high_speed_prepare(self):
        inp = p4_input()
        ctx = LaneContextV1((lane(compatible=False), lane("LEFT", 3.5, compatible=None)), "LEFT")
        decision = Stage1RuleBasedAgent().act(replace(inp, task_context=replace(inp.task_context, lane_context=ctx)))
        self.assertIn("target_route_unavailable", decision.diagnostics["lane_change"]["veto"])
        decision = Stage1RuleBasedAgent().act(p4_input(speed=40))
        self.assertIn("prepare_speed_or_alignment", decision.diagnostics["lane_change"]["veto"])
        self.assertLessEqual(decision.diagnostics["speed_limits_kmh"]["lane_change"], 24)

    def test_no_viable_route_lane_stops_before_lane_end(self):
        inp = p4_input(mandatory=False)
        ctx = LaneContextV1((lane(remaining=10, compatible=False),))
        decision = Stage1RuleBasedAgent().act(replace(inp, task_context=replace(inp.task_context, lane_context=ctx)))
        self.assertEqual(decision.diagnostics["speed_limits_kmh"]["lane_change"], 0)

    def test_unknown_lidar_surface_in_maneuver_vetoes(self):
        inp = p4_input()
        obs = inp.core_observation.features.copy()
        # A point in the target corridor, beyond the lateral transition.
        angle = math.atan2(3.5, 20)
        ray = round(angle / (2 * math.pi / 240))
        obs[19 + ray] = math.hypot(20, 3.5) / 50
        inp = replace(inp, core_observation=input_module.CoreObservationV1(obs))
        decision = Stage1RuleBasedAgent().act(inp)
        self.assertIn("swept_lidar", decision.diagnostics["lane_change"]["veto"])

    def test_return_after_pass_requires_observed_clearance_and_cooldown(self):
        agent = Stage1RuleBasedAgent()
        for step in range(12):
            agent.act(p4_input(step=step, mandatory=False, actors=((20, 0, 0),)))
        agent._frame.position[:] = (30, 3.5)
        inp = p4_input(mandatory=False, actors=((-10, -3.5, 0),))
        ctx = LaneContextV1((lane("RIGHT", -3.5, lane_id="CURRENT"), lane("CURRENT", 0, lane_id="LEFT")))
        inp = replace(inp, task_context=replace(inp.task_context, lane_context=ctx))
        for step in range(12, 22):
            agent.act(replace(inp, step_index=step))
        self.assertGreater(agent._lane_change.cooldown, 0)
        self.assertEqual(agent._lane_change.phase, "KEEP_LANE")
        for step in range(22, 82):
            decision = agent.act(replace(inp, step_index=step))
        self.assertEqual(decision.diagnostics["lane_change"]["reason"], "RETURN_AFTER_PASS")
        self.assertEqual(decision.diagnostics["lane_change"]["target_id"], "CURRENT")

    def test_mandatory_takes_priority_over_optional_return(self):
        agent = Stage1RuleBasedAgent()
        agent._lane_change.return_lane_id = "RIGHT"
        agent._lane_change.overtaken_track_id = 1
        inp = p4_input()
        ctx = LaneContextV1(inp.task_context.lane_context.lanes + (lane("RIGHT", -3.5),), "LEFT")
        decision = agent.act(replace(inp, task_context=replace(inp.task_context, lane_context=ctx)))
        self.assertEqual(decision.diagnostics["lane_change"]["reason"], "MANDATORY_ROUTE")
        self.assertEqual(decision.diagnostics["lane_change"]["target_id"], "LEFT")

    def test_junction_approach_veto_does_not_strand_existing_change(self):
        agent = Stage1RuleBasedAgent()
        agent.act(p4_input())
        agent._frame.position[:] = (10, 1.5)
        inp = p4_input(step=1, offset=1.5)
        inp = replace(inp, task_context=replace(inp.task_context, decision_point=RoadDecisionPointV1("junction", 33, ("left", "straight"))))
        decision = agent.act(inp)
        self.assertEqual(decision.diagnostics["lane_change"]["phase"], "CHANGE_LANE")
        self.assertGreater(decision.diagnostics["speed_limits_kmh"]["lane_change"], 0)

    def test_discretionary_pass_waits_outside_entire_observed_junction_approach(self):
        agent = Stage1RuleBasedAgent()
        inp = p4_input(mandatory=False, actors=((20, 0, 0),))
        inp = replace(inp, task_context=replace(inp.task_context, decision_point=RoadDecisionPointV1("junction", 49, ("left", "straight"))))
        for step in range(12):
            decision = agent.act(replace(inp, step_index=step))
        self.assertEqual(decision.diagnostics["lane_change"]["phase"], "KEEP_LANE")
        self.assertIn("junction_or_cooldown", decision.diagnostics["lane_change"]["veto"])

    def test_rotated_body_can_commit_before_center_reaches_fraction(self):
        agent = Stage1RuleBasedAgent()
        agent.act(p4_input())
        agent._frame.position[:] = (8, .4)
        agent._frame.heading = .25
        inp = p4_input(step=1, offset=.4)
        c, s = math.cos(-.25), math.sin(-.25)
        geometries = tuple(replace(l, points=tuple((d, c*x-s*y, s*x+c*y, -.25) for d, x, y, h in l.points))
                           for l in inp.task_context.lane_context.lanes)
        inp = replace(inp, task_context=replace(inp.task_context, lane_context=LaneContextV1(geometries, "LEFT")),
                      ego_motion=EgoMotionV1(0, 0, 0, 0, 0, -.25))
        decision = agent.act(inp)
        self.assertLess(decision.diagnostics["lane_change"]["progress"], .22)
        self.assertTrue(decision.diagnostics["lane_change"]["committed"])


if __name__ == "__main__":
    unittest.main()
