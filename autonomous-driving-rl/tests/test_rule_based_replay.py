"""Recorded public inputs retain all immutable P2/P3 data through replay."""

from dataclasses import replace
import unittest

import src.platform.agent_input as inputs
from scripts.replay_rule_based_trace import restore_input
from tests.test_rule_based_agent import make_input


class TestRuleBasedReplay(unittest.TestCase):
    def test_full_public_input_round_trip_includes_motion_paths_and_passage(self):
        inp = make_input(actors=((10, 3, 4),))
        path = inputs.ActorPathV1(0, "local_branch", ((0, 10, 3, 0), (10, 20, 3, 0)))
        passage = inputs.RoutePassageV1("entry", "branch", "roundabout_merge", 20, 45,
                                       "right", yield_distance_m=15, conflict_exit_distance_m=30)
        inp = replace(inp, traffic_context=replace(inp.traffic_context, motion_paths=(path,)),
                      task_context=replace(inp.task_context, route_passage=passage),
                      ego_motion=inputs.EgoMotionV1(0.1, 0, 0.01, 4, 0, 0.02))
        restored = restore_input(inp.to_dict(), inputs)
        self.assertEqual(restored.to_dict(), inp.to_dict())
        self.assertFalse(restored.core_observation.features.flags.writeable)
        self.assertFalse(restored.traffic_context.actors_array.flags.writeable)
        self.assertFalse(restored.task_context.waypoints_array.flags.writeable)

    def test_replay_rejects_nonpublic_top_level_fields(self):
        data = make_input().to_dict()
        data["reward"] = 999
        with self.assertRaises(ValueError):
            restore_input(data, inputs)


if __name__ == "__main__":
    unittest.main()
