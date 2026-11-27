import math
import unittest
from types import SimpleNamespace as NS

import numpy as np

from src.platform.agent_input import TaskContextV1
from src.training.stage5.reference_station_proposal import corrected_task_context


class Straight:
    def __init__(self, x=0, y=0, length=10):
        self.x, self.y, self.length = x, y, length

    def local_coordinates(self, pos):
        return pos[0] - self.x, pos[1] - self.y

    def position(self, station, lateral):
        return np.array([self.x + station, self.y + lateral])

    def heading_theta_at(self, station):
        return 0.0


class TestReferenceStationProposal(unittest.TestCase):
    def setUp(self):
        self.ref0 = Straight()
        self.alt = Straight(y=3.5)
        self.ref1 = Straight(x=10)
        self.graph = {"a": {"b": [self.ref0, self.alt]}, "b": {"c": [self.ref1]}}
        self.task = TaskContextV1.empty()

    def apply(self, lane, pos, checkpoint=0, heading=0, checkpoints=("a", "b", "c")):
        nav = NS(current_lane=lane, checkpoints=checkpoints, _target_checkpoints_index=[checkpoint])
        return corrected_task_context(nav, NS(position=pos, heading_theta=heading), NS(graph=self.graph), self.task)

    def test_straight_and_adjacent_keep_designated_centerline(self):
        for lane, y in ((self.ref0, 0), (self.alt, 3.5)):
            task = self.apply(lane, [6, y])
            np.testing.assert_allclose(task.waypoints_array[:2], [[2.5, -y, 0], [5, -y, 0]])

    def test_checkpoint_before_and_after_transition_agree(self):
        for lane, x in ((self.ref0, 9.9), (self.ref1, 10.1), (self.ref1, 7)):
            first = self.apply(lane, [x, 0], 0)
            second = self.apply(lane, [x, 0], 1)
            np.testing.assert_array_equal(first.waypoints_array, second.waypoints_array)
            self.assertEqual(first.waypoints[0].relative_x, 2.5)

    def test_reverse_heading_does_not_clip_valid_waypoint(self):
        task = self.apply(self.ref0, [7, 0], heading=math.pi)
        self.assertTrue(task.validity_mask[0])
        self.assertEqual(task.waypoints[0].relative_x, -2.5)

    def test_route_end_padding_and_exact_endpoint(self):
        task = self.apply(self.ref1, [17.5, 0], 1)
        self.assertTrue(task.route_end_within_lookahead)
        self.assertEqual(int(task.validity_mask.sum()), 1)
        self.assertEqual(task.waypoints[0].relative_x, 2.5)
        np.testing.assert_array_equal(task.waypoints_array[1:], 0)

    def test_invalid_route_is_masked_without_inventing_end(self):
        for shape in ("missing", "degenerate", "disconnected", "missing_checkpoints"):
            with self.subTest(shape=shape):
                self.graph["b"] = {"c": [self.ref1]}
                checkpoints = ("a", "b", "c")
                if shape == "missing":
                    self.graph["b"] = {}
                elif shape == "degenerate":
                    self.graph["b"]["c"] = [Straight(x=10, length=0)]
                elif shape == "disconnected":
                    self.graph["b"]["c"] = [Straight(x=12)]
                else:
                    checkpoints = ()
                task = self.apply(self.ref0, [5, 0], checkpoints=checkpoints)
                self.assertFalse(task.validity_mask.any())
                self.assertFalse(task.route_end_within_lookahead)
                np.testing.assert_array_equal(task.waypoints_array, 0)

    def test_nonlocal_or_unrelated_lane_is_not_snapped_to_route(self):
        near_but_unrelated = Straight(y=0.01)
        task = self.apply(near_but_unrelated, [5, 0.01])
        self.assertFalse(task.validity_mask.any())
        tail = Straight(x=20)
        last = Straight(x=30)
        self.graph["c"] = {"d": [tail]}
        self.graph["d"] = {"e": [last]}
        task = self.apply(last, [31, 0], checkpoints=("a", "b", "c", "d", "e"))
        self.assertFalse(task.validity_mask.any())
