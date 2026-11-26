"""Actual BFS decisions, input/action pairing, inspection and overlay lifecycle."""

from types import SimpleNamespace
import unittest

import numpy as np
from panda3d.core import NodePath

from src.agents.graph_search_agents import BFSAgent
from src.platform.agent import AgentDecision, AgentPublicEpisodeContext
from src.platform.action_adapter import get_action_adapter
from src.platform.agent_input import (
    AgentInputV1, CoreObservationV1, InputProfileId,
    RouteWaypointV1, TaskContextV1, TrafficContextV1,
)
from src.visualization.agent_io_overlay import AgentIoOverlay


def public_input(step=0):
    distances = np.arange(1, 21) * 2.5
    route = np.column_stack((distances, np.zeros(20), np.zeros(20)))
    task = TaskContextV1(
        waypoints=tuple(RouteWaypointV1(float(d), float(d), 0., 0.) for d in distances),
        waypoints_array=route, validity_mask=np.ones(20, dtype=bool),
    )
    return AgentInputV1(
        InputProfileId.STATE_DECISION_V1,
        CoreObservationV1(np.linspace(0, 1, 259)),
        TrafficContextV1.empty(), task, step,
    )


class TestAgentIoOverlay(unittest.TestCase):
    def setUp(self):
        self.aspect = NodePath("aspect2d")
        engine = SimpleNamespace(
            aspect2d=self.aspect,
            win=SimpleNamespace(getXSize=lambda: 1200, getYSize=lambda: 900),
        )
        self.overlay = AgentIoOverlay(engine)
        self.agent = BFSAgent()
        self.agent.reset(AgentPublicEpisodeContext(action_adapter_id="discrete9_lowbranch_v1"))

    def tearDown(self):
        self.overlay.destroy()
        self.agent.close()

    def show_input(self, inp):
        decision = self.agent.act(inp)
        action = get_action_adapter("discrete9_lowbranch_v1").to_canonical(decision.action_payload)
        self.overlay.update(inp, decision, action, agent_id="stage2_bfs", control_dt_s=.1,
                            act_latency_ms=.25)
        return decision, action

    def test_actual_bfs_input_and_decision_share_pre_action_time(self):
        inp = public_input(17)
        before = inp.core_observation.features.copy()
        decision, action = self.show_input(inp)
        snap = self.overlay.snapshot
        self.assertEqual(snap["step_index"], 17)
        self.assertAlmostEqual(snap["input_time_s"], 1.7)
        self.assertEqual(snap["agent_decision"], decision.to_dict())
        self.assertEqual(snap["canonical_action"], action.to_dict())
        self.assertIn(decision.diagnostics["planner_status"], self.overlay.body.getText())
        self.assertIn("Step 17", self.overlay.body.getText())
        np.testing.assert_array_equal(inp.core_observation.features, before)
        self.assertFalse(inp.core_observation.features.flags.writeable)

    def test_snapshot_copies_input_and_nested_diagnostics(self):
        inp = public_input()
        decision = AgentDecision(7, {"nested": {"values": [1, 2]}})
        action = get_action_adapter("discrete9_lowbranch_v1").to_canonical(7)
        self.overlay.update(inp, decision, action, agent_id="example", control_dt_s=.1)
        decision.diagnostics["nested"]["values"].append(3)
        self.overlay.snapshot["agent_input"]["core_observation"]["features"][0] = 99
        self.assertEqual(self.overlay.snapshot["agent_decision"]["diagnostics"]["nested"]["values"], [1, 2])
        self.assertEqual(inp.core_observation.features[0], 0)

    def test_all_detail_pages_include_last_lidar_and_waypoint(self):
        self.show_input(public_input())
        pages = []
        for _ in self.overlay.PAGE_NAMES:
            pages.append(self.overlay.body.getText())
            self.overlay.next_page()
        self.assertIn("nav[9]", pages[1])
        self.assertIn("235:", pages[4])
        self.assertIn("No actors", pages[5])
        self.assertIn("19: 50.00", pages[6])
        self.assertIn("DIAGNOSTICS", pages[7])
        self.assertEqual(self.overlay.page, 0)

    def test_history_and_nodes_are_bounded_and_hidden_view_stays_current(self):
        count = self.overlay.root.getNumChildren()
        self.overlay.toggle()
        for step in range(25):
            self.show_input(public_input(step))
        self.assertEqual(len(self.overlay.history), 4)
        self.assertEqual(self.overlay.root.getNumChildren(), count)
        self.overlay.toggle()
        self.assertIn("Step 24", self.overlay.body.getText())
        self.assertIn("#0024", self.overlay.body.getText())

    def test_reset_and_destroy_remove_stale_data_nodes_and_key_handlers(self):
        self.show_input(public_input())
        self.overlay.next_page()
        self.overlay.reset()
        self.assertIsNone(self.overlay.snapshot)
        self.assertFalse(self.overlay.history)
        self.assertIn("Waiting", self.overlay.body.getText())
        self.overlay.destroy()
        self.overlay.destroy()
        self.assertEqual(self.aspect.getNumChildren(), 0)
        self.assertFalse(self.overlay._events.isAccepting("tab"))


if __name__ == "__main__":
    unittest.main()
