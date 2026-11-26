"""Decision text must remain faithful to real planner diagnostics and actions."""

import unittest

from src.platform.agent import AgentDecision
from src.platform.action_adapter import get_action_adapter
from src.visualization.planner_trace import explain_search_decision


class PlannerTraceTest(unittest.TestCase):
    def test_found_path_reports_exact_diagnostics_and_applied_action(self):
        decision = AgentDecision(7, {"planner_status": "PATH_FOUND", "expansions": 28,
                                     "local_path_cost": 12.75, "target_x": 6., "target_y": -.8})
        action = get_action_adapter("discrete9_lowbranch_v1").to_canonical(decision.action_payload)
        trace = explain_search_decision("stage2_bfs", decision, action)
        self.assertIn("28 nút", trace["result"])
        self.assertIn("12.75", trace["result"])
        self.assertIn("y=-0.8", trace["control"])
        self.assertIn("tăng ga", trace["control"])
        self.assertIn("ít cạnh", trace["strategy"])
        self.assertNotIn("an toàn", trace["text"])

    def test_failed_search_does_not_invent_a_collision_cause(self):
        decision = AgentDecision(1, {"planner_status": "NO_PATH", "expansions": 256})
        action = get_action_adapter("discrete9_lowbranch_v1").to_canonical(1)
        trace = explain_search_decision("stage2_hybrid_astar", decision, action)
        self.assertIn("256 nút", trace["result"])
        self.assertIn("phanh", trace["control"])
        self.assertNotIn("va chạm", trace["text"])
        self.assertNotIn("Bám điểm", trace["control"])

    def test_missing_route_is_distinct_from_search_exhaustion(self):
        decision = AgentDecision(1, {"planner_status": "NO_PUBLIC_ROUTE"})
        action = get_action_adapter("discrete9_lowbranch_v1").to_canonical(1)
        trace = explain_search_decision("stage2_astar", decision, action)
        self.assertIn("Không có điểm dẫn đường", trace["result"])
        self.assertNotIn("nút", trace["result"])


if __name__ == "__main__":
    unittest.main()
