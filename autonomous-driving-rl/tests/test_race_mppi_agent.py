"""Public input boundary, seed replay and physically meaningful MPPI responses."""

import unittest

import numpy as np

from src.agents.mppi_agent import MPPIAgent, MPPIConfig
from src.platform import AgentPublicEpisodeContext, get_action_adapter
from src.platform.agent_input import (
    AgentInputV1, CoreObservationV1, InputProfileId, RouteWaypointV1,
    TaskContextV1, TrafficContextV1,
)


def observation(speed=3., radius=None, empty=False):
    features = np.full(259, .5, dtype=np.float32)
    features[3] = (speed * 3.6 + 1) / 81
    features[19:] = 1.
    distances = np.arange(1, 21) * 2.5
    route = np.column_stack((distances, np.zeros(20), np.zeros(20)))
    if radius:
        route = np.column_stack((radius * np.sin(distances / radius),
                                 radius * (1 - np.cos(distances / radius)), distances / radius))
    task = TaskContextV1.empty() if empty else TaskContextV1(
        waypoints=tuple(RouteWaypointV1(float(d), *map(float, p)) for d, p in zip(distances, route)),
        waypoints_array=route, validity_mask=np.ones(20, dtype=bool),
    )
    return AgentInputV1(InputProfileId.STATE_DECISION_V1, CoreObservationV1(features),
                        TrafficContextV1.empty(), task, 0)


class TestMPPI(unittest.TestCase):
    def policy(self, seed=42):
        agent = MPPIAgent()
        agent.reset(AgentPublicEpisodeContext(), seed)
        return agent

    def test_seeded_replay_resets_receding_horizon_and_validates_continuous_actions(self):
        agent = self.policy()
        inputs = observation()
        before = inputs.core_observation.features.copy()
        first = agent.act(inputs)
        agent.act(inputs)
        agent.reset(AgentPublicEpisodeContext(), 42)
        repeated = agent.act(inputs)
        np.testing.assert_array_equal(first.action_payload, repeated.action_payload)
        np.testing.assert_array_equal(before, inputs.core_observation.features)
        get_action_adapter("continuous_box2_v1").to_canonical(first.action_payload)
        self.assertTrue(first.diagnostics["prototype"])
        self.assertEqual(len(first.diagnostics["best_sample_path"]), agent.config.horizon)

    def test_curved_routes_turn_in_the_observed_direction(self):
        left, right = [], []
        for seed in (1, 7, 42, 101):
            left.append(self.policy(seed).act(observation(radius=20.)).action_payload[0])
            right.append(self.policy(seed).act(observation(radius=-20.)).action_payload[0])
        self.assertGreater(np.mean(left), .05)
        self.assertLess(np.mean(right), -.05)

    def test_straight_route_keeps_steering_small_and_throttle_responds_to_speed(self):
        slow = self.policy().act(observation(speed=0.)).action_payload
        fast = self.policy().act(observation(speed=10.)).action_payload
        self.assertLess(abs(slow[0]), .1)
        self.assertGreater(slow[1], 0.)
        self.assertLess(fast[1], slow[1])

    def test_missing_route_brakes_without_claiming_success(self):
        result = self.policy().act(observation(empty=True))
        self.assertLess(result.action_payload[1], 0.)
        self.assertEqual(result.diagnostics["planner_status"], "NO_PUBLIC_ROUTE")

    def test_invalid_context_budget_and_lifecycle_fail_loudly(self):
        for kwargs in ({"samples": 1}, {"horizon": 0}, {"temperature": float("nan")}, {"steering_noise": 0}):
            with self.assertRaises(ValueError):
                MPPIConfig(**kwargs)
        agent = MPPIAgent()
        with self.assertRaises(RuntimeError):
            agent.act(observation())
        with self.assertRaises(ValueError):
            agent.reset(AgentPublicEpisodeContext(action_adapter_id="discrete9_lowbranch_v1"))
        agent.reset(AgentPublicEpisodeContext())
        agent.close()
        with self.assertRaises(RuntimeError):
            agent.act(observation())


if __name__ == "__main__":
    unittest.main()
