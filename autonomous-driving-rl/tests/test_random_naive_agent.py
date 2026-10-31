"""Timed random decisions and actual road-topology fork detection."""

from dataclasses import fields, replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from src.agents.random_naive_agent import Stage0RandomNaiveAgent
from src.launcher.registry import build_canonical_agent_registry
from src.platform.action_adapter import Discrete9Adapter
from src.platform.agent import AgentPublicEpisodeContext, build_agent_contract_core
from src.platform.agent_input import (
    AgentInputV1, CoreObservationV1, InputProfileId, RoadDecisionPointV1,
    TaskContextV1, TrafficContextV1, build_agent_input, extract_road_decision_point,
)


def make_input(point=None):
    return AgentInputV1(
        InputProfileId.STATE_DECISION_V1,
        CoreObservationV1(np.full(259, 0.5, dtype=np.float32)),
        TrafficContextV1.empty(),
        replace(TaskContextV1.empty(), decision_point=point),
        0,
    )


def fork(distance=3.0, point_id="junction", turns=("left", "straight", "right")):
    return RoadDecisionPointV1(point_id, distance, turns)


def lane(start, end, length=30.0, heading=0.0):
    return SimpleNamespace(
        index=(start, end, 0), length=length,
        local_coordinates=lambda position: (float(position[0]), float(position[1])),
        position=lambda s, lateral: (s, lateral),
        # All entry tangents are aligned, as in real intersection connectors.
        heading_theta_at=lambda s: heading * min(1.0, max(0.0, s / length)),
    )


def road_fixture(headings=(math.pi / 2, 0.0, -math.pi / 2), position=25.0):
    approach = lane("a", "b")
    exits = {f"exit{i}": [lane("b", f"exit{i}", heading=heading)]
             for i, heading in enumerate(headings)}
    network = SimpleNamespace(graph={"a": {"b": [approach]}, "b": exits})
    vehicle = SimpleNamespace(lane=approach, position=(position, 0.0),
                              heading_theta=0.0, velocity=(0.0, 0.0))
    return vehicle, network


class TestRandomNaiveAgent(unittest.TestCase):
    def setUp(self):
        self.agent = Stage0RandomNaiveAgent(min_hold_s=2, max_hold_s=2)
        self.context = AgentPublicEpisodeContext(action_adapter_id="discrete9_lowbranch_v1")
        self.agent.reset(self.context, agent_seed=17)

    def test_timer_holds_for_exactly_one_period(self):
        decisions = [self.agent.act(make_input()) for _ in range(41)]
        self.assertEqual([i for i, d in enumerate(decisions) if d.diagnostics["action_resample"]],
                         [0, 20, 40])
        self.assertTrue(all(d.action_payload == decisions[0].action_payload for d in decisions[:20]))

    def test_junction_interrupts_and_restarts_full_period(self):
        for _ in range(5):
            self.agent.act(make_input())
        decision = self.agent.act(make_input(fork()))
        self.assertEqual(decision.diagnostics["resample_reason"], "junction")
        self.assertEqual(decision.diagnostics["steps_until_resample"], 19)
        for _ in range(19):
            self.assertFalse(self.agent.act(make_input()).diagnostics["action_resample"])
        self.assertEqual(self.agent.act(make_input()).diagnostics["resample_reason"], "timer")

    def test_persistent_junction_does_not_resample_every_step(self):
        first = self.agent.act(make_input(fork()))
        for _ in range(19):
            decision = self.agent.act(make_input(fork()))
            self.assertFalse(decision.diagnostics["action_resample"])
            self.assertEqual(decision.action_payload, first.action_payload)
        # The normal timer still runs while within a junction.
        self.assertEqual(self.agent.act(make_input(fork())).diagnostics["resample_reason"], "timer")

    def test_new_junction_interrupts_even_when_previous_is_still_visible(self):
        self.agent.act(make_input(fork()))
        self.assertEqual(self.agent.act(make_input(fork(point_id="next"))).diagnostics["resample_reason"],
                         "junction")

    def test_preview_does_not_trigger_before_distance_threshold(self):
        self.agent.act(make_input())
        self.assertFalse(self.agent.act(make_input(fork(distance=8.01))).diagnostics["action_resample"])
        self.assertTrue(self.agent.act(make_input(fork(distance=8))).diagnostics["junction_trigger"])

    def test_brief_localization_flicker_does_not_retrigger(self):
        self.agent.act(make_input(fork()))
        self.agent.act(make_input())
        self.assertFalse(self.agent.act(make_input(fork())).diagnostics["junction_trigger"])

    def test_revisit_after_leaving_rearms_same_junction(self):
        self.agent.act(make_input(fork()))
        for _ in range(3):
            self.agent.act(make_input())
        self.assertTrue(self.agent.act(make_input(fork())).diagnostics["junction_trigger"])

    def test_junction_takes_priority_when_timer_also_expires(self):
        for _ in range(20):
            self.agent.act(make_input())
        self.assertEqual(self.agent.act(make_input(fork())).diagnostics["resample_reason"], "junction")

    def test_only_available_directions_are_chosen_with_physical_steering_sign(self):
        observed = set()
        for seed in range(50):
            self.agent.reset(self.context, agent_seed=seed)
            decision = self.agent.act(make_input(fork(turns=("left", "right"))))
            steering = Discrete9Adapter().to_canonical(decision.action_payload).steering
            self.assertIn(steering, (-0.6, 0.6))
            observed.add(steering)
        self.assertEqual(observed, {-0.6, 0.6})
        self.assertEqual(self.agent._TURN_TO_STEERING["left"], 2)
        self.assertEqual(self.agent._TURN_TO_STEERING["right"], 0)

    def test_crossroad_can_choose_straight_and_both_turns(self):
        selected = set()
        for seed in range(50):
            self.agent.reset(self.context, agent_seed=seed)
            selected.add(self.agent.act(make_input(fork())).action_payload % 3)
        self.assertEqual(selected, {0, 1, 2})

    def test_same_seed_replays_actions_durations_and_reset_state(self):
        def episode(seed):
            self.agent.reset(self.context, agent_seed=seed)
            return [self.agent.act(make_input(fork() if i in (5, 6, 40) else None)).to_dict()
                    for i in range(80)]
        self.assertEqual(episode(42), episode(42))
        self.assertNotEqual(episode(42), episode(43))
        self.assertEqual(episode(42)[0]["diagnostics"]["step_counter"], 1)

    def test_simulation_time_respects_control_dt(self):
        self.agent.reset(replace(self.context, control_dt_s=0.2, control_frequency_hz=5))
        decisions = [self.agent.act(make_input()) for _ in range(11)]
        self.assertEqual([i for i, d in enumerate(decisions) if d.diagnostics["action_resample"]], [0, 10])

    def test_default_random_hold_is_within_two_to_four_seconds(self):
        agent = Stage0RandomNaiveAgent()
        agent.reset(self.context, agent_seed=42)
        durations = set()
        for _ in range(300):
            decision = agent.act(make_input())
            if decision.diagnostics["action_resample"]:
                duration = decision.diagnostics["steps_until_resample"] + 1
                self.assertLessEqual(20, duration)
                self.assertLessEqual(duration, 40)
                durations.add(duration)
            Discrete9Adapter().to_canonical(decision.action_payload)
        self.assertGreater(len(durations), 1)

    def test_lifecycle_requires_reset_and_close_disables_act(self):
        agent = Stage0RandomNaiveAgent()
        with self.assertRaises(RuntimeError):
            agent.act(make_input())
        self.agent.close()
        with self.assertRaises(RuntimeError):
            self.agent.act(make_input())

    def test_invalid_configuration(self):
        for kwargs in ({"min_hold_s": 0}, {"max_hold_s": 1}, {"max_hold_s": float("inf")},
                       {"junction_trigger_distance_m": -1}):
            with self.assertRaises(ValueError):
                Stage0RandomNaiveAgent(**kwargs)

    def test_registration_matches_descriptor_and_stays_in_development(self):
        registry = build_canonical_agent_registry()
        registration = registry.get(self.agent.descriptor.agent_id)
        for name in ("agent_version", "input_profile_id", "action_adapter_id",
                     "inference_stochasticity", "stateful_within_episode", "method_family"):
            self.assertEqual(getattr(registration, name), getattr(self.agent.descriptor, name))
        self.assertTrue(registration.sandbox_eligible)
        self.assertTrue(registration.audit_eligible)
        self.assertFalse(registration.benchmark_eligible)
        self.assertIsInstance(registry.get_factory(registration.agent_id)(), Stage0RandomNaiveAgent)


class TestRoadDecisionPoint(unittest.TestCase):
    def test_real_crossroad_uses_connector_end_headings(self):
        vehicle, network = road_fixture()
        point = extract_road_decision_point(vehicle, network)
        self.assertEqual(point, fork(distance=5, point_id="b"))

    def test_t_junction_has_no_invented_straight_exit(self):
        vehicle, network = road_fixture((math.pi / 2, -math.pi / 2))
        self.assertEqual(extract_road_decision_point(vehicle, network).available_turns, ("left", "right"))

    def test_single_curve_and_parallel_lanes_are_not_junctions(self):
        vehicle, network = road_fixture((math.pi / 2,))
        network.graph["b"]["exit0"].append(lane("b", "exit0", heading=math.pi / 2))
        self.assertIsNone(extract_road_decision_point(vehicle, network))

    def test_uturn_is_excluded_and_does_not_create_false_fork(self):
        vehicle, network = road_fixture((0.0, math.pi))
        self.assertIsNone(extract_road_decision_point(vehicle, network))
        network.graph["b"]["left"] = [lane("b", "left", heading=math.pi / 2)]
        self.assertEqual(extract_road_decision_point(vehicle, network).available_turns, ("left", "straight"))

    def test_single_connector_is_traversed_with_distance_accumulation(self):
        vehicle, network = road_fixture()
        network.graph["c"] = network.graph.pop("b")
        network.graph["b"] = {"c": [lane("b", "c", length=10)]}
        point = extract_road_decision_point(vehicle, network)
        self.assertEqual(point.point_id, "c")
        self.assertEqual(point.distance_m, 15)
        self.assertIsNone(extract_road_decision_point(vehicle, network, lookahead_range_m=14))

    def test_no_lane_or_dead_end_or_cycle_returns_no_fork(self):
        vehicle, network = road_fixture(())
        self.assertIsNone(extract_road_decision_point(vehicle, network))
        network.graph["b"] = {"a": [lane("b", "a", length=1)]}
        self.assertIsNone(extract_road_decision_point(vehicle, network))
        self.assertIsNone(extract_road_decision_point(SimpleNamespace(), network))

    def test_snapshot_copies_turns_and_validates(self):
        turns = ["left", "right"]
        point = fork(turns=turns)
        turns.clear()
        self.assertEqual(point.available_turns, ("left", "right"))
        json.dumps(point.to_dict(), allow_nan=False)
        for kwargs in ({"distance": -1}, {"distance": float("nan")},
                       {"turns": ("left",)}, {"turns": ("left", "left")},
                       {"turns": ("left", "uturn")}):
            with self.assertRaises(ValueError):
                fork(**kwargs)

    def test_input_assembly_exposes_public_topology_and_masks_core_only(self):
        vehicle, network = road_fixture()
        navigation = SimpleNamespace(
            current_lane=vehicle.lane,
            get_current_lane_width=lambda: 3.5,
            get_navi_info=lambda: [1.0, 0.5],
            _target_checkpoints_index=(0, 1),
            checkpoints=["a", "b", "exit1"],
        )
        arguments = dict(raw_obs=np.full(259, 0.5, dtype=np.float32), ego_vehicle=vehicle,
                         traffic_vehicles=[], navigation=navigation, road_network=network, step_index=0)
        public = build_agent_input(**arguments)
        self.assertEqual(public.task_context.decision_point.available_turns, ("left", "straight", "right"))
        agent = Stage0RandomNaiveAgent()
        agent.reset(AgentPublicEpisodeContext())
        self.assertTrue(agent.act(public).diagnostics["junction_trigger"])
        self.assertIn("decision_point", public.to_dict()["task_context"])
        masked = build_agent_input(**arguments, profile_id=InputProfileId.CORE_ONLY_V1)
        self.assertIsNone(masked.task_context.decision_point)

    def test_schema_describes_additive_public_context(self):
        core = build_agent_contract_core()
        self.assertEqual(core["runtime_dataclass_schemas"]["RoadDecisionPointV1"],
                         [f.name for f in fields(RoadDecisionPointV1)])
        self.assertIn("decision_point", core["agent_input_schema"]["task_context"])
        path = Path(__file__).resolve().parents[1] / "configs/platform/agent_contract_v1.json"
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), core)


if __name__ == "__main__":
    unittest.main()
