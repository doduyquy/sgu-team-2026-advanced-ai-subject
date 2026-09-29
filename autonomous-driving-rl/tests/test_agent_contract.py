"""
Unit tests for Gate 6: Agent Contract, AgentInputV1, and Action Adapters.
Pure unit tests executing in <0.05s without simulator / Panda3D dependencies.
"""

import copy
import json
from pathlib import Path
import unittest

import numpy as np

from src.platform.action_adapter import (
    CanonicalActionV1,
    ContinuousBox2Adapter,
    Discrete9Adapter,
    Discrete25Adapter,
    InvalidActionError,
    get_action_adapter,
)
from src.platform.agent import (
    AgentDecision,
    AgentDescriptor,
    AgentPublicEpisodeContext,
    DeterministicConstantFixtureAgent,
    DiscreteFixtureAgent,
    InvalidOutputFixtureAgent,
    SeededRandomFixtureAgent,
    StatefulCounterFixtureAgent,
    TechnicalFailureReason,
    build_agent_contract_core,
    validate_diagnostics_payload,
)
from src.platform.agent_input import (
    AgentInputV1,
    CoreObservationV1,
    FORBIDDEN_EVALUATOR_FIELDS,
    InputProfileId,
    RouteWaypointV1,
    TaskContextV1,
    TrafficActorV1,
    TrafficContextV1,
)
from src.platform.protocol import canonical_json_sha256


class TestCoreObservation(unittest.TestCase):
    def setUp(self):
        self.valid_features = np.zeros((259,), dtype=np.float32)
        self.valid_features[0:9] = 0.5  # Ego state
        self.valid_features[9:19] = 0.5  # Navigation
        self.valid_features[19:259] = 0.8  # LiDAR

    def test_exact_shape_and_dtype(self):
        obs = CoreObservationV1(self.valid_features)
        self.assertEqual(obs.features.shape, (259,))
        self.assertEqual(obs.features.dtype, np.float32)
        self.assertEqual(obs.ego_state.shape, (9,))
        self.assertEqual(obs.navigation_vector.shape, (10,))
        self.assertEqual(obs.lidar_points.shape, (240,))

    def test_floating_dtype_conversion_and_validation(self):
        # Converts float64 or list safely to float32
        list_features = [0.5] * 259
        obs = CoreObservationV1(list_features)
        self.assertEqual(obs.features.dtype, np.float32)

        float64_features = np.ones((259,), dtype=np.float64) * 0.5
        obs64 = CoreObservationV1(float64_features)
        self.assertEqual(obs64.features.dtype, np.float32)

    def test_invalid_shape_rejection(self):
        with self.assertRaises(ValueError):
            CoreObservationV1(np.zeros((35,), dtype=np.float32))

        with self.assertRaises(ValueError):
            CoreObservationV1(np.zeros((260,), dtype=np.float32))

    def test_non_finite_rejection(self):
        nan_feat = np.copy(self.valid_features)
        nan_feat[10] = float("nan")
        with self.assertRaises(ValueError):
            CoreObservationV1(nan_feat)

        inf_feat = np.copy(self.valid_features)
        inf_feat[50] = float("inf")
        with self.assertRaises(ValueError):
            CoreObservationV1(inf_feat)

    def test_normalized_bounds_enforcement(self):
        # Reject -0.01
        below_feat = np.copy(self.valid_features)
        below_feat[0] = -0.01
        with self.assertRaises(ValueError):
            CoreObservationV1(below_feat)

        # Reject 1.01
        above_feat = np.copy(self.valid_features)
        above_feat[5] = 1.01
        with self.assertRaises(ValueError):
            CoreObservationV1(above_feat)

        # Valid 0.0 and 1.0 boundary values accepted
        edge_feat = np.zeros((259,), dtype=np.float32)
        edge_feat[0] = 0.0
        edge_feat[-1] = 1.0
        obs = CoreObservationV1(edge_feat)
        self.assertAlmostEqual(obs.features[0], 0.0)
        self.assertAlmostEqual(obs.features[-1], 1.0)

    def test_defensive_immutability(self):
        obs = CoreObservationV1(self.valid_features)
        # Verify read-only flag
        self.assertFalse(obs.features.flags.writeable)

        # Mutating exported array must raise ValueError
        with self.assertRaises(ValueError):
            obs.features[0] = 0.9

        # Mutating original source features must not affect CoreObservation
        self.valid_features[0] = 0.9
        self.assertNotEqual(obs.features[0], 0.9)


class TestTrafficContext(unittest.TestCase):
    def test_empty_traffic_context(self):
        tc = TrafficContextV1.empty(capacity=8, radius_m=50.0)
        self.assertEqual(tc.capacity, 8)
        self.assertEqual(tc.active_count, 0)
        self.assertEqual(tc.overflow_count, 0)
        self.assertEqual(tc.actors_array.shape, (8, 7))
        self.assertEqual(tc.validity_mask.shape, (8,))
        self.assertFalse(any(tc.validity_mask))
        self.assertFalse(tc.actors_array.flags.writeable)
        self.assertFalse(tc.validity_mask.flags.writeable)

    def test_traffic_actor_fields_and_validation(self):
        actor = TrafficActorV1(
            relative_position_x=12.5,
            relative_position_y=-1.75,
            relative_velocity_x=3.2,
            relative_velocity_y=0.1,
            relative_heading=0.05,
            length=4.6,
            width=1.85
        )
        t = actor.to_tuple()
        self.assertEqual(len(t), 7)
        d = actor.to_dict()
        self.assertEqual(len(d), 7)
        self.assertNotIn("distance", d)
        self.assertNotIn("valid", d)

        # Non-finite rejection
        with self.assertRaises(ValueError):
            TrafficActorV1(float("nan"), 0.0, 0.0, 0.0, 0.0, 4.5, 1.8)

        # Negative dimensions rejection
        with self.assertRaises(ValueError):
            TrafficActorV1(10.0, 0.0, 0.0, 0.0, 0.0, -4.5, 1.8)

    def test_actors_sorting_and_validity_mask(self):
        a1 = TrafficActorV1(10.0, 0.0, 0.0, 0.0, 0.0, 4.5, 1.8)
        a2 = TrafficActorV1(5.0, 0.0, 0.0, 0.0, 0.0, 4.5, 1.8)
        empty = TrafficActorV1(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        actors = (a2, a1) + tuple(empty for _ in range(6))
        arr = np.array([a.to_tuple() for a in actors], dtype=np.float32)
        mask = np.array([True, True, False, False, False, False, False, False], dtype=bool)

        tc = TrafficContextV1(
            capacity=8,
            radius_m=50.0,
            actors=actors,
            actors_array=arr,
            validity_mask=mask,
            active_count=2,
            overflow_count=0
        )
        self.assertEqual(tc.active_count, 2)
        self.assertTrue(tc.validity_mask[0])
        self.assertTrue(tc.validity_mask[1])
        self.assertFalse(tc.validity_mask[2])
        self.assertEqual(tc.actors[0].relative_position_x, 5.0)
        self.assertEqual(tc.actors[1].relative_position_x, 10.0)

    def test_malformed_traffic_context_rejection(self):
        actors = tuple(TrafficActorV1(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(8))
        arr = np.zeros((8, 7), dtype=np.float32)
        mask = np.zeros((8,), dtype=bool)

        # Negative capacity
        with self.assertRaises(ValueError):
            TrafficContextV1(capacity=0, radius_m=50.0, actors=(), actors_array=np.zeros((0, 7)), validity_mask=np.zeros((0,)))

        # Mask mismatch with active_count
        with self.assertRaises(ValueError):
            TrafficContextV1(capacity=8, radius_m=50.0, actors=actors, actors_array=arr, validity_mask=mask, active_count=2)


class TestTaskContext(unittest.TestCase):
    def test_empty_task_context(self):
        tc = TaskContextV1.empty(lookahead_count=20, lookahead_spacing_m=2.5)
        self.assertEqual(tc.lookahead_count, 20)
        self.assertEqual(tc.lookahead_range_m, 50.0)
        self.assertEqual(tc.waypoints_array.shape, (20, 3))
        self.assertEqual(tc.validity_mask.shape, (20,))
        self.assertFalse(any(tc.validity_mask))
        self.assertFalse(tc.route_end_within_lookahead)
        self.assertFalse(tc.waypoints_array.flags.writeable)
        self.assertFalse(tc.validity_mask.flags.writeable)

    def test_task_context_waypoints(self):
        wps = []
        for i in range(1, 21):
            wps.append(RouteWaypointV1(
                lookahead_distance_m=i * 2.5,
                relative_x=i * 2.5,
                relative_y=0.0,
                relative_heading=0.0
            ))
        wps_arr = np.array([w.to_tuple() for w in wps], dtype=np.float32)
        mask = np.ones((20,), dtype=bool)

        tc = TaskContextV1(
            lookahead_count=20,
            lookahead_spacing_m=2.5,
            lookahead_range_m=50.0,
            current_lane_width=3.5,
            navigation_goal_direction=(1.0, 0.0),
            waypoints=tuple(wps),
            waypoints_array=wps_arr,
            validity_mask=mask,
            route_end_within_lookahead=False
        )
        self.assertTrue(all(tc.validity_mask))
        self.assertEqual(tc.waypoints[0].relative_x, 2.5)
        self.assertEqual(tc.waypoints[-1].relative_x, 50.0)
        d = tc.to_dict()
        self.assertNotIn("speed_limit_kmh", d)
        self.assertIn("route_end_within_lookahead", d)


class TestAgentInputAndInformationBoundary(unittest.TestCase):
    def setUp(self):
        self.core = CoreObservationV1(np.zeros((259,), dtype=np.float32))
        self.traffic = TrafficContextV1.empty()
        self.task = TaskContextV1.empty()

    def test_agent_input_construction(self):
        agent_in = AgentInputV1(
            profile_id=InputProfileId.STATE_DECISION_V1,
            core_observation=self.core,
            traffic_context=self.traffic,
            task_context=self.task,
            step_index=0
        )
        self.assertEqual(agent_in.step_index, 0)
        self.assertEqual(agent_in.profile_id, InputProfileId.STATE_DECISION_V1)

    def test_forbidden_evaluator_fields_scan(self):
        agent_in = AgentInputV1(
            profile_id=InputProfileId.STATE_DECISION_V1,
            core_observation=self.core,
            traffic_context=self.traffic,
            task_context=self.task,
            step_index=5
        )
        serialized_str = json.dumps(agent_in.to_dict())

        # Assert no forbidden private evaluator fields leak into AgentInput serialization
        for forbidden in FORBIDDEN_EVALUATOR_FIELDS:
            self.assertNotIn(f'"{forbidden}"', serialized_str, f"Forbidden field '{forbidden}' found in AgentInput!")

    def test_public_context_cleanliness(self):
        ctx = AgentPublicEpisodeContext(
            control_frequency_hz=10,
            control_dt_s=0.1,
            horizon_steps=1500,
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="continuous_box2_v1",
            mode="INFERENCE"
        )
        serialized_str = json.dumps(ctx.to_dict())
        for forbidden in ("tier", "split", "case_id", "geometry_sha256", "environment_seed", "env_seed"):
            self.assertNotIn(f'"{forbidden}"', serialized_str)


class TestActionAdaptersAndValidation(unittest.TestCase):
    def test_continuous_box2_adapter_valid_edges(self):
        adapter = ContinuousBox2Adapter()
        edges = [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [0.0, 0.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ]
        for e in edges:
            action = adapter.to_canonical(e)
            self.assertAlmostEqual(action.steering, e[0])
            self.assertAlmostEqual(action.throttle_brake, e[1])
            arr = action.to_numpy()
            self.assertEqual(arr.shape, (2,))
            self.assertEqual(arr.dtype, np.float32)

    def test_continuous_box2_out_of_bounds_no_silent_clipping(self):
        adapter = ContinuousBox2Adapter()
        invalid_inputs = [
            [1.01, 0.0],
            [-1.01, 0.0],
            [0.0, 1.01],
            [0.0, -1.01],
            [float("nan"), 0.0],
            [0.0, float("inf")],
            [0.0],
            [0.0, 0.0, 0.0],
        ]
        for inv in invalid_inputs:
            with self.assertRaises(InvalidActionError, msg=f"Failed to reject invalid action {inv}"):
                adapter.to_canonical(inv)

    def test_discrete25_all_indices(self):
        adapter = Discrete25Adapter()
        for idx in range(25):
            action = adapter.to_canonical(idx)
            expected_steer = (idx % 5) * 0.5 - 1.0
            expected_throttle = (idx // 5) * 0.5 - 1.0
            self.assertAlmostEqual(action.steering, expected_steer)
            self.assertAlmostEqual(action.throttle_brake, expected_throttle)
            self.assertTrue(-1.0 <= action.steering <= 1.0)
            self.assertTrue(-1.0 <= action.throttle_brake <= 1.0)

    def test_discrete25_invalid_indices(self):
        adapter = Discrete25Adapter()
        for inv in [-1, 25, 26, "12", None, 3.5]:
            with self.assertRaises(InvalidActionError):
                adapter.to_canonical(inv)

    def test_discrete9_all_indices(self):
        adapter = Discrete9Adapter()
        for idx in range(9):
            action = adapter.to_canonical(idx)
            st, tb = Discrete9Adapter.GRID[idx]
            self.assertAlmostEqual(action.steering, st)
            self.assertAlmostEqual(action.throttle_brake, tb)

    def test_discrete9_invalid_indices(self):
        adapter = Discrete9Adapter()
        for inv in [-1, 9, 10, None, [0]]:
            with self.assertRaises(InvalidActionError):
                adapter.to_canonical(inv)


class TestAgentPolicyLifecycleAndFixtures(unittest.TestCase):
    def setUp(self):
        self.context = AgentPublicEpisodeContext(horizon_steps=1000)
        self.sample_input = AgentInputV1(
            profile_id=InputProfileId.STATE_DECISION_V1,
            core_observation=CoreObservationV1(np.zeros((259,), dtype=np.float32)),
            traffic_context=TrafficContextV1.empty(),
            task_context=TaskContextV1.empty(),
            step_index=0
        )

    def test_deterministic_constant_agent(self):
        agent = DeterministicConstantFixtureAgent([0.1, 0.5])
        agent.reset(self.context, agent_seed=101)
        dec = agent.act(self.sample_input)
        adapter = ContinuousBox2Adapter()
        canonical = adapter.to_canonical(dec.action_payload)
        self.assertAlmostEqual(canonical.steering, 0.1)
        self.assertAlmostEqual(canonical.throttle_brake, 0.5)

    def test_seeded_random_repeatability(self):
        # Same agent_seed -> identical action sequence
        agent1 = SeededRandomFixtureAgent()
        agent1.reset(self.context, agent_seed=4242)
        acts1 = [agent1.act(self.sample_input).action_payload for _ in range(5)]

        agent2 = SeededRandomFixtureAgent()
        agent2.reset(self.context, agent_seed=4242)
        acts2 = [agent2.act(self.sample_input).action_payload for _ in range(5)]

        self.assertEqual(acts1, acts2)

        # Different agent_seed -> different action sequence
        agent3 = SeededRandomFixtureAgent()
        agent3.reset(self.context, agent_seed=9999)
        acts3 = [agent3.act(self.sample_input).action_payload for _ in range(5)]
        self.assertNotEqual(acts1, acts3)

    def test_stateful_counter_reset(self):
        agent = StatefulCounterFixtureAgent(initial_static_weight=2.5)
        agent.reset(self.context, agent_seed=101)

        # Episode 1: step 3 times
        agent.act(self.sample_input)
        agent.act(self.sample_input)
        dec = agent.act(self.sample_input)
        self.assertEqual(dec.diagnostics["step_counter"], 3)
        self.assertEqual(agent.static_weight, 2.5)

        # Reset episode
        agent.reset(self.context, agent_seed=101)
        self.assertEqual(agent.step_counter, 0)
        self.assertEqual(agent.static_weight, 2.5)  # Static weight preserved!

        # Episode 2 starts from step 1
        dec2 = agent.act(self.sample_input)
        self.assertEqual(dec2.diagnostics["step_counter"], 1)

    def test_diagnostics_validation(self):
        # Valid JSON-safe bounded primitives
        valid_diag = {
            "step": 1,
            "score": 0.95,
            "valid": True,
            "label": "test",
            "nested": {"a": [1, 2, 3]}
        }
        dec = AgentDecision(action_payload=[0.0, 0.0], diagnostics=valid_diag)
        self.assertEqual(dec.diagnostics["step"], 1)

        # Non-finite float rejected
        with self.assertRaises(ValueError):
            AgentDecision(action_payload=[0.0, 0.0], diagnostics={"nan_val": float("nan")})

        # Illegal non-primitive object rejected
        with self.assertRaises(TypeError):
            AgentDecision(action_payload=[0.0, 0.0], diagnostics={"live_obj": object()})

        # Excessively deep structure rejected
        deep_diag = {"l1": {"l2": {"l3": {"l4": {"l5": "too_deep"}}}}}
        with self.assertRaises(ValueError):
            AgentDecision(action_payload=[0.0, 0.0], diagnostics=deep_diag)

    def test_invalid_output_handling(self):
        adapter = ContinuousBox2Adapter()

        # NaN output
        agent_nan = InvalidOutputFixtureAgent(invalid_mode="nan")
        dec = agent_nan.act(self.sample_input)
        with self.assertRaises(InvalidActionError):
            adapter.to_canonical(dec.action_payload)

        # Out-of-bounds output
        agent_oob = InvalidOutputFixtureAgent(invalid_mode="out_of_bounds")
        dec_oob = agent_oob.act(self.sample_input)
        with self.assertRaises(InvalidActionError):
            adapter.to_canonical(dec_oob.action_payload)

        # Exception mode
        agent_ex = InvalidOutputFixtureAgent(invalid_mode="exception")
        with self.assertRaises(RuntimeError):
            agent_ex.act(self.sample_input)


class TestAgentContractHashMutations(unittest.TestCase):
    def test_discrete9_mutation_changes_hash(self):
        core_default = build_agent_contract_core()
        h_default = canonical_json_sha256(core_default)

        # Mutate one Discrete9 action
        mutated_grid = copy.deepcopy(core_default["actuator_contract"]["certified_adapters"]["discrete9_lowbranch_v1"]["mappings"])
        mutated_grid[0]["steering"] = -0.55  # Changed from -0.6
        core_mutated = build_agent_contract_core(custom_adapter_grids={"discrete9": mutated_grid})
        h_mutated = canonical_json_sha256(core_mutated)

        self.assertNotEqual(h_default, h_mutated)

    def test_discrete25_mutation_changes_hash(self):
        core_default = build_agent_contract_core()
        h_default = canonical_json_sha256(core_default)

        mutated_grid = copy.deepcopy(core_default["actuator_contract"]["certified_adapters"]["discrete25_native_v1"]["mappings"])
        mutated_grid[0]["steering"] = -0.99
        core_mutated = build_agent_contract_core(custom_adapter_grids={"discrete25": mutated_grid})
        h_mutated = canonical_json_sha256(core_mutated)

        self.assertNotEqual(h_default, h_mutated)

    def test_evaluation_rule_mutation_changes_hash(self):
        core_default = build_agent_contract_core()
        h_default = canonical_json_sha256(core_default)

        core_mutated = build_agent_contract_core(custom_rules={"no_online_learning_during_evaluation": "Mutated rule"})
        h_mutated = canonical_json_sha256(core_mutated)

        self.assertNotEqual(h_default, h_mutated)

    def test_latency_boundary_mutation_changes_hash(self):
        core_default = build_agent_contract_core()
        h_default = canonical_json_sha256(core_default)

        core_mutated = build_agent_contract_core(custom_latency_boundary={"nominal_realtime_budget_ms": 50.0})
        h_mutated = canonical_json_sha256(core_mutated)

        self.assertNotEqual(h_default, h_mutated)

    def test_json_whitespace_invariance(self):
        core = build_agent_contract_core()
        h1 = canonical_json_sha256(core)

        # Serialize with indentation and reload
        indented_json = json.dumps(core, indent=4)
        reloaded = json.loads(indented_json)
        h2 = canonical_json_sha256(reloaded)

        self.assertEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
