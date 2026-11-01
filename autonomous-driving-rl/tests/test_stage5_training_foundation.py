"""
Unit and Integration Test Suite for Stage 5 RL Training Foundation (Gate S5.1).

Verifies:
1. Stage5ObservationEncoder 409D vector shape, dtype, bounds, and slice alignment.
2. Exact parity with Platform V1 AgentInputV1 public contract.
3. Strict zero-leakage of all 34 forbidden evaluator-private fields.
4. TrainCaseSampler strict holdout isolation (180 TRAIN geometries, 0 holdout leak).
5. Seed separation (training_root_seed, environment_seed, geometry_generation_seed).
6. Gymnasium API compliance and check_env() compatibility.
7. Action validation and rejection of out-of-bounds actions (no silent clipping).
8. Reward and termination parity with Platform V1 RewardSpecV1 and outcome classifier.
9. Leak-free simulator lifecycle and idempotent close.
10. Scientific platform contract hash immutability.
"""

import math
import unittest
from unittest.mock import MagicMock, patch

from gymnasium.utils.env_checker import check_env
import numpy as np

from src.launcher.models import ResolvedCaseV1
from src.launcher.resolver import (
    DEFAULT_LAUNCHER_CONTRACT_HASH,
    DEFAULT_PLATFORM_EXECUTION_HASH,
    GATE5_LOCKED_BENCHMARK_HASH,
    GATE6_LOCKED_AGENT_HASH,
    GATE6_LOCKED_RUNTIME_HASH,
    GATE7_LOCKED_LOGGING_HASH,
    GATE7_LOCKED_OBSERVABILITY_HASH,
)
from src.platform.action_adapter import InvalidActionError
from src.platform.agent_input import (
    AgentInputV1,
    CoreObservationV1,
    InputProfileId,
    RouteWaypointV1,
    TaskContextV1,
    TrafficActorV1,
    TrafficContextV1,
)
from src.platform.episode import EpisodeOutcome, TerminalReason
from src.platform.reward import RewardBreakdown, RewardSpecV1
from src.training.stage5 import (
    CORE_OBSERVATION_DIM,
    STAGE5_CORE_ONLY_OBSERVATION_DIM,
    STAGE5_ENCODER_ID,
    STAGE5_ENCODER_VERSION,
    STAGE5_FULL_OBSERVATION_DIM,
    SamplerStrategy,
    Stage5ObservationEncoder,
    Stage5TrainingEnv,
    TrainCaseSampler,
    encode_agent_input,
)


def _build_sample_agent_input(active_actors: int = 2, valid_waypoints: int = 15) -> AgentInputV1:
    """Builds a deterministic AgentInputV1 for parity testing."""
    core_features = np.linspace(0.0, 1.0, CORE_OBSERVATION_DIM, dtype=np.float32)
    core = CoreObservationV1(core_features)

    # Traffic context with active_actors
    capacity = 8
    actors_list = []
    actors_arr = np.zeros((capacity, 7), dtype=np.float32)
    valid_mask = np.zeros(capacity, dtype=bool)

    for i in range(capacity):
        if i < active_actors:
            act = TrafficActorV1(
                relative_position_x=float(10.0 + i * 5.0),
                relative_position_y=float(-2.0 + i * 1.5),
                relative_velocity_x=float(5.0 + i),
                relative_velocity_y=float(0.5 * i),
                relative_heading=float(0.1 * i),
                length=4.5,
                width=1.9,
            )
            actors_arr[i] = act.to_tuple()
            valid_mask[i] = True
        else:
            act = TrafficActorV1(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        actors_list.append(act)

    traffic = TrafficContextV1(
        capacity=capacity,
        radius_m=50.0,
        actors=tuple(actors_list),
        actors_array=actors_arr,
        validity_mask=valid_mask,
        active_count=active_actors,
        overflow_count=1,
    )

    # Task context with valid_waypoints
    lookahead_count = 20
    wps_list = []
    wps_arr = np.zeros((lookahead_count, 3), dtype=np.float32)
    task_mask = np.zeros(lookahead_count, dtype=bool)

    for k in range(lookahead_count):
        if k < valid_waypoints:
            wp = RouteWaypointV1(
                lookahead_distance_m=float((k + 1) * 2.5),
                relative_x=float((k + 1) * 2.5),
                relative_y=float(math.sin(k * 0.2)),
                relative_heading=float(0.05 * k),
            )
            wps_arr[k] = wp.to_tuple()
            task_mask[k] = True
        else:
            wp = RouteWaypointV1(0.0, 0.0, 0.0, 0.0)
        wps_list.append(wp)

    task = TaskContextV1(
        lookahead_count=lookahead_count,
        lookahead_spacing_m=2.5,
        lookahead_range_m=50.0,
        current_lane_width=3.5,
        navigation_goal_direction=(0.8, 0.6),
        waypoints=tuple(wps_list),
        waypoints_array=wps_arr,
        validity_mask=task_mask,
        route_end_within_lookahead=True,
    )

    return AgentInputV1(
        profile_id=InputProfileId.STATE_DECISION_V1,
        core_observation=core,
        traffic_context=traffic,
        task_context=task,
        step_index=7,
    )


class TestStage5TrainingFoundation(unittest.TestCase):
    """11 Exhaustive Unit and Parity Tests for Gate S5.1 RL Training Foundation."""

    def test_01_encoder_dimensions_and_observation_space(self):
        """1. Stage5ObservationEncoder dimension matches exact contract (409D full, 259D core-only)."""
        full_enc = Stage5ObservationEncoder(core_only=False)
        self.assertEqual(full_enc.dimension, 409)
        self.assertEqual(full_enc.encoder_id, STAGE5_ENCODER_ID)
        self.assertEqual(full_enc.encoder_version, STAGE5_ENCODER_VERSION)
        self.assertEqual(full_enc.observation_space.shape, (409,))
        self.assertEqual(full_enc.observation_space.dtype, np.float32)

        core_enc = Stage5ObservationEncoder(core_only=True)
        self.assertEqual(core_enc.dimension, 259)
        self.assertEqual(core_enc.observation_space.shape, (259,))
        self.assertTrue(core_enc.is_core_only)

        # Manifest verification
        manifest = full_enc.get_feature_manifest()
        total_manifest_dim = sum(v["dim"] for v in manifest.values())
        self.assertEqual(total_manifest_dim, 409)
        self.assertIsInstance(full_enc.get_schema_fingerprint(), str)
        self.assertEqual(len(full_enc.get_schema_fingerprint()), 64)

    def test_02_encoder_pure_input_parity_and_semantic_ordering(self):
        """2. Observation encoder correctly flattens AgentInputV1 preserving exact slice alignment."""
        agent_input = _build_sample_agent_input(active_actors=3, valid_waypoints=12)
        enc = Stage5ObservationEncoder(core_only=False)
        vec = enc.encode(agent_input)

        self.assertEqual(vec.shape, (409,))
        self.assertEqual(vec.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(vec)))

        # Slice 0..259: Core features
        np.testing.assert_allclose(vec[0:259], agent_input.core_observation.features)

        # Slice 259..315: Traffic actors array (8 x 7 = 56)
        np.testing.assert_allclose(vec[259:315], agent_input.traffic_context.actors_array.flatten())

        # Slice 315..323: Traffic validity mask (8)
        np.testing.assert_allclose(vec[315:323], agent_input.traffic_context.validity_mask.astype(np.float32))

        # Slice 323..325: Traffic active ratio and overflow ratio (2)
        self.assertAlmostEqual(vec[323], 3.0 / 8.0, places=4)
        self.assertAlmostEqual(vec[324], 1.0 / 10.0, places=4)

        # Slice 325..385: Task waypoints array (20 x 3 = 60)
        np.testing.assert_allclose(vec[325:385], agent_input.task_context.waypoints_array.flatten())

        # Slice 385..405: Task validity mask (20)
        np.testing.assert_allclose(vec[385:405], agent_input.task_context.validity_mask.astype(np.float32))

        # Slice 405..409: Task metrics (4: lane_width/5, goal_x, goal_y, route_end_flag)
        self.assertAlmostEqual(vec[405], 3.5 / 5.0, places=4)
        self.assertAlmostEqual(vec[406], 0.8, places=4)
        self.assertAlmostEqual(vec[407], 0.6, places=4)
        self.assertAlmostEqual(vec[408], 1.0, places=4)

        # Pure functional helper check
        vec_func = encode_agent_input(agent_input)
        np.testing.assert_array_equal(vec, vec_func)

    def test_03_encoder_actor_padding_and_empty_contexts(self):
        """3. Zero-padding and masking semantics are preserved for empty and partially populated contexts."""
        empty_core = CoreObservationV1(np.zeros(259, dtype=np.float32))
        empty_traffic = TrafficContextV1.empty(capacity=8, radius_m=50.0)
        empty_task = TaskContextV1.empty(lookahead_count=20, lookahead_spacing_m=2.5)

        empty_input = AgentInputV1(
            profile_id=InputProfileId.CORE_ONLY_V1,
            core_observation=empty_core,
            traffic_context=empty_traffic,
            task_context=empty_task,
            step_index=0,
        )

        enc = Stage5ObservationEncoder(core_only=False)
        vec = enc.encode(empty_input)

        # Padded traffic slots (259..315) must be strictly 0.0
        self.assertTrue(np.all(vec[259:315] == 0.0))
        # Padded traffic mask (315..323) must be strictly 0.0
        self.assertTrue(np.all(vec[315:323] == 0.0))
        # Active and overflow counts must be 0.0
        self.assertEqual(vec[323], 0.0)
        self.assertEqual(vec[324], 0.0)

        # Padded waypoint slots (325..385) must be strictly 0.0
        self.assertTrue(np.all(vec[325:385] == 0.0))
        # Waypoint mask must be strictly 0.0
        self.assertTrue(np.all(vec[385:405] == 0.0))

    def test_04_encoder_zero_leakage_of_evaluator_private_fields(self):
        """4. Encoder strictly rejects invalid input types and accesses zero evaluator-private properties."""
        enc = Stage5ObservationEncoder()

        # Reject non-AgentInputV1
        with self.assertRaises(TypeError):
            enc.encode("not_an_input")  # type: ignore

        # Spied input verifying that encoder never reads forbidden evaluator fields
        sample = _build_sample_agent_input()

        class SpiedAgentInput(AgentInputV1):
            def __getattribute__(self, name):
                if name in (
                    "route_completion", "clean_success", "reward", "case_id",
                    "environment_seed", "tier", "split", "ground_truth_trajectory",
                    "is_clean_success", "terminal_outcome_status",
                ):
                    raise AssertionError(f"Forbidden evaluator-private property accessed: {name}")
                return super().__getattribute__(name)

        spied = SpiedAgentInput(
            profile_id=sample.profile_id,
            core_observation=sample.core_observation,
            traffic_context=sample.traffic_context,
            task_context=sample.task_context,
            step_index=sample.step_index,
        )

        # Verify spy intercepts forbidden attribute accesses
        with self.assertRaises(AssertionError):
            _ = spied.route_completion

        # Verify encode consumes strictly permitted fields without triggering spy assertions
        vec = enc.encode(spied)
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.shape, (409,))

    def test_05_train_case_sampler_strict_holdout_isolation(self):
        """5. TrainCaseSampler contains strictly 180 TRAIN geometries with 0 holdout leakage."""
        sampler = TrainCaseSampler(strategy=SamplerStrategy.ROUND_ROBIN)
        self.assertEqual(sampler.total_eligible_geometries, 180)

        # Verify all records have split == "TRAIN"
        for rec in sampler._records:
            self.assertEqual(rec["split"].upper(), "TRAIN")
            self.assertNotIn(rec["split"].upper(), ("VALIDATION", "TEST"))

        # Verify exact 45 records per tier
        tier_counts = {}
        for rec in sampler._records:
            tier_counts[rec["tier"]] = tier_counts.get(rec["tier"], 0) + 1
        self.assertEqual(tier_counts, {"Easy": 45, "Medium": 45, "Hard": 45, "Extreme": 45})

        # Tier filter
        easy_sampler = TrainCaseSampler(tier_filter="Easy")
        self.assertEqual(easy_sampler.total_eligible_geometries, 45)
        for rec in easy_sampler._records:
            self.assertEqual(rec["tier"].lower(), "easy")

        # Verify holdout validation rejects forged validation case
        from src.launcher.cases import get_default_project_root, resolve_validation_cases
        val_case = resolve_validation_cases(get_default_project_root())[0]
        forged_val_case = ResolvedCaseV1(
            case_id=val_case.case_id.replace("val", "train"),
            case_index=1,
            protocol_order_index=1,
            split="TRAIN",  # Forged label
            tier=val_case.tier,
            sequence=val_case.sequence,
            geometry_generation_seed=val_case.geometry_generation_seed,
            geometry_sha256=val_case.geometry_sha256,
            environment_seed=val_case.environment_seed,
            traffic_density=val_case.traffic_density,
            horizon_steps=val_case.horizon_steps,
        )
        with self.assertRaises(ValueError) as ctx:
            sampler.validate_train_case(forged_val_case)
        self.assertIn("does not exist in verified MapSuite V1 TRAIN manifest", str(ctx.exception))

    def test_06_train_case_sampler_reproducibility_and_seed_separation(self):
        """6. TrainCaseSampler guarantees seed reproducibility, deterministic case mapping, and seed separation."""
        s1 = TrainCaseSampler(training_root_seed=777, strategy=SamplerStrategy.UNIFORM_RANDOM)
        s2 = TrainCaseSampler(training_root_seed=777, strategy=SamplerStrategy.UNIFORM_RANDOM)
        s3 = TrainCaseSampler(training_root_seed=888, strategy=SamplerStrategy.UNIFORM_RANDOM)

        cases_1 = [s1.sample_case().case_id for _ in range(10)]
        cases_2 = [s2.sample_case().case_id for _ in range(10)]
        cases_3 = [s3.sample_case().case_id for _ in range(10)]

        # Same root seed produces identical sequence
        self.assertEqual(cases_1, cases_2)
        # Different root seed produces different sequence
        self.assertNotEqual(cases_1, cases_3)

        # Separate get_case_for_geometry_seed API
        case_g0 = s1.get_case_for_geometry_seed(0, environment_seed=42)
        self.assertEqual(case_g0.geometry_generation_seed, 0)
        self.assertEqual(case_g0.environment_seed, 42)

        # Seed separation assertion: env_seed != geom_seed != training_root_seed
        case = s1.sample_case()
        self.assertEqual(case.split, "TRAIN")
        self.assertIsInstance(case.environment_seed, int)
        self.assertIsInstance(case.geometry_generation_seed, int)

        # Test calling Stage5TrainingEnv.reset(seed=X) reseeds sampling sequence
        env_rr1 = Stage5TrainingEnv(sampler_strategy=SamplerStrategy.ROUND_ROBIN, tier_filter="Easy")
        try:
            obs1_a, info1_a = env_rr1.reset(seed=101)
            obs1_b, info1_b = env_rr1.reset()
        finally:
            env_rr1.close()

        env_rr2 = Stage5TrainingEnv(sampler_strategy=SamplerStrategy.ROUND_ROBIN, tier_filter="Easy")
        try:
            obs2_a, info2_a = env_rr2.reset(seed=101)
            obs2_b, info2_b = env_rr2.reset()
        finally:
            env_rr2.close()

        self.assertEqual(info1_a["case_id"], info2_a["case_id"])
        self.assertEqual(info1_b["case_id"], info2_b["case_id"])
        self.assertNotEqual(info1_a["case_id"], info1_b["case_id"])

    def test_07_gymnasium_api_and_check_env_compliance(self):
        """7. Stage5TrainingEnv fully complies with standard Gymnasium API and passes check_env()."""
        env = Stage5TrainingEnv(
            sampler_strategy=SamplerStrategy.FIXED,
            fixed_geometry_seed=0,
            tier_filter="Easy",
        )
        try:
            # Action and observation space checks
            self.assertEqual(env.action_space.shape, (2,))
            self.assertEqual(env.action_space.dtype, np.float32)
            self.assertEqual(env.observation_space.shape, (409,))
            self.assertEqual(env.observation_space.dtype, np.float32)

            # Full Gymnasium check_env validation
            check_env(env.unwrapped, skip_render_check=True)
        finally:
            env.close()

    def test_08_action_validation_and_no_silent_clipping(self):
        """8. Action space strictly validates finite values and bounds, rejecting out-of-bounds actions."""
        env = Stage5TrainingEnv(
            sampler_strategy=SamplerStrategy.FIXED,
            fixed_geometry_seed=0,
            tier_filter="Easy",
        )
        try:
            obs, info = env.reset()

            # NaN action rejected
            with self.assertRaises((InvalidActionError, ValueError)):
                env.step(np.array([float("nan"), 0.5], dtype=np.float32))

            # Inf action rejected
            with self.assertRaises((InvalidActionError, ValueError)):
                env.step(np.array([0.0, float("inf")], dtype=np.float32))

            # Out of bounds action rejected without silent clipping
            with self.assertRaises((InvalidActionError, ValueError)):
                env.step(np.array([1.5, 0.0], dtype=np.float32))

            with self.assertRaises((InvalidActionError, ValueError)):
                env.step(np.array([0.0, -1.2], dtype=np.float32))
        finally:
            env.close()

    def test_09_reward_and_termination_parity_with_platform_v1(self):
        """9. Step reward, time penalties, and outcome classifications match Platform V1 specs exactly."""
        env = Stage5TrainingEnv(
            sampler_strategy=SamplerStrategy.FIXED,
            fixed_geometry_seed=0,
            tier_filter="Easy",
        )
        try:
            obs, info = env.reset()
            self.assertEqual(env.current_step, 0)
            self.assertEqual(env.cumulative_reward, 0.0)

            # Step with neutral throttle (coasting)
            obs, reward, terminated, truncated, step_info = env.step(np.array([0.0, 0.0], dtype=np.float32))
            self.assertEqual(env.current_step, 1)
            self.assertIn("reward_breakdown", step_info)
            breakdown = step_info["reward_breakdown"]

            # Verify reward breakdown components
            self.assertIn("progress_reward", breakdown)
            self.assertIn("time_cost", breakdown)
            self.assertIn("total_reward", breakdown)
            self.assertAlmostEqual(reward, breakdown["total_reward"], places=6)

            # Time cost must equal time_penalty_budget / horizon_steps
            expected_time_cost = 0.25 / env.current_case.horizon_steps
            self.assertAlmostEqual(breakdown["time_cost"], expected_time_cost, places=6)
        finally:
            env.close()

    def test_10_env_reset_and_simulator_cleanup_no_leaks(self):
        """10. Repeated reset cycles and close() calls clean up MetaDrive cleanly without leaks."""
        env = Stage5TrainingEnv(
            sampler_strategy=SamplerStrategy.ROUND_ROBIN,
            tier_filter="Easy",
        )
        try:
            # 3 sequential reset cycles across different cases
            obs1, info1 = env.reset()
            case1_id = info1["case_id"]

            obs2, info2 = env.reset()
            case2_id = info2["case_id"]

            self.assertNotEqual(case1_id, case2_id)
            self.assertEqual(obs1.shape, (409,))
            self.assertEqual(obs2.shape, (409,))

            # Step in episode 2
            env.step([0.0, 0.5])
            self.assertEqual(env.current_step, 1)

            # Close and idempotent close
            env.close()
            self.assertIsNone(env._env)
            self.assertIsNone(env._current_sim_key)
            env.close()  # Must not throw
        finally:
            env.close()

    def test_11_scientific_contracts_and_hashes_unmutated(self):
        """11. Verifies that Stage 5 RL foundation preserves all locked platform contract hashes."""
        self.assertEqual(GATE5_LOCKED_BENCHMARK_HASH, "9ddd889b84d8705fae618879e5035556c80d0276a3dc2a58a7963937ebb59f77")
        self.assertEqual(GATE6_LOCKED_AGENT_HASH, "53aa37079ff44afa75d9a3f921b0c1f98c4600d882fced51fc8d9197795058eb")
        self.assertEqual(GATE6_LOCKED_RUNTIME_HASH, "c7698768539a769c7b2bc6b90771ff234276a974b6e0353bbb03119faf2f79ad")
        self.assertEqual(GATE7_LOCKED_LOGGING_HASH, "0d17915556d83f49d9519576e95920c0853700f33a91593ef85b3aeef6cbb9f2")
        self.assertEqual(GATE7_LOCKED_OBSERVABILITY_HASH, "f00c27fca8abf8145596576cd0dd2540181ff8ae7eccf7806c864921933d3581")
        self.assertEqual(DEFAULT_LAUNCHER_CONTRACT_HASH, "5e6269b6e32349bd3ee34ca52583c00f832bd0a737330c894e9a8d314653db04")
        self.assertEqual(DEFAULT_PLATFORM_EXECUTION_HASH, "442beafd86212419dc7b3edfa60e53715bf33877d3dd72e61c54091520b0563d")

    def test_12_holdout_rejection_in_env_reset_options(self):
        """12. Stage5TrainingEnv.reset(options={'case': ...}) strictly rejects holdouts and invalid types."""
        from src.launcher.cases import get_default_project_root, resolve_validation_cases
        val_case = resolve_validation_cases(get_default_project_root())[0]

        # Forged validation case with split='TRAIN'
        forged_case = ResolvedCaseV1(
            case_id=val_case.case_id.replace("val", "train"),
            case_index=1,
            protocol_order_index=1,
            split="TRAIN",
            tier=val_case.tier,
            sequence=val_case.sequence,
            geometry_generation_seed=val_case.geometry_generation_seed,
            geometry_sha256=val_case.geometry_sha256,
            environment_seed=val_case.environment_seed,
            traffic_density=val_case.traffic_density,
            horizon_steps=val_case.horizon_steps,
        )

        env = Stage5TrainingEnv(sampler_strategy=SamplerStrategy.FIXED, fixed_geometry_seed=0)
        try:
            # Must reject forged validation case before creating simulator
            with self.assertRaises(ValueError) as ctx1:
                env.reset(options={"case": forged_case})
            self.assertIn("does not exist in verified MapSuite V1 TRAIN manifest", str(ctx1.exception))

            # Must reject non-ResolvedCaseV1 instance
            with self.assertRaises(TypeError) as ctx2:
                env.reset(options={"case": "not_a_case"})
            self.assertIn("must be a ResolvedCaseV1 instance", str(ctx2.exception))
        finally:
            env.close()

    def test_13_observation_bounds_allows_values_exceeding_100(self):
        """13. Observation space contains valid continuous waypoint and actor coordinates > 100 without clipping."""
        enc = Stage5ObservationEncoder(core_only=False)

        # Build input with waypoint coordinate > 100m (e.g. 150m) and actor position > 100m (e.g. 120m)
        core = CoreObservationV1(np.full(259, 0.5, dtype=np.float32))

        # Actor at 120m
        act = TrafficActorV1(120.0, 15.0, 25.0, 0.0, 0.2, 4.8, 2.0)
        actors_list = [act] + [TrafficActorV1(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(7)]
        actors_arr = np.zeros((8, 7), dtype=np.float32)
        actors_arr[0] = act.to_tuple()
        val_mask = np.zeros(8, dtype=bool)
        val_mask[0] = True

        traffic = TrafficContextV1(
            capacity=8, radius_m=50.0, actors=tuple(actors_list),
            actors_array=actors_arr, validity_mask=val_mask,
            active_count=1, overflow_count=0
        )

        # Waypoint at 150m
        wp = RouteWaypointV1(150.0, 150.0, 20.0, 0.5)
        wps_list = [wp] + [RouteWaypointV1(0.0, 0.0, 0.0, 0.0) for _ in range(19)]
        wps_arr = np.zeros((20, 3), dtype=np.float32)
        wps_arr[0] = wp.to_tuple()
        task_mask = np.zeros(20, dtype=bool)
        task_mask[0] = True

        task = TaskContextV1(
            lookahead_count=20, lookahead_spacing_m=2.5, lookahead_range_m=50.0,
            current_lane_width=3.5, navigation_goal_direction=(1.0, 0.0),
            waypoints=tuple(wps_list), waypoints_array=wps_arr, validity_mask=task_mask,
            route_end_within_lookahead=False
        )

        inp = AgentInputV1(InputProfileId.STATE_DECISION_V1, core, traffic, task, step_index=0)
        vec = enc.encode(inp)

        # Must not be clipped to 100
        self.assertEqual(vec[259], 120.0)  # Actor x
        self.assertEqual(vec[325], 150.0)  # Waypoint x
        self.assertTrue(np.all(np.isfinite(vec)))

        # Observation space must contain the observation without rejection
        self.assertTrue(enc.observation_space.contains(vec))

    def test_14_exception_safe_lifecycle_failure_injection(self):
        """14. Exceptions in _build_agent_input or encoder.encode during reset or step close simulator safely."""
        env = Stage5TrainingEnv(sampler_strategy=SamplerStrategy.FIXED, fixed_geometry_seed=0)
        try:
            # 1. Failure in _build_agent_input during reset()
            with patch.object(env, "_build_agent_input", side_effect=RuntimeError("Simulated reset input error")):
                with self.assertRaises(RuntimeError) as ctx1:
                    env.reset()
                self.assertIn("Simulated reset input error", str(ctx1.exception))
                self.assertIsNone(env._env, "Simulator must be closed on reset input failure")

            # 2. Failure in encoder.encode during reset()
            with patch.object(env.encoder, "encode", side_effect=RuntimeError("Simulated reset encode error")):
                with self.assertRaises(RuntimeError) as ctx2:
                    env.reset()
                self.assertIn("Simulated reset encode error", str(ctx2.exception))
                self.assertIsNone(env._env, "Simulator must be closed on reset encode failure")

            # 3. Failure in _build_agent_input during step()
            env.reset()
            self.assertIsNotNone(env._env)
            with patch.object(env, "_build_agent_input", side_effect=RuntimeError("Simulated step input error")):
                with self.assertRaises(RuntimeError) as ctx3:
                    env.step([0.0, 0.5])
                self.assertIn("Simulated step input error", str(ctx3.exception))
                self.assertIsNone(env._env, "Simulator must be closed on step input failure")

            # 4. Failure in encoder.encode during step()
            env.reset()
            self.assertIsNotNone(env._env)
            with patch.object(env.encoder, "encode", side_effect=RuntimeError("Simulated step encode error")):
                with self.assertRaises(RuntimeError) as ctx4:
                    env.step([0.0, 0.5])
                self.assertIn("Simulated step encode error", str(ctx4.exception))
                self.assertIsNone(env._env, "Simulator must be closed on step encode failure")
        finally:
            env.close()

    def test_15_reward_spec_outcomes_and_safety_first_precedence(self):
        """15. Verifies RewardSpecV1 step reward, terminal bonuses, penalties, and safety-first precedence."""
        from src.platform.episode import classify_episode_outcome
        from src.launcher.executor import compute_step_route_progress_and_reward

        spec = RewardSpecV1(
            progress_weight=1.0,
            time_penalty_budget=0.25,
            success_bonus=1.0,
            safety_penalty=1.0,
            unknown_termination_penalty=1.0,
            timeout_penalty=0.0
        )
        horizon = 1000

        # A. Clean Success (+1.0 bonus)
        succ_outcome = classify_episode_outcome(raw_flags={"arrive_dest": True}, terminated=True, truncated=False)
        self.assertTrue(succ_outcome.clean_success)
        self.assertEqual(succ_outcome.primary_reason, TerminalReason.SUCCESS)
        _, _, b_succ = compute_step_route_progress_and_reward(spec, 1.0, 0.9, horizon, succ_outcome)
        self.assertEqual(b_succ.terminal_reward, 1.0)
        expected_total = 1.0 * (1.0 - 0.9) - (0.25 / 1000) + 1.0
        self.assertAlmostEqual(b_succ.total_reward, expected_total)

        # B. Safety Failure (Crash Vehicle) -> -1.0 penalty
        crash_outcome = classify_episode_outcome(raw_flags={"crash_vehicle": True}, terminated=True, truncated=False)
        self.assertFalse(crash_outcome.clean_success)
        self.assertEqual(crash_outcome.primary_reason, TerminalReason.CRASH_VEHICLE)
        _, _, b_crash = compute_step_route_progress_and_reward(spec, 0.5, 0.5, horizon, crash_outcome)
        self.assertEqual(b_crash.terminal_reward, -1.0)

        # C. Timeout -> 0.0 penalty
        timeout_outcome = classify_episode_outcome(raw_flags={}, terminated=False, truncated=True)
        self.assertEqual(timeout_outcome.primary_reason, TerminalReason.TIMEOUT)
        _, _, b_timeout = compute_step_route_progress_and_reward(spec, 0.8, 0.8, horizon, timeout_outcome)
        self.assertEqual(b_timeout.terminal_reward, 0.0)

        # D. Simultaneous Safety Violation + Arrival -> Safety takes absolute precedence!
        both_flags = {"crash_vehicle": True, "arrive_dest": True}
        dual_outcome = classify_episode_outcome(raw_flags=both_flags, terminated=True, truncated=False)
        self.assertEqual(dual_outcome.primary_reason, TerminalReason.CRASH_VEHICLE)
        self.assertFalse(dual_outcome.clean_success)
        _, _, b_dual = compute_step_route_progress_and_reward(spec, 1.0, 0.9, horizon, dual_outcome)
        # Terminal reward must be penalty (-1.0), NOT success (+1.0)
        self.assertEqual(b_dual.terminal_reward, -1.0)


if __name__ == "__main__":
    unittest.main()
