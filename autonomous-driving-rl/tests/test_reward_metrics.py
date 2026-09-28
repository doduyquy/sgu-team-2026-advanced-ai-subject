"""
Unit tests for Platform V1 RewardSpecV1 and EvaluationMetricsV1 pure helpers.
Does not require Panda3D or MetaDrive simulator initialization.
"""

import math
import unittest

from src.platform import (
    AggregateMetrics,
    EpisodeOutcome,
    EpisodeRecord,
    RewardBreakdown,
    RewardSpecV1,
    TerminalReason,
    classify_episode_outcome,
    compute_aggregate_metrics,
)


class TestRewardSpec(unittest.TestCase):
    def setUp(self):
        self.spec = RewardSpecV1(
            progress_weight=1.0,
            time_penalty_budget=0.25,
            success_bonus=1.0,
            safety_penalty=1.0,
            timeout_penalty=0.0,
        )

    def test_positive_progress_step(self):
        # 0.01 progress on horizon=1000 -> time_cost = 0.25 / 1000 = 0.00025
        b = self.spec.compute_step_reward(delta_route_completion=0.01, horizon_steps=1000)
        self.assertAlmostEqual(b.progress_reward, 0.01, places=5)
        self.assertAlmostEqual(b.time_cost, 0.00025, places=5)
        self.assertAlmostEqual(b.terminal_reward, 0.0, places=5)
        self.assertAlmostEqual(b.total_reward, 0.01 - 0.00025, places=5)

    def test_negative_progress_step(self):
        # Moving backwards: -0.005 progress
        b = self.spec.compute_step_reward(delta_route_completion=-0.005, horizon_steps=1000)
        self.assertAlmostEqual(b.progress_reward, -0.005, places=5)
        self.assertAlmostEqual(b.total_reward, -0.005 - 0.00025, places=5)

    def test_stationary_transition(self):
        # Stationary: delta = 0
        b = self.spec.compute_step_reward(delta_route_completion=0.0, horizon_steps=1000)
        self.assertAlmostEqual(b.progress_reward, 0.0, places=5)
        self.assertAlmostEqual(b.total_reward, -0.00025, places=5)

    def test_clean_success_terminal_step(self):
        raw = {"arrive_dest": True, "out_of_road": False, "crash_vehicle": False}
        outcome = classify_episode_outcome(raw, terminated=True)
        b = self.spec.compute_step_reward(delta_route_completion=0.005, horizon_steps=1000, outcome=outcome)
        self.assertAlmostEqual(b.terminal_reward, 1.0, places=5)
        self.assertAlmostEqual(b.total_reward, 0.005 - 0.00025 + 1.0, places=5)

    def test_safety_failure_terminal_step(self):
        raw = {"arrive_dest": False, "crash_vehicle": True}
        outcome = classify_episode_outcome(raw, terminated=True)
        b = self.spec.compute_step_reward(delta_route_completion=0.0, horizon_steps=1000, outcome=outcome)
        self.assertAlmostEqual(b.terminal_reward, -1.0, places=5)
        self.assertAlmostEqual(b.total_reward, 0.0 - 0.00025 - 1.0, places=5)

    def test_simultaneous_arrival_and_crash_safety_precedence(self):
        # Arrival + crash must receive safety penalty, NOT success bonus!
        raw = {"arrive_dest": True, "crash_vehicle": True}
        outcome = classify_episode_outcome(raw, terminated=True)
        self.assertFalse(outcome.clean_success)
        self.assertEqual(outcome.primary_reason, TerminalReason.CRASH_VEHICLE)

        b = self.spec.compute_step_reward(delta_route_completion=0.001, horizon_steps=1000, outcome=outcome)
        self.assertAlmostEqual(b.terminal_reward, -1.0, places=5)
        # Never positive on simultaneous crash!
        self.assertTrue(b.total_reward < 0)

    def test_timeout_step(self):
        raw = {"max_step": True}
        outcome = classify_episode_outcome(raw, truncated=True)
        b = self.spec.compute_step_reward(delta_route_completion=0.0, horizon_steps=1000, outcome=outcome)
        self.assertAlmostEqual(b.terminal_reward, 0.0, places=5)
        self.assertAlmostEqual(b.total_reward, -0.00025, places=5)

    def test_horizon_normalized_time_cost_scaling(self):
        # For horizon=2000, time cost is 0.25 / 2000 = 0.000125
        b2000 = self.spec.compute_step_reward(delta_route_completion=0.0, horizon_steps=2000)
        self.assertAlmostEqual(b2000.time_cost, 0.000125, places=6)

    def test_component_sum_consistency(self):
        b = self.spec.compute_step_reward(delta_route_completion=0.01234, horizon_steps=1500)
        expected = round(b.progress_reward - b.time_cost + b.terminal_reward, 6)
        self.assertAlmostEqual(b.total_reward, expected, places=6)

    def test_invalid_reward_spec_configuration(self):
        with self.assertRaises(ValueError):
            RewardSpecV1(progress_weight=-1.0)
        with self.assertRaises(ValueError):
            RewardSpecV1(time_penalty_budget=-0.1)
        with self.assertRaises(ValueError):
            RewardSpecV1(success_bonus=-0.5)
        with self.assertRaises(ValueError):
            RewardSpecV1(safety_penalty=-1.0)
        with self.assertRaises(ValueError):
            self.spec.compute_step_reward(0.01, horizon_steps=0)


class TestEvaluationMetrics(unittest.TestCase):
    def _create_record(
        self,
        clean_success: bool = False,
        raw_arrival: bool = False,
        primary_reason: str = "OUT_OF_ROAD",
        final_rc: float = 0.5,
        time_to_success: float = None,
        ret: float = 10.0,
        raw_crash_veh: bool = False,
        raw_out: bool = True,
    ) -> EpisodeRecord:
        return EpisodeRecord(
            tier="Medium",
            sequence="SCXCS",
            scenario_seed=0,
            terminated=True,
            truncated=(primary_reason == "TIMEOUT"),
            primary_reason=primary_reason,
            raw_arrival=raw_arrival,
            clean_success=clean_success,
            final_route_completion=final_rc,
            max_route_completion=final_rc,
            raw_crash_vehicle=raw_crash_veh,
            raw_crash_object=False,
            raw_crash_building=False,
            raw_crash_human=False,
            raw_crash_sidewalk=False,
            raw_out_of_road=raw_out,
            episode_steps=500,
            simulation_time_s=50.0,
            mean_speed_kmh=25.0,
            max_speed_kmh=35.0,
            episode_return=ret,
            time_to_clean_success_s=time_to_success,
        )

    def test_clean_success_aggregation(self):
        records = [
            self._create_record(clean_success=True, raw_arrival=True, primary_reason="SUCCESS", final_rc=1.0, time_to_success=40.0),
            self._create_record(clean_success=True, raw_arrival=True, primary_reason="SUCCESS", final_rc=1.0, time_to_success=60.0),
            self._create_record(clean_success=False, raw_arrival=False, primary_reason="OUT_OF_ROAD", final_rc=0.4),
            self._create_record(clean_success=False, raw_arrival=False, primary_reason="TIMEOUT", final_rc=0.8),
        ]
        agg = compute_aggregate_metrics(records)
        self.assertEqual(agg.total_episodes, 4)
        self.assertAlmostEqual(agg.clean_success_rate, 0.5, places=3)
        self.assertAlmostEqual(agg.raw_arrival_rate, 0.5, places=3)
        self.assertAlmostEqual(agg.timeout_rate, 0.25, places=3)
        self.assertAlmostEqual(agg.out_of_road_rate, 0.25, places=3)
        # Conditional time to success
        self.assertAlmostEqual(agg.mean_time_to_clean_success_s, 50.0, places=2)
        self.assertAlmostEqual(agg.median_time_to_clean_success_s, 50.0, places=2)

    def test_raw_arrival_greater_than_clean_success(self):
        # 1 clean success, 1 arrival WITH crash, 1 out of road
        records = [
            self._create_record(clean_success=True, raw_arrival=True, primary_reason="SUCCESS", final_rc=1.0, time_to_success=45.0),
            self._create_record(clean_success=False, raw_arrival=True, primary_reason="CRASH_VEHICLE", final_rc=1.0, raw_crash_veh=True),
            self._create_record(clean_success=False, raw_arrival=False, primary_reason="OUT_OF_ROAD", final_rc=0.3),
        ]
        agg = compute_aggregate_metrics(records)
        self.assertAlmostEqual(agg.clean_success_rate, 1 / 3, places=3)
        self.assertAlmostEqual(agg.raw_arrival_rate, 2 / 3, places=3)
        # Verify raw arrival exceeds clean success!
        self.assertTrue(agg.raw_arrival_rate > agg.clean_success_rate)

    def test_mutually_exclusive_primary_outcome_rates_sum_to_one(self):
        records = [
            self._create_record(primary_reason="SUCCESS", clean_success=True, raw_arrival=True, final_rc=1.0, time_to_success=30.0),
            self._create_record(primary_reason="CRASH_VEHICLE", raw_crash_veh=True),
            self._create_record(primary_reason="OUT_OF_ROAD"),
            self._create_record(primary_reason="TIMEOUT"),
        ]
        agg = compute_aggregate_metrics(records)
        sum_rates = (
            agg.success_rate
            + agg.timeout_rate
            + agg.crash_human_rate
            + agg.crash_vehicle_rate
            + agg.crash_object_rate
            + agg.crash_building_rate
            + agg.crash_sidewalk_rate
            + agg.out_of_road_rate
            + agg.unknown_termination_rate
        )
        self.assertAlmostEqual(sum_rates, 1.0, places=4)

    def test_no_success_conditional_time_is_none(self):
        # Zero successes -> conditional time must be None, NOT 0.0!
        records = [
            self._create_record(primary_reason="OUT_OF_ROAD"),
            self._create_record(primary_reason="TIMEOUT"),
        ]
        agg = compute_aggregate_metrics(records)
        self.assertIsNone(agg.mean_time_to_clean_success_s)
        self.assertIsNone(agg.median_time_to_clean_success_s)

    def test_route_completion_statistics(self):
        records = [
            self._create_record(final_rc=0.2),
            self._create_record(final_rc=0.4),
            self._create_record(final_rc=0.9),
        ]
        agg = compute_aggregate_metrics(records)
        self.assertAlmostEqual(agg.mean_final_route_completion, 0.5, places=3)
        self.assertAlmostEqual(agg.median_final_route_completion, 0.4, places=3)

    def test_empty_records_raises_error(self):
        with self.assertRaises(ValueError):
            compute_aggregate_metrics([])


if __name__ == "__main__":
    unittest.main()
