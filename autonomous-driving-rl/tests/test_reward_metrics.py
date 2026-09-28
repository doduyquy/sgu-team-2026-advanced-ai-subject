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
        self.assertEqual(b.progress_reward, 0.01)
        self.assertEqual(b.time_cost, 0.00025)
        self.assertEqual(b.terminal_reward, 0.0)
        self.assertAlmostEqual(b.total_reward, 0.01 - 0.00025, places=10)
        self.assertEqual(b.raw_total_reward, b.total_reward)
        self.assertFalse(b.clipping_applied)

    def test_negative_progress_step(self):
        # Moving backwards: -0.005 progress
        b = self.spec.compute_step_reward(delta_route_completion=-0.005, horizon_steps=1000)
        self.assertEqual(b.progress_reward, -0.005)
        self.assertAlmostEqual(b.total_reward, -0.005 - 0.00025, places=10)

    def test_stationary_transition(self):
        # Stationary: delta = 0
        b = self.spec.compute_step_reward(delta_route_completion=0.0, horizon_steps=1000)
        self.assertEqual(b.progress_reward, 0.0)
        self.assertEqual(b.total_reward, -0.00025)

    def test_unrounded_time_cost_telescoping(self):
        # Verify that sum of H step time costs strictly equals time_penalty_budget across diverse horizons
        horizons = [1049, 1930, 2816, 4000]
        for H in horizons:
            b = self.spec.compute_step_reward(delta_route_completion=0.0, horizon_steps=H)
            total_time_penalty = b.time_cost * H
            self.assertAlmostEqual(
                total_time_penalty,
                self.spec.time_penalty_budget,
                places=12,
                msg=f"Failed time cost telescoping for horizon {H}"
            )

    def test_clean_success_terminal_step(self):
        raw = {"arrive_dest": True, "out_of_road": False, "crash_vehicle": False}
        outcome = classify_episode_outcome(raw, terminated=True)
        b = self.spec.compute_step_reward(delta_route_completion=0.005, horizon_steps=1000, outcome=outcome)
        self.assertEqual(b.terminal_reward, 1.0)
        self.assertAlmostEqual(b.total_reward, 0.005 - 0.00025 + 1.0, places=10)

    def test_safety_failure_terminal_step(self):
        raw = {"arrive_dest": False, "crash_vehicle": True}
        outcome = classify_episode_outcome(raw, terminated=True)
        b = self.spec.compute_step_reward(delta_route_completion=0.0, horizon_steps=1000, outcome=outcome)
        self.assertEqual(b.terminal_reward, -1.0)
        self.assertAlmostEqual(b.total_reward, 0.0 - 0.00025 - 1.0, places=10)

    def test_simultaneous_arrival_and_crash_safety_precedence(self):
        raw = {"arrive_dest": True, "crash_vehicle": True}
        outcome = classify_episode_outcome(raw, terminated=True)
        self.assertFalse(outcome.clean_success)
        self.assertEqual(outcome.primary_reason, TerminalReason.CRASH_VEHICLE)

        b = self.spec.compute_step_reward(delta_route_completion=0.001, horizon_steps=1000, outcome=outcome)
        self.assertEqual(b.terminal_reward, -1.0)
        self.assertTrue(b.total_reward < 0)

    def test_timeout_step(self):
        raw = {"max_step": True}
        outcome = classify_episode_outcome(raw, truncated=True)
        b = self.spec.compute_step_reward(delta_route_completion=0.0, horizon_steps=1000, outcome=outcome)
        self.assertEqual(b.terminal_reward, 0.0)
        self.assertEqual(b.total_reward, -0.00025)

    def test_clipping_semantics(self):
        # Configure clipping: [-0.5, 0.5]
        clipped_spec = RewardSpecV1(clip_min=-0.5, clip_max=0.5, success_bonus=1.0)
        raw = {"arrive_dest": True}
        outcome = classify_episode_outcome(raw, terminated=True)
        b = clipped_spec.compute_step_reward(delta_route_completion=0.0, horizon_steps=1000, outcome=outcome)
        self.assertAlmostEqual(b.raw_total_reward, 1.0 - 0.00025, places=5)
        self.assertEqual(b.total_reward, 0.5)
        self.assertTrue(b.clipping_applied)

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
            RewardSpecV1(clip_min=1.0, clip_max=0.5)
        with self.assertRaises(ValueError):
            self.spec.compute_step_reward(0.01, horizon_steps=0)


class TestEvaluationMetrics(unittest.TestCase):
    def _create_record(
        self,
        clean_success: bool = False,
        raw_arrival: bool = False,
        primary_reason: TerminalReason = TerminalReason.OUT_OF_ROAD,
        final_rc: float = 0.5,
        time_to_success: float = None,
        ret: float = 10.0,
        raw_crash_veh: bool = False,
        raw_crash_human: bool = False,
        raw_crash_bld: bool = False,
        raw_out: bool = False,
    ) -> EpisodeRecord:
        return EpisodeRecord(
            tier="Medium",
            sequence="SCXCS",
            scenario_seed=0,
            terminated=(primary_reason != TerminalReason.TIMEOUT),
            truncated=(primary_reason == TerminalReason.TIMEOUT),
            primary_reason=primary_reason,
            raw_arrival=raw_arrival,
            clean_success=clean_success,
            final_route_completion=final_rc,
            max_route_completion=final_rc,
            raw_crash_vehicle=raw_crash_veh,
            raw_crash_object=False,
            raw_crash_building=raw_crash_bld,
            raw_crash_human=raw_crash_human,
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
            self._create_record(clean_success=True, raw_arrival=True, primary_reason=TerminalReason.SUCCESS, final_rc=1.0, time_to_success=40.0),
            self._create_record(clean_success=True, raw_arrival=True, primary_reason=TerminalReason.SUCCESS, final_rc=1.0, time_to_success=60.0),
            self._create_record(clean_success=False, raw_arrival=False, primary_reason=TerminalReason.OUT_OF_ROAD, final_rc=0.4, raw_out=True),
            self._create_record(clean_success=False, raw_arrival=False, primary_reason=TerminalReason.TIMEOUT, final_rc=0.8),
        ]
        agg = compute_aggregate_metrics(records)
        self.assertEqual(agg.total_episodes, 4)
        self.assertAlmostEqual(agg.clean_success_rate, 0.5, places=3)
        self.assertAlmostEqual(agg.raw_arrival_rate, 0.5, places=3)
        self.assertAlmostEqual(agg.timeout_rate, 0.25, places=3)
        self.assertAlmostEqual(agg.out_of_road_rate, 0.25, places=3)
        self.assertAlmostEqual(agg.mean_time_to_clean_success_s, 50.0, places=2)
        self.assertAlmostEqual(agg.median_time_to_clean_success_s, 50.0, places=2)

    def test_raw_arrival_greater_than_clean_success(self):
        records = [
            self._create_record(clean_success=True, raw_arrival=True, primary_reason=TerminalReason.SUCCESS, final_rc=1.0, time_to_success=45.0),
            self._create_record(clean_success=False, raw_arrival=True, primary_reason=TerminalReason.CRASH_VEHICLE, final_rc=1.0, raw_crash_veh=True),
            self._create_record(clean_success=False, raw_arrival=False, primary_reason=TerminalReason.OUT_OF_ROAD, final_rc=0.3, raw_out=True),
        ]
        agg = compute_aggregate_metrics(records)
        self.assertAlmostEqual(agg.clean_success_rate, 1 / 3, places=3)
        self.assertAlmostEqual(agg.raw_arrival_rate, 2 / 3, places=3)
        self.assertTrue(agg.raw_arrival_rate > agg.clean_success_rate)

    def test_mutually_exclusive_primary_outcome_rates_sum_to_one(self):
        records = [
            self._create_record(primary_reason=TerminalReason.SUCCESS, clean_success=True, raw_arrival=True, final_rc=1.0, time_to_success=30.0),
            self._create_record(primary_reason=TerminalReason.CRASH_VEHICLE, raw_crash_veh=True),
            self._create_record(primary_reason=TerminalReason.CRASH_HUMAN, raw_crash_human=True),
            self._create_record(primary_reason=TerminalReason.OUT_OF_ROAD, raw_out=True),
            self._create_record(primary_reason=TerminalReason.TIMEOUT),
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
        self.assertEqual(agg.raw_crash_human_rate, 0.2)
        self.assertEqual(agg.raw_any_safety_event_rate, 0.6)

    def test_no_success_conditional_time_is_none(self):
        records = [
            self._create_record(primary_reason=TerminalReason.OUT_OF_ROAD),
            self._create_record(primary_reason=TerminalReason.TIMEOUT),
        ]
        agg = compute_aggregate_metrics(records)
        self.assertIsNone(agg.mean_time_to_clean_success_s)
        self.assertIsNone(agg.median_time_to_clean_success_s)

    def test_episode_record_validation_errors(self):
        # clean_success True but raw_arrival False
        with self.assertRaises(ValueError):
            EpisodeRecord(
                tier="Easy", sequence="SCS", scenario_seed=0, terminated=True, truncated=False,
                primary_reason=TerminalReason.SUCCESS, raw_arrival=False, clean_success=True,
                final_route_completion=1.0, max_route_completion=1.0,
                raw_crash_vehicle=False, raw_crash_object=False, raw_crash_building=False,
                raw_crash_human=False, raw_crash_sidewalk=False, raw_out_of_road=False,
                episode_steps=100, simulation_time_s=10.0, mean_speed_kmh=20.0, max_speed_kmh=25.0,
                episode_return=1.0
            )

        # primary_reason is UNDETERMINED in completed record
        with self.assertRaises(ValueError):
            EpisodeRecord(
                tier="Easy", sequence="SCS", scenario_seed=0, terminated=True, truncated=False,
                primary_reason=TerminalReason.UNDETERMINED, raw_arrival=False, clean_success=False,
                final_route_completion=0.5, max_route_completion=0.5,
                raw_crash_vehicle=False, raw_crash_object=False, raw_crash_building=False,
                raw_crash_human=False, raw_crash_sidewalk=False, raw_out_of_road=False,
                episode_steps=100, simulation_time_s=10.0, mean_speed_kmh=20.0, max_speed_kmh=25.0,
                episode_return=0.0
            )

        # Non-finite route completion
        with self.assertRaises(ValueError):
            EpisodeRecord(
                tier="Easy", sequence="SCS", scenario_seed=0, terminated=True, truncated=False,
                primary_reason=TerminalReason.OUT_OF_ROAD, raw_arrival=False, clean_success=False,
                final_route_completion=float("nan"), max_route_completion=1.0,
                raw_crash_vehicle=False, raw_crash_object=False, raw_crash_building=False,
                raw_crash_human=False, raw_crash_sidewalk=False, raw_out_of_road=True,
                episode_steps=100, simulation_time_s=10.0, mean_speed_kmh=20.0, max_speed_kmh=25.0,
                episode_return=-1.0
            )

    def test_empty_records_raises_error(self):
        with self.assertRaises(ValueError):
            compute_aggregate_metrics([])


if __name__ == "__main__":
    unittest.main()
