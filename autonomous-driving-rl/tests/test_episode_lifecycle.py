"""
Unit tests for Platform V1 episode lifecycle contracts and pure helpers.
Does not require Panda3D or MetaDrive simulator initialization.
"""

import unittest

from src.platform import (
    EpisodeOutcome,
    EpisodeSpecV1,
    TerminalReason,
    classify_episode_outcome,
    compute_route_aware_horizon,
)


class TestEpisodeLifecycle(unittest.TestCase):
    def test_route_aware_horizon_calculation(self):
        # Base: 18 km/h = 5.0 m/s, safety margin 1.5, 10 Hz -> factor = 3.0 steps/m
        h_short = compute_route_aware_horizon(200.0, min_horizon_steps=1000)
        self.assertEqual(h_short, 1000)  # Clamped to min

        # Easy Primary (349.6 m): 349.6 * 3.0 = 1048.8 -> 1049 steps
        h_easy = compute_route_aware_horizon(349.6)
        self.assertEqual(h_easy, 1049)

        # Medium Primary (523.8 m): 523.8 * 3.0 = 1571.4 -> 1572 steps
        h_med = compute_route_aware_horizon(523.8)
        self.assertEqual(h_med, 1572)

        # Hard Primary (643.0 m): 643.0 * 3.0 = 1929.0 -> 1929 steps
        h_hard = compute_route_aware_horizon(643.0)
        self.assertEqual(h_hard, 1929)

        # Extreme Primary (938.6 m): 938.6 * 3.0 = 2815.8 -> 2816 steps
        h_ext = compute_route_aware_horizon(938.6)
        self.assertEqual(h_ext, 2816)

        # Extreme route clamping to max cap (4000)
        h_huge = compute_route_aware_horizon(2000.0, max_horizon_steps=4000)
        self.assertEqual(h_huge, 4000)

        # Zero length returns min floor
        h_zero = compute_route_aware_horizon(0.0)
        self.assertEqual(h_zero, 1000)

    def test_route_aware_horizon_validation_errors(self):
        # Negative route length
        with self.assertRaises(ValueError):
            compute_route_aware_horizon(-10.0)
        # Non-positive speed
        with self.assertRaises(ValueError):
            compute_route_aware_horizon(500.0, reference_floor_speed_kmh=0.0)
        # Non-positive safety margin
        with self.assertRaises(ValueError):
            compute_route_aware_horizon(500.0, safety_margin=0.0)
        # Non-positive frequency
        with self.assertRaises(ValueError):
            compute_route_aware_horizon(500.0, control_frequency_hz=0)
        # Max horizon < min horizon
        with self.assertRaises(ValueError):
            compute_route_aware_horizon(500.0, min_horizon_steps=2000, max_horizon_steps=1000)

    def test_clean_success_classification(self):
        raw = {"arrive_dest": True, "out_of_road": False, "crash_vehicle": False}
        outcome = classify_episode_outcome(raw, terminated=True, truncated=False)
        self.assertTrue(outcome.terminated)
        self.assertFalse(outcome.truncated)
        self.assertEqual(outcome.primary_reason, TerminalReason.SUCCESS)
        self.assertTrue(outcome.clean_success)
        self.assertEqual(outcome.raw_flags["arrive_dest"], True)

    def test_safety_first_precedence_simultaneous_success_and_crash(self):
        # Vehicle crosses destination line while simultaneously colliding with traffic
        raw = {
            "arrive_dest": True,
            "crash_vehicle": True,
            "out_of_road": False,
        }
        outcome = classify_episode_outcome(raw, terminated=True, truncated=False)
        self.assertTrue(outcome.terminated)
        self.assertEqual(outcome.primary_reason, TerminalReason.CRASH_VEHICLE)
        self.assertFalse(outcome.clean_success)
        # Verify raw flags are strictly preserved
        self.assertTrue(outcome.raw_flags["arrive_dest"])
        self.assertTrue(outcome.raw_flags["crash_vehicle"])

    def test_safety_first_precedence_simultaneous_success_and_out_of_road(self):
        raw = {
            "arrive_dest": True,
            "out_of_road": True,
            "crash_sidewalk": True,
        }
        outcome = classify_episode_outcome(raw, terminated=True, truncated=False)
        self.assertTrue(outcome.terminated)
        self.assertEqual(outcome.primary_reason, TerminalReason.CRASH_SIDEWALK)
        self.assertFalse(outcome.clean_success)

    def test_simultaneous_human_and_vehicle_crash(self):
        # Crash human takes highest precedence
        raw = {
            "crash_human": True,
            "crash_vehicle": True,
            "arrive_dest": False,
        }
        outcome = classify_episode_outcome(raw, terminated=True, truncated=False)
        self.assertEqual(outcome.primary_reason, TerminalReason.CRASH_HUMAN)
        self.assertFalse(outcome.clean_success)

    def test_timeout_and_out_of_road(self):
        # Out-of-road takes precedence over timeout
        raw = {
            "out_of_road": True,
            "max_step": True,
        }
        outcome = classify_episode_outcome(raw, terminated=True, truncated=True)
        self.assertEqual(outcome.primary_reason, TerminalReason.OUT_OF_ROAD)
        self.assertTrue(outcome.terminated)

    def test_timeout_alone(self):
        raw = {
            "max_step": True,
            "arrive_dest": False,
            "out_of_road": False,
        }
        outcome = classify_episode_outcome(raw, terminated=False, truncated=True)
        self.assertFalse(outcome.terminated)
        self.assertTrue(outcome.truncated)
        self.assertEqual(outcome.primary_reason, TerminalReason.TIMEOUT)
        self.assertFalse(outcome.clean_success)

    def test_ongoing_undetermined_step(self):
        raw = {
            "arrive_dest": False,
            "out_of_road": False,
            "crash_vehicle": False,
            "max_step": False,
        }
        outcome = classify_episode_outcome(raw, terminated=False, truncated=False)
        self.assertFalse(outcome.terminated)
        self.assertFalse(outcome.truncated)
        self.assertFalse(outcome.is_done)
        self.assertEqual(outcome.primary_reason, TerminalReason.UNDETERMINED)

    def test_episode_spec_serialization(self):
        spec = EpisodeSpecV1()
        data = spec.to_dict()
        self.assertEqual(data["spec_version"], "1.0.0")
        self.assertEqual(data["traffic_mode"], "trigger")
        self.assertEqual(data["reference_floor_speed_kmh"], 18.0)
        self.assertEqual(spec.compute_horizon(500.0), 1500)


if __name__ == "__main__":
    unittest.main()
