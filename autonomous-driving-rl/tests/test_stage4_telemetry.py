"""Chặn hồi quy vận tốc giả bằng 0 và kiểm tra quy chiếu dấu lái."""

import math
from types import SimpleNamespace
import unittest

from scripts.audit_stage4_telemetry import signed_turn, speed_summary
from src.launcher.telemetry import read_vehicle_speed_kmh
from src.platform.action_adapter import ContinuousBox2Adapter, Discrete9Adapter, Discrete25Adapter


class TestTelemetryEvidence(unittest.TestCase):
    def test_canonical_speed_property_overrides_misleading_alias(self):
        self.assertEqual(read_vehicle_speed_kmh(SimpleNamespace(speed_km_h=36.0, speed_kmh=0.0)), 36.0)
        self.assertEqual(read_vehicle_speed_kmh(SimpleNamespace(speed_km_h=0.0)), 0.0)

    def test_missing_or_invalid_sensor_cannot_be_logged_as_zero(self):
        with self.assertRaises(AttributeError):
            read_vehicle_speed_kmh(SimpleNamespace(speed_kmh=36.0))
        for value in (math.nan, math.inf, -1.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                read_vehicle_speed_kmh(SimpleNamespace(speed_km_h=value))

    def test_heading_wrap_and_rotated_ego_left_axis(self):
        start = {"heading_rad": math.pi - 0.01, "position_m": [0, 0]}
        end = {"heading_rad": -math.pi + 0.02, "position_m": [-1, -0.2]}
        result = signed_turn(start, end)
        self.assertAlmostEqual(result["heading_delta_rad"], 0.03)
        self.assertGreater(result["initial_ego_left_displacement_m"], 0)
        self.assertGreater(result["initial_ego_forward_displacement_m"], 0)

    def test_speed_summary_exposes_false_zero_and_unit_error(self):
        rows = [{"post": {"reference_speed_kmh": 36.0, "speed_km_h": 36.0,
                          "legacy_speed_kmh_fallback": 0.0}}]
        result = speed_summary(rows, 1e-5)
        self.assertEqual(result["legacy_false_zero_steps"], 1)
        self.assertTrue(result["speed_matches_body_velocity"])
        rows[0]["post"]["speed_km_h"] = 10.0
        self.assertFalse(speed_summary(rows, 1e-5)["speed_matches_body_velocity"])

    def test_numeric_adapter_mappings_keep_their_sign(self):
        self.assertIn("-1=right, +1=left", ContinuousBox2Adapter().action_space_description["semantics"][0])
        for sign in (-1, 0, 1):
            self.assertEqual(ContinuousBox2Adapter().to_canonical([sign, .2]).steering, sign)
        self.assertEqual(Discrete9Adapter().to_canonical(0).steering, -0.6)
        self.assertEqual(Discrete9Adapter().to_canonical(2).steering, 0.6)
        self.assertEqual(Discrete25Adapter().to_canonical(0).steering, -1)
        self.assertEqual(Discrete25Adapter().to_canonical(4).steering, 1)


if __name__ == "__main__":
    unittest.main()
