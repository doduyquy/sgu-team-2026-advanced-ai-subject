from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from scripts.analyze_stage5_tau_ab import _summary
from scripts.evaluate_stage5_pilot import _episode_record
from scripts.run_stage5_tau_ab import _protocol


ROOT = Path(__file__).resolve().parents[1]


class TestStage5TauAB(unittest.TestCase):
    def test_registered_protocol_is_fixed_train_only_and_tau_is_only_arm_difference(self):
        protocol = _protocol("unit-test-run")
        self.assertEqual(protocol["arms"], {"A": {"tau": 0.005}, "B": {"tau": 0.001}})
        self.assertEqual(protocol["only_registered_arm_difference"], "sac_hyperparameters.tau")
        self.assertEqual(protocol["training_steps"], 10_000)
        self.assertEqual(protocol["checkpoint_steps"], [0, 5_000, 10_000])
        self.assertEqual(len(protocol["diagnostic_schedule"]), 8)
        self.assertEqual({tier: sum(row["tier"] == tier for row in protocol["diagnostic_schedule"])
                          for tier in ("Easy", "Medium", "Hard", "Extreme")},
                         {tier: 2 for tier in ("Easy", "Medium", "Hard", "Extreme")})
        self.assertTrue(all(len(row["geometry_sha256"]) == 64 for row in protocol["diagnostic_schedule"]))
        self.assertFalse(protocol["holdouts_accessed"])

    def test_bounded_probe_is_marked_censored_and_progress_is_reset_relative(self):
        env = SimpleNamespace(
            current_case=SimpleNamespace(case_id="case", tier="Easy", sequence="SCCS"),
            last_outcome=None, _last_route_completion=0.17, max_route_completion=0.19,
        )
        row = _episode_record(
            env,
            {"geometry_id": "geom", "geometry_sha256": "a" * 64, "tier": "Easy",
             "sequence": "SCCS", "geometry_generation_seed": 1, "environment_seed": 2},
            "b" * 64, 0, [0.0], [[0.0, 0.1]], [0.2], [0.0], 0.1, True, 0.99, 0.12,
        )
        self.assertEqual(row["outcome"], "PROBE_CENSORED")
        self.assertFalse(row["timeout"])
        self.assertAlmostEqual(row["route_progress_delta"], 0.05)
        self.assertAlmostEqual(row["max_route_progress_delta"], 0.07)

    def test_summary_keeps_censored_probes_out_of_official_timeout_count(self):
        summary = _summary([
            {"outcome": "PROBE_CENSORED", "timeout": False, "route_progress_delta": 0.02,
             "max_route_progress_delta": 0.03, "mean_speed_km_h": 4.0,
             "stationary_fraction": 0.2, "mean_throttle": 0.3, "mean_brake": 0.0,
             "mean_steering": 0.1, "clean_success": False, "safety_failure": False},
        ])
        self.assertEqual(summary["probe_censored"], 1)
        self.assertEqual(summary["official_timeout_count"], 0)
        self.assertAlmostEqual(summary["mean_route_progress_delta"], 0.02)


if __name__ == "__main__":
    unittest.main()
