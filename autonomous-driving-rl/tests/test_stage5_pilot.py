"""S5.3A internal split, sampler isolation, and evaluation read-only tests."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.platform import EpisodeOutcome, TerminalReason
from src.training.stage5.pilot_protocol import build_pilot_split, load_pilot_split, save_pilot_split
from src.training.stage5.train_sampler import SamplerStrategy, TrainCaseSampler
from scripts.evaluate_stage5_pilot import evaluate_checkpoint, summarize
from src.training.stage5.env_adapter import Stage5TrainingEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestPilotProtocol(unittest.TestCase):
    def test_split_is_deterministic_train_only_and_balanced(self):
        first = build_pilot_split(PROJECT_ROOT, split_seed=5301)
        second = build_pilot_split(PROJECT_ROOT, split_seed=5301)
        self.assertEqual(first, second)
        self.assertEqual(len(first["pilot_train"]), 156)
        self.assertEqual(len(first["pilot_diagnostic"]), 24)
        train_ids = {r["geometry_id"] for r in first["pilot_train"]}
        diag_ids = {r["geometry_id"] for r in first["pilot_diagnostic"]}
        self.assertFalse(train_ids & diag_ids)
        self.assertEqual(len(train_ids | diag_ids), 180)
        for tier in {r["tier"] for r in first["pilot_train"]}:
            self.assertEqual(sum(r["tier"] == tier for r in first["pilot_train"]), 39)
            self.assertEqual(sum(r["tier"] == tier for r in first["pilot_diagnostic"]), 6)
        self.assertTrue(all(len(r["geometry_sha256"]) == 64 for r in first["pilot_train"] + first["pilot_diagnostic"]))

    def test_sampler_rejects_diagnostic_case_in_pool_and_reset_path(self):
        split = build_pilot_split(PROJECT_ROOT)
        allowed = [r["geometry_id"] for r in split["pilot_train"]]
        blocked = split["pilot_diagnostic"][0]
        sampler = TrainCaseSampler(101, SamplerStrategy.ROUND_ROBIN,
                                   allowed_geometry_ids=allowed, project_root=PROJECT_ROOT)
        self.assertEqual(sampler.total_eligible_geometries, 156)
        unrestricted = TrainCaseSampler(101, SamplerStrategy.ROUND_ROBIN, project_root=PROJECT_ROOT)
        forged_explicit_case = unrestricted.get_case_for_geometry_id(blocked["geometry_id"], 71)
        with self.assertRaisesRegex(ValueError, "PILOT SPLIT VIOLATION"):
            sampler.validate_train_case(forged_explicit_case)
        with self.assertRaisesRegex(ValueError, "outside the configured"):
            sampler.get_case_for_geometry_id(blocked["geometry_id"], 71)
        observed = {sampler.sample_case().geometry_sha256 for _ in range(312)}
        self.assertLessEqual(observed, {r["geometry_sha256"] for r in split["pilot_train"]})

    def test_training_environment_reset_options_enforces_pilot_pool(self):
        split = build_pilot_split(PROJECT_ROOT)
        train = split["pilot_train"][0]
        diagnostic = split["pilot_diagnostic"][0]
        sampler = TrainCaseSampler(101, SamplerStrategy.ROUND_ROBIN,
                                   allowed_geometry_ids=[r["geometry_id"] for r in split["pilot_train"]],
                                   project_root=PROJECT_ROOT)
        unrestricted = TrainCaseSampler(101, SamplerStrategy.ROUND_ROBIN, project_root=PROJECT_ROOT)
        env = Stage5TrainingEnv(sampler=sampler, project_root=PROJECT_ROOT)
        try:
            blocked_case = unrestricted.get_case_for_geometry_id(diagnostic["geometry_id"], 53)
            with self.assertRaisesRegex(ValueError, "PILOT SPLIT VIOLATION"):
                env.reset(options={"case": blocked_case})
            self.assertIsNone(env._env)

            # A real allowed case passes the Gymnasium reset path and initializes MetaDrive.
            allowed_case = sampler.get_case_for_geometry_id(train["geometry_id"], 54)
            observation, info = env.reset(options={"case": allowed_case})
            self.assertTrue(env.observation_space.contains(observation))
            self.assertEqual(info["geometry_id"], train["geometry_id"])
            self.assertEqual(env.current_case.geometry_sha256, train["geometry_sha256"])
        finally:
            env.close()

    def test_interrupted_probe_is_not_counted_as_timeout(self):
        summary = summarize([{"interrupted": True, "timeout": False, "final_route_completion": 0.2}])
        self.assertEqual(summary["episodes_completed"], 0)
        self.assertEqual(summary["episodes_interrupted"], 1)
        self.assertNotIn("timeout_rate", summary)

    def test_evaluator_does_not_update_model_or_sampler_and_writes_records(self):
        class FakePolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(1, 2)

        class FakeModel:
            def __init__(self):
                self.policy = FakePolicy()
                self._n_updates = 0
                self.replay_buffer = None

            def predict(self, obs, deterministic=True):
                return np.array([0.0, 0.2], dtype=np.float32), None

        class FakeSampler:
            def __init__(self, **kwargs):
                self.sampled_count = 0
                self.allowed_geometry_ids = kwargs["allowed_geometry_ids"]

            def get_case_for_geometry_id(self, geometry_id, environment_seed):
                return type("Case", (), {"geometry_id": geometry_id, "environment_seed": environment_seed})()

        class FakeEnv:
            def __init__(self, sampler, encoder, project_root):
                self.sampler = sampler
                self.observation_space = type("Space", (), {"contains": lambda _, obs: True})()
                self.current_case = None
                self.last_outcome = None
                self.max_route_completion = 0.4
                self._last_route_completion = 0.4

            def reset(self, options):
                self.current_case = options["case"]
                return np.zeros(409, dtype=np.float32), {}

            def step(self, action):
                self.last_outcome = EpisodeOutcome(True, False, TerminalReason.OUT_OF_ROAD, False,
                                                   {"out_of_road": True})
                return np.zeros(409, dtype=np.float32), 0.25, True, False, {"speed_km_h": 4.0}

            def close(self):
                self.closed = True

        split_path = Path(tempfile.mkdtemp()) / "split.json"
        split_path.write_text(json.dumps(build_pilot_split(PROJECT_ROOT)), encoding="utf-8")
        output = split_path.parent / "records"
        model = FakeModel()
        metadata = {"model_sha256": "a" * 64, "total_timesteps": 0}
        with patch("scripts.evaluate_stage5_pilot.load_sac_checkpoint", return_value=(model, metadata)), \
             patch("scripts.evaluate_stage5_pilot.TrainCaseSampler", FakeSampler), \
             patch("scripts.evaluate_stage5_pilot.Stage5TrainingEnv", FakeEnv):
            result = evaluate_checkpoint(Path("checkpoint"), split_path, output, max_cases=1)
        self.assertTrue(result["evaluation_state_unchanged"])
        self.assertEqual(result["training_sampler_sampled_count"], 0)
        row = json.loads((output / "step_00000_episodes.jsonl").read_text().splitlines()[0])
        self.assertEqual(row["outcome"], "OUT_OF_ROAD")
        self.assertTrue(row["safety_failure"])
        self.assertEqual(row["episode_steps"], 1)


if __name__ == "__main__":
    unittest.main()
