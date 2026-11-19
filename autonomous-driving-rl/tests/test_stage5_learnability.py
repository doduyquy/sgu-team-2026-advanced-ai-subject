"""S5.4 versioned observation treatment and TRAIN-only protocol checks."""
from __future__ import annotations

import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np

from src.training.stage5.pilot_observation_transform import (
    GROUPWISE_PHYSICAL_V1,
    GroupwisePhysicalObservation,
    WAYPOINT_XY50_V1,
    apply_observation_transform,
    transform_fingerprint,
)
from src.training.stage5.pilot_protocol import load_pilot_split
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.train_sampler import SamplerStrategy, TrainCaseSampler


class TestGroupwisePhysicalTransform(unittest.TestCase):
    def test_scales_only_declared_physical_fields_and_preserves_masks(self):
        obs = np.zeros(409, dtype=np.float32)
        obs[:259] = 0.25
        actors = obs[259:315].reshape(8, 7)
        actors[:] = [50.0, -25.0, 30.0, -15.0, np.pi, 5.0, 2.5]
        obs[315:325] = np.arange(10, dtype=np.float32) / 10
        waypoints = obs[325:385].reshape(20, 3)
        waypoints[:] = [100.0, -50.0, np.pi]
        obs[385:409] = np.arange(24, dtype=np.float32) / 24

        transformed = apply_observation_transform(obs, GROUPWISE_PHYSICAL_V1["transform_id"])
        np.testing.assert_array_equal(transformed[:259], obs[:259])
        np.testing.assert_allclose(transformed[259:315].reshape(8, 7)[0], [1, -.5, 1, -.5, 1, 1, .5])
        np.testing.assert_array_equal(transformed[315:325], obs[315:325])
        np.testing.assert_allclose(transformed[325:385].reshape(20, 3)[0], [2, -1, 1])
        np.testing.assert_array_equal(transformed[385:409], obs[385:409])
        self.assertEqual(transformed.dtype, np.float32)

    def test_zero_padding_is_preserved_and_batch_shape_supported(self):
        obs = np.zeros((3, 409), dtype=np.float32)
        out = apply_observation_transform(obs, GROUPWISE_PHYSICAL_V1["transform_id"])
        self.assertEqual(out.shape, obs.shape)
        self.assertTrue(np.array_equal(out, obs))

    def test_legacy_waypoint_transform_remains_compatible(self):
        obs = np.ones(409, dtype=np.float32)
        short_name = apply_observation_transform(obs, "waypoint_xy_div50m_v1")
        registered_name = apply_observation_transform(obs, WAYPOINT_XY50_V1["transform_id"])
        np.testing.assert_array_equal(short_name, registered_name)
        self.assertEqual(short_name[325], 1.0 / 50.0)
        self.assertEqual(short_name[327], 1.0)

    def test_transform_identity_is_versioned_and_fingerprinted(self):
        self.assertEqual(GROUPWISE_PHYSICAL_V1["observation_dim"], 409)
        self.assertEqual(GROUPWISE_PHYSICAL_V1["clipping"], "none")
        self.assertEqual(len(transform_fingerprint(GROUPWISE_PHYSICAL_V1)), 64)
        with self.assertRaisesRegex(ValueError, "Unsupported Stage 5 observation transform"):
            apply_observation_transform(np.zeros(409, dtype=np.float32), "unregistered")

    def test_gymnasium_wrapper_matches_pure_transform_and_space(self):
        class IdentityEnv(gym.Env):
            observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(409,), dtype=np.float32)
            action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(409, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(409, dtype=np.float32), 0.0, False, False, {}

        wrapper = GroupwisePhysicalObservation(IdentityEnv())
        obs, _ = wrapper.reset()
        self.assertEqual(obs.shape, (409,))
        self.assertTrue(wrapper.observation_space.contains(obs))
        sample = np.ones(409, dtype=np.float32)
        np.testing.assert_array_equal(wrapper.observation(sample),
                                      apply_observation_transform(sample, GROUPWISE_PHYSICAL_V1["transform_id"]))


class TestS54GeometryIsolation(unittest.TestCase):
    def test_eight_easy_train_ids_are_disjoint_from_six_easy_diagnostics(self):
        root = Path(__file__).resolve().parents[1]
        split = load_pilot_split(root / "results/stage5/pilot_split_v1.json")
        train_easy = [row for row in split["pilot_train"] if row["tier"] == "Easy"]
        selected = []
        quotas = {"SCCS": 3, "SCSS": 3, "SCS": 2}
        for family, count in quotas.items():
            candidates = sorted((row for row in train_easy if row["sequence"] == family),
                                key=lambda row: (row["geometry_generation_seed"], row["geometry_id"]))
            selected.extend(candidates[:count])
        diagnostic = [row for row in split["pilot_diagnostic"] if row["tier"] == "Easy"]
        self.assertEqual(len(selected), 8)
        self.assertEqual(len(diagnostic), 6)
        self.assertFalse({row["geometry_id"] for row in selected} & {row["geometry_id"] for row in diagnostic})
        sampler = TrainCaseSampler(101, SamplerStrategy.ROUND_ROBIN,
                                   allowed_geometry_ids=[row["geometry_id"] for row in selected],
                                   project_root=root)
        seen = [sampler.geometry_id_for_case(sampler.sample_case()) for _ in range(16)]
        self.assertEqual(set(seen), {row["geometry_id"] for row in selected})
        self.assertEqual(len(sampler._records), 8)
        foreign_case = TrainCaseSampler(101, project_root=root).get_case_for_geometry_id(
            diagnostic[0]["geometry_id"], environment_seed=8123
        )
        with self.assertRaisesRegex(ValueError, "PILOT SPLIT VIOLATION"):
            sampler.validate_train_case(foreign_case)
        env = Stage5TrainingEnv(sampler=sampler, project_root=root)
        try:
            with self.assertRaisesRegex(ValueError, "PILOT SPLIT VIOLATION"):
                env.reset(options={"case": foreign_case})
            self.assertIsNone(env._env)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
