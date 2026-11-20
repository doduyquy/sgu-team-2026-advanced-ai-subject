from __future__ import annotations

import math
import unittest

import gymnasium as gym
import numpy as np

from src.platform.reward import RewardSpecV1
from src.training.stage5.reward_experiments import (
    RewardScale,
    Stage5ExperimentalRewardWrapper,
    compute_s5_5_reward_variants,
)
from src.training.stage5.pilot_observation_transform import (
    GROUPWISE_PHYSICAL_V1,
    GroupwisePhysicalObservation,
    apply_observation_transform,
)
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.train_sampler import SamplerStrategy, TrainCaseSampler


class _FakeNavigation:
    total_length = 500.0


class _FakeAgent:
    navigation = _FakeNavigation()


class _FakeMetaDrive:
    agent = _FakeAgent()


class _FakeTrainingEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

    def __init__(self):
        self.reward_spec = RewardSpecV1()
        self._env = _FakeMetaDrive()
        self._current_case = type("Case", (), {"horizon_steps": 1000})()
        self.last_action = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(2, dtype=np.float32), {"initial_route_completion": 0.02}

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        delta_p = -0.002
        terminal = -1.0
        reward = delta_p - 0.25 / 1000 + terminal
        return (
            np.ones(2, dtype=np.float32), reward, True, False,
            {
                "delta_route": delta_p,
                "horizon_steps": 1000,
                "action_steering": float(action[0]),
                "action_throttle_brake": float(action[1]),
                "reward_breakdown": {
                    "progress_reward": delta_p,
                    "time_cost": 0.25 / 1000,
                    "terminal_reward": terminal,
                    "raw_total_reward": reward,
                    "total_reward": reward,
                    "clipping_applied": False,
                },
            },
        )


class _FakeObservationEnv(gym.Env):
    observation_space = gym.spaces.Box(-100.0, 100.0, shape=(409,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def observation(self, observation):
        return observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(409, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(409, dtype=np.float32), 0.0, False, False, {}


class TestStage5MeterReward(unittest.TestCase):
    def test_k_is_exact_median_over_ten(self):
        scale = RewardScale.from_route_lengths([200.0, 400.0, 600.0, 800.0])
        self.assertEqual(scale.median_route_length_m, 500.0)
        self.assertEqual(scale.scale_k, 50.0)

    def test_scale_and_meter_rewards_match_at_ten_k(self):
        k = 42.46479467561925
        values = compute_s5_5_reward_variants(
            delta_route=0.012,
            horizon_steps=1000,
            terminal_reward=0.0,
            effective_route_length_m=10.0 * k,
            scale_k=k,
        )
        self.assertAlmostEqual(values["reward_scale"], values["reward_meter"], places=14)

    def test_signed_reverse_progress_is_not_clipped(self):
        values = compute_s5_5_reward_variants(
            delta_route=-0.01,
            horizon_steps=1000,
            terminal_reward=0.0,
            effective_route_length_m=400.0,
            scale_k=42.0,
        )
        self.assertEqual(values["delta_s_m"], -4.0)
        self.assertLess(values["reward_v1"], 0.0)
        self.assertLess(values["reward_scale"], 0.0)
        self.assertLess(values["reward_meter"], 0.0)

    def test_terminal_reward_contributes_exactly_once(self):
        values = compute_s5_5_reward_variants(
            delta_route=0.0,
            horizon_steps=1000,
            terminal_reward=-1.0,
            effective_route_length_m=400.0,
            scale_k=42.0,
        )
        self.assertAlmostEqual(values["reward_v1"], -1.00025)
        self.assertAlmostEqual(values["reward_scale"], 42.0 * -1.00025)
        self.assertAlmostEqual(values["reward_meter"], 42.0 * -1.00025)

    def test_wrapper_returns_selected_reward_and_preserves_action(self):
        raw = _FakeTrainingEnv()
        wrapper = Stage5ExperimentalRewardWrapper(raw, "METER", RewardScale(420.0, 42.0))
        wrapper.reset(seed=3)
        action = np.asarray([0.25, 0.75], dtype=np.float32)
        _, reward, terminated, truncated, info = wrapper.step(action)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(np.array_equal(raw.last_action, action))
        self.assertAlmostEqual(reward, info["s5_5_rewards"]["reward_meter"])
        self.assertAlmostEqual(info["s5_5_rewards"]["reward_v1"], info["reward_breakdown"]["total_reward"])
        self.assertEqual(info["s5_5_reward_variant"], "METER")

    def test_wrapper_rejects_clipped_platform_reward(self):
        raw = _FakeTrainingEnv()
        raw.reward_spec = RewardSpecV1(clip_min=-1.0, clip_max=1.0)
        with self.assertRaisesRegex(ValueError, "unclipped"):
            Stage5ExperimentalRewardWrapper(raw, "V1", RewardScale(420.0, 42.0))

    def test_rejects_invalid_route_length_and_horizon(self):
        with self.assertRaises(ValueError):
            compute_s5_5_reward_variants(
                delta_route=0.0, horizon_steps=0, terminal_reward=0.0,
                effective_route_length_m=400.0, scale_k=42.0,
            )
        with self.assertRaises(ValueError):
            compute_s5_5_reward_variants(
                delta_route=0.0, horizon_steps=100, terminal_reward=0.0,
                effective_route_length_m=0.0, scale_k=42.0,
            )

    def test_meter_progress_is_signed_and_continuous_across_block_boundary(self):
        # A boundary is a route-geometry event, not a reset of route completion.
        before = compute_s5_5_reward_variants(
            delta_route=1.01 - 0.98, horizon_steps=1000, terminal_reward=0.0,
            effective_route_length_m=400.0, scale_k=42.0,
        )
        after = compute_s5_5_reward_variants(
            delta_route=1.00 - 1.01, horizon_steps=1000, terminal_reward=0.0,
            effective_route_length_m=400.0, scale_k=42.0,
        )
        self.assertAlmostEqual(before["delta_s_m"], 12.0)
        self.assertAlmostEqual(after["delta_s_m"], -4.0)

    def test_groupwise_training_wrapper_matches_inference_transform_and_preserves_masks(self):
        raw = np.zeros(409, dtype=np.float32)
        raw[259:315] = np.arange(56, dtype=np.float32)
        raw[315:325] = np.asarray([1, 0, 1, 0, 1, 0, 1, 0, 0.5, 0.25], dtype=np.float32)
        raw[325:385] = np.arange(60, dtype=np.float32)
        wrapper = GroupwisePhysicalObservation(_FakeObservationEnv())
        training = wrapper.observation(raw)
        inference = apply_observation_transform(raw, GROUPWISE_PHYSICAL_V1["transform_id"])
        self.assertTrue(np.array_equal(training, inference))
        self.assertTrue(np.array_equal(training[315:325], raw[315:325]))
        self.assertTrue(np.array_equal(training[:259], raw[:259]))

    def test_sampler_rejects_diagnostic_case_before_simulator_creation(self):
        diagnostic_id = "geom_easy_SCCS_seed4"
        diagnostic_sampler = TrainCaseSampler(
            training_root_seed=101,
            strategy=SamplerStrategy.FIXED,
            allowed_geometry_ids=[diagnostic_id],
        )
        diagnostic_case = diagnostic_sampler.get_case_for_geometry_id(diagnostic_id, environment_seed=925389)
        training_sampler = TrainCaseSampler(
            training_root_seed=101,
            strategy=SamplerStrategy.ROUND_ROBIN,
            allowed_geometry_ids=["geom_easy_SCCS_seed0"],
        )
        env = Stage5TrainingEnv(sampler=training_sampler)
        try:
            with self.assertRaisesRegex(ValueError, "outside the allowed TRAIN pool"):
                env.reset(options={"case": diagnostic_case})
            self.assertIsNone(env._env)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
