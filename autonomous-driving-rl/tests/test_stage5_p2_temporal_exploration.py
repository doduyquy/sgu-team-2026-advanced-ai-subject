from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback

from src.training.stage5.temporal_exploration import (
    OrnsteinUhlenbeckActionNoise,
    TemporalExplorationSAC,
)


class _ActionEchoEnv(gym.Env):
    observation_space = spaces.Box(-10.0, 10.0, shape=(3,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def __init__(self, truncate: bool = False) -> None:
        self.step_index = 0
        self.truncate = truncate

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_index = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self.step_index += 1
        action = np.asarray(action, dtype=np.float32)
        info = {
            "action_steering": float(action[0]),
            "action_throttle_brake": float(action[1]),
            "geometry_id": "synthetic_train_case",
            "environment_seed": 17,
            "speed_km_h": float(max(action[1], 0.0) * 10.0),
            "route_completion": self.step_index / 100.0,
            "delta_route": 0.01,
            "reward_breakdown": {"raw_total_reward": 0.01},
        }
        done = self.step_index >= 5
        terminated = done and not self.truncate
        truncated = done and self.truncate
        return np.full(3, self.step_index, dtype=np.float32), 0.01, terminated, truncated, info


class _EpisodeCountCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.completed = 0

    def _on_step(self) -> bool:
        self.completed += int(np.asarray(self.locals["dones"], dtype=bool).sum())
        return True


class TestStage5P2TemporalExploration(unittest.TestCase):
    def test_ou_stream_is_reproducible_stationary_and_temporally_correlated(self):
        first = OrnsteinUhlenbeckActionNoise(rho=0.85, stationary_std=0.15, seed=101)
        second = OrnsteinUhlenbeckActionNoise(rho=0.85, stationary_std=0.15, seed=101)
        a = np.stack([first.sample() for _ in range(20_000)])
        b = np.stack([second.sample() for _ in range(20_000)])
        self.assertTrue(np.array_equal(a, b))
        lag_one = np.corrcoef(a[:-1, 0], a[1:, 0])[0, 1]
        self.assertAlmostEqual(float(lag_one), 0.85, delta=0.03)
        first.reset_episode()
        self.assertTrue(np.array_equal(first.state, np.zeros(2, dtype=np.float32)))

    def test_sb3_environment_and_replay_receive_identical_actions_and_reset_ou(self):
        env = _ActionEchoEnv()
        model = TemporalExplorationSAC(
            "MlpPolicy", env, exploration_mode="ou_ar1_v1", ou_rho=0.85,
            ou_stationary_std=0.15, ou_seed=901, learning_rate=3e-4,
            buffer_size=256, learning_starts=10, batch_size=8, tau=0.005,
            gamma=0.99, train_freq=1, gradient_steps=1, seed=101,
            policy_kwargs={"net_arch": [16, 16]}, device="cpu", verbose=0,
        )
        rows = []
        model.action_audit_sink = rows.append
        callback = _EpisodeCountCallback()
        model.learn(total_timesteps=60, callback=callback, log_interval=1000)
        self.assertEqual(len(rows), 60)
        self.assertEqual(model.action_mismatch_count, 0)
        self.assertEqual(model.ou_episode_resets, callback.completed)
        self.assertGreater(model.action_noise_count, 0)
        for row in rows:
            np.testing.assert_allclose(row["executed_action"], row["replay_action"], rtol=0.0, atol=1e-6)
            expected = np.clip(np.asarray(row["policy_sample"]) + np.asarray(row["ou_noise"]), -1.0, 1.0)
            np.testing.assert_allclose(row["executed_action"], expected, atol=1e-7)
            self.assertGreaterEqual(min(row["executed_action"]), -1.0)
            self.assertLessEqual(max(row["executed_action"]), 1.0)

    def test_ou_state_resets_on_truncation_as_well_as_termination(self):
        env = _ActionEchoEnv(truncate=True)
        model = TemporalExplorationSAC(
            "MlpPolicy", env, exploration_mode="ou_ar1_v1", ou_rho=0.85,
            ou_stationary_std=0.15, ou_seed=902, learning_rate=3e-4,
            buffer_size=128, learning_starts=8, batch_size=8, seed=102,
            policy_kwargs={"net_arch": [16, 16]}, device="cpu", verbose=0,
        )
        callback = _EpisodeCountCallback()
        model.learn(total_timesteps=30, callback=callback, log_interval=1000)
        self.assertEqual(callback.completed, 6)
        self.assertEqual(model.ou_episode_resets, callback.completed)
        self.assertEqual(model.action_mismatch_count, 0)

    def test_standard_control_does_not_apply_external_noise(self):
        env = _ActionEchoEnv()
        model = TemporalExplorationSAC(
            "MlpPolicy", env, exploration_mode="standard", learning_rate=3e-4,
            buffer_size=64, learning_starts=64, batch_size=8, seed=101,
            policy_kwargs={"net_arch": [16, 16]}, device="cpu", verbose=0,
        )
        actions, buffer_actions = model._sample_action(learning_starts=64, n_envs=1)
        self.assertEqual(model.action_noise_count, 0)
        np.testing.assert_array_equal(actions, buffer_actions)


if __name__ == "__main__":
    unittest.main()
