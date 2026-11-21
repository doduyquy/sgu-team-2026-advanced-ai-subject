import tempfile
import unittest

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from scripts.run_stage5_p2h_terminal_replay import P2HInstrumentedSAC
from src.training.stage5.actor_activation_ablation import ActorOnlyLeakyReLUPolicy, activation_inventory


class TerminalEnv(gym.Env):
    observation_space = gym.spaces.Box(-1.0, 1.0, (409,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

    def __init__(self, kind="safety"):
        self.kind = kind
        self.count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); self.count = 0
        return np.zeros(409, dtype=np.float32), {}

    def step(self, action):
        self.count += 1
        info = {"action_steering": float(action[0]), "action_throttle_brake": float(action[1])}
        if self.count == 5:
            if self.kind == "timeout":
                return np.zeros(409, np.float32), 0.0, False, True, info
            reason = "SUCCESS" if self.kind == "clean" else "OUT_OF_ROAD"
            info.update({"primary_reason": reason,
                         "clean_success": self.kind == "clean",
                         "outcome": {"primary_reason": reason, "clean_success": self.kind == "clean"}})
            return np.zeros(409, np.float32), -1.0 if self.kind == "safety" else 1.0, True, False, info
        return np.zeros(409, np.float32), 0.0, False, False, info


class P2HTreatmentIntegrationTests(unittest.TestCase):
    def _model(self, kind):
        env = DummyVecEnv([lambda: TerminalEnv(kind)])
        model = P2HInstrumentedSAC(ActorOnlyLeakyReLUPolicy, env,
            policy_kwargs={"net_arch": [32, 32]}, learning_starts=1, buffer_size=128,
            batch_size=8, train_freq=1, gradient_steps=1, gamma=.99, tau=.005,
            ent_coef="auto", seed=101, device="cpu", verbose=0,
            p2h_sampler_config={"mixture": .2, "priority_alpha": .6,
                "priority_beta_start": .4, "priority_beta_end": 1.0,
                "priority_beta_steps": 9, "td_error_cap": 10.0,
                "epsilon": .001, "seed": 101})
        return model

    def test_real_online_terminal_and_timeout_labels_and_critic_only_sampling(self):
        model = self._model("safety")
        model.learn(40)
        kinds = model.terminal_kinds[:model.replay_buffer.size()]
        self.assertGreater(int(np.sum(kinds == 1)), 0)
        self.assertGreater(model.terminal_sampler.terminal_draws, 0)
        self.assertGreater(model._p2_gradient_sq["actor"], 0.0)
        self.assertGreater(model._p2_gradient_sq["critic"], 0.0)
        self.assertEqual(activation_inventory(model.policy)["actor_id"], "actor_leaky_relu_001_v1")
        self.assertTrue(activation_inventory(model.policy)["critic_is_relu_only"])
        with tempfile.TemporaryDirectory() as directory:
            path = directory + "/model"
            model.save(path)
            restored = SAC.load(path, device="cpu")
            self.assertEqual(activation_inventory(restored.policy)["actor_id"], "actor_leaky_relu_001_v1")
        model.env.close()

    def test_timeout_is_recorded_but_never_terminal_priority_candidate(self):
        model = self._model("timeout")
        model.learn(15)
        size = model.replay_buffer.size()
        kinds = model.terminal_kinds[:size]
        self.assertGreater(int(np.sum(kinds == 3)), 0)
        true_terminal = model.replay_buffer.dones[:size, 0].astype(bool)
        true_terminal &= ~model.replay_buffer.timeouts[:size, 0].astype(bool)
        self.assertFalse(bool(np.any(true_terminal & (kinds == 3))))
        sampled = model.replay_buffer._get_samples(np.arange(size, dtype=np.int64), env=None)
        timeout_indices = np.flatnonzero(kinds == 3)
        self.assertGreater(len(timeout_indices), 0)
        self.assertTrue(bool(np.all(sampled.dones[timeout_indices].cpu().numpy() == 0.0)))
        model.env.close()

    def test_clean_success_terminal_is_distinct_from_safety_terminal(self):
        model = self._model("clean")
        model.learn(15)
        kinds = model.terminal_kinds[:model.replay_buffer.size()]
        self.assertGreater(int(np.sum(kinds == 2)), 0)
        self.assertEqual(int(np.sum(kinds == 1)), 0)
        model.env.close()


if __name__ == "__main__":
    unittest.main()
