import numpy as np
import torch
import gymnasium as gym
import unittest
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.sac.policies import SACPolicy

from src.training.stage5.actor_activation_ablation import (
    ACTOR_LEAKY_RELU_001_V1, ACTOR_RELU_V1, ActorOnlyLeakyReLUPolicy,
    activation_inventory, actor_activation_id, tensor_state_fingerprint,
)
from src.training.stage5.checkpoint import load_sac_checkpoint, save_sac_checkpoint
from src.training.stage5.observation_encoder import Stage5ObservationEncoder


class TinyContinuousEnv(gym.Env):
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(409,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(409, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(409, dtype=np.float32), 0.0, False, False, {}


def _model(policy_class, seed):
    return SAC(
        policy_class, DummyVecEnv([TinyContinuousEnv]), seed=seed, device="cpu",
        policy_kwargs={"net_arch": [256, 256], "activation_fn": torch.nn.ReLU},
        learning_starts=2, buffer_size=16, batch_size=2, verbose=0,
    )


class ActorActivationAblationTests(unittest.TestCase):
    def test_actor_only_policy_changes_actor_and_preserves_relu_critics_and_shapes(self):
        import tempfile
        control = _model(SACPolicy, 101)
        treatment = _model(ActorOnlyLeakyReLUPolicy, 202)
        treatment.policy.load_state_dict(control.policy.state_dict(), strict=True)
        self.assertEqual(actor_activation_id(control.policy), ACTOR_RELU_V1)
        self.assertEqual(actor_activation_id(treatment.policy), ACTOR_LEAKY_RELU_001_V1)
        self.assertTrue(activation_inventory(treatment.policy)["critic_is_relu_only"])
        self.assertEqual(activation_inventory(treatment.policy)["critic_target_module_types"], ["ReLU"])
        self.assertEqual(
            [(k, tuple(v.shape)) for k, v in control.policy.state_dict().items()],
            [(k, tuple(v.shape)) for k, v in treatment.policy.state_dict().items()],
        )
        self.assertEqual(tensor_state_fingerprint(control.policy), tensor_state_fingerprint(treatment.policy))

        with tempfile.TemporaryDirectory() as directory:
            control_path, treatment_path = f"{directory}/control", f"{directory}/treatment"
            control.save(control_path); treatment.save(treatment_path)
            loaded_control = SAC.load(control_path, device="cpu")
            loaded_treatment = SAC.load(treatment_path, device="cpu")
            self.assertEqual(actor_activation_id(loaded_control.policy), ACTOR_RELU_V1)
            self.assertEqual(actor_activation_id(loaded_treatment.policy), ACTOR_LEAKY_RELU_001_V1)
            self.assertTrue(activation_inventory(loaded_treatment.policy)["critic_is_relu_only"])
            for label, model, expected in (("control", control, ACTOR_RELU_V1),
                                           ("treatment", treatment, ACTOR_LEAKY_RELU_001_V1)):
                bundle = f"{directory}/{label}_bundle"
                save_sac_checkpoint(model, __import__("pathlib").Path(bundle), Stage5ObservationEncoder(core_only=False),
                                    total_timesteps=0, training_root_seed=101,
                                    training_config={"actor_activation_id": expected},
                                    project_root=__import__("pathlib").Path.cwd())
                restored, metadata = load_sac_checkpoint(
                    __import__("pathlib").Path(bundle), expected_actor_activation_id=expected,
                )
                self.assertEqual(metadata["actor_activation"], expected)
                self.assertEqual(actor_activation_id(restored.policy), expected)
                with self.assertRaises(ValueError):
                    load_sac_checkpoint(__import__("pathlib").Path(bundle), expected_actor_activation_id=(
                        ACTOR_LEAKY_RELU_001_V1 if expected == ACTOR_RELU_V1 else ACTOR_RELU_V1
                    ))
        control.env.close(); treatment.env.close()


    def test_leaky_relu_has_nonzero_negative_side_gradient_and_relu_does_not(self):
        x_relu = torch.tensor([-2.0], requires_grad=True)
        torch.nn.ReLU()(x_relu).sum().backward()
        x_leaky = torch.tensor([-2.0], requires_grad=True)
        torch.nn.LeakyReLU(negative_slope=0.01)(x_leaky).sum().backward()
        self.assertEqual(x_relu.grad.item(), 0.0)
        self.assertAlmostEqual(x_leaky.grad.item(), 0.01)
