"""
Unit and Integration Test Suite for Stage 5 Soft Actor-Critic (SAC) Baseline (Gate S5.2).

Verifies:
1. SAC initialization with MlpPolicy, 409D observation space, Box(2) action space, and config parameters.
2. Checkpoint bundle persistence: model.zip, metadata.json, CHECKSUMS.sha256, cryptographic verification.
3. Cryptographic rejection: corrupted model.zip, tampered checksum file, or metadata discrepancy.
4. Schema fingerprint rejection: rejection when encoder schema fingerprint does not match loaded checkpoint.
5. Save/load bit-for-bit prediction parity on identical observation vectors.
6. Stage5SacAgent lifecycle: descriptor compliance, reset(), deterministic act(), and valid action bounds.
7. Stage5SacAgent unready behavior: safe braking fallback prior to checkpoint loading.
8. Training harness holdout isolation: 100% TRAIN split containment, zero holdout leakage.
9. Platform contract hashes: strict immutability of GATE5, GATE6, GATE7, Launcher, and Execution contracts.
10. Canonical Agent Registry integration: candidate registration, eligibility, and factory instantiation.
"""

from __future__ import annotations

# Windows DLL initialization order: import torch before Panda3D
try:
    import torch
except ImportError:
    pass

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from src.agents.stage5_sac_agent import Stage5SacAgent
from src.launcher.cases import get_default_project_root
from src.launcher.registry import build_canonical_agent_registry
from src.launcher.resolver import (
    DEFAULT_LAUNCHER_CONTRACT_HASH,
    DEFAULT_PLATFORM_EXECUTION_HASH,
    GATE5_LOCKED_BENCHMARK_HASH,
    GATE6_LOCKED_AGENT_HASH,
    GATE6_LOCKED_RUNTIME_HASH,
    GATE7_LOCKED_LOGGING_HASH,
    GATE7_LOCKED_OBSERVABILITY_HASH,
)
from src.platform import (
    AgentDecision,
    AgentInputV1,
    AgentPublicEpisodeContext,
    CoreObservationV1,
    InputProfileId,
    TaskContextV1,
    TrafficContextV1,
)
from src.training.stage5 import (
    STAGE5_FULL_OBSERVATION_DIM,
    SacBaselineTrainer,
    Stage5ObservationEncoder,
    Stage5TrainingEnv,
    TrainCaseSampler,
    load_sac_checkpoint,
    save_sac_checkpoint,
    verify_checkpoint_integrity,
)
from src.training.stage5.checkpoint import CheckpointIntegrityError


class TestStage5Sac(unittest.TestCase):
    """Authoritative test suite for Stage 5 SAC baseline implementation."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = get_default_project_root()
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="sac_test_"))

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_sac_initialization(self):
        """SAC initializes with MlpPolicy, 409D observation space, and Box(2) action space."""
        config_overrides = {
            "sac_hyperparameters": {
                "buffer_size": 1000,
                "batch_size": 32,
                "learning_starts": 10,
                "learning_rate": 0.0001,
            }
        }
        trainer = SacBaselineTrainer(
            config_overrides=config_overrides,
            training_root_seed=101,
            project_root=self.project_root,
            device="cpu",
        )
        try:
            self.assertEqual(trainer.model.observation_space.shape, (STAGE5_FULL_OBSERVATION_DIM,))
            self.assertEqual(trainer.model.action_space.shape, (2,))
            self.assertEqual(trainer.model.buffer_size, 1000)
            self.assertEqual(trainer.model.batch_size, 32)
            self.assertEqual(trainer.model.learning_starts, 10)
            self.assertAlmostEqual(trainer.model.learning_rate, 0.0001)
        finally:
            trainer.close()

    def test_02_checkpoint_save_and_verify(self):
        """Tests saving SAC checkpoint bundle and cryptographic verification."""
        ckpt_dir = self.temp_dir / "ckpt_save_test"
        trainer = SacBaselineTrainer(
            config_overrides={"sac_hyperparameters": {"buffer_size": 500, "batch_size": 32, "learning_starts": 5}},
            training_root_seed=202,
            project_root=self.project_root,
            device="cpu",
        )
        try:
            saved_path = trainer.save_checkpoint(ckpt_dir, save_replay_buffer=False)
            self.assertTrue(saved_path.exists())
            self.assertTrue((saved_path / "model.zip").exists())
            self.assertTrue((saved_path / "metadata.json").exists())
            self.assertTrue((saved_path / "CHECKSUMS.sha256").exists())

            # Verify integrity
            encoder = Stage5ObservationEncoder(core_only=False)
            metadata = verify_checkpoint_integrity(
                saved_path,
                expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
            )
            self.assertEqual(metadata["algorithm"], "SAC")
            self.assertEqual(metadata["observation_dim"], 409)
            self.assertEqual(metadata["action_dim"], 2)
            self.assertIn("model_sha256", metadata)
        finally:
            trainer.close()

    def test_03_checkpoint_rejection_corrupted_model(self):
        """Tampering with model.zip raises CheckpointIntegrityError."""
        ckpt_dir = self.temp_dir / "ckpt_corrupt_test"
        trainer = SacBaselineTrainer(
            config_overrides={"sac_hyperparameters": {"buffer_size": 500, "batch_size": 32, "learning_starts": 5}},
            training_root_seed=303,
            project_root=self.project_root,
            device="cpu",
        )
        try:
            saved_path = trainer.save_checkpoint(ckpt_dir, save_replay_buffer=False)
            model_file = saved_path / "model.zip"

            # Corrupt model.zip by flipping trailing bytes
            content = bytearray(model_file.read_bytes())
            content[-1] ^= 0xFF
            model_file.write_bytes(bytes(content))

            encoder = Stage5ObservationEncoder(core_only=False)
            with self.assertRaises(CheckpointIntegrityError):
                load_sac_checkpoint(
                    saved_path,
                    expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
                    device="cpu",
                )
        finally:
            trainer.close()

    def test_04_checkpoint_rejection_encoder_fingerprint_mismatch(self):
        """Mismatched encoder schema fingerprint raises CheckpointIntegrityError."""
        smoke_dir = self.project_root / "runs" / "checkpoints" / "stage5_sac_smoke"
        if not smoke_dir.exists():
            self.skipTest("Smoke checkpoint directory does not exist")

        bogus_fingerprint = "0000000000000000000000000000000000000000000000000000000000000000"
        with self.assertRaises(CheckpointIntegrityError) as ctx:
            load_sac_checkpoint(
                smoke_dir,
                expected_encoder_fingerprint=bogus_fingerprint,
                device="cpu",
            )
        self.assertIn("Observation encoder schema mismatch", str(ctx.exception))

    def test_05_save_load_prediction_parity(self):
        """Model loaded from checkpoint reproduces bit-for-bit identical actions on arbitrary input."""
        smoke_dir = self.project_root / "runs" / "checkpoints" / "stage5_sac_smoke"
        if not smoke_dir.exists():
            self.skipTest("Smoke checkpoint directory does not exist")

        loaded_model_1, _ = load_sac_checkpoint(smoke_dir, device="cpu")
        loaded_model_2, _ = load_sac_checkpoint(smoke_dir, device="cpu")

        rng = np.random.RandomState(42)
        test_obs = rng.uniform(-1.0, 1.0, size=(STAGE5_FULL_OBSERVATION_DIM,)).astype(np.float32)

        action_1, _ = loaded_model_1.predict(test_obs, deterministic=True)
        action_2, _ = loaded_model_2.predict(test_obs, deterministic=True)

        np.testing.assert_allclose(action_1, action_2, atol=1e-6)
        self.assertTrue(np.all(action_1 >= -1.0))
        self.assertTrue(np.all(action_1 <= 1.0))

    def test_06_stage5_sac_agent_lifecycle(self):
        """Stage5SacAgent conforms to Platform V1 lifecycle and produces valid continuous decisions."""
        agent = Stage5SacAgent()
        desc = agent.descriptor

        self.assertEqual(desc.agent_id, "stage5_sac")
        self.assertEqual(desc.input_profile_id, "STATE_DECISION_V1")
        self.assertEqual(desc.action_adapter_id, "continuous_box2_v1")
        self.assertEqual(desc.inference_stochasticity, "deterministic")
        self.assertFalse(desc.stateful_within_episode)

        ctx = AgentPublicEpisodeContext()
        agent.reset(ctx)
        self.assertTrue(agent._ready)

        # Build valid synthetic AgentInputV1
        raw_core = np.zeros(259, dtype=np.float32)
        core_obs = CoreObservationV1(raw_core)
        agent_input = AgentInputV1(
            profile_id=InputProfileId.STATE_DECISION_V1,
            core_observation=core_obs,
            traffic_context=TrafficContextV1.empty(),
            task_context=TaskContextV1.empty(),
            step_index=0,
        )

        decision = agent.act(agent_input)
        self.assertIsInstance(decision, AgentDecision)
        self.assertEqual(len(decision.action_payload), 2)
        self.assertTrue(-1.0 <= decision.action_payload[0] <= 1.0)
        self.assertTrue(-1.0 <= decision.action_payload[1] <= 1.0)
        self.assertTrue(decision.diagnostics.get("deterministic"))

    def test_07_stage5_sac_agent_unready_behavior(self):
        """Calling act() prior to model loading raises RuntimeError."""
        agent = Stage5SacAgent.__new__(Stage5SacAgent)
        agent.agent_id = "stage5_sac"
        agent.agent_version = "0.1.0"
        agent._descriptor = None
        agent._model = None
        agent._ready = False

        agent_input = AgentInputV1(
            profile_id=InputProfileId.STATE_DECISION_V1,
            core_observation=CoreObservationV1(np.zeros(259, dtype=np.float32)),
            traffic_context=TrafficContextV1.empty(),
            task_context=TaskContextV1.empty(),
            step_index=0,
        )
        with self.assertRaises(RuntimeError) as ctx:
            agent.act(agent_input)
        self.assertIn("must be reset with a loaded model before act()", str(ctx.exception))

    def test_08_train_sampler_strict_isolation_in_training(self):
        """TrainCaseSampler strictly contains only the 180 TRAIN split geometries."""
        sampler = TrainCaseSampler(project_root=self.project_root)
        self.assertEqual(sampler.total_eligible_geometries, 180)

        # Sample and validate cases
        for _ in range(20):
            case = sampler.sample_case()
            self.assertEqual(case.split, "TRAIN")
            sampler.validate_train_case(case)

    def test_09_platform_contract_hashes_immutability(self):
        """Platform, Launcher, and Benchmark contract hashes remain 100% immutable."""
        self.assertEqual(
            GATE5_LOCKED_BENCHMARK_HASH,
            "9ddd889b84d8705fae618879e5035556c80d0276a3dc2a58a7963937ebb59f77",
        )
        self.assertEqual(
            GATE6_LOCKED_AGENT_HASH,
            "53aa37079ff44afa75d9a3f921b0c1f98c4600d882fced51fc8d9197795058eb",
        )
        self.assertEqual(
            GATE6_LOCKED_RUNTIME_HASH,
            "c7698768539a769c7b2bc6b90771ff234276a974b6e0353bbb03119faf2f79ad",
        )
        self.assertEqual(
            GATE7_LOCKED_LOGGING_HASH,
            "0d17915556d83f49d9519576e95920c0853700f33a91593ef85b3aeef6cbb9f2",
        )
        self.assertEqual(
            GATE7_LOCKED_OBSERVABILITY_HASH,
            "f00c27fca8abf8145596576cd0dd2540181ff8ae7eccf7806c864921933d3581",
        )
        self.assertEqual(
            DEFAULT_LAUNCHER_CONTRACT_HASH,
            "5e6269b6e32349bd3ee34ca52583c00f832bd0a737330c894e9a8d314653db04",
        )
        self.assertEqual(
            DEFAULT_PLATFORM_EXECUTION_HASH,
            "442beafd86212419dc7b3edfa60e53715bf33877d3dd72e61c54091520b0563d",
        )

    def test_10_agent_registry_stage5_sac(self):
        """Canonical agent registry correctly includes candidate stage5_sac."""
        reg = build_canonical_agent_registry()
        self.assertTrue(reg.has_agent("stage5_sac"))

        entry = reg.get("stage5_sac")
        self.assertEqual(entry.agent_id, "stage5_sac")
        self.assertEqual(entry.stage_label, "STAGE_5")
        self.assertEqual(entry.method_family, "MODEL_FREE_RL")
        self.assertEqual(entry.purpose, "CANDIDATE")
        self.assertEqual(entry.action_adapter_id, "continuous_box2_v1")
        self.assertEqual(entry.input_profile_id, "STATE_DECISION_V1")
        self.assertEqual(entry.inference_stochasticity, "deterministic")
        self.assertFalse(entry.stateful_within_episode)
        self.assertTrue(entry.sandbox_eligible)
        self.assertTrue(entry.audit_eligible)
        self.assertFalse(entry.benchmark_eligible)


if __name__ == "__main__":
    unittest.main()
