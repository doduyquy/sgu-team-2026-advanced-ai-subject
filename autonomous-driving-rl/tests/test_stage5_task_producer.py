"""Checkpoint and fail-fast producer integration tests. No SAC updates."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.launcher.registry import make_sac_agent_factory
from src.platform.agent_input import build_agent_input
from src.platform.task_context_producer import (
    LEGACY_TASK_PRODUCER, REFERENCE_TASK_PRODUCER, TaskContextLocalizationError,
    require_checkpoint_producer, task_producer_identity,
)
from src.training.stage5.checkpoint import CheckpointIntegrityError, load_sac_checkpoint, save_sac_checkpoint
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.sac_trainer import SacBaselineTrainer


class TestTaskProducer(unittest.TestCase):
    def test_missing_version_is_legacy_only_and_fingerprint_must_match(self):
        self.assertEqual(require_checkpoint_producer({}, LEGACY_TASK_PRODUCER)["id"], LEGACY_TASK_PRODUCER)
        with self.assertRaisesRegex(ValueError, "producer mismatch"):
            require_checkpoint_producer({}, REFERENCE_TASK_PRODUCER)
        identity = task_producer_identity(REFERENCE_TASK_PRODUCER)
        identity["configuration"]["spacing_m"] = 3
        with self.assertRaisesRegex(ValueError, "producer mismatch"):
            require_checkpoint_producer({"task_context_producer": identity}, REFERENCE_TASK_PRODUCER)

    def test_unlocalized_observation_aborts_at_first_sample(self):
        ego = SimpleNamespace(position=(0, 0), velocity=(0, 0), heading_theta=0)
        # The native lane-width getter cannot be called before localization succeeds.
        nav = SimpleNamespace(checkpoints=None, current_lane=None)
        with self.assertRaisesRegex(TaskContextLocalizationError, "at decision 7"):
            build_agent_input(np.zeros(259, np.float32), ego, [], nav,
                              SimpleNamespace(graph={}), 7,
                              task_context_producer_id=REFERENCE_TASK_PRODUCER)

    def test_checkpoint_binds_train_env_and_inference_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle"
            trainer = SacBaselineTrainer(config_overrides={"task_context_producer_id": REFERENCE_TASK_PRODUCER})
            legacy_env = Stage5TrainingEnv()
            try:
                self.assertEqual(trainer.model._n_updates, 0)
                with self.assertRaisesRegex(CheckpointIntegrityError, "training env"):
                    save_sac_checkpoint(trainer.model, path, trainer.encoder, 0, 101)
                trainer.save_checkpoint(path)
                with self.assertRaisesRegex(CheckpointIntegrityError, "producer mismatch"):
                    make_sac_agent_factory(path)
                with self.assertRaisesRegex(CheckpointIntegrityError, "environment TaskContext"):
                    load_sac_checkpoint(path, env=legacy_env, task_context_producer_id=REFERENCE_TASK_PRODUCER)
                factory = make_sac_agent_factory(path, task_context_producer_id=REFERENCE_TASK_PRODUCER)
                agent = factory()
                try:
                    agent.reset(None)
                    with self.assertRaisesRegex(ValueError, "Policy input TaskContext producer mismatch"):
                        agent.act(None)
                    self.assertEqual(agent.checkpoint_provenance["task_context_producer"],
                                     trainer.config["task_context_producer"])
                    _, meta = load_sac_checkpoint(path, task_context_producer_id=REFERENCE_TASK_PRODUCER)
                    self.assertEqual(meta["training_config"]["task_context_producer"],
                                     factory.experiment_parameters()["task_context_producer"])
                finally:
                    agent.close()
            finally:
                trainer.close()
                legacy_env.close()
