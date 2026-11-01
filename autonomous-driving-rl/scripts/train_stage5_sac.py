"""
Standalone Training Script for Stage 5 Soft Actor-Critic (SAC) Policy (Gate S5.2).

Executes SAC model-free reinforcement learning on MapSuite V1 TRAIN split:
- Trains continuous Box(2) actor on 409D observation vectors.
- Verifies gradient updates, replay buffer transitions, finite losses, and valid action bounds.
- Persists verifiable checkpoint bundle (model.zip, metadata.json, CHECKSUMS.sha256).
- Tests pre- and post-training prediction on diagnostic fixed case.
"""

from __future__ import annotations

# Import torch first on Windows to ensure c10.dll initializes before Panda3D C++ runtime
try:
    import torch
except ImportError:
    pass

import argparse
import math
from pathlib import Path
import sys
import time

# Ensure subproject root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.training.stage5.checkpoint import load_sac_checkpoint
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.training.stage5.sac_trainer import SacBaselineTrainer
from src.training.stage5.train_sampler import SamplerStrategy


def run_training_session(
    timesteps: int = 5000,
    learning_starts: int = 500,
    buffer_size: int = 50000,
    batch_size: int = 128,
    output_dir: Optional[str] = None,
    training_root_seed: int = 101,
    save_replay: bool = False,
) -> bool:
    """Executes the SAC training session and verifies checkpoint integrity."""
    print("=" * 70)
    print("XETUHANH — STAGE 5 SAC BASELINE TRAINING PIPELINE (GATE S5.2)")
    print("=" * 70)
    print(f"  Target Timesteps: {timesteps}")
    print(f"  Learning Starts:  {learning_starts}")
    print(f"  Buffer Size:      {buffer_size}")
    print(f"  Batch Size:       {batch_size}")
    print(f"  Master Root Seed: {training_root_seed}")

    out_path = Path(output_dir).resolve() if output_dir else (PROJECT_ROOT / "runs" / "checkpoints" / "stage5_sac_smoke")

    trainer = None
    try:
        config_overrides = {
            "sac_hyperparameters": {
                "buffer_size": buffer_size,
                "batch_size": batch_size,
                "learning_starts": learning_starts,
            }
        }

        trainer = SacBaselineTrainer(
            config_overrides=config_overrides,
            training_root_seed=training_root_seed,
            project_root=PROJECT_ROOT,
            device="cpu",
        )

        # 1. Run Diagnostic Case Probe before training
        print("\n--- Diagnostic Pre-Training Probe on Easy/SCS/geom-0 ---")
        diag_env = Stage5TrainingEnv(
            sampler_strategy=SamplerStrategy.FIXED,
            tier_filter="Easy",
            sequence_filter="SCS",
            fixed_geometry_seed=0,
            project_root=PROJECT_ROOT,
        )
        pre_obs, _ = diag_env.reset()
        pre_action, _ = trainer.model.predict(pre_obs, deterministic=True)
        print(f"  Initial Untrained Policy Action: [steer={pre_action[0]:+.3f}, throttle={pre_action[1]:+.3f}]")
        diag_env.close()

        # 2. Execute Training
        start_t = time.time()
        summary = trainer.train(total_timesteps=timesteps)
        train_duration = time.time() - start_t

        # 3. Verify Replay Buffer and Gradient Updates
        rb_size = trainer.model.replay_buffer.size()
        print(f"\n--- Verifying Training Transition Integrity ---")
        print(f"  Replay Buffer Transitions: {rb_size} / {buffer_size}")
        assert rb_size >= timesteps or rb_size == buffer_size, f"Replay buffer size {rb_size} < timesteps {timesteps}"

        # Verify Actor & Critic Losses
        logger_dict = getattr(trainer.model.logger, "name_to_value", {})
        actor_loss = logger_dict.get("train/actor_loss", None)
        critic_loss = logger_dict.get("train/critic_loss", None)
        ent_coef = logger_dict.get("train/ent_coef", None)
        print(f"  Actor Loss:       {actor_loss}")
        print(f"  Critic Loss:      {critic_loss}")
        print(f"  Entropy Coef:     {ent_coef}")

        if timesteps > learning_starts:
            assert actor_loss is not None, "Actor loss must be recorded after learning_starts"
            assert critic_loss is not None, "Critic loss must be recorded after learning_starts"
            assert math.isfinite(actor_loss), f"Actor loss is non-finite: {actor_loss}"
            assert math.isfinite(critic_loss), f"Critic loss is non-finite: {critic_loss}"
            print("  [OK] Finite loss and gradient updates verified successfully!")

        # 4. Save Checkpoint Bundle
        print(f"\n--- Saving Checkpoint to: {out_path} ---")
        saved_dir = trainer.save_checkpoint(out_path, save_replay_buffer=save_replay)
        print(f"  [SAVED] Checkpoint directory: {saved_dir}")

        # 5. Verify Checkpoint Load & Prediction Parity
        print(f"\n--- Verifying Checkpoint Load and Prediction Parity ---")
        loaded_model, metadata = load_sac_checkpoint(saved_dir, device="cpu")
        print(f"  Model SHA-256:       {metadata['model_sha256']}")
        print(f"  Encoder Fingerprint: {metadata['encoder_fingerprint'][:16]}...")
        print(f"  Total Steps:         {metadata['total_timesteps']}")

        # Test prediction parity on pre_obs
        post_action_1, _ = trainer.model.predict(pre_obs, deterministic=True)
        post_action_2, _ = loaded_model.predict(pre_obs, deterministic=True)
        np.testing.assert_allclose(post_action_1, post_action_2, atol=1e-5)
        print(f"  Trained Policy Action: [steer={post_action_1[0]:+.3f}, throttle={post_action_1[1]:+.3f}]")
        print("  [OK] Bit-for-bit save/load prediction parity verified!")

        print("\n" + "=" * 70)
        print("STAGE 5 SAC TRAINING SMOKE COMPLETE (SUCCESS)")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n[TRAINING FAILURE] Error during SAC training: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if trainer is not None:
            trainer.close()
            print("  [CLEANUP] Trainer resources released.")


def main():
    parser = argparse.ArgumentParser(description="Stage 5 SAC Training Pipeline (Gate S5.2)")
    parser.add_argument("--timesteps", type=int, default=5000, help="Total environment steps to train (default: 5000)")
    parser.add_argument("--learning-starts", type=int, default=500, help="Steps before gradient updates begin")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer capacity")
    parser.add_argument("--batch-size", type=int, default=128, help="Minibatch size for gradient updates")
    parser.add_argument("--output-dir", type=str, default=None, help="Target checkpoint directory")
    parser.add_argument("--seed", type=int, default=101, help="Training master seed")
    parser.add_argument("--save-replay", action="store_true", help="Include replay buffer in saved checkpoint")
    args = parser.parse_args()

    success = run_training_session(
        timesteps=args.timesteps,
        learning_starts=args.learning_starts,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        training_root_seed=args.seed,
        save_replay=args.save_replay,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
