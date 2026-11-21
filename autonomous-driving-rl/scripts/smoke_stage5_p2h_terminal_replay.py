"""Short non-scientific MetaDrive integration smoke for P2-H treatment B."""
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import torch as th

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_stage5_p2_experiment as p2
from scripts.run_stage5_p2h_terminal_replay import P2HInstrumentedSAC, read_protocol, sampler_kwargs
from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER
from src.training.stage5.actor_activation_ablation import ActorOnlyLeakyReLUPolicy
from src.training.stage5.checkpoint import load_sac_checkpoint, save_sac_checkpoint
from src.training.stage5.observation_encoder import Stage5ObservationEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="smoke_meta_20261121_retry1")
    args = parser.parse_args()
    protocol = read_protocol()
    target = ROOT / "runs/stage5_p2h" / args.run_id
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite P2-H technical smoke: {target}")
    target.mkdir(parents=True)
    env, _ = p2.make_env(protocol, "SMOKE", training=True)
    try:
        if env.env.task_context_producer_id != REFERENCE_TASK_PRODUCER:
            raise RuntimeError("MetaDrive smoke did not construct the corrected F09 producer")
        p2.seed_everything(101)
        kwargs = p2.model_kwargs(protocol, "standard")
        kwargs.update({"policy": ActorOnlyLeakyReLUPolicy, "learning_starts": 8,
                       "buffer_size": 256, "batch_size": 8,
                       "policy_kwargs": {"net_arch": [32, 32], "activation_fn": th.nn.ReLU}})
        model = P2HInstrumentedSAC(env=env, p2h_sampler_config=sampler_kwargs(protocol), **kwargs)
        if not model.replay_buffer.handle_timeout_termination:
            raise RuntimeError("SB3 timeout bootstrap is disabled")
        executed = []
        model.action_audit_sink = executed.append
        model.learn(total_timesteps=64, log_interval=1000)
        if len(executed) != 64 or model.action_mismatch_count:
            raise RuntimeError("P2-H MetaDrive action-to-replay fidelity failed")
        if any(not np.array_equal(row["executed_action"], row["replay_action"]) for row in executed):
            raise RuntimeError("MetaDrive executed action and replay action differ")
        rewards = np.asarray([row["reward"] for row in executed], dtype=np.float64)
        if not np.isfinite(rewards).all():
            raise RuntimeError("Non-finite RewardSpecV1 values observed in MetaDrive smoke")
        encoder = Stage5ObservationEncoder(core_only=False)
        bundle = target / "checkpoint"
        save_sac_checkpoint(model, bundle, encoder, 64, 101,
            training_config={"p2h_technical_smoke": True,
                             "task_context_producer_id": REFERENCE_TASK_PRODUCER,
                             "actor_activation_id": "actor_leaky_relu_001_v1"},
            project_root=ROOT, observation_transform=protocol["observation"]["post_encoder_transform"],
            task_context_producer_id=REFERENCE_TASK_PRODUCER)
        restored, metadata = load_sac_checkpoint(bundle,
            expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
            task_context_producer_id=REFERENCE_TASK_PRODUCER,
            expected_actor_activation_id="actor_leaky_relu_001_v1")
        action = restored.predict(np.zeros(409, np.float32), deterministic=True)[0]
        summary = {"status": "TECHNICAL_SMOKE_ONLY", "scientific_training": False,
            "environment": "MetaDrive TRAIN only", "steps": 64, "optimizer_updates": model._n_updates,
            "geometry_exposure": {key: value for key, value in __import__("collections").Counter(x["geometry_id"] for x in executed).items()},
            "corrected_producer_id": REFERENCE_TASK_PRODUCER,
            "action_mismatches": model.action_mismatch_count, "finite_reward_rows": int(np.isfinite(rewards).sum()),
            "reward_range": [float(rewards.min()), float(rewards.max())],
            "timeout_bootstrap_enabled": bool(model.replay_buffer.handle_timeout_termination),
            "checkpoint_model_sha256": metadata["model_sha256"],
            "restored_action_finite": bool(np.isfinite(action).all()),
            "terminal_candidates": int(np.sum(model.terminal_kinds[:64] == 1) + np.sum(model.terminal_kinds[:64] == 2)),
            "treatment_sampling_fallback_if_no_terminals": model.last_sampling_stats,
            "official_validation_test_accessed": False, "benchmark_eligible": False}
        (target / "smoke_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
