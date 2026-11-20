"""Short technical smoke for P2-D; results are not scientific evidence."""
from __future__ import annotations

import gc
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.sac.policies import SACPolicy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_stage5_p2_experiment as p2
from scripts.evaluate_stage5_p2 import run_one
from scripts.run_stage5_p2d_actor_activation import read_protocol
from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER
from src.training.stage5.actor_activation_ablation import ActorOnlyLeakyReLUPolicy, activation_inventory
from src.training.stage5.checkpoint import load_sac_checkpoint, save_sac_checkpoint
from src.training.stage5.observation_encoder import Stage5ObservationEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="smoke_20261121_v2")
    args = parser.parse_args()
    protocol = read_protocol()
    out = ROOT / "runs/stage5_p2d" / args.run_id
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite P2-D smoke artifacts: {out}")
    out.mkdir(parents=True)
    results = []
    initial_env, _ = p2.make_env(protocol, "INITIAL", training=True)
    try:
        p2.seed_everything(101)
        base_kwargs = p2.model_kwargs(protocol, "standard")
        base_kwargs["policy_kwargs"] = {"net_arch": [32, 32], "activation_fn": torch.nn.ReLU}
        shared = p2.P2InstrumentedSAC(env=initial_env, **base_kwargs)
        shared_state = {name: tensor.detach().cpu().clone() for name, tensor in shared.policy.state_dict().items()}
    finally:
        initial_env.close()
    for arm, policy_class, expected in (
        ("A", SACPolicy, "actor_relu_v1"),
        ("B", ActorOnlyLeakyReLUPolicy, "actor_leaky_relu_001_v1"),
    ):
        env, _ = p2.make_env(protocol, arm, training=True)
        try:
            p2.seed_everything(101)
            kwargs = p2.model_kwargs(protocol, "standard")
            kwargs["policy"] = policy_class
            kwargs.update({"learning_starts": 8, "batch_size": 8, "buffer_size": 128,
                           "policy_kwargs": {"net_arch": [32, 32], "activation_fn": torch.nn.ReLU}})
            model = p2.P2InstrumentedSAC(env=env, **kwargs)
            model.policy.load_state_dict(shared_state, strict=True)
            model.install_gradient_hooks()
            if activation_inventory(model.policy)["actor_id"] != expected:
                raise AssertionError("Smoke arm reconstructed wrong actor activation")
            if not activation_inventory(model.policy)["critic_is_relu_only"]:
                raise AssertionError("Smoke treatment changed critic activation")
            records = []
            model.action_audit_sink = records.append
            model.learn(total_timesteps=24, log_interval=1000)
            if model.action_mismatch_count or len(records) != 24:
                raise AssertionError(f"Sampled/executed/replay action mismatch in arm {arm}")
            if any(not np.array_equal(row["executed_action"], row["replay_action"]) for row in records):
                raise AssertionError("Replay action differs from action executed by environment")
            if getattr(env.unwrapped, "task_context_producer_id", None) != REFERENCE_TASK_PRODUCER:
                raise AssertionError("Smoke did not use corrected F09 producer")
            encoder = Stage5ObservationEncoder(core_only=False)
            checkpoint = out / f"arm_{arm}_checkpoint"
            save_sac_checkpoint(
                model, checkpoint, encoder, int(model.num_timesteps), 101,
                training_config={"smoke": True, "actor_activation_id": expected,
                                 "task_context_producer_id": REFERENCE_TASK_PRODUCER},
                project_root=ROOT, observation_transform=protocol["observation"]["post_encoder_transform"],
                task_context_producer_id=REFERENCE_TASK_PRODUCER,
            )
            restored, metadata = load_sac_checkpoint(
                checkpoint, expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
                task_context_producer_id=REFERENCE_TASK_PRODUCER, expected_actor_activation_id=expected,
            )
            pred = restored.predict(np.zeros(409, dtype=np.float32), deterministic=True)[0]
            if not np.isfinite(pred).all():
                raise AssertionError("Restored actor returned non-finite action")
            results.append({"arm": arm, "steps": int(model.num_timesteps), "action_mismatches": model.action_mismatch_count,
                            "actor_activation_id": expected, "critic_activation": "ReLU",
                            "model_sha256": metadata["model_sha256"], "activation_fingerprint": metadata["activation_fingerprint"],
                            "restored_action_finite": True})
            env.close()
            gc.collect()
            evaluation = run_one(checkpoint, protocol, out / f"evaluation_{arm}", cap=5)
            results[-1]["evaluation_censored_cases"] = evaluation["aggregate"]["censored"]
            results[-1]["evaluation_cases"] = evaluation["case_count"]
        finally:
            env.close()
            gc.collect()
    summary = {"status": "TECHNICAL_SMOKE_ONLY", "scientific_training": False,
               "official_validation_test_accessed": False, "protocol_sha256": protocol["protocol_sha256"], "arms": results}
    (out / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
