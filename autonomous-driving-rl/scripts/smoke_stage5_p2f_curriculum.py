"""Small technical smoke for P2-F reset profiles, action fidelity and checkpoint binding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_stage5_p2_experiment as p2
from scripts.run_stage5_p2d_actor_activation import read_protocol as read_p2d_protocol
from scripts.run_stage5_p2f_curriculum import P2FCurriculumSAC, make_env
from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER
from src.training.stage5.actor_activation_ablation import ACTOR_LEAKY_RELU_001_V1, activation_inventory
from src.training.stage5.checkpoint import load_sac_checkpoint, save_sac_checkpoint
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.training.stage5.start_curriculum import CONTROL_TASK_ID, CURRICULUM_TASK_ID


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="smoke_20261121_v1")
    args = parser.parse_args()
    protocol = read_p2d_protocol()
    out = ROOT / "runs/stage5_p2f" / args.run_id
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite P2-F smoke artifacts: {out}")
    out.mkdir(parents=True)
    results = []
    for arm in ("A", "B"):
        env, base, _ = make_env(protocol, arm, training=True)
        try:
            p2.seed_everything(101)
            kwargs = p2.model_kwargs(protocol, "standard")
            kwargs["policy"] = __import__(
                "src.training.stage5.actor_activation_ablation", fromlist=["ActorOnlyLeakyReLUPolicy"]
            ).ActorOnlyLeakyReLUPolicy
            kwargs.update({"learning_starts": 8, "batch_size": 8, "buffer_size": 128,
                           "policy_kwargs": {"net_arch": [256, 256], "activation_fn": torch.nn.ReLU}})
            model = P2FCurriculumSAC(env=env, **kwargs)
            model.install_gradient_hooks()
            records = []
            model.action_audit_sink = records.append
            model.learn(total_timesteps=24, log_interval=1000)
            expected_task = CONTROL_TASK_ID if arm == "A" else CURRICULUM_TASK_ID
            if model.action_mismatch_count != 0 or len(records) != 24:
                raise AssertionError(f"P2-F action fidelity failed for {arm}")
            if activation_inventory(model.policy)["actor_id"] != ACTOR_LEAKY_RELU_001_V1:
                raise AssertionError("P2-F smoke actor activation mismatch")
            if not activation_inventory(model.policy)["critic_is_relu_only"]:
                raise AssertionError("P2-F smoke changed critic activation")
            if any(row.get("training_task_id") != expected_task for row in records):
                raise AssertionError("P2-F action audit omitted or mismatched task identity")
            if arm == "B" and any(row.get("curriculum_stage_id") != "near_60m" for row in records):
                raise AssertionError("P2-F action audit omitted the reset curriculum stage")
            encoder = Stage5ObservationEncoder(core_only=False)
            checkpoint = out / f"arm_{arm}_checkpoint"
            task_fingerprint = base.training_task_fingerprint
            save_sac_checkpoint(
                model, checkpoint, encoder, int(model.num_timesteps), 101,
                training_config={"technical_smoke": True, "training_task_id": expected_task,
                                 "training_task_fingerprint": task_fingerprint},
                project_root=ROOT, observation_transform=protocol["observation"]["post_encoder_transform"],
                task_context_producer_id=REFERENCE_TASK_PRODUCER,
            )
            restored, meta = load_sac_checkpoint(
                checkpoint, expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
                task_context_producer_id=REFERENCE_TASK_PRODUCER,
                expected_actor_activation_id=ACTOR_LEAKY_RELU_001_V1,
            )
            action = restored.predict(np.zeros(409, dtype=np.float32), deterministic=True)[0]
            if not np.isfinite(action).all():
                raise AssertionError("Restored smoke model emitted a non-finite action")
            results.append({
                "arm": arm, "steps": int(model.num_timesteps), "action_mismatches": model.action_mismatch_count,
                "task_id": expected_task, "task_fingerprint": task_fingerprint,
                "activation_id": meta["actor_activation"], "checkpoint_sha256": meta["model_sha256"],
                "task_fields_in_action_audit": True, "checkpoint_reload": "PASS",
            })
        finally:
            env.close()
    summary = {"status": "TECHNICAL_SMOKE_ONLY", "scientific_training": False,
               "official_evaluation": False, "arms": results}
    (out / "smoke_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
