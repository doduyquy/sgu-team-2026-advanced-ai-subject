"""Short, non-scientific MetaDrive smoke for P2 producer/action/checkpoint paths."""
from __future__ import annotations

import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER
from src.training.stage5.checkpoint import save_sac_checkpoint, verify_checkpoint_integrity, load_sac_checkpoint
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.training.stage5.pilot_observation_transform import GROUPWISE_PHYSICAL_V1, GroupwisePhysicalObservation
from src.training.stage5.temporal_exploration import TemporalExplorationSAC
from scripts.run_stage5_p2_experiment import model_kwargs, make_env, read_protocol, seed_everything, state_fingerprint, load_initial_into
from scripts.evaluate_stage5_p2 import run_one


def main() -> None:
    protocol = read_protocol()
    out = ROOT / "runs/stage5_p2/smoke_20261121_eval_retry4"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite smoke evidence: {out}")
    out.mkdir(parents=True)
    results = []
    for arm, mode in (("A", "standard"), ("B", "ou_ar1_v1")):
        env, _ = make_env(protocol, arm, training=True)
        ckpt = out / f"arm_{arm}_checkpoint"
        verified = None
        try:
            seed_everything(101)
            kwargs = model_kwargs(protocol, mode)
            kwargs.update({"learning_starts": 8, "batch_size": 8, "buffer_size": 128,
                           "policy_kwargs": {"net_arch": [32, 32], "activation_fn": torch.nn.ReLU}})
            model = TemporalExplorationSAC(env=env, **kwargs)
            actions = []
            model.action_audit_sink = actions.append
            model.learn(total_timesteps=24, log_interval=1000)
            if model.action_mismatch_count != 0 or len(actions) != 24:
                raise AssertionError(f"Action fidelity failed for smoke arm {arm}")
            if model.env.get_attr("task_context_producer_id") != [REFERENCE_TASK_PRODUCER]:
                raise AssertionError("Smoke environment did not use corrected F09 producer")
            if any(not np.allclose(row["executed_action"], row["replay_action"], atol=1e-6, rtol=0) for row in actions):
                raise AssertionError("Environment/replay action mismatch in smoke")
            encoder = Stage5ObservationEncoder(core_only=False)
            save_sac_checkpoint(model, ckpt, encoder, int(model.num_timesteps), 101,
                                training_config={"smoke": True, "task_context_producer_id": REFERENCE_TASK_PRODUCER},
                                project_root=ROOT, observation_transform=GROUPWISE_PHYSICAL_V1["transform_id"],
                                task_context_producer_id=REFERENCE_TASK_PRODUCER)
            verified = verify_checkpoint_integrity(ckpt, expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
                                                   task_context_producer_id=REFERENCE_TASK_PRODUCER)
            loaded, _ = load_sac_checkpoint(ckpt, expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
                                            task_context_producer_id=REFERENCE_TASK_PRODUCER)
            results.append({"arm": arm, "actual_steps": model.num_timesteps,
                            "action_mismatches": model.action_mismatch_count,
                            "corrected_producer": REFERENCE_TASK_PRODUCER,
                            "checkpoint_sha256": verified["model_sha256"],
                            "loaded_policy_prediction_finite": bool(np.isfinite(loaded.predict(np.zeros(409, dtype=np.float32), deterministic=True)[0]).all())})
        finally:
            env.close()
            gc.collect()
        # Mirror the scientific runner: frozen evaluation starts only after
        # collection has released its MetaDrive simulator and replay is closed.
        evaluation = run_one(ckpt, protocol, out / f"evaluation_arm_{arm}", cap=5)
        results[-1].update({
            "evaluation_cases": evaluation["case_count"],
            "evaluation_all_explicitly_censored": evaluation["aggregate"]["censored"] == 6,
            "evaluation_sampler_side_effects": evaluation["sampler_sampled_count"],
        })
    summary = {"status": "TECHNICAL_SMOKE_ONLY", "scientific_training": False,
               "official_validation_test_accessed": False, "arms": results}
    (out / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
