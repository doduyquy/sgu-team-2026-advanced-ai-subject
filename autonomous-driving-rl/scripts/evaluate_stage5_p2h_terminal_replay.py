"""Frozen deterministic TRAIN evaluation for the P2-H checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_stage5_p2 as p2eval
from scripts.run_stage5_p2h_terminal_replay import PROTOCOL_PATH, read_protocol
from src.training.stage5.actor_activation_ablation import ACTOR_LEAKY_RELU_001_V1
from src.training.stage5.checkpoint import load_sac_checkpoint
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/stage5_p2h/s5_p2h_terminal_replay_seed101_20261121")
    parser.add_argument("--arm", choices=("A", "B", "BOTH"), default="BOTH")
    parser.add_argument("--checkpoint-step", type=int, choices=(0, 2000, 5000, 10000), action="append")
    args = parser.parse_args()
    protocol = read_protocol()
    arms = ("A", "B") if args.arm == "BOTH" else (args.arm,)
    steps = args.checkpoint_step or [0, 2000, 5000, 10000]
    encoder = Stage5ObservationEncoder(core_only=False)
    for arm in arms:
        for step in steps:
            checkpoint = args.run_dir / f"arm_{arm}/checkpoints/step_{step:06d}"
            model, metadata = load_sac_checkpoint(checkpoint, device="cpu",
                expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
                task_context_producer_id=REFERENCE_TASK_PRODUCER,
                expected_actor_activation_id=ACTOR_LEAKY_RELU_001_V1)
            output = args.run_dir / "evaluation" / f"arm_{arm}"
            probe = p2eval.run_one(checkpoint, protocol, output, cap=300)
            full = p2eval.run_one(checkpoint, protocol, output, cap=None) if step == 10000 else None
            row = {"protocol_sha256": protocol["protocol_sha256"], "arm": arm, "step": step,
                   "checkpoint_sha256": metadata["model_sha256"], "checkpoint_identity": metadata.get("checkpoint_identity"),
                   "activation_id": metadata["actor_activation"], "probe_result_sha256": probe["result_sha256"],
                   "probe_aggregate": probe["aggregate"], "actor_responsiveness": probe["actor_responsiveness"],
                   "full_result_sha256": full["result_sha256"] if full else None,
                   "full_aggregate": full["aggregate"] if full else None,
                   "official_validation_test_accessed": False, "benchmark_eligible": False}
            target = output / f"p2h_checkpoint_{step:05d}_summary.json"
            target.write_text(json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            print(f"{arm} {step}: probe={probe['aggregate']}" + (f" full={full['aggregate']}" if full else ""), flush=True)
            del model


if __name__ == "__main__":
    main()
