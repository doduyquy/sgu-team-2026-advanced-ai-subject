"""Run P2-F evaluation with the explicitly registered SAC root-seed alias.

The shared P2 evaluator reads ``training.root_seed``; P2-F registers the same
value as ``sac.root_seed``. This adapter supplies only that equivalent metadata
in memory and records its own source hash; it does not change training or the
frozen P2-F protocol JSON.
"""
from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_stage5_p2f_curriculum as p2f


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="s5_p2f_start_distance_seed101_20261121_retry3")
    args = parser.parse_args()
    original_loader = p2f.load_protocol

    def load_with_seed_alias():
        protocol = original_loader()
        protocol = dict(protocol)
        protocol["training"] = dict(protocol.get("training", {}))
        protocol["training"]["root_seed"] = int(protocol["sac"]["root_seed"])
        return protocol

    p2f.load_protocol = load_with_seed_alias
    from scripts import evaluate_stage5_p2f_curriculum as evaluator
    from scripts import evaluate_stage5_p2d_actor_activation as p2d_actor_analysis
    evaluator.p2d.audit_observations = p2d_actor_analysis.audit_observations

    run_home = ROOT / "runs/stage5_p2f" / args.run_id
    old_evaluation = run_home / "evaluation"
    if old_evaluation.exists():
        preserved = run_home / "evaluation_attempt_1_incomplete"
        if preserved.exists():
            raise FileExistsError(f"Refusing to overwrite preserved evaluation attempt: {preserved}")
        shutil.move(str(old_evaluation), str(preserved))
    sys.argv = [sys.argv[0], "--run-id", args.run_id]
    evaluator.main()
    run_root = run_home / "evaluation"
    source = Path(__file__).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    actor_helper = Path(p2d_actor_analysis.__file__).resolve()
    record = {
        "evaluation_adapter": "p2f_p2_evaluator_root_seed_alias_v1",
        "adapter_source_sha256": digest,
        "mapping": "training.root_seed = sac.root_seed (both registered as 101)",
        "actor_diagnostic_helper": "evaluate_stage5_p2d_actor_activation.audit_observations",
        "actor_diagnostic_helper_source_sha256": hashlib.sha256(actor_helper.read_bytes()).hexdigest(),
        "training_protocol_sha256": original_loader()["protocol_sha256"],
        "training_artifacts_modified": False,
        "scope": "TRAIN/PILOT_DIAGNOSTIC only",
    }
    output = run_root / "EVALUATION_ADAPTER.json"
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".sha256").write_text(f"{checksum}  {output.name}\n", encoding="ascii")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
