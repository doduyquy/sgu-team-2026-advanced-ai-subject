"""Build a compact, hash-bound summary of the completed P2-D artifacts."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage5_p2d_actor_activation import read_protocol
from src.training.stage5.actor_activation_ablation import ACTOR_LEAKY_RELU_001_V1, ACTOR_RELU_V1
from src.training.stage5.checkpoint import verify_checkpoint_integrity
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER

RUN = ROOT / "runs/stage5_p2d/s5_p2d_actor_activation_seed101_20261121"
OUT = ROOT / "results/stage5/p2d_actor_activation_seed101_20261121"
STEPS = (0, 2000, 5000, 10000)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    protocol = read_protocol()
    run_manifest = read_json(RUN / "run_manifest.json")
    encoder = Stage5ObservationEncoder(core_only=False)
    arms = {}
    file_hashes = {"run_manifest.json": sha256(RUN / "run_manifest.json")}
    checkpoint_rows = {}
    for arm in ("A", "B"):
        arm_dir = RUN / f"arm_{arm}"
        training = read_json(arm_dir / "training_summary.json")
        expected_activation = ACTOR_RELU_V1 if arm == "A" else ACTOR_LEAKY_RELU_001_V1
        checkpoint_data = {}
        probes = {}
        activation_data = {}
        for step in STEPS:
            checkpoint = arm_dir / "checkpoints" / f"step_{step:06d}"
            meta = verify_checkpoint_integrity(
                checkpoint, expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
                task_context_producer_id=REFERENCE_TASK_PRODUCER,
            )
            if meta.get("actor_activation") != expected_activation:
                raise ValueError(f"Unexpected activation in {arm} checkpoint {step}")
            checkpoint_data[str(step)] = {
                "model_sha256": meta["model_sha256"],
                "metadata_sha256": sha256(checkpoint / "metadata.json"),
                "activation_id": meta["actor_activation"],
                "activation_fingerprint": meta["activation_fingerprint"],
                "replay_buffer_sha256": meta.get("replay_buffer_sha256"),
                "optimizer_updates": meta["training_summary"].get("optimizer_updates"),
                "alpha": meta["training_summary"].get("alpha"),
            }
            for suffix in ("CHECKSUMS.sha256", "metadata.json", "model.zip", "replay_buffer.pkl"):
                path = checkpoint / suffix
                if path.is_file():
                    file_hashes[f"{arm}/checkpoints/step_{step:06d}/{suffix}"] = sha256(path)
            probe_path = RUN / "evaluation" / f"arm_{arm}" / f"step_{step:05d}_probe.json"
            probe = read_json(probe_path)
            if probe["checkpoint_sha256"] != meta["model_sha256"] or probe["case_count"] != 6:
                raise ValueError(f"Evaluation/checkpoint schedule mismatch for {arm}/{step}")
            probes[str(step)] = probe
            probe_summary = probe_path.relative_to(RUN).as_posix()
            file_hashes[probe_summary] = sha256(probe_path)
            diagnostic_path = RUN / "evaluation" / f"arm_{arm}" / f"activation_diagnostics_step_{step:05d}.json"
            activation_data[str(step)] = read_json(diagnostic_path)
            file_hashes[diagnostic_path.relative_to(RUN).as_posix()] = sha256(diagnostic_path)
        full_path = RUN / "evaluation" / f"arm_{arm}" / "step_10000_full.json"
        full = read_json(full_path)
        if full["checkpoint_sha256"] != checkpoint_data["10000"]["model_sha256"] or full["case_count"] != 6:
            raise ValueError(f"Full diagnostic/checkpoint mismatch for arm {arm}")
        file_hashes[full_path.relative_to(RUN).as_posix()] = sha256(full_path)
        layer_audit = OUT / "posthoc_layer_gradient_audit.json"
        if layer_audit.is_file():
            file_hashes["../results/" + layer_audit.relative_to(ROOT / "results").as_posix()] = sha256(layer_audit)
        for name in ("training_summary.json", "checkpoint_telemetry.json", "training_episodes.jsonl",
                     "optimizer_telemetry.jsonl", "replay_action_trace.jsonl"):
            path = arm_dir / name
            if path.is_file():
                file_hashes[path.relative_to(RUN).as_posix()] = sha256(path)
        for path in sorted((arm_dir / "replay_samples").glob("*.npz")):
            file_hashes[path.relative_to(RUN).as_posix()] = sha256(path)
        arms[arm] = {
            "training": training,
            "checkpoints": checkpoint_data,
            "probes": probes,
            "activation_diagnostics": activation_data,
            "full_episode_10000": full,
        }
    paired = {}
    for step in STEPS:
        a = {x["geometry_id"]: x for x in arms["A"]["probes"][str(step)]["cases"]}
        b = {x["geometry_id"]: x for x in arms["B"]["probes"][str(step)]["cases"]}
        if set(a) != set(b):
            raise ValueError(f"Unpaired probe cases at {step}")
        paired[str(step)] = {
            "schedule_match": True,
            "route_progress_delta_B_minus_A": {
                "mean": float(np.mean([b[k]["route_progress_delta"] - a[k]["route_progress_delta"] for k in sorted(a)])),
                "per_case": {k: float(b[k]["route_progress_delta"] - a[k]["route_progress_delta"]) for k in sorted(a)},
            },
            "mean_speed_kmh_B_minus_A": float(np.mean([b[k]["mean_speed_kmh"] - a[k]["mean_speed_kmh"] for k in sorted(a)])),
            "stationary_fraction_B_minus_A": float(np.mean([b[k]["stationary_fraction_below_1kmh"] - a[k]["stationary_fraction_below_1kmh"] for k in sorted(a)])),
            "clean_successes_A_B": [arms["A"]["probes"][str(step)]["aggregate"]["clean_successes"],
                                    arms["B"]["probes"][str(step)]["aggregate"]["clean_successes"]],
            "safety_failures_A_B": [arms["A"]["probes"][str(step)]["aggregate"]["safety_failures"],
                                    arms["B"]["probes"][str(step)]["aggregate"]["safety_failures"]],
            "censored_A_B": [arms["A"]["probes"][str(step)]["aggregate"]["censored"],
                             arms["B"]["probes"][str(step)]["aggregate"]["censored"]],
        }
    full_a = {x["geometry_id"]: x for x in arms["A"]["full_episode_10000"]["cases"]}
    full_b = {x["geometry_id"]: x for x in arms["B"]["full_episode_10000"]["cases"]}
    if set(full_a) != set(full_b):
        raise ValueError("Full-episode diagnostic case schedules are not paired")
    paired_full = {
        "schedule_match": True,
        "route_progress_delta_B_minus_A_mean": float(np.mean([full_b[k]["route_progress_delta"] - full_a[k]["route_progress_delta"] for k in sorted(full_a)])),
        "mean_speed_kmh_B_minus_A_mean": float(np.mean([full_b[k]["mean_speed_kmh"] - full_a[k]["mean_speed_kmh"] for k in sorted(full_a)])),
        "stationary_fraction_B_minus_A_mean": float(np.mean([full_b[k]["stationary_fraction_below_1kmh"] - full_a[k]["stationary_fraction_below_1kmh"] for k in sorted(full_a)])),
        "outcome_A_B": {
            "A": arms["A"]["full_episode_10000"]["outcome_counts"],
            "B": arms["B"]["full_episode_10000"]["outcome_counts"],
        },
        "per_case": {
            k: {
                "environment_seed_A_B": [full_a[k]["environment_seed"], full_b[k]["environment_seed"]],
                "outcome_A_B": [full_a[k]["outcome"], full_b[k]["outcome"]],
                "route_progress_delta_B_minus_A": full_b[k]["route_progress_delta"] - full_a[k]["route_progress_delta"],
                "mean_speed_kmh_B_minus_A": full_b[k]["mean_speed_kmh"] - full_a[k]["mean_speed_kmh"],
            } for k in sorted(full_a)
        },
    }
    result = {
        "schema": "stage5_p2d_result_summary_v1", "status": "COMPLETE",
        "protocol": protocol["protocol"], "protocol_sha256": protocol["protocol_sha256"],
        "training_source_commit": run_manifest["source_commit"],
        "source_file_sha256": protocol["source_files_sha256"],
        "producer": protocol["producer"], "observation": protocol["observation"],
        "geometry": protocol["geometry"], "run_manifest": run_manifest,
        "arms": arms, "paired_probes_B_minus_A": paired, "paired_full_episodes_10000_B_minus_A": paired_full,
        "posthoc_layer_gradient_audit": {
            "path": "results/stage5/p2d_actor_activation_seed101_20261121/posthoc_layer_gradient_audit.json",
            "sha256": sha256(OUT / "posthoc_layer_gradient_audit.json") if (OUT / "posthoc_layer_gradient_audit.json").is_file() else None,
            "analysis_script_sha256": sha256(ROOT / "scripts/audit_stage5_p2d_actor_layer_gradients.py"),
        },
        "provenance": {
            "training_run_directory": str(RUN.relative_to(ROOT)).replace("\\", "/"),
            "official_validation_test_accessed": False, "benchmark_eligible": False,
            "all_probe_censoring_preserved": True,
            "file_sha256": file_hashes,
        },
        "scientific_interpretation": {
            "pre_registered_control": "A=ReLU actor/ReLU critics",
            "pre_registered_treatment": "B=LeakyReLU(0.01) actor/ReLU critics",
            "one_root_seed_limit": True,
            "clean_driving_established": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "result_summary.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    digest = sha256(result_path)
    (OUT / "result_summary.sha256").write_text(f"{digest}  result_summary.json\n", encoding="ascii")
    print(json.dumps({"result_summary": result_path.relative_to(ROOT).as_posix(), "sha256": digest,
                      "paired_10k": paired["10000"]}, indent=2))


if __name__ == "__main__":
    main()
