"""Create compact, checksum-bound P2-H results from frozen local artifacts."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUN = ROOT / "runs/stage5_p2h/s5_p2h_terminal_replay_seed101_20261121"
OUT = ROOT / "results/stage5/p2h_terminal_replay_seed101_20261121"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def compact_eval(path: Path) -> dict:
    data = read(path)
    cases = data["cases"]
    return {"result_path": path.relative_to(ROOT).as_posix(), "result_sha256": sha(path),
            "evaluation_mode": data["evaluation_mode"], "evaluation_cap": data["evaluation_cap"],
            "outcome_counts": data["outcome_counts"], "aggregate": data["aggregate"],
            "actor_responsiveness": data["actor_responsiveness"],
            "cases": [{"geometry_id": x["geometry_id"], "environment_seed": x["environment_seed"],
                       "outcome": x["outcome"], "clean_success": x["clean_success"],
                       "safety_failure": x["safety_failure"], "timeout": x["timeout"],
                       "probe_censored": x["probe_censored"],
                       "route_progress_delta": x["route_progress_delta"],
                       "final_route_completion": x.get("final_route_completion"),
                       "mean_speed_kmh": x["mean_speed_kmh"],
                       "stationary_fraction_below_1kmh": x["stationary_fraction_below_1kmh"],
                       "longest_moving_period_steps": x.get("longest_moving_period_steps"),
                       "throttle_mean": x.get("throttle_mean"), "steering_std": x.get("steering_std")}
                      for x in cases]}


def main() -> None:
    protocol = read(OUT / "protocol.json")
    run_manifest = read(RUN / "run_manifest.json")
    if run_manifest["status"] != "COMPLETE":
        raise RuntimeError("Refusing to summarize an incomplete A/B run")
    from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER
    from src.training.stage5.actor_activation_ablation import ACTOR_LEAKY_RELU_001_V1
    from src.training.stage5.checkpoint import verify_checkpoint_integrity
    from src.training.stage5.observation_encoder import Stage5ObservationEncoder
    encoder = Stage5ObservationEncoder(core_only=False)

    arm_data = {}
    for arm in "AB":
        training = read(RUN / f"arm_{arm}/training_summary.json")
        episode_rows = [json.loads(line) for line in (RUN / f"arm_{arm}/training_episodes.jsonl").read_text(encoding="utf-8").splitlines() if line]
        training_checkpoints = {int(row["step"]): row for row in training["checkpoints"]}
        checkpoints = []
        for step in (0, 2000, 5000, 10000):
            cp_path = RUN / f"arm_{arm}/checkpoints/step_{step:06d}"
            metadata = verify_checkpoint_integrity(cp_path,
                expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
                task_context_producer_id=REFERENCE_TASK_PRODUCER)
            if metadata.get("actor_activation") != ACTOR_LEAKY_RELU_001_V1:
                raise RuntimeError(f"Checkpoint actor activation mismatch at {arm}/{step}")
            eval_root = RUN / f"evaluation/arm_{arm}"
            probe = compact_eval(eval_root / f"step_{step:05d}_probe.json")
            full = compact_eval(eval_root / f"step_{step:05d}_full.json") if step == 10000 else None
            checkpoints.append({"step": step, "model_sha256": metadata["model_sha256"],
                "metadata_sha256": sha(cp_path / "metadata.json"),
                "model_size_bytes": (cp_path / "model.zip").stat().st_size,
                "checkpoint_identity": metadata.get("checkpoint_identity"),
                "replay_buffer_sha256": metadata.get("replay_buffer_sha256"),
                "replay_buffer_size_bytes": (cp_path / "replay_buffer.pkl").stat().st_size if (cp_path / "replay_buffer.pkl").is_file() else None,
                "optimizer_updates": int(training_checkpoints[step]["optimizer_updates"]),
                "alpha": float(training_checkpoints[step]["alpha"]),
                "probe": probe, "full_episode": full})
        arm_data[arm] = {"training": training, "training_episodes": episode_rows,
                         "checkpoint_evaluations": checkpoints}

    full_a = arm_data["A"]["checkpoint_evaluations"][-1]["full_episode"]["cases"]
    full_b = arm_data["B"]["checkpoint_evaluations"][-1]["full_episode"]["cases"]
    if [(x["geometry_id"], x["environment_seed"]) for x in full_a] != [(x["geometry_id"], x["environment_seed"]) for x in full_b]:
        raise RuntimeError("A/B full-evaluation case schedule is not paired")
    paired = []
    for a, b in zip(full_a, full_b):
        paired.append({"geometry_id": a["geometry_id"], "environment_seed": a["environment_seed"],
                       "control_outcome": a["outcome"], "treatment_outcome": b["outcome"],
                       "route_progress_delta_B_minus_A": b["route_progress_delta"] - a["route_progress_delta"],
                       "mean_speed_kmh_B_minus_A": b["mean_speed_kmh"] - a["mean_speed_kmh"],
                       "stationary_fraction_B_minus_A": b["stationary_fraction_below_1kmh"] - a["stationary_fraction_below_1kmh"]})

    bterm = read(RUN / "arm_B/terminal_inventory.json")
    b_final = bterm[-1]
    a_outcomes = [row["outcome"] for row in arm_data["A"]["training_episodes"]]
    a_counts = {"none_or_nonterminal": 10000 - len(a_outcomes), "safety_terminal": sum(x not in ("TIMEOUT", "SUCCESS") for x in a_outcomes),
                "clean_terminal": sum(x == "SUCCESS" for x in a_outcomes),
                "timeout_truncation": sum(x == "TIMEOUT" for x in a_outcomes),
                "true_terminal_mask_count": sum(x not in ("TIMEOUT", None) for x in a_outcomes)}
    summary = {
        "experiment": "Stage5 P2-H terminal-aware prioritized critic replay A/B",
        "status": "P2H_INCONCLUSIVE",
        "status_reason": [
            "Arm B observed zero eligible clean or safety terminal transitions at every checkpoint and therefore used uniform fallback for the entire run.",
            "The implementation used an independent uniform actor/alpha batch in B while control A retained SB3's shared uniform batch; this changes critic/actor minibatch coupling, so post hoc A/B differences cannot be attributed to terminal prioritization.",
        ],
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_path": "results/stage5/p2h_terminal_replay_seed101_20261121/protocol.json",
        "source_commit": run_manifest["source_commit"],
        "initial_parameter_fingerprint": run_manifest["initialization"]["policy_state_fingerprint"],
        "initial_state_file_sha256": run_manifest["initialization"]["sha256"],
        "runtime": arm_data["A"]["training"]["runtime_versions"],
        "arms": arm_data,
        "terminal_exposure": {"A": a_counts,
                              "B": b_final["counts"],
                              "B_critic_draws_total": b_final["sampled"]["cumulative_sampler"]["draws"] if b_final["sampled"] else 0,
                              "B_critic_draws_last_update_call": b_final["sampled"]["critic_draws"] if b_final["sampled"] else 0,
                              "B_terminal_priority_draws": b_final["sampled"]["cumulative_sampler"]["terminal_draws"] if b_final["sampled"] else 0,
                              "B_timeout_uniform_draws": b_final["sampled"]["cumulative_sampler"]["timeout_draws"] if b_final["sampled"] else 0,
                              "B_priority_candidates_by_checkpoint": [x["counts"]["true_terminal_mask_count"] for x in bterm]},
        "paired_full_episode_10k": paired,
        "paired_mean_B_minus_A": {key: float(np.mean([row[key] for row in paired])) for key in (
            "route_progress_delta_B_minus_A", "mean_speed_kmh_B_minus_A", "stationary_fraction_B_minus_A")},
        "scientific_conclusion": {
            "clean_successes_A": 0, "clean_successes_B": 0,
            "full_episode_safety_failures_A": 6, "full_episode_safety_failures_B": 6,
            "driving_skill_demonstrated": False,
            "terminal_prioritization_hypothesis_tested": False,
            "actor_observation_sensitivity_collapsed_by_10k": True,
            "inference": "The 10k agents did not demonstrate clean driving. The terminal replay mechanism was not activated because online replay contained only timeouts and nonterminal transitions. P2-H cannot determine whether terminal-aware replay would help when useful terminal records exist.",
            "benchmark_eligible": False,
            "validation_test_accessed": False,
        },
    }
    target = OUT / "result_summary.json"
    target.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (OUT / "result_summary.sha256").write_text(f"{sha(target)}  result_summary.json\n", encoding="ascii")
    print(json.dumps({"result_summary": target.as_posix(), "sha256": sha(target),
                      "status": summary["status"], "paired_mean": summary["paired_mean_B_minus_A"]}, indent=2))


if __name__ == "__main__":
    main()
