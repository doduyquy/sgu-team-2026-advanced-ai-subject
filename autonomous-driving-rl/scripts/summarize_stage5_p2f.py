"""Aggregate immutable P2-F run artifacts into a compact, hashed evidence summary."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage5_p2_experiment import runtime_versions

RUN_ID = "s5_p2f_start_distance_seed101_20261121_retry3"
RUN = ROOT / "runs/stage5_p2f" / RUN_ID
OUT = ROOT / "results/stage5/p2f_training_curriculum_20261121/result_summary.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    return float(np.mean(values)) if values else None


def case_map(rows, key="geometry_id"):
    return {row[key]: row for row in rows}


def paired(left_rows, right_rows, metrics, key="geometry_id"):
    left, right = case_map(left_rows, key), case_map(right_rows, key)
    if left.keys() != right.keys():
        raise ValueError(f"Paired schedule mismatch: {left.keys()} != {right.keys()}")
    rows = []
    for case_id in left:
        for identity in ("environment_seed", "geometry_sha256"):
            if left[case_id].get(identity) != right[case_id].get(identity):
                raise ValueError(f"Paired case identity mismatch for {case_id}: {identity}")
        row = {"case_id": case_id}
        for metric in metrics:
            a, b = left[case_id].get(metric), right[case_id].get(metric)
            row[f"A_{metric}"] = a
            row[f"B_{metric}"] = b
            row[f"B_minus_A_{metric}"] = (float(b) - float(a)) if a is not None and b is not None else None
        rows.append(row)
    means = {}
    for metric in metrics:
        diffs = [row[f"B_minus_A_{metric}"] for row in rows if row[f"B_minus_A_{metric}"] is not None]
        means[metric] = mean(diffs)
    return {"pair_count": len(rows), "mean_B_minus_A": means, "per_case": rows}


def summarize_episode(track: Path):
    payload = read(track)
    rows = payload["cases"]
    return {
        "case_count": len(rows), "clean_successes": payload["clean_successes"],
        "safety_failures": payload["safety_failures"], "timeouts": payload["timeouts"],
        "censored": payload["censored"],
        "mean_speed_kmh": mean([x["mean_speed_kmh"] for x in rows]),
        "mean_route_progress_delta": mean([x["route_progress_delta"] for x in rows]),
        "mean_stationary_fraction": mean([x["stationary_fraction_below_1kmh"] for x in rows]),
        "checkpoint_sha256": sorted({x["checkpoint_sha256"] for x in rows}),
        "result_sha256": sha(track),
        "cases": rows,
    }


def main():
    protocol_path = ROOT / "results/stage5/p2f_training_curriculum_20261121/protocol.json"
    protocol = read(protocol_path)
    run_manifest_path = RUN / "run_manifest.json"
    run_manifest = read(run_manifest_path)
    paired_training = read(RUN / "paired_training_summary.json")
    runtime_path = RUN / "RUNTIME_ENVIRONMENT.json"
    runtime_record = {
        "runtime_versions": runtime_versions(),
        "source_commit_from_training_manifest": run_manifest["source_commit"],
        "capture_note": "Captured from the same named conda environment after the completed run; training process used this environment throughout.",
    }
    if runtime_path.exists():
        if read(runtime_path) != runtime_record:
            raise FileExistsError(f"Refusing to replace runtime provenance: {runtime_path}")
    else:
        runtime_path.write_text(json.dumps(runtime_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    arms = {}
    for arm in ("A", "B"):
        train = read(RUN / f"arm_{arm}/training_summary.json")
        episodes = train["training_task_episodes"]
        checkpoint_rows = read(RUN / f"arm_{arm}/checkpoint_telemetry.json")
        diag = {step: read(RUN / f"evaluation/arm_{arm}/checkpoint_{step:05d}_diagnostics.json")
                for step in (0, 2000, 5000, 10000)}
        transfer_full = diag[10000]["original_task_full_cases"]
        curriculum_file = RUN / f"evaluation/arm_{arm}/step_10000_curriculum/curriculum_full_episodes.json"
        curriculum_full = read(curriculum_file)
        replay = []
        with (RUN / f"arm_{arm}/replay_action_trace.jsonl").open(encoding="utf-8") as handle:
            replay = [json.loads(line) for line in handle if line.strip()]
        replay_categories = {
            "transitions": len(replay),
            "stationary_speed_below_1kmh": sum(float(x["speed_km_h"]) < 1.0 for x in replay),
            "moving_speed_at_least_1kmh": sum(float(x["speed_km_h"]) >= 1.0 for x in replay),
            "positive_progress": sum(float(x["delta_route"]) > 0 for x in replay),
            "negative_progress": sum(float(x["delta_route"]) < 0 for x in replay),
            "safety_terminal_transitions": sum(bool(x.get("terminated")) and not bool((x.get("outcome") or {}).get("clean_success", False)) for x in replay),
            "clean_terminal_transitions": sum(bool((x.get("outcome") or {}).get("clean_success", False)) for x in replay),
            "turn_context_sample": None,
            "turn_context_definition": "Unavailable in per-step trace; not inferred from an undocumented threshold",
        }
        sample_path = RUN / f"arm_{arm}/replay_samples/observations_step_10000.npz"
        sample = np.load(sample_path, allow_pickle=False)
        obs = sample["observations"]
        waypoints = obs[:, 325:385].reshape(-1, 20, 3)
        masks = obs[:, 385:405] > 0.5
        # Diagnostic-only operational definition: at least one valid waypoint
        # has >=10 degrees absolute relative heading, after undoing /pi scaling.
        heading_turn = np.any((np.abs(waypoints[:, :, 2] * np.pi) >= np.deg2rad(10)) & masks, axis=1)
        replay_categories["turn_context_sample"] = {
            "sampling": "1024 evenly-spaced replay indices saved at step 10k",
            "sample_count": int(len(obs)), "turn_context_count": int(heading_turn.sum()),
            "turn_context_fraction": float(heading_turn.mean()),
            "definition": "at least one valid Task waypoint relative heading magnitude >=10 degrees",
            "sample_sha256": sha(sample_path),
        }
        arms[arm] = {
            "training": {
                "status": train["status"], "actual_environment_steps": train["actual_environment_steps"],
                "optimizer_updates": train["optimizer_updates"], "completed_episodes": train["completed_episodes"],
                "episode_outcomes": train["episode_outcomes"], "training_episode_clean_successes": sum(x["clean_success"] for x in episodes),
                "training_episode_safety_failures": sum(x["outcome"] == "OUT_OF_ROAD" or x["outcome"] == "SAFETY_FAILURE" for x in episodes),
                "unique_geometries_visited": train["unique_geometries_visited"],
                "geometry_step_counts": train["geometry_step_counts"],
                "wall_time_seconds": train["wall_time_seconds"],
                "throughput_steps_per_second": train["throughput_steps_per_second"],
                "episodes": episodes,
                "replay_action_distribution": train["replay_action_distribution"],
                "replay_transition_categories": replay_categories,
            },
            "checkpoints": {
                str(step): {
                    "checkpoint_sha256": diag[step]["checkpoint_sha256"],
                    "training_step": checkpoint_rows[[x["step"] for x in checkpoint_rows].index(step)]["step"],
                    "optimizer_updates": checkpoint_rows[[x["step"] for x in checkpoint_rows].index(step)]["optimizer_updates"],
                    "alpha": checkpoint_rows[[x["step"] for x in checkpoint_rows].index(step)]["alpha"],
                    "actor_deterministic_action_std_across_fixed_anchors": diag[step]["actor_fixed_anchor_sensitivity"]["deterministic_action_std"],
                    "original_task_probe": diag[step]["original_task_probe"],
                } for step in (0, 2000, 5000, 10000)
            },
            "original_task_full_10k": {
                "aggregate": diag[10000]["original_task_full"],
                "cases": transfer_full,
                "paired_vs_other_arm": None,
            },
            "curriculum_full_10k": {
                "case_count": curriculum_full["case_count"], "clean_successes": curriculum_full["clean_successes"],
                "safety_failures": curriculum_full["safety_failures"], "timeouts": curriculum_full["timeouts"],
                "censored": curriculum_full["censored"],
                "mean_speed_kmh": mean([x["mean_speed_kmh"] for x in curriculum_full["cases"]]),
                "mean_route_progress_delta": mean([x["route_progress_delta"] for x in curriculum_full["cases"]]),
                "mean_stationary_fraction": mean([x["stationary_fraction_below_1kmh"] for x in curriculum_full["cases"]]),
                "cases": curriculum_full["cases"], "result_sha256": sha(curriculum_file),
            },
        }
    a_full = arms["A"]["original_task_full_10k"]["cases"]
    b_full = arms["B"]["original_task_full_10k"]["cases"]
    original_paired = paired(a_full, b_full, ["route_progress_delta", "mean_speed_kmh", "stationary_fraction_below_1kmh", "clean_success", "safety_failure", "timeout"])
    a_cur = arms["A"]["curriculum_full_10k"]["cases"]
    b_cur = arms["B"]["curriculum_full_10k"]["cases"]
    for row in a_cur:
        row["pair_key"] = f"{row['geometry_id']}|{row['curriculum_stage_id']}"
    for row in b_cur:
        row["pair_key"] = f"{row['geometry_id']}|{row['curriculum_stage_id']}"
    curriculum_paired = paired(a_cur, b_cur, ["route_progress_delta", "mean_speed_kmh", "stationary_fraction_below_1kmh", "clean_success", "safety_failure", "timeout"], key="pair_key")
    summary = {
        "experiment": protocol["protocol"], "protocol_revision": protocol["protocol_revision"],
        "protocol_sha256": protocol["protocol_sha256"], "protocol_file_sha256": sha(protocol_path),
        "source_commit": run_manifest["source_commit"], "current_code_after_training_differs_only_for_evaluation_adapter": True,
        "training_run_id": RUN_ID, "training_status": "COMPLETE", "evaluation_status": "COMPLETE",
        "evaluation_adapter": read(RUN / "evaluation/EVALUATION_ADAPTER.json"),
        "decision": "CURRICULUM_NO_IMPROVEMENT",
        "decision_rationale": "No clean success in either arm. B's curriculum episodes and curriculum evaluation remained stationary TIMEOUTs; original full-route transfer was stationary TIMEOUT. A moved more but incurred safety failures in all curriculum and full-route episodes. No useful clean trajectory support or driving skill emerged.",
        "training_scope": "TRAIN/PILOT_TRAIN only", "evaluation_scope": "TRAIN/PILOT_DIAGNOSTIC only",
        "validation_test_accessed": False, "benchmark_eligible": False,
        "feasibility_evidence_sha256": protocol["feasibility"]["evidence_sha256"],
        "initialization": run_manifest["initialization"], "arms": arms,
        "paired_original_full_route_10k": original_paired,
        "paired_curriculum_10k": curriculum_paired,
        "limitations": [
            "One training root seed; this pilot cannot establish generalization or arm superiority.",
            "Only 7 A and 8 B training episodes completed; though both visited all eight geometries, per-geometry episode coverage is sparse.",
            "MetaDrive route_completion denominator is recomputed on the suffix after curriculum spawn; route-progress fractions are not directly interpretable as equal physical distance across arms.",
            "Curriculum evaluation episodes start at 60m or 120m remaining and are not original full-route performance or benchmark results.",
            "P2-F training telemetry did not persist turn-context for every transition; turn context was estimated on a fixed 1024-transition sample using valid Task waypoint heading >=10 degrees.",
            "pytest was unavailable in the metadrive_sb3 environment; Python compile, protocol checks, reset feasibility and live simulator smoke were used instead.",
        ],
        "artifacts": {},
    }
    paths = [RUN / "run_manifest.json", RUN / "paired_training_summary.json", runtime_path]
    for arm in ("A", "B"):
        paths += [RUN / f"arm_{arm}/training_summary.json", RUN / f"arm_{arm}/checkpoint_telemetry.json",
                  RUN / f"arm_{arm}/replay_action_trace.jsonl", RUN / f"arm_{arm}/training_task_episodes.jsonl",
                  RUN / f"arm_{arm}/checkpoints/step_010000/metadata.json",
                  RUN / f"arm_{arm}/checkpoints/step_010000/model.zip",
                  RUN / f"arm_{arm}/checkpoints/step_010000/replay_buffer.pkl",
                  RUN / f"evaluation/arm_{arm}/checkpoint_10000_diagnostics.json",
                  RUN / f"evaluation/arm_{arm}/step_10000_transfer/step_10000_full.json",
                  RUN / f"evaluation/arm_{arm}/step_10000_curriculum/curriculum_full_episodes.json"]
        for step in (0, 2000, 5000, 10000):
            paths += [RUN / f"arm_{arm}/checkpoints/step_{step:06d}/metadata.json",
                      RUN / f"arm_{arm}/checkpoints/step_{step:06d}/CHECKSUMS.sha256"]
    summary["artifacts"] = {str(path.relative_to(ROOT)).replace("\\", "/"): {"sha256": sha(path), "bytes": path.stat().st_size} for path in paths}
    summary["summary_generator_sha256"] = sha(Path(__file__).resolve())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (OUT.with_suffix(".sha256")).write_text(f"{sha(OUT)}  {OUT.name}\n", encoding="ascii")
    print(json.dumps({"status": summary["evaluation_status"], "decision": summary["decision"],
                      "path": str(OUT.relative_to(ROOT)), "sha256": sha(OUT),
                      "training_steps": {a: summary["arms"][a]["training"]["actual_environment_steps"] for a in "AB"},
                      "paired_original": original_paired["mean_B_minus_A"],
                      "paired_curriculum": curriculum_paired["mean_B_minus_A"]}, indent=2))


if __name__ == "__main__":
    main()
