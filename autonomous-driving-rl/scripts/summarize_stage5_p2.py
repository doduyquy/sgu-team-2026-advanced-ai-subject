"""Build a compact, hash-bound summary from the frozen P2 run and evaluations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs/stage5_p2/s5_p2_correlated_exploration_seed101_20261121_retry1"
DEFAULT_PROTOCOL = ROOT / "results/stage5/p2_correlated_exploration_seed101_20261120/protocol.json"
DEFAULT_OUT = ROOT / "results/stage5/p2_correlated_exploration_seed101_20261121"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    run_manifest_path = args.run_dir / "run_manifest.json"
    protocol = read_json(args.protocol)
    run = read_json(run_manifest_path)
    if run.get("status") != "COMPLETE" or len(run.get("completed_arms", [])) != 2:
        raise SystemExit("P2 run is not a complete two-arm experiment")
    if run.get("protocol_sha256") != protocol.get("protocol_sha256"):
        raise SystemExit("Run/protocol fingerprint mismatch")

    evaluations: dict[str, dict[str, Any]] = {"A": {}, "B": {}}
    evaluation_hashes: dict[str, str] = {}
    for arm in ("A", "B"):
        folder = args.run_dir / "evaluation" / f"arm_{arm}"
        for step in (0, 2000, 5000, 10000):
            kind = "full" if step == 10000 else "probe"
            path = folder / f"step_{step:05d}_{kind}.json"
            if not path.is_file():
                raise SystemExit(f"Missing frozen evaluation: {path}")
            data = read_json(path)
            if data.get("benchmark_eligible") is not False or data.get("official_validation_test_accessed") is not False:
                raise SystemExit(f"Evaluation eligibility/holdout invariant failed: {path}")
            if data.get("protocol_sha256") != protocol.get("protocol_sha256"):
                raise SystemExit(f"Evaluation protocol mismatch: {path}")
            if len(data.get("cases", [])) != 6:
                raise SystemExit(f"Expected six fixed diagnostic cases: {path}")
            evaluations[arm][str(step)] = data
            evaluation_hashes[path.relative_to(ROOT).as_posix()] = sha256(path)

    checkpoints: dict[str, dict[str, Any]] = {}
    for arm_run in run["completed_arms"]:
        checkpoints[arm_run["arm"]] = {
            str(item["step"]): {
                "model_sha256": item["model_sha256"],
                "metadata_sha256": item["metadata_sha256"],
                "replay_buffer_sha256": item.get("replay_buffer_sha256"),
                "alpha": item["alpha"],
                "optimizer_updates": item["optimizer_updates"],
                "completed_episodes": item["completed_episodes"],
                "replay_size": item["replay_size"],
            }
            for item in arm_run["checkpoints"]
        }

    paired: dict[str, list[dict[str, Any]]] = {}
    for step in (0, 2000, 5000, 10000):
        rows_a = { (x["geometry_id"], x["environment_seed"]): x for x in evaluations["A"][str(step)]["cases"] }
        rows_b = { (x["geometry_id"], x["environment_seed"]): x for x in evaluations["B"][str(step)]["cases"] }
        if rows_a.keys() != rows_b.keys():
            raise SystemExit(f"Unpaired diagnostic schedule at step {step}")
        paired[str(step)] = []
        for key in sorted(rows_a):
            a, b = rows_a[key], rows_b[key]
            paired[str(step)].append({
                "geometry_id": key[0], "environment_seed": key[1],
                "A_outcome": a["outcome"], "B_outcome": b["outcome"],
                "A_route_progress_delta": a["route_progress_delta"],
                "B_route_progress_delta": b["route_progress_delta"],
                "B_minus_A_route_progress_delta": b["route_progress_delta"] - a["route_progress_delta"],
                "A_mean_speed_kmh": a["mean_speed_kmh"], "B_mean_speed_kmh": b["mean_speed_kmh"],
                "A_stationary_fraction": a["stationary_fraction_below_1kmh"],
                "B_stationary_fraction": b["stationary_fraction_below_1kmh"],
                "A_probe_censored": a["probe_censored"], "B_probe_censored": b["probe_censored"],
            })

    arm_summary = {}
    for arm_run in run["completed_arms"]:
        arm = arm_run["arm"]
        action = arm_run["replay_action_distribution"]
        arm_summary[arm] = {
            "status": arm_run["status"], "actual_environment_steps": arm_run["actual_environment_steps"],
            "optimizer_updates": arm_run["optimizer_updates"], "completed_episodes": arm_run["completed_episodes"],
            "episode_outcomes": arm_run["episode_outcomes"], "partial_episode_at_end": arm_run["partial_episode_at_end"],
            "geometry_step_counts": arm_run["geometry_step_counts"], "unique_geometries_visited": arm_run["unique_geometries_visited"],
            "wall_time_seconds": arm_run["wall_time_seconds"],
            "throughput_steps_per_second": arm_run["throughput_steps_per_second"],
            "peak_sampled_rss_mb": max(x["rss_mb"] for x in arm_run["ram_samples"]),
            "action_mismatch_count": arm_run["action_mismatch_count"],
            "replay_action_distribution": action,
            "initial_policy_fingerprint": arm_run["initial_policy_fingerprint"],
            "runtime_versions": arm_run["runtime_versions"],
        }

    summary = {
        "experiment_id": run["run_id"], "status": "COMPLETE", "benchmark_eligible": False,
        "official_validation_test_accessed": False,
        "source_commit": run["training_source_commit"],
        "protocol_sha256": run["protocol_sha256"],
        "run_manifest_sha256": sha256(run_manifest_path),
        "protocol_file_sha256": sha256(args.protocol),
        "initialization_sha256": sha256(args.run_dir / "shared_initial/initial_state.pt"),
        "initial_policy_fingerprint": run["completed_arms"][0]["initial_policy_fingerprint"],
        "producer": protocol["producer"], "observation": protocol["observation"],
        "action": protocol["action"], "training_geometry": protocol["geometry"]["training_pool"],
        "diagnostic_geometry": protocol["geometry"]["diagnostic_pool"],
        "arms": arm_summary, "checkpoints": checkpoints,
        "evaluation": {
            arm: {
                step: {
                    "evaluation_file_sha256": evaluation_hashes[
                        (args.run_dir / "evaluation" / f"arm_{arm}" / f"step_{int(step):05d}_{'full' if int(step) == 10000 else 'probe'}.json").relative_to(ROOT).as_posix()
                    ],
                    "evaluation_cap": result["evaluation_cap"],
                    "aggregate": result["aggregate"],
                    "actor_responsiveness": result["actor_responsiveness"],
                    "sampled_observations_sha256": result["sampled_observations_sha256"],
                    "cases": result["cases"],
                }
                for step, result in evaluations[arm].items()
            }
            for arm in ("A", "B")
        },
        "paired_cases": paired,
        "evidence_file_sha256": evaluation_hashes,
        "interpretation": {
            "temporal_noise_autocorrelation_confirmed": True,
            "deterministic_policy_improvement_confirmed": False,
            "clean_success_observed": False,
            "root_cause_identified": False,
            "decision": "NO_IMPROVEMENT",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "result_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    digest = sha256(out)
    (args.output_dir / "result_summary.sha256").write_text(f"{digest}  result_summary.json\n", encoding="ascii")
    print(f"summary={out.relative_to(ROOT).as_posix()}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
