"""Validate and summarize completed S5.5 training/evaluation artifacts."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.stage5.checkpoint import verify_checkpoint_integrity


RUN_ID = "s5_5_meter_reward_seed101_20261120"
RESULTS = ROOT / "results" / "stage5" / RUN_ID
RUNS = ROOT / "runs" / "stage5_meter_reward" / RUN_ID
ARMS = {"A": "V1", "B": "SCALE", "C": "METER"}
CHECKPOINT_STEPS = (0, 2000, 5000, 10000, 30000)
FULL_EPISODE_STEPS = {10000, 30000}
PAIR_METRICS = (
    "route_progress_delta",
    "mean_speed_km_h",
    "stationary_fraction_speed_lt_1_km_h",
    "clean_success",
    "safety_failure",
    "timeout",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def checkpoint_fingerprints() -> tuple[dict[str, Any], list[str]]:
    bundles: dict[str, Any] = {}
    initial_fingerprints: list[str] = []
    for arm, variant in ARMS.items():
        arm_bundles: dict[str, Any] = {}
        for step in CHECKPOINT_STEPS:
            folder = RUNS / "arms" / arm / "checkpoints" / f"step_{step:06d}"
            verified = verify_checkpoint_integrity(folder)
            metadata = read_json(folder / "metadata.json")
            if metadata.get("total_timesteps") != step:
                raise ValueError(f"Checkpoint step mismatch: {folder}")
            if metadata.get("git_commit") != PROTOCOL["source_commit"]:
                raise ValueError(f"Training source SHA mismatch: {folder}")
            if metadata.get("training_config", {}).get("protocol_sha256") != PROTOCOL["protocol_sha256"]:
                raise ValueError(f"Checkpoint protocol mismatch: {folder}")
            if metadata.get("training_config", {}).get("reward_variant") != variant:
                raise ValueError(f"Checkpoint reward arm mismatch: {folder}")
            if step == 0:
                initial_fingerprints.append(
                    read_json(RUNS / "arms" / arm / "initial_state.json")["weights_fingerprint"]
                )
            arm_bundles[str(step)] = {
                "model_sha256": verified["model_sha256"],
                "integrity": "PASS",
                "training_steps": metadata["total_timesteps"],
                "encoder_fingerprint": verified["encoder_fingerprint"],
                "observation_transform_fingerprint": verified.get("observation_transform_fingerprint"),
                "source_commit": metadata["git_commit"],
            }
        bundles[arm] = arm_bundles
    if len(set(initial_fingerprints)) != 1:
        raise ValueError("Fresh initial actor/critic/entropy state differs across arms")
    return bundles, initial_fingerprints


def training_summaries() -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    expected_geometry = {row["geometry_id"] for row in PROTOCOL["training_geometry_pool"]}
    initial_fingerprints = set()
    for arm, variant in ARMS.items():
        data = read_json(RUNS / "arms" / arm / "training_summary.json")
        if data.get("status") != "COMPLETE" or data.get("actual_training_steps") != 30000:
            raise ValueError(f"Arm {arm} did not complete the registered 30k steps")
        if data.get("optimizer_updates") != 29000:
            raise ValueError(f"Arm {arm} optimizer update count is unexpected")
        if data.get("source_commit") != PROTOCOL["source_commit"]:
            raise ValueError(f"Arm {arm} source SHA does not match the frozen protocol")
        if data.get("protocol_sha256") != PROTOCOL["protocol_sha256"]:
            raise ValueError(f"Arm {arm} protocol identity mismatch")
        seen = set(data.get("geometry_steps", {}))
        if seen != expected_geometry:
            raise ValueError(f"Arm {arm} did not visit exactly the registered eight training geometries")
        initial_fingerprints.add(data["initial_weights_fingerprint"])
        episodes = data.get("episodes", [])
        outcome_counts: dict[str, int] = defaultdict(int)
        for episode in episodes:
            outcome_counts[str(episode.get("outcome", "UNKNOWN"))] += 1
        checkpoint_windows = {}
        for checkpoint in data.get("checkpoint_summaries", []):
            summary = checkpoint["summary"]
            checkpoint_windows[str(checkpoint["training_step"])] = {
                key: summary.get(key)
                for key in (
                    "window_mean_speed_km_h",
                    "window_stationary_fraction_speed_lt_1_km_h",
                    "window_throttle_mean",
                    "window_throttle_positive_fraction",
                    "window_throttle_negative_fraction",
                    "window_steering_std",
                    "max_positive_throttle_run_steps",
                    "max_negative_throttle_run_steps",
                    "max_sustained_moving_run_steps_speed_ge_1",
                    "direct_throttle_brake_sign_switches",
                    "alpha",
                    "actor_loss",
                    "critic_loss",
                    "ent_coef_loss",
                    "ram_last_sample_mb",
                )
            }
        summaries[arm] = {
            "reward_variant": variant,
            "status": data["status"],
            "actual_environment_steps": data["actual_training_steps"],
            "optimizer_updates": data["optimizer_updates"],
            "completed_episodes": data["completed_episodes"],
            "training_episode_outcomes": dict(sorted(outcome_counts.items())),
            "unique_training_geometry_count": len(seen),
            "geometry_steps": data["geometry_steps"],
            "completed_episodes_per_geometry": data["geometry_completed_episodes"],
            "wall_time_seconds": data["wall_time_s"],
            "throughput_steps_per_second": data["actual_training_steps"] / data["wall_time_s"],
            "ram_samples": data.get("ram_samples", []),
            "runtime_versions": data.get("runtime_versions"),
            "initial_weights_fingerprint": data["initial_weights_fingerprint"],
            "checkpoint_windows": checkpoint_windows,
        }
    if len(initial_fingerprints) != 1:
        raise ValueError("Initial weight fingerprint differs across the three arms")
    return summaries


def evaluate_summary() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = RESULTS / "evaluation_summary.json"
    eval_summary = read_json(summary_path)
    if eval_summary.get("benchmark_eligible") is not False or eval_summary.get("holdouts_accessed") is not False:
        raise ValueError("Evaluation summary must remain non-eligible and holdout-free")
    evaluation_commit = str(eval_summary.get("evaluation_source_commit", ""))
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", evaluation_commit, "HEAD"], cwd=ROOT, check=False
    )
    if ancestor.returncode != 0:
        raise ValueError("Evaluator source revision is not an ancestor of the analyzed source")
    git_root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True).strip())
    evaluator_path = ROOT / "scripts" / "evaluate_stage5_s55.py"
    relative_evaluator_path = evaluator_path.relative_to(git_root).as_posix()
    recorded_blob = subprocess.check_output(
        ["git", "rev-parse", f"{evaluation_commit}:{relative_evaluator_path}"], cwd=ROOT, text=True
    ).strip()
    current_blob = subprocess.check_output(["git", "hash-object", str(evaluator_path)], cwd=ROOT, text=True).strip()
    if recorded_blob != current_blob:
        raise ValueError("The evaluator source file changed after its recorded evaluation revision")
    if eval_summary.get("protocol_sha256") != PROTOCOL["protocol_sha256"]:
        raise ValueError("Evaluation protocol identity mismatch")
    records_path = ROOT / eval_summary["per_episode_path"]
    if sha256(records_path) != eval_summary["per_episode_sha256"]:
        raise ValueError("Per-episode evaluation file SHA-256 mismatch")
    records = read_jsonl_gz(records_path)
    for arm in ARMS:
        for step in CHECKPOINT_STEPS:
            rows = [r for r in records if r["arm"] == arm and r["checkpoint_step"] == step]
            probes = [r for r in rows if r["probe"]]
            full = [r for r in rows if not r["probe"]]
            if len(probes) != 6:
                raise ValueError(f"Expected six fixed probes for {arm}/{step}, got {len(probes)}")
            if len(full) != (6 if step in FULL_EPISODE_STEPS else 0):
                raise ValueError(f"Full episode schedule mismatch for {arm}/{step}")
            if any(r["split"] != "TRAIN" for r in rows):
                raise ValueError("Non-TRAIN record found in development evaluation")

    compact_checkpoints = []
    for item in eval_summary["checkpoint_summaries"]:
        compact_checkpoints.append({
            key: item[key]
            for key in (
                "arm", "checkpoint_step", "checkpoint_model_sha256", "checkpoint_integrity", "source_commit",
                "probe_cases", "probe_censored_cases", "probe_clean_successes", "probe_safety_failures",
                "probe_mean_speed_km_h", "probe_mean_route_progress_delta", "probe_mean_stationary_fraction",
                "probe_metrics_overall", "probe_metrics_by_tier", "full_episode_count", "full_clean_successes",
                "full_safety_failures", "full_timeouts", "full_mean_final_route_completion",
                "full_mean_route_progress_delta", "full_mean_speed_km_h", "full_mean_stationary_fraction",
                "full_metrics_overall", "full_metrics_by_tier", "actor_responsiveness",
                "actor_responsiveness_on_own_probe_states", "own_probe_observation_feature_summary",
            )
            if key in item
        })
    return {"summary": eval_summary, "checkpoint_metrics": compact_checkpoints}, records


def paired_differences(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, bool], dict[tuple[str, int], dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for record in records:
        schedule = (int(record["checkpoint_step"]), bool(record["probe"]))
        case = (str(record["geometry_id"]), int(record["environment_seed"]))
        grouped[schedule][case][str(record["arm"])] = record

    output: dict[str, Any] = {}
    for (step, is_probe), cases in sorted(grouped.items()):
        expected_arms = set(ARMS)
        if any(set(row) != expected_arms for row in cases.values()):
            raise ValueError(f"Unpaired arm records at checkpoint {step}, probe={is_probe}")
        per_pair = {}
        for left, right in itertools.combinations(ARMS, 2):
            case_differences = []
            for (geometry_id, env_seed), rows in sorted(cases.items()):
                a, b = rows[left], rows[right]
                differences = {}
                for metric in PAIR_METRICS:
                    av, bv = a[metric], b[metric]
                    differences[metric] = (int(bv) - int(av)) if isinstance(av, bool) else float(bv) - float(av)
                case_differences.append({
                    "geometry_id": geometry_id,
                    "environment_seed": env_seed,
                    "left_arm": left,
                    "right_arm": right,
                    "right_minus_left": differences,
                })
            aggregate = {}
            for metric in PAIR_METRICS:
                aggregate[metric] = mean([float(x["right_minus_left"][metric]) for x in case_differences])
            per_pair[f"{right}_minus_{left}"] = {
                "paired_case_count": len(case_differences),
                "mean_right_minus_left": aggregate,
                "per_case": case_differences,
            }
        output[f"step_{step}_{'probe' if is_probe else 'full_episode'}"] = per_pair
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULTS / "analysis_summary.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite analysis output: {args.output}")

    global PROTOCOL
    PROTOCOL = read_json(RESULTS / "protocol.json")
    preflight = read_json(RESULTS / "reward_preflight.json")
    if preflight["protocol_sha256"] != PROTOCOL["protocol_sha256"] or preflight["preflight"]["status"] != "PASS":
        raise ValueError("Reward preflight identity/status mismatch")
    if preflight["source_commit"] != PROTOCOL["source_commit"]:
        raise ValueError("Reward preflight source SHA mismatch")

    checkpoints, initial_fingerprints = checkpoint_fingerprints()
    training = training_summaries()
    evaluation, records = evaluate_summary()
    output = {
        "status": "COMPLETE",
        "protocol_sha256": PROTOCOL["protocol_sha256"],
        "training_source_commit": PROTOCOL["source_commit"],
        "evaluation_source_commit": evaluation["summary"]["evaluation_source_commit"],
        "analysis_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "initial_weights_fingerprint_shared_across_arms": initial_fingerprints[0],
        "checkpoint_bundles": checkpoints,
        "training_arms": training,
        "evaluation_checkpoint_metrics": evaluation["checkpoint_metrics"],
        "paired_differences": paired_differences(records),
        "evaluation_artifacts": {
            "per_episode_path": evaluation["summary"]["per_episode_path"],
            "per_episode_sha256": evaluation["summary"]["per_episode_sha256"],
            "policy_response_eval_states_path": evaluation["summary"]["policy_response_eval_states_path"],
            "policy_response_eval_states_sha256": evaluation["summary"]["policy_response_eval_states_sha256"],
            "holdouts_accessed": False,
            "benchmark_eligible": False,
        },
        "runtime_versions": training["A"].get("runtime_versions"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": output["status"],
                      "training_steps": {arm: data["actual_environment_steps"] for arm, data in training.items()},
                      "evaluation_records": len(records), "checkpoints": 15}, indent=2))


if __name__ == "__main__":
    PROTOCOL: dict[str, Any] = {}
    main()
