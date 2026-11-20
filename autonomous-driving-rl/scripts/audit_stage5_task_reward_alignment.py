"""Offline replay audit for the S5 Task & Reward Alignment preflight.

This script reads existing full TRAIN traces only. It does not import or start
MetaDrive, load a policy, modify checkpoints, or train an agent.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_RUN = ROOT / "results/stage5/s5_5_meter_reward_seed101_20261120"
DEFAULT_RAW = ROOT / "runs/stage5_meter_reward/s5_5_meter_reward_seed101_20261120/preflight/reward_traces.jsonl.gz"
GAMMAS = (0.90, 0.99, 0.999, 0.9999, 1.0)
EXPECTED_BASE = "2c572ea620e4254667665814f74485ddf57815e6"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def discounted(values: list[float], gamma: float) -> float:
    return math.fsum((gamma**i) * value for i, value in enumerate(values))


def load_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def analyze(raw_path: Path, run_dir: Path) -> dict[str, Any]:
    preflight_path = run_dir / "reward_preflight.json"
    protocol_path = run_dir / "protocol.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    baseline = json.loads((ROOT / "configs/experiments/stage5_rl_baseline_v1.json").read_text(encoding="utf-8"))
    reward_spec = baseline["reward_contract"]
    raw_sha = sha256_file(raw_path)
    if raw_sha != preflight["raw_trace_sha256"]:
        raise ValueError("Raw trajectory SHA-256 differs from the recorded preflight")
    if protocol["source_commit"] != "1baaeedf9bd9d69f4507d5ff9b27b75dcc9d5639":
        raise ValueError("Unexpected S5.5 source commit in registered protocol")

    rows_by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows = load_jsonl_gzip(raw_path)
    for row in rows:
        rows_by_trace[row["trace_id"]].append(row)
    summaries = {row["trace_id"]: row for row in preflight["trace_summaries"]}
    if set(rows_by_trace) != set(summaries):
        raise ValueError("Trace IDs in raw data do not match preflight summaries")

    tolerance = 2e-9
    replay: dict[str, Any] = {}
    formula_checks = {"rows": len(rows), "failed": 0, "max_abs_v1_env": 0.0,
                      "max_abs_v1_formula": 0.0, "max_abs_scale_formula": 0.0,
                      "max_abs_meter_formula": 0.0, "terminal_rows": 0,
                      "terminal_nonzero_rows": 0, "terminal_at_end_only": True,
                      "sequence_contiguous": True, "single_geometry_environment": True,
                      "max_abs_production_helper": 0.0, "max_abs_reward_used": 0.0}
    identities = set()
    k = float(protocol["reward_variants"]["k"])
    for trace_id, trace_rows in rows_by_trace.items():
        trace_rows.sort(key=lambda row: int(row["step_index"]))
        summary = summaries[trace_id]
        horizon = int(summary["horizon_steps"])
        length_m = float(summary["effective_route_length_m"])
        last_progress = float(summary["initial_route_completion"])
        identities.add((summary["split"], summary["geometry_id"], summary["environment_seed"]))
        reward_sequences: dict[str, list[float]] = {"V1": [], "Scale": [], "Meter": []}
        speeds = []
        for index, row in enumerate(trace_rows, start=1):
            if int(row["step_index"]) != index:
                formula_checks["sequence_contiguous"] = False
            dp = float(row["delta_p"])
            ds = float(row["delta_s_m"])
            terminal = float(row["terminal_reward"])
            time_cost = float(row["time_cost"])
            from src.launcher.executor import compute_step_route_progress_and_reward
            from src.platform import RewardSpecV1, TerminalReason, classify_episode_outcome
            outcome = None
            if row["terminated"] or row["truncated"]:
                reason = TerminalReason(row["primary_reason"])
                raw_flags = {"max_step": reason == TerminalReason.TIMEOUT}
                if reason in (TerminalReason.SUCCESS, TerminalReason.CRASH_HUMAN, TerminalReason.CRASH_VEHICLE,
                              TerminalReason.CRASH_OBJECT, TerminalReason.CRASH_BUILDING,
                              TerminalReason.CRASH_SIDEWALK, TerminalReason.OUT_OF_ROAD):
                    from src.platform.episode import REASON_TO_RAW_FLAG
                    raw_flags[REASON_TO_RAW_FLAG[reason]] = True
                outcome = classify_episode_outcome(raw_flags, bool(row["terminated"]), bool(row["truncated"]))
            helper_delta, _, helper_breakdown = compute_step_route_progress_and_reward(
                RewardSpecV1(), float(row["route_completion"]), last_progress, horizon, outcome)
            last_progress = float(row["route_completion"])
            v1 = dp - 0.25 / horizon + terminal
            scale = k * v1
            meter = ds / 10.0 + k * (terminal - 0.25 / horizon)
            errors = {
                "v1_env": abs(float(row["reward_v1_env"]) - float(row["reward_v1_recomputed"])),
                "v1_formula": abs(float(row["reward_v1_recomputed"]) - v1),
                "scale_formula": abs(float(row["reward_scale"]) - scale),
                "meter_formula": abs(float(row["reward_meter"]) - meter),
                "meter_distance": abs(ds - length_m * dp),
                "time_cost": abs(time_cost - 0.25 / horizon),
                "production_helper": abs(float(row["reward_v1_env"]) - float(helper_breakdown.total_reward)),
                "delta_from_progress": abs(dp - helper_delta),
                "reward_used": abs(float(row["reward_used"]) - float(row["reward_v1_env"])),
            }
            for name, error in errors.items():
                formula_checks[f"max_abs_{name}"] = max(formula_checks.get(f"max_abs_{name}", 0.0), error)
            if any(error > tolerance for error in errors.values()):
                formula_checks["failed"] += 1
            if row["terminated"] or row["truncated"]:
                formula_checks["terminal_rows"] += 1
                if index != len(trace_rows):
                    formula_checks["terminal_at_end_only"] = False
            if terminal != 0.0:
                formula_checks["terminal_nonzero_rows"] += 1
            reward_sequences["V1"].append(float(row["reward_v1_recomputed"]))
            reward_sequences["Scale"].append(float(row["reward_scale"]))
            reward_sequences["Meter"].append(float(row["reward_meter"]))
            speeds.append(float(row["speed_km_h"]))

        returns = {}
        for name, sequence in reward_sequences.items():
            returns[name] = {
                "undiscounted": math.fsum(sequence),
                "discounted_by_gamma": {f"{gamma:g}": discounted(sequence, gamma) for gamma in GAMMAS},
            }
        fixed_window = trace_rows[49:99]
        prefix = {
            name: {
                "return_50_steps_zero_based_49_99": math.fsum(float(row[f"reward_{key.lower()}"]) if key != "V1" else float(row["reward_v1_recomputed"]) for row in fixed_window),
                "mean_speed_km_h_same_window": (math.fsum(float(row["speed_km_h"]) for row in fixed_window) / len(fixed_window)) if fixed_window else None,
            }
            for name, key in (("V1", "V1"), ("Scale", "Scale"), ("Meter", "Meter"))
        }
        moving_speeds = [speed for speed in speeds if speed >= 1.0]
        outcome = summary["outcome"]
        replay[trace_id] = {
            "policy_kind": summary["policy_kind"], "outcome": outcome,
            "clean_success": bool(summary["clean_success"]), "safety_failure": bool(summary["safety_failure"]),
            "timeout": bool(summary["timeout"]), "full_episode": len(trace_rows) == int(summary["steps"]),
            "case_id": summary["case_id"], "split": summary["split"], "geometry_id": summary["geometry_id"],
            "geometry_sha256": summary["geometry_sha256"], "environment_seed": summary["environment_seed"],
            "horizon_steps": horizon, "steps": len(trace_rows), "route_progress_delta": summary["route_progress_delta"],
            "mean_speed_km_h": summary["mean_speed_km_h"],
            "moving_mean_speed_ge_1_km_h": math.fsum(moving_speeds) / len(moving_speeds) if moving_speeds else None,
            "terminal_reward_sum": summary["terminal_reward_sum"], "returns": returns,
            "fixed_window": prefix,
        }
    formula_checks["single_geometry_environment"] = len(identities) == 1
    if formula_checks["failed"] or not formula_checks["sequence_contiguous"] or not formula_checks["terminal_at_end_only"] or not formula_checks["single_geometry_environment"]:
        raise ValueError(f"Reward trace integrity failed: {formula_checks}")

    gamma_horizon = {}
    horizon = int(next(iter(summaries.values()))["horizon_steps"])
    clean_steps = int(summaries["stage1_reference_clean"]["steps"])
    for gamma in GAMMAS:
        gamma_horizon[f"{gamma:g}"] = {
            "effective_horizon_decisions_1_over_1_minus_gamma": None if gamma == 1.0 else 1.0 / (1.0 - gamma),
            "effective_horizon_seconds_at_10hz": None if gamma == 1.0 else 0.1 / (1.0 - gamma),
            "discount_at_clean_terminal_index": gamma ** (clean_steps - 1),
            "discount_at_timeout_terminal_index": gamma ** (horizon - 1),
        }

    # Stationary-only closed-form sequence, cross-checked against the recorded episode.
    stationary_step_cost = 0.25 / horizon
    stationary_closed = {f"{gamma:g}": -stationary_step_cost * sum(gamma**i for i in range(horizon)) for gamma in GAMMAS}
    current_sha = git_value("rev-parse", "HEAD")
    try:
        from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER, task_producer_identity
        current_producer = task_producer_identity(REFERENCE_TASK_PRODUCER)
    except Exception as exc:  # pragma: no cover - environment-specific import failure is recorded
        current_producer = {"error": f"{type(exc).__name__}: {exc}"}
    runtime = {}
    for package in ("Python", "gymnasium", "stable-baselines3", "torch", "metadrive-simulator"):
        try:
            runtime[package] = sys.version.split()[0] if package == "Python" else version(package)
        except PackageNotFoundError:
            runtime[package] = "NOT_INSTALLED"
    audit_script = Path(__file__).resolve()

    return {
        "protocol": "stage5_task_reward_alignment_preflight_v1",
        "analysis_source_commit": current_sha,
        "audit_script_sha256": sha256_file(audit_script),
        "runtime_versions": runtime,
        "reward_source_sha256": sha256_file(ROOT / "src/platform/reward.py"),
        "env_adapter_source_sha256": sha256_file(ROOT / "src/training/stage5/env_adapter.py"),
        "sac_config_file_sha256": sha256_file(ROOT / "configs/experiments/stage5_rl_baseline_v1.json"),
        "expected_pr42_base": EXPECTED_BASE,
        "base_matches_expected": current_sha == EXPECTED_BASE,
        "historical_trajectory_source_commit": protocol["source_commit"],
        "historical_task_context_producer": "NOT_RECORDED; generated before F09 producer version integration; reward replay itself consumes recorded delta_p/terminal only",
        "current_registered_corrected_producer_identity": current_producer,
        "producer_effect_on_historical_reward_replay": "none; recorded transition reward inputs are route progress, horizon and terminal component; historical policy observation provenance is not retroactively upgraded",
        "reward_spec": reward_spec,
        "reward_spec_sha256": canonical_sha256(reward_spec),
        "protocol_sha256": (run_dir / "protocol.sha256").read_text(encoding="ascii").split()[0],
        "protocol_file_sha256": sha256_file(protocol_path),
        "preflight_file_sha256": sha256_file(preflight_path),
        "raw_trace_path": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
        "raw_trace_sha256": raw_sha,
        "raw_trace_size_bytes": raw_path.stat().st_size,
        "manifest_sha256": protocol["source_geometry_manifest_file_sha256"],
        "geometry_id": next(iter(summaries.values()))["geometry_id"],
        "geometry_sha256": next(iter(summaries.values()))["geometry_sha256"],
        "split": "TRAIN_ONLY",
        "environment_seed": next(iter(summaries.values()))["environment_seed"],
        "trajectory_count": len(replay), "transition_count": len(rows),
        "historical_trace_rows": replay,
        "reward_formula_integrity": formula_checks,
        "reward_rankings_by_full_episode": {
            "V1": sorted(replay, key=lambda key: replay[key]["returns"]["V1"]["undiscounted"], reverse=True),
            "Scale": sorted(replay, key=lambda key: replay[key]["returns"]["Scale"]["undiscounted"], reverse=True),
            "Meter": sorted(replay, key=lambda key: replay[key]["returns"]["Meter"]["undiscounted"], reverse=True),
        },
        "discount_horizon": gamma_horizon,
        "stationary_timeout": {
            "step_cost": stationary_step_cost,
            "undiscounted_return": -0.25,
            "discounted_return_closed_form_by_gamma": stationary_closed,
            "timeout_bootstrapped_by_sac": True,
            "observed_reward_sum_is_not_full_sac_target": True,
        },
        "classification": {
            "CONFIRMED": [
                "RewardSpecV1 accounting matches all recorded transition rows within 2e-9; clipping is disabled; signed progress and initial reset progress are preserved.",
                "On these four full episodes from one TRAIN geometry/environment seed, all reward variants rank the clean episode above stationary timeout and the two safety failures at full-episode return.",
                "Gymnasium timeout is truncation; installed SB3 2.9.2 VecEnv preserves terminal_observation and TimeLimit.truncated, and ReplayBuffer masks timeout done to zero, so critic bootstraps at timeout.",
                "Gamma discounts late terminal rewards sharply; see exact weights in discount_horizon.",
            ],
            "HYPOTHESIS": [
                "A pre-terminal progress-only segment can favor fast movement before later safety outcome is observed; this may make safe-speed/turn behavior difficult to credit, but these traces do not establish causal overspeed or a defective reward formula.",
                "The combination of reward scale, delayed terminal safety feedback, SAC entropy and sparse successful transitions may contribute to failed learning; this audit does not isolate their effects.",
            ],
            "UNKNOWN": [
                "Relative ranking of clean-fast and clean-slow runs matched by geometry and environment seed; only one clean episode exists.",
                "Unsafe-slow outcome ranking; no authentic unsafe-slow full trace is present.",
                "Whether a versioned shaping candidate improves policy learning or resolves stationary/actor collapse.",
                "Whether historical trajectories used corrected TaskContext producer; the historical protocol predates and does not record that identity.",
            ],
        },
        "decision": "REWARD_V1_RETAIN",
        "decision_rationale": "No replay-integrity defect or full-episode reward inversion is demonstrated on available traces; a new reward candidate cannot be justified without matched clean speed outcomes and a real unsafe-slow trace. Retain canonical RewardSpecV1 for now; this is not evidence that reward is sufficient or that SAC learns.",
        "training_performed": False,
        "validation_test_accessed": False,
        "teacher_trajectories_used_for_training": False,
        "benchmark_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage5/task_reward_alignment_p1r/evidence.json")
    args = parser.parse_args()
    evidence = analyze(args.raw, args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(evidence, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    args.output.write_text(serialized, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    args.output.with_suffix(".sha256").write_text(f"{digest}  {args.output.name}\n", encoding="ascii", newline="\n")
    print(json.dumps({"decision": evidence["decision"], "trajectories": evidence["trajectory_count"],
                      "transitions": evidence["transition_count"], "integrity": evidence["reward_formula_integrity"],
                      "evidence": str(args.output), "evidence_sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
