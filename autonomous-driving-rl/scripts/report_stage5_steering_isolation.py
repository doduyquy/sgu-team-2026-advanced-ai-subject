"""Summarize the completed S5.3G steering isolation probe with paired cases."""
from __future__ import annotations

import hashlib
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/stage5/s5_3g_steering_isolation_seed101_20261115"
RUN = ROOT / "runs/stage5_pilot/s5_3g_throttle_hold5_seed101_20261115"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mean(values):
    values = [float(v) for v in values if v is not None]
    return statistics.mean(values) if values else None


def main():
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (OUT / "episode_records.jsonl").read_text(encoding="utf-8").splitlines()]
    arms = list(summary["conditions"])
    keys = {(r["geometry_id"], r["environment_seed"]) for r in records}
    if len(records) != 24 or len(keys) != 8 or any(sum((r["geometry_id"], r["environment_seed"]) == k for r in records) != 3 for k in keys):
        raise RuntimeError("Expected 24 records: exactly one result per arm on each of 8 paired cases")

    indexed = {(r["geometry_id"], r["environment_seed"], r["arm"]): r for r in records}
    paired = {}
    for lhs, rhs in ((arms[1], arms[0]), (arms[2], arms[0]), (arms[2], arms[1])):
        differences = [indexed[(g, e, lhs)]["route_progress_delta"] - indexed[(g, e, rhs)]["route_progress_delta"]
                       for g, e in sorted(keys)]
        paired[f"{lhs}_minus_{rhs}_route_progress"] = {
            "n": len(differences), "mean_difference": mean(differences),
            "median_difference": statistics.median(differences),
            "positive_cases": sum(x > 0 for x in differences),
            "negative_cases": sum(x < 0 for x in differences), "ties": sum(x == 0 for x in differences),
        }

    per_tier = {}
    for tier in sorted({r["tier"] for r in records}):
        per_tier[tier] = {}
        for arm in arms:
            rows = [r for r in records if r["tier"] == tier and r["arm"] == arm]
            per_tier[tier][arm] = {
                "n": len(rows), "route_progress_delta_mean": mean(r["route_progress_delta"] for r in rows),
                "mean_speed_km_h": mean(r["mean_speed_km_h"] for r in rows),
                "outcomes": {reason: sum(r["outcome"] == reason for r in rows)
                             for reason in sorted({r["outcome"] for r in rows})},
            }

    source_paths = [Path("scripts/diagnose_stage5_steering_isolation.py"),
                    Path("scripts/report_stage5_steering_isolation.py")]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except Exception:
        commit, dirty = "unavailable", None
    provenance = {
        "protocol": "S5.3G steering isolation diagnostic v1",
        "status": summary["status"], "code_commit": commit,
        "worktree_dirty_during_run": dirty,
        "checkpoint_source_commit": summary["code_commit"],
        "checkpoint_model_sha256": summary["checkpoint_sha256"],
        "training_step": summary["training_step"],
        "pilot_split_sha256": summary["pilot_split_sha256"],
        "schedule_seed": summary["schedule_seed"],
        "case_schedule": summary["case_schedule"],
        "episode_records_sha256": sha(OUT / "episode_records.jsonl"),
        "source_file_sha256": {p.as_posix(): sha(ROOT / p) for p in source_paths},
        "python": sys.version, "platform": platform.platform(),
        "conditions": summary["conditions"], "max_steps_per_case": summary["max_steps_per_case"],
        "censoring_rule": summary["censoring_rule"],
        "evaluation_isolation": summary["evaluation_isolation"],
        "limitations": [
            "Eight cases total, two per tier; screening probe only, not an inferential estimate.",
            "Throttle was overridden to fixed +0.5 in every arm to isolate steering; this is not SAC policy performance.",
            "Condition C is a simple pure-pursuit steering diagnostic without a speed governor; its high speed and crashes do not assess a complete lane-following controller.",
            "Condition A episodes censored at 300 steps are not timeouts and cannot be counted as safe completions.",
            "No weights, optimizer, normalization, or replay buffer were updated; no VALIDATION/TEST cases were accessed.",
        ],
    }
    (OUT / "paired_analysis.json").write_text(json.dumps({"paired_route_progress": paired, "per_tier": per_tier}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# S5.3G — Steering isolation diagnostic",
        "",
        "## Protocol and scope",
        "",
        "Completed 24 short probes: 8 fixed internal PILOT_DIAGNOSTIC TRAIN geometries (2 per tier, distinct sequence families), each run once in each arm with the same geometry and environment seed. The 10k throttle-hold SAC checkpoint was used only to sample steering in A/B. All arms used a fixed +0.5 throttle override to isolate steering. This is an actuator diagnostic, not an SAC policy evaluation or benchmark result.",
        "",
        "A: stochastic SAC steering each step. B: same stochastic SAC steering with a 0.15 action-unit/step slew limit. C: waypoint pure-pursuit steering from public encoded waypoints, 8 m lookahead, with no speed governor. Each probe was capped at 300 steps. A cap is `PROBE_CENSORED`, never TIMEOUT. Simulator terminal outcomes use Platform V1 classification.",
        "",
        "## Results",
        "",
        "| Arm | Cases | Observed terminal outcomes | Censored | Mean route progress delta | Mean speed (km/h) | Mean absolute lateral offset (m) |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for arm in arms:
        a = summary["arm_aggregates"][arm]
        outcomes = ", ".join(f"{k} {v}" for k, v in a["outcome_counts"].items())
        lines.append(f"| {arm} | {a['n_cases']} | {outcomes} | {a['probe_censored']} | {a['mean_route_progress_delta']:.4f} | {a['mean_speed_km_h']:.2f} | {a['mean_abs_lateral_deviation_m']:.3f} |")
    lines += ["", "### Paired route progress deltas", "", "Positive means the first arm progressed farther on the same geometry and environment seed.", "", "| Paired contrast | Mean difference | Median | Cases higher / lower / tied |", "|---|---:|---:|---:|"]
    for label, stats in paired.items():
        lines.append(f"| {label} | {stats['mean_difference']:+.4f} | {stats['median_difference']:+.4f} | {stats['positive_cases']} / {stats['negative_cases']} / {stats['ties']} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "The steering slew limit did not make positive-throttle motion safe in this screening set: all 8 B probes ended in OUT_OF_ROAD or CRASH_VEHICLE. A reached the 300-step cap in 7/8 probes, so those are censored observations, not timeout or verified safe runs; the remaining A probe ended OUT_OF_ROAD. C made the greatest route progress on average and had the smallest lateral offset, but it was not safe at this throttle: 6/8 CRASH_VEHICLE, 1/8 OUT_OF_ROAD, and 1/8 SUCCESS. Its mean speed was about 40.9 km/h and mean maximum speed about 70.9 km/h. A steering-only pure-pursuit rule without speed control is therefore not evidence that steering alone solves the task.",
        "",
        "This experiment does not identify why SAC learned a near-zero deterministic throttle. It shows that under a fixed +0.5 throttle intervention, steering rate-limiting alone did not prevent safety failures, and route following without a speed governor can yield more progress while still crashing. The original SAC policy jointly chooses throttle and steering, so this probe cannot attribute its learning failure to steering alone.",
        "",
        "## Action fidelity audit status",
        "",
        "The separate runtime action-fidelity audit exercised Stable-Baselines3 2.9.2 `collect_rollouts` on a synthetic Box(2) environment during warm-up and after learning starts. In both captured transitions, the environment action and replay action matched (within float32 representation). This establishes the custom sampler/collector code path; the historical S5.3G replay buffer was not saved, so its arrays cannot be inspected retrospectively. See `../s5_3g_throttle_hold5_seed101_20261115/action_fidelity_runtime_audit.json`.",
        "",
        "## Provenance and limitations",
        "",
        f"Checkpoint SHA-256: `{summary['checkpoint_sha256']}`. Internal split SHA-256: `{summary['pilot_split_sha256']}`. Source commit recorded in checkpoint: `{summary['code_commit']}`. Per-case records, paired analysis, and full provenance are adjacent JSON files. The sample is small (8 cases), diagnostic, and should not be generalized to all TRAIN geometries. No model training was performed.",
        "",
    ]
    (OUT / "CONCLUSION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"paired": paired, "aggregates": summary["arm_aggregates"], "provenance": str(OUT / "provenance.json")}, indent=2))


if __name__ == "__main__":
    main()
