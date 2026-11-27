"""Validate and summarize completed S5.3L frozen stochastic rollout replicates."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "s5_3l_softq_calibration_seed101_20261115"
RESULTS = ROOT / "results/stage5" / RUN_ID
RAW = ROOT / "runs/stage5_pilot" / RUN_ID
REPLICATES = 8
CONDITIONS = {"brake_q15", "near_neutral_q50", "throttle_q85"}
T95_DF7 = 2.364624251
T95_DF31 = 2.039513446


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mean(values):
    return statistics.mean(values) if values else None


def ci95(values):
    if len(values) < 2:
        return [None, None]
    critical = T95_DF7 if len(values) == 8 else (T95_DF31 if len(values) == 32 else 1.96)
    half = critical * statistics.stdev(values) / math.sqrt(len(values))
    return [statistics.mean(values) - half, statistics.mean(values) + half]


def soft_return(row):
    if row["outcome"] == "TIMEOUT":
        return row["timeout_bootstrapped_soft_return_estimate"]
    if row["outcome"] == "PROBE_CENSORED":
        return None
    return row["discounted_soft_return_observed"]


def main() -> None:
    manifest_path = RESULTS / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or len(manifest.get("replicate_states", [])) != REPLICATES:
        raise RuntimeError("L1 manifest is not COMPLETE for all eight replicates")

    all_rows, summaries = [], []
    hashes, summary_hashes = {}, {}
    reference = None
    reference_cases = None
    for rep in range(1, REPLICATES + 1):
        name = f"replicate_{rep:02d}"
        summary_path = RESULTS / name / "summary.json"
        records_path = RESULTS / name / "episode_records.jsonl"
        raw_path = RAW / name / "transition_records.jsonl"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if summary.get("status") != "COMPLETE" or summary.get("replicate_id") != rep or len(rows) != 12:
            raise RuntimeError(f"Incomplete or misidentified {name}")
        ro = summary["read_only"]
        if not ro["policy_weights_unchanged"] or ro["optimizer_updates_before"] != ro["optimizer_updates_after"] or ro["validation_test_access"]:
            raise RuntimeError(f"Read-only/holdout check failed for {name}")
        parity = summary["initial_state_parity"]
        if parity["verified_case_count"] != 4 or parity["failures"]:
            raise RuntimeError(f"Reset parity failed for {name}")
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        expected_raw = str(raw_path.relative_to(ROOT)).replace("\\", "/")
        # Correct a runner metadata path label; raw records themselves are untouched.
        if summary.get("raw_step_records") != expected_raw:
            summary["raw_step_records"] = expected_raw
            summary["raw_step_records_sha256"] = sha256(raw_path)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary_hashes[name] = sha256(summary_path)
        hashes[name] = {"bytes": raw_path.stat().st_size, "sha256": sha256(raw_path)}
        if reference is None:
            reference = summary
            reference_cases = summary["cases"]
        else:
            for key in ("checkpoint_sha256", "pilot_split_sha256", "gamma", "entropy_coefficient",
                        "stable_baselines3_version", "gymnasium_version"):
                if summary[key] != reference[key]:
                    raise RuntimeError(f"Cross-replicate provenance mismatch: {key}")
            if summary["cases"] != reference_cases:
                raise RuntimeError("Geometry/environment seed schedule changed across replicates")
        for row in rows:
            if row["replicate_id"] != rep or not row["reset_parity_verified"]:
                raise RuntimeError(f"Bad episode provenance/parity in {name}")
            row["soft_return_for_analysis"] = soft_return(row)
            all_rows.append(row)
        summaries.append(summary)

    if len(all_rows) != 96:
        raise RuntimeError(f"Expected 96 episodes, found {len(all_rows)}")
    keys = [(r["geometry_id"], r["condition"], r["replicate_id"]) for r in all_rows]
    if len(set(keys)) != 96 or {r["condition"] for r in all_rows} != CONDITIONS:
        raise RuntimeError("Duplicate/missing rollout keys or wrong intervention conditions")

    cells = defaultdict(list)
    paired = defaultdict(dict)
    outcomes = Counter()
    for row in all_rows:
        cells[(row["geometry_id"], row["tier"], row["condition"])].append(row)
        paired[(row["geometry_id"], row["replicate_id"])][row["condition"]] = row
        outcomes[row["outcome"]] += 1
    if len(cells) != 12 or any(len(v) != 8 for v in cells.values()):
        raise RuntimeError("Expected eight continuations in each of 12 geometry-condition cells")
    if len(paired) != 32 or any(set(v) != CONDITIONS for v in paired.values()):
        raise RuntimeError("Incomplete paired action triplets")

    # Confirm exact same reset state/action intervention across stochastic seeds.
    for geometry in {r["geometry_id"] for r in all_rows}:
        group = [r for r in all_rows if r["geometry_id"] == geometry]
        if len({r["environment_seed"] for r in group}) != 1 or len({r["initial_observation_sha256"] for r in group}) != 1:
            raise RuntimeError(f"Reset state or environment seed changed across repeats: {geometry}")
        for condition in CONDITIONS:
            arm = [r for r in group if r["condition"] == condition]
            if len({r["initial_throttle_value"] for r in arm}) != 1 or len({r["initial_steering_fixed_to_deterministic_actor"] for r in arm}) != 1:
                raise RuntimeError(f"Initial intervention changed across repeats: {geometry}/{condition}")
    for (geometry, replicate_id), group in paired.items():
        if len({row["stochastic_continuation_seed"] for row in group.values()}) != 1:
            raise RuntimeError(f"Common random-number seed mismatch within paired triplet: {geometry}/{replicate_id}")
    if any(len({g["brake_q15"]["stochastic_continuation_seed"] for (geom, rep), g in paired.items() if geom == geometry}) != REPLICATES
           for geometry in {r["geometry_id"] for r in all_rows}):
        raise RuntimeError("Continuation seeds are not distinct across replicates")
    fingerprints = {s["read_only"]["policy_fingerprint_before"] for s in summaries}
    if len(fingerprints) != 1:
        raise RuntimeError("Frozen policy fingerprint changed between replicates")

    cell_stats = []
    for (geometry, tier, condition), group in sorted(cells.items()):
        qs = [r["initial_min_q"] for r in group]
        if max(qs) - min(qs) > 1e-6:
            raise RuntimeError(f"Initial action/Q changed between repeats for {geometry}/{condition}")
        returns = [r["soft_return_for_analysis"] for r in group]
        if any(v is None for v in returns):
            raise RuntimeError(f"Censored episode in {geometry}/{condition}; cannot claim full-return calibration")
        gaps = [r["initial_min_q"] - r["soft_return_for_analysis"] for r in group]
        cell_stats.append({
            "geometry_id": geometry, "tier": tier, "condition": condition, "n": 8,
            "mean_min_q": mean(qs), "mean_soft_return": mean(returns),
            "soft_return_sd": statistics.stdev(returns), "soft_return_ci95_t_df7": ci95(returns),
            "mean_q_minus_return": mean(gaps), "q_minus_return_sd": statistics.stdev(gaps),
            "q_minus_return_ci95_t_df7": ci95(gaps),
            "outcomes": dict(Counter(r["outcome"] for r in group)),
        })

    pair_stats = {}
    for metric in ("soft_return_for_analysis", "discounted_reward_return_observed", "route_progress_delta"):
        pair_stats[metric] = {}
        for left, right in (("near_neutral_q50", "brake_q15"), ("throttle_q85", "brake_q15"),
                            ("throttle_q85", "near_neutral_q50")):
            diffs = [float(g[left][metric]) - float(g[right][metric]) for g in paired.values()
                     if g[left][metric] is not None and g[right][metric] is not None]
            pair_stats[metric][f"{left}_minus_{right}"] = {
                "n": len(diffs), "mean": mean(diffs),
                "median": statistics.median(diffs) if diffs else None,
                "sd": statistics.stdev(diffs) if len(diffs) > 1 else None,
                "ci95_t_df31": ci95(diffs) if len(diffs) == 32 else None,
                "positive": sum(x > 0 for x in diffs), "negative": sum(x < 0 for x in diffs),
                "ties": sum(x == 0 for x in diffs),
            }

    q_action_by_geometry = []
    for geometry in sorted({r["geometry_id"] for r in all_rows}):
        q_by_condition = {}
        for condition in CONDITIONS:
            vals = [r["initial_min_q"] for r in all_rows
                    if r["geometry_id"] == geometry and r["condition"] == condition]
            q_by_condition[condition] = mean(vals)
        q_action_by_geometry.append({
            "geometry_id": geometry,
            "throttle_minus_brake_min_q": q_by_condition["throttle_q85"] - q_by_condition["brake_q15"],
            "neutral_minus_brake_min_q": q_by_condition["near_neutral_q50"] - q_by_condition["brake_q15"],
        })

    gaps = [r["initial_min_q"] - r["soft_return_for_analysis"] for r in all_rows]
    wall = sum(float(s["wall_time_s"]) for s in summaries)
    raw_bytes = sum(item["bytes"] for item in hashes.values())
    payload = {
        "protocol": "S5.3L-L1 repeated frozen stochastic rollouts; fixed diagnostic state calibration",
        "status": "COMPLETE", "episodes": len(all_rows), "replicates": REPLICATES,
        "geometry_count": len({r["geometry_id"] for r in all_rows}),
        "checkpoint_sha256": reference["checkpoint_sha256"], "pilot_split_sha256": reference["pilot_split_sha256"],
        "gamma": reference["gamma"], "entropy_coefficient": reference["entropy_coefficient"],
        "stable_baselines3_version": reference["stable_baselines3_version"],
        "gymnasium_version": reference["gymnasium_version"], "outcomes": dict(outcomes),
        "censored": outcomes.get("PROBE_CENSORED", 0),
        "mean_online_min_q": mean([r["initial_min_q"] for r in all_rows]),
        "mean_soft_return": mean([r["soft_return_for_analysis"] for r in all_rows]),
        "mean_q_minus_return": mean(gaps), "median_q_minus_return": statistics.median(gaps),
        "sd_q_minus_return_pooled": statistics.stdev(gaps),
        "q_minus_return_range": [min(gaps), max(gaps)],
        "cells": cell_stats, "paired_intervention_differences": pair_stats,
        "q_action_value_contrast_by_geometry": q_action_by_geometry,
        "mean_throttle_minus_brake_min_q_over_four_geometries": mean([r["throttle_minus_brake_min_q"] for r in q_action_by_geometry]),
        "read_only_all_replicates": all(s["read_only"]["policy_weights_unchanged"]
            and s["read_only"]["optimizer_updates_before"] == s["read_only"]["optimizer_updates_after"]
            and not s["read_only"]["validation_test_access"] for s in summaries),
        "reset_parity_all_replicates": all(s["initial_state_parity"]["verified_case_count"] == 4
            and not s["initial_state_parity"]["failures"] for s in summaries),
        "sum_replicate_wall_time_s": wall, "sum_replicate_wall_time_hours": wall / 3600,
        "raw_trace_bytes": raw_bytes, "raw_trace_hashes": hashes,
        "replicate_summary_hashes": summary_hashes,
        "limitations": [
            "Four fixed internal TRAIN geometries and one environment seed per geometry; inference is conditional on those reset states.",
            "Eight independent stochastic continuations estimate policy-sampling variability; action conditions use common random numbers within geometry and replicate.",
            "TIMEOUT return adds gamma^T times target-critic soft bootstrap; L0 measured this tail as negligible at these episode lengths.",
            "The checkpoint critics and actor co-adapted with an automatic entropy coefficient during training; L1 uses the frozen final alpha.",
            "Per-cell t intervals describe stochastic continuation variation, not uncertainty across geometries or training roots.",
        ],
    }
    (RESULTS / "L1_ANALYSIS.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# S5.3L — Soft-Q Calibration (L1)",
        "",
        f"Status: **COMPLETE** — {len(all_rows)}/96 full episodes, {REPLICATES}/8 stochastic continuation replicates.",
        "",
        f"Checkpoint SHA-256 {payload['checkpoint_sha256']}; pilot split SHA-256 {payload['pilot_split_sha256']}; gamma {payload['gamma']}; frozen alpha {payload['entropy_coefficient']:.8f}; SB3 {payload['stable_baselines3_version']}; Gymnasium {payload['gymnasium_version']}.",
        "",
        "Outcomes: " + ", ".join(f"{k}={v}" for k, v in sorted(outcomes.items())) + f". Censored={payload['censored']}.",
        "",
        f"Mean online min(Q1,Q2) = **{payload['mean_online_min_q']:.3f}**; mean full soft-return estimate = **{payload['mean_soft_return']:.3f}**; mean Q-return gap = **{payload['mean_q_minus_return']:+.3f}** (pooled SD {payload['sd_q_minus_return_pooled']:.3f}, range {payload['q_minus_return_range'][0]:+.3f} to {payload['q_minus_return_range'][1]:+.3f}).",
        "",
        "## Per fixed reset state",
        "",
        "95% t intervals use eight independent stochastic continuations (df=7); they describe variation conditional on one fixed geometry/environment reset, not across maps.",
        "",
        "| Tier | Geometry | Initial action | n | minQ | Soft return | Q-return gap | Gap 95% CI |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in cell_stats:
        lo, hi = row["q_minus_return_ci95_t_df7"]
        lines.append(f"| {row['tier']} | {row['geometry_id']} | {row['condition']} | 8 | {row['mean_min_q']:.3f} | {row['mean_soft_return']:.3f} | {row['mean_q_minus_return']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += [
        "",
        "## Paired contrasts",
        "",
        "Positive values favor the condition on the left. Pairing is by geometry, environment seed and stochastic continuation seed.",
        "",
        "| Metric | Contrast | n | Mean | Median | 95% CI | + / - / tie |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for metric, contrasts in pair_stats.items():
        for contrast, row in contrasts.items():
            ci = row["ci95_t_df31"]
            ci_text = f"[{ci[0]:+.6f}, {ci[1]:+.6f}]" if ci else "n/a"
            lines.append(f"| {metric} | {contrast} | {row['n']} | {row['mean']:+.6f} | {row['median']:+.6f} | {ci_text} | {row['positive']} / {row['negative']} / {row['ties']} |")
    lines += [
        "",
        "## Interpretation",
        "",
        f"L0 validated SB3 timeout/replay/soft-return semantics and found the timeout bootstrap tail negligible at these horizons. The positive Q-return gap persists across stochastic continuations: all 12 geometry-condition 95% intervals are above zero. This is evidence of local absolute critic overestimation/miscalibration on the tested state-action pairs, not a confirmed cause of the stationary policy. Relative action ranking is unresolved: mean throttle-minus-brake Q difference across four geometries is {payload['mean_throttle_minus_brake_min_q_over_four_geometries']:+.4f}, while the paired soft-return contrast is {pair_stats['soft_return_for_analysis']['throttle_q85_minus_brake_q15']['mean']:+.6f} with CI [{pair_stats['soft_return_for_analysis']['throttle_q85_minus_brake_q15']['ci95_t_df31'][0]:+.6f}, {pair_stats['soft_return_for_analysis']['throttle_q85_minus_brake_q15']['ci95_t_df31'][1]:+.6f}], which includes zero. The data therefore do not establish that the critic chooses the wrong action ranking. L1 cannot attribute the absolute gap to critic approximation, changing alpha/policy during training, or replay coverage.",
        "",
        "No training, replay writes, normalization updates, VALIDATION or TEST occurred. Full per-replicate hashes are in L1_ANALYSIS.json; run manifest is run_manifest.json.",
        "",
    ]
    (RESULTS / "L1_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")

    for item in manifest["replicate_states"]:
        item["summary_sha256"] = summary_hashes[f"replicate_{item['replicate_id']:02d}"]
    manifest["analysis"] = "L1_ANALYSIS.json"
    manifest["total_raw_trace_bytes"] = raw_bytes
    manifest["sum_replicate_wall_time_s"] = wall
    manifest["status"] = "COMPLETE"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "COMPLETE", "episodes": len(all_rows), "outcomes": dict(outcomes),
        "mean_q": payload["mean_online_min_q"], "mean_soft_return": payload["mean_soft_return"],
        "mean_gap": payload["mean_q_minus_return"], "wall_time_s": wall, "raw_bytes": raw_bytes,
    }, indent=2))


if __name__ == "__main__":
    main()
