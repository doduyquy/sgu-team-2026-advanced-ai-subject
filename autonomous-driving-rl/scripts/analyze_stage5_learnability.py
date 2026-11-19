"""Summarize completed S5.4 runs, paired TRAIN diagnostics, and plots."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, Iterable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mean(values: Iterable[float]) -> Optional[float]:
    items = [float(value) for value in values if value is not None]
    return statistics.mean(items) if items else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_paths(value: Any) -> Any:
    """Make in-repository absolute paths portable in committed summaries."""
    if isinstance(value, dict):
        return {key: _portable_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_paths(item) for item in value]
    if not isinstance(value, str):
        return value
    candidate = Path(value)
    if not candidate.is_absolute():
        return value
    try:
        return candidate.resolve(strict=False).relative_to(ROOT).as_posix()
    except ValueError:
        return value


def _load_arm(run_root: Path, arm: str, steps: List[int]) -> Dict[str, Any]:
    arm_root = run_root / f"arm_{arm}"
    status_path = arm_root / "run_status.json"
    status = _read_json(status_path) if status_path.is_file() else {"status": "MISSING"}
    result: Dict[str, Any] = {"status": status, "checkpoints": {}, "coverage": None}
    coverage_path = arm_root / "checkpoints/training_geometry_coverage.json"
    if coverage_path.is_file():
        result["coverage"] = _read_json(coverage_path)
    for step in steps:
        root = arm_root / "diagnostics" / f"step_{step:05d}"
        probe_dir = root / "probe_5404"
        probe_summary = probe_dir / f"step_{step:05d}_summary.json"
        full_dir = root / "full_5404"
        full_summary = full_dir / f"step_{step:05d}_summary.json"
        checkpoint_dir = arm_root / "checkpoints" / f"step_{step:05d}"
        metadata_path = checkpoint_dir / "metadata.json"
        if not metadata_path.is_file():
            result["checkpoints"][step] = {"available": False}
            continue
        metadata = _read_json(metadata_path)
        checkpoint = {
            "available": True,
            "model_sha256": metadata.get("model_sha256"),
            "metadata_sha256": _sha256(metadata_path),
            "optimizer_updates": metadata.get("training_summary", {}).get("optimizer_updates"),
            "actor_loss": metadata.get("training_summary", {}).get("actor_loss"),
            "critic_loss": metadata.get("training_summary", {}).get("critic_loss"),
            "entropy_coefficient": metadata.get("training_summary", {}).get("entropy_coefficient"),
            "probe": None,
            "full_episode": None,
            "probe_records": [],
            "full_records": [],
        }
        if probe_summary.is_file():
            probe = _read_json(probe_summary)
            checkpoint["probe"] = probe
            checkpoint["probe_records"] = _read_jsonl(probe_dir / probe["records_file"])
        if full_summary.is_file():
            full = _read_json(full_summary)
            checkpoint["full_episode"] = full
            checkpoint["full_records"] = _read_jsonl(full_dir / full["records_file"])
        result["checkpoints"][step] = checkpoint
    return result


def _paired(a_records: List[Dict[str, Any]], b_records: List[Dict[str, Any]],
            metrics: List[str], data_kind: str, step: int) -> Dict[str, Any]:
    a_index = {(row.get("geometry_id"), row.get("environment_seed")): row for row in a_records}
    b_index = {(row.get("geometry_id"), row.get("environment_seed")): row for row in b_records}
    shared = sorted(set(a_index) & set(b_index))
    fields: Dict[str, Any] = {}
    for metric in metrics:
        deltas = []
        per_case = []
        for key in shared:
            av, bv = a_index[key].get(metric), b_index[key].get(metric)
            if av is None or bv is None:
                continue
            delta = float(bv) - float(av)
            deltas.append(delta)
            per_case.append({"geometry_id": key[0], "environment_seed": key[1],
                             "control_a": float(av), "treatment_b": float(bv), "delta_b_minus_a": delta})
        fields[metric] = {
            "paired_n": len(deltas), "mean_delta_b_minus_a": _mean(deltas),
            "median_delta_b_minus_a": statistics.median(deltas) if deltas else None,
            "treatment_b_better_count": sum(value > 0 for value in deltas),
            "control_a_better_count": sum(value < 0 for value in deltas),
            "ties": sum(value == 0 for value in deltas),
            "per_case": per_case,
        }
    return {"training_step": step, "data_kind": data_kind, "paired_case_count": len(shared),
            "pairing_keys": ["geometry_id", "environment_seed"], "metrics": fields}


def analyze(run_id: str) -> Dict[str, Any]:
    run_root = ROOT / "runs/stage5_learnability" / run_id
    results_dir = ROOT / "results/stage5" / run_id
    protocol_path = results_dir / "protocol.json"
    protocol = _read_json(protocol_path)
    steps = list(protocol["checkpoint_steps"])
    arms = {arm: _load_arm(run_root, arm, steps) for arm in ("A", "B")}
    checkpoints: Dict[str, Any] = {}
    metric_rows: List[Dict[str, Any]] = []
    paired_rows: List[Dict[str, Any]] = []
    for step in steps:
        checkpoints[str(step)] = {}
        for arm in ("A", "B"):
            cp = arms[arm]["checkpoints"].get(step, {"available": False})
            checkpoints[str(step)][arm] = cp
            row: Dict[str, Any] = {"training_step": step, "arm": arm,
                                   "checkpoint_available": bool(cp.get("available"))}
            if cp.get("available"):
                row.update({"model_sha256": cp["model_sha256"], "optimizer_updates": cp["optimizer_updates"],
                            "actor_loss": cp["actor_loss"], "critic_loss": cp["critic_loss"],
                            "entropy_coefficient": cp["entropy_coefficient"]})
                probe = cp.get("probe")
                if probe:
                    agg = probe["aggregate"]
                    actor = probe.get("actor_state_sensitivity", {})
                    row.update({"probe_speed_km_h": agg.get("probe_mean_speed_km_h"),
                                "probe_route_progress_delta": agg.get("probe_mean_route_progress_delta"),
                                "probe_stationary_fraction": agg.get("probe_stationary_fraction"),
                                "probe_censored_count": agg.get("probe_censored_count"),
                                "probe_safety_failure_count": agg.get("probe_safety_failure_count"),
                                "actor_steering_sensitivity": (actor.get("deterministic_action_std_across_fixed_observations") or [None, None])[0],
                                "actor_throttle_sensitivity": (actor.get("deterministic_action_std_across_fixed_observations") or [None, None])[1],
                                "first_layer_nonpositive_fraction": actor.get("first_layer_nonpositive_fraction")})
                full = cp.get("full_episode")
                if full:
                    agg = full["aggregate"]
                    row.update({"full_mean_final_route_completion": agg.get("final_route_completion"),
                                "full_median_final_route_completion": agg.get("median_route_completion"),
                                "full_mean_route_progress_delta": agg.get("route_progress_delta"),
                                "full_clean_success_rate": agg.get("clean_success_rate"),
                                "full_safety_failure_rate": agg.get("safety_failure_rate"),
                                "full_timeout_rate": agg.get("timeout_rate"),
                                "full_mean_episode_return": agg.get("episode_return"),
                                "full_mean_speed_km_h": agg.get("mean_speed_km_h"),
                                "full_stationary_fraction": agg.get("stationary_fraction"),
                                "full_mean_time_to_clean_success_s": agg.get("time_to_clean_success_s")})
            metric_rows.append(row)
        a_cp, b_cp = arms["A"]["checkpoints"].get(step, {}), arms["B"]["checkpoints"].get(step, {})
        if a_cp.get("probe") and b_cp.get("probe"):
            paired_rows.append(_paired(a_cp["probe_records"], b_cp["probe_records"],
                                       ["route_progress_delta", "mean_speed_km_h", "stationary_fraction",
                                        "mean_throttle", "mean_brake"], "PROBE_CENSORED_DIAGNOSTIC", step))
        if a_cp.get("full_episode") and b_cp.get("full_episode"):
            paired_rows.append(_paired(a_cp["full_records"], b_cp["full_records"],
                                       ["route_progress_delta", "final_route_completion", "episode_return",
                                        "mean_speed_km_h", "stationary_fraction", "clean_success", "safety_failure"],
                                       "FULL_EPISODE", step))

    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "checkpoint_metrics.csv").open("w", newline="", encoding="utf-8") as sink:
        fields = sorted({key for row in metric_rows for key in row})
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric_rows)
    (results_dir / "paired_comparisons.json").write_text(
        json.dumps(paired_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    _make_plots(arms, steps, plots_dir)
    summary = {
        "protocol_sha256": protocol["protocol_sha256"], "source_commit": protocol["source_commit"],
        "run_id": run_id, "arms": {key: value["status"] for key, value in arms.items()},
        "checkpoints": checkpoints,
        "paired_comparisons": paired_rows,
        "training_geometry_coverage": {arm: data["coverage"] for arm, data in arms.items()},
        "diagnostic_only": True, "official_validation_or_test_used": False,
        "interpretation_limits": [
            "One root seed does not establish superiority or generalization.",
            "300-step PROBE_CENSORED records are not full-episode safety/success/timeout outcomes.",
            "Only completed full episodes contribute to clean-success, safety and timeout rates.",
        ],
    }
    summary = _portable_paths(summary)
    summary_path = results_dir / "scientific_results.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_files = [summary_path, results_dir / "checkpoint_metrics.csv", results_dir / "paired_comparisons.json", *plots_dir.glob("*.png")]
    manifest = {str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path) for path in manifest_files}
    (results_dir / "summary_artifact_checksums.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _make_plots(arms: Dict[str, Any], steps: List[int], output: Path) -> None:
    colors = {"A": "#2563eb", "B": "#dc2626"}
    metrics = [
        ("probe_mean_speed_km_h", "Probe mean speed (km/h)", "mean_speed_km_h"),
        ("probe_mean_route_progress_delta", "Probe route progress delta", "route_progress_delta"),
        ("probe_stationary_fraction", "Probe stationary fraction", "stationary_fraction"),
    ]
    for key, title, aggregate_key in metrics:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for arm in ("A", "B"):
            xs, ys = [], []
            for step in steps:
                cp = arms[arm]["checkpoints"].get(step, {})
                if cp.get("probe"):
                    value = cp["probe"]["aggregate"].get(key)
                    if value is not None:
                        xs.append(step); ys.append(value)
            if xs:
                ax.plot(xs, ys, marker="o", label=f"{arm} — {aggregate_key}", color=colors[arm])
        ax.set_title(title); ax.set_xlabel("Training environment steps"); ax.grid(alpha=.25); ax.legend()
        fig.tight_layout(); fig.savefig(output / f"{key}.png", dpi=150); plt.close(fig)

    for metric, title, aggregate_key in (
        ("clean_success_rate", "Full episode clean success rate", "clean_success_rate"),
        ("safety_failure_rate", "Full episode safety failure rate", "safety_failure_rate"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for arm in ("A", "B"):
            xs, ys = [], []
            for step in steps:
                cp = arms[arm]["checkpoints"].get(step, {})
                if cp.get("full_episode"):
                    value = cp["full_episode"]["aggregate"].get(aggregate_key)
                    if value is not None:
                        xs.append(step); ys.append(value)
            if xs:
                ax.plot(xs, ys, marker="o", label=arm, color=colors[arm])
        ax.set_ylim(-.05, 1.05); ax.set_title(title); ax.set_xlabel("Training environment steps")
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(output / f"full_{metric}.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for arm in ("A", "B"):
        xs, ys = [], []
        for step in steps:
            cp = arms[arm]["checkpoints"].get(step, {})
            if cp.get("probe"):
                sensitivity = cp["probe"].get("actor_state_sensitivity", {}).get("deterministic_action_std_across_fixed_observations")
                if sensitivity:
                    xs.append(step); ys.append(max(sensitivity))
        if xs:
            ax.plot(xs, ys, marker="o", label=arm, color=colors[arm])
    ax.set_title("Actor action sensitivity across fixed diagnostic observations")
    ax.set_xlabel("Training environment steps"); ax.set_ylabel("Maximum action SD")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(output / "actor_sensitivity.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    for axis, arm in zip(axes, ("A", "B")):
        coverage = arms[arm].get("coverage") or {}
        counts = coverage.get("geometry_step_counts", {})
        names = list(counts)
        axis.bar(names, [counts[name] for name in names], color=colors[arm])
        axis.set_title(f"{arm} — training steps by encountered geometry")
        axis.set_ylabel("Environment decisions"); axis.tick_params(axis="x", labelrotation=35)
        axis.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(output / "training_geometry_coverage.png", dpi=150); plt.close(fig)

    # Paired probe delta chart is filled after pairing JSON exists by its own data source.
    paired_path = output.parent / "paired_comparisons.json"
    if paired_path.is_file():
        paired = json.loads(paired_path.read_text(encoding="utf-8"))
        rows = [row for row in paired if row["data_kind"] == "PROBE_CENSORED_DIAGNOSTIC"]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        xs, ys = [], []
        for row in rows:
            value = row["metrics"]["route_progress_delta"]["mean_delta_b_minus_a"]
            if value is not None:
                xs.append(row["training_step"]); ys.append(value)
        if xs:
            ax.axhline(0, color="#444", linewidth=1); ax.plot(xs, ys, marker="o", color="#7c3aed")
        ax.set_title("Paired route-progress difference (B − A)")
        ax.set_xlabel("Training environment steps"); ax.set_ylabel("Mean paired difference")
        ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(output / "paired_route_progress_difference.png", dpi=150); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="s5_4_learnability_seed101_20261119")
    args = parser.parse_args()
    result = analyze(args.run_id)
    print(json.dumps({"run_id": result["run_id"], "arms": result["arms"],
                      "protocol_sha256": result["protocol_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
