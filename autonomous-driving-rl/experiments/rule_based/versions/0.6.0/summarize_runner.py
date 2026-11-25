"""Verify completed TRAIN experiment artifacts and export an immutable paired report."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.logging.agent_experiment import sha256_file, write_json


def load_run(directory, variant):
    directory = Path(directory)
    state = json.loads((directory / "run_state.json").read_text(encoding="utf-8"))
    if state["status"] != "COMPLETED":
        raise ValueError(f"Cannot publish a complete comparison from run status {state['status']}: {directory}")
    manifest = json.loads((directory / "resolved_cases.json").read_text(encoding="utf-8"))
    if any(case["split"] != "TRAIN" for case in manifest["cases"]):
        raise ValueError("Development report may contain TRAIN cases only")
    provenance = json.loads((directory / variant / "agent_provenance.json").read_text(encoding="utf-8"))
    for name, expected in provenance["source_sha256"].items():
        if sha256_file(directory / variant / "sources" / name) != expected:
            raise ValueError(f"Source snapshot was modified: {name}")
    config_hash = hashlib.sha256(json.dumps(provenance["configuration"], sort_keys=True,
                               separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if config_hash != provenance["configuration_sha256"]:
        raise ValueError("Configuration hash mismatch")
    with (directory / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["variant"] == variant]
    if len(rows) != len(manifest["cases"]) or state["completed_episodes"] != state["expected_episodes"]:
        raise ValueError("Episode count does not match declared run coverage")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate case rows")
    by_case = {row["case_id"]: row for row in rows}
    traces = {}
    for index, case in enumerate(manifest["cases"]):
        row = by_case[case["case_id"]]
        if row["technical_error"]:
            raise ValueError("Technical errors must be resolved/reported separately before this comparison")
        path = directory / variant / f"episode_{index:03d}" / "decisions.jsonl.gz"
        count = 0
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for count, line in enumerate(stream, 1):
                trace = json.loads(line)
                if trace["case_id"] != case["case_id"] or trace["step_index"] != count - 1:
                    raise ValueError(f"Non-contiguous/wrong-case decision trace: {path}")
                if len(trace["agent_input"]["core_observation"]["features"]) != 259:
                    raise ValueError("Invalid observation dimension in recorded trace")
        if count != int(row["steps"]):
            raise ValueError(f"Trace length does not match result row: {path}")
        traces[case["case_id"]] = {"rows": count, "sha256": sha256_file(path)}
    return by_case, provenance, traces


def validate_paired_definitions(first, second):
    """Shard order/index may differ; physical TRAIN case definitions may not."""
    keys = ("case_id", "split", "tier", "sequence", "geometry_generation_seed",
            "geometry_sha256", "environment_seed", "traffic_density", "horizon_steps")
    differences = [key for key in keys if first.get(key) != second.get(key)]
    if differences:
        raise ValueError("Mismatched paired case definition: " + ", ".join(differences))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--baseline-variant", choices=("baseline", "current"), default="baseline")
    parser.add_argument("--candidate-variant", choices=("baseline", "current"), default="current")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    baseline, baseline_provenance, baseline_traces = load_run(args.baseline_run, args.baseline_variant)
    candidate, candidate_provenance, candidate_traces = load_run(args.candidate_run, args.candidate_variant)
    cases = [case for case in candidate if case in baseline]
    if not cases:
        raise ValueError("No shared cases for a paired comparison")
    baseline_cases = json.loads((args.baseline_run / "resolved_cases.json").read_text(encoding="utf8"))["cases"]
    candidate_cases = json.loads((args.candidate_run / "resolved_cases.json").read_text(encoding="utf8"))["cases"]
    for case_id in cases:
        old_index, old_case = next((i, c) for i, c in enumerate(baseline_cases) if c["case_id"] == case_id)
        new_index, new_case = next((i, c) for i, c in enumerate(candidate_cases) if c["case_id"] == case_id)
        validate_paired_definitions(old_case, new_case)
        def environment_config(root, variant, index):
            return json.loads((root / variant / f"episode_{index:03d}" / "environment.json").read_text(encoding="utf8"))["configuration"]
        if environment_config(args.baseline_run, args.baseline_variant, old_index) != environment_config(args.candidate_run, args.candidate_variant, new_index):
            raise ValueError("Mismatched paired simulator configuration: " + case_id)
    paired = []
    for case in cases:
        before, after = baseline[case], candidate[case]
        old_success, new_success = before["clean_success"] == "True", after["clean_success"] == "True"
        paired.append({
            "case_id": case, "baseline_success": old_success, "candidate_success": new_success,
            "baseline_reason": before["terminal_reason"], "candidate_reason": after["terminal_reason"],
            "baseline_completion": float(before["route_completion"]),
            "candidate_completion": float(after["route_completion"]),
            "baseline_simulation_time_s": float(before["simulation_time_s"]),
            "candidate_simulation_time_s": float(after["simulation_time_s"]),
            "baseline_speed_kmh": float(before["mean_speed_kmh"]),
            "candidate_speed_kmh": float(after["mean_speed_kmh"]),
            "candidate_latency_p95_ms": float(after["act_latency_p95_ms"]),
            "new_success": not old_success and new_success,
            "lost_success": old_success and not new_success,
            "lower_completion": float(after["route_completion"]) + 1e-6 < float(before["route_completion"]),
        })
    summary = {
        "noncanonical": True, "split": "TRAIN", "paired_cases": len(cases),
        "baseline_run": str(args.baseline_run), "candidate_run": str(args.candidate_run),
        "baseline_variant": args.baseline_variant, "candidate_variant": args.candidate_variant,
        "baseline_successes": sum(row["baseline_success"] for row in paired),
        "candidate_successes": sum(row["candidate_success"] for row in paired),
        "baseline_crashes": sum(baseline[case]["crash"] == "True" for case in cases),
        "candidate_crashes": sum(candidate[case]["crash"] == "True" for case in cases),
        "new_success_cases": [row["case_id"] for row in paired if row["new_success"]],
        "lost_success_cases": [row["case_id"] for row in paired if row["lost_success"]],
        "lower_completion_cases": [row["case_id"] for row in paired if row["lower_completion"]],
        "unmatched_baseline_cases": [case for case in baseline if case not in candidate],
        "unmatched_candidate_cases": [case for case in candidate if case not in baseline],
        "baseline_provenance": baseline_provenance, "candidate_provenance": candidate_provenance,
        "verified_trace_hashes": {"baseline": baseline_traces, "candidate": candidate_traces},
        "comparison_scope": "Policy plus shared input-extractor change; not a policy-only ablation",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "comparison.json", summary, exclusive=True)
    with (args.output_dir / "paired_cases.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    lines = ["# Rule-based paired TRAIN development comparison", "",
             f"Baseline: `{args.baseline_run.name}/{args.baseline_variant}`; candidate: `{args.candidate_run.name}/{args.candidate_variant}`.", "",
             f"Matched cases: {len(cases)}; clean success {summary['baseline_successes']} → {summary['candidate_successes']}; "
             f"crashes {summary['baseline_crashes']} → {summary['candidate_crashes']}.", "",
             "Both runs completed. Source/configuration hashes and every recorded trace row count were verified.", "",
             "This compares the policy and extractor together. It is not a canonical benchmark or policy-only ablation.", "",
             "| Case | Baseline | Candidate | Completion before → after |", "|---|---|---|---|"]
    for row in paired:
        lines.append(f"| {row['case_id']} | {row['baseline_reason']} | {row['candidate_reason']} | "
                     f"{row['baseline_completion']:.4f} → {row['candidate_completion']:.4f} |")
    lines += ["", "Unmatched baseline cases (excluded from paired score):", ""]
    lines.extend("- " + case for case in summary["unmatched_baseline_cases"])
    lines += ["", "Lower completion remains visible in paired_cases.csv; faster failure is not an improvement.", ""]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("baseline_provenance", "candidate_provenance", "verified_trace_hashes")}, indent=2))


if __name__ == "__main__":
    main()
