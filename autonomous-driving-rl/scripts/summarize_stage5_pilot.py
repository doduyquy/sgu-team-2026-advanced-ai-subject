"""Aggregate observed S5.3A episode records and produce empirical learning curves."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]

METRICS = ("final_route_completion", "clean_success", "safety_failure", "timeout", "episode_return",
           "stationary_fraction", "mean_moving_speed_km_h", "mean_throttle", "mean_brake",
           "mean_steering", "steering_std", "steering_saturation_fraction")

def read_records(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def avg(values: List[Any]) -> float | None:
    return float(statistics.mean(values)) if values else None

def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    complete = [r for r in records if not r.get("interrupted", False)]
    result: Dict[str, Any] = {"episodes_completed": len(complete), "episodes_interrupted": len(records)-len(complete)}
    for metric in METRICS:
        result[metric] = avg([float(r[metric]) for r in complete if r.get(metric) is not None])
    if complete:
        result["median_route_completion"] = float(statistics.median(r["final_route_completion"] for r in complete))
    return result

def build_report(run_dir: Path, out_dir: Path) -> Dict[str, Any]:
    episode_paths = {step: run_dir / "diagnostics" / f"step_{step:05d}_episodes.jsonl" for step in (0, 5000, 10000)}
    series = {step: read_records(path) for step, path in episode_paths.items()}
    expected = [(r["geometry_id"], r["environment_seed"]) for r in series[0]]
    for step, rows in series.items():
        actual = [(r["geometry_id"], r["environment_seed"]) for r in rows]
        if actual != expected:
            raise ValueError(f"Diagnostic schedule mismatch at checkpoint step {step}")
        if len(rows) != 24 or any(r.get("interrupted") for r in rows):
            raise ValueError(f"Expected 24 full episodes at {step} steps, got {len(rows)}")
    report: Dict[str, Any] = {
        "classification": "internal PILOT_DIAGNOSTIC development data; not official VALIDATION or TEST",
        "checkpoints": {}, "paired_differences_vs_untrained": {}, "per_tier": {},
    }
    for step, rows in series.items():
        report["checkpoints"][str(step)] = aggregate(rows)
    baseline = {r["geometry_id"]: r for r in series[0]}
    for step in (5000, 10000):
        current = {r["geometry_id"]: r for r in series[step]}
        report["paired_differences_vs_untrained"][str(step)] = {}
        for metric in METRICS:
            diffs = []
            for geometry_id, row in current.items():
                a, b = row.get(metric), baseline[geometry_id].get(metric)
                if a is not None and b is not None:
                    diffs.append(float(a) - float(b))
            report["paired_differences_vs_untrained"][str(step)][metric] = {
                "mean": avg(diffs), "n": len(diffs),
            }
    tiers = sorted({r["tier"] for rows in series.values() for r in rows})
    for tier in tiers:
        report["per_tier"][tier] = {
            str(step): aggregate([r for r in rows if r["tier"] == tier])
            for step, rows in series.items()
        }
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "scientific_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    with (out_dir / "learning_curve_route_completion.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["training_steps", "mean_route_completion", "median_route_completion", "clean_success_rate", "safety_failure_rate", "timeout_rate"])
        for step in (0,5000,10000):
            item=report["checkpoints"][str(step)]
            w.writerow([step,item.get("final_route_completion"),item.get("median_route_completion"),item.get("clean_success"),item.get("safety_failure"),item.get("timeout")])
    points=[]
    for x, step in enumerate((0,5000,10000)):
        y=report["checkpoints"][str(step)].get("final_route_completion")
        points.append((90+x*250, 220-(float(y or 0)*180), step, float(y or 0)))
    poly=" ".join(f"{x},{y:.1f}" for x,y,_,_ in points)
    circles="".join(f'<circle cx="{x}" cy="{y:.1f}" r="5"/><text x="{x-25}" y="{y-10:.1f}">{v:.3f}</text><text x="{x-25}" y="250">{step}</text>' for x,y,step,v in points)
    svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="620" height="300" viewBox="0 0 620 300"><rect width="100%" height="100%" fill="white"/><text x="190" y="24" font-family="sans-serif" font-size="16">Observed route completion (internal diagnostic)</text><line x1="70" y1="220" x2="580" y2="220" stroke="black"/><line x1="70" y1="40" x2="70" y2="220" stroke="black"/><polyline points="{poly}" fill="none" stroke="#2457a7" stroke-width="3"/>{circles}<text x="270" y="280" font-family="sans-serif">Training steps</text><text x="8" y="130" transform="rotate(-90 8,130)" font-family="sans-serif">Mean route completion</text><text x="80" y="290" font-family="sans-serif" font-size="11">PILOT_DIAGNOSTIC internal TRAIN split; empirical checkpoint measurements</text></svg>'''
    (out_dir / "learning_curve_route_completion.svg").write_text(svg, encoding="utf-8")
    return report

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--run-dir",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
    print(json.dumps(build_report(a.run_dir,a.output_dir),indent=2))

if __name__ == "__main__": main()
