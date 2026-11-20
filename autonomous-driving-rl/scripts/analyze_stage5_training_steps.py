"""Summarize existing per-step SAC pilot telemetry without simulator access."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def analyze(path: Path, bucket_steps: int = 1000) -> Dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result: Dict[str, Any] = {
        "source": str(path), "rows": len(rows), "bucket_steps": bucket_steps,
        "training_geometry_ids": sorted({r.get("geometry_id") for r in rows if r.get("geometry_id")}),
        "terminal_reasons": dict(Counter(r.get("primary_reason") for r in rows if r.get("primary_reason"))),
        "buckets": [],
    }
    for start in range(0, max((int(r["training_step"]) for r in rows), default=0), bucket_steps):
        bucket = [r for r in rows if start < int(r["training_step"]) <= start + bucket_steps]
        if not bucket:
            continue
        def mean(field: str) -> float | None:
            values = [float(r[field]) for r in bucket if r.get(field) is not None]
            return float(np.mean(values)) if values else None
        actions = np.asarray([
            [r["steering"], r["throttle_brake"]]
            for r in bucket if r.get("steering") is not None and r.get("throttle_brake") is not None
        ], dtype=np.float64)
        result["buckets"].append({
            "start_step_exclusive": start, "end_step_inclusive": start + bucket_steps,
            "n": len(bucket), "mean_steering": float(actions[:, 0].mean()) if len(actions) else None,
            "mean_throttle_brake": float(actions[:, 1].mean()) if len(actions) else None,
            "throttle_fraction": float(np.mean(actions[:, 1] > 0.05)) if len(actions) else None,
            "brake_fraction": float(np.mean(actions[:, 1] < -0.05)) if len(actions) else None,
            "mean_speed_km_h": mean("speed_km_h"), "mean_route_completion": mean("route_completion"),
            "mean_delta_route": mean("delta_route"), "mean_progress_reward": mean("progress_reward"),
            "mean_time_cost": mean("time_cost"),
            "terminal_reward_sum": float(sum(float(r.get("terminal_reward") or 0.0) for r in bucket)),
            "terminal_outcomes": dict(Counter(r.get("primary_reason") for r in bucket if r.get("primary_reason"))),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.steps)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
