"""Convert paired launcher traces to chart-ready episode and tier CSVs."""
import argparse
import csv
from pathlib import Path

import numpy as np

from src.platform.episode import SAFETY_CRITICAL_FLAGS

PROFILES = ("off", "fixed", "adaptive")
TIERS = ("Easy", "Medium", "Hard", "Extreme")


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return round(float(np.mean(values)), 8) if len(values) else None


def rms(values):
    return round(float(np.sqrt(np.mean(np.square(values)))), 8) if len(values) else None


def metrics(trace):
    def series(field):
        return np.array([float(row[field]) for row in trace if row[field] != ""])

    raw = series("steering_raw")
    applied = series("steering_executed")
    actual = series("steering_simulator")
    changes = np.diff(applied)
    signs = np.sign(applied[np.abs(applied) >= 0.02])
    return {
        "steering_change_mean_abs": mean(np.abs(changes)),
        "steering_change_rms": rms(changes),
        "steering_reversal_rate": round(float(np.sum(np.diff(signs) != 0)) / max(len(applied) - 1, 1), 8),
        "steering_saturation_rate": mean(np.abs(applied) >= 0.98),
        "raw_steering_change_mean_abs": mean(np.abs(np.diff(raw))),
        "max_raw_to_simulator_steering_difference": round(float(np.max(np.abs(raw - actual))), 8),
        "max_executed_to_simulator_steering_difference": round(float(np.max(np.abs(applied - actual))), 8),
        "max_throttle_override": round(float(np.max(np.abs(
            series("throttle_brake_raw") - series("throttle_brake_executed")))), 8),
        "throttle_change_mean_abs": mean(np.abs(np.diff(series("throttle_brake_executed")))),
        "cross_track_mean_abs_m": mean(np.abs(series("cross_track_m"))),
        "yaw_rate_rms_rad_s": rms(series("yaw_rate_rad_s")),
        "lateral_accel_rms_m_s2": rms(series("lateral_accel_m_s2")),
        "lateral_jerk_rms_m_s3": rms(series("lateral_jerk_m_s3")),
    }


def build_rows(cases, root):
    rows = []
    for index, case in enumerate(cases, 1):
        prefix = f"{case['tier'].lower()}-{index:02d}"
        recorded = {}
        for profile in PROFILES:
            run = root / f"{prefix}-{profile}"
            episode, = read_rows(run / "episodes.csv")
            assert episode["split"] == "TRAIN"
            assert (episode["tier"], episode["sequence"], episode["geometry_generation_seed"],
                    episode["environment_seed"]) == tuple(case[k] for k in (
                        "tier", "family", "geometry_seed", "environment_seed"))
            if episode["has_technical_failure"] != "False":
                raise ValueError(f"Unverifiable steering comparison in {prefix}-{profile}")
            trace = read_rows(next(run.glob("steering-*.csv")))
            assert len(trace) == int(episode["episode_steps"])
            recorded[profile] = episode, trace
        compared_steps = min(len(trace) for _, trace in recorded.values())
        for profile, (episode, trace) in recorded.items():
            rows.append({
                **case, "profile": profile, "outcome": episode["primary_terminal_reason"],
                "clean_success": int(episode["clean_success"] == "True"),
                "safety_failure": int(any(episode[flag] == "True" for flag in SAFETY_CRITICAL_FLAGS)),
                "timeout": int(episode["timeout"] == "True"),
                "route_completion": float(episode["final_route_completion"]),
                "mean_speed_kmh": float(episode["mean_speed_kmh"]),
                "time_to_clean_success_s": float(episode["time_to_clean_success_s"])
                if episode["clean_success"] == "True" else None,
                "episode_steps": len(trace), "compared_steps": compared_steps,
                **metrics(trace[:compared_steps]),
            })
    return rows


def summarize(rows):
    summaries = []
    for tier in (*TIERS, "All"):
        for profile in PROFILES:
            selected = [r for r in rows if r["profile"] == profile and (tier == "All" or r["tier"] == tier)]
            reference = [r for r in rows if r["profile"] == "off" and (tier == "All" or r["tier"] == tier)]
            common = [(r, ref) for r, ref in zip(selected, reference) if r["clean_success"] and ref["clean_success"]]
            summarized = {k: mean([r[k] for r in selected if r[k] is not None])
                          for k in ("route_completion", "mean_speed_kmh", *metrics_columns())}
            summaries.append({
                "tier": tier, "profile": profile, "cases": len(selected),
                **{k: sum(r[k] for r in selected) for k in ("clean_success", "safety_failure", "timeout")},
                "common_clean_vs_off": len(common),
                "clean_to_safety_failure": sum(r["safety_failure"] and ref["clean_success"]
                                               for r, ref in zip(selected, reference)),
                "common_clean_time_delta_s": mean([r["time_to_clean_success_s"] - ref["time_to_clean_success_s"]
                                                   for r, ref in common]),
                **summarized,
            })
    return summaries


def metrics_columns():
    return ("steering_change_mean_abs", "steering_change_rms", "steering_reversal_rate",
            "steering_saturation_rate", "throttle_change_mean_abs", "cross_track_mean_abs_m",
            "yaw_rate_rms_rad_s", "lateral_accel_rms_m_s2", "lateral_jerk_rms_m_s3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    rows = build_rows(read_rows(args.cases), args.runs)
    write_rows(args.out / "steering_episodes.csv", rows)
    write_rows(args.out / "steering_summary.csv", summarize(rows))


if __name__ == "__main__":
    main()
