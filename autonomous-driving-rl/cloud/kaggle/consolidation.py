"""Summarize a finished Stage 5 window without counting inherited replay twice."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path

from cloud.kaggle import campaign, control
from cloud.kaggle.common import WORK, read, verify, write
from cloud.kaggle.pilot import panel, summarize


def by_job(directory):
    return {row["job"]: row for path in directory.glob("*.json") if (row := read(path))}


def verified_metrics(job, directories):
    for directory in directories:
        marker_path, metrics_path = (directory / name for name in ("COMPLETE.json", "metrics.json"))
        if marker_path.is_file() and metrics_path.is_file():
            marker = read(marker_path)
            if marker["job"] != job:
                raise ValueError("Checkpoint metrics belong to another job")
            verify(metrics_path, marker["files"]["metrics.json"])
            return read(metrics_path), marker
    return {}, {}


def attempt(row, entry, observation, logs, archive, window):
    version = entry["version"]
    job, name = row["job"], row["session"]
    if any(x and (x["job"], x["version"]) != (job, version) for x in (observation, logs, archive)):
        raise ValueError("Post-campaign evidence does not match the pinned job version")
    plan = entry["planned"]
    candidates = [WORK / "harvest" / job, window / "final-shortlist" / name]
    verified = [a for a in archive.get("artifacts", {}).values() if a.get("path")]
    final = archive.get("artifacts", {}).get("final/state", {})
    if final.get("path"):
        candidates.insert(0, Path(final["path"]))
    stats, marker = verified_metrics(job, candidates)
    complete = [m for m in logs.get("markers", ()) if m[0] == "COMPLETE"]
    marks = logs.get("markers", ())
    parent_dir = WORK / "harvest" / plan["parent_job"]
    if plan.get("parent_step") is not None:
        parent_dir = WORK / "periodic" / plan["parent_job"] / f"step-{plan['parent_step']:012d}"
    verify(parent_dir / "COMPLETE.json", plan["parent_sha"])
    parent = read(parent_dir / "COMPLETE.json")
    baselines = parent["transitions"], parent["updates"]
    checkpoint_bounds = [
        (a["transitions"] - baselines[0], a["updates"] - baselines[1])
        for a in verified if a.get("purpose") == "SCREENING_ONLY"
    ]
    lower = [tuple(m[1:]) for m in marks] + checkpoint_bounds
    decisions = stats.get("transitions", max((m[0] for m in lower), default=None))
    updates = stats.get("updates", max((m[1] for m in lower), default=None))
    discards = None
    if stats:
        verify(parent_dir / "metrics.json", parent["files"]["metrics.json"])
        inherited = read(parent_dir / "metrics.json").get("localization_invalid_rows", 0)
        discards = stats.get("localization_invalid_rows", 0) - inherited
    if discards is not None and (not 0 <= discards <= decisions or discards % plan["workers"]):
        raise ValueError("Invalid final localization-discard count")
    outcomes = Counter(x["primary_reason"] for x in stats.get("episodes", ()) if x.get("primary_reason"))
    state_path = final.get("path")
    if entry.get("state_sha") and entry.get("output_version") == version:
        state_path = str(WORK/"harvest"/job)
    panels = {}
    for directory in candidates:
        for tier, name in (("Easy", "panel.json"), ("Medium", "panel-medium.json"),
                           ("Hard", "panel-hard.json"), ("Extreme", "panel-extreme.json")):
            path = directory / name
            if path.is_file() and marker:
                report = read(path)
                if report["model_sha"] != marker["files"]["model.zip"]:
                    raise ValueError("Evaluation panel belongs to another model")
                panels[tier] = summarize(panel(report, tier=tier))
    return dict(job=job, session=row["session"], account=row["account"], version=version,
        final_status=observation.get("status", archive.get("api_status", entry["status"])),
        parent=plan["parent_job"], parent_sha=plan["parent_sha"], device=plan["device"],
        curriculum=plan["phase"], profile={k: plan.get(k) for k in (
            "guided_actor_lr_scale", "guided_gradient_steps", "traffic_guidance", "traffic_teacher_sha",
            "autonomous_traffic", "localization_guard")},
        wall_log_seconds=logs.get("log_last_elapsed_seconds"), training_seconds=stats.get("seconds"),
        decisions=decisions, updates=updates, counters_exact=bool(stats or complete),
        valid_rows=decisions - discards if discards is not None else None, localization_discard=discards,
        online_live=stats.get("online_live_rows"), guided_live=stats.get("guided_live_rows"),
        online_evicted=stats.get("online_evicted_rows"), geometry_coverage=stats.get("decisions_by_geometry"),
        tier_decisions=stats.get("online_tier_decisions"), episode_sample_outcomes=dict(outcomes),
        full_state_path=state_path, verified_artifacts=verified, panels=panels,
        final_model_sha=marker.get("files", {}).get("model.zip"), f09=logs.get("f09"),
        learning_curve=[{"event": m[0], "decisions": m[1], "updates": m[2]} for m in marks],
        log_checkpoint_counts={key: len(logs.get(key, {})) for key in ("screen", "periodic")})


def campaign_report(window, guards, archive_root, output):
    document = campaign.manifest(window)
    registry = control.registry()
    statuses, logs = (by_job(window / key) for key in ("post-deadline-api", "post-deadline-metrics"))
    archived = {row["job"]: row for path in archive_root.glob("*/*/receipt.json") if (row := read(path))}
    grouped = {row["job"]: {"session": row["session"], "attempts": []} for row in document["sessions"]}
    for row in campaign.job_queue(guards, document):
        job = row["job"]
        entry = registry["jobs"][job]
        if entry.get("version"):
            grouped[row.get("supersedes", job)]["attempts"].append(
                attempt(row, entry, statuses.get(job, {}), logs.get(job, {}), archived.get(job, {}), window))
    workloads = list(grouped.values())
    attempts = [a for row in workloads for a in row["attempts"]]
    keys = ("decisions", "updates", "valid_rows", "wall_log_seconds", "training_seconds")
    totals = {k: {"value": sum(row[k] for row in attempts if row[k] is not None),
                  "attempts_observed": sum(row[k] is not None for row in attempts)} for k in keys}
    ledgers = {}
    for parent_id in {a["parent"] for a in attempts}:
        path, digest = control.parent_checkpoint(registry, parent_id)
        verify(path / "COMPLETE.json", digest)
        ledgers[parent_id] = read(path / "COMPLETE.json")["ledger"]
    shared = {}
    for ledger in ledgers.values():
        for key, digest in ledger.items():
            if shared.setdefault(key, digest) != digest:
                raise ValueError("Replay chunk identity has conflicting content")
    output.mkdir(parents=True, exist_ok=True)
    write(output / "campaign.json", {"window": document["window"], "recorded_at": control.now(),
        "workloads": workloads, "totals": totals, "session_hours_from_logs": totals["wall_log_seconds"]["value"] / 3600,
        "unique_inherited_guided_chunks": shared, "eligible_new_trajectory_chunks": 0,
        "runtime_note": "Elapsed time to last captured log event, not billed quota.",
        "replay_policy": "Full-state online rings are eligible for resume. Consolidation requires verified TRAIN trajectory chunks."})
    with (output / "workloads.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ("session", "account", "job", "version", "parent", "device", "curriculum", "profile",
                  "wall_log_seconds", "training_seconds", "decisions", "valid_rows", "updates", "localization_discard",
                  "attempts", "geometry_count", "episode_sample_outcomes", "full_state_path", "final_model_sha", "final_status")
        table = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        table.writeheader()
        for row in workloads:
            if not row["attempts"]:
                table.writerow({"session": row["session"], "attempts": 0})
                continue
            latest = row["attempts"][-1]
            table.writerow({**latest, "session": row["session"], "attempts": len(row["attempts"]),
                            "geometry_count": len(latest["geometry_coverage"]) if latest["geometry_coverage"] is not None else None})
    print(totals)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--window", type=Path, required=True)
    args.add_argument("--guards", type=Path, required=True)
    args.add_argument("--archive", type=Path, required=True)
    args.add_argument("--out", type=Path, required=True)
    p = args.parse_args()
    campaign_report(p.window, p.guards, p.archive, p.out)


if __name__ == "__main__":
    main()
