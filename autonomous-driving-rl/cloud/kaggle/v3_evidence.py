"""Export local CSV summaries and checksum manifests for the two Stage 5 tracks."""
from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path
import shutil
import stat

from cloud.kaggle.common import WORK, read, sha, write
from cloud.kaggle.local_risk import LOCK, TIERS
from cloud.kaggle.second_window_harvest import archive_root
from cloud.kaggle.second_window_inventory import ROOT, source_inventory
from cloud.kaggle.smooth_traffic_eval import ROOT as SMOOTH


def table(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def evidence(directory, files):
    return {p.relative_to(directory).as_posix(): sha(p) for p in sorted(files) if p.is_file()}


def logical_bytes(roots):
    total = 0
    seen = set()
    for directory in roots:
        for item in directory.rglob("*"):
            try:
                file = item.stat()
            except FileNotFoundError:
                continue  # A download staging file may have already been finalized.
            if stat.S_ISREG(file.st_mode):
                identity = (file.st_dev, file.st_ino) if file.st_ino else item
                if identity not in seen:
                    total += file.st_size
                    seen.add(identity)
    return total


def main():
    from cloud.kaggle.runtime import check_state
    from cloud.kaggle.common import verify

    lock = read(LOCK)
    verify(Path(lock["model"]), lock["model_sha256"])
    check_state(Path(lock["full_state"]), lock["full_state_manifest_sha256"])
    sandbox = read(SMOOTH / "sandbox.json")
    table(SMOOTH / "sandbox.csv", [{
        **{key: row[key] for key in ("geometry", "seed", "outcome", "progress", "clean", "speed_km_h")},
        **row["traffic"],
    } for row in sandbox["cases"]])
    table(SMOOTH / "edge.csv", [{k: row[k] for k in ("step", "baseline_jump", "blended_jump", "reduction")}
                                  for row in read(SMOOTH / "offline.json")["edge"]])
    write(SMOOTH / "manifest.json", {"model_sha": lock["model_sha256"], "wrapper_promoted": False, "updates": 0,
          "files": evidence(SMOOTH, [SMOOTH / x for x in ("offline.json", "sandbox.json", "sandbox.csv", "edge.csv")])})

    inventory = source_inventory()
    write(ROOT / "inventory22.json", inventory)
    pipeline = ROOT / "v3"
    result = read(pipeline / "result.json")
    waves = {path.parent.name: read(path) for path in ROOT.glob("donor-*/WAVE.json")}
    trained = Counter()
    admitted_by_tier = Counter()
    for row in result.get("audit", []):
        if row.get("admitted"):
            trained[row["wave"]] += row["admitted"]
            admitted_by_tier[row["kind"], row["tier"]] += row["admitted"]
    sensitivities = read(ROOT / "source-sensitivity.json")
    overview = []
    for session, item in sorted(inventory["learners"].items()):
        pin = inventory["sources"].get(session, {})
        wave = waves.get("donor-" + session, {})
        recorded = wave.get("records", [])
        sampled = sensitivities.get(session, {})
        screen_file = ROOT / "source-screen" / f"{session}.json"
        screen = read(screen_file)["cases"] if screen_file.is_file() else {}
        attempt = next((a for a in item["attempts"] if a["job"] == pin.get("job") and
                        a.get("api") == "CLOUD_SUCCEEDED"), {})
        row = {"session": session, "account": item["account"], "job": pin.get("job"),
               "version": pin.get("version"), "api": attempt.get("api"), "artifact_result": pin.get("result"),
               "source_status": item["source_status"],
               "metadata_and_policy": bool(pin), "full_state_local": bool(pin) and not pin["policy_only"],
               "periodic_full_state_local": any(a["periodic_local"] > 0 for a in item["attempts"]),
               "source_kind": pin.get("kind"), "policy_sha256": pin.get("model_sha"),
               "trajectory_policy_sha256": wave.get("snapshot", {}).get("policy_sha"),
               "valid_collected_rows": sum(r.get("rows", 0) for r in recorded),
               "admitted_rows": trained["donor-" + session],
               "screening_markers_seen": max((a["policy_remote"] for a in item["attempts"]), default=0),
               "periodic_markers_seen": max((a["periodic_remote"] for a in item["attempts"]), default=0),
               "marker_catalog_available": any(a["remote_marker_evidence"] != "unobserved" for a in item["attempts"]),
               "range_jump": max((abs(sampled[f"edge{i}_50.0"][1]-sampled[f"edge{i}_50.01"][1])
                                  for i in (2456, 2973)), default=None) if sampled else None}
        row.update({f"screen_{tier.lower()}": screen.get(tier, {}).get("outcome") for tier in TIERS})
        overview.append(row)
    table(ROOT / "learners22.csv", overview)
    table(ROOT / "admitted-tiers.csv", [{"pool": p, "tier": t, "rows": n}
                                       for (p, t), n in sorted(admitted_by_tier.items())])
    comparison = []
    for directory in (ROOT / "tuned-pilot", pipeline):
        for record in sorted(directory.glob("confirmation-*.json")):
            for tier, pair in read(record)["pairs"].items():
                fields = ("clean_successes", "safety_failures", "timeouts", "mean_progress",
                          "mean_speed_km_h", "mean_success_seconds")
                comparison.append({"run": directory.name, "milestone": record.stem, "tier": tier,
                                   **{f"v2_{key}": pair["reference"][key] for key in fields},
                                   **{f"candidate_{key}": pair["candidate"][key] for key in fields}})
    table(ROOT / "comparison.csv", comparison)
    archive = archive_root()
    archives = sorted((p for p in archive.iterdir() if p.is_dir() and p.name.startswith("1795782600")))
    write(ROOT / "manifest.json", {
        "selected": result["selected"], "parent_sha": lock["full_state_manifest_sha256"],
        "complete_policies": sum(row["metadata_and_policy"] for row in overview),
        "complete_full_states": sum(row["full_state_local"] for row in overview),
        "periodic_full_state_learners": sum(row["periodic_full_state_local"] for row in overview),
        "historical_cross_learner_replay_rows": 0, "donors_used": len([row for row in overview if row["admitted_rows"]]),
        "new_unique_donor_rows": sum(trained[label] for label in trained if label.startswith("donor")),
        "archive_bytes": logical_bytes(archives), "local_v3_bytes": logical_bytes([ROOT]),
        "free_archive_bytes": shutil.disk_usage(archive).free, "free_local_bytes": shutil.disk_usage(ROOT).free,
        "files": evidence(ROOT, [ROOT / name for name in ("inventory22.json", "learners22.csv", "source-sensitivity.csv",
                   "admitted-tiers.csv", "comparison.csv", "pilot/result.json", "tuned-pilot/result.json",
                   "v3/result.json", "v3/resume-proof.json", "preservation.json")]
                   + list(ROOT.glob("source-screen/S*.json"))
                   + list(ROOT.glob("final-demo/*/run_integrity.json"))
                   + list(ROOT.glob("final-demo/*/episodes.csv"))
                   + list(pipeline.glob("confirmation-*.json"))),
    })


if __name__ == "__main__":
    main()
