"""Resolve the 22 logical learners from versioned second-window harvest receipts."""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cloud.kaggle import campaign
from cloud.kaggle.common import WORK, read, sha, verify, write

ROOT = WORK / "second-window-v3"


def source_inventory():
    from cloud.kaggle.consolidated_replay import assert_contract
    from cloud.kaggle.local_risk import LOCK

    archive = Path(os.environ["LOCALAPPDATA"]) / "XeTuHanh" / "Stage5CampaignHarvest"
    anchor = read(Path(read(LOCK)["full_state"]) / "COMPLETE.json")
    rows, candidates = {}, defaultdict(list)
    for suffix in ("", "-extension"):
        window = WORK / f"central12-second{suffix}"
        document = campaign.manifest(window)
        deadline = document["window"]["deadline_epoch"]
        by_job = {q["job"]: q for q in campaign.job_queue(window, document)}
        for session in document["sessions"]:
            rows[session["session"]] = {"session": session["session"], "account": session["account"],
                                        "attempts": [], "source_status": "AWAITING_ARCHIVE_RECEIPT"}
        original = archive / f"{deadline}{suffix}"
        selective = original.with_name(original.name + "-selective")
        catalog_dir = original.with_name(original.name + "-catalog")
        receipts = [receipt for directory in (original, selective, catalog_dir)
                    for receipt in sorted(directory.glob("*/*/receipt.json"))]
        for receipt in receipts:
            r = read(receipt)
            if r["job"] not in by_job:
                continue
            q = by_job[r["job"]]
            logical = q.get("logical_session", q["session"])
            current = rows[logical]
            markers = r.get("registered_markers", {})
            catalog = r.get("file_catalog") or {}
            artifacts = r.get("artifacts", {})
            current["attempts"].append({
                "job": r["job"], "version": r["version"], "api": r.get("api_status"), "at": r.get("at"),
                "selective": receipt.is_relative_to(selective),
                "final_local": sum(a.get("path") is not None and not a.get("policy_only")
                                   for k,a in artifacts.items() if k.startswith("final/")),
                "periodic_local": sum(a.get("path") is not None and not a.get("policy_only")
                                      for k,a in artifacts.items() if k.startswith("periodic/")),
                "policy_local": sum(a.get("path") is not None for k,a in artifacts.items() if k.startswith("screening/")),
                "periodic_remote": max(len(markers.get("periodic", {})),
                                       max(0, catalog.get("state_indexes", 0)-int(catalog.get("execution_envelope", False)))),
                "policy_remote": max(len(markers.get("screening", {})), catalog.get("screening_markers", 0)),
                "remote_marker_evidence": "log" if markers else "file_catalog" if catalog else "unobserved",
                "complete_scan": bool(r.get("scan_complete")),
                "last_log_steps": r.get("log_steps") or r.get("last_log_steps"),
                "errors": {k:x["status"] for k,x in artifacts.items() if x.get("status")},
                "archive_error": r.get("small_error") or r.get("full_error"),
            })
            if r.get("api_status"):
                current["source_status"] = "NO_VERIFIED_COMPATIBLE_STATE"
            for kind, item in artifacts.items():
                if not item.get("path") or not kind.startswith(("final/", "periodic/")):
                    continue
                folder = Path(item["path"])
                marker = folder / "COMPLETE.json"
                try:
                    verify(marker, item["sha256"])
                    metadata = read(marker)
                    assert_contract(metadata["identity"], anchor["identity"])
                    model = folder / "model.zip"
                    verify(model, metadata["files"]["model.zip"])
                except (OSError, ValueError, KeyError) as exc:
                    current["source_status"] = f"STATE_REJECTED_{type(exc).__name__}"
                    continue
                candidates[logical].append({"session": logical, "kind": kind, "job": r["job"], "version": r["version"],
                    "state": str(folder), "state_sha": item["sha256"], "model": str(model),
                    "model_sha": metadata["files"]["model.zip"], "updates": metadata["updates"],
                    "curriculum": q["workload"], "recovery": "logical_session" in q, "result": metadata["result"],
                    "policy_only": bool(item.get("policy_only"))})
    sources = {}
    for logical, choices in candidates.items():
        # Use a final state when available, preferring the newer recovery of that logical session.
        best = max(choices, key=lambda c: (c["recovery"], c["kind"].startswith("final/"), c["updates"],
                                           not c["policy_only"]))
        sources[logical] = best
        rows[logical]["source_status"] = "COMPATIBLE_POLICY_VERIFIED"
    return {"learners": rows, "sources": sources, "source_counts": dict(Counter(r["source_status"] for r in rows.values())),
            "historical_replay_admitted": 0}


def probes():
    from cloud.kaggle.observation_sensitivity import pack_traffic
    from cloud.kaggle.observation_capture import OUTPUT

    obs = np.load(OUTPUT / "capture/extreme_exact.npz", allow_pickle=False)["policy"]
    frames = read(OUTPUT / "capture/extreme_exact.json")["frames"]
    samples = {}
    for step in (2456, 2973):
        actors = np.asarray(frames[step]["actors"], dtype=np.float32)
        for radius in (50., 50.01):
            updated = actors.copy()
            updated[3, :2] *= radius / np.linalg.norm(actors[3, :2])
            samples[f"edge{step}_{radius}"] = pack_traffic(obs[step], updated)
        samples[f"no_target_{step}"] = pack_traffic(obs[step], actors[1:])
        samples[f"original_{step}"] = obs[step]
    for name, row in {"closing": [25, 0, -10, 0, 0, 4.5, 1.8],
                      "crossing": [23, -8, 0, 7, np.pi/2, 4.5, 1.8],
                      "separating": [31.4, 0, 6, 0, 0, 4.5, 1.8]}.items():
        samples[name] = pack_traffic(obs[2973], [row])
    with np.load(OUTPUT / "capture/medium_turn.npz", allow_pickle=False) as medium:
        samples["medium_turn"] = medium["policy"][81]
    return samples


def sensitivity(model, samples):
    result, _ = model.predict(np.stack(list(samples.values())), deterministic=True)
    return dict(zip(samples, result.tolist(), strict=True))


def screen_sources(data):
    from cloud.kaggle.local_risk import V2_ROOT, TIERS
    from cloud.kaggle.local_workers import Simulators
    from cloud.kaggle.smooth_traffic_eval import memory, worker_budget
    from cloud.kaggle.v3_train import screen_fixtures

    baseline = read(V2_ROOT / "confirmation-05000.json")["cases"]
    short = screen_fixtures(baseline)
    reference = {tier: {**doc, "cases": doc["cases"][-1:]} for tier, doc in short.items()}
    results = ROOT / "source-screen"
    results.mkdir(exist_ok=True)
    budget = memory()
    with Simulators(worker_budget(budget)) as sim:
        # Two sources give 8 tasks while each policy stays cached for the wave.
        from cloud.kaggle.runtime import pool

        lookup = {row["geometry_id"]: row for row in pool("eval")}
        pending = []
        for session, pin in data["sources"].items():
            path = results / f"{session}.json"
            if not path.exists() or read(path)["model_sha"] != pin["model_sha"]:
                pending.append((session, pin))
        for offset in range(0, len(pending), 2):
            group = pending[offset:offset+2]
            jobs = [{"kind": "eval", "policy": Path(pin["model"]), "policy_sha": pin["model_sha"],
                     "row": lookup[row["geometry"]], "seed": row["seed"], "risk_metrics": True}
                    for session, pin in group for tier in TIERS for row in reference[tier]["cases"]]
            outputs, performance = sim.run(jobs)
            for i, (session, pin) in enumerate(group):
                write(results / f"{session}.json", {"model_sha": pin["model_sha"], "session": session,
                      "cases": dict(zip(TIERS, outputs[i*4:i*4+4], strict=True)), "performance": performance,
                      "budget_at_start": budget})
            print("SOURCE_SCREEN", [session for session, _ in group],
                  [(r["outcome"], r["progress"]) for r in outputs], flush=True)


def main():
    import torch
    from cloud.kaggle.runtime import load_policy
    from cloud.kaggle.local_risk import LOCK

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory-only", action="store_true")
    p.add_argument("--screen", action="store_true")
    args = p.parse_args()
    ROOT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    value = source_inventory()
    write(ROOT / "inventory22.json", value)
    print("INVENTORY", value["source_counts"], flush=True)
    if args.inventory_only:
        return
    samples = probes()
    lock = read(LOCK)
    survey = {"V2": sensitivity(load_policy(Path(lock["model"])), samples)}
    for session, entry in value["sources"].items():
        survey[session] = sensitivity(load_policy(Path(entry["model"])), samples)
    write(ROOT / "source-sensitivity.json", survey)
    with (ROOT / "source-sensitivity.csv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(("session", "probe", "steering", "throttle_brake"))
        writer.writerows((session, probe, *action) for session, results in survey.items() for probe, action in results.items())
    if args.screen:
        screen_sources(value)


if __name__ == "__main__":
    main()
