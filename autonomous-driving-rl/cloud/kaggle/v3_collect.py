"""Collect checksum-bound TRAIN trajectories for Consolidation V3."""
from __future__ import annotations

import argparse
from pathlib import Path

from cloud.kaggle.common import read, sha, write
from cloud.kaggle.local_workers import Simulators, stage_source
from cloud.kaggle.second_window_inventory import ROOT, source_inventory
from cloud.kaggle.smooth_traffic_eval import memory, worker_budget

TIERS = ("Easy", "Medium", "Hard", "Extreme")


def prepare_wave(label, state, digest, kind, stream, cases):
    destination = ROOT / label
    destination.mkdir(parents=True, exist_ok=True)
    donor = stage_source(state / "model.zip", state / "COMPLETE.json", digest, destination / "donor")
    jobs = []
    for index, (tier, geometry, seed) in enumerate(cases):
        folder = destination / f"chunk-{index:03d}"
        jobs.append({"kind": "collect", "snapshot": donor["snapshot"], "snapshot_sha": donor["snapshot_sha"],
                     "policy": donor["snapshot"] / "policy.zip", "policy_sha": donor["policy_sha"],
                     "geometry": geometry, "seed": seed, "stream": stream, "sequence": index, "out": folder})
    return {"name": label, "folder": destination, "kind": kind, "stream": stream, "source_state": digest,
            "donor": {**donor, "snapshot": str(donor["snapshot"])}, "cases": cases, "jobs": jobs}


def collect_waves(specs, simulator=None):
    jobs = [job for wave in specs for job in wave["jobs"]]
    if not jobs:
        return []
    submission = [job for job in jobs if not (job["out"] / "CHUNK.json").is_file()]
    start_memory = memory()
    workers = worker_budget(start_memory)
    if submission and simulator is None:
        with Simulators(workers) as sim:
            data, perf = sim.run(submission)
    elif submission:
        data, perf = simulator.run(submission)
    else:
        data, perf = (), {}
    outputs = {job["out"]: result for job, result in zip(submission, data, strict=True)}
    result = []
    for wave in specs:
        records = []
        for (tier, geometry, seed), job in zip(wave["cases"], wave["jobs"], strict=True):
            path = job["out"]
            row = {"tier": tier, "geometry": geometry, "seed": seed, "chunk": path.name}
            if path in outputs and "discard" in outputs[path]:
                records.append({**row, **outputs[path]})
                continue
            chunk = read(path / "CHUNK.json")
            marker = sha(path / "CHUNK.json")
            clean = chunk["guided"]["outcome"] == "SUCCESS"
            records.append({**row, "rows": chunk["count"], "sha": marker,
                            "outcome": chunk["guided"]["outcome"], "admit_limit": 900 if clean else 450})
        marker = wave["folder"] / "WAVE.json"
        write(marker, {"schema": 1, "label": wave["name"], "kind": wave["kind"], "stream": wave["stream"],
                       "source_state": wave["source_state"], "snapshot": wave["donor"], "records": records})
        result.append(marker)
    write(ROOT / f"performance-{specs[0]['name']}.json", {"measured": perf, "start_memory": start_memory,
          "end_memory": memory(), "workers": workers if simulator is None else len(perf.get("workers", {}))})
    print("COLLECTED", [wave["name"] for wave in specs], perf.get("decisions"), perf.get("decisions_per_second"),
          flush=True)
    return result


def train_cases(owner, tiers):
    from cloud.kaggle.runtime import pool

    rows = pool("mixed")
    result = []
    for i, tier in enumerate(tiers):
        choices = sorted(row["geometry_id"] for row in rows if row["tier"] == tier)
        result.append((tier, choices[(owner * 11 + 7 * i) % len(choices)], 62600 + 20 * owner + i))
    return result


def donor_waves(inventory):
    specs = []
    seen = set()
    for session, entry in sorted(inventory["sources"].items()):
        label = f"donor-{session}"
        if entry["model_sha"] in seen or (ROOT / label / "WAVE.json").is_file():
            continue
        seen.add(entry["model_sha"])
        num = int(session[1:])
        tiers = ("Medium", "Hard", "Extreme", "Extreme") if "Extreme" in entry["curriculum"] else (
            "Medium", "Hard", "Hard", "Extreme")
        specs.append(prepare_wave(label, Path(entry["state"]), entry["state_sha"], "risk",
                                  601 + num, train_cases(num, tiers)))
    return specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    inventory = source_inventory()
    write(ROOT / "inventory22.json", inventory)
    specs = donor_waves(inventory)
    if args.warmup and not (ROOT / "online-01/WAVE.json").is_file():
        from cloud.kaggle.local_risk import v2_parent

        anchor, digest = v2_parent()
        tiers = ("Easy", "Medium", "Hard", "Extreme", "Medium", "Hard", "Extreme", "Medium")
        specs.append(prepare_wave("online-01", anchor, digest, "online", 640, train_cases(100, tiers)))
    if specs:
        collect_waves(specs)


if __name__ == "__main__":
    main()
