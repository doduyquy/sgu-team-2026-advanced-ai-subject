"""Discover the first-window donors and materialize TRAIN collection waves."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from cloud.kaggle.common import WORK, read, sha, write
from cloud.kaggle.local_workers import PARENT, PARENT_SHA, Simulators, collect, parity, stage_source


def survey(out, stage_archive):
    root = WORK / "central12-recovery"
    campaign = read(root / "post-campaign/campaign.json")["workloads"]
    sessions = {attempt["job"]: record["session"] for record in campaign for attempt in record["attempts"]}
    marker_paths = set()
    markers_by_session = defaultdict(list)
    indexed = Counter()
    for item in read(stage_archive)["candidates"]:
        session = sessions.get(item["job"])
        if session is None:
            continue
        kind = item["artifact"].split("/")[0]
        indexed[session, kind] += 1
        if kind not in ("periodic", "final"):
            continue
        state = Path(item["path"])
        marker = state / "COMPLETE.json"
        if not marker.is_file() or not (state / "model.zip").is_file() or marker in marker_paths:
            continue
        marker_paths.add(marker)
        digest = sha(marker)
        if digest != item.get("sha256"):
            continue
        payload = read(marker)
        if payload["files"]["model.zip"] != item["files"]["model.zip"]:
            continue
        markers_by_session[session].append({
            "path": str(state), "digest": digest, "kind": kind, "updates": payload["updates"],
            "model_sha": payload["files"]["model.zip"], "replay_present": (state / "replay.pkl").exists(),
            "position": payload.get("replay_pos"), "parent_sha": payload.get("parent_sha"),
            "has_externally_attested_row_provenance": False,
        })
    old = read(WORK / "first-window-local/collection-01.json")["sources"]
    explicit = {item["label"]: (Path(item["state"]), item["state_sha"]) for item in old}
    explicit["S01"] = PARENT, PARENT_SHA
    # Prefer previously evaluated finals, then a local verified periodic state.
    sources = {}
    for row in campaign:
        session = row["session"]
        records = markers_by_session[session]
        if session in explicit:
            state, digest = explicit[session]
        elif records:
            top = max(records, key=lambda item: (item["kind"] == "final", item["updates"]))
            state, digest = Path(top["path"]), top["digest"]
        else:
            continue
        sources[session] = stage_source(state / "model.zip", state / "COMPLETE.json", digest,
                                        out / "donors" / session)
    inventory = {"historical_new_rows_admitted": 0,
                 "historical_replay_evidence": read(root / "post-campaign/campaign.json")["replay_policy"],
                 "learners": [{"session": row["session"],
                               "indexed": {kind: indexed[row["session"], kind] for kind in ("final", "periodic", "screening")},
                               "full_state_candidates": markers_by_session[row["session"]],
                               "donor": row["session"] in sources} for row in campaign]}
    write(out / "inventory.json", inventory)
    return sources


def assignment_schedule(sources, repeats):
    from cloud.kaggle.runtime import pool

    previously_seen = {row["geometry"] for row in read(WORK / "first-window-local/stage-10000/metrics.json")
                       ["consolidation"]["sources"]}
    orders = {"Easy": ("S03", "S06", "S10", "S16", "S01", "S07", "S11"),
              "Medium": ("S01", "S09", "S13", "S11", "S02", "S05", "S07", "S14", "S15")}
    pools = {tier: sorted((row["geometry_id"] for row in pool("mixed") if row["tier"] == tier),
                          key=lambda geometry: (geometry in previously_seen, geometry)) for tier in orders}
    schedule = []
    for repeat in range(repeats):
        for index, (easy, medium) in enumerate(zip(pools["Easy"], pools["Medium"], strict=True)):
            for tier, geometry in (("Medium", medium), ("Easy", easy)):
                known = [donor for donor in orders[tier] if donor in sources]
                if repeat:
                    known += [donor for donor in sources if donor not in known]
                label = known[(index + repeat * 3) % len(known)]
                seed = 41200 + repeat * 80 + index * 2 + (tier == "Easy")
                schedule.append((label, geometry, seed))
    return schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sources = survey(args.out, args.archive)
    schedule = assignment_schedule(sources, 8)
    write(args.out / "schedule.json", {"assignments": schedule,
                                      "sources": {name: {**data, "snapshot": str(data["snapshot"])}
                                                  for name, data in sources.items()}})
    with Simulators() as sim:
        print("PARITY", parity(sim, args.out / "parity.json"), flush=True)
        first = collect(sim, schedule[:24], sources, args.out / "external-01", 860)
        print("FIRST_BATCH", first["performance"], flush=True)


if __name__ == "__main__":
    main()
