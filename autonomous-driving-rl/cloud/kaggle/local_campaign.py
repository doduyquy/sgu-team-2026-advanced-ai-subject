"""Run the local Stage 5 hybrid experiment and keep accepted artifacts on disk."""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import os
from pathlib import Path

from cloud.kaggle.common import read, sha, write
from cloud.kaggle.local_workers import PARENT, PARENT_SHA, Simulators, assess, collect, references, stage_source


TIERS = ("Easy", "Medium", "Hard", "Extreme")
MILESTONES = (128, 256, 512, 1000, 2000)


def screen_panel(reference):
    picked = {}
    for tier in ("Easy", "Medium"):
        source = reference[tier]
        failures = [case for case in source["cases"] if not case["clean"]]
        family_first = {}
        for case in source["cases"]:
            if case["clean"]:
                family_first.setdefault(case["family"], case)
        good = list(family_first.values())
        good += [case for case in source["cases"] if case["clean"] and case not in good]
        picked[tier] = {**source, "cases": (failures + good)[:4]}
    return picked


def current(root, folder):
    return root / folder, sha(root / folder / "COMPLETE.json")


def screen_rank(record):
    easy, medium = (record["summary"][tier] for tier in ("Easy", "Medium"))
    return (easy["technical_failures"] + medium["technical_failures"], -easy["clean_successes"],
            easy["safety_failures"], -medium["clean_successes"], medium["safety_failures"],
            -medium["mean_progress"], record["updates"])


def accept(pairs):
    clean = {tier: pair["candidate"]["clean_successes"] - pair["reference"]["clean_successes"]
             for tier, pair in pairs.items()}
    safety = {tier: pair["candidate"]["safety_failures"] - pair["reference"]["safety_failures"]
              for tier, pair in pairs.items()}
    return (all(pair["candidate"]["technical_failures"] == 0 for pair in pairs.values())
            and min(clean.values()) >= 0 and max(safety.values()) <= 0
            and (clean["Medium"] > 0 or clean["Hard"] > 0 or clean["Extreme"] > 0 or min(safety.values()) < 0))


def confirmation_rank(item):
    tiers = [item["pairs"][tier]["candidate"] for tier in TIERS]
    return (not item["eligible"], sum(row["technical_failures"] for row in tiers),
            *((-row["clean_successes"], row["safety_failures"]) for row in tiers),
            -sum(row["mean_progress"] or 0 for row in tiers), item["updates"])


def confirm(sim, root, record, reference):
    from cloud.kaggle.final_evaluation import compare_panels
    from cloud.kaggle.runtime import check_state

    out = root / f"confirmation-{record['updates']:05d}.json"
    if out.exists():
        cached = read(out)
        if cached["sha"] != record["sha"]:
            raise ValueError("Confirmation belongs to a different checkpoint")
        return cached
    folder, digest = root / record["state"], record["sha"]
    pin = check_state(folder, digest)
    data, performance = assess(sim, folder / "model.zip", pin["files"]["model.zip"], reference, TIERS)
    pairs = {tier: compare_panels(reference[tier], data[tier]) for tier in TIERS}
    result = {"state": folder.name, "sha": digest, "pairs": pairs, "cases": data,
              "performance": performance, "eligible": accept(pairs), "updates": record["updates"]}
    write(out, result)
    print("CONFIRMATION", record["updates"], result["eligible"],
          {tier: pair["candidate"] for tier, pair in pairs.items()}, flush=True)
    return result


def main():
    import psutil
    import torch
    from cloud.kaggle.common import WORK
    from cloud.kaggle.delivery import export
    from cloud.kaggle.final_evaluation import summarize
    from cloud.kaggle.local_hybrid import checkpoint, restore, update
    from cloud.kaggle.runtime import pool, check_state

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--deadline", type=datetime.fromisoformat, required=True)
    parser.add_argument("--target-rows", type=int, default=200000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha")
    parser.add_argument("--confirm-only", type=int)
    parser.add_argument("--extend-to", type=int, choices=(5000, 10000))
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(WORK.resolve()) or not (root / "parity.json").is_file():
        raise ValueError("Use the verified local four-worker workspace")
    if not 10000 <= args.target_rows <= 250000:
        raise ValueError("Target must fit the bounded local replay")
    if bool(args.resume) != bool(args.resume_sha) or (args.extend_to and not args.resume):
        parser.error("Continuation requires the path and SHA-256 of a verified full state")
    reference = references()
    if args.confirm_only:
        with Simulators() as sim:
            confirm(sim, root, read(root / f"screen-{args.confirm_only:05d}.json"), reference)
        return
    torch.set_num_threads(2)
    if os.name == "nt":
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    schedule = read(root / "schedule.json")
    assignments = [tuple(spec) for spec in schedule["assignments"]]
    donors = {name: {**spec, "snapshot": Path(spec["snapshot"])} for name, spec in schedule["sources"].items()}
    screening = screen_panel(reference)
    write(root / "screen-panel.json", screening)
    baseline = {tier: summarize(doc) for tier, doc in screening.items()}
    parent = check_state(PARENT, PARENT_SHA)
    state = args.resume.resolve() if args.resume else PARENT
    state_sha = args.resume_sha if args.resume else PARENT_SHA
    model, mix, marker = restore(root, state, state_sha)
    print("RESTORED", model._n_updates, flush=True)
    config = marker.get("hybrid", {})
    cursor = config.get("cursor", {"external_index": 24, "external_wave": 1, "online_wave": 0})
    screens = [read(path) for path in sorted(root.glob("screen-*.json")) if path.name != "screen-panel.json"]
    previous_result = read(root / "result.json") if args.extend_to else None
    if previous_result:
        if not config:
            raise ValueError("Extension requires a verified V2 checkpoint")
        for spec in previous_result["datasets"]:
            if spec["online_limit"] is None and spec not in mix.datasets:
                mix.admit(root / spec["path"], spec["sha"])
        external_paths = list(root.glob("external-*/CONSOLIDATED.json"))
        cursor["external_wave"] = max(int(path.parent.name.split("-")[-1]) for path in external_paths)
        cursor["external_index"] = sum(len(read(path)["records"]) for path in external_paths)
        cursor["online_wave"] = max(int(path.parent.name.split("-")[-1])
                                    for path in root.glob("online-*/CONSOLIDATED.json"))
    if not config:
        mix.admit(root / "external-01", sha(root / "external-01/CONSOLIDATED.json"))
    rows = pool("mixed")
    online_choices = {tier: [row["geometry_id"] for row in rows if row["tier"] == tier] for tier in ("Easy", "Medium")}

    def remaining():
        return (args.deadline - datetime.now(args.deadline.tzinfo)).total_seconds()

    def external_rows():
        return sum(row["admitted"] for row in mix.audit if row.get("pool") == "external")

    def external_wave(sim):
        index = cursor["external_index"]
        wave = cursor["external_wave"] + 1
        paths = assignments[index:index + 40]
        if not paths or wave >= 60:
            raise RuntimeError("Exhausted locked external collection schedule")
        folder = root / f"external-{wave:02d}"
        manifest = collect(sim, paths, donors, folder, 859 + wave)
        mix.admit(folder, sha(folder / "CONSOLIDATED.json"))
        cursor["external_index"] += len(paths)
        cursor["external_wave"] = wave
        print("EXTERNAL", wave, external_rows(), manifest["performance"]["decisions_per_second"], flush=True)

    def online_wave(sim, target):
        wave = cursor["online_wave"] + 1
        stage = root / "online-policies" / f"wave-{wave:02d}"
        source = stage_source(state / "model.zip", state / "COMPLETE.json", state_sha, stage)
        tiers = ("Easy", "Medium", "Medium", "Medium") if wave % 3 != 0 else ("Easy", "Easy", "Medium", "Medium")
        cases = []
        for index, tier in enumerate(tiers):
            choices = online_choices[tier]
            geometry = choices[(wave * 11 + 7 * index) % len(choices)]
            cases.append(("V2-online", geometry, 48500 + wave * 4 + index))
        folder = root / f"online-{wave:02d}"
        manifest = collect(sim, cases, {"V2-online": source}, folder, 929 + wave, online=True)
        mix.admit(folder, sha(folder / "CONSOLIDATED.json"), online_limit=2 * target - mix.online_rows)
        cursor["online_wave"] = wave
        print("ONLINE", wave, mix.online_rows, manifest["performance"]["decisions_per_second"], flush=True)

    milestones = ((3000, 5000) if args.extend_to == 5000 else (7500, 10000)) if args.extend_to else MILESTONES
    run_name = f"extension-{args.extend_to}" if args.extend_to else ""
    previous_screens = {record["updates"] for record in screens}
    resume_screen = next((record["summary"] for record in screens if record["state"] == state.name), None)
    with Simulators() as sim:
        for target in milestones:
            if target <= model._n_updates - parent["updates"]:
                continue
            if remaining() < 1200:
                break
            while mix.online_rows < 2 * target:
                online_wave(sim, target)
                if cursor["online_wave"] >= 45:
                    raise RuntimeError("Online collection exhausted its bounded namespace")
            learning = update(model, mix, parent, target)
            folder = root / f"stage-{target:05d}"
            digest = checkpoint(model, mix, parent, folder, learning, cursor.copy())
            state, state_sha = folder, digest
            print("TRAINED", target, learning, flush=True)
            prediction, performance = assess(sim, folder / "model.zip", sha(folder / "model.zip"), screening, ("Easy", "Medium"))
            summary = {tier: summarize(doc) for tier, doc in prediction.items()}
            record = {"state": folder.name, "sha": digest, "updates": target, "summary": summary,
                      "cases": prediction, "performance": performance}
            screens.append(record)
            write(root / f"screen-{target:05d}.json", record)
            print("SCREEN", target, summary, flush=True)
            severe = any(summary[tier]["clean_successes"] <= baseline[tier]["clean_successes"] - 2
                         or summary[tier]["technical_failures"] for tier in summary)
            if severe:
                print("STOP_SCREEN_REGRESSION", target, flush=True)
                break
            if args.extend_to and resume_screen and any(
                    summary[tier]["clean_successes"] < resume_screen[tier]["clean_successes"] for tier in summary):
                print("STOP_BELOW_EXTENSION_PARENT", target, flush=True)
                break
            if target != milestones[-1] and external_rows() < args.target_rows and remaining() > 1500:
                external_wave(sim)

        new_screens = [record for record in screens if record["updates"] not in previous_screens]
        if args.extend_to:
            new_screens = [record for record in new_screens if all(
                result["clean_successes"] >= baseline[tier]["clean_successes"] and not result["technical_failures"]
                for tier, result in record["summary"].items())]
        rows_ranked = sorted(new_screens if args.extend_to else screens, key=screen_rank)
        confirmations = ([read(path) for path in sorted(root.glob("confirmation-*.json"))]
                         if args.extend_to else [])
        for record in rows_ranked[:2]:
            if remaining() < 600:
                break
            confirmations.append(confirm(sim, root, record, reference))
        if not confirmations:
            raise RuntimeError("No V2 checkpoint completed the 48-case development confirmation")
        full_rank = sorted(confirmations, key=confirmation_rank)
        best = full_rank[0]
        selected = best["state"] if best["eligible"] else "S01"
        best_state, best_sha = current(root, best["state"])
        export(best_state, best_sha, root / (f"best-v2-{run_name}-bundle" if run_name else "best-v2-bundle"))

        # Preserve the final replay even when screening stops learner updates.
        while (external_rows() < args.target_rows or cursor["external_index"] < 78) and remaining() > 900:
            external_wave(sim)
            write(root / "progress.json", {"external_rows": external_rows(), "cursor": cursor,
                                          "selected": selected, "best": best["state"]})
    write(root / (f"{run_name}-result.json" if run_name else "result.json"),
          {"selected": selected, "best": best, "confirmations": confirmations,
                                 "datasets": mix.datasets, "external_rows_new": external_rows(),
                                 "online_rows": mix.online_rows, "sampling": dict(mix.draws),
                                 "updates": model._n_updates - parent["updates"],
                                 "replay_audit": mix.audit})
    del model, mix
    gc.collect()
    _, resumed, saved = restore(root, best_state, best_sha)
    write(root / (f"{run_name}-resume-proof.json" if run_name else "resume-proof.json"),
          {"state": best_state.name, "sha": best_sha, "updates": saved["updates"],
                                       "online_rows": resumed.online_rows, "sampling": dict(resumed.draws),
                                       "next_minibatch_sha256": resumed.proof(), "passed": True})
    print("SELECTED", selected, "BEST_V2", best_state.name, flush=True)


if __name__ == "__main__":
    main()
