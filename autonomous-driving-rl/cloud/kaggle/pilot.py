"""Pair Easy pilot evaluations with the frozen source panel."""
import argparse
from pathlib import Path
from statistics import mean

from cloud.kaggle.common import SOURCE_SHA, read, sha, write


def panel(document):
    cases = {}
    for row in document["cases"]:
        key = (row["geometry"], row["family"], row["seed"])
        if key in cases or row["censored"] or not isinstance(row["clean"], bool):
            raise ValueError("Easy panel contains duplicate, censored or invalid cases")
        cases[key] = row
    if document["tier"] != "Easy" or len(cases) != 12:
        raise ValueError("Expected the complete 12-case Easy panel")
    return cases


def summarize(cases):
    rows = list(cases.values())
    times = [row["success_seconds"] for row in rows if row["clean"]]
    return {"clean_successes": len(times),
            "safety_failures": sum(row["outcome"] == "OUT_OF_ROAD" or row["outcome"].startswith("CRASH_") for row in rows),
            "mean_route_completion": mean(row["progress"] for row in rows),
            "mean_speed_km_h": mean(row["speed_km_h"] for row in rows),
            "mean_success_seconds": mean(times) if times else None}


def compare(baseline, candidates, *, reference_sha=SOURCE_SHA):
    frozen = panel(baseline)
    reference_summary = summarize(frozen)
    if baseline["model_sha"] != reference_sha or (reference_sha == SOURCE_SHA and reference_summary["clean_successes"] != 8):
        raise ValueError("Reference checkpoint does not match the registered panel")
    evaluated = {reference_sha}
    milestones = []
    for result in candidates:
        current = panel(result)
        model = result["model_sha"]
        if frozen.keys() != current.keys() or model in evaluated:
            raise ValueError("Milestones must use distinct checkpoints and the exact baseline case IDs")
        evaluated.add(model)
        summary = summarize(current)
        common = [row["success_seconds"] - frozen[key]["success_seconds"]
                  for key, row in current.items() if row["clean"] and frozen[key]["clean"]]
        paired_fields = ("outcome", "clean", "progress", "speed_km_h", "success_seconds")
        pairs = [{"geometry": key[0], "family": key[1], "seed": key[2],
                  "baseline": {field: frozen[key][field] for field in paired_fields},
                  "candidate": {field: row[field] for field in paired_fields}} for key, row in current.items()]
        milestones.append({"model_sha": model, "cases": len(current), **summary,
                           "improved_cases": sum(row["clean"] and not frozen[key]["clean"] for key, row in current.items()),
                           "regressed_cases": sum(not row["clean"] and frozen[key]["clean"] for key, row in current.items()),
                           "mean_time_delta_on_common_successes_seconds": mean(common) if common else None,
                           "paired_cases": pairs,
                           "meets_thresholds": summary["clean_successes"] >= 10 and summary["safety_failures"] <= 2})
    admissible = [row for row in milestones if
                  row["clean_successes"] >= reference_summary["clean_successes"]
                  and row["safety_failures"] <= reference_summary["safety_failures"]]
    best = max([{"model_sha": reference_sha, **reference_summary}, *admissible],
               key=lambda row: (row["clean_successes"], -row["safety_failures"], row["mean_route_completion"],
                                -(row.get("mean_time_delta_on_common_successes_seconds") or 0)))
    return {"baseline_clean_successes": reference_summary["clean_successes"], "baseline_cases": 12,
            "reference_model_sha": reference_sha, "baseline": reference_summary, "milestones": milestones,
            "recommended_model_sha": best["model_sha"],
            "two_milestones_meet_thresholds": len(milestones) >= 2 and all(row["meets_thresholds"] for row in milestones)}


def screen_candidates(reference, candidates, *, reference_sha):
    reference_cases = panel(reference)
    if reference["model_sha"] != reference_sha:
        raise ValueError("Reference checkpoint does not match the registered panel")
    seed = min(key[2] for key in reference_cases)
    frozen = {key: row for key, row in reference_cases.items() if key[2] == seed}
    rows = []
    seen = {reference_sha}
    for document in candidates:
        candidate = {(row["geometry"], row["family"], row["seed"]): row for row in document["cases"]}
        if (len(frozen) != 6 or len(document["cases"]) != 6 or candidate.keys() != frozen.keys()
                or document["tier"] != "Easy" or any(row["censored"] for row in candidate.values())
                or document["model_sha"] in seen):
            raise ValueError("Screening needs distinct models on the locked six Easy cases")
        seen.add(document["model_sha"])
        rows.append({"model_sha": document["model_sha"], **summarize(candidate)})
    rank = sorted(rows, key=lambda row: (-row["clean_successes"], row["safety_failures"], -row["mean_route_completion"]))
    return {"purpose": "SCREENING_ONLY", "seed": seed, "reference_model_sha": reference_sha,
            "reference": summarize(frozen), "ranked": rank}


def easy_coverage(split, runs):
    easy = {row["geometry_id"]: row["sequence"] for row in split["pilot_train"] if row["tier"] == "Easy"}
    diagnostic = {row["geometry_id"] for row in split["pilot_diagnostic"]}
    if len(easy) != 39 or easy.keys() & diagnostic:
        raise ValueError("Expected the locked 39-geometry Easy TRAIN pool")
    totals = dict.fromkeys(sorted(easy), 0)
    for metrics in runs:
        counts = metrics["decisions_by_geometry"]
        if counts.keys() - easy.keys() or any(type(value) is not int or value < 0 for value in counts.values()):
            raise ValueError("Telemetry falls outside the locked Easy TRAIN pool")
        for key, value in counts.items():
            totals[key] += value
    return {"pool": len(easy), "visited": sum(value > 0 for value in totals.values()),
            "unvisited": [key for key, value in totals.items() if not value], "decisions": totals}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-sha", default=SOURCE_SHA)
    parser.add_argument("--screen", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    action = screen_candidates if args.screen else compare
    result = action(read(args.baseline), [read(path) for path in args.candidate], reference_sha=args.reference_sha)
    result["baseline_artifact_sha256"] = sha(args.baseline)
    write(args.out, result)
    print(result)


if __name__ == "__main__":
    main()
