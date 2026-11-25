"""Pair Easy pilot evaluations with the frozen source panel."""
import argparse
from pathlib import Path

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


def compare(baseline, candidates):
    frozen = panel(baseline)
    if baseline["model_sha"] != SOURCE_SHA or sum(row["clean"] for row in frozen.values()) != 8:
        raise ValueError("Baseline must be the registered source with 8/12 clean successes")
    evaluated = {SOURCE_SHA}
    milestones = []
    for result in candidates:
        current = panel(result)
        model = result["model_sha"]
        if frozen.keys() != current.keys() or model in evaluated:
            raise ValueError("Milestones must use distinct checkpoints and the exact baseline case IDs")
        evaluated.add(model)
        clean = sum(row["clean"] for row in current.values())
        safety = sum(row["outcome"] == "OUT_OF_ROAD" or row["outcome"].startswith("CRASH_")
                     for row in current.values())
        milestones.append({"model_sha": model, "cases": len(current), "clean_successes": clean,
                           "safety_failures": safety,
                           "improved_cases": sum(row["clean"] and not frozen[key]["clean"] for key, row in current.items()),
                           "regressed_cases": sum(not row["clean"] and frozen[key]["clean"] for key, row in current.items()),
                           "meets_thresholds": clean >= 10 and safety <= 2})
    return {"baseline_clean_successes": 8, "baseline_cases": 12, "milestones": milestones,
            "two_milestones_meet_thresholds": len(milestones) >= 2 and all(row["meets_thresholds"] for row in milestones)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = compare(read(args.baseline), [read(path) for path in args.candidate])
    result["baseline_artifact_sha256"] = sha(args.baseline)
    write(args.out, result)
    print(result)


if __name__ == "__main__":
    main()
