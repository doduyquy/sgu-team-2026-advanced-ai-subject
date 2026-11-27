"""Evaluate local consolidation on the original first-window confirmation cases."""
import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from cloud.kaggle.common import WORK, read, sha, write
from cloud.kaggle.final_evaluation import compare_panels, evaluate

TIERS = ("Easy", "Medium", "Hard", "Extreme")


def evaluate_checkpoint(state, expected, out):
    from cloud.kaggle.runtime import check_state

    marker = check_state(state, expected)
    evidence = read(WORK / "central12-recovery" / "post-campaign" / "model-evidence.json")
    references = {tier: read(WORK / evidence["confirmation"][tier]["file"]) for tier in TIERS}
    seeds = sorted({case["seed"] for panel in references.values() for case in panel["cases"]})
    if len({panel["model_sha"] for panel in references.values()}) != 1:
        raise ValueError("The confirmation baseline uses more than one model")
    plan = {"scope": "development confirmation", "seeds": seeds, "tiers": TIERS,
            "model_sha": marker["files"]["model.zip"]}
    out.mkdir(parents=True, exist_ok=False)
    pairs = {}
    for tier in TIERS:
        data = evaluate(plan, state / "model.zip", tier)
        pairs[tier] = compare_panels(references[tier], data)
        write(out / f"{tier.lower()}.json", data)
        print(tier, pairs[tier]["candidate"], flush=True)
    write(out / "paired.json", pairs)
    return pairs


def write_comparison(root, checkpoints, out):
    """Rank evaluated offline checkpoints while keeping S01 as the safety fallback."""
    from cloud.kaggle.delivery import export
    from cloud.kaggle.runtime import check_state

    frozen = read(root / "parent-copy" / "COMPLETE.json")
    version = frozen["files"]["model.zip"]
    references = {}
    candidates = []
    table = []
    for name, eval_name in checkpoints:
        state = root / name
        state_sha = sha(state / "COMPLETE.json")
        marker = check_state(state, state_sha)
        pairs = read(root / eval_name / "paired.json")
        metrics = read(state / "metrics.json")["consolidation"]
        additions = marker["updates"] - frozen["updates"]
        if set(pairs) != set(TIERS) or any(
            pair["reference_sha"] != version or pair["candidate_sha"] != marker["files"]["model.zip"]
            for pair in pairs.values()
        ):
            raise ValueError("Checkpoint evaluation does not match the pinned comparison")
        for tier in TIERS:
            pair = pairs[tier]
            if references.setdefault(tier, pair["reference"]) != pair["reference"]:
                raise ValueError("Baseline statistics vary across checkpoints")
            table.append({"variant": f"Local SAC {additions}", "offline_updates": additions, "tier": tier,
                          **pair["candidate"], "common_clean_cases": pair["common_clean_cases"],
                          "time_delta_on_common_clean_s": pair["mean_time_delta_seconds"]})
        safety = sum(pair["candidate"]["safety_failures"] for pair in pairs.values())
        clean = sum(pair["candidate"]["clean_successes"] for pair in pairs.values())
        technical = sum(pair["candidate"]["technical_failures"] for pair in pairs.values())
        easy_medium_regressions = sum(
            max(0, pairs[tier]["reference"]["clean_successes"] - pairs[tier]["candidate"]["clean_successes"])
            for tier in ("Easy", "Medium"))
        per_tier_retention = all(
            pair["candidate"]["clean_successes"] >= pair["reference"]["clean_successes"]
            and pair["candidate"]["safety_failures"] <= pair["reference"]["safety_failures"]
            for pair in pairs.values())
        gain = any(pair["candidate"]["clean_successes"] > pair["reference"]["clean_successes"]
                   or pair["candidate"]["safety_failures"] < pair["reference"]["safety_failures"]
                   for pair in pairs.values())
        rank = (technical, easy_medium_regressions, safety, -clean,
                -sum(pair["candidate"]["mean_progress"] or 0 for pair in pairs.values()), additions)
        candidates.append({"path": str(state), "state_sha256": state_sha, "offline_updates": additions,
                           "external_rows": metrics["external_rows"], "rank": rank,
                           "eligible_to_replace_s01": not technical and per_tier_retention and gain})
    baseline_rows = [{"variant": "S01", "offline_updates": 0, "tier": tier, **references[tier],
                      "common_clean_cases": references[tier]["clean_successes"], "time_delta_on_common_clean_s": None}
                     for tier in TIERS]
    best = min(candidates, key=lambda row: row["rank"])
    eligible = [row for row in candidates if row["eligible_to_replace_s01"]]
    selected = min(eligible, key=lambda row: row["rank"]) if eligible else None
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as temporary:
        delivery = Path(temporary) / "selection"
        delivery.mkdir()
        with (delivery / "comparison.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(baseline_rows[0]))
            writer.writeheader()
            writer.writerows(baseline_rows + table)
        export(Path(best["path"]), best["state_sha256"], delivery / "best-trained")
        if selected and selected != best:
            export(Path(selected["path"]), selected["state_sha256"], delivery / "deployment-candidate")
        write(delivery / "SELECTION.json", {"best_trained": best, "selected": selected or "S01",
                                           "checkpoints": candidates})
        delivery.rename(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--state-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evaluate_checkpoint(args.state, args.state_sha, args.out)


if __name__ == "__main__":
    main()
