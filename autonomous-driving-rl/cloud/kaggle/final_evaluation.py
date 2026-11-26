"""Evaluate a frozen development candidate with separate F09 failure accounting."""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from cloud.kaggle.common import read, verify, write
from src.platform.task_context_producer import TaskContextLocalizationError


def evaluate(plan, model_path, tier):
    from cloud.kaggle import runtime

    if plan["scope"] != "development confirmation" or tier not in plan["tiers"]:
        raise ValueError("The requested tier is outside the frozen development panel")
    seeds = plan["seeds"]
    if len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError("The confirmation panel requires two distinct pinned seeds")
    verify(model_path, plan["model_sha"])
    runtime.torch.set_num_threads(1)
    model = runtime.load_policy(model_path)
    rows = [row for row in runtime.pool("eval") if row["tier"] == tier]
    if len(rows) != 6:
        raise ValueError("Unexpected locked diagnostic geometry count")
    cases = []
    for row in rows:
        for seed in seeds:
            env = runtime.ScheduledEnv("eval", 0, 0, fixed=(row["geometry_id"], seed))
            info, step, technical_failure = {}, 0, False
            try:
                obs, reset = env.reset()
                for step in range(1, reset["horizon_steps"] + 1):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                else:
                    raise RuntimeError("The locked diagnostic horizon did not emit a terminal or timeout")
            except TaskContextLocalizationError:
                technical_failure = True
            finally:
                env.close()
            clean = not technical_failure and bool(info.get("clean_success"))
            cases.append({"geometry": row["geometry_id"], "family": row["sequence"], "seed": seed,
                "clean": clean, "outcome": "TASK_CONTEXT_LOCALIZATION" if technical_failure else info["primary_reason"],
                "technical_failure": technical_failure, "progress": info.get("route_completion"),
                "speed_km_h": info.get("episode_mean_speed_km_h"), "success_seconds": step / 10 if clean else None})
    return {"model_sha": plan["model_sha"], "scope": plan["scope"], "tier": tier, "cases": cases}


def summarize(document):
    cases = document["cases"]
    summary = {
        "cases": len(cases), "clean_successes": sum(row["clean"] for row in cases),
        "safety_failures": sum(row["outcome"] == "OUT_OF_ROAD" or row["outcome"].startswith("CRASH_") for row in cases),
        "timeouts": sum(row["outcome"] == "TIMEOUT" for row in cases),
        "technical_failures": sum(row["technical_failure"] for row in cases),
    }
    for field in ("progress", "speed_km_h", "success_seconds"):
        sample = [row[field] for row in cases if row[field] is not None]
        summary[f"mean_{field}"] = mean(sample) if sample else None
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tier", choices=("Easy", "Medium", "Hard", "Extreme"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    doc = evaluate(read(args.plan), args.model, args.tier)
    write(args.out, {**doc, "summary": summarize(doc)})


if __name__ == "__main__":
    main()
