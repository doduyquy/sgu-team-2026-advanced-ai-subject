"""Paired traffic study for the four deterministic Stage 2 planners."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
from statistics import mean, stdev
import sys

from evaluate_candidate_test import (
    PROJECT, check_snapshot, evaluate_job, initialize_worker, snapshot_sources, write_json,
)

AGENTS = ("stage2_hybrid_astar", "stage2_bfs", "stage2_dijkstra", "stage2_astar")
SEEDS = {"VALIDATION": (5101, 5102, 5103), "TEST": (9101, 9102, 9103)}
TIERS = ("Easy", "Medium", "Hard", "Extreme")


def schedule(split, manifest_cases, scope):
    """Pair the same cases across planners and traffic modes."""
    if scope == "locked":
        cases_by_seed = {}
        for case in manifest_cases:
            cases_by_seed.setdefault(case.environment_seed, []).append(case.to_dict())
    else:
        cases_by_seed = three_seed_cases(split, manifest_cases, scope)

    for agent in AGENTS:
        for seed, seed_cases in sorted(cases_by_seed.items()):
            for traffic_mode in ("on", "off"):
                cases = [dict(case, traffic_density=(max(.08, case["traffic_density"])
                                                     if traffic_mode == "on" else 0.))
                         for case in seed_cases]
                yield dict(agent_id=agent, agent_seed=None, smoke=False, split=split,
                           evaluation_kind=f"SEARCH_TRAFFIC_STUDY_{scope.upper()}",
                           job_id=f"{split.lower()}-{agent}-{seed}-{traffic_mode}",
                           traffic_mode=traffic_mode, cases=cases)


def three_seed_cases(split, manifest_cases, scope):
    """Build the custom three-seed schedule from a fixed set of geometries."""
    geometries = {}
    for case in manifest_cases:
        key = (case.tier, case.sequence, case.geometry_generation_seed)
        current = case.to_dict()
        current.pop("environment_seed")
        current.pop("case_index")
        current.pop("protocol_order_index")
        current["case_id"] = current["case_id"].rsplit("/env-", 1)[0]
        if key in geometries and geometries[key] != current:
            raise ValueError(f"Manifest varies geometry metadata by seed: {key}")
        geometries[key] = current
    expected = 48 if split == "VALIDATION" else 12
    if len(geometries) != expected:
        raise ValueError(f"{split}: expected {expected} unique geometries")
    keys = sorted(geometries, key=lambda key: (TIERS.index(key[0]), key[1], key[2]))
    if scope == "pilot":
        keys = [next(key for key in keys if key[0] == "Hard")]
    cases_by_seed = {}
    for seed in SEEDS[split]:
        cases = []
        for index, key in enumerate(keys, 1):
            case = dict(geometries[key])
            case.update(case_id=f"{case['case_id']}/env-{seed}", case_index=index,
                        protocol_order_index=index, environment_seed=seed)
            cases.append(case)
        cases_by_seed[seed] = cases
    return cases_by_seed


def identity(row):
    return row["split"], row["case_id"], row["agent_id"], row["traffic_mode"]


def summarize(jobs, records, output):
    expected = {identity({**case, **{k: job[k] for k in ("agent_id", "traffic_mode")}}): case
                for job in jobs for case in job["cases"]}
    actual = {identity(r): r for r in records}
    if actual.keys() != expected.keys() or len(actual) != len(records):
        raise ValueError("Incomplete or duplicate paired episode coverage")
    for key, row in actual.items():
        if any(row[name] != expected[key][name] for name in ("geometry_sha256", "environment_seed",
                                                             "traffic_density", "horizon_steps")):
            raise ValueError(f"Episode metadata drift: {key}")
    scorecards = []
    for split in SEEDS:
        for agent in AGENTS:
            for tier in ("ALL", *TIERS):
                matched = [r for r in records if r["split"] == split and r["agent_id"] == agent
                           and (tier == "ALL" or r["tier"] == tier)]
                if not matched:
                    continue
                card = dict(split=split, agent=agent, tier=tier)
                for mode in ("on", "off"):
                    rows = [r for r in matched if r["traffic_mode"] == mode]
                    seeds = sorted({r["environment_seed"] for r in rows})
                    seed_rates = {seed: mean(r["clean_success"] for r in rows if r["environment_seed"] == seed)
                                  for seed in seeds}
                    card[mode] = dict(episodes=len(rows), successes=sum(r["clean_success"] for r in rows),
                        clean_success_by_seed=seed_rates, mean_seed_success=mean(seed_rates.values()),
                        std_seed_success=stdev(seed_rates.values()), mean_route=mean(r["final_route_completion"] for r in rows),
                        reason_counts=dict(Counter(r["primary_reason"] for r in rows)),
                        episodes_with_traffic=sum(r["peak_traffic"] > 0 for r in rows),
                        mean_act_ms=sum(r["mean_act_ms"] * r["episode_steps"] for r in rows)
                            / sum(r["episode_steps"] for r in rows))
                on = {r["case_id"]: r["clean_success"] for r in matched if r["traffic_mode"] == "on"}
                off = {r["case_id"]: r["clean_success"] for r in matched if r["traffic_mode"] == "off"}
                if on.keys() != off.keys():
                    raise ValueError("Unpaired traffic treatment")
                card.update(gained_clean_success=sum(off[k] and not on[k] for k in on),
                            lost_clean_success=sum(on[k] and not off[k] for k in on))
                scorecards.append(card)
    write_json(output / "comparison.json", {"scorecards": scorecards})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("pilot", "full", "locked"), default="pilot",
                        help="locked uses all existing 96 VALIDATION and 60 TEST cases with their manifest seeds")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(PROJECT))
    from PySide6.QtCore import QProcess  # Qt must load first on Windows
    from src.launcher.cases import resolve_test_cases, resolve_validation_cases
    from src.launcher.registry import build_canonical_agent_registry
    from src.platform import (PINNED_METADRIVE_COMMIT, PINNED_METADRIVE_VERSION,
                              capture_environment_provenance, capture_git_provenance)
    from src.training.runtime import local_training_runtime

    with local_training_runtime(PROJECT):
        environment = capture_environment_provenance(PROJECT)
        if (environment["metadrive_version"], environment["metadrive_commit"]) != (
                PINNED_METADRIVE_VERSION, PINNED_METADRIVE_COMMIT):
            raise RuntimeError("Exact MetaDrive pin required")
        cohort = {"VALIDATION": list(schedule("VALIDATION", resolve_validation_cases(PROJECT), args.scope)),
                  "TEST": list(schedule("TEST", resolve_test_cases(PROJECT), args.scope))}
        registry = build_canonical_agent_registry()
        methods = {agent: registry.get_factory(agent).experiment_parameters() for agent in AGENTS}
        if any(registry.get(agent).inference_stochasticity != "deterministic" for agent in AGENTS):
            raise ValueError("This schedule assumes deterministic agents")
        jobs = [job for group in cohort.values() for job in group]
        seeds = {split: sorted({case["environment_seed"] for job in group for case in job["cases"]})
                 for split, group in cohort.items()}
        manifest = dict(status="PLANNED", canonical_run=False, scope=args.scope,
                        environment=environment, git=capture_git_provenance(PROJECT.parent),
                        seeds=seeds, workers=args.workers, methods=methods, jobs=jobs,
                        expected_episodes=sum(len(job["cases"]) for job in jobs))
        if not args.run:
            print(json.dumps(manifest, indent=2))
            return 0

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = (args.output or PROJECT / "artifacts/search_traffic" / stamp).resolve()
        if any(output.is_relative_to(PROJECT / folder) for folder in ("src", "configs", "results")):
            raise ValueError("Put output outside the execution sources and manifests")
        output.mkdir(parents=True, exist_ok=False)
        source, hashes = snapshot_sources(output)
        shutil.copy2(__file__, source / Path(__file__).name)
        hashes[Path(__file__).name] = hashlib.sha256((source / Path(__file__).name).read_bytes()).hexdigest()
        manifest.update(status="RUNNING", source_sha256=hashes, completed_episodes=0)
        write_json(output / "session.json", manifest)
        print(f"{args.scope}: {manifest['expected_episodes']} episodes | {output}", flush=True)
        all_records = []
        try:
            with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=initialize_worker, initargs=(source,)) as pool:
                for split, group in cohort.items():
                    manifest["active_split"] = split
                    write_json(output / "session.json", manifest)
                    pending = [pool.submit(evaluate_job, job, output, hashes) for job in group]
                    try:
                        for future in as_completed(pending):
                            all_records.extend(future.result())
                            manifest["completed_episodes"] = len(all_records)
                            write_json(output / "session.json", manifest)
                    except BaseException:
                        (output / "STOP").touch()
                        for future in pending:
                            future.cancel()
                        raise
            check_snapshot(source, hashes)
            summarize(jobs, all_records, output)
            manifest["status"] = "COMPLETE"
        except BaseException as error:
            manifest.update(status="INTERRUPTED" if isinstance(error, (KeyboardInterrupt, InterruptedError))
                            else "FAILED", error=str(error) or type(error).__name__)
            raise
        finally:
            records = []
            for job in jobs:
                file = output / job["job_id"] / "episodes.jsonl"
                if file.exists():
                    records.extend(json.loads(line) for line in file.read_text(encoding="utf-8").splitlines())
            if records:
                with (output / "episodes.csv").open("w", encoding="utf-8-sig", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=list(records[0]))
                    writer.writeheader()
                    writer.writerows({**r, "raw_flags": json.dumps(r["raw_flags"])} for r in records)
            manifest.update(completed_episodes=len(records), active_split=None,
                            finished_at_utc=datetime.now(timezone.utc).isoformat())
            write_json(output / "session.json", manifest)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
