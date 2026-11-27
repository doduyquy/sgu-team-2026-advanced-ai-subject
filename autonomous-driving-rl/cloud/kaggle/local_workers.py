"""Persistent, single-threaded MetaDrive processes for Stage 5 SAC."""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache
import logging
from multiprocessing import get_context
import os
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import psutil

from cloud.kaggle.common import WORK, read, sha, verify, write


WORKERS = 4
PARENT_SHA = "5fb35ed4a744d45000e2d3ce4eb120bc88f90032898bed5ca35f3787566e07fd"
PARENT = WORK / "first-window-local" / "parent-copy"


def initialize():
    import torch

    torch.set_num_threads(1)
    logging.disable(logging.INFO)
    if os.name == "nt":
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


@lru_cache(maxsize=4)
def policy(path, digest):
    from cloud.kaggle.runtime import load_policy

    verify(path, digest)
    return load_policy(path)


@lru_cache(maxsize=2)
def environment(phase):
    from cloud.kaggle.runtime import ScheduledEnv

    return ScheduledEnv(phase, 0, 0)


def worker(task):
    from cloud.kaggle import replay
    from cloud.kaggle.final_evaluation import evaluate_case
    from src.platform.task_context_producer import TaskContextLocalizationError

    started = monotonic()
    process = psutil.Process()
    with open(os.devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
        model = policy(task["policy"], task["policy_sha"])
        if task["kind"] == "eval":
            measured = {}
            observer = None
            if task.get("risk_metrics"):
                from cloud.kaggle.risk_metrics import RiskMetrics

                observer = RiskMetrics()
            if task.get("smooth_traffic"):
                from cloud.kaggle.smooth_traffic_eval import BlendedPolicy, TrafficObserver

                model, observer = BlendedPolicy(model), TrafficObserver()
            result = evaluate_case(model, task["row"], task["seed"], environment(task.get("phase", "eval")), measured, observer)
            if task.get("smooth_traffic"):
                result["traffic"] = {**observer.summary(), "interventions": model.interventions}
            if observer is not None and not task.get("smooth_traffic"):
                result["risk_metrics"] = observer.summary()
            decisions = measured["decisions"]
        else:
            guide = {}
            if task.get("risk_guidance"):
                from cloud.kaggle.risk_guidance import PROFILE, risk_action

                guide = {"action_guide": risk_action, "guide_profile": PROFILE}
            try:
                digest = replay.collect(SimpleNamespace(
                    snapshot=task["snapshot"], snapshot_sha=task["snapshot_sha"], rows=replay.MAX_ROWS,
                    stream=task["stream"], sequence=task["sequence"], geometry=task["geometry"],
                    collection_phase="mixed", seed=task["seed"], out=task["out"]),
                    model=model, env=environment("mixed"), quiet=True, **guide)
            except TaskContextLocalizationError:
                result, decisions = {"discard": "LOCALIZATION"}, 0
            else:
                decisions = read(task["out"] / "CHUNK.json")["count"]
                result = {"sha": digest, "rows": decisions}
    return result, {"pid": process.pid, "decisions": decisions, "seconds": monotonic() - started,
                    "rss_bytes": process.memory_info().rss}


class Simulators:
    def __init__(self, workers=WORKERS):
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=initialize)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.pool.shutdown(cancel_futures=True)

    def run(self, jobs):
        began = monotonic()
        data, samples = zip(*self.pool.map(worker, jobs), strict=True)
        elapsed = monotonic() - began
        processes = defaultdict(lambda: {"decisions": 0, "busy_seconds": 0.0, "peak_rss_bytes": 0})
        for sample in samples:
            stat = processes[sample["pid"]]
            stat["decisions"] += sample["decisions"]
            stat["busy_seconds"] += sample["seconds"]
            stat["peak_rss_bytes"] = max(stat["peak_rss_bytes"], sample["rss_bytes"])
        for stat in processes.values():
            stat["decisions_per_second"] = stat["decisions"] / stat["busy_seconds"]
            stat["real_time_factor"] = stat["decisions_per_second"] / 10
        steps = sum(item["decisions"] for item in samples)
        return data, {"wall_seconds": elapsed, "decisions": steps, "decisions_per_second": steps / elapsed,
                      "real_time_factor": steps / elapsed / 10, "workers": dict(processes),
                      "available_ram_bytes": psutil.virtual_memory().available}


def references():
    graph = read(WORK / "central12-recovery/post-campaign/model-evidence.json")
    return {tier: read(WORK / item["file"]) for tier, item in graph["confirmation"].items()}


def assess(sim, model, digest, panel, tiers, *, risk_metrics=False):
    from cloud.kaggle.runtime import pool

    lookup = {row["geometry_id"]: row for row in pool("eval")}
    locked = [(tier, case) for tier in tiers for case in panel[tier]["cases"]]
    data, performance = sim.run([{"kind": "eval", "policy": model, "policy_sha": digest,
                                  "row": lookup[case["geometry"]], "seed": case["seed"],
                                  "risk_metrics": risk_metrics} for _, case in locked])
    results = {tier: {"tier": tier, "scope": "development confirmation", "model_sha": digest, "cases": []}
               for tier in tiers}
    for (tier, _), case in zip(locked, data, strict=True):
        results[tier]["cases"].append(case)
    return results, performance


def parity(sim, out):
    frozen = references()
    panel = {tier: {**frozen[tier], "cases": frozen[tier]["cases"][:2]} for tier in ("Easy", "Medium")}
    results, performance = assess(sim, PARENT / "model.zip", frozen["Easy"]["model_sha"], panel, tuple(panel))
    for tier, document in results.items():
        for expected, actual in zip(panel[tier]["cases"], document["cases"], strict=True):
            for field, b in actual.items():
                a = expected.get(field, False) if field == "technical_failure" else expected[field]
                equal = abs(a - b) <= 1e-6 if isinstance(a, float) and isinstance(b, (float, int)) else a == b
                if not equal:
                    raise ValueError(f"Four-worker parity mismatch in {tier} {field}: {a} vs {b}")
    write(out, {"cases": results, "performance": performance, "parity": "PASS"})
    return performance


def stage_source(model, marker, expected, folder):
    from cloud.kaggle import replay
    from cloud.kaggle.consolidated_replay import assert_contract

    verify(marker, expected)
    source = read(marker)
    assert_contract(source["identity"], read(PARENT / "COMPLETE.json")["identity"])
    folder.mkdir(parents=True, exist_ok=True)
    snapshot = folder / "snapshot"
    info = {"snapshot": snapshot, "identity": source["identity"], "policy_sha": source["files"]["model.zip"],
            "policy_version": source["updates"], "source_state": expected}
    if not snapshot.exists():
        (folder / "SOURCE.json").write_bytes(marker.read_bytes())
        replay.publish(model, info["policy_sha"], info["identity"], info["policy_version"], snapshot)
    info["snapshot_sha"] = sha(snapshot / "SNAPSHOT.json")
    version = replay.check_snapshot(snapshot, info["snapshot_sha"], info["identity"])
    if version != info["policy_version"] or read(snapshot / "SNAPSHOT.json")["policy_sha"] != info["policy_sha"]:
        raise ValueError("Staged snapshot differs from the donor")
    return info


def collect(sim, assignments, sources, directory, stream, *, online=False):
    from cloud.kaggle.runtime import pool

    locked = {row["geometry_id"]: row for row in pool("mixed") if row["tier"] in ("Easy", "Medium")}
    if len(assignments) > 1001 or len(set(assignments)) != len(assignments):
        raise ValueError("Collection batch repeats cases or exceeds its namespace")
    complete = directory / "CONSOLIDATED.json"
    if complete.is_file():
        previous = read(complete)
        cases = [(record["label"], record["geometry"], record["seed"]) for record in previous["records"]]
        if ((previous["parent_sha"], previous["stream"], previous["online"], cases)
                != (PARENT_SHA, stream, online, assignments)
                or any(pin["snapshot_sha"] != sources[name]["snapshot_sha"]
                       for name, pin in previous["sources"].items())):
            raise ValueError("Existing replay directory belongs to another batch")
        return previous
    jobs = []
    directory.mkdir(parents=True)
    for index, (label, geometry, seed) in enumerate(assignments):
        source = sources[label]
        if geometry not in locked:
            raise ValueError("Expected a locked Easy/Medium TRAIN geometry")
        jobs.append({"kind": "collect", "snapshot": source["snapshot"], "snapshot_sha": source["snapshot_sha"],
                     "policy": source["snapshot"] / "policy.zip", "policy_sha": source["policy_sha"],
                     "geometry": geometry, "seed": seed, "stream": stream, "sequence": index,
                     "out": directory / f"chunk-{index:04d}"})
    data, performance = sim.run(jobs)
    records = [{"label": label, "tier": locked[geometry]["tier"], "geometry": geometry, "seed": seed,
                "chunk": job["out"].name, **result}
               for (label, geometry, seed), job, result in zip(assignments, jobs, data, strict=True)]
    manifest = {"schema": 2, "parent_sha": PARENT_SHA, "online": online, "stream": stream,
                "sources": {label: {**source, "snapshot": str(source["snapshot"])}
                            for label, source in sources.items() if any(label == case[0] for case in assignments)},
                "records": records, "performance": performance}
    write(complete, manifest)
    return manifest
