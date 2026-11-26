"""Headless, uncertified candidate evaluation; never changes benchmark eligibility.

TEST always uses all 60 locked cases and inference seeds 101/202/303. A separate
--smoke operation uses one TRAIN case and at most ten decisions per agent.
Workers import a recorded source snapshot so edits to the working tree during a
long evaluation cannot silently change its policies. Results are not Gate-7 runs.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback


PROJECT = Path(__file__).resolve().parents[1]
AGENTS = {"random": "stage0_random_naive", "mppi": "stage3_mppi"}
SEEDS = (101, 202, 303)
FLAGS = ("arrive_dest", "out_of_road", "crash_vehicle", "crash_object",
         "crash_building", "crash_human", "crash_sidewalk", "max_step")


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def make_jobs(agent_names, cases, smoke=False, seeds=SEEDS):
    expected_split, expected_count = ("TRAIN", 1) if smoke else ("TEST", 60)
    if len(cases) != expected_count or any(c["split"] != expected_split for c in cases):
        raise ValueError("TEST requires exactly 60 locked cases; smoke requires one TRAIN case")
    if len({c["case_id"] for c in cases}) != expected_count:
        raise ValueError("Duplicate case IDs are forbidden")
    if not smoke and [c["protocol_order_index"] for c in cases] != list(range(1, 61)):
        raise ValueError("TEST cases must retain the locked manifest order")
    target_seeds = (101,) if smoke else tuple(dict.fromkeys(seeds))
    return [{"agent_id": AGENTS[name], "agent_seed": seed, "cases": cases,
             "smoke": smoke} for name in dict.fromkeys(agent_names)
            for seed in target_seeds]


def snapshot_sources(destination):
    """Copy only execution sources/configuration and locked protocol evidence."""
    source = destination / "source"
    for relative in ("src", "configs", "results/audits/evaluation_protocol"):
        shutil.copytree(PROJECT / relative, source / relative,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    shutil.copy2(__file__, source / "evaluate_candidate_test.py")
    hashes = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(source.rglob("*")) if p.is_file()}
    return source, hashes


def check_snapshot(source, hashes):
    for relative, digest in hashes.items():
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source snapshot changed: {relative}")


def initialize_worker(source):
    # Avoid nested BLAS thread pools competing with independent simulator workers.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def evaluate_job(job, output, hashes):
    source = output / "source"
    label = job.get("job_id", f"{job['agent_id']}_seed{job['agent_seed']}")
    folder = output / label
    folder.mkdir()
    state = {"canonical_run": False, "status": "RUNNING", "completed_episodes": 0,
             "expected_episodes": len(job["cases"]), "agent_id": job["agent_id"],
             "agent_seed": job["agent_seed"]}
    write_json(folder / "progress.json", state)
    started = time.perf_counter()
    records, geometry_cache = [], {}
    try:
        check_snapshot(source, hashes)
        from PySide6.QtCore import QProcess  # Qt must load before Panda3D on Windows.
        from metadrive.engine.logger import get_logger

        logger = get_logger()
        with (folder / "runtime.log").open("w", encoding="utf-8", buffering=1) as log, ExitStack() as stack:
            for handler in logger.handlers:
                if hasattr(handler, "setStream"):
                    stack.callback(handler.setStream, handler.setStream(log))
            with redirect_stdout(log), redirect_stderr(log):
                from metadrive.envs.metadrive_env import MetaDriveEnv
                from src.launcher.cases import load_locked_geometry_block_sequence
                from src.launcher.executor import build_metadrive_case_config
                from src.launcher.models import ResolvedCaseV1
                from src.launcher.registry import build_canonical_agent_registry
                from src.platform import (AgentPublicEpisodeContext, InputProfileId,
                                          build_agent_input, classify_episode_outcome,
                                          get_action_adapter)

                registry = build_canonical_agent_registry()
                registration = registry.get(job["agent_id"])
                agent = registry.get_factory(job["agent_id"])()
                implementation = Path(sys.modules[type(agent).__module__].__file__).resolve()
                if not implementation.is_relative_to(source):
                    agent.close()
                    raise RuntimeError(f"Policy was not imported from the frozen snapshot: {implementation}")
                state["policy_source"] = implementation.relative_to(source).as_posix()
                adapter = get_action_adapter(registration.action_adapter_id)
                try:
                    with (folder / "episodes.jsonl").open("w", encoding="utf-8", buffering=1) as stream:
                        for index, raw_case in enumerate(job["cases"], 1):
                            if (output / "STOP").exists():
                                raise InterruptedError("Stopped by STOP file or Ctrl+C")
                            case = ResolvedCaseV1(**raw_case)
                            state["current_case_id"] = case.case_id
                            write_json(folder / "progress.json", state)
                            if case.geometry_sha256 not in geometry_cache:
                                geometry_cache[case.geometry_sha256] = load_locked_geometry_block_sequence(
                                    raw_case, source)[0]
                            config = build_metadrive_case_config(
                                case, copy.deepcopy(geometry_cache[case.geometry_sha256]), "OFF")
                            env = MetaDriveEnv(config)
                            try:
                                obs, info = env.reset(seed=case.environment_seed)
                                agent.reset(AgentPublicEpisodeContext(
                                    control_frequency_hz=10, control_dt_s=.1,
                                    horizon_steps=case.horizon_steps,
                                    input_profile_id=registration.input_profile_id,
                                    action_adapter_id=registration.action_adapter_id,
                                    mode="INFERENCE"), agent_seed=job["agent_seed"])
                                flags, latencies = {}, []
                                route = maximum_route = 0.
                                peak_traffic = 0
                                limit = min(10, case.horizon_steps) if job["smoke"] else case.horizon_steps
                                for step in range(limit):
                                    if (output / "STOP").exists():
                                        raise InterruptedError("Stopped by STOP file or Ctrl+C")
                                    traffic = list(getattr(env.engine.traffic_manager, "traffic_vehicles", []))
                                    if job.get("traffic_mode") == "off" and traffic:
                                        raise RuntimeError(f"Traffic appeared in the empty-road case {case.case_id}")
                                    peak_traffic = max(peak_traffic, len(traffic))
                                    inputs = build_agent_input(
                                        raw_obs=obs, ego_vehicle=env.agent,
                                        traffic_vehicles=traffic,
                                        navigation=env.agent.navigation,
                                        road_network=env.current_map.road_network, step_index=step,
                                        profile_id=InputProfileId(registration.input_profile_id))
                                    t0 = time.perf_counter_ns()
                                    decision = agent.act(inputs)
                                    latencies.append((time.perf_counter_ns() - t0) / 1e6)
                                    action = adapter.to_canonical(decision.action_payload)
                                    obs, _, terminated, truncated, info = env.step(action.to_numpy())
                                    flags.update({key: True for key in FLAGS if info.get(key, False)})
                                    outcome = classify_episode_outcome(
                                        raw_flags=flags, terminated=terminated,
                                        truncated=truncated or step + 1 >= case.horizon_steps)
                                    route = float(info.get("route_completion", 0.))
                                    maximum_route = max(maximum_route, route)
                                    if outcome.is_done:
                                        break
                                reason = outcome.primary_reason.value if outcome.is_done else "SMOKE_LIMIT"
                                record = {**raw_case, "agent_id": job["agent_id"],
                                          "agent_seed": job["agent_seed"], "canonical_run": False,
                                          "evaluation_kind": job.get("evaluation_kind",
                                              "TRAIN_SMOKE" if job["smoke"] else "UNCERTIFIED_CANDIDATE_TEST"),
                                          "episode_steps": step + 1, "simulation_time_s": (step + 1) * .1,
                                          "primary_reason": reason, "clean_success": outcome.clean_success,
                                          "final_route_completion": route, "max_route_completion": maximum_route,
                                          "mean_act_ms": sum(latencies) / len(latencies), "raw_flags": flags}
                                if "traffic_mode" in job:
                                    record.update(traffic_mode=job["traffic_mode"], peak_traffic=peak_traffic)
                                stream.write(json.dumps(record, allow_nan=False) + "\n")
                                records.append(record)
                            finally:
                                env.close()
                            state["completed_episodes"] = index
                            write_json(folder / "progress.json", state)
                            print(f"[{label}] {index}/{len(job['cases'])} "
                                  f"{case.case_id} | {reason} | route={route:.1%}", file=sys.__stdout__, flush=True)
                finally:
                    agent.close()
        check_snapshot(source, hashes)
        state.update(status="SMOKE_COMPLETE" if job["smoke"] else "COMPLETE",
                     wall_time_s=time.perf_counter() - started)
        write_json(folder / "progress.json", state)
        return records
    except BaseException as error:
        state.update(status="INTERRUPTED" if isinstance(error, InterruptedError) else "FAILED",
                     error=str(error), traceback=traceback.format_exc(), wall_time_s=time.perf_counter() - started)
        write_json(folder / "progress.json", state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", nargs="+", choices=tuple(AGENTS), default=list(AGENTS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=(101, 202, 303), default=list(SEEDS))
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", action="store_true", help="Read-only: resolve and verify, do not simulate")
    parser.add_argument("--smoke", action="store_true", help="Only a ten-decision TRAIN runtime check")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(PROJECT))
    from src.training.runtime import local_training_runtime
    with local_training_runtime(PROJECT):
        from PySide6.QtCore import QProcess  # noqa: F401
        from src.launcher.cases import resolve_sandbox_case, resolve_test_cases
        from src.launcher.models import LaunchRequestV1, LauncherMode
        from src.launcher.registry import build_canonical_agent_registry
        from src.platform import (PINNED_METADRIVE_COMMIT, PINNED_METADRIVE_VERSION,
                                  capture_environment_provenance, capture_git_provenance)
        environment = capture_environment_provenance(PROJECT)
        if (environment["metadrive_version"], environment["metadrive_commit"]) != (
                PINNED_METADRIVE_VERSION, PINNED_METADRIVE_COMMIT):
            raise RuntimeError(f"Exact MetaDrive pin required: {environment}")
        if args.smoke:
            cases = [resolve_sandbox_case(LaunchRequestV1(
                LauncherMode.SANDBOX, "stage0_random_naive", tier="Easy"), PROJECT)]
        else:
            cases = resolve_test_cases(PROJECT)
        jobs = make_jobs(args.agents, [c.to_dict() for c in cases], args.smoke, seeds=args.seeds)
        registry = build_canonical_agent_registry()
        agents = {}
        for name in dict.fromkeys(args.agents):
            aid = AGENTS[name]
            factory = registry.get_factory(aid)
            agents[aid] = {"registration": registry.get(aid).to_dict(),
                           "parameters": factory.experiment_parameters() if hasattr(factory, "experiment_parameters") else None}
        manifest = {"kind": "TRAIN_SMOKE" if args.smoke else "UNCERTIFIED_CANDIDATE_TEST",
                    "canonical_run": False, "status": "PLANNED", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "expected_episodes": sum(len(j["cases"]) for j in jobs), "workers": args.workers,
                    "control_dt_s": .1, "render": "OFF", "wandb": "DISABLED", "blas_threads_per_worker": 1,
                    "environment": environment, "git": capture_git_provenance(PROJECT.parent),
                    "agents": agents, "jobs": jobs,
                    "notice": "Uncertified candidate study. Not a canonical benchmark. TEST must not be used to tune policies."}
        if args.plan:
            print(json.dumps(manifest, indent=2))
            return 0
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = (args.output or PROJECT / "artifacts/candidate_test" / stamp).resolve()
        for relative in ("src", "configs", "results/audits/evaluation_protocol"):
            if output.is_relative_to(PROJECT / relative):
                raise ValueError("Output must be outside the execution source/configuration directories")
        output.mkdir(parents=True, exist_ok=False)
        source, hashes = snapshot_sources(output)
        manifest.update(status="RUNNING", source_sha256=hashes)
        write_json(output / "session.json", manifest)
        print(f"{manifest['kind']} | {manifest['expected_episodes']} episodes | workers={args.workers}", flush=True)
        print(f"Results: {output}\nCtrl+C or create {output / 'STOP'} to stop.", flush=True)
        status, failure = "COMPLETE", None
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialize_worker, initargs=(source,)) as pool:
            futures = {pool.submit(evaluate_job, job, output, hashes): i for i, job in enumerate(jobs)}
            try:
                for future in as_completed(futures):
                    future.result()
            except (Exception, KeyboardInterrupt) as error:
                status = "INTERRUPTED" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "FAILED"
                failure = str(error) or type(error).__name__
                (output / "STOP").touch()
                for future in futures:
                    future.cancel()
        # Preserve every completed episode, including partial replicates after an interruption.
        records = []
        for job in jobs:
            path = output / f"{job['agent_id']}_seed{job['agent_seed']}" / "episodes.jsonl"
            if path.exists():
                records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
        if records:
            with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows({**r, "raw_flags": json.dumps(r["raw_flags"])} for r in records)
        if status == "COMPLETE" and len(records) != manifest["expected_episodes"]:
            status, failure = "FAILED", "Incomplete episode coverage"
        if status == "COMPLETE" and args.smoke:
            status = "SMOKE_COMPLETE"
        manifest.update(status=status, completed_episodes=len(records), error=failure,
                        finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "session.json", manifest)
        print(f"{status}: {len(records)}/{manifest['expected_episodes']} | {output}", flush=True)
        if failure:
            print(f"[ERROR] {failure}; see each worker's progress.json and runtime.log", file=sys.stderr)
        return 0 if status in ("COMPLETE", "SMOKE_COMPLETE") else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        raise SystemExit(1)
