"""Re-run the four Stage 2 planners on the paired 12-family TRAIN panel."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
METHODS = ("stage2_bfs", "stage2_dijkstra", "stage2_astar", "stage2_hybrid_astar")
TIERS = ("Easy", "Medium", "Hard", "Extreme")


def source_hash(runner_path=Path(__file__)):
    digest = hashlib.sha256()
    for folder in ("src", "configs"):
        for file in sorted((ROOT / folder).rglob("*")):
            if file.is_file() and "__pycache__" not in file.parts and file.suffix in {".py", ".json", ".csv"}:
                digest.update(file.relative_to(ROOT).as_posix().encode() + b"\0" + file.read_bytes())
    digest.update(runner_path.read_bytes())
    return digest.hexdigest()


def schedule():
    from src.launcher.cases import LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256, load_manifest_csv

    rows = load_manifest_csv(ROOT / "results/audits/evaluation_protocol/geometry_split_manifest.csv",
                             expected_sha256=LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256)
    cases = []
    for tier in TIERS:
        eligible = [row for row in rows if row["tier"] == tier and row["split"] == "TRAIN"]
        for family in sorted({row["sequence"] for row in eligible}):
            selected = min((row for row in eligible if row["sequence"] == family),
                           key=lambda row: int(row["geometry_generation_seed"]))
            cases.append(dict(tier=tier, sequence=family, geom_seed=int(selected["geometry_generation_seed"])))
    if len(cases) != 12 or any(sum(c["tier"] == tier for c in cases) != 3 for tier in TIERS):
        raise ValueError("Expected three TRAIN families in each tier")
    return [{**case, "agent": agent} for case in cases for agent in METHODS]


def run_job(job, output, expected_source, no_traffic=False):
    from PySide6.QtCore import QProcess  # Qt before MetaDrive on Windows
    from metadrive.engine.logger import get_logger

    logger = get_logger()
    label = f"{job['agent']}-{job['tier']}-{job['sequence']}"
    with (output / f"{label}.log").open("w", encoding="utf-8", buffering=1) as log, ExitStack() as stack:
        for handler in logger.handlers:
            if hasattr(handler, "setStream"):
                stack.callback(handler.setStream, handler.setStream(log))
        with redirect_stdout(log), redirect_stderr(log):
            from src.training.runtime import local_training_runtime
            from src.launcher import executor
            from src.launcher.models import LaunchRequestV1, LauncherMode, compute_resolved_plan_sha256
            from src.launcher.resolver import resolve_experiment_plan
            from src.platform import WandbMode, capture_environment_provenance
            from src.platform import PINNED_METADRIVE_COMMIT, PINNED_METADRIVE_VERSION
            from src.workbench.results_repository import RunArtifactRepository

            if source_hash() != expected_source:
                raise RuntimeError("Source changed after scheduling")
            request = LaunchRequestV1(LauncherMode.SANDBOX, job["agent"], tier=job["tier"],
                sequence=job["sequence"], geometry_generation_seed=job["geom_seed"],
                environment_seed=0, render_mode="OFF", wandb_mode=WandbMode.DISABLED)
            with local_training_runtime(ROOT):
                provenance = capture_environment_provenance(ROOT)
                if (provenance["metadrive_version"], provenance["metadrive_commit"]) != (
                        PINNED_METADRIVE_VERSION, PINNED_METADRIVE_COMMIT):
                    raise RuntimeError("MetaDrive pin mismatch")
                plan = resolve_experiment_plan(request, project_root=ROOT)
                case, = plan.resolved_cases
                if case.split != "TRAIN" or plan.canonical_run:
                    raise RuntimeError("Expected one TRAIN case")
                reference_density = case.traffic_density
                checked = 0
                if no_traffic:
                    case = replace(case, traffic_density=0.)
                    plan = replace(plan, resolved_cases=(case,))
                    plan = replace(plan, resolved_plan_sha256=compute_resolved_plan_sha256(plan.to_dict()))
                    build_input = executor.build_agent_input

                    def check_empty_traffic(**kwargs):
                        nonlocal checked
                        if kwargs["traffic_vehicles"]:
                            raise RuntimeError("A traffic vehicle was observed in the empty-road run")
                        checked += 1
                        return build_input(**kwargs)

                    stack.enter_context(patch.object(executor, "build_agent_input", check_empty_traffic))
                runs = output / "runs"
                report = executor.ExperimentExecutor(plan, project_root=ROOT, runs_root=runs,
                                                      custom_run_id=label).execute()
                snapshot = RunArtifactRepository(runs).load_run_snapshot(report.run_id)
            if (report.status != "COMPLETE" or report.completed_episodes != 1
                    or snapshot.integrity_status.value != "VERIFIED"
                    or source_hash() != expected_source):
                raise RuntimeError(f"Failed to verify {label}")
            episode, = snapshot.episode_rows
            timing, = snapshot.timing_rows
            if no_traffic and (checked == 0 or checked != int(episode["episode_steps"])):
                raise RuntimeError("Incomplete traffic-vehicle verification")
            metrics = snapshot.summary_payload["overall_metrics"]
            return dict(**job, case_id=case.case_id, geometry_sha256=case.geometry_sha256,
                environment_seed=case.environment_seed, traffic_density=case.traffic_density,
                reference_traffic_density=reference_density, traffic_frames_checked=checked,
                horizon_steps=case.horizon_steps, run_id=report.run_id,
                terminal_reason=episode["primary_terminal_reason"], steps=int(episode["episode_steps"]),
                success=bool(metrics["clean_success_rate"]), unsafe=bool(metrics["safety_failure_rate"]),
                route=float(metrics["mean_final_route_completion"]), timing=timing)


def write_progress(output, state):
    temp = output / "progress.tmp"
    temp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(output / "progress.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-traffic", action="store_true",
                        help="Set traffic density to zero for each TRAIN case and verify every decision")
    args = parser.parse_args()
    jobs = schedule()
    if not args.run:
        print(json.dumps([dict(job, traffic_density=0.) for job in jobs] if args.no_traffic else jobs, indent=2))
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or ROOT / "artifacts" / "search_recheck" / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    expected_source = source_hash()
    state = dict(status="RUNNING", split="TRAIN", canonical_run=False, workers=args.workers,
                 source_sha256=expected_source, no_traffic=args.no_traffic, jobs=jobs, completed=[], errors=[])
    write_progress(output, state)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    print(f"{len(jobs)} paired TRAIN episodes | {output}", flush=True)
    try:
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(run_job, job, output, expected_source, args.no_traffic): job for job in jobs}
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as error:
                    state["errors"].append(dict(job=futures[future], error=str(error)))
                    print(f"FAILED {futures[future]}: {error}", flush=True)
                else:
                    state["completed"].append(row)
                    print(f"{len(state['completed'])}/{len(jobs)} {row['agent']} {row['case_id']} "
                          f"{row['terminal_reason']} {row['route']:.2%}", flush=True)
                write_progress(output, state)
    except KeyboardInterrupt:
        state["status"] = "INTERRUPTED"
        write_progress(output, state)
        raise
    state["status"] = "COMPLETE" if len(state["completed"]) == len(jobs) else "FAILED"
    write_progress(output, state)
    rows = [{key: value for key, value in row.items() if key != "timing"} for row in state["completed"]]
    if rows:
        with (output / "episodes.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: (TIERS.index(row["tier"]), row["sequence"], METHODS.index(row["agent"]))))
    return int(state["status"] != "COMPLETE")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
