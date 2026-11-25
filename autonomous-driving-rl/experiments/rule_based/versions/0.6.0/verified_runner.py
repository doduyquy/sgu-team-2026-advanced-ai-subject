"""Reproducible paired TRAIN development runs with immutable source snapshots.

This is not a substitute for the launcher's canonical VALIDATION/TEST suites.
"""

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import types
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import metadrive
from metadrive import MetaDriveEnv
from src.agents.rule_based_agent import Stage1RuleBasedAgent
from src.launcher.cases import load_locked_geometry_block_sequence, resolve_sandbox_case
from src.launcher.executor import build_metadrive_case_config
from src.launcher.models import LauncherMode, LaunchRequestV1, validate_custom_run_id
from src.logging.agent_experiment import DecisionTrace, save_agent_provenance, sha256_file, write_json
from src.platform.agent import AgentPublicEpisodeContext
from src.platform.agent_input import build_agent_input
from src.platform.action_adapter import ContinuousBox2Adapter
from src.platform.episode import classify_episode_outcome

ACTIVE_RUN_DIR = None

def load_snapshot(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def main(argv=None):
    global ACTIVE_RUN_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "experiments/rule_based/train_development_v1.json")
    parser.add_argument("--variants", nargs="+", choices=("baseline", "current"), default=["baseline", "current"])
    parser.add_argument("--env-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--limit-cases", type=int, default=None, help="Explicit TRAIN smoke subset; recorded in manifest")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--baseline-version", default="0.4.0")
    args = parser.parse_args(argv)
    if len(set(args.variants)) != len(args.variants):
        parser.error("Variants must be distinct")
    if args.limit_cases is not None and args.limit_cases <= 0:
        parser.error("--limit-cases must be positive")
    if not args.baseline_version or any(part in args.baseline_version for part in ("/", "\\", "..")):
        parser.error("Invalid baseline version")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = []
    for geometry in manifest["geometries"]:
        for seed in (args.env_seeds if args.env_seeds is not None else manifest["environment_seeds"]):
            case = resolve_sandbox_case(LaunchRequestV1(
                mode=LauncherMode.SANDBOX, agent_id="stage1_rule_based",
                tier=geometry["tier"], sequence=geometry["sequence"],
                geometry_generation_seed=geometry["geometry_seed"], environment_seed=seed))
            if case.split != "TRAIN":
                raise ValueError("Experiment may consume TRAIN cases only")
            cases.append(case)
    if args.limit_cases is not None:
        cases = cases[:args.limit_cases]
    if len({case.case_id for case in cases}) != len(cases):
        parser.error("Selected case manifest contains duplicates")
    run_id = args.run_id or ("rule_based_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:6])
    validate_custom_run_id(run_id, ROOT / "runs")
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ACTIVE_RUN_DIR = run_dir
    write_json(run_dir / "run_state.json", {"status": "RUNNING", "run_id": run_id,
               "started_at_utc": timestamp(), "phase": "SETUP", "noncanonical": True})
    write_json(run_dir / "resolved_cases.json", {
        "noncanonical": True, "split": "TRAIN", "source_manifest_sha256": sha256_file(args.manifest),
        "request": vars(args) | {"manifest": str(args.manifest)},
        "cases": [case.to_dict() for case in cases],
    }, exclusive=True)
    (run_dir / "runner.py").write_bytes(Path(__file__).read_bytes())
    versions = {}
    for package in ("numpy", "panda3d", "metadrive-simulator"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    simulator_root = Path(metadrive.__file__).resolve().parent.parent
    simulator_head = subprocess.run(["git", "-c", f"safe.directory={simulator_root.as_posix()}",
                                     "rev-parse", "HEAD"], cwd=simulator_root,
                                    capture_output=True, text=True, encoding="utf-8")
    write_json(run_dir / "runtime.json", {"python": sys.version, "executable": sys.executable,
               "platform": platform.platform(), "packages": versions, "git_head": head.stdout.strip(),
               "git_dirty": bool(dirty.stdout.strip()), "git_status": dirty.stdout,
               "metadrive_module_version": getattr(metadrive, "__version__", None),
               "metadrive_source_path": str(simulator_root),
               "metadrive_commit": simulator_head.stdout.strip() or None,
               "control_dt_s": 0.1, "runner_sha256": sha256_file(__file__)}, exclusive=True)
    baseline_dir = ROOT / "experiments/rule_based/versions" / args.baseline_version
    variants = {}
    for variant in args.variants:
        if variant == "baseline":
            package_name = "_baseline_package_" + run_id
            package = types.ModuleType(package_name)
            package.__path__ = [str(baseline_dir)]
            sys.modules[package_name] = package
            policy = load_snapshot(package_name + ".policy", baseline_dir / "policy.py")
            inputs = load_snapshot("_baseline_input_" + run_id, baseline_dir / "agent_input.py")
            agent = policy.Stage1RuleBasedAgent()
            builder = inputs.build_agent_input
            sources = {"policy.py": baseline_dir / "policy.py", "agent_input.py": baseline_dir / "agent_input.py"}
            for helper in baseline_dir.glob("rule_based_*.py"):
                sources[helper.name] = helper
        else:
            agent, builder, sources = Stage1RuleBasedAgent(), build_agent_input, None
        variant_dir = run_dir / variant
        variant_dir.mkdir()
        save_agent_provenance(variant_dir, agent, sources)
        variants[variant] = (agent, builder, variant_dir)
    state = {"status": "RUNNING", "started_at_utc": timestamp(), "run_id": run_id,
             "expected_episodes": len(cases) * len(variants), "completed_episodes": 0,
             "technical_errors": 0, "noncanonical": True}
    write_json(run_dir / "run_state.json", state)
    results = []
    fields = ["case_id", "variant", "agent_version", "tier", "sequence", "geometry_seed", "environment_seed",
              "steps", "clean_success", "terminal_reason", "crash", "out_of_road", "boundary_contact",
              "route_completion", "mean_speed_kmh", "max_speed_kmh", "max_abs_lane_offset_m",
              "simulation_time_s", "min_observed_gap_m", "act_latency_p95_ms", "technical_error"]
    print("RUN_DIR", run_dir, flush=True)
    try:
        with (run_dir / "episodes.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, case in enumerate(cases):
                blocks, _, _ = load_locked_geometry_block_sequence(case.to_dict())
                for variant, (agent, builder, variant_dir) in variants.items():
                    if (run_dir / "STOP_REQUESTED").exists():
                        raise KeyboardInterrupt("Experiment cancellation marker detected")
                    episode_dir = variant_dir / f"episode_{index:03d}"
                    episode_dir.mkdir()
                    config = build_metadrive_case_config(case, blocks)
                    config["log_level"] = 50
                    write_json(episode_dir / "environment.json", {"case": case.to_dict(), "configuration": config}, exclusive=True)
                    env, trace = None, None
                    row = dict.fromkeys(fields, None)
                    row.update(case_id=case.case_id, variant=variant, agent_version=agent.descriptor.agent_version,
                               tier=case.tier, sequence=case.sequence, geometry_seed=case.geometry_generation_seed,
                               environment_seed=case.environment_seed, technical_error=None)
                    try:
                        env = MetaDriveEnv(config)
                        obs, info = env.reset(seed=case.environment_seed)
                        assumptions = {
                            "speed_normalization_max_kmh": float(env.agent.max_speed_km_h),
                            "vehicle_length_m": float(env.agent.LENGTH), "vehicle_width_m": float(env.agent.WIDTH),
                            "max_lane_width_m": float(env.current_map.MAX_LANE_WIDTH),
                            "road_scale_m": float((env.current_map.MAX_LANE_NUM + 1) * env.current_map.MAX_LANE_WIDTH),
                            "lidar": dict(env.agent.config["lidar"]),
                            "random_agent_model": bool(env.config["random_agent_model"]),
                        }
                        write_json(episode_dir / "observation_assumptions.json", assumptions, exclusive=True)
                        if (assumptions["speed_normalization_max_kmh"] != agent.observation_max_speed_kmh
                                or abs(assumptions["vehicle_length_m"] - agent.vehicle_length_m) > 1e-6
                                or abs(assumptions["vehicle_width_m"] - agent.vehicle_width_m) > 1e-6
                                or assumptions["max_lane_width_m"] != agent.observation_max_lane_width_m
                                or assumptions["road_scale_m"] != agent.observation_road_scale_m
                                or assumptions["random_agent_model"]
                                or assumptions["lidar"]["num_lasers"] != 240
                                or assumptions["lidar"]["distance"] != 50
                                or assumptions["lidar"]["gaussian_noise"] != 0
                                or assumptions["lidar"]["dropout_prob"] != 0):
                            raise ValueError("Simulator observation configuration differs from policy assumptions")
                        agent.reset(AgentPublicEpisodeContext(action_adapter_id="continuous_box2_v1",
                                                             horizon_steps=case.horizon_steps))
                        trace = DecisionTrace(episode_dir / "decisions.jsonl.gz")
                        flags, speeds, latencies, gaps = {}, [], [], []
                        max_offset, boundary = 0.0, False
                        for step in range(case.horizon_steps):
                            if step % 10 == 0 and (run_dir / "STOP_REQUESTED").exists():
                                write_json(episode_dir / "interrupted.json", {
                                    "case_id": case.case_id, "completed_steps": step,
                                    "reason": "STOP_REQUESTED", "partial_episode": True,
                                }, exclusive=True)
                                raise KeyboardInterrupt("Experiment cancellation marker detected")
                            actors = list(getattr(env.engine.traffic_manager, "traffic_vehicles", []))
                            inp = builder(obs, env.agent, actors, env.agent.navigation, env.current_map.road_network, step)
                            start = time.perf_counter_ns()
                            decision = agent.act(inp)
                            latency = (time.perf_counter_ns() - start) / 1e6
                            action = ContinuousBox2Adapter().to_canonical(decision.action_payload).to_numpy()
                            trace.write(case.case_id, inp, decision, action, latency)
                            latencies.append(latency)
                            gap = (decision.diagnostics or {}).get("front_gap_m")
                            if gap is not None:
                                gaps.append(float(gap))
                            speeds.append(float(env.agent.speed_km_h))
                            max_offset = max(max_offset, abs(float(env.agent.lane.local_coordinates(env.agent.position)[1])))
                            obs, _, terminated, truncated, info = env.step(action)
                            boundary = boundary or bool(env.agent.on_yellow_continuous_line or env.agent.on_white_continuous_line)
                            for key in ("arrive_dest", "out_of_road", "crash_vehicle", "crash_object", "crash_building", "crash_human", "crash_sidewalk"):
                                flags[key] = flags.get(key, False) or bool(info.get(key, False))
                            if terminated or truncated:
                                break
                        outcome = classify_episode_outcome(flags, terminated, truncated or len(speeds) >= case.horizon_steps)
                        row.update(steps=len(speeds), clean_success=outcome.clean_success,
                                   terminal_reason=outcome.primary_reason.value,
                                   crash=any(flags.get(key, False) for key in ("crash_vehicle", "crash_object", "crash_building", "crash_human", "crash_sidewalk")),
                                   out_of_road=flags.get("out_of_road", False), boundary_contact=boundary,
                                   route_completion=float(info.get("route_completion", 0)),
                                   mean_speed_kmh=float(np.mean(speeds)), max_speed_kmh=max(speeds),
                                   max_abs_lane_offset_m=max_offset, simulation_time_s=len(speeds) * 0.1,
                                   min_observed_gap_m=min(gaps) if gaps else None,
                                   act_latency_p95_ms=float(np.percentile(latencies, 95)))
                    except Exception as exc:
                        row["technical_error"] = f"{type(exc).__name__}: {exc}"
                        (episode_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                        state["technical_errors"] += 1
                    finally:
                        if trace is not None:
                            trace.close()
                        if env is not None:
                            env.close()
                    write_json(episode_dir / "result.json", row, exclusive=True)
                    writer.writerow(row)
                    stream.flush()
                    results.append(row)
                    state["completed_episodes"] += 1
                    write_json(run_dir / "run_state.json", state)
                    print(json.dumps(row, allow_nan=False), flush=True)
        summary = {}
        for variant in variants:
            rows = [row for row in results if row["variant"] == variant]
            completed = [row for row in rows if row["technical_error"] is None]
            summary[variant] = {"episodes": len(rows), "technical_errors": len(rows) - len(completed),
                                "clean_successes": sum(bool(row["clean_success"]) for row in completed),
                                "crashes": sum(bool(row["crash"]) for row in completed),
                                "out_of_road": sum(bool(row["out_of_road"]) for row in completed),
                                "mean_route_completion": float(np.mean([row["route_completion"] for row in completed])) if completed else None}
        write_json(run_dir / "summary.json", summary, exclusive=True)
        state["status"] = "COMPLETED_WITH_ERRORS" if state["technical_errors"] else "COMPLETED"
    except KeyboardInterrupt:
        state["status"] = "INTERRUPTED"
        raise
    except Exception:
        state["status"] = "FAILED"
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        for agent, _, _ in variants.values():
            agent.close()
        state["finished_at_utc"] = timestamp()
        write_json(run_dir / "run_state.json", state)
    print("SUMMARY", json.dumps(summary), flush=True)
    return 1 if state["technical_errors"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseException as exc:
        # Setup errors and cancellation must leave a durable non-success state,
        # even before the episode loop's own exception handlers are installed.
        if ACTIVE_RUN_DIR is not None:
            state_path = ACTIVE_RUN_DIR / "run_state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["status"] == "RUNNING":
                state.update(status="INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                             finished_at_utc=timestamp())
                write_json(state_path, state)
                (ACTIVE_RUN_DIR / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
