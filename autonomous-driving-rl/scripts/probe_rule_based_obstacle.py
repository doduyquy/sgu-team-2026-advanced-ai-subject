"""Controlled TRAIN-derived LiDAR obstacle stop/restart and side-object probes.

Traffic is disabled and a barrier is spawned deliberately. These are sensor
integration scenarios, NOT canonical map-driving scores or benchmark episodes.
"""

import argparse
import importlib.metadata
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import metadrive
from metadrive import MetaDriveEnv
from metadrive.component.static_object.traffic_object import TrafficBarrier
from src.agents.rule_based_agent import Stage1RuleBasedAgent
from src.launcher.cases import resolve_sandbox_case, load_locked_geometry_block_sequence
from src.launcher.executor import build_metadrive_case_config
from src.launcher.models import LaunchRequestV1, LauncherMode, validate_custom_run_id
from src.logging.agent_experiment import save_agent_provenance, write_json, DecisionTrace
from src.platform.agent import AgentPublicEpisodeContext
from src.platform.agent_input import build_agent_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    validate_custom_run_id(args.run_id, ROOT / "runs")
    root = ROOT / "runs" / args.run_id
    root.mkdir(exist_ok=False)
    state = {"status": "RUNNING", "noncanonical": True, "controlled_scenario": True,
             "started_at_utc": datetime.now(timezone.utc).isoformat(), "completed_scenarios": 0}
    write_json(root / "run_state.json", state, exclusive=True)
    results = []
    try:
        (root / "runner.py").write_bytes(Path(__file__).read_bytes())
        simulator_root = Path(metadrive.__file__).resolve().parent.parent
        commit = subprocess.run(["git", "-c", f"safe.directory={simulator_root.as_posix()}",
                                 "rev-parse", "HEAD"], cwd=simulator_root,
                                capture_output=True, text=True, encoding="utf8")
        try:
            simulator_version = importlib.metadata.version("metadrive-simulator")
        except importlib.metadata.PackageNotFoundError:
            simulator_version = None
        write_json(root / "runtime.json", {"python": sys.version, "executable": sys.executable,
                   "platform": platform.platform(), "numpy": np.__version__,
                   "metadrive": simulator_version, "metadrive_commit": commit.stdout.strip(),
                   "control_dt_s": 0.1}, exclusive=True)
        for label, lateral in (("on_path_stop_restart", 0.0), ("side_object", 6.0)):
            folder = root / label
            folder.mkdir()
            agent = Stage1RuleBasedAgent()
            save_agent_provenance(folder, agent)
            case = resolve_sandbox_case(LaunchRequestV1(mode=LauncherMode.SANDBOX,
                 agent_id="stage1_rule_based", tier="Easy", sequence="SCS",
                 geometry_generation_seed=0, environment_seed=0))
            blocks, _, _ = load_locked_geometry_block_sequence(case.to_dict())
            config = build_metadrive_case_config(case, blocks)
            config.update(traffic_density=0.0, log_level=50)
            write_json(folder / "environment.json", {"case": case.to_dict(), "configuration": config,
                       "intervention": {"traffic_density": 0, "barrier_distance_m": 20,
                                        "barrier_lateral_m": lateral,
                                        "remove_after_stable_stop": lateral == 0}}, exclusive=True)
            env, trace = None, None
            try:
                env = MetaDriveEnv(config)
                env.reset(seed=0)
                ego = env.agent
                origin = np.asarray(ego.position).copy()
                heading = ego.heading_theta
                forward = np.array((math.cos(heading), math.sin(heading)))
                left = np.array((-forward[1], forward[0]))
                barrier = env.engine.spawn_object(TrafficBarrier,
                    position=origin + forward * 20 + left * lateral,
                    heading_theta=heading + math.pi / 2, static=True, random_seed=0)
                barrier_id = barrier.id
                write_json(folder / "sensor_object.json", {
                    "object_class": "TrafficBarrier", "length_m": barrier.LENGTH,
                    "width_m": barrier.WIDTH, "height_m": barrier.HEIGHT,
                    "lidar_mask": int(ego.engine.get_sensor("lidar").mask.getWord()),
                    "physics_world": "dynamic_world", "static_body_flag": True,
                    "lidar_distance_m": ego.config["lidar"]["distance"],
                }, exclusive=True)
                obs = next(iter(env.observations.values())).observe(ego)
                agent.reset(AgentPublicEpisodeContext())
                trace = DecisionTrace(folder / "decisions.jsonl.gz")
                detected, stop_step, removed_step = False, None, None
                any_lidar_hit, min_lidar_range_m = False, 50.0
                stopped_frames, crash = 0, False
                max_speed_before, max_speed_after, min_center_distance = 0.0, 0.0, float("inf")
                barrier_position = np.asarray(barrier.position).copy()
                for step in range(350):
                    inp = build_agent_input(obs, ego, [], ego.navigation, env.current_map.road_network, step)
                    decision = agent.act(inp)
                    trace.write(case.case_id + "/" + label, inp, decision, decision.action_payload)
                    gap = decision.diagnostics.get("lidar_obstacle_gap_m")
                    any_lidar_hit = any_lidar_hit or bool(np.any(inp.core_observation.lidar_points < 0.9999))
                    min_lidar_range_m = min(min_lidar_range_m, float(np.min(inp.core_observation.lidar_points)) * 50)
                    detected = detected or gap is not None
                    speed = float(ego.speed_km_h)
                    if removed_step is None:
                        max_speed_before = max(max_speed_before, speed)
                        min_center_distance = min(min_center_distance, float(np.linalg.norm(ego.position - barrier_position)))
                    else:
                        max_speed_after = max(max_speed_after, speed)
                    if lateral == 0 and step > 30 and removed_step is None:
                        stopped_frames = stopped_frames + 1 if speed < 0.3 and gap is not None else 0
                        if stopped_frames >= 10:
                            stop_step = step
                            env.engine.clear_objects([barrier_id])
                            removed_step = step
                    obs, _, terminated, truncated, info = env.step(decision.action_payload)
                    crash = crash or any(info.get(key, False) for key in ("crash_vehicle", "crash_object", "crash_building", "out_of_road"))
                    if terminated or truncated or (removed_step is not None and step - removed_step >= 100):
                        break
                    if lateral != 0 and float((np.asarray(ego.position) - origin) @ forward) > 30:
                        break
                passed = (detected and stop_step is not None and max_speed_after > 15 and not crash) if lateral == 0 else (
                    any_lidar_hit and not detected and max_speed_before > 20 and not crash)
                result = {"scenario": label, "passed": passed, "steps": step + 1,
                          "lidar_detected_in_corridor": detected, "stable_stop_step": stop_step,
                          "raw_lidar_hit_observed": any_lidar_hit, "min_raw_lidar_range_m": min_lidar_range_m,
                          "removed_step": removed_step, "max_speed_before_kmh": max_speed_before,
                          "max_speed_after_kmh": max_speed_after, "crash_or_out_of_road": crash,
                          "min_center_distance_before_removal_m": min_center_distance,
                          "note": "Center distance is not mesh clearance; controlled intervention, not driving score"}
                write_json(folder / "result.json", result, exclusive=True)
                results.append(result)
                state["completed_scenarios"] += 1
                print(json.dumps(result), flush=True)
            finally:
                if trace is not None:
                    trace.close()
                if env is not None:
                    env.close()
                agent.close()
        state["status"] = "COMPLETED" if all(row["passed"] for row in results) else "COMPLETED_WITH_FAILED_EXPECTATIONS"
        write_json(root / "summary.json", results, exclusive=True)
    except BaseException as exc:
        state["status"] = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
        state["error"] = f"{type(exc).__name__}: {exc}"
        (root / "error.txt").write_text(traceback.format_exc(), encoding="utf8")
        raise
    finally:
        state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(root / "run_state.json", state)
    return 0 if state["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())
