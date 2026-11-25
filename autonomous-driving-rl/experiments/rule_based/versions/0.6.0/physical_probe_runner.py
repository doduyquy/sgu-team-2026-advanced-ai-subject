"""Controlled TRAIN-derived physical P4 probes, never benchmark driving scores.

Disable background traffic, warm up to a sufficiently long straight route edge,
then spawn a parked source-lane vehicle. A scripted target-lane rear vehicle is
used in the rear-gap scenario. Every intervention and physical sample is logged.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import metadrive
import numpy as np
from metadrive import MetaDriveEnv
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from src.agents.rule_based_agent import Stage1RuleBasedAgent
from src.launcher.cases import resolve_sandbox_case, load_locked_geometry_block_sequence
from src.launcher.executor import build_metadrive_case_config
from src.launcher.models import LaunchRequestV1, LauncherMode, validate_custom_run_id
from src.logging.agent_experiment import save_agent_provenance, write_json, DecisionTrace, sha256_file
from src.platform.agent import AgentPublicEpisodeContext
from src.platform.action_adapter import ContinuousBox2Adapter
from src.platform.agent_input import build_agent_input


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scenarios", nargs="+", choices=("safe_overtake", "fast_rear", "mandatory_route", "mandatory_left", "precommit_abort", "postcommit_front_block"),
                        default=["safe_overtake", "fast_rear", "mandatory_route", "mandatory_left", "precommit_abort", "postcommit_front_block"])
    args = parser.parse_args()
    validate_custom_run_id(args.run_id, ROOT / "runs")
    root = ROOT / "runs" / args.run_id
    root.mkdir(exist_ok=False)
    state = {"status": "RUNNING", "noncanonical": True, "controlled_scenario": True,
             "started_at_utc": now(), "completed_scenarios": 0}
    write_json(root / "run_state.json", state, exclusive=True)
    results = []
    try:
        (root / "runner.py").write_bytes(Path(__file__).read_bytes())
        simulator_root = Path(metadrive.__file__).resolve().parent.parent
        commit = subprocess.run(["git", "-c", f"safe.directory={simulator_root.as_posix()}", "rev-parse", "HEAD"],
                                cwd=simulator_root, capture_output=True, text=True, encoding="utf8")
        write_json(root / "runtime.json", {"python": sys.version, "executable": sys.executable,
                   "platform": platform.platform(), "numpy": np.__version__,
                   "panda3d": importlib.metadata.version("panda3d"),
                   "metadrive_module_version": getattr(metadrive, "__version__", None), "metadrive_commit": commit.stdout.strip(),
                   "control_dt_s": .1, "runner_sha256": sha256_file(__file__)}, exclusive=True)
        for label in args.scenarios:
            folder = root / label
            folder.mkdir()
            agent = Stage1RuleBasedAgent()
            save_agent_provenance(folder, agent)
            case = resolve_sandbox_case(LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="stage1_rule_based",
                   tier="Easy", sequence="SCS", geometry_generation_seed=0, environment_seed=0))
            if case.split != "TRAIN":
                raise ValueError("Probes require TRAIN geometry")
            blocks, _, _ = load_locked_geometry_block_sequence(case.to_dict())
            config = build_metadrive_case_config(case, blocks)
            config.update(traffic_density=0., log_level=50)
            if label == "mandatory_left":
                config["vehicle_config"] = {"spawn_lane_index": (">", ">>", 1)}
            write_json(folder / "environment.json", {"case": case.to_dict(), "configuration": config,
                       "intervention": {"traffic_density": 0, "parked_leader_distance_m": 23,
                                        "rear_start_distance_m": -30, "rear_speed_mps": 15,
                                        "warmup": "First straight route edge with >=50m remaining and a legal neighbor; bring ego to rest before spawning"}}, exclusive=True)
            env = trace = None
            try:
                env = MetaDriveEnv(config)
                obs, _ = env.reset(seed=0)
                ego = env.agent
                agent.reset(AgentPublicEpisodeContext())
                trace = DecisionTrace(folder / "decisions.jsonl.gz")
                actors, spawned, spawn_step, rear = [], False, None, None
                boundary = crash = out = False
                completed = veto_rear = passed_leader = aborted = mandatory_started = False
                phases = set()
                max_lateral = 0.
                selected_lane = original_lane = None
                target_blocker = None
                blocker_cleared = postcommit_blocked = False
                target_preserved = True
                original_target_id = None
                stopped_frames = 0
                metadata = {}
                with (folder / "physical.jsonl").open("x", encoding="utf8") as physical:
                    for step in range(1100):
                        inp = build_agent_input(obs, ego, actors, ego.navigation, env.current_map.road_network, step)
                        if not spawned:
                            lane = ego.navigation.current_lane
                            station, _ = lane.local_coordinates(ego.position)
                            ctx = inp.task_context.lane_context
                            options = [item for item in ctx.lanes if item.side != "CURRENT" and item.route_compatible is True] if ctx else []
                            straight = abs(lane.heading_theta_at(station + 50) - lane.heading_theta_at(station)) < .01
                            if lane.length - station >= 50 and options and straight and not inp.task_context.route_passage:
                                target = options[0]
                                current = next(item for item in ctx.lanes if item.side == "CURRENT")
                                if agent._lane_change._permission(current, target):
                                    # Intervention occurs before the next public snapshot.
                                    ego.set_velocity((0, 0))
                                    original_lane = lane
                                    selected_lane = next(item for item in env.current_map.road_network.graph[lane.index[0]][lane.index[1]]
                                                         if "|".join(map(str, item.index)) == target.lane_id)
                                    heading = float(lane.heading_theta_at(station))
                                    leader = None
                                    route_filter = None
                                    if label in {"mandatory_route", "mandatory_left"}:
                                        nav = ego.navigation
                                        edge = tuple(lane.index[:2])
                                        ri = next(i for i in range(len(nav.checkpoints)-2) if tuple(nav.checkpoints[i:i+2]) == edge)
                                        next_edge = nav.checkpoints[ri+1:ri+3]
                                        next_lanes = env.current_map.road_network.graph[next_edge[0]][next_edge[1]]
                                        endpoint = np.asarray(selected_lane.position(selected_lane.length, 0))
                                        successor = min(next_lanes, key=lambda n: float(np.linalg.norm(np.asarray(n.position(0, 0)) - endpoint)))
                                        route_filter = {"edge": next_edge, "original_lane_indices": [list(n.index) for n in next_lanes],
                                                        "retained_lane_index": list(successor.index),
                                                        "meaning": "Controlled next-route-edge availability restriction; physics/rendered lanes unchanged; stop before reaching restricted edge"}
                                        env.current_map.road_network.graph[next_edge[0]][next_edge[1]] = [successor]
                                    else:
                                        leader = env.engine.spawn_object(DefaultVehicle, vehicle_config=dict(ego.config),
                                                 position=lane.position(station + 23, 0), heading=heading, random_seed=1)
                                        leader.set_static(True)
                                        actors.append(leader)
                                    if label == "fast_rear":
                                        rear = env.engine.spawn_object(DefaultVehicle, vehicle_config=dict(ego.config),
                                               position=selected_lane.position(station - 30, 0), heading=heading, random_seed=2)
                                        actors.append(rear)
                                    metadata = {"spawn_step": step, "station_m": station, "lane_index": list(lane.index),
                                                "lane_length_m": lane.length, "target_lane_index": list(selected_lane.index),
                                                "leader_length_m": leader.LENGTH if leader else None, "leader_width_m": leader.WIDTH if leader else None,
                                                "route_availability_intervention": route_filter,
                                                "leader_station_m": station + 23, "rear_start_station_m": station - 30}
                                    write_json(folder / "intervention.json", metadata, exclusive=True)
                                    spawned, spawn_step = True, step
                                    obs = next(iter(env.observations.values())).observe(ego)
                                    inp = build_agent_input(obs, ego, actors, ego.navigation, env.current_map.road_network, step)
                        if label == "postcommit_front_block" and spawned and target_blocker is None and not blocker_cleared and agent._lane_change.committed:
                            blocker_station = original_lane.local_coordinates(ego.position)[0] + 15
                            target_blocker = env.engine.spawn_object(DefaultVehicle, vehicle_config=dict(ego.config),
                                position=selected_lane.position(blocker_station, 0), heading=ego.heading_theta, random_seed=3)
                            target_blocker.set_heading_theta(selected_lane.heading_theta_at(blocker_station))
                            target_blocker.set_static(True)
                            actors.append(target_blocker)
                            original_target_id = agent._lane_change.target_id
                            write_json(folder / "postcommit_intrusion.json", {"step": step, "station_m": blocker_station,
                                "target_id": original_target_id, "progress": agent._lane_change.last_progress,
                                "clearance_condition": "remove after 10 consecutive BLOCKED_CHANGE frames below 0.3km/h"}, exclusive=True)
                            obs = next(iter(env.observations.values())).observe(ego)
                            inp = build_agent_input(obs, ego, actors, ego.navigation, env.current_map.road_network, step)
                        if label == "precommit_abort" and spawned and rear is None and agent._lane_change.phase == "PREPARE_CHANGE":
                            rear_station = original_lane.local_coordinates(ego.position)[0] - 15
                            rear = env.engine.spawn_object(DefaultVehicle, vehicle_config=dict(ego.config),
                                   position=selected_lane.position(rear_station, 0), heading=ego.heading_theta, random_seed=2)
                            actors.append(rear)
                            metadata["rear_start_station_m"] = rear_station - 15 * (step - spawn_step) * .1
                            write_json(folder / "intrusion.json", {"step": step, "rear_station_m": rear_station,
                                       "speed_mps": 15, "policy_phase_before_injection": agent._lane_change.phase,
                                       "progress_before_injection": agent._lane_change.last_progress}, exclusive=True)
                        if rear is not None:
                            rear_station = metadata["rear_start_station_m"] + 15 * (step - spawn_step) * .1
                            rear.set_position(selected_lane.position(rear_station, 0))
                            rear.set_heading_theta(selected_lane.heading_theta_at(rear_station))
                            rear.set_velocity((math.cos(rear.heading_theta), math.sin(rear.heading_theta)), 15)
                            obs = next(iter(env.observations.values())).observe(ego)
                            inp = build_agent_input(obs, ego, actors, ego.navigation, env.current_map.road_network, step)
                        started = time.perf_counter_ns()
                        decision = agent.act(inp)
                        action = ContinuousBox2Adapter().to_canonical(decision.action_payload).to_numpy()
                        trace.write(case.case_id + "/" + label, inp, decision, action,
                                    (time.perf_counter_ns() - started) / 1e6)
                        diag = decision.diagnostics.get("lane_change", {})
                        phases.add(diag.get("phase"))
                        completed |= bool(diag.get("completed"))
                        aborted |= diag.get("phase") == "ABORT_CHANGE"
                        mandatory_started |= diag.get("reason") == "MANDATORY_ROUTE" and diag.get("phase") != "KEEP_LANE"
                        if original_target_id is not None:
                            target_preserved &= diag.get("target_id") == original_target_id
                            postcommit_blocked |= diag.get("phase") == "BLOCKED_CHANGE"
                        if target_blocker is not None:
                            stopped_frames = stopped_frames + 1 if ego.speed_km_h < .3 and diag.get("phase") == "BLOCKED_CHANGE" else 0
                            if stopped_frames >= 10:
                                write_json(folder / "postcommit_clearance.json", {"step": step, "speed_kmh": float(ego.speed_km_h)}, exclusive=True)
                                actors.remove(target_blocker)
                                env.engine.clear_objects([target_blocker.id])
                                target_blocker = None
                                blocker_cleared = True
                        veto_rear |= "rear_gap" in diag.get("veto", [])
                        obs, _, terminated, truncated, info = env.step(action)
                        crash |= any(info.get(key, False) for key in ("crash_vehicle", "crash_object", "crash_sidewalk", "crash_building"))
                        out |= bool(info.get("out_of_road"))
                        boundary |= bool(ego.on_yellow_continuous_line or ego.on_white_continuous_line)
                        if spawned:
                            station_now, lateral = original_lane.local_coordinates(ego.position)
                            max_lateral = max(max_lateral, abs(float(lateral)))
                            if leader is not None:
                                passed_leader |= station_now > metadata["leader_station_m"] + (ego.LENGTH + leader.LENGTH) / 2 + 1
                        physical.write(json.dumps({"step": step, "speed_kmh": float(ego.speed_km_h),
                              "position": list(map(float, ego.position)), "heading": float(ego.heading_theta),
                              "crash": crash, "out_of_road": out, "solid_boundary_contact": boundary,
                              "passed_leader": passed_leader, "maneuver_completed": completed}) + "\n")
                        if terminated or truncated or crash or out or boundary:
                            break
                        if spawned and ((label == "safe_overtake" and passed_leader and completed) or
                                        (label == "fast_rear" and step - spawn_step > 90) or
                                        (label in {"mandatory_route", "mandatory_left"} and mandatory_started and completed) or
                                        (label == "precommit_abort" and aborted and completed) or
                                        (label == "postcommit_front_block" and postcommit_blocked and blocker_cleared and completed)):
                            break
                expectation = {"safe_overtake": completed and passed_leader, "fast_rear": veto_rear,
                               "mandatory_route": mandatory_started and completed,
                               "mandatory_left": mandatory_started and completed,
                               "precommit_abort": aborted and completed,
                               "postcommit_front_block": postcommit_blocked and blocker_cleared and target_preserved and completed}[label]
                passed = spawned and not (crash or out or boundary) and expectation
                result = {"scenario": label, "passed": bool(passed), "steps": step + 1,
                          "spawned": spawned, "spawn_step": spawn_step, "phases": sorted(p for p in phases if p),
                          "maneuver_completed": completed, "passed_leader": bool(passed_leader),
                          "mandatory_started": mandatory_started, "safe_abort_observed": aborted,
                          "postcommit_blocked": postcommit_blocked, "blocker_cleared": blocker_cleared,
                          "postcommit_target_preserved": target_preserved,
                          "fast_rear_veto_observed": veto_rear, "crash": crash, "out_of_road": out,
                          "solid_boundary_contact": boundary, "max_abs_source_lane_lateral_m": max_lateral,
                          "note": "Controlled TRAIN-derived traffic/spawn interventions; not a canonical driving outcome"}
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
        write_json(root / "summary.json", results, exclusive=True)
        state["status"] = "COMPLETED" if all(row["passed"] for row in results) else "COMPLETED_WITH_FAILED_EXPECTATIONS"
    except BaseException as exc:
        state["status"] = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
        state["error"] = f"{type(exc).__name__}: {exc}"
        (root / "error.txt").write_text(traceback.format_exc(), encoding="utf8")
        raise
    finally:
        state["finished_at_utc"] = now()
        write_json(root / "run_state.json", state)
    return 0 if state["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())
