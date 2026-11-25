"""Replay frozen policy/helpers against every recorded public input, without simulator.

This verifies decision/state reproducibility, not a new closed-loop driving score.
The shared AgentDecision API/adapter stays live; its current hashes are reported.
"""

import argparse
import dataclasses
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.logging.agent_experiment import sha256_file, write_json
from src.platform.agent import AgentPublicEpisodeContext
from src.platform.action_adapter import ContinuousBox2Adapter


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def restore_input(data, module):
    fields = {field.name for field in dataclasses.fields(module.AgentInputV1)}
    if set(data) - fields:
        raise ValueError("Unexpected fields in public input payload")
    core_data, traffic_data, task_data = data["core_observation"], data["traffic_context"], data["task_context"]
    core = module.CoreObservationV1(np.array(core_data["features"], dtype=core_data["dtype"]))
    actors = tuple(module.TrafficActorV1(**actor) for actor in traffic_data["actors"])
    extra_traffic = {}
    if hasattr(module, "ActorPathV1"):
        extra_traffic = {"motion_paths": tuple(module.ActorPathV1(path["actor_slot"], path["branch_id"], path["points"])
                                               for path in traffic_data.get("motion_paths", ())),
                         "motion_paths_overflow_count": traffic_data.get("motion_paths_overflow_count", 0)}
    traffic = module.TrafficContextV1(
        capacity=traffic_data["capacity"], radius_m=traffic_data["radius_m"], actors=actors,
        actors_array=np.array([actor.to_tuple() for actor in actors], dtype=np.float32),
        validity_mask=np.array(traffic_data["validity_mask"], dtype=bool),
        active_count=traffic_data["active_count"], overflow_count=traffic_data["overflow_count"], **extra_traffic)
    waypoints = tuple(module.RouteWaypointV1(**point) for point in task_data["waypoints"])
    extra_task = {}
    if hasattr(module, "LaneContextV1"):
        lane_data = task_data.get("lane_context")
        extra_task["lane_context"] = module.LaneContextV1(
            tuple(module.LaneGeometryV1(**lane) for lane in lane_data["lanes"]),
            lane_data["required_side"]) if lane_data else None
    if hasattr(module, "RoutePassageV1"):
        extra_task["route_passage"] = module.RoutePassageV1(**task_data["route_passage"]) if task_data.get("route_passage") else None
    task = module.TaskContextV1(
        lookahead_count=task_data["lookahead_count"], lookahead_spacing_m=task_data["lookahead_spacing_m"],
        lookahead_range_m=task_data["lookahead_range_m"], current_lane_width=task_data["current_lane_width"],
        navigation_goal_direction=tuple(task_data["navigation_goal_direction"]), waypoints=waypoints,
        waypoints_array=np.array([point.to_tuple() for point in waypoints], dtype=np.float32),
        validity_mask=np.array(task_data["validity_mask"], dtype=bool),
        route_end_within_lookahead=task_data["route_end_within_lookahead"],
        decision_point=module.RoadDecisionPointV1(**task_data["decision_point"]) if task_data.get("decision_point") else None,
        **extra_task)
    extra_input = {}
    if hasattr(module, "EgoMotionV1"):
        extra_input["ego_motion"] = module.EgoMotionV1(**data["ego_motion"]) if data.get("ego_motion") else None
    return module.AgentInputV1(module.InputProfileId(data["profile_id"]), core, traffic, task, data["step_index"], **extra_input)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-dir", type=Path, required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--episode-index", type=int)
    selector.add_argument("--controlled-scenario", action="store_true", help="variant-dir is a physical-probe scenario folder")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    state = json.loads((args.variant_dir.parent / "run_state.json").read_text(encoding="utf8"))
    if state["status"] != "COMPLETED":
        raise ValueError("Only completed driving runs may be verified as complete replays")
    provenance = json.loads((args.variant_dir / "agent_provenance.json").read_text(encoding="utf8"))
    sources = args.variant_dir / "sources"
    for name, expected in provenance["source_sha256"].items():
        if sha256_file(sources / name) != expected:
            raise ValueError(f"Modified source snapshot: {name}")
    config_hash = hashlib.sha256(json.dumps(provenance["configuration"], sort_keys=True,
                               separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if config_hash != provenance["configuration_sha256"]:
        raise ValueError("Modified configuration")
    adapter = ROOT / "src/platform/action_adapter.py"
    if sha256_file(adapter) != provenance["source_sha256"]["action_adapter.py"]:
        raise ValueError("Live adapter differs from recorded source")
    package_name = "_rb_replay_" + uuid.uuid4().hex
    package = types.ModuleType(package_name)
    package.__path__ = [str(sources / "agents" if (sources / "agents").is_dir() else sources)]
    sys.modules[package_name] = package
    policy = load_module(package_name + ".policy", sources / "policy.py")
    inputs = load_module(package_name + ".inputs", sources / "agent_input.py")
    agent = policy.Stage1RuleBasedAgent()
    for name, value in provenance["configuration"].items():
        if name not in agent.export_config():
            raise ValueError(f"Unknown archived parameter: {name}")
        setattr(agent, name, value)
    episode = args.variant_dir if args.controlled_scenario else args.variant_dir / f"episode_{args.episode_index:03d}"
    environment = json.loads((episode / "environment.json").read_text(encoding="utf8"))
    result = json.loads((episode / "result.json").read_text(encoding="utf8"))
    case_id = environment["case"]["case_id"] + "/" + result["scenario"] if args.controlled_scenario else result["case_id"]
    trace = episode / "decisions.jsonl.gz"
    report = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "variant_dir": str(args.variant_dir),
              "case_id": case_id, "controlled_scenario": args.controlled_scenario, "configuration_sha256": config_hash,
              "source_sha256": provenance["source_sha256"], "trace_sha256": sha256_file(trace),
              "shared_platform_api_sha256": sha256_file(ROOT / "src/platform/agent.py"),
              "adapter_sha256": sha256_file(adapter), "excluded_from_comparison": ["act_latency_ms"],
              "scope": "Frozen policy/helpers/input types; live unchanged shared AgentDecision API and verified adapter",
              "completed_replay": False, "steps": 0, "decision_mismatches": [], "action_mismatches": []}
    try:
        agent.reset(AgentPublicEpisodeContext(horizon_steps=environment["case"]["horizon_steps"]))
        with gzip.open(trace, "rt", encoding="utf8") as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                if row["step_index"] != index or row["case_id"] != case_id:
                    raise ValueError("Wrong case or non-contiguous trace")
                inp = restore_input(row["agent_input"], inputs)
                if inp.to_dict() != row["agent_input"]:
                    raise ValueError(f"Input round-trip mismatch at step {index}")
                decision = agent.act(inp)
                if decision.to_dict() != row["decision"]:
                    report["decision_mismatches"].append(index)
                action = ContinuousBox2Adapter().to_canonical(decision.action_payload).to_numpy().tolist()
                if action != row["canonical_action"]:
                    report["action_mismatches"].append(index)
                report["steps"] += 1
        if report["steps"] != result["steps"]:
            raise ValueError("Recorded step count differs from episode result")
        report["completed_replay"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        agent.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report, exclusive=True)
    passed = not report["decision_mismatches"] and not report["action_mismatches"]
    print(json.dumps({"passed": passed, "steps": report["steps"], "output": str(args.output)}))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
