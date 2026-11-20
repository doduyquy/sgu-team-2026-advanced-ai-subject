"""Replay F09 on TRAIN and evaluate absolute reference-route arc length."""
import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_stage5_task_alignment as p0

POSITION_TOL = 0.00015
HEADING_TOL = 0.00008


def capture(env, obs, agent_input, step, candidate):
    vehicle = env._env.agent
    nav = vehicle.navigation
    lane = nav.current_lane
    pos = np.asarray(vehicle.position[:2])
    heading = vehicle.heading_theta
    edges = list(zip(nav.checkpoints, nav.checkpoints[1:]))
    checkpoint = int(nav._target_checkpoints_index[0])
    nearby = range(max(0, checkpoint - 1), min(len(edges), checkpoint + 2))
    matches = [i for i in nearby if edges[i] == tuple(lane.index[:2])]
    index = matches[0] if len(matches) == 1 else checkpoint
    refs = [nav.map.road_network.graph[u][v][0] for u, v in edges]
    starts = np.r_[0., np.cumsum([ref.length for ref in refs])]
    ref_station = float(refs[index].local_coordinates(pos)[0])
    absolute = starts[index] + ref_station
    before = agent_input.task_context
    after = candidate(nav, vehicle, nav.map.road_network, before) if candidate else before

    def measure(task):
        points = []
        for offset, wp in enumerate(task.waypoints, 1):
            target = absolute + 2.5 * offset
            valid = bool(task.validity_mask[offset - 1])
            expected_valid = 0 <= target <= starts[-1]
            row = {"valid": valid, "expected_valid": bool(expected_valid), "target_station_m": target,
                   "ego_relative": list(wp.to_tuple())}
            if valid and expected_valid:
                segment = min(len(refs) - 1, int(np.searchsorted(starts, target, side="left")) - 1)
                segment = max(0, segment)
                ref = refs[segment]
                x, y, yaw = wp.to_tuple()
                world = pos + [x * np.cos(heading) - y * np.sin(heading),
                               x * np.sin(heading) + y * np.cos(heading)]
                local = target - starts[segment]
                world_heading = p0.wrap(yaw + heading)
                row.update(world=world.tolist(), reference_lane=list(ref.index),
                           expected_world=ref.position(local, 0).tolist(),
                           station_error_m=float(ref.local_coordinates(world)[0] - local),
                           world_error_m=float(np.linalg.norm(world - ref.position(local, 0))),
                           heading_error_rad=abs(p0.wrap(world_heading - ref.heading_theta_at(local))))
            points.append(row)
        return points

    from dataclasses import replace
    encoded = env.encoder.encode(replace(agent_input, task_context=after))
    assert encoded.shape == (409,) and np.array_equal(obs[:325], encoded[:325])
    assert np.array_equal(obs[405:408], encoded[405:408])
    return {"step": step, "ego_position": pos.tolist(), "ego_heading": heading,
            "current_lane": list(lane.index), "reference_lane": list(refs[index].index),
            "checkpoint_indices": list(nav._target_checkpoints_index), "lane_type": type(lane).__name__,
            "current_lane_station_m": float(lane.local_coordinates(pos)[0]),
            "reference_projected_station_m": ref_station,
            "station_delta_m": float(lane.local_coordinates(pos)[0]) - ref_station,
            "absolute_reference_station_m": absolute, "before": measure(before), "after": measure(after),
            "same_task_84d": bool(np.array_equal(obs[325:], encoded[325:]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("legacy", "proposal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    candidate = None
    if args.mode == "proposal":
        from src.training.stage5.reference_station_proposal import corrected_task_context
        candidate = corrected_task_context
    with gzip.open(p0.RUN / "stage1_reference_decisions.jsonl.gz", "rt", encoding="utf-8") as stream:
        actions = [json.loads(line)["canonical_action"] for line in stream]
    with patch.object(p0, "sample", side_effect=lambda *x: capture(*x, candidate)):
        runs = {
            "lane_zero_clean": p0.rollout(actions),
            "adjacent_straight": p0.rollout([[0, .45]] * 45, offset_lane=True),
            "adjacent_curve": p0.rollout([[0, 0]] * 900, offset_lane=True, diagnostic_follower=True),
            "delayed_checkpoint": p0.rollout([[0, 0]] * 3, offset_lane="skip_segment"),
        }
    witness = next(r for r in runs["adjacent_curve"]["rows"] if r["step"] == 614)
    assert witness["current_lane"][-1] == 1 and abs(witness["station_delta_m"] + 7.684567) < .001
    violations = {}
    for name, run in runs.items():
        assert run["physics_calls"] == 5 * run["decisions"] and run["action_max_error"] < 1e-7
        violations[name] = int(sum(
            point["valid"] != point["expected_valid"]
            or (not point["valid"] and any(point["ego_relative"]))
            or point.get("world_error_m", 0) > POSITION_TOL
            or abs(point.get("station_error_m", 0)) > POSITION_TOL
            or point.get("heading_error_rad", 0) > HEADING_TOL
            for row in run["rows"] for point in row["after"]
        ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"mode": args.mode, "source": subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(),
                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "runs": runs, "violations": violations}, stream, default=p0.encode_scalar, allow_nan=False)
    print(json.dumps(violations))
    if any(violations.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
