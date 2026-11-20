"""TRAIN-only P0 simulator audit. Replays a recorded clean reference for measurement."""
import argparse
import gzip
import hashlib
import json
import math
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.train_sampler import TrainCaseSampler
import numpy as np

GEOMETRY = "geom_easy_SCS_seed0"
HISTORY = "s5_5_meter_reward_seed101_20261120"
RUN = ROOT / "runs/stage5_meter_reward" / HISTORY / "preflight"
RESULT = ROOT / "results/stage5" / HISTORY


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def sample(env, obs, agent_input, step):
    v = env._env.agent
    n = v.navigation
    idx = int(n._target_checkpoints_index[0])
    route = list(zip(n.checkpoints[idx:-1], n.checkpoints[idx + 1:]))
    lanes = [n.map.road_network.graph[u][w] for u, w in route]
    lane = n.current_lane
    s, lateral = lane.local_coordinates(v.position)
    heading = float(v.heading_theta)
    pos = np.asarray(v.position[:2])
    cs, sn = math.cos(heading), math.sin(heading)
    task = agent_input.task_context
    points = []
    for i, wp in enumerate(task.waypoints):
        if not task.validity_mask[i]:
            continue
        x, y, h = wp.to_tuple()
        world = pos + [x * cs - y * sn, x * sn + y * cs]
        remaining = s + (i + 1) * task.lookahead_spacing_m
        for group in lanes:
            selected = group[0]
            if remaining <= selected.length:
                reference = selected.position(remaining, 0)
                points.append({
                    "world": world.tolist(), "heading": wrap(h + heading),
                    "frame_error_m": float(np.linalg.norm(world - reference)),
                    "heading_error_rad": abs(wrap(h + heading - selected.heading_theta_at(remaining))),
                    "lane": list(selected.index), "along_lane_m": float(remaining),
                })
                break
            remaining -= selected.length
    seams = [
        {"gap_m": float(np.linalg.norm(g[0].position(g[0].length, 0) - j[0].position(0, 0))),
         "tangent_rad": abs(wrap(g[0].heading_theta_at(g[0].length) - j[0].heading_theta_at(0)))}
        for g, j in zip(lanes, lanes[1:])
    ]
    lane0_gap = None
    station_delta = None
    if lanes and lane in lanes[0] and lane is not lanes[0][0]:
        lane0_gap = float(np.linalg.norm(lane.position(s, 0) - lanes[0][0].position(s, 0)))
        reference_s, _ = lanes[0][0].local_coordinates(v.position)
        station_delta = float(s - reference_s)
    traffic = agent_input.traffic_context
    return {
        "step": step, "ego": obs[:9].tolist(), "navigation": obs[9:19].tolist(),
        "navigation_error": float(np.max(np.abs(obs[9:19] - n.get_navi_info()))),
        "lidar_min": float(obs[19:259].min()), "lidar_max": float(obs[19:259].max()),
        "traffic_valid": int(obs[315:323].sum()), "traffic_nonzero": int(np.count_nonzero(obs[259:325])),
        "traffic_padding_zero": bool(np.all(obs[259:315].reshape(8, 7)[~traffic.validity_mask] == 0)),
        "traffic_distances_m": np.linalg.norm(traffic.actors_array[traffic.validity_mask, :2], axis=1).tolist(),
        "task_padding_zero": bool(np.all(obs[325:385].reshape(20, 3)[~task.validity_mask] == 0)),
        "valid_waypoints": int(task.validity_mask.sum()), "route_end": task.route_end_within_lookahead,
        "heading": heading, "speed": float(v.speed_km_h), "lane": list(lane.index),
        "lane_type": type(lane).__name__, "lateral": float(lateral), "lane_count": len(lanes[0]) if lanes else 0,
        "lane0_gap_m": lane0_gap, "checkpoints": list(n._target_checkpoints_index),
        "current_minus_reference_station_m": station_delta,
        "task": obs[325:].tolist(), "points": points, "seams": seams,
        "last_action": np.asarray(v.last_current_action[1]).tolist(),
        "speed_encoding_error": abs(float(obs[3]) - np.clip((v.speed_km_h+1)/(v.max_speed_km_h+1), 0, 1)),
        "previous_action_error": float(np.max(np.abs(obs[5:7] - (np.asarray(v.last_current_action[1])+1)/2))),
        "finite_409": bool(obs.shape == (409,) and np.isfinite(obs).all()),
    }


def rollout(actions, *, offset_lane=False, geometry=GEOMETRY, diagnostic_follower=False):
    sampler = TrainCaseSampler(allowed_geometry_ids=[geometry])
    case = sampler.get_case_for_geometry_id(geometry, environment_seed=0)
    env = Stage5TrainingEnv(sampler=sampler)
    latest = []
    original = env._build_agent_input

    def capture(raw, step_idx):
        result = original(raw, step_idx)
        latest[:] = [result]
        return result

    env._build_agent_input = capture
    try:
        obs, reset = env.reset(options={"case": case})
        md = env._env
        v = md.agent
        config = md.config
        received = []
        calls = []
        old_step = md.step
        old_physics = md.engine.step_physics_world

        def counted_step(action):
            received.append(np.array(action))
            return old_step(action)

        def counted_physics(*args, **kwargs):
            calls.append(1)
            return old_physics(*args, **kwargs)

        md.step = counted_step
        md.engine.step_physics_world = counted_physics
        if offset_lane:
            n = v.navigation
            other = next(lane for lane in n.current_ref_lanes if lane.index[-1] != 0)
            along = 20 if offset_lane == "skip_segment" else n.current_lane.local_coordinates(v.position)[0]
            v.set_position(other.position(along, 0))
            v.set_heading_theta(other.heading_theta_at(along))
        rows = [sample(env, obs, latest[0], 0)] if not offset_lane else []
        max_error = 0.0
        for i, action in enumerate(actions, 1):
            if diagnostic_follower:
                # Low-speed fixture for collecting physical geometry, never SAC evaluation.
                lane = v.navigation.current_lane
                along, lateral = lane.local_coordinates(v.position)
                heading_error = wrap(lane.heading_theta_at(min(along + 3, lane.length)) - v.heading_theta)
                steer = np.clip(heading_error - .15 * lateral, -.5, .5)
                drive = np.clip(.1 * (16 - v.speed_km_h), -.35, .45)
                action = [float(steer), float(drive)]
            obs, reward, term, trunc, info = env.step(action)
            max_error = max(max_error, float(np.max(np.abs(received[-1] - action))),
                            abs(info["action_steering"] - action[0]),
                            abs(info["action_throttle_brake"] - action[1]))
            row = sample(env, obs, latest[0], i)
            row.update(action=list(action), reward=reward, progress=info["route_completion"],
                       outcome=info.get("outcome"), terminated=term, truncated=trunc)
            rows.append(row)
            if term or trunc:
                break
        return {"geometry": geometry, "case_id": case.case_id,
                "initial_progress": reset["initial_route_completion"], "rows": rows,
                "decision_repeat": config["decision_repeat"], "physics_dt": config["physics_world_step_size"],
                "decisions": len(received), "physics_calls": len(calls), "action_max_error": max_error}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    reference = RUN / "stage1_reference_decisions.jsonl.gz"
    with gzip.open(reference, "rt", encoding="utf-8") as stream:
        actions = [json.loads(line)["canonical_action"] for line in stream]
    preflight = json.loads((RESULT / "reward_preflight.json").read_text(encoding="utf-8"))
    raw = RUN / "reward_traces.jsonl.gz"
    if digest(raw) != preflight["raw_trace_sha256"]:
        raise ValueError("historical reward trace hash mismatch")
    episodes = {x["trace_id"]: x for x in preflight["trace_summaries"]}
    clean = episodes["stage1_reference_clean"]
    gamma = .99
    horizon = clean["horizon_steps"]
    length = clean["effective_route_length_m"]
    prefix_returns = {}
    with gzip.open(raw, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            trace = row["trace_id"]
            prefix_returns.setdefault(trace, []).append(row["reward_v1_env"])
    credit = {}
    for trace, rewards in prefix_returns.items():
        end = min(clean["first_curve_step_from_public_task_waypoints"], len(rewards)-1)
        start = max(0, end-50)
        credit[trace] = {"window_start": start, "window_end_exclusive": end,
                         "window_return_v1": sum(rewards[start:end]),
                         "window_start_discounted_tail": sum(gamma**i*r for i,r in enumerate(rewards[start:]))}
    data = {
        "base_source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": digest(Path(__file__)),
        "metadrive": version("metadrive-simulator"),
        "metadrive_source": subprocess.check_output(
            ["git", "-C", str(Path(sys.modules["metadrive"].__file__).parent.parent), "rev-parse", "HEAD"],
            text=True).strip(),
        "sb3_distribution": version("stable-baselines3"),
        "reference_sha256": digest(reference), "reward_trace_sha256": digest(raw),
        "reward": {"horizon": horizon, "length_m": length, "time_cost": .25/horizon,
                   "positive_step_threshold_ideal_kmh": 3.6 * length*.25 / (.1*horizon),
                   "gamma_power_horizon": gamma**horizon,
                   "success_bonus_discount_at_start": gamma**(clean["steps"]-1),
                   "episodes": episodes, "prefix_and_tail_credit": credit,
                   "matched_clean_fast_slow": "MISSING"},
        "pulses": {},
    }
    for steer in (-.6, -.3, -.1, 0, .1, .3, .6):
        data["pulses"][str(steer)] = rollout([[0, .45]]*14 + [[steer, 0]]*4 + [[0, -.45]]*4)
    data["adjacent_lane_probe"] = rollout([[0, .45]]*45, offset_lane=True)
    data["localization_stress_probe"] = rollout([[0, 0]]*3, offset_lane="skip_segment")
    data["traffic_probe"] = rollout([[0, .65]]*65, geometry="geom_medium_SCTCS_seed11")
    data["adjacent_lane_curve"] = rollout([[0, 0]]*900, offset_lane=True, diagnostic_follower=True)
    data["historical_clean_replay"] = rollout(actions)
    payload = json.dumps(data, allow_nan=False, separators=(",", ":"), default=encode_scalar)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def encode_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unexpected audit value: {type(value).__name__}")


if __name__ == "__main__":
    main()
