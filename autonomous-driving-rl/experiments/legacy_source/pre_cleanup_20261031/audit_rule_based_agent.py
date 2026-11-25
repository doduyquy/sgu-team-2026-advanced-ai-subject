"""Headless TRAIN-only driving check; simulator telemetry stays outside the agent."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metadrive import MetaDriveEnv
from src.agents.rule_based_agent import Stage1RuleBasedAgent
from src.launcher.cases import load_locked_geometry_block_sequence, resolve_sandbox_case
from src.launcher.executor import build_metadrive_case_config
from src.launcher.models import LauncherMode, LaunchRequestV1
from src.platform.agent import AgentPublicEpisodeContext
from src.platform.agent_input import build_agent_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="Easy")
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--geom-seed", type=int, default=None)
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cruise-speed-kmh", type=float, default=45.0,
                        help="Open-road target cap; other speed constraints still apply")
    parser.add_argument("--steering-probe", type=float, default=None)
    parser.add_argument("--trace", type=Path, default=None)
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if args.steering_probe is not None and not -1.0 <= args.steering_probe <= 1.0:
        parser.error("--steering-probe must be in [-1, 1]")
    if not math.isfinite(args.cruise_speed_kmh) or not 0 < args.cruise_speed_kmh <= 80:
        parser.error("--cruise-speed-kmh must be finite and in (0, 80]")
    case = resolve_sandbox_case(LaunchRequestV1(
        mode=LauncherMode.SANDBOX, agent_id="stage1_rule_based",
        tier=args.tier, sequence=args.sequence,
        geometry_generation_seed=args.geom_seed, environment_seed=args.env_seed,
    ))
    blocks, _, _ = load_locked_geometry_block_sequence(case.to_dict())
    env = MetaDriveEnv(build_metadrive_case_config(case, blocks))
    agent = Stage1RuleBasedAgent(cruise_speed_kmh=args.cruise_speed_kmh)
    rows = []
    yellow_contact = False
    white_contact = False
    try:
        obs, info = env.reset(seed=case.environment_seed)
        if (env.config["random_agent_model"] or
                env.agent.config["side_detector"]["num_lasers"] != 0 or
                env.agent.config["lane_line_detector"]["num_lasers"] != 0 or
                env.agent.max_speed_km_h != agent.observation_max_speed_kmh or
                env.current_map.MAX_LANE_WIDTH != agent.observation_max_lane_width_m or
                (env.current_map.MAX_LANE_NUM + 1) * env.current_map.MAX_LANE_WIDTH != agent.observation_road_scale_m or
                abs(env.agent.WIDTH - agent.vehicle_width_m) > 1e-6 or
                abs(env.agent.LENGTH - agent.vehicle_length_m) > 1e-6):
            raise ValueError("Simulator configuration differs from agent observation/vehicle assumptions")
        agent.reset(AgentPublicEpisodeContext(action_adapter_id="continuous_box2_v1"))
        heading_start = float(env.agent.heading_theta)
        print(json.dumps({"case": case.case_id, "max_speed_kmh": env.agent.max_speed_km_h,
                          "agent_version": agent.descriptor.agent_version,
                          "cruise_speed_kmh": agent.cruise_speed_kmh,
                          "width_m": env.agent.WIDTH, "length_m": env.agent.LENGTH,
                          "steering_probe": args.steering_probe}), flush=True)
        limit = args.steps or case.horizon_steps
        for step in range(limit):
            traffic = list(getattr(env.engine.traffic_manager, "traffic_vehicles", []))
            inp = build_agent_input(obs, env.agent, traffic, env.agent.navigation,
                                    env.current_map.road_network, step)
            if args.steering_probe is None:
                decision = agent.act(inp)
                action = decision.action_payload
                diagnostics = decision.diagnostics or {}
            else:
                action = [args.steering_probe, 0.4]
                diagnostics = {}
            rows.append({"step": step, "speed_kmh": float(env.agent.speed_km_h),
                         "lane_offset_m": float(env.agent.lane.local_coordinates(env.agent.position)[1]),
                         "obs_heading": float(obs[2]), "obs_offset": float(obs[8]),
                         "waypoint_x": float(inp.task_context.waypoints[0].relative_x),
                         "waypoint_y": float(inp.task_context.waypoints[0].relative_y),
                         "steering": float(action[0]), "throttle": float(action[1]),
                         "diagnostics": json.dumps(diagnostics)})
            obs, _, terminated, truncated, info = env.step(action)
            yellow_contact = yellow_contact or bool(env.agent.on_yellow_continuous_line)
            white_contact = white_contact or bool(env.agent.on_white_continuous_line)
            if terminated or truncated:
                break
        delta = math.atan2(math.sin(env.agent.heading_theta - heading_start),
                           math.cos(env.agent.heading_theta - heading_start))
        summary = {
            "case": case.case_id, "steps": len(rows),
            "agent_version": agent.descriptor.agent_version,
            "cruise_speed_kmh": agent.cruise_speed_kmh,
            "simulation_duration_s": len(rows) * 0.1,
            "mean_speed_kmh": sum(row["speed_kmh"] for row in rows) / len(rows),
            "arrive_dest": bool(info.get("arrive_dest", False)),
            "out_of_road": bool(info.get("out_of_road", False)),
            "yellow_contact": yellow_contact,
            "white_contact": white_contact,
            "route_completion": float(info.get("route_completion", 0.0)),
            "delta_heading_rad": float(delta),
            "max_abs_lane_offset_m": max(abs(row["lane_offset_m"]) for row in rows),
            "max_speed_kmh": max(row["speed_kmh"] for row in rows),
            "crash": bool(info.get("crash", False)),
            "terminated": bool(terminated), "truncated": bool(truncated),
        }
        print(json.dumps(summary), flush=True)
        if args.trace:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            with args.trace.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    finally:
        agent.close()
        env.close()


if __name__ == "__main__":
    main()
