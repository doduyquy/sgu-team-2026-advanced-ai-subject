"""TRAIN-only reset-integrity preflight for the P2-F start-distance curriculum."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metadrive.envs.metadrive_env import MetaDriveEnv
from src.launcher.cases import load_locked_geometry_block_sequence
from src.launcher.executor import build_metadrive_case_config
from src.platform import InputProfileId
from src.platform.agent_input import build_agent_input
from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER, task_producer_identity
from src.training.stage5.start_curriculum import apply_route_spawn_plan, resolve_route_spawn_plan
from src.training.stage5.train_sampler import SamplerStrategy, TrainCaseSampler

POOL = (("SCCS", 0), ("SCCS", 2), ("SCCS", 3), ("SCSS", 0), ("SCSS", 3), ("SCSS", 4), ("SCS", 0), ("SCS", 1))
DISTANCES_M = (60.0, 120.0)
TERMINAL_FLAGS = ("arrive_dest", "out_of_road", "crash_vehicle", "crash_object", "crash_building",
                  "crash_human", "crash_sidewalk", "max_step")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict:
    rows: list[dict] = []
    manifest_path = ROOT / "results/audits/evaluation_protocol/geometry_split_manifest.csv"
    manifest_sha = sha256(manifest_path)
    for sequence, generation_seed in POOL:
        sampler = TrainCaseSampler(
            training_root_seed=101, strategy=SamplerStrategy.FIXED, tier_filter="Easy",
            sequence_filter=sequence, fixed_geometry_seed=generation_seed, project_root=ROOT,
        )
        case = sampler.get_case(generation_seed, tier="Easy", sequence=sequence, environment_seed=0)
        geometry_id = sampler.geometry_id_for_case(case)
        if case.split != "TRAIN" or case.traffic_density != 0.0:
            raise RuntimeError(f"Unexpected non-Easy/ non-TRAIN P2-F geometry: {geometry_id}")
        blocks, _, _ = load_locked_geometry_block_sequence(case.to_dict(), ROOT)
        for target_m in DISTANCES_M:
            plan = resolve_route_spawn_plan(
                case=case, blocks=blocks, project_root=ROOT,
                requested_remaining_route_distance_m=target_m, render_mode="OFF",
            )
            cfg = apply_route_spawn_plan(build_metadrive_case_config(case, blocks, "OFF"), plan)
            cfg["log_level"] = logging.ERROR
            env = MetaDriveEnv(cfg)
            try:
                raw, reset_info = env.reset(seed=case.environment_seed)
                nav = env.agent.navigation
                position_error = float(np.linalg.norm(np.asarray(env.agent.position) - np.asarray(plan.spawn_position_xy)))
                lane = nav.current_ref_lanes[0]
                projected_station = float(lane.local_coordinates(env.agent.position)[0])
                remaining = float(nav.total_length - nav.travelled_length)
                task = build_agent_input(
                    raw_obs=raw, ego_vehicle=env.agent, traffic_vehicles=[], navigation=nav,
                    road_network=env.current_map.road_network, step_index=0,
                    profile_id=InputProfileId.STATE_DECISION_V1,
                    task_context_producer_id=REFERENCE_TASK_PRODUCER,
                ).task_context
                flags = {name: bool(reset_info.get(name, False)) for name in TERMINAL_FLAGS}
                checkpoint_match = tuple(str(x) for x in nav.checkpoints) == plan.expected_checkpoint_suffix
                destination_match = str(nav.checkpoints[-1]) == plan.original_destination
                lane_match = tuple(lane.index) == plan.reference_lane_index
                checks = {
                    "train_case": case.split == "TRAIN",
                    "checkpoint_suffix_preserved": checkpoint_match,
                    "original_destination_preserved": destination_match,
                    "reference_lane_preserved": lane_match,
                    "position_error_le_0_05m": position_error <= 0.05,
                    "station_error_le_0_05m": abs(projected_station - plan.reference_lane_local_station_m) <= 0.05,
                    "remaining_distance_error_le_0_10m": abs(remaining - target_m) <= 0.10,
                    "f09_waypoints_valid": int(task.validity_mask.sum()) >= 19,
                    "observation_finite": bool(np.isfinite(raw).all() and np.isfinite(task.waypoints_array).all()),
                    "zero_initial_speed": float(env.agent.speed_km_h) == 0.0,
                    "no_initial_terminal_or_safety_flags": not any(flags.values()),
                    "case_horizon_unchanged": int(cfg["horizon"]) == int(case.horizon_steps),
                }
                row = {
                    "geometry_id": geometry_id, "geometry_sha256": case.geometry_sha256,
                    "sequence": sequence, "geometry_generation_seed": generation_seed,
                    "environment_seed": case.environment_seed, "split": case.split,
                    "requested_remaining_route_distance_m": target_m,
                    "measured_remaining_route_distance_m": remaining,
                    "reference_station_m": plan.reference_station_m,
                    "reference_lane_local_station_m": plan.reference_lane_local_station_m,
                    "position_error_m": position_error,
                    "reference_station_error_m": abs(projected_station - plan.reference_lane_local_station_m),
                    "route_completion_reset": float(nav.route_completion),
                    "meta_route_total_length_m": float(nav.total_length),
                    "case_horizon_steps": case.horizon_steps,
                    "original_destination": plan.original_destination,
                    "original_checkpoints": list(plan.original_checkpoints),
                    "actual_checkpoints": [str(x) for x in nav.checkpoints],
                    "valid_f09_waypoints": int(task.validity_mask.sum()),
                    "speed_kmh": float(env.agent.speed_km_h), "initial_flags": flags,
                    "checks": checks, "pass": all(checks.values()),
                }
                rows.append(row)
            finally:
                env.close()

    result = {
        "status": "PASS" if len(rows) == len(POOL) * len(DISTANCES_M) and all(r["pass"] for r in rows) else "FAIL",
        "scope": "TRAIN/PILOT_TRAIN only; simulator reset feasibility, not policy competence",
        "meta_drive_version": "0.4.3", "environment_seed": 0,
        "producer": task_producer_identity(REFERENCE_TASK_PRODUCER),
        "geometry_manifest_sha256": manifest_sha, "rows": rows,
        "full_route_horizon_preserved": True,
        "reward_spec_changed": False, "validation_test_accessed": False,
    }
    out = ROOT / "results/stage5/p2f_curriculum_preflight_20261121"
    out.mkdir(parents=True, exist_ok=True)
    evidence = out / "reset_integrity.json"
    evidence.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (out / "reset_integrity.sha256").write_text(f"{sha256(evidence)}  reset_integrity.json\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps({k: v for k, v in run().items() if k != "rows"} | {
        "rows": len(json.loads((ROOT / "results/stage5/p2f_curriculum_preflight_20261121/reset_integrity.json").read_text())["rows"]),
        "all_cases_pass": all(x["pass"] for x in json.loads((ROOT / "results/stage5/p2f_curriculum_preflight_20261121/reset_integrity.json").read_text())["rows"]),
    }, indent=2, allow_nan=False))
