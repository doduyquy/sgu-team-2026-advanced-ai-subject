"""Training-only, versioned start-state curricula for Stage 5 experiments."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from src.launcher.executor import build_metadrive_case_config
from src.launcher.models import ResolvedCaseV1


CONTROL_TASK_ID = "p2f_official_start_control_v1"
CURRICULUM_TASK_ID = "p2f_remaining_route_distance_v1"
CURRICULUM_CONFIG = {
    "id": CURRICULUM_TASK_ID,
    "episode_index_ranges": [
        {"start": 0, "stop_exclusive": 3, "remaining_route_distance_m": 60.0},
        {"start": 3, "stop_exclusive": 6, "remaining_route_distance_m": 120.0},
        {"start": 6, "stop_exclusive": None, "remaining_route_distance_m": None},
    ],
    "schedule_basis": "training_episode_index_at_reset",
    "velocity_at_spawn_m_s": 0.0,
    "destination": "original_MetaDrive_destination",
    "horizon": "unchanged_case_horizon_steps",
    "route_completion": "MetaDrive_route_completion_from_curriculum_spawn; reset baseline captured before first reward delta",
    "training_only": True,
}
CONTROL_CONFIG = {
    "id": CONTROL_TASK_ID,
    "start": "official_case_spawn",
    "velocity_at_spawn_m_s": 0.0,
    "horizon": "unchanged_case_horizon_steps",
    "training_only": True,
}


def _fingerprint(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


CONTROL_FINGERPRINT = _fingerprint(CONTROL_CONFIG)
CURRICULUM_FINGERPRINT = _fingerprint(CURRICULUM_CONFIG)


@dataclass(frozen=True)
class CurriculumStage:
    stage_id: str
    requested_remaining_route_distance_m: float | None


def curriculum_stage(episode_index: int) -> CurriculumStage:
    """Return the pre-registered stage; selection never reads evaluation results."""
    episode_index = int(episode_index)
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    if episode_index < 3:
        return CurriculumStage("near_60m", 60.0)
    if episode_index < 6:
        return CurriculumStage("mid_120m", 120.0)
    return CurriculumStage("full_route", None)


@dataclass(frozen=True)
class RouteSpawnPlan:
    geometry_id: str
    environment_seed: int
    requested_remaining_route_distance_m: float
    effective_remaining_route_distance_m: float
    original_destination: str
    original_checkpoints: tuple[str, ...]
    expected_checkpoint_suffix: tuple[str, ...]
    reference_lane_index: tuple[Any, ...]
    spawn_position_xy: tuple[float, float]
    spawn_heading_rad: float
    reference_station_m: float
    reference_lane_local_station_m: float
    route_edge_index: int
    reference_route_total_length_m: float
    original_reset_route_completion: float
    source: str = "pinned_MetaDrive_route_reference_centerline"


def resolve_route_spawn_plan(
    *,
    case: ResolvedCaseV1,
    blocks: list[Any],
    project_root,
    requested_remaining_route_distance_m: float,
    render_mode: str = "OFF",
) -> RouteSpawnPlan:
    """Resolve a physical spawn by replaying the authentic TRAIN case route.

    The probe uses the same map, environment seed, default spawn and MetaDrive
    destination selection as the official case. The resulting point lies on
    that selected reference route, and the final destination is pinned when the
    curriculum environment is subsequently constructed.
    """
    if str(case.split).upper() != "TRAIN":
        raise ValueError("Training-only start curriculum accepts TRAIN cases only")
    target = float(requested_remaining_route_distance_m)
    if not np.isfinite(target) or target <= 0:
        raise ValueError("requested_remaining_route_distance_m must be finite and positive")

    from metadrive.envs.metadrive_env import MetaDriveEnv

    probe = MetaDriveEnv(build_metadrive_case_config(case, blocks, render_mode=render_mode))
    try:
        probe.reset(seed=int(case.environment_seed))
        navigation = probe.agent.navigation
        checkpoints = list(navigation.checkpoints)
        if len(checkpoints) < 2:
            raise RuntimeError("MetaDrive produced fewer than two route checkpoints")
        destination = str(checkpoints[-1])
        total_length = float(navigation.total_length)
        start_station = float(navigation.travelled_length)
        if not np.isfinite(total_length) or not np.isfinite(start_station) or total_length <= 0:
            raise RuntimeError("MetaDrive produced invalid route station/length")
        if target >= total_length - start_station - 5.0:
            raise ValueError(
                f"Requested remaining distance {target:.3f}m is not ahead of the official spawn "
                f"with a 5m margin (available={total_length - start_station:.3f}m)"
            )

        edge_lanes = [
            probe.current_map.road_network.graph[u][v][0]
            for u, v in zip(checkpoints[:-1], checkpoints[1:])
        ]
        route_station = total_length - target
        traversed = 0.0
        chosen = None
        for edge_index, lane in enumerate(edge_lanes):
            edge_length = float(lane.length)
            if traversed - 1e-6 <= route_station <= traversed + edge_length + 1e-6:
                local_station = float(np.clip(route_station - traversed, 0.0, edge_length))
                position = lane.position(local_station, 0.0)
                heading = float(lane.heading_theta_at(local_station))
                chosen = (edge_index, lane, local_station, position, heading)
                break
            traversed += edge_length
        if chosen is None:
            raise RuntimeError("Requested start station could not be located on the designated route")

        edge_index, lane, local_station, position, heading = chosen
        expected_suffix = tuple(checkpoints[edge_index:])
        return RouteSpawnPlan(
            geometry_id=str(case.case_id),
            environment_seed=int(case.environment_seed),
            requested_remaining_route_distance_m=target,
            effective_remaining_route_distance_m=target,
            original_destination=destination,
            original_checkpoints=tuple(str(value) for value in checkpoints),
            expected_checkpoint_suffix=tuple(str(value) for value in expected_suffix),
            reference_lane_index=tuple(lane.index),
            spawn_position_xy=(float(position[0]), float(position[1])),
            spawn_heading_rad=heading,
            reference_station_m=float(route_station),
            reference_lane_local_station_m=float(local_station),
            route_edge_index=int(edge_index),
            reference_route_total_length_m=total_length,
            original_reset_route_completion=float(navigation.route_completion),
        )
    finally:
        probe.close()


def apply_route_spawn_plan(config: dict[str, Any], plan: RouteSpawnPlan) -> dict[str, Any]:
    """Return a copied MetaDrive config with the exact route destination pinned."""
    result = dict(config)
    vehicle_config = dict(result.get("vehicle_config", {}))
    vehicle_config.update({
        "spawn_position_heading": (plan.spawn_position_xy, float(plan.spawn_heading_rad)),
        "spawn_lane_index": plan.reference_lane_index,
        "destination": plan.original_destination,
        "spawn_velocity": [0.0, 0.0, 0.0],
        "spawn_velocity_car_frame": False,
    })
    result["vehicle_config"] = vehicle_config
    return result
