"""Capture frozen Stage 5 inference and read-only simulator metadata."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import math
import multiprocessing
from pathlib import Path

import numpy as np

from cloud.kaggle.common import WORK, read, verify, write
from cloud.kaggle.local_workers import initialize

OUTPUT = WORK / "observation-sensitivity"
CASES = (
    ("extreme_exact", "geom_extreme_SCTXORyCCS_seed2", 33702, "eval"),
    ("easy", "geom_easy_SCS_seed19", 59600, "mixed"),
    ("medium", "geom_medium_SCXCCS_seed2", 59601, "mixed"),
    ("medium_turn", "geom_medium_SCTCS_seed2", 59602, "mixed"),
    ("hard", "geom_hard_XTOCS_seed16", 59604, "mixed"),
    ("extreme_transfer", "geom_extreme_CrXROSTR_seed18", 59606, "mixed"),
)


def lidar_hits(sim, ego, observed):
    """Recast positive returns and require sensor fraction parity before attribution."""
    from metadrive.utils.math import get_laser_end, panda_vector
    from metadrive.utils.utils import get_object_from_node

    sensor = sim.engine.get_sensor("lidar")
    distance = ego.config["lidar"]["distance"]
    angles = sensor._get_lidar_range(len(observed), sensor.start_phase_offset)
    world = sim.engine.physics_world.dynamic_world
    start = panda_vector(*ego.position, sensor.DEFAULT_HEIGHT)
    excluded = set(ego.dynamic_nodes)
    rays = {}
    for i in np.flatnonzero(observed < 1 - 1e-6):
        end = get_laser_end(angles, distance, i, ego.heading_theta, *ego.position)
        ray = world.rayTestAll(start, panda_vector(*end, sensor.DEFAULT_HEIGHT), sensor.mask)
        hits = sorted((hit for hit in ray.getHits() if hit.getNode() not in excluded),
                      key=lambda hit: hit.getHitFraction())
        if not hits or abs(hits[0].getHitFraction() - observed[i]) > 2e-5:
            continue
        if len(hits) > 1 and abs(hits[1].getHitFraction() - hits[0].getHitFraction()) < 1e-6:
            # A fraction alone cannot disambiguate simultaneous returns.
            continue
        matches = []
        for hit in hits:
            obj = get_object_from_node(hit.getNode())
            matches.append([str(getattr(obj, "id", "")), float(hit.getHitFraction()), hit.getNode().getName()])
        rays[int(i)] = matches
    return rays


def capture(task):
    from cloud.kaggle.runtime import ScheduledEnv, load_policy
    from src.platform.agent_input import extract_traffic_context
    from src.platform.task_context_producer import TaskContextLocalizationError

    case, geometry, seed, split, out = task
    lock = read(WORK / "postwindow-product/cleanup-locked-v2/delivery-lock.json")
    path = Path(lock["model"])
    verify(path, lock["model_sha256"])
    actor = load_policy(path)
    names = ("encoder", "policy", "actions")
    arrays = {key: [] for key in names}
    frames = []

    class CaptureEnv(ScheduledEnv):
        def transform(self, obs):
            self.encoder_observation = obs.copy()
            return super().transform(obs)

    env = CaptureEnv(split, 0, 0, fixed=(geometry, seed))
    outcome = None
    try:
        obs, reset = env.reset()
        sim = env.env.env._env
        ego = sim.agent
        lidar = ego.config["lidar"]
        config = {
            "decision_repeat": sim.config["decision_repeat"],
            "physics_step_s": sim.config["physics_world_step_size"],
            "lidar": {k: lidar[k] for k in ("num_lasers", "distance", "num_others", "gaussian_noise", "dropout_prob")},
            "lidar_start_radians": sim.engine.get_sensor("lidar").start_phase_offset,
            "side_lasers": ego.config["side_detector"]["num_lasers"],
            "line_lasers": ego.config["lane_line_detector"]["num_lasers"],
            "max_speed_kmh": ego.max_speed_km_h,
            "max_steering": ego.MAX_STEERING,
            "road_distance_scale_m": (sim.current_map.MAX_LANE_NUM + 1) * sim.current_map.MAX_LANE_WIDTH,
            "lane_offset_scale_m": sim.current_map.MAX_LANE_WIDTH,
        }
        if (config["decision_repeat"], config["physics_step_s"], lidar["num_lasers"], lidar["distance"],
            lidar["num_others"], lidar["gaussian_noise"], lidar["dropout_prob"]) != (5, .02, 240, 50, 0, 0, 0):
            raise ValueError("Unexpected cadence or LiDAR config")
        info = reset
        for step in range(reset["horizon_steps"] + 1):
            action, _ = actor.predict(obs, deterministic=True)
            manager = sim.engine.traffic_manager
            active = [v for v in manager.traffic_vehicles if v is not ego]
            pending = {key for group in manager.block_triggered_vehicles for key in group.vehicles}
            full = extract_traffic_context(ego, active, capacity=max(8, len(active)))
            actors = full.actors_array[:full.active_count]
            if not np.array_equal(actors[:8], env.encoder_observation[259:315].reshape(8, 7)[:len(actors)]):
                raise ValueError("Producer rows differ from actor observation")
            if abs(float(obs[324]) - min(max(0, len(actors)-8), 10) / 10) > 1e-6:
                raise ValueError("Producer overflow differs from actor observation")
            cos, sin = math.cos(ego.heading_theta), math.sin(ego.heading_theta)
            heading_basis = np.array([[cos, -sin], [sin, cos]])
            positions = [(np.asarray(v.position) - ego.position) for v in active]
            sort_keys = [(math.hypot(*p), *(round(float(c), 4) for c in p @ heading_basis)) for p in positions]
            indices = [i for i in sorted(range(len(active)), key=sort_keys.__getitem__) if sort_keys[i][0] <= 50]
            if len(indices) != len(actors) or not np.allclose(
                    np.array([sort_keys[i][1:] for i in indices]).reshape(-1, 2), actors[:, :2]):
                raise ValueError("Cannot verify producer actor identity")
            rays = lidar_hits(sim, ego, obs[19:259])
            station, lateral = ego.lane.local_coordinates(ego.position)
            heading_error = math.remainder(ego.heading_theta - ego.lane.heading_theta_at(station), 2 * math.pi)
            frames.append({
                "step": step, "progress": float(info.get("route_completion") or 0),
                "speed_kmh": float(ego.speed_km_h), "ego_velocity_mps": (ego.velocity @ heading_basis).tolist(),
                "lane": list(ego.lane_index), "lane_offset_m": float(lateral),
                "heading_error_rad": float(heading_error), "lane_type": type(ego.lane).__name__,
                "actors": actors.tolist(), "actor_ids": [str(active[i].id) for i in indices],
                "actor_lanes": [list(active[i].lane_index or ()) for i in indices],
                "pending_ids": sorted(pending & {h[0] for hits in rays.values() for h in hits}),
                "pending_total": len(pending), "rays": rays,
                "terminal_input": outcome is not None,
            })
            for key, value in zip(names, (env.encoder_observation, obs, action)):
                arrays[key].append(value.copy())
            if outcome is not None:
                break
            try:
                obs, _, terminated, truncated, info = env.step(action)
            except TaskContextLocalizationError:
                outcome = "TASK_CONTEXT_LOCALIZATION"
                break
            if terminated or truncated:
                outcome = info["primary_reason"]
    finally:
        env.close()
    np.savez_compressed(out / f"{case}.npz", **{k: np.stack(v) for k, v in arrays.items()})
    write(out / f"{case}.json", {"case": case, "geometry": geometry, "seed": seed, "split": split,
                                "outcome": outcome, "model_sha256": lock["model_sha256"], "config": config,
                                "frames": frames})
    print(case, outcome, len(frames), frames[-1]["progress"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT / "capture")
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize) as pool:
        list(pool.map(capture, [(*case, args.out) for case in CASES]))


if __name__ == "__main__":
    main()
