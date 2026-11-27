"""Offline-only counterfactual inference for the frozen 409D Stage 5 actor."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from cloud.kaggle.common import WORK, read, sha, verify, write
from cloud.kaggle.observation_capture import CASES, OUTPUT
from cloud.kaggle.risk_guidance import _path, _project
from src.training.stage5.pilot_observation_transform import GROUPWISE_PHYSICAL_V1

DIVISORS = np.asarray(GROUPWISE_PHYSICAL_V1["traffic_actors"]["divisors"], dtype=np.float32)
THRESHOLDS = (45, 40, 35, 30, 25, 20, 15, 10, 5)
DEADBAND = .02


def pack_traffic(observation, actors):
    """Recreate valid prefix, padding, active count and overflow from producer rows."""
    result = observation.copy()
    actors = np.asarray(actors, dtype=np.float32).reshape(-1, 7)
    distances = np.linalg.norm(actors[:, :2], axis=1)
    actors, distances = actors[distances <= 50], distances[distances <= 50]
    order = np.lexsort((actors[:, 1], actors[:, 0], distances))
    valid = min(len(order), 8)
    result[259:325] = 0
    result[259:315].reshape(8, 7)[:valid] = actors[order[:valid]] / DIVISORS
    result[315:315 + valid] = 1
    result[323:325] = valid / 8, min(len(order) - valid, 10) / 10
    return result


def mask_rays(observation, rays, removed_ids):
    """Reveal the next verified ray hit when the nearest hit is removed."""
    result = observation.copy()
    changed = 0
    for index, hits in rays.items():
        if hits[0][0] not in removed_ids:
            continue
        replacement = next((hit[1] for hit in hits if hit[0] not in removed_ids), 1.)
        result[19 + int(index)] = replacement
        changed += 1
    return result, changed


def boundary_pair(observation, side, distance, scale):
    result = observation.copy()
    total = float(observation[:2].sum()) * scale
    i = 0 if side == "left" else 1
    if not 0 < distance < total:
        raise ValueError("Counterfactual border lies outside the observed road width")
    result[i], result[1 - i] = distance / scale, (total - distance) / scale
    return result


def geometry_features(observation):
    points = observation[325:385].reshape(20, 3)[observation[385:405] > .5] * [50., 50., np.pi]
    path = _path(points[:, :2]) if len(points) >= 3 else None
    bends = np.abs(np.diff(np.unwrap(points[:8, 2]))) / 2.5
    return points, path, float(np.max(bends)) if len(bends) else 0.


def observed_scenario(frame, observation, scale):
    left, right = observation[:2] * scale
    angle = frame["heading_error_rad"]
    if min(left, right) < 2.8:
        if abs(angle) <= .09:
            return "near_edge_parallel"
        if (left < right and angle >= .14) or (right < left and angle <= -.14):
            return "near_edge_outward"
    if abs(frame["lane_offset_m"]) < .35:
        return "lane_center"
    return "other"


def sample_steps(metadata, actions, target):
    """Use a fixed grid plus outcome-independent temporal and physical strata."""
    frames = metadata["frames"]
    speed = np.array([f["speed_kmh"] for f in frames])
    pedal = actions[:, 1]
    first_pending = next((i for i, f in enumerate(frames)
                          if any(hits[0][0] in f["pending_ids"] for hits in f["rays"].values())), None)
    first_actor = next((i for i, f in enumerate(frames) if f["actors"]), None)
    onset = np.flatnonzero((pedal[:-1] >= -.05) & (pedal[1:] < -.05)) + 1
    stop = len(speed) - len(np.trim_zeros((speed >= 1).astype(int), trim="b"))
    stop_start = len(speed) - stop if stop else len(speed) - 1
    first_motion = int(np.argmax(speed > 10)) if (speed > 10).any() else 0
    step_labels = {
        "moving": first_motion, "first_active_context": first_actor, "first_verified_pending_ray": first_pending,
        "last_stop_start": min(stop_start, len(frames) - 1), "final": len(frames) - 1,
    }
    if len(onset):
        step_labels["first_brake_crossing"] = int(onset[0])
        before_stop = onset[(onset <= stop_start) & (onset >= first_motion)]
        if len(before_stop):
            step_labels["brake_before_final_stop"] = int(before_stop[-1])
    if len(speed) > 10:
        pedal_drop = np.flatnonzero((pedal[10:] <= pedal[:-10] - .15) & (speed[10:] > 10)) + 10
        deceleration = np.flatnonzero((speed[10:] <= speed[:-10] - 3) & (speed[:-10] > 10)) + 10
        if len(pedal_drop):
            step_labels["first_1s_pedal_drop"] = int(pedal_drop[0])
        if len(deceleration):
            step_labels["first_1s_speed_drop"] = int(deceleration[0])
    step_labels = {k: v for k, v in step_labels.items() if v is not None}
    selected = set(range(0, len(frames), 10))
    selected.update(i for center in step_labels.values() for i in range(max(0, center-8), min(len(frames), center+9)))
    if target:
        selected.update(i for i, f in enumerate(frames) if target in f["actor_ids"] and i % 3 == 0)
    for i in range(1, len(frames)):
        if frames[i]["lane"][:2] == frames[i-1]["lane"][:2] and frames[i]["lane"] != frames[i-1]["lane"]:
            selected.update(j for j in range(max(0, i-2), min(len(frames), i+3)))
    return sorted(selected), step_labels


def action_row(base, action):
    return {
        "raw_throttle_brake": float(base[1]), "counterfactual_throttle_brake": float(action[1]),
        "delta_throttle_brake": float(action[1]-base[1]),
        "raw_steering": float(base[0]), "counterfactual_steering": float(action[0]),
        "delta_steering": float(action[0]-base[0]),
        "brake_to_gas": int(base[1] < -DEADBAND and action[1] > DEADBAND),
        "gas_to_brake": int(base[1] > DEADBAND and action[1] < -DEADBAND),
        "steering_reversal": int(abs(base[0]) > DEADBAND and abs(action[0]) > DEADBAND and base[0]*action[0] < 0),
    }


def analyze_case(actor, meta, arrays, tables):
    frames = meta["frames"]
    obs, raw = arrays["policy"], arrays["actions"]
    case = meta["case"]
    scale = meta["config"]["road_distance_scale_m"]
    restored, _ = actor.predict(obs, deterministic=True)
    if np.max(np.abs(restored - raw)) > 2e-5:
        raise ValueError("The frozen actor failed original-input inference parity")
    final = frames[-1]
    target = None
    if case == "extreme_exact":
        across = [(j, a) for j, a in enumerate(final["actors"]) if a[0] > 0]
        index, position = min(across, key=lambda row: abs(row[1][0] - 31.4))
        if abs(position[0] - 31.4) > .25:
            raise ValueError("Could not match the archived 31.4 m target")
        target = final["actor_ids"][index]
    selected, landmarks = sample_steps(meta, raw, target)
    for i, f in enumerate(frames):
        hits = f["rays"]
        nearest = min((math.hypot(*a[:2]) for a in f["actors"]), default=None)
        forward = min((a[0] for a in f["actors"] if a[0] > 0 and abs(a[1]) < 2), default=None)
        active_ids = set(f["actor_ids"])
        pending_ids = set(f["pending_ids"])
        tables["temporal_braking_trace"].append({
            "case": case, "geometry": meta["geometry"], "seed": meta["seed"], "step": i, "t_s": i / 10,
            "landmark": "|".join(k for k, s in landmarks.items() if s == i),
            "terminal_input_only": int(f["terminal_input"]),
            "speed_kmh": f["speed_kmh"], "progress_pct": f["progress"]*100,
            "steering": float(raw[i, 0]), "throttle_brake": float(raw[i, 1]),
            "throttle_brake_change": float(raw[i, 1]-raw[i-1, 1]) if i else 0.,
            "closest_active_m": nearest, "closest_forward_active_x_m": forward,
            "target_x_m": f["actors"][f["actor_ids"].index(target)][0] if target in f["actor_ids"] else None,
            "active_in_radius": len(active_ids), "pending_total": f["pending_total"],
            "confirmed_active_rays": sum(h[0][0] in active_ids for h in hits.values()),
            "confirmed_pending_rays": sum(h[0][0] in pending_ids for h in hits.values()),
            "unattributed_hit_rays": int(np.count_nonzero(obs[i, 19:259] < 1-1e-6))-len(hits),
            "nearest_lidar_hit_m": float(np.min(obs[i, 19:259])*50),
            "left_edge_m": float(obs[i, 0]*scale), "right_edge_m": float(obs[i, 1]*scale),
            "lane_offset_m": f["lane_offset_m"], "heading_error_rad": f["heading_error_rad"],
            "lane_type": f["lane_type"], "waypoint_curvature_proxy_1pm": geometry_features(obs[i])[2],
        })

    for offset in range(0, len(selected), 24):
        vectors, rows = [], []
        for i in selected[offset:offset + 24]:
            build_cases(i, meta, obs[i], target, vectors, rows)
        predicted, _ = actor.predict(np.stack(vectors), deterministic=True)
        for (i, table, fields), action in zip(rows, predicted):
            tables[table].append({**fields, **action_row(raw[i], action)})
    return {"landmarks": landmarks, "sample_frames": len(selected), "matched_final_target": target is not None,
            "baseline_inference_max_error": float(np.max(np.abs(restored-raw)))}


def build_cases(i, meta, obs, target, vectors, rows):
    f = meta["frames"][i]
    ids = f["actor_ids"]
    actors = np.asarray(f["actors"], dtype=np.float32).reshape(-1, 7)
    all_active = set(ids)
    pending = set(f["pending_ids"])
    rays = f["rays"]
    scale = meta["config"]["road_distance_scale_m"]
    points, path, curvature = geometry_features(obs)
    scene = observed_scenario(f, obs, scale)
    side = "left" if obs[0] < obs[1] else "right"
    changed_lane = (i > 0 and f["lane"][:2] == meta["frames"][i-1]["lane"][:2]
                    and f["lane"] != meta["frames"][i-1]["lane"])
    if changed_lane:
        scene = "same_road_lane_index_change"
    if f["lane_type"] != "StraightLane" and scene == "other":
        scene = "curve"
    speed = f["speed_kmh"]
    speed_band = "stopped" if speed < 1 else "low" if speed < 15 else "moving"
    base = {"case": meta["case"], "geometry": meta["geometry"], "seed": meta["seed"], "step": i, "speed_kmh": speed,
            "speed_band": speed_band, "scene": scene, "lane_type": f["lane_type"], "curvature_proxy_1pm": curvature}

    def add(table, vector, name, setting=None, value=None, removed_actors=0, changed_rays=0, synthetic=False):
        vectors.append(vector)
        rows.append((i, table, {**base, "variant": name, "setting": setting, "value": value,
                               "removed_actors": removed_actors, "changed_rays": changed_rays,
                               "synthetic_probe": int(synthetic)}))

    def traffic_without(removed):
        return pack_traffic(obs, [a for k, a in zip(ids, actors) if k not in removed]) if removed else obs

    def ablate(group, traffic_ids, ray_ids):
        b = traffic_without(traffic_ids)
        c, count = mask_rays(obs, rays, ray_ids)
        d, _ = mask_rays(b, rays, ray_ids)
        for code, vector, actors_n, rays_n in (("A", obs, 0, 0), ("B", b, len(traffic_ids), 0),
                                               ("C", c, 0, count), ("D", d, len(traffic_ids), count),
                                               ("E", obs.copy(), 0, 0)):
            add("lidar_traffic_ablation", vector, code, group, removed_actors=actors_n, changed_rays=rays_n)

    add("observation_sensitivity", obs, "frozen_identity")
    distance = np.linalg.norm(actors[:, :2], axis=1)
    far = {k for k, d in zip(ids, distance) if d > 30}
    ablate("active_gt30m", far, far)
    ablate("pending_only", set(), pending)
    ablate("all_active", all_active, all_active)
    ablate("active_and_pending", all_active, all_active | pending)
    if target and target in all_active:
        ablate("exact_target", {target}, {target})
        ablate("other_active", all_active - {target}, all_active - {target})
        for rank, key in enumerate(ids[:8]):
            ablate(f"remove_slot_{rank + 1}", {key}, {key})
    if target and len(actors) >= 4 and 48 < distance[3] <= 50:
        for radius in (48., 49., 49.5, 49.9, 49.99, 50., 50.01, 50.1, 51.):
            updated = actors.copy()
            updated[3, :2] *= radius / distance[3]
            add("traffic_distance_sensitivity", pack_traffic(obs, updated), "range_edge_fourth_actor",
                "meters", radius, int(radius > 50), synthetic=True)
    widths = float(obs[:2].sum() * scale)
    route_distance = _project(actors[:, :2], path)[:2] if path is not None and len(actors) else None
    for cutoff in THRESHOLDS:
        removed = {k for k, d in zip(ids, distance) if d > cutoff}
        add("traffic_distance_sensitivity", traffic_without(removed), "euclidean_cutoff", "meters", cutoff, len(removed))
        if route_distance is not None:
            station, lateral = route_distance
            removed = {k for k, s, y in zip(ids, station, lateral) if abs(s) > cutoff and y <= 3}
            add("traffic_distance_sensitivity", traffic_without(removed), "route_station_cutoff", "meters", cutoff, len(removed))

    front = np.flatnonzero((actors[:, 0] > 6) & (np.abs(actors[:, 1]) < 3.5))
    chosen = ids.index(target) if target in all_active else int(front[0]) if len(front) else None
    if chosen is not None:
        source_actor = actors[chosen]
        length = float(distance[chosen])
        def edit(name, change, table, setting, value):
            updated = actors.copy()
            updated[chosen] = change
            add(table, pack_traffic(obs, updated), name, setting, value, synthetic=True)

        for metres in (5, 10, 15, 20, 25, 30, 31.4, 35, 40, 45, 49):
            a = source_actor.copy()
            a[:2] *= metres / length
            edit("change_target_distance", a, "traffic_distance_sensitivity", "meters", metres)
        ego_v = np.asarray(f["ego_velocity_mps"])
        ego_speed = max(0., float(ego_v[0]))
        profiles = [("stopped", 0., 0.), ("same_speed", ego_speed, 0.),
                    ("slower", max(0., ego_speed-3), 0.), ("faster", ego_speed+3, 0.),
                    ("pulling_away", ego_speed+8, 0.), ("crossing_left", 5., np.pi/2)]
        for label, rate, heading in profiles:
            a = source_actor.copy()
            a[2:4] = rate * np.array([math.cos(heading), math.sin(heading)]) - ego_v
            a[4] = heading
            edit("change_target_velocity", a, "relative_velocity_sensitivity", label, a[2])
        heading = float(source_actor[4])
        for rate in (0., 2., 5., 10.):
            a = source_actor.copy()
            a[2:4] = rate * np.array([math.cos(heading), math.sin(heading)]) - ego_v
            edit("fixed_heading_speed", a, "relative_velocity_sensitivity", f"abs_{rate:g}_mps", a[2])
        r = min(length, 45.)
        if r > 5.:
            configurations = (("front", r, 0., 0.), ("adjacent_left", math.sqrt(r*r-3.5**2), 3.5, 0.),
                              ("adjacent_right", math.sqrt(r*r-3.5**2), -3.5, 0.),
                              ("behind", -r, 0., 0.), ("oncoming", r, 0., -np.pi))
            for label, x, y, heading in configurations:
                a = source_actor.copy()
                a[:2], a[4] = [x, y], heading
                a[2:4] = 5. * np.array([math.cos(heading), math.sin(heading)]) - ego_v
                edit("change_target_position", a, "observation_sensitivity", label, y)

    if widths > 3:
        for edge in ("left", "right"):
            for distance_m in (.85, 1.1, 1.5, 2., 2.75, widths/2):
                add("boundary_sensitivity", boundary_pair(obs, edge, distance_m, scale),
                    "redistribute_road_offsets", edge, distance_m, synthetic=True)
    for lateral in (-1.5, -.75, 0., .75, 1.5):
        changed = obs.copy()
        changed[8] = np.clip(lateral/meta["config"]["lane_offset_scale_m"] + .5, 0, 1)
        add("boundary_sensitivity", changed, "ego_lane_offset", side, lateral, synthetic=True)
    for alignment in (-.24, -.12, 0., .12, .24):
        changed = obs.copy()
        changed[2] = (1 + alignment) / 2
        add("boundary_sensitivity", changed, "ego_heading_field", "signed_lateral_alignment", alignment, synthetic=True)
    if len(points) >= 3:
        valid = obs[385:405] > .5
        for shift in (-1.5, 0., 1.5):
            changed = obs.copy()
            changed[325:385].reshape(20, 3)[valid, 1] += shift/50
            add("boundary_sensitivity", changed, "waypoint_lateral_shift", side, shift, synthetic=True)
        anchor = points[0, :2]
        heading = points[0, 2]
        distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1))]
        changed = obs.copy()
        changed[325:385].reshape(20, 3)[valid] = np.column_stack(
            ((anchor + distances[:, None]*[math.cos(heading), math.sin(heading)])/50,
             np.full(len(points), heading/np.pi)))
        add("boundary_sensitivity", changed, "straighten_waypoint_suffix", side, curvature, synthetic=True)
    changed = obs.copy()
    changed[9:19] = .5
    add("boundary_sensitivity", changed, "navigation_midpoint_probe", side, synthetic=True)
    changed = obs.copy()
    changed[11:14] = changed[16:19] = [0., .5, .5]
    add("boundary_sensitivity", changed, "navigation_curve_fields_to_straight", side, synthetic=True)


def main():
    from cloud.kaggle.runtime import check_state, load_policy
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=OUTPUT / "capture")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    lock = read(WORK / "postwindow-product/cleanup-locked-v2/delivery-lock.json")
    path = Path(lock["model"])
    verify(path, lock["model_sha256"])
    actor = load_policy(path)
    tables = {name: [] for name in ("observation_sensitivity", "traffic_distance_sensitivity",
              "relative_velocity_sensitivity", "lidar_traffic_ablation", "boundary_sensitivity",
              "temporal_braking_trace")}
    inputs, samples = {}, {}
    for case, geometry, seed, split in CASES:
        files = [args.capture / f"{case}.{ext}" for ext in ("json", "npz")]
        meta = read(files[0])
        if (meta["geometry"], meta["seed"], meta["split"], meta["model_sha256"]) != (
                geometry, seed, split, lock["model_sha256"]):
            raise ValueError("Unexpected capture provenance")
        with np.load(files[1], allow_pickle=False) as arrays:
            samples[case] = analyze_case(actor, meta, arrays, tables)
        inputs.update({file.name: sha(file) for file in files})
        print(case, samples[case], flush=True)
    outputs = {}
    for name, rows in tables.items():
        path_csv = args.out / f"{name}.csv"
        with path_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        outputs[path_csv.name] = {"sha256": sha(path_csv), "rows": len(rows)}
    verify(path, lock["model_sha256"])
    check_state(Path(lock["full_state"]), lock["full_state_manifest_sha256"])
    write(args.out / "manifest.json", {"capture_sha256": inputs, "report_files": outputs, "samples": samples,
          "model_sha256": lock["model_sha256"], "full_state_sha256": lock["full_state_manifest_sha256"],
          "gradient_updates": 0, "replay_rows": 0, "counterfactual_closed_loop_steps": 0})


if __name__ == "__main__":
    main()
