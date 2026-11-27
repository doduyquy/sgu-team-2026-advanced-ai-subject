"""Measure slot-entry discontinuities using frozen observations and producer eligibility."""
import csv
from pathlib import Path

import numpy as np

from cloud.kaggle.common import read, sha, verify, write
from cloud.kaggle.local_risk import LOCK
from cloud.kaggle.observation_capture import CASES, OUTPUT
from cloud.kaggle.observation_sensitivity import DIVISORS, pack_traffic
from cloud.kaggle.traffic_context_experiment import PROFILES, ROOT, dynamic_mask

SCENARIOS = {
    "ahead_constant_gap": (0., 0., 0.),
    "closing_moderate": (0., -3., 0.),
    "closing_fast": (0., -9., 0.),
    "adjacent_crossing": (-22., -1., 6.),
    "lateral_separating": (80., 0., 2.),
}


def actor_at(radius, case, ego_speed=0.):
    angle, vx, vy = case
    a = np.deg2rad(angle)
    heading = (np.arctan2(vy, ego_speed+vx)+np.pi) % (2*np.pi)-np.pi
    return np.array([radius*np.cos(a), radius*np.sin(a), vx, vy, heading, 4.5, 1.8], dtype=np.float32)


def selected(profile, actors, speed):
    distance = np.linalg.norm(actors[:, :2], axis=1)
    if profile == "F":
        return dynamic_mask(actors[:, :2], actors[:, 2:4], speed)[1]
    return distance <= PROFILES[profile]


def boundary(profile, scenario, speed):
    if profile != "F":
        return PROFILES[profile]
    ranges = np.linspace(.5, 50.5, 201)
    candidates = np.stack([actor_at(r, scenario) for r in ranges])
    mask = selected(profile, candidates, speed)
    ends = np.flatnonzero(mask[:-1] & ~mask[1:])
    if not len(ends):
        return None
    left, right = ranges[ends[-1]], ranges[ends[-1] + 1]
    for _ in range(18):
        middle = (left + right)*.5
        if selected(profile, actor_at(middle, scenario)[None], speed)[0]:
            left = middle
        else:
            right = middle
    return (left + right)*.5


def main():
    import torch
    from cloud.kaggle.runtime import load_policy

    ROOT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    lock = read(LOCK)
    model_path = Path(lock["model"])
    verify(model_path, lock["model_sha256"])
    model = load_policy(model_path)
    details = []
    vectors = []
    for key, *_ in CASES:
        meta = read(OUTPUT/"capture"/f"{key}.json")
        with np.load(OUTPUT/"capture"/f"{key}.npz", allow_pickle=False) as z:
            obs = z["policy"]
        candidates = np.flatnonzero((obs[:, 324] == 0) & (obs[:, 323] < 1) & (obs[:, 408] < .5))
        if not len(candidates):
            continue
        for i in candidates[np.linspace(0, len(candidates)-1, min(18, len(candidates)), dtype=int)]:
            row = obs[i]
            actors = row[259:315].reshape(8, 7)[row[315:323] > .5]*DIVISORS
            speed = max(0., (float(row[3])*81-1)/3.6)
            speed_for_radius = min(25., max(speed, 8.)) if (
                speed > 3. and meta["frames"][i]["lane_type"] != "StraightLane") else speed
            for profile in PROFILES:
                for name, scenario in SCENARIOS.items():
                    limit = boundary(profile, scenario, speed_for_radius)
                    if limit is None:
                        continue
                    frames = []
                    masks = []
                    for delta in (-.01, .01):
                        synthetic = np.vstack([actors, actor_at(limit + delta, scenario, speed)])
                        mask = selected(profile, synthetic, speed_for_radius)
                        masks.append(bool(mask[-1]))
                        frames.append(pack_traffic(row, synthetic[mask]))
                    if masks != [True, False]:
                        continue
                    vectors.extend(frames)
                    details.append({"profile": profile, "scene": key, "frame": int(i), "scenario": name,
                                    "boundary_m": float(limit), "ego_speed_mps": speed})
    actions, _ = model.predict(np.stack(vectors), deterministic=True)
    for item, (a,b) in zip(details, actions.reshape(-1, 2, 2), strict=True):
        item.update(steering_inside=float(a[0]), steering_outside=float(b[0]),
                    pedal_inside=float(a[1]), pedal_outside=float(b[1]),
                    steering_jump=float(abs(b[0]-a[0])), pedal_jump=float(abs(b[1]-a[1])))
    csv_path = ROOT/"boundary-probes.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, details[0].keys())
        writer.writeheader()
        writer.writerows(details)
    summaries = {}
    for name in PROFILES:
        rows = [r for r in details if r["profile"] == name]
        pedal = [r["pedal_jump"] for r in rows]
        summaries[name] = {"pairs": len(rows), "median_pedal_jump": float(np.median(pedal)),
                           "p90_pedal_jump": float(np.quantile(pedal, .9)), "max_pedal_jump": max(pedal),
                           "max_steering_jump": max(r["steering_jump"] for r in rows),
                           "min_boundary_m": min(r["boundary_m"] for r in rows),
                           "max_boundary_m": max(r["boundary_m"] for r in rows)}
    write(ROOT/"boundary.json", {"summary": summaries, "source": str(OUTPUT/"manifest.json"),
                                "source_sha256": sha(OUTPUT/"manifest.json"), "csv_sha256": sha(csv_path),
                                "producer_candidates_only": True, "training_updates": 0})
    print(summaries, flush=True)


if __name__ == "__main__":
    main()
