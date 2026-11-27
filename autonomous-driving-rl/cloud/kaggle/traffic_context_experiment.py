"""Opt-in producer-only TrafficContext profiles. Frozen SAC consumes the normal 409D state."""
from __future__ import annotations

import argparse
from collections import deque
from functools import partial
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import monotonic

import numpy as np

from cloud.kaggle.common import WORK, read, verify, write
from cloud.kaggle.local_risk import LOCK
from cloud.kaggle.local_workers import initialize, policy

ROOT = WORK / "short-dynamic-traffic"
PROFILES = {"A": 50., "B": 20., "C": 15., "D": 10., "E": 5., "F": None}
EXACT = ("geom_extreme_SCTXORyCCS_seed2", 33702, "eval")


def core_radius(speed_mps):
    return float(np.clip(6. + .65 * speed_mps + speed_mps ** 2 / 7., 5., 50.))


def dynamic_mask(position, relative_velocity, speed_mps, *, reliable=True):
    """Use a near radius with an exception for a possible future conflict within 50 m."""
    distance = np.linalg.norm(position, axis=1)
    observable = distance <= 50.
    if not reliable or not np.isfinite(speed_mps) or not np.isfinite(relative_velocity).all():
        return 50., observable
    radius = core_radius(speed_mps)
    time = np.linspace(.25, 4., 16)
    projected = position[:, None, :] + relative_velocity[:, None, :] * time[None, :, None]
    # Inflate the envelope with forecast time. Rear and crossing actors
    # use the same footprint test as leads. It does not require lane membership.
    conflict = np.any((np.abs(projected[:, :, 0]) <= 6.5 + .35*time**2)
                      & (np.abs(projected[:, :, 1]) <= 2.5 + .20*time**2), axis=1)
    return radius, observable & ((distance <= radius) | conflict)


def relative_states(ego, vehicles):
    if not vehicles:
        return np.empty((0, 2)), np.empty((0, 2))
    heading = float(ego.heading_theta)
    c, s = math.cos(heading), math.sin(heading)
    basis = np.array([[c, -s], [s, c]])
    position = (np.asarray([v.position for v in vehicles])-ego.position) @ basis
    velocity = (np.asarray([v.velocity for v in vehicles])-ego.velocity) @ basis
    return position, velocity


def assemble(profile, base, raw_obs, step_idx):
    from src.platform.agent_input import InputProfileId, build_agent_input

    sim = base._env
    ego = sim.agent
    traffic = [v for v in sim.engine.traffic_manager.traffic_vehicles if v is not ego]
    if profile == "F":
        pos, vel = relative_states(ego, traffic)
        speed = ego.speed_km_h / 3.6
        navigation = ego.navigation
        lanes = getattr(navigation, "current_ref_lanes", None)
        reliable = bool(lanes and ego.lane is not None)
        if reliable and speed > 3. and type(ego.lane).__name__ != "StraightLane":
            # A constant-vector forecast is less reliable in a bend.
            # Retain a wider floor without extending the 50 m observation limit.
            speed = min(25., max(speed, 8.))
        radius, mask = dynamic_mask(pos, vel, speed, reliable=reliable)
        selected = [v for v, keep in zip(traffic, mask, strict=True) if keep]
        cutoff = 50.  # Allows the risk exceptions selected above.
        base._experimental_radius = radius
        base._experimental_exceptions = int(np.count_nonzero(mask & (np.linalg.norm(pos, axis=1) > radius)))
    else:
        selected = traffic
        cutoff = PROFILES[profile]
        base._experimental_radius = cutoff
        base._experimental_exceptions = 0
    return build_agent_input(
        raw_obs=raw_obs, ego_vehicle=ego, traffic_vehicles=selected, navigation=ego.navigation,
        road_network=sim.current_map.road_network, step_index=step_idx, profile_id=InputProfileId.STATE_DECISION_V1,
        traffic_radius_m=cutoff, task_context_producer_id=base.task_context_producer_id)


class Monitor:
    def __init__(self, base):
        self.base = base
        self.tail = deque(maxlen=50)
        self.steps = self.stopped = self.current_stop = self.longest_stop = 0
        self.braking_steps = self.brake_onsets = self.false_stop_proxy = 0
        self.risk_decel_frames = self.braking_on_decel_frames = self.fast_closing_frames = 0
        self.activate = self.activated_npcs = self.moving_after_trigger = self.exceptions = 0
        self.last_pedal = self.last_steer = None
        self.sum_abs_steer_delta = 0.
        self.pending_start = self.previous_pending = self.pending_min = None
        self.radius_min, self.radius_max = 50., 0.
        self.previous_lead = None
        self.first_trigger_progress = None
        self.decels = []

    def before_step(self, obs, action):
        self.obs, self.action = obs, action
        manager = self.base._env.engine.traffic_manager
        pending = sum(len(group.vehicles) for group in manager.block_triggered_vehicles)
        if self.pending_start is None:
            sim = self.base._env
            lidar = sim.agent.config["lidar"]
            if (sim.config["physics_world_step_size"], sim.config["decision_repeat"],
                lidar["distance"], lidar["num_lasers"]) != (.02, 5, 50, 240):
                raise ValueError("Unexpected locked simulator or LiDAR parameters")
            self.pending_start = self.previous_pending = self.pending_min = pending
        self.radius_min = min(self.radius_min, self.base._experimental_radius)
        self.radius_max = max(self.radius_max, self.base._experimental_radius)
        self.exceptions += self.base._experimental_exceptions
        steer, pedal = map(float, action)
        if self.last_steer is not None:
            self.sum_abs_steer_delta += abs(steer - self.last_steer)
        self.brake_onsets += pedal < -.1 and (self.last_pedal is None or self.last_pedal >= -.1)
        self.braking_steps += pedal < -.1
        self.last_pedal, self.last_steer = pedal, steer
        ego = self.base._env.agent
        traffic = list(manager.traffic_vehicles)
        pos, rel = relative_states(ego, traffic)
        selected = np.flatnonzero((pos[:, 0] > 0) & (np.abs(pos[:, 1]) < 2.2)
                                  & (pos[:, 0] < min(50, 12 + ego.speed_km_h / 3.6*3)))
        self.lead = None
        if len(selected):
            index = int(selected[np.argmin(pos[selected, 0])])
            self.lead = (traffic[index].id, float(traffic[index].speed_km_h/3.6),
                         float(pos[index, 0]), float(rel[index, 0]))
        if self.lead and self.lead[3] < -2:
            self.fast_closing_frames += 1
        if self.lead and self.previous_lead and self.lead[0] == self.previous_lead[0]:
            deceleration = (self.previous_lead[1]-self.lead[1])*10
            if deceleration >= 1.8 and self.lead[3] < -.5:
                self.risk_decel_frames += 1
                self.braking_on_decel_frames += pedal < -.1
                if len(self.decels) < 25:
                    self.decels.append({"decision": self.steps, "lead_decel_mps2": deceleration,
                                        "lead_x_m": self.lead[2], "closing_mps": -self.lead[3], "pedal": pedal})
        self.previous_lead = self.lead

    def after_step(self, ego):
        self.steps += 1
        obs = self.obs
        manager = ego.engine.traffic_manager
        pending = sum(len(group.vehicles) for group in manager.block_triggered_vehicles)
        self.pending_min = min(self.pending_min, pending)
        if pending < self.previous_pending:
            self.activate += 1
            self.activated_npcs += self.previous_pending-pending
        if self.previous_pending >= 20 > pending and self.first_trigger_progress is None:
            self.first_trigger_progress = self.steps
        self.previous_pending = pending
        moving = sum(v.speed_km_h >= 1. for v in manager.traffic_vehicles)
        if pending < 20:
            self.moving_after_trigger = max(self.moving_after_trigger, moving)
        stopped = ego.speed_km_h < 1.
        self.stopped += stopped
        self.current_stop = self.current_stop+1 if stopped else 0
        self.longest_stop = max(self.longest_stop, self.current_stop)
        self.false_stop_proxy += bool(stopped and self.last_pedal < -.1
                                     and np.min(obs[np.r_[19:24,254:259]])*50 > 15)
        station, offset = ego.lane.local_coordinates(ego.position)
        self.tail.append({"step": self.steps, "speed_kmh": float(ego.speed_km_h), "action": self.action.tolist(),
                          "lane_offset_m": float(offset), "heading_error_rad": math.remainder(
                              ego.heading_theta-ego.lane.heading_theta_at(station), 2*math.pi),
                          "road_edges_m_input": (obs[:2]*18).tolist(), "lane": list(ego.lane_index or ()),
                          "active": len(manager.traffic_vehicles), "moving": moving, "pending": pending,
                          "radius_m": self.base._experimental_radius,
                          "traffic_selected": int(np.count_nonzero(obs[315:323])),
                          "traffic_rows": obs[259:315].reshape(8, 7)[obs[315:323] > .5].tolist()})

    def summary(self):
        return {"steps": self.steps, "stationary_s": self.stopped/10, "longest_stationary_s": self.longest_stop/10,
                "final_stationary_s": self.current_stop/10, "braking_s": self.braking_steps/10,
                "brake_onsets": self.brake_onsets, "stationary_braking_clear_proxy_s": self.false_stop_proxy/10,
                "mean_abs_steering_change": self.sum_abs_steer_delta/max(1, self.steps-1),
                "activated_npcs": self.activated_npcs, "activated_batches": self.activate,
                "pending_start": self.pending_start, "pending_final": self.previous_pending, "pending_min": self.pending_min,
                "first_below_20_pending_step": self.first_trigger_progress, "peak_moving_below_20_pending": self.moving_after_trigger,
                "radius_min_m": self.radius_min, "radius_max_m": self.radius_max, "dynamic_exception_actor_frames": self.exceptions,
                "fast_closing_frames": self.fast_closing_frames, "lead_deceleration_frames": self.risk_decel_frames,
                "braking_on_lead_deceleration_frames": self.braking_on_decel_frames, "lead_deceleration_examples": self.decels,
                "tail": list(self.tail)}


def episode(task):
    from cloud.kaggle.final_evaluation import evaluate_case
    from cloud.kaggle.runtime import ScheduledEnv

    profile, geometry, seed, phase, path, digest = task
    env = ScheduledEnv(phase, 0, 0, fixed=(geometry, seed))
    base = env.env.env
    # The instance binding leaves the official producer and V2/V4 provenance
    # files untouched and affects both reset and every encoded decision.
    base._build_agent_input = partial(assemble, profile, base)
    row = next(r for r in env.rows if r["geometry_id"] == geometry)
    monitor = Monitor(base)
    result = evaluate_case(policy(path, digest), row, seed, env, observer=monitor)
    return {"profile": profile, **result, **monitor.summary()}


def safe(row):
    return not row["technical_failure"] and row["outcome"] in ("SUCCESS", "TIMEOUT")


def main():
    from cloud.kaggle.smooth_traffic_eval import memory, worker_budget

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("exact", "paired", "all"), default="all")
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    lock = read(LOCK)
    path, digest = Path(lock["model"]), lock["model_sha256"]
    verify(path, digest)
    args_exact = [(name, *EXACT, path, digest) for name in PROFILES]

    def execute(pool, jobs, name):
        start = monotonic()
        rows = list(pool.map(episode, jobs))
        seconds = monotonic() - start
        decisions = sum(row["steps"] for row in rows)
        write(ROOT / f"{name}.json", {"cases": rows, "seconds": seconds, "decisions": decisions,
                                     "decisions_per_second": decisions/seconds, "memory_after": memory()})
        print(name, decisions, round(decisions/seconds, 2), [(x["profile"], x["outcome"], x["progress"]) for x in rows], flush=True)
        return rows

    profile = memory()
    n = worker_budget(profile)
    write(ROOT / "execution.json", {"simulator_workers": n, "start_memory": profile, "sha256": digest,
                                   "physics_dt": .02, "decision_repeat": 5, "lidar_range_m": 50, "training_updates": 0})
    with ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize) as pool:
        exact = execute(pool, args_exact, "exact") if args.phase != "paired" else read(ROOT/"exact.json")["cases"]
        if args.phase == "exact":
            return
        candidates = sorted(exact[1:], key=lambda row: (safe(row), row["clean"], row["progress"] or 0), reverse=True)
        # Keep a dynamic comparison even if it fails the exact case; baseline
        # is paired with any static candidate that reaches farthest safely.
        best_static = next(row["profile"] for row in candidates if row["profile"] != "F")
        winners = ["A", best_static, "F"]
        prior = read(WORK/"first-window-predictive/train-gate.json")["cases"]
        paired = [(r["geometry"], r["seed"], "mixed") for r in prior
                  if r["variant"] == "raw" and not r["geometry"].startswith("geom_easy")]
        execute(pool, [(name, geom, seed, phase, path, digest)
                       for geom, seed, phase in paired for name in winners], "paired")


if __name__ == "__main__":
    main()
