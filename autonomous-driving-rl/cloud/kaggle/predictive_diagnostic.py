"""Matched local simulator runs for the predictive controller and privileged IDM."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import wraps
import multiprocessing
from pathlib import Path
from time import monotonic

import numpy as np

from cloud.kaggle.common import WORK, read, sha, write
from cloud.kaggle.local_workers import initialize

OUT = WORK / "first-window-predictive"
EXACT = ("geom_extreme_SCTXORyCCS_seed2", 33702)
SWITCH = 1360


def traffic_probe(env, trace, hooked):
    from metadrive.policy.idm_policy import IDMPolicy

    manager = env.engine.traffic_manager
    live = manager.traffic_vehicles
    for vehicle in live:
        name = vehicle.id
        if name in hooked:
            continue
        policy = env.engine.get_policy(name)
        if not isinstance(policy, IDMPolicy):
            continue
        hooked.add(name)

        def observe(p, key):
            accelerate, steer, route = p.acceleration, p.steering_control, p.move_to_next_road

            @wraps(accelerate)
            def measured(front, gap):
                result = accelerate(front, gap)
                trace.setdefault(key, {}).update(front=getattr(front, "id", None), gap=float(gap),
                      acceleration=float(result), target_speed=float(p.target_speed),
                      fallback_signature=front is None and gap == 5)
                return result

            @wraps(steer)
            def target(lane):
                trace.setdefault(key, {})["steering_target"] = getattr(lane, "index", None)
                return steer(lane)

            @wraps(route)
            def routing():
                result = route()
                trace.setdefault(key, {}).update(route_connected=bool(result),
                     routing_target=getattr(p.routing_target_lane, "index", None))
                return result

            p.acceleration, p.steering_control, p.move_to_next_road = measured, target, routing

        observe(policy, name)


def episode(task):
    from metadrive.policy.idm_policy import IDMPolicy
    from cloud.kaggle.runtime import ScheduledEnv, load_policy
    from cloud.kaggle.risk_metrics import RiskMetrics
    from src.platform.task_context_producer import TaskContextLocalizationError

    geometry, seed, split, variant = task
    lock = read(WORK / "postwindow-product/cleanup-locked-v2/delivery-lock.json")
    model = Path(lock["model"])
    if sha(model) != lock["model_sha256"]:
        raise ValueError("Frozen model checksum mismatch")
    actor = load_policy(model)
    env = ScheduledEnv(split, 0, 0, fixed=(geometry, seed))
    controller = None
    if variant in ("predictive", "guarded"):
        from cloud.kaggle.predictive_risk import PredictiveController

        controller = PredictiveController(guarded=variant == "guarded")
    exact = (geometry, seed) == EXACT
    events, trace, hooked, snapshots = Counter(), {}, set(), {}
    interventions = []
    observer = RiskMetrics()
    t0 = monotonic()
    step, info, peak, min_clearance = 0, {}, 0.0, None
    try:
        obs, reset = env.reset()
        sim = env.env.env._env
        if (sim.config["physics_world_step_size"], sim.config["decision_repeat"]) != (.02, 5):
            raise RuntimeError("Predictive comparison requires the locked simulator timestep and control frequency")
        manager = sim.engine.traffic_manager
        ego = sim.agent
        idm = IDMPolicy(ego, seed) if variant.startswith("idm") else None
        for step in range(1, reset["horizon_steps"] + 1):
            if exact and (step == SWITCH - 25 or (step > SWITCH - 25 and step % 10 == 0)):
                traffic_probe(sim, trace, hooked)
            raw, _ = actor.predict(obs, deterministic=True)
            action = raw
            evidence = None
            reason = "actor"
            if variant == "idm" or (variant == "idm-switch" and step > SWITCH):
                action = np.clip(idm.act(), -1., 1.).astype(np.float32)
                reason = "idm"
            elif controller is not None:
                action, evidence = controller.act(obs, raw)
                reason = evidence["reason"]
                margin = evidence.get("min_clearance_m")
                if margin is not None:
                    min_clearance = margin if min_clearance is None else min(margin, min_clearance)
            events[reason] += 1
            observer.before_step(obs, action)
            obs, _, terminal, truncated, info = env.step(action)
            observer.after_step(ego)
            progress = float(info["route_completion"])
            peak = max(peak, progress)
            if controller is not None and reason != "actor":
                interventions.append({
                    "step": step, "progress": progress, "risk": evidence, "raw": raw.tolist(),
                    "guided": action.tolist(),
                    "executed": [float(info["action_steering"]), float(info["action_throttle_brake"])],
                })
            if exact and (step in (SWITCH, SWITCH + 100, SWITCH + 400, 2400) or terminal or truncated):
                live = manager.traffic_vehicles
                pending = manager.block_triggered_vehicles
                pending_ids = {key for group in pending for key in group.vehicles}
                active_ids = {v.id for v in live}
                for record in trace.values():
                    front = record.get("front")
                    record["front_state"] = ("pending" if front in pending_ids else
                                             "active" if front in active_ids else
                                             "ego" if front == ego.id else "unresolved" if front else "none")
                snapshots[str(step)] = {
                    "progress": progress, "ego_lane": list(ego.lane_index or ()), "speed_kmh": float(ego.speed_km_h),
                    "raw": raw.tolist(), "executed": action.tolist(), "pending": sum(len(g.vehicles) for g in pending),
                    "next_trigger": str(pending[-1].trigger_road) if pending else None, "active": len(live),
                    "stopped_npcs": [{"id": v.id, "lane": list(v.lane_index or ()), "pos": list(map(float, v.position)),
                      "speed_kmh": float(v.speed_km_h), **trace.get(v.id, {})}
                      for v in live if v.speed_km_h < 1.0]}
            if terminal or truncated:
                break
    except TaskContextLocalizationError:
        info["primary_reason"] = "TASK_CONTEXT_LOCALIZATION"
    finally:
        env.close()
    return {"geometry": geometry, "seed": seed, "split": split, "variant": variant, "steps": step,
            "seconds": monotonic() - t0, "clean": bool(info.get("clean_success")),
            "outcome": info.get("primary_reason"), "progress": info.get("route_completion"), "max_progress": peak,
            "min_predicted_clearance_m": min_clearance, "events": dict(events),
            "snapshots": snapshots, "metrics": observer.summary(), "interventions": interventions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("idm", "exact", "train-gate", "revision-exact", "revision-train"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == "idm":
        tasks = [(*EXACT, "eval", name) for name in ("raw", "idm", "idm-switch")]
    elif args.phase == "exact":
        tasks = [(*EXACT, "eval", "predictive")]
    elif args.phase == "revision-exact":
        tasks = [(*EXACT, "eval", variant) for variant in ("raw", "idm-switch", "guarded")]
    else:
        from cloud.kaggle.runtime import pool

        rows = pool("mixed")
        # Pair the same seed on representative TRAIN families before results.
        choices = ("SCS", "SCXCCS", "SCTCS", "SCTXrCS", "XTOCS", "SCTXORyCCS", "CrXROSTR")
        fixtures = []
        for index, family in enumerate(choices):
            eligible = sorted(row["geometry_id"] for row in rows if row["sequence"] == family)
            if eligible:
                fixtures.append((eligible[len(eligible)//2], 59600 + index))
        variants = ("guarded",) if args.phase == "revision-train" else ("raw", "predictive")
        tasks = [(geometry, seed, "mixed", variant) for geometry, seed in fixtures for variant in variants]
    file = args.out / (args.phase + ".json")
    scratch = args.out / (args.phase + "-partials")
    scratch.mkdir()
    start = monotonic()
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize) as workers:
        futures = {workers.submit(episode, task): i for i, task in enumerate(tasks)}
        results = {}
        for future in as_completed(futures):
            index = futures[future]
            result = future.result()
            results[index] = result
            write(scratch / f"{index:02d}.json", result)
            print(result["variant"], result["geometry"], result["outcome"], result["progress"], flush=True)
    elapsed = monotonic() - start
    write(file, {"wall_seconds": elapsed, "decisions_per_second": sum(r["steps"] for r in results.values()) / elapsed,
                 "cases": [results[i] for i in range(len(tasks))]})


if __name__ == "__main__":
    main()
