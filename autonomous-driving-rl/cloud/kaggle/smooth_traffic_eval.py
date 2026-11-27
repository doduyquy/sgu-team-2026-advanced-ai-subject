"""Read-only qualification for the isolated Smooth TrafficContext sandbox."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil

from cloud.kaggle.common import WORK, read, verify, write
from cloud.kaggle.observation_capture import CASES, OUTPUT
from cloud.kaggle.observation_sensitivity import pack_traffic
from cloud.kaggle.smooth_traffic import blend, variants

ROOT = WORK / "smooth-traffic"


class BlendedPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.interventions = 0

    def predict(self, observation, deterministic=True):
        raw, state = self.policy.predict(observation, deterministic=True)
        action = blend(self.policy, observation, raw)
        self.interventions += abs(float(action[1] - raw[1])) > .001
        return action, state


class TrafficObserver:
    def __init__(self):
        self.stationary = self.run = self.longest = self.steps = 0
        self.last = 0.
        self.change = 0.
        self.pending = None
        self.min_pending = None
        self.max_active = 0

    def before_step(self, observation, action):
        steering = float(action[0])
        if self.steps:
            self.change += abs(steering - self.last)
        self.last = steering

    def after_step(self, ego):
        self.steps += 1
        stopped = ego.speed_km_h < 1.
        self.stationary += stopped
        self.run = self.run + 1 if stopped else 0
        self.longest = max(self.longest, self.run)
        manager = ego.engine.traffic_manager
        count = sum(len(x.vehicles) for x in manager.block_triggered_vehicles)
        if self.pending is None:
            self.pending = count
            self.min_pending = count
        self.min_pending = min(self.min_pending, count)
        self.max_active = max(self.max_active, len(manager.traffic_vehicles))

    def summary(self):
        return {"stationary_s": self.stationary / 10, "longest_stationary_s": self.longest / 10,
                "pending_start": self.pending, "pending_min": self.min_pending, "peak_active": self.max_active,
                "mean_abs_steering_change": self.change / max(1, self.steps - 1)}


def batch_blend(actor, observations):
    out = []
    for start in range(0, len(observations), 160):
        batch = observations[start:start+160]
        raw, _ = actor.predict(batch, deterministic=True)
        weights, vectors, parents = [], [], []
        blended = raw.copy()
        for j, obs in enumerate(batch):
            choices = variants(obs)
            if len(choices) == 1 and choices[0][1] is obs:
                continue
            for weight, vector in choices:
                parents.append(j)
                weights.append(weight)
                vectors.append(vector)
            blended[j, 1] = 0
        if vectors:
            predicted, _ = actor.predict(np.stack(vectors), deterministic=True)
            np.add.at(blended[:, 1], parents, np.asarray(weights)*predicted[:, 1])
        out.append((raw, blended))
    return tuple(np.concatenate(group) for group in zip(*out))


def memory():
    ram = psutil.virtual_memory()
    result = {"available_bytes": ram.available, "physical_total_bytes": ram.total,
              "paging": psutil.swap_memory()._asdict(), "cpu_percent": psutil.cpu_percent()}
    if __import__("os").name == "nt":
        import ctypes
        class WindowsMemory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
                (name, ctypes.c_uint64) for name in ("total", "available", "commit_limit", "commit_available",
                                                    "virtual_total", "virtual_available", "extended")]
        stat = WindowsMemory()
        stat.length = ctypes.sizeof(stat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            result.update(commit_limit_bytes=stat.commit_limit,
                          commit_available_bytes=stat.commit_available)
    return result


def worker_budget(state):
    free = state["available_bytes"] / 1024**3
    commit = state.get("commit_available_bytes", state["available_bytes"]) / 1024**3
    return 8 if free >= 6 and commit >= 9 else 6 if free >= 4.5 and commit >= 6 else 4


def offline(actor):
    check = {}
    for name, _, _, _ in CASES:
        with np.load(OUTPUT / "capture" / f"{name}.npz", allow_pickle=False) as arrays:
            obs = arrays["policy"]
        a, b = batch_blend(actor, obs)
        delta = b - a
        check[name] = {"frames": len(a), "changed_pedal": int(np.count_nonzero(abs(delta[:, 1]) > .001)),
                       "steer_max": float(np.max(abs(delta[:, 0]))),
                       "abs_pedal_p95": float(np.quantile(abs(delta[:, 1]), .95)),
                       "gas_to_brake": int(np.count_nonzero((a[:, 1] > .02) & (b[:, 1] < -.02))),
                       "brake_to_gas": int(np.count_nonzero((a[:, 1] < -.02) & (b[:, 1] > .02)))}
        if name == "extreme_exact":
            exact = obs
    meta = read(OUTPUT / "capture/extreme_exact.json")
    edge = []
    for step in (2456, 2973):
        rows = np.asarray(meta["frames"][step]["actors"], dtype=np.float32)
        observations = []
        for radius in (50., 50.01):
            modified = rows.copy()
            modified[3, :2] *= radius / np.linalg.norm(rows[3, :2])
            observations.append(pack_traffic(exact[step], modified))
        a, b = batch_blend(actor, np.stack(observations))
        baseline, changed = abs(float(a[1, 1] - a[0, 1])), abs(float(b[1, 1] - b[0, 1]))
        edge.append({"step": step, "raw": a[:, 1].tolist(), "blended": b[:, 1].tolist(),
                     "baseline_jump": baseline, "blended_jump": changed, "reduction": 1 - changed / baseline})
    probes = {}
    for name, row in {
        "adjacent_far": [41., 5., 1., 0., 0., 4.5, 1.8],
        "separating_far": [40., 0., 2., 0., 0., 4.5, 1.8],
        "closing_fast": [32., 0., -12., 0., 0., 4.5, 1.8],
        "near_blocking": [9., 0., 0., 0., 0., 4.5, 1.8],
        "crossing_fast": [29., -8., 0., 8., np.pi/2, 4.5, 1.8],
    }.items():
        obs = pack_traffic(exact[2973], [row])
        a, b = batch_blend(actor, obs[None])
        probes[name] = {"raw": a[0].tolist(), "blended": b[0].tolist(), "delta_pedal": float(b[0, 1] - a[0, 1])}
    a, b = batch_blend(actor, exact[[2455, 2456, 2457, 2972, 2973]])
    timeline = [(i, float(x[1]), float(y[1])) for i, x, y in zip((2455, 2456, 2457, 2972, 2973), a, b)]
    passed = (all(row["reduction"] >= .8 for row in edge)
              and all(abs(probes[key]["delta_pedal"]) <= .02 for key in ("closing_fast", "near_blocking", "crossing_fast"))
              and all(row["steer_max"] == 0 for row in check.values()))
    return {"passed": passed, "edge": edge, "probes": probes, "campaign_frames": check, "exact_actions": timeline}


def main():
    import torch
    from cloud.kaggle.local_risk import LOCK, V2_ROOT, TIERS
    from cloud.kaggle.local_workers import Simulators
    from cloud.kaggle.runtime import load_policy, pool
    from cloud.kaggle.final_evaluation import compare_panels

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", required=True, type=datetime.fromisoformat)
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    lock = read(LOCK)
    path = Path(lock["model"])
    verify(path, lock["model_sha256"])
    evidence = offline(load_policy(path))
    write(ROOT / "offline.json", evidence)
    print("OFFLINE", evidence["passed"], evidence["edge"], flush=True)
    if not evidence["passed"] or datetime.now(args.deadline.tzinfo) >= args.deadline:
        return
    cases = [case for case in CASES if case[0] != "easy"]
    lookup = {phase: {r["geometry_id"]: r for r in pool(phase)} for phase in ("mixed", "eval")}
    original = {}
    for phase in ("idm", "train-gate"):
        for r in read(WORK / "first-window-predictive" / f"{phase}.json")["cases"]:
            if r["variant"] == "raw":
                original[r["geometry"], r["seed"]] = r
    start_memory = memory()
    count = worker_budget(start_memory)
    setup = {"kind": "eval", "smooth_traffic": True, "policy": path, "policy_sha": lock["model_sha256"]}
    with Simulators(count) as sim:
        measured, perf = sim.run([{**setup, "phase": phase, "row": lookup[phase][geometry], "seed": seed}
                                  for _, geometry, seed, phase in cases])
        losses = []
        for result in measured:
            raw = original[result["geometry"], result["seed"]]
            if ((raw["clean"] and not result["clean"])
                or (raw["outcome"] not in ("OUT_OF_ROAD",) and not raw["outcome"].startswith("CRASH_")
                    and (result["outcome"] == "OUT_OF_ROAD" or result["outcome"].startswith("CRASH_")))):
                losses.append(result["geometry"])
        exact = measured[0]
        qualified = not losses and (exact["clean"] or (
            exact["outcome"] == "TIMEOUT" and exact["progress"] >= .42))
        write(ROOT / "sandbox.json", {"cases": measured, "performance": perf, "start_memory": start_memory,
                                     "end_memory": memory(), "configured_workers": count, "regressions": losses,
                                     "eligible_for_confirmation": qualified})
        print("SANDBOX", qualified, [(r["outcome"], r["progress"]) for r in measured], flush=True)
        if qualified and (args.deadline - datetime.now(args.deadline.tzinfo)).total_seconds() > 9 * 60:
            baseline = read(V2_ROOT / "confirmation-05000.json")["cases"]
            jobs = [{**setup, "row": lookup["eval"][r["geometry"]], "seed": r["seed"]}
                    for tier in TIERS for r in baseline[tier]["cases"]]
            outcomes, throughput = sim.run(jobs)
            panels = {tier: {**baseline[tier], "controller": "smooth-traffic-v1",
                             "cases": list(outcomes[i*12:(i+1)*12])} for i, tier in enumerate(TIERS)}
            write(ROOT / "confirmation.json", {"performance": throughput, "cases": panels,
                  "comparison": {tier: compare_panels(baseline[tier], panels[tier]) for tier in TIERS}})


if __name__ == "__main__":
    main()
