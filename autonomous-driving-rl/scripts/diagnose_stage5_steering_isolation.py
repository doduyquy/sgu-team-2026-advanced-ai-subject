"""Short TRAIN-only steering isolation probe for the S5.3G throttle-hold pilot.

This is a diagnostic actuator intervention, not policy evaluation: all arms use
the same fixed positive throttle and case/environment seeds. No learning occurs.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.stage5.checkpoint import load_sac_checkpoint
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.training.stage5.pilot_protocol import diagnostic_schedule, load_pilot_split
from src.training.stage5.train_sampler import SamplerStrategy, TrainCaseSampler
from src.platform import SAFETY_CRITICAL_FLAGS


PILOT_ROOT = ROOT / "runs/stage5_pilot/s5_3g_throttle_hold5_seed101_20261115"
CHECKPOINT = PILOT_ROOT / "checkpoints/step_10000"
SPLIT_PATH = ROOT / "results/stage5/pilot_split_v1.json"
OUTPUT = ROOT / "results/stage5/s5_3g_steering_isolation_seed101_20261115"
SCHEDULE_SEED = 5302
MAX_STEPS = 300
THROTTLE = 0.5
STEERING_RATE_LIMIT_PER_STEP = 0.15
LOOKAHEAD_M = 8.0
WHEELBASE_M = 2.5
MAX_STEERING_RAD = math.radians(60.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values):
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.mean(vals) if vals else None


def pure_pursuit_steering(obs: np.ndarray) -> float:
    points = np.asarray(obs[325:385], dtype=np.float32).reshape(20, 3)
    mask = np.asarray(obs[385:405], dtype=np.float32) > 0.5
    valid = points[mask, :2]
    if len(valid) == 0:
        return 0.0
    distances = np.linalg.norm(valid, axis=1)
    candidates = np.flatnonzero(distances >= LOOKAHEAD_M)
    target_index = int(candidates[0]) if len(candidates) else len(valid) - 1
    target = valid[target_index]
    distance = max(float(np.linalg.norm(target)), 1.0)
    angle = math.atan2(float(target[1]), float(target[0]))
    return float(np.clip(2.0 * WHEELBASE_M * math.sin(angle) /
                         (distance * MAX_STEERING_RAD), -1.0, 1.0))


def lane_lateral_deviation(env: Stage5TrainingEnv):
    vehicle = getattr(getattr(env, "_env", None), "agent", None)
    lane = getattr(vehicle, "lane", None)
    position = getattr(vehicle, "position", None)
    if lane is None or position is None:
        return None
    try:
        _, lateral = lane.local_coordinates(position)
        return float(lateral)
    except Exception:
        return None


def choose_cases(split):
    schedule = diagnostic_schedule(split, SCHEDULE_SEED)
    selected = []
    for tier in sorted({row["tier"] for row in schedule}, key=str.lower):
        tier_rows = sorted((r for r in schedule if r["tier"] == tier),
                           key=lambda r: (r["sequence"], r["geometry_id"]))
        chosen, families = [], set()
        for row in tier_rows:
            if row["sequence"] not in families:
                chosen.append(row)
                families.add(row["sequence"])
            if len(chosen) == 2:
                break
        if len(chosen) != 2:
            raise RuntimeError(f"Could not choose two sequence families for {tier}")
        selected.extend(chosen)
    return selected


def run_arm(model, item, arm, repeat_seed):
    sampler = TrainCaseSampler(
        training_root_seed=101,
        strategy=SamplerStrategy.FIXED,
        allowed_geometry_ids=[item["geometry_id"]],
        project_root=ROOT,
    )
    env = Stage5TrainingEnv(sampler=sampler, encoder=Stage5ObservationEncoder(core_only=False),
                           project_root=ROOT)
    case = sampler.get_case_for_geometry_id(item["geometry_id"], item["environment_seed"])
    obs, _ = env.reset(options={"case": case})
    # Shared stochastic-noise seed across A/B for this paired geometry.
    torch.manual_seed(repeat_seed)
    np.random.seed(repeat_seed)
    random.seed(repeat_seed)
    speeds, throttle, steering, rewards, route_deltas, lateral = [], [], [], [], [], []
    start_progress = float(env._last_route_completion)
    done = False
    started = time.perf_counter()
    last_steering = 0.0
    try:
        while not done and len(speeds) < MAX_STEPS:
            if arm in ("A_actor_stochastic_steering", "B_rate_limited_actor_steering"):
                raw, _ = model.predict(obs, deterministic=False)
                requested_steering = float(np.asarray(raw).reshape(-1)[0])
            else:
                requested_steering = pure_pursuit_steering(obs)
            if arm == "B_rate_limited_actor_steering":
                executed_steering = float(np.clip(requested_steering,
                                                  last_steering - STEERING_RATE_LIMIT_PER_STEP,
                                                  last_steering + STEERING_RATE_LIMIT_PER_STEP))
            else:
                executed_steering = float(np.clip(requested_steering, -1.0, 1.0))
            last_steering = executed_steering
            obs, reward, terminated, truncated, info = env.step([executed_steering, THROTTLE])
            speeds.append(float(info.get("speed_km_h", 0.0)))
            steering.append(executed_steering)
            throttle.append(THROTTLE)
            rewards.append(float(reward))
            route_deltas.append(float(info.get("delta_route", 0.0)))
            lateral.append(lane_lateral_deviation(env))
            done = bool(terminated or truncated)
        outcome = env.last_outcome
        actual_terminal = bool(done)
        reason = outcome.primary_reason.value if outcome else "PROBE_CENSORED"
        if not actual_terminal:
            reason = "PROBE_CENSORED"
        final_progress = float(env._last_route_completion)
        record = {
            "arm": arm, "geometry_id": item["geometry_id"], "geometry_sha256": item["geometry_sha256"],
            "tier": item["tier"], "sequence": item["sequence"],
            "geometry_generation_seed": item["geometry_generation_seed"],
            "environment_seed": item["environment_seed"], "repeat_seed": repeat_seed,
            "split": "PILOT_DIAGNOSTIC_INTERNAL_TRAIN", "throttle_override": THROTTLE,
            "max_steps": MAX_STEPS, "episode_steps": len(speeds), "terminal_observed": actual_terminal,
            "outcome": reason, "clean_success": bool(outcome.clean_success) if outcome and actual_terminal else False,
            "safety_failure": (any(bool(outcome.raw_flags.get(flag, False)) for flag in SAFETY_CRITICAL_FLAGS)
                               if outcome and actual_terminal else False),
            "probe_censored": not actual_terminal,
            "initial_route_completion": start_progress, "final_route_completion": final_progress,
            "route_progress_delta": final_progress - start_progress,
            "max_route_completion": float(env.max_route_completion),
            "cumulative_reward_observed": float(sum(rewards)),
            "mean_speed_km_h": mean(speeds), "max_speed_km_h": max(speeds) if speeds else None,
            "mean_abs_lateral_deviation_m": mean(abs(x) for x in lateral if x is not None),
            "mean_steering": mean(steering), "steering_std": float(np.std(steering)) if steering else None,
            "mean_abs_steering_delta": mean(abs(b-a) for a, b in zip(steering, steering[1:])),
            "mean_throttle": mean(throttle), "duration_s": time.perf_counter() - started,
            "diagnostic_label": "actuator intervention; not policy performance or benchmark metric",
        }
        return record
    finally:
        env.close()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    split = load_pilot_split(SPLIT_PATH)
    cases = choose_cases(split)
    model, metadata = load_sac_checkpoint(
        CHECKPOINT, device="cpu",
        expected_encoder_fingerprint=Stage5ObservationEncoder(core_only=False).get_schema_fingerprint(),
    )
    model_sha = metadata["model_sha256"]
    arms = ["A_actor_stochastic_steering", "B_rate_limited_actor_steering", "C_waypoint_pure_pursuit"]
    records = []
    records_path = OUTPUT / "episode_records.jsonl"
    started = time.perf_counter()
    with records_path.open("w", encoding="utf-8", newline="\n") as sink:
        for case_index, item in enumerate(cases):
            # Order arms by case, so simulator resources are closed between every run.
            for arm_index, arm in enumerate(arms):
                seed = 73000 + case_index * 101
                row = run_arm(model, item, arm, seed)
                row["checkpoint_sha256"] = model_sha
                row["training_step"] = int(metadata["total_timesteps"])
                records.append(row)
                sink.write(json.dumps(row, sort_keys=True) + "\n")
                sink.flush()
                print(json.dumps({"completed": len(records), "planned": len(cases)*len(arms),
                                  "case": item["geometry_id"], "arm": arm,
                                  "steps": row["episode_steps"], "outcome": row["outcome"]}), flush=True)

    aggregates = {}
    for arm in arms:
        rows = [r for r in records if r["arm"] == arm]
        terminal = [r for r in rows if r["terminal_observed"]]
        aggregates[arm] = {
            "n_cases": len(rows), "terminal_observed": len(terminal),
            "probe_censored": sum(r["probe_censored"] for r in rows),
            "outcome_counts": {reason: sum(r["outcome"] == reason for r in rows)
                               for reason in sorted({r["outcome"] for r in rows})},
            "mean_route_progress_delta": mean(r["route_progress_delta"] for r in rows),
            "median_route_progress_delta": statistics.median(r["route_progress_delta"] for r in rows),
            "mean_speed_km_h": mean(r["mean_speed_km_h"] for r in rows),
            "mean_max_speed_km_h": mean(r["max_speed_km_h"] for r in rows),
            "mean_abs_lateral_deviation_m": mean(r["mean_abs_lateral_deviation_m"] for r in rows),
            "safety_failure_rate_among_terminal": (sum(r["safety_failure"] for r in terminal)/len(terminal)
                                                    if terminal else None),
        }
    summary = {
        "protocol": "S5.3G steering isolation diagnostic v1",
        "status": "COMPLETE" if len(records) == len(cases)*len(arms) else "INCOMPLETE",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "checkpoint_sha256": model_sha,
        "training_step": int(metadata["total_timesteps"]), "code_commit": metadata.get("git_commit"),
        "pilot_split_sha256": split["split_sha256"], "schedule_seed": SCHEDULE_SEED,
        "case_count": len(cases), "case_schedule": cases,
        "conditions": {
            arms[0]: "SAC stochastic steering sampled every step; throttle overridden to +0.5",
            arms[1]: f"same SAC stochastic steering with slew limit {STEERING_RATE_LIMIT_PER_STEP:.2f}/step; throttle +0.5",
            arms[2]: f"public-observation waypoint pure pursuit, {LOOKAHEAD_M:.1f}m lookahead; throttle +0.5",
        },
        "max_steps_per_case": MAX_STEPS,
        "censoring_rule": "A probe reaching the step cap without simulator terminal outcome is PROBE_CENSORED, never TIMEOUT.",
        "evaluation_isolation": "No optimizer, weight update, normalization update, replay insertion, VALIDATION, or TEST access.",
        "arm_aggregates": aggregates, "wall_time_s": time.perf_counter() - started,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "aggregates": aggregates,
                      "wall_time_s": summary["wall_time_s"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
