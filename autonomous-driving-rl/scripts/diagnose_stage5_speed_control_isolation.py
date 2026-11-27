"""S5.3H: compare Stage 5 steering choices under a shared speed controller.

TRAIN-only development diagnostic. SAC weights, optimizer, normalizers, and
replay are read-only. This is not a benchmark evaluation.
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

from src.platform import SAFETY_CRITICAL_FLAGS, get_action_adapter
from src.training.stage5.checkpoint import load_sac_checkpoint
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.training.stage5.pilot_protocol import load_pilot_split
from src.training.stage5.train_sampler import SamplerStrategy, TrainCaseSampler

PREVIOUS = ROOT / "results/stage5/s5_3g_steering_isolation_seed101_20261115"
CHECKPOINT = ROOT / "runs/stage5_pilot/s5_3g_throttle_hold5_seed101_20261115/checkpoints/step_10000"
SPLIT_PATH = ROOT / "results/stage5/pilot_split_v1.json"
OUTPUT = ROOT / "results/stage5/s5_3h_speed_control_seed101_20261115"
MAX_STEPS = 300
TARGET_SPEED_KMH = 12.0
SPEED_KP = 0.08
THROTTLE_MIN, THROTTLE_MAX = -0.45, 0.60
LOOKAHEAD_M, WHEELBASE_M = 8.0, 2.5
MAX_STEERING_RAD = math.radians(60.0)
STOCHASTIC_SAMPLES = 32


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def policy_fingerprint(model) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.policy.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def mean(values):
    seq = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.mean(seq) if seq else None


def pure_pursuit(obs: np.ndarray) -> float:
    points = np.asarray(obs[325:385], dtype=np.float32).reshape(20, 3)
    mask = np.asarray(obs[385:405], dtype=np.float32) > 0.5
    points = points[mask, :2]
    if not len(points):
        return 0.0
    distances = np.linalg.norm(points, axis=1)
    candidates = np.flatnonzero(distances >= LOOKAHEAD_M)
    target = points[int(candidates[0])] if len(candidates) else points[-1]
    distance = max(float(np.linalg.norm(target)), 1.0)
    angle = math.atan2(float(target[1]), float(target[0]))
    return float(np.clip(2 * WHEELBASE_M * math.sin(angle) /
                         (distance * MAX_STEERING_RAD), -1.0, 1.0))


def lateral_offset(env):
    vehicle = getattr(getattr(env, "_env", None), "agent", None)
    lane, pos = getattr(vehicle, "lane", None), getattr(vehicle, "position", None)
    if lane is None or pos is None:
        return None
    try:
        return float(lane.local_coordinates(pos)[1])
    except Exception:
        return None


def actor_action_audit(model, obs, adapter, seed):
    det, _ = model.predict(obs, deterministic=True)
    det = np.asarray(det, dtype=np.float32).reshape(2)
    det_canonical = adapter.to_canonical(det).to_numpy().astype(float).tolist()
    torch.manual_seed(seed)
    samples = []
    for _ in range(STOCHASTIC_SAMPLES):
        action, _ = model.predict(obs, deterministic=False)
        raw = np.asarray(action, dtype=np.float32).reshape(2)
        canonical = adapter.to_canonical(raw).to_numpy().astype(float).tolist()
        samples.append({"raw": raw.astype(float).tolist(), "canonical": canonical})
    return {
        "deterministic_raw": det.astype(float).tolist(), "deterministic_canonical": det_canonical,
        "stochastic_samples": samples,
        "stochastic_mean_raw": np.mean([r["raw"] for r in samples], axis=0).astype(float).tolist(),
        "stochastic_std_raw": np.std([r["raw"] for r in samples], axis=0).astype(float).tolist(),
        "stochastic_throttle_positive_fraction": float(np.mean([r["raw"][1] > 0 for r in samples])),
        "adapter_identity_for_probe": bool(np.allclose(det, det_canonical, atol=1e-7)),
    }


def run_case(model, item, arm, seed, audit):
    sampler = TrainCaseSampler(training_root_seed=101, strategy=SamplerStrategy.FIXED,
        allowed_geometry_ids=[item["geometry_id"]], project_root=ROOT)
    env = Stage5TrainingEnv(sampler=sampler, encoder=Stage5ObservationEncoder(core_only=False), project_root=ROOT)
    case = sampler.get_case_for_geometry_id(item["geometry_id"], item["environment_seed"])
    obs, _ = env.reset(options={"case": case})
    adapter = get_action_adapter("continuous_box2_v1")
    if audit is not None:
        audit.update(actor_action_audit(model, obs, adapter, seed + 17000))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    speeds, steering, throttles, raw_throttles, rewards, lateral = [], [], [], [], [], []
    initial_progress = float(env._last_route_completion)
    done = False
    started = time.perf_counter()
    try:
        while not done and len(speeds) < MAX_STEPS:
            if arm == "C_waypoint_pure_pursuit":
                steer = pure_pursuit(obs)
                actor_throttle = None
            else:
                raw, _ = model.predict(obs, deterministic=(arm == "A_sac_deterministic"))
                raw = np.asarray(raw, dtype=np.float32).reshape(2)
                steer, actor_throttle = float(raw[0]), float(raw[1])
            # Shared proportional speed controller replaces actor throttle in every arm.
            speed_now = speeds[-1] if speeds else 0.0
            control_throttle = float(np.clip(SPEED_KP * (TARGET_SPEED_KMH - speed_now), THROTTLE_MIN, THROTTLE_MAX))
            canonical = adapter.to_canonical([steer, control_throttle])
            executed = [float(canonical.steering), float(canonical.throttle_brake)]
            obs, reward, terminated, truncated, info = env.step(executed)
            speeds.append(float(info.get("speed_km_h", 0.0)))
            steering.append(executed[0]); throttles.append(executed[1])
            raw_throttles.append(actor_throttle); rewards.append(float(reward))
            lateral.append(lateral_offset(env))
            done = bool(terminated or truncated)
        outcome = env.last_outcome
        terminal = bool(done)
        reason = outcome.primary_reason.value if outcome and terminal else "PROBE_CENSORED"
        safety = any(bool(outcome.raw_flags.get(flag, False)) for flag in SAFETY_CRITICAL_FLAGS) if outcome and terminal else False
        return {
            "arm": arm, "geometry_id": item["geometry_id"], "geometry_sha256": item["geometry_sha256"],
            "tier": item["tier"], "sequence": item["sequence"],
            "geometry_generation_seed": item["geometry_generation_seed"],
            "environment_seed": item["environment_seed"], "rollout_seed": seed,
            "split": "PILOT_DIAGNOSTIC_INTERNAL_TRAIN", "checkpoint_training_step": 10000,
            "episode_steps": len(speeds), "terminal_observed": terminal, "outcome": reason,
            "probe_censored": not terminal, "safety_failure": safety,
            "clean_success": bool(outcome.clean_success) if outcome and terminal else False,
            "initial_route_completion": initial_progress,
            "final_route_completion": float(env._last_route_completion),
            "route_progress_delta": float(env._last_route_completion) - initial_progress,
            "max_route_completion": float(env.max_route_completion),
            "mean_speed_km_h": mean(speeds), "max_speed_km_h": max(speeds) if speeds else None,
            "mean_abs_lateral_deviation_m": mean(abs(x) for x in lateral if x is not None),
            "mean_steering": mean(steering), "steering_std": float(np.std(steering)) if steering else None,
            "mean_control_throttle": mean(throttles),
            "mean_actor_raw_throttle": mean(x for x in raw_throttles if x is not None),
            "fraction_actor_raw_throttle_positive": (mean(float(x > 0) for x in raw_throttles if x is not None)),
            "fraction_speed_over_target_plus_3": mean(float(x > TARGET_SPEED_KMH + 3) for x in speeds),
            "cumulative_reward_observed": sum(rewards), "duration_s": time.perf_counter() - started,
            "initial_action_audit": audit,
            "diagnostic_label": "shared speed-control intervention; not policy benchmark performance",
        }
    finally:
        env.close()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    split = load_pilot_split(SPLIT_PATH)
    previous = json.loads((PREVIOUS / "summary.json").read_text(encoding="utf-8"))
    cases = previous["case_schedule"]
    diagnostic_ids = {r["geometry_id"] for r in split["pilot_diagnostic"]}
    if len(cases) != 8 or any(r["geometry_id"] not in diagnostic_ids for r in cases):
        raise ValueError("S5.3H cases must be the same eight internal diagnostic TRAIN geometries as S5.3G")
    encoder = Stage5ObservationEncoder(core_only=False)
    model, metadata = load_sac_checkpoint(CHECKPOINT, device="cpu", expected_encoder_fingerprint=encoder.get_schema_fingerprint())
    checkpoint_sha = metadata["model_sha256"]
    weights_before = policy_fingerprint(model)
    updates_before = int(getattr(model, "_n_updates", 0))
    replay = getattr(model, "replay_buffer", None)
    replay_size_before = replay.size() if replay is not None else None
    arms = ["A_sac_deterministic", "B_sac_stochastic", "C_waypoint_pure_pursuit"]
    records_path = OUTPUT / "episode_records.jsonl"
    audits_path = OUTPUT / "initial_action_audits.jsonl"
    n_done = 0
    started = time.perf_counter()
    with records_path.open("w", encoding="utf-8", newline="\n") as sink, audits_path.open("w", encoding="utf-8", newline="\n") as audit_sink:
        for case_index, item in enumerate(cases):
            for arm in arms:
                seed = 81000 + case_index * 137
                audit_box = {} if arm == "A_sac_deterministic" else None
                row = run_case(model, item, arm, seed, audit_box)
                row.update({"checkpoint_sha256": checkpoint_sha, "training_root_seed": metadata.get("training_root_seed"),
                            "checkpoint_source_commit": metadata.get("git_commit")})
                sink.write(json.dumps(row, sort_keys=True) + "\n"); sink.flush()
                if audit_box is not None:
                    audit_row = {"geometry_id": item["geometry_id"], "environment_seed": item["environment_seed"],
                                 "rollout_seed": seed, **audit_box}
                    audit_sink.write(json.dumps(audit_row, sort_keys=True) + "\n"); audit_sink.flush()
                n_done += 1
                print(json.dumps({"completed": n_done, "planned": 24, "geometry": item["geometry_id"],
                                  "arm": arm, "steps": row["episode_steps"], "outcome": row["outcome"]}), flush=True)

    weights_after = policy_fingerprint(model)
    updates_after = int(getattr(model, "_n_updates", 0))
    replay_after = getattr(model, "replay_buffer", None)
    replay_size_after = replay_after.size() if replay_after is not None else None
    rows = [json.loads(x) for x in records_path.read_text(encoding="utf-8").splitlines()]
    aggregates = {}
    for arm in arms:
        subset = [r for r in rows if r["arm"] == arm]
        terminal = [r for r in subset if r["terminal_observed"]]
        aggregates[arm] = {
            "n": len(subset), "terminal_count": len(terminal),
            "probe_censored": sum(r["probe_censored"] for r in subset),
            "outcome_counts": {k: sum(r["outcome"] == k for r in subset) for k in sorted({r["outcome"] for r in subset})},
            "mean_route_progress_delta": mean(r["route_progress_delta"] for r in subset),
            "median_route_progress_delta": statistics.median(r["route_progress_delta"] for r in subset),
            "mean_speed_km_h": mean(r["mean_speed_km_h"] for r in subset),
            "mean_max_speed_km_h": mean(r["max_speed_km_h"] for r in subset),
            "mean_abs_lateral_deviation_m": mean(r["mean_abs_lateral_deviation_m"] for r in subset),
            "safety_failure_rate_among_terminal": mean(float(r["safety_failure"]) for r in terminal) if terminal else None,
            "mean_actor_raw_throttle": mean(r["mean_actor_raw_throttle"] for r in subset),
            "mean_control_throttle": mean(r["mean_control_throttle"] for r in subset),
            "fraction_speed_over_target_plus_3": mean(r["fraction_speed_over_target_plus_3"] for r in subset),
        }
    summary = {
        "protocol": "S5.3H shared speed-control isolation v1", "status": "COMPLETE" if n_done == 24 else "INCOMPLETE",
        "checkpoint_sha256": checkpoint_sha, "training_step": int(metadata["total_timesteps"]),
        "checkpoint_source_commit": metadata.get("git_commit"), "pilot_split_sha256": split["split_sha256"],
        "same_case_schedule_as": "s5_3g_steering_isolation_seed101_20261115", "cases": cases,
        "arms": arms,
        "longitudinal_controller": {"type": "proportional speed controller", "target_speed_km_h": TARGET_SPEED_KMH,
            "kp_action_per_km_h": SPEED_KP, "throttle_brake_clip": [THROTTLE_MIN, THROTTLE_MAX],
            "formula": "clip(0.08 * (12.0 - measured_speed_km_h), -0.45, +0.60)"},
        "pure_pursuit": {"lookahead_m": LOOKAHEAD_M, "wheelbase_m": WHEELBASE_M, "steering_rad_limit": MAX_STEERING_RAD},
        "max_steps_per_case": MAX_STEPS,
        "censoring_rule": "300-step cap without simulator terminal outcome is PROBE_CENSORED, never TIMEOUT.",
        "isolation": {"policy_weights_unchanged": weights_before == weights_after,
            "policy_fingerprint_before": weights_before, "policy_fingerprint_after": weights_after,
            "optimizer_updates_before": updates_before, "optimizer_updates_after": updates_after,
            "replay_size_before": replay_size_before, "replay_size_after": replay_size_after,
            "normalization_update": False, "validation_test_access": False},
        "arm_aggregates": aggregates, "wall_time_s": time.perf_counter() - started,
    }
    summary["source_file_sha256"] = {p: sha256(ROOT / p) for p in [
        "scripts/diagnose_stage5_speed_control_isolation.py", "src/training/stage5/env_adapter.py",
        "src/training/stage5/observation_encoder.py", "src/platform/action_adapter.py"]}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# S5.3H — Deterministic policy and speed-control isolation", "",
        f"Status: **{summary['status']}**. 24 probes across the same eight internal TRAIN diagnostic geometries/seeds used in S5.3G. Each arm was capped at {MAX_STEPS} steps; censored probes are not timeouts.", "",
        "All arms used the same proportional longitudinal speed controller; actor throttle was recorded but overridden. The controller target is 12 km/h with `clip(0.08 * (12 - speed_km_h), -0.45, +0.60)`. Arms: deterministic SAC steering, stochastic SAC steering, and public-waypoint pure pursuit (8 m lookahead). This intervention is diagnostic only, not benchmark policy performance.", "",
        "## Results", "", "| Arm | n | Outcomes | Censored | Mean progress delta | Mean speed km/h | Mean abs lateral offset m | Raw actor throttle mean | Speed controller throttle mean |", "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for arm in arms:
        a = aggregates[arm]
        outcomes = ", ".join(f"{k} {v}" for k, v in a["outcome_counts"].items())
        rawt = "n/a" if a["mean_actor_raw_throttle"] is None else f"{a['mean_actor_raw_throttle']:.3f}"
        lines.append(f"| {arm} | {a['n']} | {outcomes} | {a['probe_censored']} | {a['mean_route_progress_delta']:.4f} | {a['mean_speed_km_h']:.2f} | {a['mean_abs_lateral_deviation_m']:.3f} | {rawt} | {a['mean_control_throttle']:.3f} |")
    audits = [json.loads(x) for x in audits_path.read_text(encoding="utf-8").splitlines()]
    lines += ["", "## Same-observation policy action audit", "",
        f"At each of the {len(audits)} reset observations, deterministic action was compared with {STOCHASTIC_SAMPLES} stochastic samples on that exact observation before any step. Across reset states, deterministic actor mean throttle was {mean(r['deterministic_raw'][1] for r in audits):.4f}; mean stochastic-sample throttle was {mean(r['stochastic_mean_raw'][1] for r in audits):.4f}; stochastic positive-throttle fraction averaged {mean(r['stochastic_throttle_positive_fraction'] for r in audits):.3f}. Adapter output matched raw action on each deterministic check: {all(r['adapter_identity_for_probe'] for r in audits)}. Per-case samples are in `initial_action_audits.jsonl`.", "",
        "## Interpretation", "",
        "This compares steering sources under one shared speed controller, so it asks whether steering choice limits progress/safety when throttle is externally stabilized around 12 km/h. It still does not represent the SAC policy acting jointly. Compare terminal outcomes and route progress together; a censored rollout is not safe. The proportional speed controller is a diagnostic intervention and was not tuned as an agent or benchmark baseline.", "",
        f"Read-only checks: policy weights unchanged = `{summary['isolation']['policy_weights_unchanged']}`; optimizer updates {updates_before} -> {updates_after}; replay size {replay_size_before} -> {replay_size_after}. No VALIDATION/TEST access.", "",
        f"Checkpoint SHA-256 `{checkpoint_sha}`; internal split SHA-256 `{split['split_sha256']}`. Full records and source hashes are in adjacent JSON files.", ""]
    (OUTPUT / "CONCLUSION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "aggregates": aggregates, "isolation": summary["isolation"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
