"""S5.3K-K1: paired initial-throttle intervention with frozen stochastic SAC continuation."""
from __future__ import annotations

import argparse
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
import stable_baselines3
import gymnasium

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.platform import SAFETY_CRITICAL_FLAGS, get_action_adapter
from src.training.stage5.checkpoint import load_sac_checkpoint
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.training.stage5.pilot_protocol import load_pilot_split
from src.training.stage5.train_sampler import SamplerStrategy, TrainCaseSampler

RUN_ID = "s5_3g_throttle_hold5_seed101_20261115"
CHECKPOINT = ROOT / f"runs/stage5_pilot/{RUN_ID}/checkpoints/step_10000"
SPLIT_PATH = ROOT / "results/stage5/pilot_split_v1.json"
SCHEDULE_PATH = ROOT / "results/stage5/s5_3g_steering_isolation_seed101_20261115/summary.json"
OUTPUT = ROOT / "results/stage5/s5_3k_action_intervention_seed101_20261115"
RAW = ROOT / "runs/stage5_pilot/s5_3k_action_intervention_seed101_20261115"
CONDITIONS = ("brake_q15", "near_neutral_q50", "throttle_q85")
ACTION_SAMPLES = 512
BOOTSTRAP_SAMPLES = 64
BOOTSTRAP_BATCH = 256


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mean(values):
    seq = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.mean(seq) if seq else None


def fingerprint(model):
    h = hashlib.sha256()
    for key, value in sorted(model.policy.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def sample_policy_actions(model, obs, n, seed):
    torch.manual_seed(seed)
    obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=model.device).reshape(1, -1)
    repeated = obs_t.expand(n, -1)
    with torch.no_grad():
        actions, log_prob = model.actor.action_log_prob(repeated)
    return actions.detach().cpu().numpy().astype(np.float32), log_prob.reshape(-1).detach().cpu().numpy().astype(np.float32)


def choose_supported_throttles(samples):
    throttle = samples[:, 1]
    candidates = {}
    for label, quantile in zip(CONDITIONS, (.15, .50, .85)):
        target = float(np.quantile(throttle, quantile))
        candidates[label] = int(np.argmin(np.abs(throttle - target)))
    values = {label: float(samples[idx, 1]) for label, idx in candidates.items()}
    if not (values[CONDITIONS[0]] < -0.15 and abs(values[CONDITIONS[1]]) < 0.30 and values[CONDITIONS[2]] > 0.15):
        raise RuntimeError(f"Sampled stochastic policy does not support brake/near-neutral/throttle conditions: {values}")
    return candidates, values


def q_at_action(model, obs, action):
    obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=model.device).reshape(1, -1)
    act_t = torch.as_tensor(np.asarray(action, dtype=np.float32), device=model.device).reshape(1, -1)
    with torch.no_grad():
        q1, q2 = model.critic(obs_t, act_t)
    return float(q1.item()), float(q2.item())


def timeout_bootstrap(model, obs, alpha):
    soft_values = []
    for start in range(0, BOOTSTRAP_SAMPLES, BOOTSTRAP_BATCH):
        count = min(BOOTSTRAP_BATCH, BOOTSTRAP_SAMPLES - start)
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=model.device).reshape(1, -1).expand(count, -1)
        with torch.no_grad():
            acts, log_prob = model.actor.action_log_prob(obs_t)
            q1, q2 = model.critic_target(obs_t, acts)
            soft_values.extend((torch.minimum(q1, q2) - alpha * log_prob.reshape(-1, 1)).reshape(-1).cpu().numpy().tolist())
    return mean(soft_values), float(np.std(soft_values))


def get_case_subset(schedule):
    selected = []
    for tier in ("Easy", "Medium", "Hard", "Extreme"):
        rows = [r for r in schedule if r["tier"].lower() == tier.lower()]
        if not rows:
            raise RuntimeError(f"No fixed diagnostic case for tier {tier}")
        selected.append(sorted(rows, key=lambda r: (r["sequence"], r["geometry_id"]))[0])
    return selected


def main():
    parser = argparse.ArgumentParser(description="Run one paired K1 or S5.3L Soft-Q calibration replicate.")
    parser.add_argument("--replicate-id", type=int, default=0,
                        help="Independent stochastic continuation replicate; 0 preserves the original K1 run path.")
    parser.add_argument("--run-id", default=None,
                        help="Optional results/runs namespace. Required with a nonzero replicate ID.")
    args = parser.parse_args()
    if args.replicate_id < 0:
        raise ValueError("replicate-id must be non-negative")
    if args.replicate_id and not args.run_id:
        raise ValueError("run-id is required for nonzero replicate IDs to prevent artifact overwrite")
    global OUTPUT, RAW
    if args.run_id:
        if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in args.run_id):
            raise ValueError("run-id may contain only letters, digits, underscore, and hyphen")
        leaf = f"replicate_{args.replicate_id:02d}" if args.replicate_id else "replicate_00"
        OUTPUT = ROOT / "results/stage5" / args.run_id / leaf
        RAW = ROOT / "runs/stage5_pilot" / args.run_id / leaf
    OUTPUT.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    if any(OUTPUT.iterdir()) or any(RAW.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty replicate artifacts: {OUTPUT} / {RAW}")
    split = load_pilot_split(SPLIT_PATH)
    schedule_all = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))["case_schedule"]
    cases = get_case_subset(schedule_all)
    diagnostic_ids = {r["geometry_id"] for r in split["pilot_diagnostic"]}
    if any(r["geometry_id"] not in diagnostic_ids for r in cases):
        raise RuntimeError("K1 must remain inside the locked internal TRAIN diagnostic split")
    encoder = Stage5ObservationEncoder(core_only=False)
    model, metadata = load_sac_checkpoint(CHECKPOINT, device="cpu",
        expected_encoder_fingerprint=encoder.get_schema_fingerprint())
    model.policy.set_training_mode(False)
    checkpoint_sha = metadata["model_sha256"]
    weights_before = fingerprint(model)
    update_count_before = int(model._n_updates)
    alpha = float(model.log_ent_coef.detach().exp().item()) if getattr(model, "log_ent_coef", None) is not None else float(model.ent_coef_tensor.item())
    adapter = get_action_adapter("continuous_box2_v1")
    sampler = TrainCaseSampler(training_root_seed=101, strategy=SamplerStrategy.TIER_ROUND_ROBIN,
        allowed_geometry_ids=[r["geometry_id"] for r in split["pilot_diagnostic"]], project_root=ROOT)
    env = Stage5TrainingEnv(sampler=sampler, encoder=encoder, project_root=ROOT)
    rows = []
    traces_path = RAW / "transition_records.jsonl"
    records_path = OUTPUT / "episode_records.jsonl"
    parity_failures = []
    planned = len(cases) * len(CONDITIONS)
    completed = 0
    started_all = time.perf_counter()
    try:
        with records_path.open("w", encoding="utf-8", newline="\n") as ep_sink, traces_path.open("w", encoding="utf-8", newline="\n") as trace_sink:
            for case_idx, item in enumerate(cases):
                case = sampler.get_case_for_geometry_id(item["geometry_id"], item["environment_seed"])
                # Establish one reset observation and policy-supported throttle samples.
                torch.manual_seed(340000 + case_idx)
                np.random.seed(340000 + case_idx); random.seed(340000 + case_idx)
                initial_obs, reset_info = env.reset(options={"case": case})
                initial_obs = np.asarray(initial_obs, dtype=np.float32).copy()
                initial_hash = sha(initial_obs.tobytes())
                det_steer_action, _ = model.predict(initial_obs, deterministic=True)
                fixed_steering = float(np.asarray(det_steer_action).reshape(-1)[0])
                sampled_actions, sampled_logp = sample_policy_actions(model, initial_obs, ACTION_SAMPLES, 350000 + case_idx)
                selected_indices, throttle_values = choose_supported_throttles(sampled_actions)

                for condition_idx, condition in enumerate(CONDITIONS):
                    # Every intervention starts from a fresh reset with identical geometry/env seed.
                    obs, info_reset = env.reset(options={"case": case})
                    obs = np.asarray(obs, dtype=np.float32).copy()
                    reset_hash = sha(obs.tobytes())
                    exact_reset_match = bool(reset_hash == initial_hash and np.array_equal(obs, initial_obs))
                    if not exact_reset_match:
                        parity_failures.append({"geometry_id": item["geometry_id"], "condition": condition,
                            "initial_observation_sha256": initial_hash, "branch_observation_sha256": reset_hash})
                        break

                    candidate_index = selected_indices[condition]
                    intervention = [fixed_steering, throttle_values[condition]]
                    canonical = adapter.to_canonical(intervention)
                    q1, q2 = q_at_action(model, obs, [canonical.steering, canonical.throttle_brake])
                    rollout_seed = 810000 + case_idx * 31 + args.replicate_id * 1000003
                    torch.manual_seed(rollout_seed)
                    np.random.seed(rollout_seed)
                    random.seed(rollout_seed)
                    discounted_reward_return = 0.0
                    discounted_soft_return = 0.0
                    rewards = []; speeds = []; controls = []; latencies = []
                    trace_rows = []
                    done = False
                    initial_progress = float(info_reset.get("initial_route_completion", env._last_route_completion))
                    start = time.perf_counter()
                    step_index = 0
                    last_soft_bootstrap = None
                    while not done and step_index < int(case.horizon_steps) + 1:
                        obs_before = np.asarray(obs, dtype=np.float32).copy()
                        inference_start = time.perf_counter_ns()
                        if step_index == 0:
                            action = np.asarray(intervention, dtype=np.float32)
                            log_prob = None  # Q(s,a) has no entropy term on the fixed initial action.
                            raw_actor_action = sampled_actions[candidate_index].astype(float).tolist()
                        else:
                            action_t, logp_t = sample_policy_actions(model, obs_before, 1,
                                810000 + case_idx * 31 + args.replicate_id * 1000003 + step_index * 104729)
                            action = action_t[0]
                            log_prob = float(logp_t[0])
                            raw_actor_action = action.astype(float).tolist()
                        latencies.append((time.perf_counter_ns() - inference_start) / 1e6)
                        canonical = adapter.to_canonical(action)
                        executed = [float(canonical.steering), float(canonical.throttle_brake)]
                        obs, reward, terminated, truncated, info = env.step(executed)
                        step_index += 1
                        done = bool(terminated or truncated)
                        rewards.append(float(reward)); speeds.append(float(info.get("speed_km_h", 0.0)))
                        controls.append(executed)
                        if step_index == 1:
                            discounted_reward_return += float(reward)
                            discounted_soft_return += float(reward)
                        else:
                            discounted_reward_return += (float(model.gamma) ** (step_index - 1)) * float(reward)
                            discounted_soft_return += (float(model.gamma) ** (step_index - 1)) * (float(reward) - alpha * float(log_prob))
                        outcome = env.last_outcome
                        trace = {
                            "geometry_id": item["geometry_id"], "environment_seed": item["environment_seed"],
                            "replicate_id": args.replicate_id,
                            "condition": condition, "step_index": step_index,
                            "observation": obs_before.tolist(), "actor_action": raw_actor_action,
                            "executed_action": executed, "action_log_prob": log_prob,
                            "reward": float(reward), "delta_route": float(info.get("delta_route", 0.0)),
                            "route_completion": float(info.get("route_completion", 0.0)),
                            "speed_km_h": float(info.get("speed_km_h", 0.0)),
                            "terminated": bool(terminated), "truncated": bool(truncated),
                            "outcome": outcome.primary_reason.value if outcome and done else None,
                        }
                        trace_rows.append(trace)
                        done_reason = outcome.primary_reason.value if outcome and done else None
                        if done and done_reason == "TIMEOUT":
                            last_soft_bootstrap, _ = timeout_bootstrap(model, obs, alpha)
                        if not done and step_index >= int(case.horizon_steps):
                            break

                    outcome = env.last_outcome
                    full_episode = bool(done)
                    outcome_name = outcome.primary_reason.value if outcome and full_episode else "PROBE_CENSORED"
                    safety_failure = (any(bool(outcome.raw_flags.get(flag, False)) for flag in SAFETY_CRITICAL_FLAGS)
                                      if outcome and full_episode else False)
                    bootstrapped_soft_return = None
                    if full_episode and outcome_name == "TIMEOUT" and last_soft_bootstrap is not None:
                        bootstrapped_soft_return = discounted_soft_return + (float(model.gamma) ** step_index) * last_soft_bootstrap
                    record = {
                        "checkpoint_sha256": checkpoint_sha, "training_step": 10000,
                        "replicate_id": args.replicate_id,
                        "stochastic_continuation_seed": 810000 + case_idx * 31 + args.replicate_id * 1000003,
                        "geometry_id": item["geometry_id"], "geometry_sha256": item["geometry_sha256"],
                        "tier": item["tier"], "sequence": item["sequence"],
                        "geometry_generation_seed": item["geometry_generation_seed"],
                        "environment_seed": item["environment_seed"], "case_id": case.case_id,
                        "condition": condition, "initial_observation_sha256": initial_hash,
                        "reset_parity_verified": exact_reset_match,
                        "initial_raw_policy_sample": sampled_actions[candidate_index].astype(float).tolist(),
                        "initial_steering_fixed_to_deterministic_actor": fixed_steering,
                        "initial_throttle_value": throttle_values[condition],
                        "initial_sampled_throttle_quantile": {"brake_q15": 0.15, "near_neutral_q50": 0.50, "throttle_q85": 0.85}[condition],
                        "initial_action_source_log_prob": float(sampled_logp[candidate_index]),
                        "initial_q1": q1, "initial_q2": q2, "initial_min_q": min(q1, q2),
                        "episode_steps": step_index, "episode_completed": full_episode,
                        "outcome": outcome_name, "clean_success": bool(outcome.clean_success) if outcome and full_episode else False,
                        "safety_failure": safety_failure,
                        "probe_censored": not full_episode,
                        "initial_route_completion": initial_progress,
                        "final_route_completion": float(env._last_route_completion),
                        "route_progress_delta": float(env._last_route_completion) - initial_progress,
                        "max_route_completion": float(env.max_route_completion),
                        "discounted_reward_return_observed": discounted_reward_return,
                        "discounted_soft_return_observed": discounted_soft_return,
                        "timeout_soft_bootstrap_value": last_soft_bootstrap,
                        "timeout_bootstrapped_soft_return_estimate": bootstrapped_soft_return,
                        "mean_speed_km_h": mean(speeds), "max_speed_km_h": max(speeds) if speeds else None,
                        "stationary_fraction": mean(float(x < 1.0) for x in speeds),
                        "mean_throttle": mean(x[1] for x in controls),
                        "mean_brake": mean(max(0.0, -x[1]) for x in controls),
                        "mean_abs_steering": mean(abs(x[0]) for x in controls),
                        "mean_inference_latency_ms": mean(latencies), "duration_s": time.perf_counter() - start,
                        "diagnostic_label": "one-step throttle intervention then frozen stochastic SAC; internal TRAIN only",
                    }
                    rows.append(record)
                    ep_sink.write(json.dumps(record, sort_keys=True) + "\n"); ep_sink.flush()
                    for trace_row in trace_rows:
                        trace_sink.write(json.dumps(trace_row, sort_keys=True) + "\n")
                    trace_sink.flush()
                    completed += 1
                    print(json.dumps({"completed": completed, "planned": planned,
                        "geometry": item["geometry_id"], "condition": condition,
                        "steps": step_index, "outcome": outcome_name,
                        "route_progress_delta": record["route_progress_delta"]}), flush=True)
                if parity_failures:
                    break
    finally:
        env.close()

    weights_after = fingerprint(model)
    update_count_after = int(model._n_updates)
    # Paired contrasts only use geometries with all three arms and verified reset-state equality.
    complete_cases = []
    for item in cases:
        case_rows = [r for r in rows if r["geometry_id"] == item["geometry_id"]]
        if len(case_rows) == len(CONDITIONS) and all(r["reset_parity_verified"] for r in case_rows):
            complete_cases.append(item["geometry_id"])
    paired = {}
    for metric in ("route_progress_delta", "discounted_reward_return_observed", "discounted_soft_return_observed"):
        contrasts = {}
        for left, right in (("near_neutral_q50", "brake_q15"), ("throttle_q85", "brake_q15"), ("throttle_q85", "near_neutral_q50")):
            diffs = []
            for geometry_id in complete_cases:
                lm = next(r[metric] for r in rows if r["geometry_id"] == geometry_id and r["condition"] == left)
                rm = next(r[metric] for r in rows if r["geometry_id"] == geometry_id and r["condition"] == right)
                diffs.append(lm - rm)
            contrasts[f"{left}_minus_{right}"] = {"n": len(diffs), "mean": mean(diffs),
                "median": statistics.median(diffs) if diffs else None,
                "positive": sum(x > 0 for x in diffs), "negative": sum(x < 0 for x in diffs), "ties": sum(x == 0 for x in diffs)}
        paired[metric] = contrasts
    status = "COMPLETE" if len(rows) == planned and not parity_failures and len(complete_cases) == len(cases) else "INCOMPLETE_OR_PARITY_FAILED"
    summary = {
        "protocol": ("S5.3L-L1 paired soft-Q calibration replicate v1" if args.run_id and args.run_id.startswith("s5_3l") else "S5.3K-K1 paired throttle action intervention v1"),
        "status": status, "replicate_id": args.replicate_id,
        "checkpoint_sha256": checkpoint_sha, "training_step": 10000,
        "gamma": float(model.gamma), "entropy_coefficient": alpha,
        "stable_baselines3_version": stable_baselines3.__version__, "gymnasium_version": gymnasium.__version__,
        "checkpoint_source_commit": metadata.get("git_commit"), "training_root_seed": 101,
        "pilot_split_sha256": split["split_sha256"], "cases": cases,
        "conditions": list(CONDITIONS), "future_policy": "same frozen stochastic SAC, sampled each decision; no throttle persistence intervention",
        "initial_action_design": "At each reset observation, draw 512 stochastic SAC actions; choose throttle samples nearest conditional q15/q50/q85. Set initial steering to the same deterministic actor steering in all three conditions. Continue with stochastic SAC after this one action.",
        "initial_state_parity": {"verified_case_count": len(complete_cases), "planned_case_count": len(cases), "failures": parity_failures},
        "paired_contrasts": paired,
        "censoring_rule": "No artificial episode timeout; each rollout runs to Platform outcome/horizon. If no terminal by route-aware horizon plus one step, record PROBE_CENSORED and do not call it timeout.",
        "soft_return_definition": "For initial fixed action a0: r0 + sum_{t=1}^{T-1} gamma^t*(r_t-alpha*log_pi(a_t|s_t)); timeout adds gamma^T*E[min(Q1_target,Q2_target)-alpha*log_pi(a_T|s_T)] at terminal observation. True terminals have no bootstrap. This matches SB3 SAC's soft Bellman recursion; the timeout tail is separately labeled and not empirical.",
        "timeout_semantics": "TIMEOUT episodes have a separately reported finite observed soft return and an SB3-style target-critic soft bootstrap estimate; do not treat the bootstrapped estimate as empirical return.",
        "read_only": {"policy_weights_unchanged": weights_before == weights_after,
            "policy_fingerprint_before": weights_before, "policy_fingerprint_after": weights_after,
            "optimizer_updates_before": update_count_before, "optimizer_updates_after": update_count_after,
            "normalization_update": False, "persistent_replay_write": False,
            "validation_test_access": False},
        "episodes": rows,
        "limitations": ["Small screening sample: four diagnostic TRAIN geometries, one environment seed each.",
            "Initial steering is fixed to deterministic actor mean; throttle values are chosen from the stochastic actor's conditional throttle samples, so the pair is an explicit intervention rather than a sampled joint policy action.",
            "A single rollout per condition/case has high stochastic-return variance.",
            "No replay buffer or optimizer state was changed; Q/return agreement here does not certify global critic calibration."],
        "raw_step_records": str((RAW / "transition_records.jsonl").relative_to(ROOT)),
        "wall_time_s": time.perf_counter() - started_all,
    }
    (RAW / "summary_internal.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["raw_step_records_sha256"] = hashlib.sha256((RAW / "transition_records.jsonl").read_bytes()).hexdigest()
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT / "paired_contrasts.json").write_text(json.dumps(paired, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_title = "S5.3L — Soft-Q Calibration replicate" if args.run_id and args.run_id.startswith("s5_3l") else "S5.3K — K1 paired initial-throttle intervention"
    lines = [f"# {report_title}", "",
        f"Status: **{status}**. Frozen SAC checkpoint; no training updates.", "",
        "At each of four fixed internal TRAIN diagnostic reset states, 512 stochastic SAC actions were sampled. Three throttle values near conditional q15/q50/q85 were selected for brake / near-neutral / throttle. Initial steering was fixed to the deterministic actor output in all conditions. After this single intervention, the same frozen stochastic SAC policy sampled both action dimensions at every decision until a real Platform episode outcome/horizon.", "",
        "Reset observations were checked for exact array equality before each condition. Only exact-parity triplets are included in paired contrasts. No external step cap was used; any incomplete episode is explicitly censored.", "",
        "| Tier | Condition | Initial throttle | Q1 / Q2 / minQ | Steps | Outcome | Route progress delta | Discounted reward return | Discounted soft return | Speed km/h |", "|---|---|---:|---|---:|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['tier']} | {r['condition']} | {r['initial_throttle_value']:+.3f} | {r['initial_q1']:.3f} / {r['initial_q2']:.3f} / {r['initial_min_q']:.3f} | {r['episode_steps']} | {r['outcome']} | {r['route_progress_delta']:+.4f} | {r['discounted_reward_return_observed']:+.4f} | {r['discounted_soft_return_observed']:+.4f} | {r['mean_speed_km_h']:.2f} |")
    lines += ["", "## Paired differences", "", "Positive values favor the left condition. n is the number of exact-reset-parity geometry pairs.", "", "| Metric | Contrast | n | Mean | Median | Higher / lower / tie |", "|---|---|---:|---:|---:|---|"]
    for metric, contrasts in paired.items():
        for contrast, values in contrasts.items():
            if values["n"]:
                lines.append(f"| {metric} | {contrast} | {values['n']} | {values['mean']:+.4f} | {values['median']:+.4f} | {values['positive']} / {values['negative']} / {values['ties']} |")
            else:
                lines.append(f"| {metric} | {contrast} | 0 | n/a | n/a | n/a |")
    lines += ["", "## Interpretation limits", "",
        "The bootstrapped timeout soft return uses the target critic and is reported separately from the realized episode return; it is not independent empirical evidence for critic calibration. Initial Q1/Q2 are critic predictions. One paired seed per geometry is a small diagnostic screen, not a statistical conclusion. If any row is PROBE_CENSORED, its observed return is partial and excluded from full-episode claims.", "",
        f"Exact reset parity: {len(complete_cases)}/{len(cases)} case triplets. Policy weights unchanged: `{weights_before == weights_after}`; optimizer updates {update_count_before} → {update_count_after}. No VALIDATION/TEST access. Checkpoint SHA-256 `{checkpoint_sha}`; split SHA-256 `{split['split_sha256']}`.", ""]
    (OUTPUT / "CONCLUSION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "completed": len(rows), "planned": planned,
        "parity": summary["initial_state_parity"], "paired": paired,
        "read_only": summary["read_only"], "artifact": str(OUTPUT / "CONCLUSION.md")}, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
