"""S5.3J: action support and read-only soft Bellman residual diagnostic."""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
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

RUN_ID = "s5_3g_throttle_hold5_seed101_20261115"
RUN = ROOT / f"runs/stage5_pilot/{RUN_ID}"
SPLIT_PATH = ROOT / "results/stage5/pilot_split_v1.json"
SCHEDULE_PATH = ROOT / "results/stage5/s5_3g_steering_isolation_seed101_20261115/summary.json"
OUTPUT = ROOT / "results/stage5/s5_3j_critic_reliability_seed101_20261115"
RAW_OUTPUT = ROOT / "runs/stage5_pilot/s5_3j_critic_reliability_seed101_20261115"
MAX_STEPS = 150
NEXT_ACTION_SAMPLES = 8
ACTION_BATCH = 128


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mean(values):
    seq = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.mean(seq) if seq else None


def policy_fingerprint(model) -> str:
    h = hashlib.sha256()
    for key, tensor in sorted(model.policy.state_dict().items()):
        h.update(key.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def action_support_audit():
    rows = [json.loads(line) for line in (RUN / "checkpoints/training_steps.jsonl").read_text(encoding="utf-8").splitlines()]
    steering = np.asarray([float(r["steering"]) for r in rows], dtype=np.float64)
    throttle = np.asarray([float(r["throttle_brake"]) for r in rows], dtype=np.float64)
    gaps = np.abs(np.diff(throttle))
    same_run_lengths = []
    start = 0
    for idx in range(1, len(throttle) + 1):
        if idx == len(throttle) or not np.isclose(throttle[idx], throttle[start], atol=1e-7):
            same_run_lengths.append(idx - start)
            start = idx
    def dist(x):
        return {"min": float(np.min(x)), "q01_q05_q25_q50_q75_q95_q99": np.quantile(x, [.01,.05,.25,.5,.75,.95,.99]).astype(float).tolist(),
                "max": float(np.max(x)), "mean": float(np.mean(x)), "std": float(np.std(x)),
                "fraction_abs_ge_0_95": float(np.mean(np.abs(x) >= .95))}
    throttle_bins = {name: int(count) for name, count in Counter(
        "brake" if x < -0.05 else "throttle" if x > 0.05 else "near_zero" for x in throttle).items()}
    return {
        "n_logged_training_actions": len(rows),
        "source_log_sha256": sha(RUN / "checkpoints/training_steps.jsonl"),
        "training_actions_are_observed_execution_actions": "SAC collector passes its returned actions to env.step; Stage5TrainingEnv reports same canonical action in info. Callback records collector action first. The custom sampler changed actions and replay actions together.",
        "steering_distribution": dist(steering), "throttle_brake_distribution": dist(throttle),
        "throttle_bins": throttle_bins,
        "throttle_hold_blocks": {"count_of_constant_value_runs": len(same_run_lengths),
            "run_length_quantiles": np.quantile(same_run_lengths, [.1,.5,.9,.99]).astype(float).tolist(),
            "max_run_length": max(same_run_lengths)},
        "missing_state_support": ["observations", "next_observations", "behavior log probabilities", "replay indices", "target-Q samples"],
        "replay_buffer_saved": False,
        "interpretation_limit": "This reconstructs marginal executed-action support from the 10k telemetry, not state-conditional replay support or the exact replay buffer contents.",
    }


def collect_probe(model, cases):
    split = load_pilot_split(SPLIT_PATH)
    sampler = TrainCaseSampler(training_root_seed=101, strategy=SamplerStrategy.TIER_ROUND_ROBIN,
        allowed_geometry_ids=[r["geometry_id"] for r in split["pilot_diagnostic"]], project_root=ROOT)
    env = Stage5TrainingEnv(sampler=sampler, encoder=Stage5ObservationEncoder(core_only=False), project_root=ROOT)
    adapter = get_action_adapter("continuous_box2_v1")
    samples = []
    episode_rows = []
    try:
        for case_idx, item in enumerate(cases):
            for mode_idx, mode in enumerate(("deterministic", "stochastic")):
                case = sampler.get_case_for_geometry_id(item["geometry_id"], item["environment_seed"])
                obs, _ = env.reset(options={"case": case})
                torch_seed = 720000 + case_idx * 101 + mode_idx
                torch.manual_seed(torch_seed); np.random.seed(torch_seed); random.seed(torch_seed)
                start = float(env._last_route_completion)
                n = 0; done = False; start_time = time.perf_counter()
                while n < MAX_STEPS and not done:
                    obs_before = np.asarray(obs, dtype=np.float32).copy()
                    raw_action, _ = model.predict(obs_before, deterministic=(mode == "deterministic"))
                    raw_action = np.asarray(raw_action, dtype=np.float32).reshape(2)
                    canonical = adapter.to_canonical(raw_action)
                    executed = np.asarray([canonical.steering, canonical.throttle_brake], dtype=np.float32)
                    obs, reward, terminated, truncated, info = env.step(executed)
                    obs_after = np.asarray(obs, dtype=np.float32).copy()
                    done = bool(terminated or truncated)
                    outcome = env.last_outcome
                    sample = {
                        "observation": obs_before,
                        "action": executed.copy(),
                        "reward": float(reward),
                        "next_observation": obs_after,
                        "terminated": bool(terminated), "truncated": bool(truncated),
                        "bootstrap_done_mask": float(bool(terminated)),
                        "geometry_id": item["geometry_id"], "tier": item["tier"],
                        "environment_seed": int(item["environment_seed"]), "mode": mode,
                        "step_index": n + 1, "speed_km_h": float(info.get("speed_km_h", 0.0)),
                        "delta_route": float(info.get("delta_route", 0.0)),
                        "primary_reason": outcome.primary_reason.value if outcome and done else None,
                        "safety_failure": any(bool(outcome.raw_flags.get(flag, False)) for flag in SAFETY_CRITICAL_FLAGS) if outcome and done else False,
                    }
                    samples.append(sample); n += 1
                last_outcome = env.last_outcome
                episode_rows.append({
                    "geometry_id": item["geometry_id"], "tier": item["tier"], "environment_seed": item["environment_seed"],
                    "mode": mode, "steps": n, "probe_censored": not done,
                    "outcome": last_outcome.primary_reason.value if last_outcome and done else "PROBE_CENSORED",
                    "safety_failure": any(bool(last_outcome.raw_flags.get(flag, False)) for flag in SAFETY_CRITICAL_FLAGS) if last_outcome and done else False,
                    "initial_route_completion": start, "final_route_completion": float(env._last_route_completion),
                    "route_progress_delta": float(env._last_route_completion) - start,
                    "duration_s": time.perf_counter() - start_time,
                })
                print(json.dumps({"completed_episodes": len(episode_rows), "planned": 16,
                    "geometry": item["geometry_id"], "mode": mode, "steps": n,
                    "outcome": episode_rows[-1]["outcome"]}), flush=True)
    finally:
        env.close()
    return samples, episode_rows


def compute_soft_bellman(model, samples, alpha, gamma):
    n = len(samples)
    arrays = {
        "observations": np.stack([s["observation"] for s in samples]).astype(np.float32),
        "actions": np.stack([s["action"] for s in samples]).astype(np.float32),
        "rewards": np.asarray([s["reward"] for s in samples], dtype=np.float32),
        "next_observations": np.stack([s["next_observation"] for s in samples]).astype(np.float32),
        "done_masks": np.asarray([s["bootstrap_done_mask"] for s in samples], dtype=np.float32),
    }
    residuals = np.zeros((n, 6), dtype=np.float64)
    # Columns: Q1, Q2, minQ, target mean, target MC SD, minQ - target mean.
    for start in range(0, n, ACTION_BATCH):
        end = min(start + ACTION_BATCH, n)
        obs = torch.as_tensor(arrays["observations"][start:end], device=model.device)
        act = torch.as_tensor(arrays["actions"][start:end], device=model.device)
        rew = torch.as_tensor(arrays["rewards"][start:end], device=model.device).reshape(-1, 1)
        nxt = torch.as_tensor(arrays["next_observations"][start:end], device=model.device)
        done = torch.as_tensor(arrays["done_masks"][start:end], device=model.device).reshape(-1, 1)
        with torch.no_grad():
            q1, q2 = model.critic(obs, act)
            repeated_nxt = nxt.repeat_interleave(NEXT_ACTION_SAMPLES, dim=0)
            nxt_actions, nxt_log_prob = model.actor.action_log_prob(repeated_nxt)
            tq = torch.cat(model.critic_target(repeated_nxt, nxt_actions), dim=1).min(dim=1, keepdim=True).values
            soft_samples = (tq - alpha * nxt_log_prob.reshape(-1, 1)).reshape(end-start, NEXT_ACTION_SAMPLES)
            soft_mean = soft_samples.mean(dim=1, keepdim=True)
            soft_sd = soft_samples.std(dim=1, unbiased=False, keepdim=True)
            target = rew + gamma * (1.0 - done) * soft_mean
            minq = torch.minimum(q1, q2)
            block = torch.cat([q1, q2, minq, target, soft_sd, minq-target], dim=1)
            residuals[start:end] = block.cpu().numpy()
    return arrays, residuals


def summarize_residuals(samples, residuals):
    out = {}
    for mode in ("deterministic", "stochastic"):
        indices = [i for i, s in enumerate(samples) if s["mode"] == mode]
        block = residuals[indices]
        out[mode] = {
            "n_transitions": len(indices),
            "q1_mean": mean(block[:,0]), "q2_mean": mean(block[:,1]),
            "mean_abs_q1_q2_disagreement": mean(abs(block[:,0]-block[:,1])),
            "critic_disagreement_q50_q90_q99": np.quantile(abs(block[:,0]-block[:,1]), [.5,.9,.99]).astype(float).tolist(),
            "soft_bellman_target_mean": mean(block[:,3]),
            "next_soft_value_mc_sd_mean_8_samples": mean(block[:,4]),
            "mean_signed_min_q_minus_target": mean(block[:,5]),
            "mean_absolute_bellman_residual": mean(abs(block[:,5])),
            "bellman_residual_q50_q90_q99": np.quantile(abs(block[:,5]), [.5,.9,.99]).astype(float).tolist(),
            "terminated_count": sum(samples[i]["terminated"] for i in indices),
            "timeout_truncated_count": sum(samples[i]["truncated"] and samples[i]["primary_reason"] == "TIMEOUT" for i in indices),
        }
    return out


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True); RAW_OUTPUT.mkdir(parents=True, exist_ok=True)
    split = load_pilot_split(SPLIT_PATH)
    cases = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))["case_schedule"]
    if len(cases) != 8 or not {r["geometry_id"] for r in cases}.issubset({r["geometry_id"] for r in split["pilot_diagnostic"]}):
        raise RuntimeError("S5.3J schedule must use the same eight internal TRAIN cases as S5.3G/H/I")
    support = action_support_audit()
    model, metadata = load_sac_checkpoint(RUN / "checkpoints/step_10000", device="cpu",
        expected_encoder_fingerprint=Stage5ObservationEncoder(core_only=False).get_schema_fingerprint())
    model.policy.set_training_mode(False)
    weights_before = policy_fingerprint(model); n_updates_before = int(model._n_updates)
    replay = getattr(model, "replay_buffer", None); replay_size_before = replay.size() if replay is not None else None
    alpha = float(model.log_ent_coef.detach().exp().item()) if getattr(model, "log_ent_coef", None) is not None else float(model.ent_coef_tensor.item())
    samples, episodes = collect_probe(model, cases)
    arrays, residuals = compute_soft_bellman(model, samples, alpha, float(model.gamma))
    weights_after = policy_fingerprint(model); n_updates_after = int(model._n_updates)
    replay_after = getattr(model, "replay_buffer", None); replay_size_after = replay_after.size() if replay_after is not None else None

    metadata_rows = [{k: v for k, v in s.items() if k not in ("observation", "next_observation", "action", "reward")} for s in samples]
    (RAW_OUTPUT / "episode_records.jsonl").write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in episodes), encoding="utf-8")
    np.savez_compressed(RAW_OUTPUT / "transitions.npz", observations=arrays["observations"], actions=arrays["actions"],
        rewards=arrays["rewards"], next_observations=arrays["next_observations"], done_masks=arrays["done_masks"],
        q1=residuals[:,0], q2=residuals[:,1], min_q=residuals[:,2], soft_target=residuals[:,3],
        soft_target_mc_sd=residuals[:,4], bellman_residual=residuals[:,5])
    (RAW_OUTPUT / "transition_metadata.jsonl").write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in metadata_rows), encoding="utf-8")

    # Action support from the newly collected diagnostic and comparison with historical marginal range.
    train_actions = [json.loads(x) for x in (RUN / "checkpoints/training_steps.jsonl").read_text(encoding="utf-8").splitlines()]
    old_steer = np.asarray([float(r["steering"]) for r in train_actions]); old_thr = np.asarray([float(r["throttle_brake"]) for r in train_actions])
    new_action_summary = {}
    for mode in ("deterministic", "stochastic"):
        idx = [i for i, s in enumerate(samples) if s["mode"] == mode]
        acts = arrays["actions"][idx]
        new_action_summary[mode] = {
            "n": len(idx), "steering_q01_q25_q50_q75_q99": np.quantile(acts[:,0], [.01,.25,.5,.75,.99]).astype(float).tolist(),
            "throttle_q01_q25_q50_q75_q99": np.quantile(acts[:,1], [.01,.25,.5,.75,.99]).astype(float).tolist(),
            "fraction_throttle_within_historical_marginal_minmax": float(np.mean((acts[:,1]>=old_thr.min())&(acts[:,1]<=old_thr.max()))),
            "fraction_steering_within_historical_marginal_minmax": float(np.mean((acts[:,0]>=old_steer.min())&(acts[:,0]<=old_steer.max()))),
            "fraction_throttle_positive": float(np.mean(acts[:,1]>.05)),
            "fraction_throttle_braking": float(np.mean(acts[:,1]<-.05)),
        }
    bellman = summarize_residuals(samples, residuals)
    isolation = {"policy_weights_unchanged": weights_before == weights_after,
        "policy_fingerprint_before": weights_before, "policy_fingerprint_after": weights_after,
        "optimizer_updates_before": n_updates_before, "optimizer_updates_after": n_updates_after,
        "replay_size_before": replay_size_before, "replay_size_after": replay_size_after,
        "normalization_update": False, "validation_test_access": False}
    summary = {
        "protocol": "S5.3J action-support and critic reliability diagnostic v1", "status": "COMPLETE",
        "checkpoint_sha256": metadata["model_sha256"], "training_step": 10000,
        "code_commit_from_checkpoint": metadata["git_commit"], "training_root_seed": 101,
        "pilot_split_sha256": split["split_sha256"], "case_schedule": cases,
        "J1_historical_action_support": support, "J1_new_diagnostic_action_support": new_action_summary,
        "J2_soft_bellman_diagnostic": {"gamma": float(model.gamma), "entropy_coefficient_alpha": alpha,
            "next_action_monte_carlo_samples": NEXT_ACTION_SAMPLES,
            "formula": "r + gamma*(1-done_mask)*(min(Q_target1,Q_target2) - alpha*log_pi(a_next|s_next))",
            "timeout_bootstrap": "done_mask=0 for truncation/timeout, matching SB3 timeout handling; safety/goal termination uses mask=1; external probe cap is censored and bootstrap mask=0.",
            "metrics_by_policy_mode": bellman,
            "limitations": ["New diagnostic transitions are not historical replay transitions.",
                "Bellman targets use 8 Monte Carlo next-action samples and have sampling noise.",
                "The probe is short and uses one checkpoint/training seed.",
                "Residuals are not a matched empirical return calibration and can be large due stochastic targets."]},
        "J2_episode_records": episodes, "read_only_isolation": isolation,
        "raw_transition_npz": "runs/stage5_pilot/s5_3j_critic_reliability_seed101_20261115/transitions.npz",
    }
    (RAW_OUTPUT / "summary_internal.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    source_files = ["scripts/diagnose_stage5_critic_reliability.py", "src/training/stage5/env_adapter.py",
        "src/training/stage5/checkpoint.py", "src/training/stage5/pilot_action_persistence.py",
        "src/platform/reward.py", "src/platform/action_adapter.py"]
    summary["source_file_sha256"] = {p: sha(ROOT / p) for p in source_files if (ROOT/p).exists()}
    summary["raw_transition_npz_sha256"] = sha(RAW_OUTPUT / "transitions.npz")
    summary["transition_metadata_sha256"] = sha(RAW_OUTPUT / "transition_metadata.jsonl")
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    (OUTPUT / "J1_action_support.json").write_text(json.dumps({"historical": support, "new_diagnostic": new_action_summary}, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    (OUTPUT / "J2_bellman_residuals.json").write_text(json.dumps(summary["J2_soft_bellman_diagnostic"], indent=2, sort_keys=True)+"\n", encoding="utf-8")

    lines = ["# S5.3J — Action support and critic reliability audit", "", "Status: **COMPLETE diagnostic**. Frozen 10k SAC checkpoint only; no learning updates.", "",
        "## J1 — Action support", "",
        f"The historical telemetry contains {support['n_logged_training_actions']} executed action rows and confirms marginal support: steering `{support['steering_distribution']['min']:.3f}..{support['steering_distribution']['max']:.3f}`, throttle/brake `{support['throttle_brake_distribution']['min']:.3f}..{support['throttle_brake_distribution']['max']:.3f}`. It does not contain state/next-state vectors, behavior log-probabilities, replay indices, or target-Q samples; replay buffer was not saved. Therefore exact historical state-conditional replay support and historical Bellman residual cannot be reconstructed.", "",
        f"New frozen-policy probes collected {len(samples)} transitions across the same eight TRAIN diagnostic geometries, deterministic and stochastic modes (cap {MAX_STEPS} steps per case/mode). Raw arrays are local under `{summary['raw_transition_npz']}`; this is a fresh diagnostic sample, not replay history.", "",
        "## J2 — Soft Bellman residual on fresh transitions", "",
        f"Checkpoint entropy coefficient `alpha={alpha:.6f}`, `gamma={float(model.gamma):.4f}`. The target uses SB3's soft target critic and stochastic next action, with {NEXT_ACTION_SAMPLES} next-action samples per transition. TIMEOUT truncations bootstrap (done mask 0), safety/success terminations do not, and the external cap is censored rather than converted to timeout.", "",
        "| Policy mode | transitions | mean |Q1−Q2| | mean |minQ−target| | residual q50 / q90 / q99 | mean next-target MC SD |", "|---|---:|---:|---:|---|---:|"]
    for mode in ("deterministic", "stochastic"):
        r = bellman[mode]
        lines.append(f"| {mode} | {r['n_transitions']} | {r['mean_abs_q1_q2_disagreement']:.4f} | {r['mean_absolute_bellman_residual']:.4f} | {r['bellman_residual_q50_q90_q99'][0]:.4f} / {r['bellman_residual_q50_q90_q99'][1]:.4f} / {r['bellman_residual_q50_q90_q99'][2]:.4f} | {r['next_soft_value_mc_sd_mean_8_samples']:.4f} |")
    lines += ["", "These are one-step, sampled soft Bellman residuals on transitions generated now by a frozen policy. They are not residuals on the historical replay and do not prove critic miscalibration. High residuals can reflect stochastic target sampling, state/action distribution shift, or actual critic error; this probe alone does not separate those causes.", "",
        "## J3 decision gate", "",
        "Historical replay support is unrecoverable, but a new state/action sample was available and J2 was completed. A further controlled action intervention should proceed only if these fresh residuals/disagreements indicate a material and repeatable concern; it must use reset-and-seed parity and verify identical initial observations before comparing interventions. This audit does not itself assert an intervention result.", "",
        f"Read-only check: weights unchanged `{isolation['policy_weights_unchanged']}`, optimizer updates {n_updates_before} → {n_updates_after}, replay size {replay_size_before} → {replay_size_after}. No VALIDATION/TEST. Checkpoint SHA-256 `{metadata['model_sha256']}`, split SHA-256 `{split['split_sha256']}`.", ""]
    (OUTPUT / "CONCLUSION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status":"COMPLETE", "historical_actions": support["n_logged_training_actions"],
        "fresh_transitions":len(samples), "bellman":bellman, "isolation":isolation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
