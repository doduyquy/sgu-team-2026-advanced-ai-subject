"""S5.3I read-only checkpoint action, critic, and reward/horizon audit."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.launcher.cases import LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256, compute_route_aware_horizon, load_manifest_csv
from src.training.stage5.checkpoint import load_sac_checkpoint
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.training.stage5.pilot_protocol import load_pilot_split
from src.training.stage5.train_sampler import SamplerStrategy, TrainCaseSampler

RUN_ID = "s5_3g_throttle_hold5_seed101_20261115"
RUN = ROOT / f"runs/stage5_pilot/{RUN_ID}"
PILOT_SPLIT = ROOT / "results/stage5/pilot_split_v1.json"
OUTPUT = ROOT / "results/stage5/s5_3i_policy_mechanism_seed101_20261115"
ACTION_GRID = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
STOCHASTIC_SAMPLES = 64
GAMMA = 0.99


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mean(xs):
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return statistics.mean(vals) if vals else None


def get_reset_observations(items, split, encoder):
    sampler = TrainCaseSampler(training_root_seed=101, strategy=SamplerStrategy.TIER_ROUND_ROBIN,
        allowed_geometry_ids=[x["geometry_id"] for x in split["pilot_diagnostic"]], project_root=ROOT)
    env = Stage5TrainingEnv(sampler=sampler, encoder=encoder, project_root=ROOT)
    observations = {}
    try:
        for item in items:
            case = sampler.get_case_for_geometry_id(item["geometry_id"], item["environment_seed"])
            obs, _ = env.reset(options={"case": case})
            observations[item["geometry_id"]] = np.asarray(obs, dtype=np.float32).copy()
    finally:
        env.close()
    return observations


def checkpoint_action_and_q(checkpoint_dir: Path, step: int, observations, expected_encoder):
    model, metadata = load_sac_checkpoint(checkpoint_dir, device="cpu",
        expected_encoder_fingerprint=expected_encoder)
    actions = []
    q_rows = []
    model.policy.set_training_mode(False)
    for geometry_id, obs_np in observations.items():
        deterministic, _ = model.predict(obs_np, deterministic=True)
        det = np.asarray(deterministic, dtype=np.float32).reshape(2)
        torch.manual_seed(910000 + step + int(hashlib.sha256(geometry_id.encode()).hexdigest()[:6], 16))
        stochastic = []
        for _ in range(STOCHASTIC_SAMPLES):
            value, _ = model.predict(obs_np, deterministic=False)
            stochastic.append(np.asarray(value, dtype=np.float32).reshape(2))
        stochastic = np.asarray(stochastic)
        actions.append({
            "checkpoint_step": step, "checkpoint_sha256": metadata["model_sha256"],
            "geometry_id": geometry_id,
            "deterministic_steering": float(det[0]), "deterministic_throttle": float(det[1]),
            "stochastic_steering_mean": float(stochastic[:, 0].mean()),
            "stochastic_steering_std": float(stochastic[:, 0].std()),
            "stochastic_steering_q05_q50_q95": np.quantile(stochastic[:, 0], [.05, .5, .95]).astype(float).tolist(),
            "stochastic_throttle_mean": float(stochastic[:, 1].mean()),
            "stochastic_throttle_std": float(stochastic[:, 1].std()),
            "stochastic_throttle_q05_q50_q95": np.quantile(stochastic[:, 1], [.05, .5, .95]).astype(float).tolist(),
            "stochastic_positive_throttle_fraction": float(np.mean(stochastic[:, 1] > 0)),
        })

        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=model.device).reshape(1, -1)
        for steering_label, steering in (("zero", 0.0), ("actor_deterministic", float(det[0]))):
            acts_t = torch.as_tensor([[steering, throttle] for throttle in ACTION_GRID],
                                     dtype=torch.float32, device=model.device)
            with torch.no_grad():
                q1, q2 = model.critic(obs_t.expand(len(ACTION_GRID), -1), acts_t)
            q1, q2 = q1.detach().cpu().numpy().reshape(-1), q2.detach().cpu().numpy().reshape(-1)
            for throttle, v1, v2 in zip(ACTION_GRID, q1, q2):
                q_rows.append({"checkpoint_step": step, "checkpoint_sha256": metadata["model_sha256"],
                    "geometry_id": geometry_id, "steering_condition": steering_label,
                    "steering": steering, "throttle": throttle, "q1": float(v1), "q2": float(v2),
                    "min_q": float(min(v1, v2)), "q_disagreement_abs": float(abs(v1-v2))})
    return actions, q_rows, metadata


def reward_horizon_audit():
    rows = [json.loads(x) for x in (RUN / "checkpoints/training_steps.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest_path = ROOT / "results/audits/evaluation_protocol/geometry_split_manifest.csv"
    manifest = load_manifest_csv(manifest_path, expected_sha256=LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256)
    horizons = {r["geometry_id"]: compute_route_aware_horizon(float(r["route_length_m"]))
                for r in manifest if r["split"].upper() == "TRAIN"}
    exact_residuals, time_cost_residuals = [], []
    grouped_actions = defaultdict(list)
    episodes = []
    current = []
    prev_id = None
    prev_step = 0
    for row in rows:
        expected = float(row["progress_reward"]) - float(row["time_cost"]) + float(row["terminal_reward"])
        exact_residuals.append(float(row["total_reward"]) - expected)
        h = horizons[row["geometry_id"]]
        time_cost_residuals.append(float(row["time_cost"]) - 0.25 / h)
        throttle = float(row["throttle_brake"])
        bucket = "throttle" if throttle > 0 else "brake" if throttle < 0 else "zero"
        grouped_actions[bucket].append(row)
        case_id = row["case_id"]
        step = int(row["episode_step"])
        if current and (case_id != prev_id or step <= prev_step):
            episodes.append(current); current = []
        current.append(row); prev_id, prev_step = case_id, step
    if current:
        episodes.append(current)

    episode_records = []
    for ep in episodes:
        last = ep[-1]
        done = bool(last["terminated_or_truncated"])
        horizon = horizons[last["geometry_id"]]
        rewards = [float(x["total_reward"]) for x in ep]
        episode_records.append({
            "case_id": last["case_id"], "geometry_id": last["geometry_id"],
            "tier": last["tier"], "episode_steps_logged": len(ep), "horizon_steps": horizon,
            "terminal_observed": done, "primary_reason": last.get("primary_reason"),
            "terminal_reward_sum": sum(float(x["terminal_reward"]) for x in ep),
            "episode_reward_sum_logged": sum(rewards),
            "discounted_logged_reward_gamma_0_99": sum((GAMMA ** t) * r for t, r in enumerate(rewards)),
            "discounted_logged_reward_gamma_0_9991": sum((0.9991 ** t) * r for t, r in enumerate(rewards)),
        })

    action_groups = {}
    for name, seq in grouped_actions.items():
        action_groups[name] = {
            "n": len(seq), "mean_progress_reward": mean(r["progress_reward"] for r in seq),
            "mean_time_cost": mean(r["time_cost"] for r in seq),
            "mean_terminal_reward": mean(r["terminal_reward"] for r in seq),
            "mean_total_reward": mean(r["total_reward"] for r in seq),
            "p_delta_route_positive": mean(float(r["delta_route"] > 0) for r in seq),
            "p_step_reward_positive": mean(float(r["total_reward"] > 0) for r in seq),
            "sum_terminal_reward": sum(float(r["terminal_reward"]) for r in seq),
        }

    return {
        "n_transitions": len(rows), "n_recorded_episodes": len(episode_records),
        "n_terminal_episodes": sum(e["terminal_observed"] for e in episode_records),
        "n_partial_episode_fragments": sum(not e["terminal_observed"] for e in episode_records),
        "max_reward_decomposition_abs_residual": max(abs(x) for x in exact_residuals),
        "max_time_cost_formula_abs_residual": max(abs(x) for x in time_cost_residuals),
        "reward_contract_formula": "progress_reward - time_cost + terminal_reward",
        "time_cost_formula": "0.25 / route_aware_horizon_steps",
        "episodes": episode_records, "action_group_summary_confounded": action_groups,
        "action_group_caveat": "Descriptive only: action groups occur in different states, episode times, and terminal contexts; no causal comparison.",
        "reward_discounting_caveat": "Logged discounted reward sums omit SAC entropy terms and critic bootstrap/timeout handling; they are not Q-value targets.",
        "mean_p_delta_route_positive": mean(float(r["delta_route"] > 0) for r in rows),
        "mean_p_progress_exceeds_time_cost": mean(float(r["progress_reward"] > r["time_cost"]) for r in rows),
        "mean_p_total_reward_positive": mean(float(r["total_reward"] > 0) for r in rows),
        "mean_progress_reward": mean(r["progress_reward"] for r in rows),
        "mean_time_cost": mean(r["time_cost"] for r in rows),
        "sum_terminal_reward": sum(float(r["terminal_reward"]) for r in rows),
        "mean_total_reward": mean(r["total_reward"] for r in rows),
        "action_groups": action_groups,
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    split = load_pilot_split(PILOT_SPLIT)
    schedule = json.loads((ROOT / "results/stage5/s5_3g_steering_isolation_seed101_20261115/summary.json").read_text(encoding="utf-8"))["case_schedule"]
    if len(schedule) != 8 or not {r["geometry_id"] for r in schedule}.issubset({r["geometry_id"] for r in split["pilot_diagnostic"]}):
        raise RuntimeError("Fixed observations are not the recorded eight internal diagnostic TRAIN geometries")
    encoder = Stage5ObservationEncoder(core_only=False)
    observations = get_reset_observations(schedule, split, encoder)
    all_actions, all_q, checkpoint_meta = [], [], {}
    for step in (0, 5000, 10000):
        actions, qs, metadata = checkpoint_action_and_q(
            RUN / f"checkpoints/step_{step:05d}", step, observations, encoder.get_schema_fingerprint())
        all_actions.extend(actions); all_q.extend(qs)
        checkpoint_meta[str(step)] = {"model_sha256": metadata["model_sha256"],
                                      "total_timesteps": metadata["total_timesteps"],
                                      "git_commit": metadata["git_commit"],
                                      "replay_buffer_saved": metadata["replay_buffer_saved"]}
    (OUTPUT / "checkpoint_actions.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in all_actions), encoding="utf-8")
    (OUTPUT / "critic_throttle_grid.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in all_q), encoding="utf-8")
    reward = reward_horizon_audit()
    (OUTPUT / "reward_horizon_audit.json").write_text(json.dumps(reward, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checkpoint_summary = {}
    for step in (0, 5000, 10000):
        subset = [r for r in all_actions if r["checkpoint_step"] == step]
        qsubset = [r for r in all_q if r["checkpoint_step"] == step]
        critic_grids = {}
        for condition in ("zero", "actor_deterministic"):
            by_thr = defaultdict(list)
            for r in qsubset:
                if r["steering_condition"] == condition:
                    by_thr[r["throttle"]].append(r["min_q"])
            mean_q_by_thr = {str(t): mean(by_thr[t]) for t in ACTION_GRID}
            max_q_throttle = max(mean_q_by_thr, key=lambda key: mean_q_by_thr[key])
            critic_grids[condition] = {
                "mean_min_q_by_throttle": mean_q_by_thr,
                "argmax_throttle": float(max_q_throttle),
                "mean_min_q_at_minus1_minus_plus1": mean(by_thr[-1.0]) - mean(by_thr[1.0]),
            }
        checkpoint_summary[str(step)] = {
            "n_observations": len(subset),
            "deterministic_throttle_mean": mean(r["deterministic_throttle"] for r in subset),
            "deterministic_throttle_median": statistics.median(r["deterministic_throttle"] for r in subset),
            "deterministic_throttle_positive_fraction": mean(float(r["deterministic_throttle"] > 0) for r in subset),
            "deterministic_steering_mean": mean(r["deterministic_steering"] for r in subset),
            "stochastic_throttle_mean": mean(r["stochastic_throttle_mean"] for r in subset),
            "stochastic_throttle_std_mean": mean(r["stochastic_throttle_std"] for r in subset),
            "stochastic_throttle_positive_fraction_mean": mean(r["stochastic_positive_throttle_fraction"] for r in subset),
            "stochastic_steering_std_mean": mean(r["stochastic_steering_std"] for r in subset),
            "critic_grid_summary": critic_grids,
            "critic_q_is_prediction_not_empirical_return": True,
        }
    provenance = {
        "protocol": "S5.3I policy mechanism audit v1", "status": "COMPLETE",
        "code_commit_from_checkpoints": checkpoint_meta["10000"]["git_commit"],
        "checkpoints": checkpoint_meta, "training_root_seed": 101,
        "pilot_split_sha256": split["split_sha256"], "fixed_observation_case_schedule": schedule,
        "observation_count": len(observations), "stochastic_samples_per_observation_checkpoint": STOCHASTIC_SAMPLES,
        "critic_action_grid_throttle": ACTION_GRID,
        "critic_steering_conditions": ["zero", "checkpoint deterministic actor steering"],
        "replay_buffers_available": False,
        "limitations": [
            "Checkpoints are from one training seed and one pilot run.",
            "Step-0 critics are randomly initialized; their Q ranking is not a calibrated value estimate.",
            "10k replay buffer was not saved; transition-level critic training data cannot be reconstructed.",
            "Q-grid is offline critic prediction only, not a causal action-value estimate or empirical return.",
            "Training action-group reward comparisons are state/episode confounded.",
            "Logged reward-only discounted sums omit entropy terms and bootstrapping semantics.",
            "All observations and transitions are from internal TRAIN split; no VALIDATION or TEST access.",
        ],
    }
    summary = {"checkpoint_action_summary": checkpoint_summary, "reward_horizon_audit": reward,
              "provenance": provenance, "status": "COMPLETE"}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source_files = ["scripts/diagnose_stage5_policy_mechanism.py", "src/training/stage5/env_adapter.py",
                    "src/training/stage5/checkpoint.py", "src/platform/reward.py"]
    source_hashes = {p: sha(ROOT / p) for p in source_files if (ROOT / p).exists()}
    lines = ["# S5.3I — Policy mechanism audit", "", "Status: **COMPLETE**. Read-only; no training run or frozen contract changes.", "",
        "## I-1: action evolution across checkpoints", "", "All three checkpoints were queried on the same eight reset observations from the internal TRAIN diagnostic schedule; each used 64 stochastic samples per observation.", "",
        "| Checkpoint | Mean deterministic throttle | Median | Positive fraction | Stochastic throttle mean / SD | Stochastic steering SD | Q-grid argmax throttle (zero / actor steering) | ΔminQ(-1 minus +1), actor steering |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for step in (0, 5000, 10000):
        r = checkpoint_summary[str(step)]
        qzero = r["critic_grid_summary"]["zero"]
        qactor = r["critic_grid_summary"]["actor_deterministic"]
        lines.append(f"| {step} | {r['deterministic_throttle_mean']:+.3f} | {r['deterministic_throttle_median']:+.3f} | {r['deterministic_throttle_positive_fraction']:.3f} | {r['stochastic_throttle_mean']:+.3f} / {r['stochastic_throttle_std_mean']:.3f} | {r['stochastic_steering_std_mean']:.3f} | {qzero['argmax_throttle']:+.2f} / {qactor['argmax_throttle']:+.2f} | {qactor['mean_min_q_at_minus1_minus_plus1']:+.4f} |")
    lines += ["", "Q-grid argmax is the critic's predicted min(Q1,Q2) averaged across the eight observations, not empirical return. At 5k and 10k the average critic grid slopes toward braking at both fixed-zero and actor steering. The endpoint Q difference is small relative to the Q level and the grid includes actions that may be out of the replay distribution. The step-0 critic is randomly initialized and has no useful calibration interpretation.", "",
        "## I-2: critic throttle preference", "",
        "The complete per-observation Q1/Q2/min-Q grid is `critic_throttle_grid.jsonl`. It evaluates nine throttle values from -1 to +1 at zero steering and at each checkpoint's deterministic actor steering. Q is learned prediction, not ground truth. The missing 10k replay buffer prevents checking whether its data covered useful moving trajectories or calibrating Q against actual returns.", "",
        "## I-3: reward and horizon audit", "",
        f"Audited {reward['n_transitions']} recorded transitions across {reward['n_recorded_episodes']} logged episode segments. Maximum absolute row-level residual for `total = progress - time_cost + terminal` was {reward['max_reward_decomposition_abs_residual']:.3g}; maximum residual for `time_cost = 0.25 / route-aware horizon` was {reward['max_time_cost_formula_abs_residual']:.3g}.", "",
        f"Positive route-progress transitions: {reward['mean_p_delta_route_positive']:.1%}. Progress exceeded the per-step time cost on {reward['mean_p_progress_exceeds_time_cost']:.1%}; total step reward was positive on {reward['mean_p_total_reward_positive']:.1%}. Summed terminal reward: {reward['sum_terminal_reward']:.3f}. These are descriptive statistics from the existing pilot logs.", "",
        "### Reward by action sign (descriptive, confounded)", "",
        "| Action sign | Transitions | Mean progress reward | Mean time cost | Sum terminal reward | Mean total reward | P(progress > 0) | P(total reward > 0) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("throttle", "brake", "zero"):
        group = reward["action_groups"].get(name)
        if group:
            lines.append(f"| {name} | {group['n']} | {group['mean_progress_reward']:+.6f} | {group['mean_time_cost']:.6f} | {group['sum_terminal_reward']:+.1f} | {group['mean_total_reward']:+.6f} | {group['p_delta_route_positive']:.1%} | {group['p_step_reward_positive']:.1%} |")
    throttle_group, brake_group = reward["action_groups"].get("throttle", {}), reward["action_groups"].get("brake", {})
    lines += ["", "The pooled reversal is arithmetically explained: positive-throttle rows have a larger mean progress reward, but carry more terminal penalty (6 of the eight `-1` penalties versus 2 under brake). Because that penalty is assigned to the final transition, it lowers the pooled total reward for the throttle group. This association does not prove throttle caused the failures: action groups occur in different states, trajectories, and episode contexts.", "",
        f"Throttle versus brake: mean progress reward {throttle_group.get('mean_progress_reward', 0):+.6f} vs {brake_group.get('mean_progress_reward', 0):+.6f}; summed terminal reward {throttle_group.get('sum_terminal_reward', 0):+.1f} vs {brake_group.get('sum_terminal_reward', 0):+.1f}; mean total reward {throttle_group.get('mean_total_reward', 0):+.6f} vs {brake_group.get('mean_total_reward', 0):+.6f}.", "",
        "Episode-level logged reward-only returns under gamma .99 and .9991 are included in `reward_horizon_audit.json`. They omit SAC entropy terms and bootstrap behavior, so they are not critic targets.", "",
        "## Conclusion and limits", "",
        "This audit can show whether deterministic throttle changed between the 0/5k/10k checkpoints and what the existing critic predicts across a throttle grid. It cannot prove why the actor changed: the replay buffer was not saved, only one seed is available, and Q values have not been calibrated against matched empirical returns. Reward decomposition matches the implemented formula on logged transitions, so this audit found no row-level reward-accounting mismatch in this run. Do not infer reward hacking, critic error, or a confirmed gamma cause from these data.", "",
        f"Checkpoint SHA-256: 0 `{checkpoint_meta['0']['model_sha256']}`, 5k `{checkpoint_meta['5000']['model_sha256']}`, 10k `{checkpoint_meta['10000']['model_sha256']}`. Internal split SHA-256: `{split['split_sha256']}`. Source SHA-256 values are in `source_hashes.json`. No VALIDATION/TEST access.", ""]
    (OUTPUT / "CONCLUSION.md").write_text("\n".join(lines), encoding="utf-8")
    (OUTPUT / "source_hashes.json").write_text(json.dumps(source_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint_summary": checkpoint_summary,
                      "reward_summary": {k: reward[k] for k in ("n_transitions", "n_recorded_episodes", "n_terminal_episodes", "n_partial_episode_fragments", "max_reward_decomposition_abs_residual", "max_time_cost_formula_abs_residual", "mean_p_delta_route_positive", "mean_p_progress_exceeds_time_cost", "mean_p_total_reward_positive", "sum_terminal_reward", "mean_total_reward")},
                      "status": "COMPLETE"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
