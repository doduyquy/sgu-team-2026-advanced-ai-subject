"""Offline TRAIN-only P2-E critic/action support audit for frozen P2-D runs."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import stable_baselines3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage5_p2d_actor_activation import read_protocol
from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER
from src.training.stage5.actor_activation_ablation import ACTOR_LEAKY_RELU_001_V1, ACTOR_RELU_V1
from src.training.stage5.checkpoint import load_sac_checkpoint
from src.training.stage5.observation_encoder import Stage5ObservationEncoder

RUN = ROOT / "runs/stage5_p2d/s5_p2d_actor_activation_seed101_20261121"
OUT = ROOT / "results/stage5/p2e_critic_learning_signal_20261121"
STEPS = (2000, 5000, 10000)
DEVICE = torch.device("cpu")
GROUPS = {
    "ego_9d": (0, 9), "navigation_10d": (9, 19), "lidar_240d": (19, 259),
    "traffic_66d": (259, 325), "task_84d": (325, 409),
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qstats(values: Any) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if not x.size:
        return {"count": 0, "mean": None, "std": None, "p10": None, "p50": None, "p90": None}
    return {"count": int(x.size), "mean": float(x.mean()), "std": float(x.std()),
            "p10": float(np.quantile(x, .1)), "p50": float(np.quantile(x, .5)),
            "p90": float(np.quantile(x, .9))}


def turn_score(obs: np.ndarray) -> np.ndarray:
    rows = np.asarray(obs)[:, 325:385].reshape(-1, 20, 3)
    masks = np.asarray(obs)[:, 385:405] > .5
    out = np.zeros(len(obs), dtype=np.float32)
    for i in range(len(obs)):
        headings = rows[i, masks[i], 2] * math.pi
        if headings.size == 1:
            out[i] = abs(headings[0])
        elif headings.size > 1:
            out[i] = max(float(np.max(np.abs(headings))), float(np.max(np.abs(np.diff(headings)))))
    return out


def layer_group(name: str) -> str:
    pieces = name.split(".")
    if name.startswith("latent_pi."):
        return ".".join(pieces[:2])
    return pieces[0]


def flat_grads(loss: torch.Tensor, named: list[tuple[str, torch.nn.Parameter]], retain: bool) -> dict[str, torch.Tensor]:
    grads = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True, retain_graph=retain)
    grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
    for (name, param), grad in zip(named, grads):
        grouped[layer_group(name)].append(torch.zeros_like(param).reshape(-1) if grad is None else grad.detach().reshape(-1))
    return {key: torch.cat(value) for key, value in grouped.items()}


def grad_summary(model: Any, obs_np: np.ndarray, actions_np: np.ndarray | None = None) -> dict[str, Any]:
    actor, critic = model.policy.actor, model.policy.critic
    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=DEVICE)
    # Use the same states for both SAC objective terms. The actor's sampled actions
    # are the actions at which entropy and Q terms are differentiated.
    sampled_action, log_prob = actor.action_log_prob(obs)
    q1, q2 = critic(obs, sampled_action)
    alpha = model.log_ent_coef.detach().exp() if getattr(model, "log_ent_coef", None) is not None else torch.as_tensor(float(model.ent_coef), device=DEVICE)
    entropy_loss = (alpha * log_prob.reshape(-1, 1)).mean()
    negative_q_loss = -torch.minimum(q1, q2).mean()
    params = list(actor.named_parameters())
    ge = flat_grads(entropy_loss, params, True)
    gq = flat_grads(negative_q_loss, params, False)
    by_layer: dict[str, Any] = {}
    for key in sorted(set(ge) | set(gq)):
        a, b = ge.get(key), gq.get(key)
        if a is None or b is None:
            continue
        na, nb = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
        cosine = float(torch.dot(a, b) / (na * nb).clamp_min(1e-30))
        combined = a + b
        by_layer[key] = {
            "entropy_grad_l2": float(na), "negative_q_grad_l2": float(nb),
            "combined_grad_l2": float(torch.linalg.vector_norm(combined)),
            "entropy_vs_negative_q_cosine": cosine,
            "negative_q_to_entropy_norm_ratio": float(nb / na.clamp_min(1e-30)),
        }
    # Critic sensitivity to actions at deterministic and behavior actions.
    with torch.no_grad():
        mean, _, _ = actor.get_action_dist_params(obs)
        det = torch.tanh(mean)
    action_rows: dict[str, Any] = {}
    paired_q_values: dict[str, torch.Tensor] = {}
    candidates = {"actor_deterministic": det.detach()}
    if actions_np is not None and len(actions_np) == len(obs_np):
        candidates["recorded_behavior"] = torch.as_tensor(actions_np, dtype=torch.float32, device=DEVICE)
    for label, candidate in candidates.items():
        candidate = candidate.detach().clone().requires_grad_(True)
        c1, c2 = critic(obs, candidate)
        q = torch.minimum(c1, c2).mean()
        dq = torch.autograd.grad(q, candidate, retain_graph=False)[0]
        action_rows[label] = {
            "mean_q1": float(c1.detach().mean()), "mean_q2": float(c2.detach().mean()),
            "mean_abs_twin_gap": float((c1.detach() - c2.detach()).abs().mean()),
            "mean_abs_dq_d_steering": float(dq[:, 0].abs().mean()),
            "mean_abs_dq_d_throttle_brake": float(dq[:, 1].abs().mean()),
            "mean_action": candidate.detach().mean(dim=0).cpu().tolist(),
        }
        paired_q_values[label] = torch.minimum(c1.detach(), c2.detach()).reshape(-1)
    if "actor_deterministic" in paired_q_values and "recorded_behavior" in paired_q_values:
        gap = paired_q_values["actor_deterministic"] - paired_q_values["recorded_behavior"]
        action_rows["paired_deterministic_minus_recorded_behavior_q"] = {
            "mean": float(gap.mean()), "std": float(gap.std(unbiased=False)),
            "fraction_positive": float((gap > 0).float().mean()),
            "fraction_negative": float((gap < 0).float().mean()),
            "interpretation_limit": "A critic comparison at the same sampled states; one behavior action per state is not sufficient to establish correct ranking or empirical value.",
        }
    return {
        "sample_count": int(len(obs_np)), "alpha": float(alpha),
        "mean_log_prob": float(log_prob.detach().mean()),
        "entropy_objective_component": float(entropy_loss.detach()),
        "negative_q_objective_component": float(negative_q_loss.detach()),
        "actor_gradient_by_layer": by_layer,
        "critic_action_sensitivity": action_rows,
        "warning": "Gradient terms are an offline snapshot at frozen weights; Q estimates are not empirical returns.",
    }


def local_support(obs: np.ndarray, behavior_actions: np.ndarray, query_actions: np.ndarray,
                  geometry_ids: list[str] | None = None, query_mask: np.ndarray | None = None,
                  k: int = 20) -> dict[str, Any]:
    n = len(obs)
    if n < 3:
        return {"status": "INSUFFICIENT_NEIGHBORS", "count": n}
    if geometry_ids is None:
        geometry_ids = ["POOLED"] * n
    if len(geometry_ids) != n:
        raise ValueError("geometry_ids must align with the replay observations")
    groups = np.asarray(geometry_ids)
    candidate_q = np.flatnonzero(np.ones(n, dtype=bool) if query_mask is None else np.asarray(query_mask, dtype=bool))
    if not len(candidate_q):
        return {"status": "NO_QUERIES_IN_STRATUM", "count": 0}
    # Deterministically bound query/reference work while preserving TRAIN geometry identity.
    qidx = candidate_q[np.unique(np.linspace(0, len(candidate_q) - 1, min(len(candidate_q), 512), dtype=int))]
    ridx = np.unique(np.linspace(0, n - 1, min(n, 2048), dtype=int))
    scales_by_geometry: dict[str, np.ndarray] = {}
    for group_id in np.unique(groups):
        scale = np.std(obs[groups == group_id], axis=0)
        scales_by_geometry[str(group_id)] = np.where(scale > 1e-5, scale, 1.0)
    selected_global: list[np.ndarray] = []
    distances: list[float] = []
    nearest_state_distances: list[float] = []
    # For every query, restrict candidate behavior actions to the same geometry.
    # This reduces cross-map support inflation but remains only a local heuristic.
    for qi in qidx:
        g = groups[qi]
        candidate_refs = ridx[groups[ridx] == g]
        if len(candidate_refs) < 2:
            selected_global.append(np.array([], dtype=int))
            distances.append(float("inf")); nearest_state_distances.append(float("inf"))
            continue
        scale = scales_by_geometry[str(g)]
        delta = (obs[candidate_refs] - obs[qi]) / scale
        d = np.sqrt(np.maximum(0.0, np.sum(delta * delta, axis=1)))
        d[candidate_refs == qi] = np.inf
        count = min(k, len(candidate_refs) - (1 if qi in candidate_refs else 0))
        if count < 1:
            selected_global.append(np.array([], dtype=int)); distances.append(float("inf")); nearest_state_distances.append(float("inf")); continue
        local = np.argpartition(d, count - 1)[:count]
        selected_global.append(candidate_refs[local])
        distances.extend(d[local].tolist())
        nearest_state_distances.append(float(np.min(d)))
    valid = [i for i, x in enumerate(selected_global) if len(x)]
    if not valid:
        return {"status": "NO_SAME_GEOMETRY_NEIGHBORS", "count": int(len(qidx))}
    query_kept = qidx[valid]
    q_actions = query_actions[query_kept]
    neighbor_actions = np.stack([behavior_actions[selected_global[i]] for i in valid])
    # Pad to a common k for a concise array calculation; skipped rows are excluded.
    min_k = min(row.shape[0] for row in neighbor_actions)
    neighbor_actions = neighbor_actions[:, :min_k]
    action_delta = neighbor_actions - q_actions[:, None, :]
    euclidean = np.linalg.norm(action_delta, axis=2)
    local_action_std = neighbor_actions.std(axis=1)
    local_action_rms = np.sqrt(np.mean(local_action_std ** 2, axis=1)).clip(min=1e-8)
    within_box_cloud = ((q_actions[:, None, :] >= neighbor_actions.min(axis=1, keepdims=True)) &
                        (q_actions[:, None, :] <= neighbor_actions.max(axis=1, keepdims=True)))
    return {
        "query_count": int(len(query_kept)), "query_population_count": int(len(candidate_q)),
        "reference_count": int(len(ridx)), "k_neighbors": int(min_k),
        "observation_neighbor_distance_standardized": qstats(distances),
        "nearest_observation_distance_standardized": qstats([nearest_state_distances[i] for i in valid]),
        "fraction_queries_with_duplicate_observation_neighbor": float(np.mean(np.asarray([nearest_state_distances[i] for i in valid]) <= 1e-8)),
        "deterministic_action_to_nearest_behavior_action_l2": qstats(euclidean.min(axis=1)),
        "deterministic_action_to_local_behavior_centroid_l2": qstats(np.linalg.norm(q_actions-neighbor_actions.mean(axis=1), axis=1)),
        "nearest_action_distance_over_local_action_rms_std": qstats(euclidean.min(axis=1) / local_action_rms),
        "fraction_inside_local_behavior_action_axis_aligned_bounds": float(within_box_cloud.all(axis=2).any(axis=1).mean()),
        "mean_local_behavior_action_std_by_dimension": local_action_std.mean(axis=0).tolist(),
        "limitation": "A per-geometry kNN heuristic over sampled replay observations/actions, not a formal density estimate or proof of temporal policy support.",
    }


def transition_groups(obs: np.ndarray, speed: np.ndarray, delta: np.ndarray, rewards: np.ndarray | None, actions: np.ndarray, outcome: list[Any]) -> dict[str, Any]:
    turns = turn_score(obs)
    def is_safety(item: Any) -> bool:
        if isinstance(item, dict):
            reason = str(item.get("primary_reason", "")).upper()
            raw_flags = item.get("raw_flags", {})
            return reason in {"OUT_OF_ROAD", "CRASH", "SAFETY_FAILURE"} or bool(raw_flags.get("out_of_road") or raw_flags.get("crash"))
        return str(item).upper() in {"OUT_OF_ROAD", "CRASH", "SAFETY_FAILURE"}
    masks = {
        "stationary_lt_1_kmh": speed < 1,
        "moving_ge_1_kmh": speed >= 1,
        "positive_progress": delta > 0,
        "turn_context_ge_0_05rad": turns >= .05,
        "safety_failure_terminal": np.asarray([is_safety(x) for x in outcome]),
    }
    out = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        out[name] = {
            "count": count, "fraction": float(mask.mean()) if len(mask) else 0,
            "speed_km_h": qstats(speed[mask]), "delta_route": qstats(delta[mask]),
            "reward": qstats(rewards[mask]) if rewards is not None else None, "steering": qstats(actions[mask, 0]),
            "throttle_brake": qstats(actions[mask, 1]),
            "positive_throttle_fraction": float((actions[mask, 1] > .05).mean()) if count else None,
            "brake_fraction": float((actions[mask, 1] < -.05).mean()) if count else None,
        }
    return {"thresholds": {"moving_kmh": 1.0, "positive_progress": 0.0, "turn_context_rad": .05,
                           "throttle_positive": .05, "brake_negative": -.05}, "groups": out}


def critic_replay_strata(model: Any, rb: Any, obs: np.ndarray, speed: np.ndarray, delta: np.ndarray,
                         outcome: list[Any], seed: int) -> dict[str, Any]:
    """Measure SB3-style critic TD signal on sampled 10k replay strata."""
    n = len(obs)
    turns = turn_score(obs)
    masks = {
        "stationary_lt_1_kmh": speed < 1,
        "moving_ge_1_kmh": speed >= 1,
        "positive_progress": delta > 0,
        "turn_context_ge_0_05rad": turns >= .05,
        "safety_failure_terminal": np.asarray([
            isinstance(x, dict) and outcome_name(x) in {"OUT_OF_ROAD", "CRASH", "SAFETY_FAILURE"}
            for x in outcome
        ]),
    }
    out: dict[str, Any] = {}
    generator = np.random.default_rng(seed)
    for name, mask in masks.items():
        pool = np.flatnonzero(mask)
        if len(pool) == 0:
            out[name] = {"population_count": 0, "status": "NO_TRANSITIONS"}
            continue
        chosen = generator.choice(pool, size=min(256, len(pool)), replace=False)
        ob = torch.as_tensor(rb.observations[chosen, 0], dtype=torch.float32, device=DEVICE)
        next_ob = torch.as_tensor(rb.next_observations[chosen, 0], dtype=torch.float32, device=DEVICE)
        action = torch.as_tensor(rb.actions[chosen, 0], dtype=torch.float32, device=DEVICE)
        reward = torch.as_tensor(rb.rewards[chosen, 0], dtype=torch.float32, device=DEVICE).reshape(-1, 1)
        done = torch.as_tensor(rb.dones[chosen, 0], dtype=torch.float32, device=DEVICE).reshape(-1, 1)
        timeout = torch.as_tensor(rb.timeouts[chosen, 0], dtype=torch.float32, device=DEVICE).reshape(-1, 1)
        # Match SB3 ReplayBuffer timeout semantics: truncations bootstrap, true
        # terminations do not. The action trace provides the semantic parity check.
        done_for_target = done * (1.0 - timeout)
        with torch.no_grad():
            next_action, next_log_prob = model.policy.actor.action_log_prob(next_ob)
            target_q1, target_q2 = model.policy.critic_target(next_ob, next_action)
            alpha = model.log_ent_coef.detach().exp() if getattr(model, "log_ent_coef", None) is not None else torch.as_tensor(float(model.ent_coef), device=DEVICE)
            target = reward + (1.0 - done_for_target) * float(model.gamma) * (
                torch.minimum(target_q1, target_q2) - alpha * next_log_prob.reshape(-1, 1)
            )
        q1, q2 = model.policy.critic(ob, action)
        # Exact SB3 SAC 2.9.2 critic-loss scale: 0.5 * sum(MSE per critic).
        loss = 0.5 * (torch.nn.functional.mse_loss(q1, target) + torch.nn.functional.mse_loss(q2, target))
        named = list(model.policy.critic.named_parameters())
        grads = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True)
        layer_sq: dict[str, float] = defaultdict(float)
        for (param_name, _), grad in zip(named, grads):
            key = ".".join(param_name.split(".")[:2])
            if grad is not None:
                layer_sq[key] += float(grad.detach().pow(2).sum())
        td1 = (q1.detach() - target).reshape(-1).cpu().numpy()
        td2 = (q2.detach() - target).reshape(-1).cpu().numpy()
        out[name] = {
            "population_count": int(len(pool)), "sample_count": int(len(chosen)),
            "mean_abs_td_q1": float(np.mean(np.abs(td1))), "mean_abs_td_q2": float(np.mean(np.abs(td2))),
            "mean_signed_td_q1": float(np.mean(td1)), "mean_signed_td_q2": float(np.mean(td2)),
            "critic_loss_sum": float(loss.detach()),
            "critic_gradient_l2_by_layer": {k: float(math.sqrt(v)) for k, v in layer_sq.items()},
            "mean_target": float(target.mean()), "mean_q1": float(q1.detach().mean()), "mean_q2": float(q2.detach().mean()),
            "timeout_count_bootstrapped": int(timeout.sum().item()),
            "true_terminal_count_not_bootstrapped": int((done * (1-timeout)).sum().item()),
        }
    return out


def load_trace(arm: str) -> list[dict[str, Any]]:
    path = RUN / f"arm_{arm}/replay_action_trace.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def outcome_name(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("primary_reason") or "UNKNOWN")
    return str(item)


def main() -> None:
    protocol = read_protocol()
    encoder = Stage5ObservationEncoder(core_only=False)
    anchor_path = ROOT / protocol["anchors"]["path"]
    anchors = np.load(anchor_path, allow_pickle=False)
    output: dict[str, Any] = {
        "schema": "stage5_p2e_critic_learning_signal_v1", "protocol_sha256": protocol["protocol_sha256"],
        "anchors_sha256": sha(anchor_path), "source_commit": "c322cb1f220d7491356a9566c3add39d49054ab9",
        "scope": "offline TRAIN-only analysis of frozen P2-D checkpoints/replay; no training or evaluation rollout",
        "runtime": {"stable_baselines3": stable_baselines3.__version__, "torch": torch.__version__, "device": str(DEVICE)},
        "input_artifact_sha256": {
            "p2c_evidence": sha(ROOT / "results/stage5/p2c_representation_critic_audit_20261121/evidence.json"),
            "p2c_fixed_anchors": sha(anchor_path),
            "p2d_protocol_v3": sha(ROOT / "results/stage5/p2d_actor_activation_seed101_20261121/protocol_v3.json"),
            "p2d_result_summary": sha(ROOT / "results/stage5/p2d_actor_activation_seed101_20261121/result_summary.json"),
            "p2d_layer_gradient_audit": sha(ROOT / "results/stage5/p2d_actor_activation_seed101_20261121/posthoc_layer_gradient_audit.json"),
            "p2d_run_manifest": sha(RUN / "run_manifest.json"),
        },
        "arms": {}, "comparison": {},
        "limitations": [
            "Only full replay at 10k is stored; 2k and 5k are deterministic 1024-transition replay samples, not complete buffers.",
            "Critic Q and local action derivatives are learned estimates, not ground truth.",
            "kNN action support is a descriptive heuristic, not a formal support estimator.",
            "Observed action consequences are confounded by state, policy, geometry and episode selection; no causal long-horizon Q conclusion is made.",
        ],
    }
    trace_data: dict[str, list[dict[str, Any]]] = {}
    for arm in ("A", "B"):
        expected = ACTOR_RELU_V1 if arm == "A" else ACTOR_LEAKY_RELU_001_V1
        trace = load_trace(arm)
        trace_data[arm] = trace
        steps_out: dict[str, Any] = {}
        for step in STEPS:
            sample_path = RUN / f"arm_{arm}/replay_samples/observations_step_{step:05d}.npz"
            with np.load(sample_path, allow_pickle=False) as sample:
                obs = sample["observations"].astype(np.float32)
                act = sample["actions"].astype(np.float32)
                speed = sample["speed_km_h"].astype(np.float32)
                delta = sample["delta_route"].astype(np.float32)
            sample_indices = np.linspace(0, step - 1, min(1024, step), dtype=np.int64)
            sample_rows = [trace[int(i)] for i in sample_indices]
            if len(sample_rows) != len(act) or np.any(np.abs(np.asarray([r["replay_action"] for r in sample_rows], dtype=np.float32) - act) > 1e-6):
                raise ValueError(f"P2-D replay sample does not align with recorded trace: arm={arm}, step={step}")
            sample_geometry = [r["geometry_id"] for r in sample_rows]
            checkpoint = RUN / f"arm_{arm}/checkpoints/step_{step:06d}"
            model, metadata = load_sac_checkpoint(
                checkpoint, expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
                task_context_producer_id=REFERENCE_TASK_PRODUCER, expected_actor_activation_id=expected,
                load_replay_buffer=(step == 10000),
            )
            torch.manual_seed(20261121 + step)
            with torch.no_grad():
                ob_t = torch.as_tensor(obs, dtype=torch.float32)
                mean, log_std, _ = model.policy.actor.get_action_dist_params(ob_t)
                deterministic = torch.tanh(mean).cpu().numpy()
                stochastic, _ = model.policy.actor.action_log_prob(ob_t)
                stochastic = stochastic.cpu().numpy()
            sample_groups = transition_groups(obs, speed, delta, None, act, [None]*len(obs))
            torch.manual_seed(20261121 + step)
            objective = grad_summary(model, anchors, None)
            torch.manual_seed(20262121 + step)
            replay_objective = grad_summary(model, obs, act)
            support = local_support(obs, act, deterministic, geometry_ids=sample_geometry)
            sample_turn = turn_score(obs)
            sample_outcomes = [r.get("outcome") for r in sample_rows]
            support_masks = {
                "stationary_lt_1_kmh": speed < 1,
                "moving_ge_1_kmh": speed >= 1,
                "positive_progress": delta > 0,
                "turn_context_ge_0_05rad": sample_turn >= .05,
                "safety_failure_terminal": np.asarray([
                    isinstance(v, dict) and outcome_name(v) in {"OUT_OF_ROAD", "CRASH", "SAFETY_FAILURE"}
                    for v in sample_outcomes
                ]),
            }
            support_by_stratum = {
                name: local_support(obs, act, deterministic, geometry_ids=sample_geometry, query_mask=mask)
                for name, mask in support_masks.items()
            }
            replay_integrity = {"loaded": False}
            if step == 10000:
                rb = model.replay_buffer
                n = int(rb.pos if not rb.full else rb.buffer_size)
                full_obs = rb.observations[:n, 0].astype(np.float32)
                full_actions = rb.actions[:n, 0].astype(np.float32)
                rows = trace[:n]
                trace_actions = np.asarray([r["replay_action"] for r in rows], dtype=np.float32)
                trace_rewards = np.asarray([r["reward"] for r in rows], dtype=np.float32)
                trace_speed = np.asarray([r["speed_km_h"] for r in rows], dtype=np.float32)
                trace_delta = np.asarray([r["delta_route"] for r in rows], dtype=np.float32)
                trace_outcome = [r.get("outcome") for r in rows]
                timeout_trace = np.asarray([bool(r["truncated"] and not r["terminated"]) for r in rows])
                replay_integrity = {
                    "loaded": True, "transition_count": n,
                    "action_mismatch_count": int(np.count_nonzero(np.abs(full_actions-trace_actions) > 1e-6)),
                    "reward_mismatch_count": int(np.count_nonzero(np.abs(rb.rewards[:n, 0]-trace_rewards) > 1e-6)),
                    "done_mismatch_count": int(np.count_nonzero((rb.dones[:n, 0] > .5) != np.asarray([r["terminated"] or r["truncated"] for r in rows]))),
                    "timeout_mismatch_count": int(np.count_nonzero((rb.timeouts[:n, 0] > .5) != timeout_trace)),
                    "exact_episode_outcome_counts": dict(Counter(outcome_name(x) for x in trace_outcome if x is not None)),
                }
                full_groups = transition_groups(full_obs, trace_speed, trace_delta, trace_rewards, full_actions, trace_outcome)
                full_turn = turn_score(full_obs)
                # Actor-vs-replay action support on exact sampled full-buffer state rows.
                with torch.no_grad():
                    full_mean, _, _ = model.policy.actor.get_action_dist_params(torch.as_tensor(full_obs, dtype=torch.float32))
                    full_det = torch.tanh(full_mean).numpy()
                full_geometry = [r["geometry_id"] for r in rows]
                full_support = local_support(full_obs, full_actions, full_det, geometry_ids=full_geometry)
                replay_integrity["full_replay_groups"] = full_groups
                turn_mask = full_turn >= .05
                moving_mask = trace_speed >= 1.0
                replay_integrity["coverage_intersections"] = {
                    "moving_and_turn_context": int(np.sum(moving_mask & turn_mask)),
                    "moving_without_turn_context": int(np.sum(moving_mask & ~turn_mask)),
                    "stationary_turn_context": int(np.sum(~moving_mask & turn_mask)),
                    "stationary_nonturn": int(np.sum(~moving_mask & ~turn_mask)),
                    "note": "Categories use speed <1 km/h as stationary and task-waypoint curvature score >=0.05 rad as turn context; these describe replay state coverage, not successful control.",
                }
                replay_integrity["full_replay_turn_score"] = qstats(full_turn)
                replay_integrity["full_replay_actor_support"] = full_support
                full_support_masks = {
                    "stationary_lt_1_kmh": trace_speed < 1,
                    "moving_ge_1_kmh": trace_speed >= 1,
                    "positive_progress": trace_delta > 0,
                    "turn_context_ge_0_05rad": full_turn >= .05,
                    "safety_failure_terminal": np.asarray([
                        isinstance(v, dict) and outcome_name(v) in {"OUT_OF_ROAD", "CRASH", "SAFETY_FAILURE"}
                        for v in trace_outcome
                    ]),
                }
                replay_integrity["full_replay_actor_support_by_stratum"] = {
                    name: local_support(full_obs, full_actions, full_det, geometry_ids=full_geometry, query_mask=mask)
                    for name, mask in full_support_masks.items()
                }
                torch.manual_seed(20263121 + (0 if arm == "A" else 1))
                replay_integrity["critic_learning_signal_by_replay_stratum"] = critic_replay_strata(
                    model, rb, full_obs, trace_speed, trace_delta, trace_outcome,
                    seed=20264121 + (0 if arm == "A" else 1),
                )
                replay_integrity["full_replay_sha256"] = metadata.get("replay_buffer_sha256")
            steps_out[str(step)] = {
                "checkpoint_sha256": metadata["model_sha256"], "sample_sha256": sha(sample_path),
                "replay_sample_groups": sample_groups, "actor_behavior": {
                    "deterministic_actions": {"steering": qstats(deterministic[:, 0]), "throttle_brake": qstats(deterministic[:, 1])},
                    "stochastic_actions": {"steering": qstats(stochastic[:, 0]), "throttle_brake": qstats(stochastic[:, 1])},
                    "deterministic_vs_stochastic_l2": qstats(np.linalg.norm(deterministic-stochastic, axis=1)),
                    "actor_mean_log_std": float(np.asarray(log_std.detach().cpu()).mean()),
                },
                "fixed_anchor_objective": objective, "replay_sample_objective": replay_objective,
                "local_action_support": support, "local_action_support_by_replay_stratum": support_by_stratum,
                "replay_integrity_and_consequences": replay_integrity,
            }
            del model
        output["arms"][arm] = {
            "steps": steps_out,
            "training_trace": {
                "transition_count": len(trace),
                "sampling_modes": dict(Counter(str(r.get("mode")) for r in trace)),
                "replay_action_mismatch_count": int(sum(np.max(np.abs(np.asarray(r["replay_action"])-np.asarray(r["executed_action"]))) > 1e-7 for r in trace)),
                "episode_outcome_counts_from_terminal_transition": dict(Counter(outcome_name(r.get("outcome")) for r in trace if r.get("outcome") is not None)),
                "throttle_behavior": {
                    "positive_fraction": float(np.mean([r["replay_action"][1] > .05 for r in trace])),
                    "brake_fraction": float(np.mean([r["replay_action"][1] < -.05 for r in trace])),
                    "neutral_fraction": float(np.mean([abs(r["replay_action"][1]) <= .05 for r in trace])),
                    "switch_count": int(sum((r["replay_action"][1] > 0) != (trace[i-1]["replay_action"][1] > 0) for i, r in enumerate(trace) if i > 0 and r["episode_index"] == trace[i-1]["episode_index"])),
                },
                "outcome_action_context": outcome_context(trace),
            },
        }
    output["comparison"] = {
            "deterministic_full_episode_outcomes_10k": {
            arm: {
                "outcome_counts": json.loads((RUN / f"evaluation/arm_{arm}/step_10000_full.json").read_text(encoding="utf-8"))["outcome_counts"],
                "aggregate": json.loads((RUN / f"evaluation/arm_{arm}/step_10000_full.json").read_text(encoding="utf-8"))["aggregate"],
            }
            for arm in ("A", "B")
        },
        "interpretation": "P2-D B's stochastic training episode outcomes differ from deterministic full-episode evaluation; do not treat them as contradictory or interchangeable. Evaluation is deterministic actor means, training collection is stochastic SAC action sampling.",
    }
    output["source_sha256"] = {p.relative_to(ROOT).as_posix(): sha(p) for p in [Path(__file__), ROOT/"scripts/run_stage5_p2d_actor_activation.py", ROOT/"scripts/audit_stage5_p2d_actor_layer_gradients.py"]}
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "evidence.json"
    target.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    digest = sha(target)
    (OUT / "evidence.sha256").write_text(f"{digest}  evidence.json\n", encoding="ascii")
    print(json.dumps({"evidence": target.relative_to(ROOT).as_posix(), "sha256": digest}, indent=2))


def outcome_context(trace: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({outcome_name(r.get("outcome")) for r in trace if r.get("outcome") is not None})
    out = {}
    for key in keys:
        rows = [r for r in trace if outcome_name(r.get("outcome")) == key]
        out[key] = {
            "terminal_transition_count": len(rows),
            "terminal_speed_km_h": qstats([r["speed_km_h"] for r in rows]),
            "terminal_delta_route": qstats([r["delta_route"] for r in rows]),
            "terminal_throttle_brake": qstats([r["replay_action"][1] for r in rows]),
            "terminal_steering": qstats([r["replay_action"][0] for r in rows]),
            "caution": "Association at terminal transitions only; not causal action effect.",
        }
    return out


if __name__ == "__main__":
    main()
