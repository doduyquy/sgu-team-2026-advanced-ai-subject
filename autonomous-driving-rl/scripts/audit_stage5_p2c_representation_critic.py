"""Read-only representation/critic audit for the frozen P2 TRAIN experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import Linear, ReLU

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/stage5_p2/s5_p2_correlated_exploration_seed101_20261121_retry1"
PROTOCOL = ROOT / "results/stage5/p2_correlated_exploration_seed101_20261120/protocol.json"
SUMMARY = ROOT / "results/stage5/p2_correlated_exploration_seed101_20261121/result_summary.json"
OUT = ROOT / "results/stage5/p2c_representation_critic_audit_20261121"
STEPS = (0, 2000, 5000, 10000)
GROUPS = {
    "ego_9d": (0, 9), "navigation_10d": (9, 19), "lidar_240d": (19, 259),
    "traffic_66d": (259, 325), "task_84d": (325, 409),
}
DEVICE = torch.device("cpu")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def quantiles(values: np.ndarray | list[float]) -> dict[str, float | None]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if not x.size:
        return {"count": 0, "mean": None, "std": None, "p50": None, "p90": None, "min": None, "max": None}
    return {"count": int(x.size), "mean": float(np.mean(x)), "std": float(np.std(x)), "p50": float(np.quantile(x, .5)),
            "p90": float(np.quantile(x, .9)), "min": float(np.min(x)), "max": float(np.max(x))}


def path_for_eval(arm: str, step: int, full: bool = False) -> Path:
    suffix = "full" if full else "probe"
    return RUN / f"evaluation/arm_{arm}/step_{step:05d}_{suffix}.json"


def obs_path_for_eval(arm: str, step: int, full: bool = False) -> Path:
    suffix = "full" if full else "probe"
    return RUN / f"evaluation/arm_{arm}/actor_observations_step_{step:05d}_{suffix}.npy"


def curvature_score(obs: np.ndarray) -> float:
    valid = np.asarray(obs[385:405] > .5)
    if not valid.any():
        return 0.0
    headings = obs[325:385].reshape(20, 3)[valid, 2] * math.pi
    if headings.size < 2:
        return float(abs(headings[0])) if headings.size else 0.0
    return float(max(np.max(np.abs(headings)), np.max(np.abs(np.diff(headings)))))


def build_anchors() -> tuple[np.ndarray, list[dict[str, Any]], dict[str, str]]:
    records: list[tuple[np.ndarray, dict[str, Any]]] = []
    source_hashes: dict[str, str] = {}
    # These are genuine evaluator observations generated on the frozen TRAIN-only
    # diagnostic schedule. The union is fixed once and reused at every checkpoint.
    sources = [(arm, step, False) for arm in ("A", "B") for step in (0, 2000, 5000, 10000)]
    sources += [(arm, 10000, True) for arm in ("A", "B")]
    for arm, step, full in sources:
        result_path, states_path = path_for_eval(arm, step, full), obs_path_for_eval(arm, step, full)
        result = read_json(result_path)
        observations = np.load(states_path, allow_pickle=False)
        if observations.ndim != 2 or observations.shape[1] != 409 or not np.isfinite(observations).all():
            raise ValueError(f"Invalid saved TRAIN observations: {states_path}")
        source_hashes[result_path.relative_to(ROOT).as_posix()] = file_sha(result_path)
        source_hashes[states_path.relative_to(ROOT).as_posix()] = file_sha(states_path)
        cursor = 0
        for case in result["cases"]:
            count = int(case["sampled_real_observation_count"])
            case_obs = observations[cursor:cursor + count]
            cursor += count
            if len(case_obs) != count:
                raise ValueError(f"Observation/case alignment failure: {states_path}")
            # Keep a deterministic, evenly-spaced sample per case and source. This
            # bounds analysis cost while retaining stationary-to-moving variation.
            take = min(4, count)
            indices = np.unique(np.linspace(0, max(count - 1, 0), take, dtype=int)) if count else np.array([], dtype=int)
            # Also retain up to four highest observed normalized ego-speed values
            # from 2k captures, where the frozen probes actually moved.
            if step == 2000 and count:
                indices = np.unique(np.concatenate([indices, np.argsort(case_obs[:, 3])[-min(4, count):]]))
            for index in indices:
                obs = np.asarray(case_obs[int(index)], dtype=np.float32)
                key = hashlib.sha256(obs.tobytes()).hexdigest()
                records.append((obs, {
                    "anchor_sha256": key, "geometry_id": case["geometry_id"],
                    "environment_seed": int(case["environment_seed"]), "source_arm": arm,
                    "source_checkpoint": int(step), "source_kind": "full_episode" if full else "300_step_probe",
                    "ego_speed_feature_3": float(obs[3]), "turn_context_score_rad": curvature_score(obs),
                }))
        if cursor != len(observations):
            raise ValueError(f"Unassigned observation rows: {states_path}, rows={len(observations)} assigned={cursor}")

    # Deduplicate bit-identical observations while preserving deterministic order.
    unique: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    for obs, meta in records:
        unique.setdefault(meta["anchor_sha256"], (obs, meta))
    ordered = [unique[key] for key in sorted(unique)]
    anchors = np.stack([item[0] for item in ordered]).astype(np.float32)
    metadata = [item[1] for item in ordered]
    return anchors, metadata, source_hashes


def layer_activation_metrics(actor: torch.nn.Module, obs: torch.Tensor) -> dict[str, Any]:
    x = obs
    metrics = []
    linear_i = 0
    for module in actor.latent_pi:
        if isinstance(module, Linear):
            pre = module(x)
            post = pre
            is_actor_first = linear_i == 0
            linear_i += 1
            # Follow the actual immediately subsequent activation module.
            next_module = actor.latent_pi[list(actor.latent_pi).index(module) + 1]
            if isinstance(next_module, ReLU):
                post = torch.relu(pre)
            positive = post > 0
            metrics.append({
                "linear_index": linear_i - 1,
                "preactivation": quantiles(pre.detach().cpu().numpy()),
                "post_activation": quantiles(post.detach().cpu().numpy()),
                "nonpositive_fraction": float((pre <= 0).float().mean().cpu()),
                "zero_activation_fraction": float((post == 0).float().mean().cpu()),
                "neurons_never_positive_count": int((positive.sum(dim=0) == 0).sum().cpu()),
                "neuron_count": int(post.shape[1]),
                "activation_feature_std_mean": float(post.detach().std(dim=0, unbiased=False).mean().cpu()),
                "weight_l2": float(torch.linalg.vector_norm(module.weight).detach().cpu()),
                "weight_frobenius": float(torch.linalg.matrix_norm(module.weight).detach().cpu()),
                "bias_l2": float(torch.linalg.vector_norm(module.bias).detach().cpu()),
                "bias_quantiles": quantiles(module.bias.detach().cpu().numpy()),
            })
            x = post
        elif isinstance(module, ReLU):
            # ReLU has already been applied to obtain post above; skip duplicate.
            continue
        else:
            x = module(x)
    return {"layers": metrics}


def actor_metrics(model: Any, anchors: np.ndarray, anchor_meta: list[dict[str, Any]]) -> dict[str, Any]:
    actor = model.actor.eval()
    obs = torch.as_tensor(anchors, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        mean, log_std, _ = actor.get_action_dist_params(obs)
        deterministic = torch.tanh(mean)
        layer_data = layer_activation_metrics(actor, obs)
    # Exact per-anchor action Jacobians over the deterministic squashed mean.
    def single_action(x: torch.Tensor) -> torch.Tensor:
        batch = x.unsqueeze(0)
        features = actor.features_extractor(batch)
        latent = actor.latent_pi(features)
        return torch.tanh(actor.mu(latent).squeeze(0))
    jacobian = torch.func.vmap(torch.func.jacrev(single_action))(obs)
    jacobian_np = jacobian.detach().cpu().numpy()
    group_jacobian = {}
    for name, (start, end) in GROUPS.items():
        chunk = jacobian_np[:, :, start:end]
        # RMS sensitivity of action to a one empirical-SD perturbation in group.
        group_std = np.std(anchors[:, start:end], axis=0, ddof=0)
        scaled = chunk * group_std[None, None, :]
        group_jacobian[name] = {
            "jacobian_abs_mean_per_action": np.mean(np.abs(chunk), axis=(0, 2)).tolist(),
            "one_observed_sd_rms_per_action": np.sqrt(np.mean(np.square(scaled), axis=(0, 2))).tolist(),
            "anchor_feature_variation_mean_sd": float(np.mean(group_std)),
            "anchor_nonconstant_feature_count": int(np.count_nonzero(group_std > 1e-8)),
        }
    sens_subsets = {}
    speed = anchors[:, 3]
    turn = np.asarray([x["turn_context_score_rad"] for x in anchor_meta])
    masks = {
        "all_fixed_real_anchors": np.ones(len(anchors), dtype=bool),
        "higher_normalized_ego_speed_feature_gt_0_02": speed > .02,
        "lower_normalized_ego_speed_feature_le_0_02": speed <= .02,
        "turn_context_score_at_least_0_05rad": turn >= .05,
    }
    activation_by_subset, jacobian_by_subset = {}, {}
    for name, mask in masks.items():
        if not mask.any():
            sens_subsets[name] = {"count": 0}
            activation_by_subset[name] = {"count": 0}
            jacobian_by_subset[name] = {"count": 0}
            continue
        sens_subsets[name] = {
            "count": int(mask.sum()),
            "steering_mean_std": [float(deterministic[mask, 0].mean()), float(deterministic[mask, 0].std(unbiased=False))],
            "throttle_mean_std": [float(deterministic[mask, 1].mean()), float(deterministic[mask, 1].std(unbiased=False))],
            "log_std_mean": log_std[mask].mean(dim=0).detach().cpu().tolist(),
            "log_std_std": log_std[mask].std(dim=0, unbiased=False).detach().cpu().tolist(),
        }
        activation_by_subset[name] = {"count": int(mask.sum()), **layer_activation_metrics(actor, obs[mask])}
        jacobian_by_subset[name] = {"count": int(mask.sum()), "groups": {}}
        for group_name, (start, end) in GROUPS.items():
            chunk = jacobian_np[mask, :, start:end]
            feature_std = np.std(anchors[mask, start:end], axis=0, ddof=0)
            jacobian_by_subset[name]["groups"][group_name] = {
                "one_observed_sd_rms_per_action": np.sqrt(np.mean(np.square(chunk * feature_std[None, None, :]), axis=(0, 2))).tolist(),
                "anchor_nonconstant_feature_count": int(np.count_nonzero(feature_std > 1e-8)),
            }
    # Offline snapshot actor-gradient decomposition on fixed real anchors.
    sampled, log_prob = actor.action_log_prob(obs)
    q1, q2 = model.critic(obs, sampled)
    alpha = float(model.log_ent_coef.detach().exp().cpu()) if getattr(model, "log_ent_coef", None) is not None else float(model.ent_coef)
    entropy_term = (alpha * log_prob).mean()
    q_term = -torch.minimum(q1, q2).mean()
    losses = {"entropy_term": entropy_term, "negative_q_term": q_term, "combined_actor_objective": entropy_term + q_term}
    grad_components = {}
    params = list(actor.parameters())
    for loss_name, loss in losses.items():
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        per_name = defaultdict(float)
        for (name, _), grad in zip(actor.named_parameters(), grads):
            if grad is not None:
                layer = "first_linear" if name.startswith("latent_pi.0.") else "second_linear" if name.startswith("latent_pi.2.") else "output_heads"
                per_name[layer] += float(grad.detach().pow(2).sum().cpu())
        grad_components[loss_name] = {name: math.sqrt(value) for name, value in per_name.items()}
    return {
        "action_mean": {
            "steering": quantiles(deterministic[:, 0].detach().cpu().numpy()),
            "throttle_brake": quantiles(deterministic[:, 1].detach().cpu().numpy()),
        },
        "log_std": {"steering": quantiles(log_std[:, 0].detach().cpu().numpy()),
                    "throttle_brake": quantiles(log_std[:, 1].detach().cpu().numpy())},
        "output_head_parameters": {
            name: {
                "weight_l2": float(torch.linalg.vector_norm(module.weight).detach().cpu()),
                "weight_frobenius": float(torch.linalg.matrix_norm(module.weight).detach().cpu()),
                "bias_l2": float(torch.linalg.vector_norm(module.bias).detach().cpu()),
                "bias": quantiles(module.bias.detach().cpu().numpy()),
            }
            for name, module in (("mu", actor.mu), ("log_std", actor.log_std))
        },
        "alpha": alpha,
        "actor_latent_layers": layer_data["layers"],
        "actor_latent_layers_by_real_state_subset": activation_by_subset,
        "jacobian_by_input_group": group_jacobian,
        "jacobian_by_real_state_subset": jacobian_by_subset,
        "action_behavior_by_real_state_subset": sens_subsets,
        "offline_actor_gradient_norm_by_component": grad_components,
        "gradient_semantics": "offline objective snapshot on fixed TRAIN observations; not historical optimizer gradients",
    }


def load_replay_records(arm: str, replay: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace_path = RUN / f"arm_{arm}/replay_action_trace.jsonl"
    trace = read_jsonl(trace_path)
    if len(trace) != replay.size():
        raise ValueError(f"Trace/replay count differs for {arm}: {len(trace)} != {replay.size()}")
    n = len(trace)
    actions = replay.actions[:n, 0]
    rewards = replay.rewards[:n, 0]
    dones = replay.dones[:n, 0]
    timeouts = replay.timeouts[:n, 0]
    mismatch_actions = mismatch_rewards = mismatch_dones = 0
    for i, row in enumerate(trace):
        logged = np.asarray(row["replay_action"], dtype=np.float32)
        mismatch_actions += int(not np.array_equal(logged, actions[i]))
        mismatch_rewards += int(abs(float(row["reward"]) - float(rewards[i])) > 1e-7)
        expected_done = bool(row["terminated"] or row["truncated"])
        mismatch_dones += int(bool(dones[i]) != expected_done or bool(timeouts[i]) != bool(row["truncated"]))
    if mismatch_actions or mismatch_rewards or mismatch_dones:
        raise ValueError(f"Replay fidelity failed {arm}: action={mismatch_actions}, reward={mismatch_rewards}, done={mismatch_dones}")
    info = {
        "transition_count": n, "trace_sha256": file_sha(trace_path),
        "action_mismatch_count": mismatch_actions, "reward_mismatch_count": mismatch_rewards,
        "termination_timeout_mismatch_count": mismatch_dones,
        "replay_sha256": file_sha(RUN / f"arm_{arm}/checkpoints/step_010000/replay_buffer.pkl"),
        "replay_observation_shape": list(replay.observations[:n, 0].shape),
        "episode_boundaries": int(np.sum(np.asarray(dones, dtype=bool))),
    }
    return trace, info


def replay_policy_representation(model: Any, trace: list[dict[str, Any]]) -> dict[str, Any]:
    rb = model.replay_buffer
    n = len(trace)
    obs_np = np.asarray(rb.observations[:n, 0], dtype=np.float32)
    action_np = np.asarray(rb.actions[:n, 0], dtype=np.float32)
    speed = np.asarray([float(row["speed_km_h"]) for row in trace])
    delta = np.asarray([float(row["delta_route"]) for row in trace])
    safety = np.asarray(["OUT_OF_ROAD" in json.dumps(row.get("outcome")) for row in trace], dtype=bool)
    turns = np.asarray([curvature_score(row) for row in obs_np], dtype=np.float64)
    first_wp_y = []
    for row in obs_np:
        valid = np.flatnonzero(row[385:405] > .5)
        first_wp_y.append(float(row[325 + 3 * int(valid[0]) + 1] * 50.0) if len(valid) else 0.0)
    first_wp_y = np.asarray(first_wp_y, dtype=np.float64)
    feature_variation = {}
    for group_name, (start, end) in GROUPS.items():
        std = np.std(obs_np[:, start:end], axis=0, ddof=0)
        feature_variation[group_name] = {"mean_per_feature_sd": float(np.mean(std)),
                                         "nonconstant_feature_count": int(np.count_nonzero(std > 1e-8)),
                                         "feature_count": end - start}
    masks = {
        "all_replay_states": np.ones(n, dtype=bool),
        "stationary_speed_lt_1kmh": speed < 1.0,
        "moving_speed_ge_1kmh": speed >= 1.0,
        "turn_context_score_ge_0_05rad": turns >= .05,
        "abs_first_valid_waypoint_y_ge_0_25m": np.abs(first_wp_y) >= .25,
        "positive_route_progress": delta > 0,
        "safety_failure_transition": safety,
    }
    result = {}
    actor = model.actor.eval()
    critic = model.critic.eval()
    with torch.no_grad():
        for name, mask in masks.items():
            if not mask.any():
                result[name] = {"count": 0}
                continue
            obs = torch.as_tensor(obs_np[mask], dtype=torch.float32, device=DEVICE)
            action = actor(obs, deterministic=True)
            q1, q2 = critic(obs, torch.as_tensor(action_np[mask], dtype=torch.float32, device=DEVICE))
            result[name] = {
                "count": int(mask.sum()), "speed_kmh": quantiles(speed[mask]),
                "turn_context_score_rad": quantiles(turns[mask]),
                "first_valid_waypoint_y_m": quantiles(first_wp_y[mask]),
                "deterministic_action": {
                    "steering": quantiles(action[:, 0].cpu().numpy()),
                    "throttle_brake": quantiles(action[:, 1].cpu().numpy()),
                },
                "first_layer_activation": layer_activation_metrics(actor, obs)["layers"][0],
                "second_layer_activation": layer_activation_metrics(actor, obs)["layers"][1],
                "critic_q_on_behavior_action": {"q1": quantiles(q1.cpu().numpy()), "q2": quantiles(q2.cpu().numpy())},
                "critic_q_on_deterministic_action": {
                    "q1": quantiles(critic(obs, action)[0].cpu().numpy()),
                    "q2": quantiles(critic(obs, action)[1].cpu().numpy()),
                },
            }
    return {
        "groups": result,
        "real_replay_task_context_coverage": {
            "turn_context_score_rad": quantiles(turns),
            "turn_context_ge_0_05rad_count": int(np.sum(turns >= .05)),
            "first_valid_waypoint_y_m": quantiles(first_wp_y),
            "abs_first_valid_waypoint_y_ge_0_25m_count": int(np.sum(np.abs(first_wp_y) >= .25)),
            "note": "Descriptive strata from actual replay observations; thresholds do not define success or a desired action.",
        },
        "input_feature_group_variation": feature_variation,
        "observation_source": "actual transformed observations in the 10k replay buffer",
        "interpretation_limit": "speed strata are from logged simulator telemetry; replay state/action transitions are aligned and hash verified",
    }


def critic_hidden_representation(model: Any, trace: list[dict[str, Any]]) -> dict[str, Any]:
    rb = model.replay_buffer
    n = len(trace)
    obs_np = np.asarray(rb.observations[:n, 0], dtype=np.float32)
    actions_np = np.asarray(rb.actions[:n, 0], dtype=np.float32)
    speed = np.asarray([float(row["speed_km_h"]) for row in trace])
    delta = np.asarray([float(row["delta_route"]) for row in trace])
    turns = np.asarray([curvature_score(row) for row in obs_np], dtype=np.float64)
    masks = {
        "all_replay_transitions": np.ones(n, dtype=bool),
        "stationary_speed_lt_1kmh": speed < 1.0,
        "moving_speed_ge_1kmh": speed >= 1.0,
        "turn_context_score_ge_0_05rad": turns >= .05,
        "positive_route_progress": delta > 0,
    }
    output = {}
    with torch.no_grad():
        for group_name, mask in masks.items():
            if not mask.any():
                output[group_name] = {"count": 0}
                continue
            x0 = torch.as_tensor(obs_np[mask], dtype=torch.float32, device=DEVICE)
            a = torch.as_tensor(actions_np[mask], dtype=torch.float32, device=DEVICE)
            x0 = torch.cat([x0, a], dim=1)
            q_metrics = {}
            for q_name, qnet in (("q1", model.critic.qf0), ("q2", model.critic.qf1)):
                x = x0
                layers = []
                linear_index = 0
                modules = list(qnet)
                for index, module in enumerate(modules):
                    if isinstance(module, Linear):
                        pre = module(x)
                        post = pre
                        if index + 1 < len(modules) and isinstance(modules[index + 1], ReLU):
                            post = torch.relu(pre)
                        positive = post > 0
                        layers.append({
                            "linear_index": linear_index,
                            "preactivation": quantiles(pre.cpu().numpy()),
                            "post_activation": quantiles(post.cpu().numpy()),
                            "nonpositive_fraction": float((pre <= 0).float().mean().cpu()),
                            "neurons_never_positive_count": int((positive.sum(dim=0) == 0).sum().cpu()),
                            "neuron_count": int(post.shape[1]),
                            "activation_feature_std_mean": float(post.std(dim=0, unbiased=False).mean().cpu()),
                        })
                        linear_index += 1
                        x = post
                    elif isinstance(module, ReLU):
                        continue
                    else:
                        x = module(x)
                q_metrics[q_name] = layers
            output[group_name] = {"count": int(mask.sum()), "q_network_hidden_layers": q_metrics}
    return output


def replay_critic_metrics(model: Any, trace: list[dict[str, Any]], batch_cap: int = 256) -> dict[str, Any]:
    rb = model.replay_buffer
    n = len(trace)
    # Use the replay buffer's own sampler to preserve SB3 timeout bootstrapping.
    samples = rb._get_samples(np.arange(n), env=None)
    obs, actions = samples.observations, samples.actions
    next_obs, rewards, dones = samples.next_observations, samples.rewards, samples.dones
    with torch.no_grad():
        q1, q2 = model.critic(obs, actions)
        next_actions, next_log_prob = model.actor.action_log_prob(next_obs)
        target_q1, target_q2 = model.critic_target(next_obs, next_actions)
        alpha = model.log_ent_coef.detach().exp() if getattr(model, "log_ent_coef", None) is not None else torch.as_tensor(float(model.ent_coef), device=DEVICE)
        next_q = torch.minimum(target_q1, target_q2) - alpha * next_log_prob.reshape(-1, 1)
        target = rewards + (1.0 - dones) * float(model.gamma) * next_q
        td1, td2 = target - q1, target - q2
        actor_actions = model.actor(obs, deterministic=True)
        actor_q1, actor_q2 = model.critic(obs, actor_actions)
    arrays = {
        "speed_kmh": np.asarray([float(row["speed_km_h"]) for row in trace]),
        "delta_route": np.asarray([float(row["delta_route"]) for row in trace]),
        "safety": np.asarray(["OUT_OF_ROAD" in json.dumps(row.get("outcome")) for row in trace], dtype=bool),
    }
    task_turn = np.asarray([curvature_score(row) for row in np.asarray(rb.observations[:n, 0], dtype=np.float32)], dtype=np.float64)
    masks = {
        "all": np.ones(n, dtype=bool),
        "stationary_speed_lt_1kmh": arrays["speed_kmh"] < 1.0,
        "moving_speed_ge_1kmh": arrays["speed_kmh"] >= 1.0,
        "turn_context_score_ge_0_05rad": task_turn >= .05,
        "positive_route_progress": arrays["delta_route"] > 0.0,
        "nonpositive_route_progress": arrays["delta_route"] <= 0.0,
        "safety_failure_transition": arrays["safety"],
    }
    q1n, q2n = q1.detach().cpu().numpy().reshape(-1), q2.detach().cpu().numpy().reshape(-1)
    targetn = target.detach().cpu().numpy().reshape(-1)
    td1n, td2n = td1.detach().cpu().numpy().reshape(-1), td2.detach().cpu().numpy().reshape(-1)
    aq1n, aq2n = actor_q1.detach().cpu().numpy().reshape(-1), actor_q2.detach().cpu().numpy().reshape(-1)
    actor_dist = torch.linalg.vector_norm(actor_actions - actions, dim=1).detach().cpu().numpy()
    groups = {}
    gradient_metrics = {}
    q_action_gradients = {}
    for name, mask_np in masks.items():
        idx = np.flatnonzero(mask_np)
        groups[name] = {
            "count": int(len(idx)), "speed_kmh": quantiles(arrays["speed_kmh"][idx]),
            "delta_route": quantiles(arrays["delta_route"][idx]),
            "q1_replay_action": quantiles(q1n[idx]), "q2_replay_action": quantiles(q2n[idx]),
            "twin_q_abs_gap": quantiles(np.abs(q1n[idx] - q2n[idx])),
            "soft_target_q": quantiles(targetn[idx]),
            "td1_abs": quantiles(np.abs(td1n[idx])), "td2_abs": quantiles(np.abs(td2n[idx])),
            "actor_minus_replay_q1": quantiles(aq1n[idx] - q1n[idx]),
            "actor_minus_replay_q2": quantiles(aq2n[idx] - q2n[idx]),
            "actor_action_distance_from_recorded_action": quantiles(actor_dist[idx]),
            "actor_action_comparison_support": "same state has one behavior action; actor action may be outside local replay action support",
        }
        if len(idx):
            selected = idx[np.linspace(0, len(idx) - 1, min(batch_cap, len(idx)), dtype=int)]
            sample_obs = samples.observations[selected]
            sample_action = samples.actions[selected].detach().clone().requires_grad_(True)
            s_q1, s_q2 = model.critic(sample_obs, sample_action)
            grad_q1 = torch.autograd.grad(s_q1.sum(), sample_action, retain_graph=True)[0]
            grad_q2 = torch.autograd.grad(s_q2.sum(), sample_action)[0]
            q_action_gradients[name] = {
                "sample_count": int(len(selected)),
                "abs_dq1_d_steering": quantiles(grad_q1[:, 0].abs().detach().cpu().numpy()),
                "abs_dq1_d_throttle": quantiles(grad_q1[:, 1].abs().detach().cpu().numpy()),
                "abs_dq2_d_steering": quantiles(grad_q2[:, 0].abs().detach().cpu().numpy()),
                "abs_dq2_d_throttle": quantiles(grad_q2[:, 1].abs().detach().cpu().numpy()),
                "support_note": "local critic derivative at replay-observed actions only",
            }
        if len(idx):
            # Deterministic contiguous sampling from the group; no optimizer step.
            chosen = idx[np.linspace(0, len(idx) - 1, min(batch_cap, len(idx)), dtype=int)]
            batch = rb._get_samples(chosen, env=None)
            with torch.no_grad():
                na, nlp = model.actor.action_log_prob(batch.next_observations)
                tq1, tq2 = model.critic_target(batch.next_observations, na)
                aa = model.log_ent_coef.detach().exp() if getattr(model, "log_ent_coef", None) is not None else torch.as_tensor(float(model.ent_coef), device=DEVICE)
                y = batch.rewards + (1.0 - batch.dones) * float(model.gamma) * (torch.minimum(tq1, tq2) - aa * nlp.reshape(-1, 1))
            model.critic.optimizer.zero_grad(set_to_none=True)
            cq1, cq2 = model.critic(batch.observations, batch.actions)
            loss = torch.nn.functional.mse_loss(cq1, y) + torch.nn.functional.mse_loss(cq2, y)
            loss.backward()
            per_q = {}
            for qname, qnet in (("q1", model.critic.qf0), ("q2", model.critic.qf1)):
                per_q[qname] = {}
                for module_name, module in qnet.named_modules():
                    if isinstance(module, Linear):
                        norm = math.sqrt(sum(float(p.grad.detach().pow(2).sum().cpu()) for p in module.parameters() if p.grad is not None))
                        per_q[qname][module_name] = norm
            model.critic.optimizer.zero_grad(set_to_none=True)
            gradient_metrics[name] = {"sample_count": int(len(chosen)), "critic_mse_sum": float(loss.detach().cpu()), "gradient_l2_by_q_and_linear": per_q,
                                      "gradient_semantics": "offline TD loss gradient recomputed at frozen 10k checkpoint; not historical optimizer gradient"}
    return {
        "alpha": float(alpha.detach().cpu()), "groups": groups,
        "local_action_gradient_at_replay_support": q_action_gradients,
        "offline_critic_gradient_by_group": gradient_metrics,
        "actor_representation_on_replay_states": replay_policy_representation(model, trace),
        "critic_hidden_representation_on_replay_states": critic_hidden_representation(model, trace),
        "replay_data_coverage": {
            "speed_ge_1kmh_fraction": float(np.mean(arrays["speed_kmh"] >= 1.0)),
            "positive_delta_route_fraction": float(np.mean(arrays["delta_route"] > 0)),
            "mean_positive_throttle_run_length": None,
            "episode_outcomes": dict(Counter(
                str(row.get("outcome", "NONE").get("primary_reason", "UNKNOWN")) if isinstance(row.get("outcome"), dict) else str(row.get("outcome") or "NONE")
                for row in trace if row.get("terminated") or row.get("truncated")
            )),
        },
    }


def optimizer_timecourse(arm: str) -> dict[str, Any]:
    path = RUN / f"arm_{arm}/optimizer_telemetry.jsonl"
    rows = read_jsonl(path)
    fields = ("alpha", "actor_loss", "critic_loss", "actor_gradient_global_norm",
              "critic_gradient_global_norm", "entropy_gradient_l2")
    result = {}
    for step in STEPS[1:]:
        available = [row for row in rows if int(row["training_step"]) <= step]
        latest = available[-1] if available else None
        window = [row for row in rows if max(1000, step - 500) < int(row["training_step"]) <= step]
        result[str(step)] = {
            "latest_sample_at_or_before_checkpoint": None if latest is None else {
                "training_step": int(latest["training_step"]), **{key: latest.get(key) for key in fields}
            },
            "preceding_500_steps_sample_count": len(window),
            "preceding_500_steps_median": {
                key: quantiles([float(row[key]) for row in window if row.get(key) is not None]) for key in fields
            },
            "sampling_note": "periodic telemetry snapshots, not every optimizer update",
        }
    return {"telemetry_rows": len(rows), "source_sha256": file_sha(path), "checkpoints": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER
    from src.training.stage5.checkpoint import load_sac_checkpoint
    from src.training.stage5.observation_encoder import Stage5ObservationEncoder

    protocol, summary = read_json(PROTOCOL), read_json(SUMMARY)
    protocol_sha = protocol["protocol_sha256"]
    summary_sha = file_sha(SUMMARY)
    encoder = Stage5ObservationEncoder(core_only=False)
    anchors, anchor_meta, anchor_sources = build_anchors()
    anchor_path = args.output_dir / "fixed_train_anchors.npy"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(anchor_path, anchors)
    fixed_anchor_sha = file_sha(anchor_path)
    if len({m["geometry_id"] for m in anchor_meta}) != 6:
        raise ValueError("Fixed anchor set does not cover all six locked diagnostic TRAIN geometries")
    source_shas = {"protocol": file_sha(PROTOCOL), "result_summary": summary_sha,
                   "anchor_file": fixed_anchor_sha, **anchor_sources}

    checkpoint_rows: list[dict[str, Any]] = []
    models: dict[tuple[str, int], Any] = {}
    for arm in ("A", "B"):
        for step in STEPS:
            checkpoint = RUN / f"arm_{arm}/checkpoints/step_{step:06d}"
            model, metadata = load_sac_checkpoint(
                checkpoint, device="cpu", expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
                task_context_producer_id=REFERENCE_TASK_PRODUCER,
                load_replay_buffer=(step == 10000),
            )
            if metadata["total_timesteps"] != step:
                raise ValueError(f"Checkpoint step mismatch {arm}/{step}")
            if metadata["observation_transform_fingerprint"] != "265103242e3ba3ad6f8c493d8f0bad39644c375a2696dbcdad4857052796797e":
                raise ValueError(f"Observation transform fingerprint mismatch {arm}/{step}")
            metrics = actor_metrics(model, anchors, anchor_meta)
            # Compare critics only on the same fixed anchor observations and actor action.
            x = torch.as_tensor(anchors, dtype=torch.float32)
            with torch.no_grad():
                aa = model.actor(x, deterministic=True)
                q1, q2 = model.critic(x, aa)
            metrics["critic_on_fixed_anchor_policy_action"] = {
                "q1": quantiles(q1.cpu().numpy()), "q2": quantiles(q2.cpu().numpy()),
                "absolute_twin_gap": quantiles((q1 - q2).abs().cpu().numpy()),
                "warning": "critic values are estimates, not empirical returns or ground truth",
            }
            row = {"arm": arm, "checkpoint_step": step, "model_sha256": metadata["model_sha256"],
                   "metadata_sha256": file_sha(checkpoint / "metadata.json"),
                   "producer_id": metadata["task_context_producer"]["id"],
                   "producer_fingerprint": metadata["task_context_producer"]["fingerprint"],
                   "transform_id": metadata["observation_transform"]["transform_id"],
                   "transform_fingerprint": metadata["observation_transform_fingerprint"],
                   "metrics": metrics}
            checkpoint_rows.append(row)
            source_shas[f"checkpoint_{arm}_{step}_model"] = file_sha(checkpoint / "model.zip")
            models[(arm, step)] = model
            print(f"audited actor {arm}/{step}", flush=True)

    replay_results = {}
    for arm in ("A", "B"):
        model = models[(arm, 10000)]
        trace, fidelity = load_replay_records(arm, model.replay_buffer)
        replay_results[arm] = {"fidelity": fidelity, "critic": replay_critic_metrics(model, trace)}
        source_shas[f"replay_trace_{arm}"] = fidelity["trace_sha256"]
        source_shas[f"replay_buffer_{arm}"] = fidelity["replay_sha256"]
        print(f"audited replay/critic {arm}", flush=True)

    # Fixed-anchor status and physical/context coverage are stated from the data,
    # not interpreted as labels of driving competence.
    anchor_summary = {
        "count": int(len(anchors)), "geometry_count": len({m["geometry_id"] for m in anchor_meta}),
        "geometry_ids": sorted({m["geometry_id"] for m in anchor_meta}),
        "anchor_sha256": fixed_anchor_sha,
        "ego_speed_feature_3": quantiles(anchors[:, 3]),
        "task_turn_context_score_rad": quantiles([m["turn_context_score_rad"] for m in anchor_meta]),
        "turn_context_ge_0_05rad_count": int(sum(m["turn_context_score_rad"] >= .05 for m in anchor_meta)),
        "source_kinds": dict(Counter(m["source_kind"] for m in anchor_meta)),
        "source_checkpoint_steps": dict(Counter(str(m["source_checkpoint"]) for m in anchor_meta)),
        "anchor_provenance": anchor_meta,
        "interpretation_note": "All anchors are actual saved observations from fixed internal TRAIN diagnostic episodes; the union was frozen and reused for all eight checkpoints.",
    }
    evidence = {
        "audit_id": "stage5_p2c_representation_critic_audit_v1",
        "status": "COMPLETE_READ_ONLY",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "run_id": RUN.name, "protocol_sha256": protocol_sha, "result_summary_sha256": summary_sha,
        "producer_id": REFERENCE_TASK_PRODUCER,
        "observation_dim": 409, "action_contract": "continuous_box2_v1",
        "official_validation_test_accessed": False, "scientific_training_performed": False,
        "checkpoint_audit": checkpoint_rows, "fixed_anchor_summary": anchor_summary,
        "replay_audit": replay_results,
        "optimizer_timecourse": {arm: optimizer_timecourse(arm) for arm in ("A", "B")},
        "timepoint_limitations": {
            "replay_buffers_saved_at": [10000],
            "replay_buffers_missing_at": [0, 2000, 5000],
            "historical_optimizer_gradient_logs": "not available before P2 training telemetry; P2 optimizer log is periodic and not per-update parameter-gradient history",
            "causal_ordering": "cannot identify whether actor sensitivity loss preceded first-layer inactivity between 2k and 5k; only endpoint snapshots exist",
        },
        "source_sha256": {**source_shas, "audit_script": file_sha(Path(__file__).resolve())},
        "interpretation": {
            "confirmed": [
                "On the fixed 201-anchor set from six internal TRAIN diagnostic geometries, the actor first ReLU layer had all 256 neurons never positive at 5k and 10k for both arms; deterministic action Jacobians were zero across all measured varying input groups at those checkpoints.",
                "At the 10k replay checkpoint, the same first actor layer was inactive for all 10,000 actual replay observations, including 1,680 A and 1,693 B transitions with logged speed at least 1 km/h; deterministic actor output variation was effectively zero in those groups.",
                "Offline actor-objective gradients through the first actor linear layer were zero at 5k and 10k on the fixed real anchors; periodic training telemetry shows global actor gradient norms declined while entropy-gradient magnitude stayed near 3.36.",
                "The 10k replay buffer and action trace align exactly for action, reward, termination and timeout fields in both arms; mismatch counts are zero.",
                "The replay contains both stationary and moving transitions, but only about 16.8-16.9 percent of transitions have speed at least 1 km/h; about 58.6-59.0 percent have positive route delta, usually very small. There are two recorded safety-failure transitions per arm.",
                "The 10k replay contains actual route-turn context (4,351 A and 4,179 B transitions score at least 0.05 rad) and substantial first-waypoint lateral-coordinate variation; actor outputs remain constant on these real states.",
            ],
            "supported_hypotheses": [
                "Dead first-layer ReLUs are a directly observed proximate mechanism that blocks actor gradients from reaching the 409D input on sampled states by 5k/10k; this can explain loss of state sensitivity but does not establish why training entered that regime or prove it is the sole cause of stationary driving.",
                "Critic action dependence is weak at the recorded replay actions in this snapshot and critic hidden units are sparse, but actor-action versus replay-action Q comparisons are not established as in-support because only one behavior action is stored per exact state and mean actor-to-recorded action distance is large.",
            ],
            "unknown": [
                "Whether actor sensitivity collapse precedes first-layer inactivity between 2k and 5k; there are only endpoint model snapshots, not per-update activation or per-layer gradient histories.",
                "Whether critic Q rankings are correct on alternative supported actions; counterfactual action support at matched states is insufficient and Q is not ground truth.",
                "Whether the steering chosen on actual turn-context states is correct for the road: the actor output is constant there, but no intervention or expert/ground-truth steering label is available to assess a desired action.",
                "Whether LiDAR or traffic features are useful: these feature groups are constant across the saved fixed diagnostic anchors, and traffic is absent in this Easy schedule; the replay feature-variation audit narrows but does not establish their causal utility.",
                "Replay buffers at 2k and 5k and the optimizer batches from those periods are absent, so early replay coverage and critic evolution cannot be reconstructed.",
                "Whether reward/entropy interactions, representation collapse, critic limitations, or their combination initiated the behavior.",
            ],
            "decision": "REPRESENTATION_INTERVENTION_JUSTIFIED",
            "decision_scope": "Justifies one controlled representation/activation experiment focused on preserving gradient flow; it is not a finding that activation choice alone causes or will solve stationary driving.",
        },
    }
    out = args.output_dir / "evidence.json"
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "evidence.sha256").write_text(f"{file_sha(out)}  evidence.json\n", encoding="ascii")
    print(f"evidence={out.relative_to(ROOT).as_posix()} sha256={file_sha(out)}", flush=True)


if __name__ == "__main__":
    main()
