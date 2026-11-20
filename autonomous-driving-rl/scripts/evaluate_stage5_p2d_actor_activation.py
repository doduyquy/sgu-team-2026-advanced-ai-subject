"""Frozen, TRAIN-only evaluation and representation audit for S5 P2-D."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import evaluate_stage5_p2 as p2eval
from scripts import run_stage5_p2d_actor_activation as p2d
from src.training.stage5.actor_activation_ablation import (
    ACTOR_LEAKY_RELU_001_V1, ACTOR_RELU_V1, activation_inventory,
)
from src.training.stage5.checkpoint import load_sac_checkpoint
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER

ANCHORS = ROOT / "results/stage5/p2c_representation_critic_audit_20261121/fixed_train_anchors.npy"
SLICES = {"ego_9d": slice(0, 9), "navigation_10d": slice(9, 19), "lidar_240d": slice(19, 259),
          "traffic_66d": slice(259, 325), "task_84d": slice(325, 409)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_observations(model, observations: np.ndarray, name: str) -> dict:
    obs_np = np.asarray(observations, dtype=np.float32)
    if obs_np.ndim != 2 or obs_np.shape[1] != 409 or len(obs_np) == 0:
        return {"source": name, "count": int(len(obs_np)), "status": "NO_VALID_OBSERVATIONS"}
    actor = model.policy.actor
    tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=model.device)
    tensor.requires_grad_(True)
    features = actor.extract_features(tensor, actor.features_extractor)
    pre = actor.latent_pi[0](features)
    activation = actor.latent_pi[1]
    post = activation(pre)
    actions = actor(tensor, deterministic=True)
    if isinstance(activation, torch.nn.LeakyReLU):
        derivative = torch.where(pre > 0, torch.ones_like(pre), torch.full_like(pre, activation.negative_slope))
        derivative_kind = "LeakyReLU derivative 1 on positive, slope on negative; negative inputs are not dead"
    else:
        derivative = (pre > 0).to(pre.dtype)
        derivative_kind = "ReLU derivative zero on nonpositive inputs"
    jacobian: dict[str, float] = {}
    for action_index, action_name in enumerate(("steering", "throttle_brake")):
        grad = torch.autograd.grad(actions[:, action_index].sum(), tensor, retain_graph=True)[0]
        jacobian[action_name] = float(torch.linalg.vector_norm(grad, dim=0).detach().cpu().mean())
    group_jacobian = {}
    # Per-observation group Jacobian norms (summed over input dimensions).
    group_grads = {group: {"steering": [], "throttle_brake": []} for group in SLICES}
    for action_index, action_name in enumerate(("steering", "throttle_brake")):
        grad = torch.autograd.grad(actions[:, action_index].sum(), tensor, retain_graph=True)[0]
        for group, slc in SLICES.items():
            group_grads[group][action_name] = torch.linalg.vector_norm(grad[:, slc], dim=1).detach().cpu().numpy()
    for group in SLICES:
        group_jacobian[group] = {f"{action}_mean_l2": float(np.mean(group_grads[group][action]))
                                 for action in ("steering", "throttle_brake")}

    sampled_actions, log_prob = actor.action_log_prob(tensor.detach())
    q1, q2 = model.policy.critic(tensor.detach(), sampled_actions)
    alpha = model.log_ent_coef.detach().exp() if getattr(model, "log_ent_coef", None) is not None else torch.as_tensor(model.ent_coef, device=model.device)
    entropy_loss = (alpha * log_prob.reshape(-1, 1)).mean()
    q_loss = -torch.minimum(q1, q2).mean()
    full_actor_loss = entropy_loss + q_loss
    parameters = list(actor.named_parameters())
    entropy_grads = torch.autograd.grad(entropy_loss, [p for _, p in parameters], retain_graph=True, allow_unused=True)
    q_grads = torch.autograd.grad(q_loss, [p for _, p in parameters], retain_graph=True, allow_unused=True)
    total_grads = torch.autograd.grad(full_actor_loss, [p for _, p in parameters], allow_unused=True)
    def layer_norms(grads):
        values = {}
        for (param_name, _), grad in zip(parameters, grads):
            key = param_name.split(".")[0]
            values[key] = values.get(key, 0.0) + (float(grad.detach().pow(2).sum().cpu()) if grad is not None else 0.0)
        return {key: float(np.sqrt(value)) for key, value in values.items()}
    with torch.no_grad():
        mu, log_std, _ = actor.get_action_dist_params(tensor.detach())
        q_gap = (q1 - q2).abs()
    return {
        "source": name, "count": len(obs_np), "activation": type(activation).__name__,
        "negative_slope": float(activation.negative_slope) if isinstance(activation, torch.nn.LeakyReLU) else None,
        "deterministic_action_mean": actions.detach().mean(0).cpu().tolist(),
        "deterministic_action_std_across_observations": actions.detach().std(0, unbiased=False).cpu().tolist(),
        "actor_mean_std": mu.detach().std(0, unbiased=False).cpu().tolist(),
        "log_std_mean": log_std.detach().mean(0).cpu().tolist(), "alpha": float(alpha.cpu()),
        "first_layer_preactivation": {
            "mean": float(pre.detach().mean().cpu()), "std": float(pre.detach().std(unbiased=False).cpu()),
            "positive_fraction": float((pre.detach() > 0).float().mean().cpu()),
            "negative_fraction": float((pre.detach() < 0).float().mean().cpu()),
            "zero_fraction": float((pre.detach() == 0).float().mean().cpu()),
            "derivative_mean": float(derivative.detach().mean().cpu()),
            "zero_derivative_fraction": float((derivative == 0).float().mean().cpu()),
            "neurons_never_positive": int((pre.detach() <= 0).all(0).sum().cpu()),
            "neurons_with_constant_activation_std_le_1e_8": int((post.detach().std(0, unbiased=False) <= 1e-8).sum().cpu()),
            "derivative_interpretation": derivative_kind,
        },
        "action_input_jacobian_mean_l2": jacobian,
        "action_input_jacobian_by_observation_group": group_jacobian,
        "actor_loss_gradient_norm_by_layer": {
            "entropy_component": layer_norms(entropy_grads), "negative_q_component": layer_norms(q_grads),
            "combined_actor_objective": layer_norms(total_grads),
        },
        "online_critic_snapshot_on_policy_sample": {
            "q1_mean": float(q1.detach().mean().cpu()), "q2_mean": float(q2.detach().mean().cpu()),
            "absolute_twin_q_gap_mean": float(q_gap.detach().mean().cpu()),
            "sampled_log_prob_mean": float(log_prob.detach().mean().cpu()),
            "warning": "Q values are learned estimates, not environment ground truth",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/stage5_p2d/s5_p2d_actor_activation_seed101_20261121")
    parser.add_argument("--arm", choices=("A", "B", "BOTH"), default="BOTH")
    parser.add_argument("--checkpoint-step", type=int, choices=(0, 2000, 5000, 10000), action="append")
    args = parser.parse_args()
    protocol = p2d.read_protocol()
    anchors = np.load(ANCHORS, allow_pickle=False)
    if anchors.shape != (201, 409) or sha256(ANCHORS) != protocol["anchors"]["sha256"]:
        raise ValueError("P2-C fixed anchor array is missing, changed, or has unexpected shape")
    arms = ("A", "B") if args.arm == "BOTH" else (args.arm,)
    steps = args.checkpoint_step or [0, 2000, 5000, 10000]
    encoder = Stage5ObservationEncoder(core_only=False)
    for arm in arms:
        expected = ACT_ID = protocol["arms"][arm]["activation_id"]
        for step in steps:
            checkpoint = args.run_dir / f"arm_{arm}/checkpoints/step_{step:06d}"
            model, metadata = load_sac_checkpoint(
                checkpoint, device="cpu", expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
                task_context_producer_id=REFERENCE_TASK_PRODUCER, expected_actor_activation_id=expected,
            )
            if activation_inventory(model.policy)["actor_id"] != ACT_ID:
                raise RuntimeError("Loaded activation differs from registered arm identity")
            output = args.run_dir / "evaluation" / f"arm_{arm}"
            output.mkdir(parents=True, exist_ok=True)
            probe = p2eval.run_one(checkpoint, protocol, output, cap=300)
            if step == 10000:
                full = p2eval.run_one(checkpoint, protocol, output, cap=None)
            else:
                full = None
            anchor_metrics = audit_observations(model, anchors, "fixed_P2C_201_train_anchors")
            sample_path = args.run_dir / f"arm_{arm}/replay_samples/observations_step_{step:05d}.npz"
            replay_metrics = None
            if sample_path.is_file():
                sample = np.load(sample_path, allow_pickle=False)
                obs = sample["observations"]
                replay_metrics = {
                    "sample_sha256": sha256(sample_path), "sample_count": int(len(obs)),
                    "speed_below_1_fraction": float(np.mean(sample["speed_km_h"] < 1.0)),
                    "positive_progress_fraction": float(np.mean(sample["delta_route"] > 0)),
                    "moving_states": audit_observations(model, obs[sample["speed_km_h"] >= 1.0], "sampled_replay_moving"),
                    "stationary_states": audit_observations(model, obs[sample["speed_km_h"] < 1.0], "sampled_replay_stationary"),
                    "positive_progress_states": audit_observations(model, obs[sample["delta_route"] > 0], "sampled_replay_positive_progress"),
                }
            result = {
                "protocol_sha256": protocol["protocol_sha256"], "arm": arm, "checkpoint_step": step,
                "checkpoint_sha256": metadata["model_sha256"], "checkpoint_activation": metadata["actor_activation"],
                "checkpoint_activation_fingerprint": metadata["activation_fingerprint"],
                "fixed_anchor_sha256": sha256(ANCHORS), "fixed_anchor_metrics": anchor_metrics,
                "replay_sample_metrics": replay_metrics,
                "probe_result_sha256": probe["result_sha256"],
                "full_episode_result_sha256": full["result_sha256"] if full else None,
                "benchmark_eligible": False, "official_validation_test_accessed": False,
            }
            path = output / f"activation_diagnostics_step_{step:05d}.json"
            path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            print(f"{arm} step={step} activation={metadata['actor_activation']} anchor_action_std={anchor_metrics['deterministic_action_std_across_observations']}", flush=True)
            del model


if __name__ == "__main__":
    main()
