"""Read-only mechanism audit of the historical S5.5 checkpoints and TRAIN states."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.platform.task_context_producer import LEGACY_TASK_PRODUCER
from src.training.stage5.checkpoint import load_sac_checkpoint
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.training.stage5.pilot_observation_transform import apply_observation_transform

RUN = ROOT / "runs/stage5_meter_reward/s5_5_meter_reward_seed101_20261120"
RESULTS = ROOT / "results/stage5/s5_5_meter_reward_seed101_20261120"
OUT = ROOT / "results/stage5/p2_mechanism_audit"
TRANSFORM = "pilot_groupwise_physical_v1"
STEPS = (0, 2_000, 5_000, 10_000, 30_000)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _float_list(value: torch.Tensor) -> list[float]:
    return [float(x) for x in value.detach().cpu().reshape(-1).tolist()]


def main() -> None:
    states_path = RUN / "evaluation/policy_response_eval_states.jsonl.gz"
    states_sha = sha256(states_path)
    analysis = json.loads((RESULTS / "analysis_summary.json").read_text(encoding="utf-8"))
    if states_sha != analysis["evaluation_artifacts"]["policy_response_eval_states_sha256"]:
        raise ValueError("Historical TRAIN observation-state file SHA does not match its run manifest")
    protocol = json.loads((RESULTS / "protocol.json").read_text(encoding="utf-8"))
    diagnostics = {(item["geometry_id"], int(item["environment_seed"])) for item in protocol["diagnostic_pool"]}
    rows = load_rows(states_path)
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["arm"], int(row["checkpoint_step"]))
        if (row["geometry_id"], int(row["environment_seed"])) not in diagnostics:
            raise ValueError(f"Historical observation is outside the locked PILOT_DIAGNOSTIC schedule: {row}")
        observation = np.asarray(row["raw_observation"], dtype=np.float32)
        if observation.shape != (409,) or not np.isfinite(observation).all():
            raise ValueError("Invalid non-finite or non-409D historical observation")
        row["_obs"] = apply_observation_transform(observation, TRANSFORM)
        grouped[key].append(row)

    encoder = Stage5ObservationEncoder(core_only=False)
    checkpoints: list[dict] = []
    for arm in ("A", "B", "C"):
        for step in STEPS:
            checkpoint = RUN / f"arms/{arm}/checkpoints/step_{step:06d}"
            checkpoint_rows = grouped.get((arm, step), [])
            if not checkpoint_rows:
                raise ValueError(f"No provenance-matched observations for {arm}/{step}")
            model, metadata = load_sac_checkpoint(
                checkpoint, device="cpu", expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
                task_context_producer_id=LEGACY_TASK_PRODUCER,
            )
            observations = np.stack([row["_obs"] for row in checkpoint_rows]).astype(np.float32)
            tensor = torch.as_tensor(observations, dtype=torch.float32)
            model.policy.set_training_mode(False)
            with torch.no_grad():
                deterministic, _ = model.predict(observations, deterministic=True)
                first_linear = next(layer for layer in model.actor.latent_pi if isinstance(layer, torch.nn.Linear))
                preactivation = first_linear(tensor.to(model.device))
                activation = torch.relu(preactivation)
                actions_tensor = torch.as_tensor(np.asarray(deterministic, dtype=np.float32), device=model.device)
                critic_values = model.critic(tensor.to(model.device), actions_tensor)
                det_q1 = critic_values[0].reshape(-1)
                det_q2 = critic_values[1].reshape(-1)
                det_actions = actions_tensor
                action_grid = []
                for axis, values in ((1, (-0.5, 0.0, 0.5)), (0, (-0.5, 0.0, 0.5))):
                    for value in values:
                        candidate = det_actions.clone()
                        candidate[:, axis] = value
                        q1, q2 = model.critic(tensor.to(model.device), candidate)
                        action_grid.append({
                            "axis": "throttle_brake" if axis == 1 else "steering",
                            "value": value,
                            "mean_q1": float(q1.mean().cpu()),
                            "mean_q2": float(q2.mean().cpu()),
                            "mean_min_q": float(torch.minimum(q1, q2).mean().cpu()),
                            "mean_min_q_delta_from_deterministic": float((torch.minimum(q1, q2).reshape(-1) - torch.minimum(det_q1, det_q2)).mean().cpu()),
                        })

            # Snapshot gradient proxy: recompute the actor objective on these frozen
            # diagnostic states. This is not historical optimizer-gradient telemetry.
            model.actor.optimizer.zero_grad(set_to_none=True)
            obs_grad = tensor.to(model.device)
            sampled_actions, log_prob = model.actor.action_log_prob(obs_grad)
            q_pi = model.critic(obs_grad, sampled_actions)
            alpha = float(model.log_ent_coef.detach().exp().cpu()) if getattr(model, "log_ent_coef", None) is not None else float(model.ent_coef)
            actor_loss_proxy = (alpha * log_prob - torch.minimum(q_pi[0], q_pi[1])).mean()
            actor_loss_proxy.backward()
            grad_sq = math.fsum(float(parameter.grad.detach().pow(2).sum().cpu()) for parameter in model.actor.parameters() if parameter.grad is not None)
            actor_grad_norm_proxy = math.sqrt(grad_sq)
            model.actor.optimizer.zero_grad(set_to_none=True)

            deterministic = np.asarray(deterministic, dtype=np.float64)
            speeds = [float(row["speed_km_h"]) for row in checkpoint_rows]
            curves = [float(row["task_curvature_proxy_max_abs_heading_rad"]) for row in checkpoint_rows]
            feature_sd = {}
            for name, (start, end) in {"ego": (0, 9), "navigation": (9, 19), "lidar": (19, 259), "traffic": (259, 325), "task": (325, 409)}.items():
                values = observations[:, start:end]
                feature_sd[name] = float(np.mean(np.std(values, axis=0)))
            checkpoints.append({
                "arm": arm, "checkpoint_step": step, "checkpoint_model_sha256": metadata["model_sha256"],
                "metadata_producer_id": metadata.get("task_context_producer_id", LEGACY_TASK_PRODUCER),
                "metadata_transform_id": metadata.get("observation_transform"),
                "observation_rows": len(checkpoint_rows),
                "geometry_count": len({row["geometry_id"] for row in checkpoint_rows}),
                "speed_range_km_h": [min(speeds), max(speeds)],
                "curvature_proxy_nonzero_rows": sum(value > 1e-5 for value in curves),
                "mean_per_feature_sd_by_group": feature_sd,
                "deterministic_steering_mean_std": [float(np.mean(deterministic[:, 0])), float(np.std(deterministic[:, 0]))],
                "deterministic_throttle_mean_std": [float(np.mean(deterministic[:, 1])), float(np.std(deterministic[:, 1]))],
                "first_layer_preactivation_nonpositive_fraction": float((preactivation <= 0).float().mean().cpu()),
                "first_layer_post_relu_zero_fraction": float((activation == 0).float().mean().cpu()),
                "first_layer_post_relu_mean_feature_sd": float(activation.std(dim=0, unbiased=False).mean().cpu()),
                "alpha": alpha,
                "offline_actor_loss_proxy": float(actor_loss_proxy.detach().cpu()),
                "offline_actor_grad_norm_proxy": actor_grad_norm_proxy,
                "offline_critic_q1_mean": float(det_q1.mean().cpu()),
                "offline_critic_q2_mean": float(det_q2.mean().cpu()),
                "offline_abs_twin_q_gap_mean": float((det_q1 - det_q2).abs().mean().cpu()),
                "action_value_grid": action_grid,
                "critic_values_are_ground_truth": False,
                "replay_support_for_grid": "UNKNOWN; historical replay buffer was not saved",
                "td_error_history": "NOT_RECORDED; cannot reconstruct without replay and optimizer batches",
                "gradient_note": "offline actor-objective snapshot on frozen diagnostic observations; not the gradient used during historical training",
            })

    payload = {
        "protocol": "stage5_p2_historical_mechanism_audit_v1",
        "source_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "historical_training_source_commit": protocol["source_commit"],
        "historical_task_context_semantics": "legacy_v1 (S5.5 predates versioned corrected producer; current loader explicitly requires legacy)",
        "producer_id_for_checkpoint_load": LEGACY_TASK_PRODUCER,
        "observation_transform_id": TRANSFORM,
        "observation_dim": 409,
        "action_contract": "continuous_box2_v1",
        "state_file": "runs/stage5_meter_reward/s5_5_meter_reward_seed101_20261120/evaluation/policy_response_eval_states.jsonl.gz",
        "state_file_sha256": states_sha,
        "protocol_sha256": protocol["protocol_sha256"],
        "analysis_summary_sha256": sha256(RESULTS / "analysis_summary.json"),
        "state_rows": len(rows),
        "checkpoints": checkpoints,
        "confirmed": [
            "Every loaded checkpoint passed its repository SHA/encoder/legacy-producer verification.",
            "The historical diagnostics show near-constant deterministic actions and mostly inactive first actor layers on the sampled states; extent is limited to these states.",
            "S5.5 did not persist replay buffers or per-training-step terminal outcome records, so exact replay coverage, training TD error history and actual optimizer-batch gradient history cannot be recovered.",
            "The prior throttle-hold-5 experiment changed the executed and replay-stored throttle together, reduced sign changes and increased logged movement, but left deterministic evaluation stationary and produced many training out-of-road endings.",
        ],
        "hypotheses": [
            "Correlated exploration may create longer moving transitions than iid actor sampling; whether those transitions are safe/useful and shape the actor is untested.",
            "First-layer inactivity and low deterministic sensitivity may contribute to weak learned control, but do not alone identify cause.",
        ],
        "unknown": [
            "Exact historical replay action/state support and historical TD-error/gradient distributions.",
            "Whether critic action ranking on these local grids is reliable; candidate actions may be outside replay support.",
        ],
        "training_performed": False,
        "holdout_accessed": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "historical_mechanism_audit.json"
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    out.write_text(serialized, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    out.with_suffix(".sha256").write_text(f"{digest}  {out.name}\n", encoding="ascii", newline="\n")
    print(json.dumps({"checkpoints": len(checkpoints), "states": len(rows), "sha256": digest, "path": str(out)}, indent=2))


if __name__ == "__main__":
    main()
