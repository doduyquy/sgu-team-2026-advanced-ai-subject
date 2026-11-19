"""One bounded, read-only analysis of existing S5.3M checkpoints and anchors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import torch
from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.training.stage5.checkpoint import compute_file_sha256, verify_checkpoint_integrity


def _actor_stats(model: SAC, observations: np.ndarray) -> dict[str, Any]:
    actions, _ = model.predict(observations, deterministic=True)
    tensor, _ = model.policy.obs_to_tensor(observations)
    actor = model.policy.actor
    with torch.no_grad():
        features = actor.extract_features(tensor, actor.features_extractor)
        preactivation = actor.latent_pi[0](features)
        first_activation = actor.latent_pi[1](preactivation)
    return {
        "deterministic_action_mean": actions.mean(axis=0).astype(float).tolist(),
        "deterministic_action_std_across_anchors": actions.std(axis=0).astype(float).tolist(),
        "first_layer_preactivation_mean": float(preactivation.mean().cpu()),
        "first_layer_preactivation_std": float(preactivation.std(unbiased=False).cpu()),
        "first_layer_nonpositive_fraction": float((preactivation <= 0).float().mean().cpu()),
        "first_layer_activation_std": float(first_activation.std(unbiased=False).cpu()),
        "entropy_coefficient": float(model.log_ent_coef.detach().exp().cpu())
            if getattr(model, "log_ent_coef", None) is not None else float(model.ent_coef),
    }


def analyze() -> dict[str, Any]:
    source = ROOT / "results/stage5/s5_3m_tau_ab_seed101_20261119_retry1"
    anchor_path = source / "anchor_observations.json"
    anchors_doc = json.loads(anchor_path.read_text(encoding="utf-8"))
    names = sorted(anchors_doc["anchors"])
    observations = np.stack([anchors_doc["anchors"][name]["observation"] for name in names]).astype(np.float32)
    group_ranges = {
        "core_259d": [0, 259], "traffic_actor_56d": [259, 315],
        "traffic_masks_metrics_10d": [315, 325], "task_waypoints_60d": [325, 385],
        "task_masks_scalars_24d": [385, 409],
    }
    feature_stats: dict[str, Any] = {}
    for name, (start, stop) in group_ranges.items():
        group = observations[:, start:stop]
        feature_stats[name] = {
            "raw_min": float(group.min()), "raw_max": float(group.max()),
            "per_feature_std_mean_across_eight_reset_anchors": float(group.std(axis=0).mean()),
            "per_feature_abs_max_mean": float(np.abs(group).max(axis=0).mean()),
        }

    checkpoints: dict[str, Any] = {}
    checkpoint_root = ROOT / "runs/stage5_tau_ab/s5_3m_tau_ab_seed101_20261119_retry1"
    for arm in ("A", "B"):
        checkpoints[arm] = {}
        for step in (0, 5000, 10000):
            checkpoint_dir = checkpoint_root / f"arm_{arm}/checkpoints/step_{step:05d}"
            metadata = verify_checkpoint_integrity(checkpoint_dir)
            model = SAC.load(str(checkpoint_dir / "model.zip"), device="cpu")
            checkpoints[arm][str(step)] = {
                "model_sha256": metadata["model_sha256"],
                "metadata_sha256": compute_file_sha256(checkpoint_dir / "metadata.json"),
                "training_step": metadata["total_timesteps"],
                "actor": _actor_stats(model, observations),
            }
            del model

    analysis_path = source / "analysis.json"
    historical = json.loads(analysis_path.read_text(encoding="utf-8"))
    endpoint_behavior = {
        arm: {
            str(step): {
                key: historical["arms"][arm][str(step)]["overall"].get(key)
                for key in ("mean_throttle", "mean_brake", "mean_speed_km_h",
                            "stationary_fraction", "mean_route_progress_delta",
                            "safety_failure_count", "probe_censored", "clean_success_count")
            }
            for step in (0, 5000, 10000)
        }
        for arm in ("A", "B")
    }
    training_actions = {
        arm: json.loads((checkpoint_root / f"arm_{arm}/checkpoints/training_action_summary.json").read_text(encoding="utf-8"))
        for arm in ("A", "B")
    }
    l1_path = ROOT / "results/stage5/s5_3l_softq_calibration_seed101_20261115/L1_ANALYSIS.md"
    l1_excerpt = None
    if l1_path.is_file():
        l1_excerpt = "Mean min(Q1,Q2)=15.232; mean full soft-return estimate=9.097; mean Q-return gap=+6.135 over 96 TRAIN-only stochastic rollouts. This is local absolute Q overestimation evidence, not proof it caused stationary behavior."
    return {
        "protocol": "stage5_s54_bounded_mechanism_analysis_v1",
        "read_only": True,
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         check=True, capture_output=True, text=True).stdout.strip(),
        "analysis_script_sha256": compute_file_sha256(Path(__file__).resolve()),
        "historical_source_run": "s5_3m_tau_ab_seed101_20261119_retry1",
        "anchor_file_sha256": compute_file_sha256(anchor_path),
        "anchor_count": len(names), "anchor_names": names,
        "feature_group_statistics_raw_409d": feature_stats,
        "checkpoints": checkpoints,
        "historical_diagnostic_behavior": endpoint_behavior,
        "historical_training_action_summary": training_actions,
        "reward_and_objective": {
            "reward_spec": "delta_route - 0.25 / horizon + terminal_reward",
            "reward_contract_unchanged": True,
            "gamma": 0.99,
            "entropy_coefficient_at_10k": {
                arm: checkpoints[arm]["10000"]["actor"]["entropy_coefficient"] for arm in ("A", "B")
            },
            "historical_q_diagnostic": l1_excerpt,
        },
        "direct_measurements": [
            "Deterministic actions across the same eight real, distinct fixed reset observations.",
            "First-layer preactivation/activation statistics for the historical 409D ReLU actor.",
            "Raw feature-group ranges and across-anchor dispersion.",
            "Previously recorded deterministic probes and training action summaries; no new simulation was run.",
        ],
        "limitations": [
            "The eight anchors are reset observations, not a representative replay-buffer or moving-state distribution.",
            "The sampled reset observations have zero traffic actor values, so traffic scaling cannot be judged from these anchors.",
            "Historical replay buffers are unavailable; this analysis cannot reconstruct exact training state coverage.",
            "The S5.3L Q-return gap does not prove a stationary-policy cause or wrong relative action ranking.",
        ],
        "mechanism_decision": {
            "selected_treatment": "pilot_groupwise_physical_v1",
            "decision": "EXPLORATORY_HYPOTHESIS",
            "rationale": "Across eight fixed anchors, actor action dispersion collapsed from the shared initialization to near-constant outputs by 10k; first-layer nonpositive fraction rose markedly, while waypoint/task physical fields showed much larger raw scale dispersion than the normalized core. This supports testing fixed, public-field unit scaling as one isolated intervention; it does not establish normalization as root cause.",
            "alternatives_not_selected": ["reward scaling", "entropy coefficient modification", "gamma change", "architecture/activation change"],
            "freeze_before_training": True,
        },
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["mechanism_decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
