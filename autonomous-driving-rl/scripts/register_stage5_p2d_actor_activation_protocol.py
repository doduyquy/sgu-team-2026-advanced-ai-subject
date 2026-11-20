"""Freeze the P2-D actor-only activation protocol before training."""
from __future__ import annotations

import hashlib
import json
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/stage5/p2_correlated_exploration_seed101_20261120/protocol.json"
OUT = ROOT / "results/stage5/p2d_actor_activation_seed101_20261121/protocol_v3.json"
SOURCES = (
    "scripts/run_stage5_p2d_actor_activation.py",
    "scripts/evaluate_stage5_p2d_actor_activation.py",
    "scripts/smoke_stage5_p2d_actor_activation.py",
    "src/training/stage5/actor_activation_ablation.py",
    "src/training/stage5/checkpoint.py",
    "src/training/stage5/temporal_exploration.py",
    "scripts/run_stage5_p2_experiment.py",
    "scripts/evaluate_stage5_p2.py",
    "tests/test_stage5_actor_activation_ablation.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def main() -> None:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    parent_digest = base.pop("protocol_sha256", None)
    if parent_digest != "6a54151d0a3e52bcd237811f0b9a7d3912adadb512615e92eab71281f7564652":
        raise ValueError("Unexpected parent P2 protocol identity")
    anchors = ROOT / "results/stage5/p2c_representation_critic_audit_20261121/fixed_train_anchors.npy"
    if sha256(anchors) != "6ded06b27deec488dac1952b8a4a7d22755b8bbbca0d774854d53fe58ac6f13a":
        raise ValueError("P2-C fixed anchor file integrity check failed")
    base["protocol"] = "stage5_p2d_actor_only_activation_v1"
    base["registered_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    base["anchors"] = {"path": "results/stage5/p2c_representation_critic_audit_20261121/fixed_train_anchors.npy",
                       "sha256": sha256(anchors), "shape": [201, 409], "source": "P2-C real TRAIN anchors"}
    base["parent_protocol"] = "stage5_p2_ou_ar1_exploration_v1; P2 standard-SAC arm is the control reference"
    base["arms"] = {
        "A": {"id": "actor_relu_critic_relu_v1", "activation_id": "actor_relu_v1", "actor_policy_class": "stable_baselines3.sac.policies.SACPolicy",
              "actor_activation": "ReLU", "critic_activation": "ReLU"},
        "B": {"id": "actor_leaky_relu_001_critic_relu_v1", "activation_id": "actor_leaky_relu_001_v1", "actor_policy_class": "ActorOnlyLeakyReLUPolicy",
              "actor_activation": "LeakyReLU(0.01)", "critic_activation": "ReLU"},
        "only_registered_difference": "actor hidden activation function; same compatible initialized tensors",
    }
    base["sac"]["activation"] = "actor-only ablation; critic ReLU both arms"
    base["sac"]["policy_classes"] = {
        "A": "stable_baselines3.sac.policies.SACPolicy",
        "B": "src.training.stage5.actor_activation_ablation.ActorOnlyLeakyReLUPolicy",
    }
    base["sac"]["actor_activation_negative_slope"] = {"A": None, "B": 0.01}
    base["training"]["checkpoint_steps"] = [0, 2000, 5000, 10000]
    base["training"]["max_environment_steps_per_arm"] = 10000
    base["training"]["root_seed"] = 101
    base["training"]["initialization_note"] = (
        "Fresh shared state initialized under ReLU control then loaded into both arms; tensor shapes and values match. "
        "Because nonlinearities differ, initial policy outputs are not asserted equal."
    )
    base["telemetry"].update({
        "fixed_p2c_201_train_anchors": True,
        "real_replay_state_activation_and_gradient_diagnostics": True,
        "activation_derivative_aware_leaky_metrics": True,
        "replay_buffer_saved_at_10000": True,
        "checkpoint_activation_identity_and_roundtrip": True,
    })
    base["s5_3c_comparison"] = {
        "result": "24/24 full diagnostic cases TIMEOUT, stationary_fraction=1.0, mean throttle=0; no clean success",
        "differences": [
            "S5.3C used waypoint_xy_div50m_v1, not pilot_groupwise_physical_v1",
            "S5.3C passed LeakyReLU as SB3 policy-wide activation_fn, changing actor and critic",
            "S5.3C had 10k steps and 24 diagnostic maps; P2-D has matched actor-only treatment and six locked cases",
        ],
        "interpretation": "historical negative evidence, not a direct actor-only comparison",
    }
    base["source_files_sha256"] = {name: sha256(ROOT / name) for name in SOURCES}
    base["status"] = "PRE_REGISTERED"
    base["constraints"]["training_started"] = False
    base["constraints"]["validation_test_accessed"] = False
    base["protocol_sha256"] = canonical_hash(base)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite registered protocol: {OUT}")
    OUT.write_text(json.dumps(base, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"path": OUT.relative_to(ROOT).as_posix(), "sha256": base["protocol_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
