"""Create the immutable P2 protocol from the already-locked S5.5 TRAIN pools."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER, task_producer_identity


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def main() -> None:
    history_path = ROOT / "results/stage5/s5_5_meter_reward_seed101_20261120/protocol.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    train, diagnostic = history["training_geometry_pool"], history["diagnostic_pool"]
    if len(train) != 8 or len(diagnostic) != 6:
        raise ValueError("S5.5 locked pool does not match required P2 schedule")
    source_files = [
        "scripts/audit_stage5_p2_mechanism.py", "src/training/stage5/temporal_exploration.py",
        "scripts/run_stage5_p2_experiment.py", "scripts/evaluate_stage5_p2.py",
        "scripts/smoke_stage5_p2.py", "tests/test_stage5_p2_temporal_exploration.py",
    ]
    p = {
        "protocol": "stage5_p2_ou_ar1_exploration_v1", "status": "PRE_REGISTERED",
        "registered_at_utc": "2026-11-20T16:58:27Z",
        "parent_commit": "8d38c4a8135ff41258fd08eb9a230e59efff5edd",
        "registration_note": "Protocol and implementation hashes are committed before scientific training. Exact training source commit is recorded in each run manifest.",
        "implementation_amendment_after_interrupted_attempt": {
            "amended_at": "2026-11-21T00:18:35+07:00",
            "reason": "Flush live action telemetry every 100 recorded transitions and add geometry sequence/hash to episode ledger. This is instrumentation/provenance only; no scientific variable, arm, seed, budget, or evaluation schedule changed.",
            "interrupted_attempt": "runs/stage5_p2/s5_p2_correlated_exploration_seed101_20261120; control arm only, 6,150 recorded transitions, checkpoints 0/2k/5k, excluded from scientific comparison; artifacts preserved.",
        },
        "source_files_sha256": {name: sha256(ROOT / name) for name in source_files},
        "historical_mechanism_audit": {
            "path": "results/stage5/p2_mechanism_audit/historical_mechanism_audit.json",
            "sha256": sha256(ROOT / "results/stage5/p2_mechanism_audit/historical_mechanism_audit.json"),
        },
        "geometry": {
            "manifest": "results/audits/evaluation_protocol/geometry_split_manifest.csv",
            "manifest_sha256": "ec1099eb93252f445d7afa95fd19270e72d5a8f37282701d6935e283a0997134",
            "training_pool_source_protocol": "results/stage5/s5_5_meter_reward_seed101_20261120/protocol.json",
            "training_pool_sha256": history["training_pool_sha256"], "training_pool": train,
            "diagnostic_pool_source_protocol": "results/stage5/s5_5_meter_reward_seed101_20261120/protocol.json",
            "diagnostic_pool_sha256": history["diagnostic_pool_sha256"], "diagnostic_pool": diagnostic,
        },
        "training": {
            "root_seed": 101, "max_environment_steps_per_arm": 10000,
            "checkpoint_steps": [0, 2000, 5000, 10000], "same_pool_and_sampler_rule": True,
            "sampler_strategy": "ROUND_ROBIN",
            "environment_seed_derivation": history["sampler"]["environment_seed_derivation"],
            "seed_schedule_note": "Same seed derivation and ordered pool in both arms; episode counts may diverge as policies diverge, so actual per-episode exposure is recorded rather than forced to match.",
        },
        "arms": {
            "A": {"id": "control_standard_sac_v1", "sampler": "SB3 standard SAC action collection; random warm-up then stochastic actor sampling"},
            "B": {"id": "treatment_ou_ar1_v1", "sampler": "OU AR(1) additive action perturbation after warm-up; perturbed normalized action is used by both env and replay"},
            "only_registered_difference": "exploration sampler; reward, observation, network and SAC hyperparameters unchanged",
        },
        "exploration": {
            "mode": "ou_ar1_v1", "process": "stationary zero-mean AR(1): n[t]=rho*n[t-1]+epsilon[t]",
            "rho": 0.85, "stationary_std": 0.15, "innovation_std": 0.07905694150420949,
            "action_dimensions": ["steering", "throttle_brake"], "applied_after_learning_starts_only": True,
            "clip": "componentwise to [-1,1]", "reset": "zero OU state after every terminated or truncated episode",
            "random_warmup_steps": 1000, "seed": 101, "not_action_hold": True,
        },
        "sac": {
            "algorithm": "SAC", "policy": "MlpPolicy", "net_arch": [256, 256], "activation": "ReLU",
            "learning_rate": 0.0003, "gamma": 0.99, "tau": 0.005, "buffer_size": 100000,
            "batch_size": 256, "learning_starts": 1000, "ent_coef": "auto", "train_freq": 1,
            "gradient_steps": 1, "device": "cpu", "shared_fresh_initial_policy_and_entropy_state": True,
            "optimizer_and_replay": "independent per arm",
        },
        "observation": {
            "canonical_encoder": "stage5_flat409_v1", "dimension": 409,
            "post_encoder_transform": history["observations"]["post_encoder_transform"],
            "transform_fingerprint": history["observations"]["transform_sha256"], "same_both_arms": True,
            "task_context_producer_id": "reference_route_station_proposal_v1",
        },
        "producer": task_producer_identity(REFERENCE_TASK_PRODUCER),
        "action": {"contract": "continuous_box2_v1", "shape": [2], "range": [-1, 1],
                   "order": ["steering", "throttle_brake"], "decision_frequency_hz": 10},
        "reward": {"version": "RewardSpecV1", "only_stage5_reward": True, "changed": False},
        "evaluation": {
            "dataset": "TRAIN/PILOT_DIAGNOSTIC only", "fixed_cases": diagnostic, "fast_probe_steps": 300,
            "fast_probe_checkpoints": [0, 2000, 5000, 10000], "full_episode_checkpoints": [10000],
            "full_episode_cases": diagnostic,
            "outcome_labels": ["CLEAN_SUCCESS", "SAFETY_FAILURE", "TIMEOUT", "PROBE_CENSORED"],
            "censored_is_not_timeout_or_safe": True, "frozen_deterministic_policy": True,
            "separate_environment_no_training_replay_or_optimizer": True,
        },
        "telemetry": {
            "per_step_requested_executed_replay_actions": True, "per_episode_outcomes": True,
            "geometry_and_seed_exposure": True, "optimizer_actor_critic_entropy_gradients_losses": True,
            "alpha": True, "replay_q_td_snapshot_at_checkpoints": True,
            "activation_and_actor_sensitivity": True, "checkpoint_integrity_and_hashes": True,
            "runtime_and_memory": True,
        },
        "constraints": {
            "validation_test_accessed": False, "teacher_data_used": False, "stage3_used": False,
            "platform_contracts_changed": False, "benchmark_eligible": False, "training_started": False,
        },
    }
    p["protocol_sha256"] = canonical_hash(p)
    out = ROOT / "results/stage5/p2_correlated_exploration_seed101_20261120/protocol.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(p, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    out.write_text(serialized, encoding="utf-8", newline="\n")
    out.with_suffix(".sha256").write_text(
        f"{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}  {out.name}\n",
        encoding="ascii", newline="\n",
    )
    print(json.dumps({"path": str(out), "protocol_sha256": p["protocol_sha256"],
                      "train_geometry_count": len(train), "diagnostic_geometry_count": len(diagnostic)}, indent=2))


if __name__ == "__main__":
    main()
