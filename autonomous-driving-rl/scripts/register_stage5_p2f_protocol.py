"""Freeze the P2-F reset-curriculum A/B protocol before scientific training."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_stage5_p2_experiment as p2
from scripts.run_stage5_p2d_actor_activation import read_protocol as read_p2d_protocol
from src.training.stage5.start_curriculum import (
    CONTROL_CONFIG, CONTROL_FINGERPRINT, CONTROL_TASK_ID, CURRICULUM_CONFIG,
    CURRICULUM_FINGERPRINT, CURRICULUM_TASK_ID,
)

OUT = ROOT / "results/stage5/p2f_training_curriculum_20261121/protocol.json"
P2D_JSON = ROOT / "results/stage5/p2d_actor_activation_seed101_20261121/protocol_v3.json"
P2C_ANCHORS = ROOT / "results/stage5/p2c_representation_critic_audit_20261121/fixed_train_anchors.npy"
SOURCES = (
    "src/platform/agent_input.py", "src/platform/task_context_producer.py", "src/platform/reference_station.py",
    "src/platform/reward.py", "src/platform/episode.py", "src/launcher/cases.py", "src/launcher/executor.py",
    "src/training/stage5/env_adapter.py", "src/training/stage5/start_curriculum.py",
    "src/training/stage5/train_sampler.py", "src/training/stage5/observation_encoder.py",
    "src/training/stage5/pilot_observation_transform.py", "src/training/stage5/actor_activation_ablation.py",
    "src/training/stage5/temporal_exploration.py", "src/training/stage5/checkpoint.py",
    "scripts/run_stage5_p2_experiment.py", "scripts/run_stage5_p2d_actor_activation.py",
    "scripts/evaluate_stage5_p2.py", "scripts/evaluate_stage5_p2d_actor_activation.py",
    "scripts/run_stage5_p2f_curriculum.py", "scripts/evaluate_stage5_p2f_curriculum.py",
    "scripts/check_stage5_p2f_feasibility.py", "scripts/smoke_stage5_p2f_curriculum.py",
    "tests/test_stage5_start_curriculum.py",
    "results/stage5/p2d_actor_activation_seed101_20261121/protocol_v3.json",
    "results/audits/evaluation_protocol/geometry_split_manifest.csv",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"P2-F protocol already exists; refusing to rewrite: {OUT}")
    parent = read_p2d_protocol()
    feasibility_path = ROOT / "results/stage5/p2f_curriculum_preflight_20261121/reset_integrity.json"
    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
    if feasibility.get("status") != "PASS" or len(feasibility.get("rows", [])) != 16 or not all(x["pass"] for x in feasibility["rows"]):
        raise RuntimeError("P2-F preflight must PASS all 16 TRAIN reset cases before preregistration")
    if sha256(P2C_ANCHORS) != parent["anchors"]["sha256"]:
        raise RuntimeError("P2-C fixed TRAIN anchor identity mismatch")
    source_hashes = {name: sha256(ROOT / name) for name in SOURCES}
    protocol = {
        "protocol": "stage5_p2f_training_only_remaining_route_distance_curriculum_v1",
        "protocol_revision": 2,
        "supersedes_protocol_sha256": "8f0257bc144ee02009753483072eeae8b617f9e7c03faedea12d23f0658042ae",
        "status": "PRE_REGISTERED",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_parent_commit": p2.git_head(),
        "parent_protocol": {
            "path": "results/stage5/p2d_actor_activation_seed101_20261121/protocol_v3.json",
            "sha256": parent["protocol_sha256"],
            "experiment_result": "P2-D actor-only LeakyReLU; reused architecture/transform/action setup, not its outcomes as training data",
        },
        "preregistration_note": "Only reset start-state distribution differs between A/B. The MetaDrive route and destination remain the original case destination; no route shortening, terminal relabeling or horizon change.",
        "question": "Does a fixed remaining-route-distance curriculum create useful clean trajectory support and transfer to the original full-start TRAIN task?",
        "training_tasks": {
            "A": {"id": CONTROL_TASK_ID, "fingerprint": CONTROL_FINGERPRINT, "configuration": CONTROL_CONFIG},
            "B": {"id": CURRICULUM_TASK_ID, "fingerprint": CURRICULUM_FINGERPRINT, "configuration": CURRICULUM_CONFIG},
        },
        "geometry": {
            "manifest": "results/audits/evaluation_protocol/geometry_split_manifest.csv",
            "manifest_sha256": sha256(ROOT / "results/audits/evaluation_protocol/geometry_split_manifest.csv"),
            "training_pool": parent["geometry"]["training_pool"],
            "diagnostic_pool": parent["geometry"]["diagnostic_pool"],
            "sampler": "ROUND_ROBIN sorted P2-D eight-geometry PILOT_TRAIN allowlist",
            "environment_seed_rule": "TrainCaseSampler.derive_environment_seed(root_seed=101, episode_index); same rule in both arms",
            "curriculum_environment_seed": "exact original episode seed is used for route-resolution reset and actual curriculum reset",
            "official_full_start_transfer": True,
            "all_data_scope": "TRAIN only; diagnostic cases never enter either replay buffer",
        },
        "feasibility": {
            "status": "PASS", "evidence_path": "results/stage5/p2f_curriculum_preflight_20261121/reset_integrity.json",
            "evidence_sha256": sha256(feasibility_path), "case_distance_combinations": 16,
            "requested_remaining_distances_m": [60.0, 120.0],
            "tolerance_position_and_station_m": 0.05, "tolerance_remaining_distance_m": 0.10,
            "pinned_metadrive": "0.4.3", "corrected_producer": parent["producer"],
        },
        "arms": {
            "A": {"start_distribution": "official full-case reset start on every episode"},
            "B": {"start_distribution": "episodes 0-2: 60m remaining; 3-5: 120m remaining; episode 6 onward: official full-route start"},
            "only_registered_difference": "Stage 5 TRAIN reset start-state distribution; no action, policy, exploration, reward or horizon differences",
        },
        "sac": {
            "algorithm": "Stable-Baselines3 SAC from fresh initialization",
            "actor_hidden_activation": "LeakyReLU(0.01), actor only",
            "critic_hidden_activation": "ReLU, both twin critics and target critics",
            "net_arch": [256, 256], "action_shape": [2], "observation_dim": 409,
            "learning_rate": parent["sac"]["learning_rate"], "buffer_size": parent["sac"]["buffer_size"],
            "batch_size": parent["sac"]["batch_size"], "gamma": parent["sac"]["gamma"],
            "tau": parent["sac"]["tau"], "learning_starts": parent["sac"]["learning_starts"],
            "train_frequency": parent["sac"]["train_freq"], "gradient_steps": parent["sac"]["gradient_steps"],
            "ent_coef": parent["sac"]["ent_coef"], "exploration": "standard SB3 SAC stochastic sampling; no OU, no action hold",
            "root_seed": 101, "initial_actor_critic_entropy_shared": True,
            "optimizer_and_replay_states": "fresh and independent per arm",
        },
        "exploration": {
            "mode": "standard_sb3_sac_stochastic",
            "ou_action_noise": False,
            "compatibility_parameters_for_shared_runner": {"rho": 0.85, "stationary_std": 0.15},
            "compatibility_parameters_effective": False,
        },
        "observation": parent["observation"],
        "anchors": parent["anchors"],
        "action": parent["action"],
        "reward": parent["reward"],
        "training": {
            "max_environment_steps_per_arm": 10_000,
            "checkpoint_steps": [0, 2_000, 5_000, 10_000],
            "continuous_run": True, "resume": "not supported; interrupted run marked INCOMPLETE, no concatenation",
            "no_teacher_or_stage3_replay": True,
        },
        "evaluation": {
            "probe_steps": 300, "probe_checkpoints": [0, 2_000, 5_000, 10_000],
            "probe_cases": parent["geometry"]["diagnostic_pool"],
            "full_original_task_checkpoints": [10_000],
            "full_original_task_cases": parent["geometry"]["diagnostic_pool"],
            "curriculum_competence_checkpoint": 10_000,
            "curriculum_competence_cases": parent["geometry"]["diagnostic_pool"],
            "curriculum_evaluation_stages": [
                {"stage_index": 0, "requested_remaining_route_distance_m": 60.0},
                {"stage_index": 3, "requested_remaining_route_distance_m": 120.0},
            ],
            "outcomes": ["CLEAN_SUCCESS", "SAFETY_FAILURE", "TIMEOUT", "PROBE_CENSORED"],
            "frozen_actor_controls_both_action_dimensions": True,
            "evaluation_environment_and_replay_isolation": True,
            "benchmark_eligible": False,
        },
        "accounting": {
            "route_progress": "MetaDrive route_completion; Stage5TrainingEnv seeds last_route_completion from reset value, so no reset jump is rewarded",
            "route_destination": "original destination selected by same geometry/environment seed; pinned into curriculum env",
            "horizon_steps": "unchanged ResolvedCaseV1 official full-case horizon in both arms",
            "spawn_velocity": "0 m/s; no teacher or action-derived velocity",
            "reward_contract": "RewardSpecV1 unchanged; no success/failure relabeling; full original destination required for arrive_dest",
            "route_completion_denominator": "MetaDrive recomputes route fraction over the current edge suffix after a mid-route spawn; logged explicitly. This is an inherent consequence of the reset treatment and a limit on attributing reward-scale effects separately.",
        },
        "metrics": ["clean_success", "safety_failure", "timeout", "probe_censored", "route_progress_delta",
                    "mean_speed_kmh", "peak_speed_kmh", "stationary_fraction", "moving_run_length",
                    "throttle_brake_distribution", "steering_variability", "actor_sensitivity", "alpha", "critic_td_diagnostic"],
        "constraints": {
            "platform_contracts_changed": False, "reward_spec_changed": False, "benchmark_eligible": False,
            "validation_test_accessed": False, "stage3_teacher_or_replay": False,
            "training_steps_limit_per_arm": 10_000, "automatic_extension": False,
            "p2e_deterministic_episode_mixture": False,
        },
        "source_files_sha256": source_hashes,
    }
    protocol["protocol_sha256"] = p2.canonical_hash(protocol)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(protocol, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": protocol["protocol"], "path": OUT.relative_to(ROOT).as_posix(),
                      "sha256": protocol["protocol_sha256"], "source_files": len(source_hashes),
                      "feasibility_sha256": protocol["feasibility"]["evidence_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
