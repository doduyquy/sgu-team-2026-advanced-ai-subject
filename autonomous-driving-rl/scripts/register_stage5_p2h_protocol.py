"""Freeze P2-H terminal-aware replay protocol before its scientific run."""
from __future__ import annotations

import json
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_stage5_p2_experiment as p2
from src.training.stage5.actor_activation_ablation import ACTOR_LEAKY_RELU_001_V1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amend-before-training", action="store_true")
    args = parser.parse_args()
    out = ROOT / "results/stage5/p2h_terminal_replay_seed101_20261121"
    if out.exists():
        existing_path = out / "protocol.json"
        if not args.amend_before_training or not existing_path.is_file():
            raise FileExistsError("P2-H registration exists; only explicit pre-training amendment is allowed")
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        if existing.get("status") != "PRE_REGISTERED":
            raise RuntimeError("Cannot amend P2-H protocol after training status changes")
    else:
        out.mkdir(parents=True)
    base_path = ROOT / "results/stage5/p2d_actor_activation_seed101_20261121/protocol_v3.json"
    protocol = json.loads(base_path.read_text(encoding="utf-8"))
    protocol.pop("protocol_sha256", None)
    protocol.update({
        "protocol": "stage5_p2h_terminal_aware_critic_replay_v1",
        "status": "PRE_REGISTERED",
        "parent_commit": p2.git_head(),
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Does terminal-aware prioritized critic replay improve state-dependent SAC driving under otherwise fixed P2-D conditions?",
        "arms": {
            "A": {"id": "uniform_replay_control_v1", "critic_replay": "uniform", "actor_and_alpha_replay": "uniform"},
            "B": {"id": "terminal_aware_prioritized_critic_replay_v1", "critic_replay": "uniform_mixture_plus_capped_terminal_td_priority", "actor_and_alpha_replay": "uniform"},
            "only_registered_difference": "critic replay minibatch distribution; treatment uses importance-weighted critic loss",
        },
        "sac": {**protocol["sac"], "actor_activation_negative_slope": {"A": 0.01, "B": 0.01},
                "policy_classes": {"A": "ActorOnlyLeakyReLUPolicy", "B": "ActorOnlyLeakyReLUPolicy"},
                "actor_activation_id": ACTOR_LEAKY_RELU_001_V1,
                "critic_activation": "ReLU", "optimizer_and_replay": "independent per arm"},
        "terminal_replay": {
            "sampler_id": "terminal_aware_prioritized_critic_replay_v1",
            "eligible_terminal_kinds": {"safety": 1, "clean_success": 2},
            "excluded_kind": {"timeout_truncation": 3},
            "terminal_mask": "replay.dones AND NOT replay.timeouts; terminal info must classify as safety or clean success",
            "uniform_mixture_probability": 0.80,
            "terminal_priority_mixture_probability": 0.20,
            "priority": "(min(max(abs(td1),abs(td2)),10.0)+0.001)^0.60",
            "td_error_cap": 10.0,
            "priority_epsilon": 0.001,
            "priority_alpha": 0.60,
            "importance_sampling": "w_i=(N*P(i))^-beta; normalize by maximum possible weight over the full current buffer; apply only to per-sample critic MSE",
            "importance_beta": {"start": 0.40, "end": 1.0, "optimizer_updates": 9000, "schedule": "linear"},
            "priority_update": "after each critic minibatch using max absolute twin TD error; newly inserted slots initialize to min(max(current priorities),10), floored at 1",
            "sampling": "with replacement; independent uniform actor/entropy batch in B; control uses standard SB3 uniform batch for all losses",
            "empty_terminal_pool": "uniform fallback until first true safety or clean terminal; log fallback and terminal exposure",
            "timeout_bootstrap": "preserve SB3 2.9.2 handle_timeout_termination and original SAC soft target; timeout dones are masked by ReplayBuffer",
            "rng_seed": 101,
        },
        "preflight": {
            "required_cases": ["0 terminal", "1 terminal", "2 terminals", "4 terminals", "safety vs clean", "timeout excluded", "priority cap", "IS weights", "effective sample size", "checkpoint save/load", "action/reward parity", "timeout bootstrap"],
            "synthetic_sampler_unit_test_only": True,
            "technical_smoke_not_scientific": True,
            "fail_policy": "stop before scientific A/B if sampler tests, action integrity, checkpoint load, timeout semantics, or measurable treatment distribution fail",
        },
        "constraints": {**protocol["constraints"], "training_started": False, "validation_test_accessed": False, "benchmark_eligible": False},
        "training": {**protocol["training"], "checkpoint_steps": [0, 2000, 5000, 10000], "max_environment_steps_per_arm": 10000,
                     "same_initial_policy_tensors": True, "replay_buffer_saved_at_10000": True},
        "evaluation": {**protocol["evaluation"], "full_episode_checkpoints": [10000],
                       "diagnostic_cases": "same six locked Easy PILOT_DIAGNOSTIC cases and seeds as P2-D",
                       "benchmark_eligible": False},
        "interpretation": {
            "driving_improvement_requires": "state-dependent actor behavior and safer useful driving, not Q/TD metrics or speed alone",
            "clean_success": "terminal success classified by Platform V1 outcome classifier",
            "censored": "not timeout, clean success, or safety success",
            "single_seed_limit": True,
        },
    })
    sources = [
        "scripts/run_stage5_p2h_terminal_replay.py",
        "scripts/register_stage5_p2h_protocol.py",
        "scripts/evaluate_stage5_p2h_terminal_replay.py",
        "scripts/smoke_stage5_p2h_terminal_replay.py",
        "src/training/stage5/terminal_aware_replay.py",
        "tests/test_stage5_terminal_aware_replay.py",
        "tests/test_stage5_p2h_replay_training.py",
        "scripts/run_stage5_p2_experiment.py",
        "src/training/stage5/temporal_exploration.py",
        "src/training/stage5/actor_activation_ablation.py",
        "src/training/stage5/checkpoint.py",
        "src/training/stage5/env_adapter.py",
        "src/training/stage5/observation_encoder.py",
        "src/training/stage5/pilot_observation_transform.py",
        "src/training/stage5/train_sampler.py",
        "src/platform/reward.py",
        "src/platform/task_context_producer.py",
        "src/platform/action_adapter.py",
        "src/platform/episode.py",
        "src/platform/reference_station.py",
    ]
    protocol["source_files_sha256"] = {path: p2.sha256(ROOT / path) for path in sources}
    protocol["status"] = "PRE_REGISTERED"
    protocol["protocol_sha256"] = p2.canonical_hash(protocol)
    path = out / "protocol.json"
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (out / "protocol.sha256").write_text(protocol["protocol_sha256"] + "  protocol.json\n", encoding="ascii")
    print(f"P2-H protocol sha256={protocol['protocol_sha256']}")


if __name__ == "__main__":
    main()
