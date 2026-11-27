"""Run the bounded, preregistered S5.4 fresh SAC A/B learnability experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

try:  # Import Torch before Panda3D on Windows.
    import torch
except ImportError:  # pragma: no cover
    torch = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from src.training.stage5.checkpoint import compute_file_sha256, verify_checkpoint_integrity
from src.training.stage5.pilot_callbacks import PilotCheckpointCallback
from src.training.stage5.pilot_observation_transform import GROUPWISE_PHYSICAL_V1
from src.training.stage5.pilot_protocol import canonical_sha256, diagnostic_schedule, load_pilot_split
from src.training.stage5.sac_trainer import SacBaselineTrainer
from src.training.stage5.train_sampler import SamplerStrategy
from src.platform import AgentInputV1, AgentPublicEpisodeContext, CoreObservationV1, InputProfileId, TaskContextV1, TrafficContextV1
from src.agents.stage5_sac_agent import Stage5SacAgent


RUN_ID = "s5_4_learnability_seed101_20261119"
MAX_STEPS = 100_000
CHECKPOINT_STEPS = (10_000, 30_000, 50_000, 100_000)
PRIMARY_SCHEDULE_SEED = 5404
CONFIRMATION_SCHEDULE_SEED = 5405


def _policy_fingerprint(model: SAC) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.policy.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    log_ent = getattr(model, "log_ent_coef", None)
    if log_ent is not None:
        digest.update(b"log_ent_coef")
        digest.update(log_ent.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _alpha(model: SAC) -> float:
    log_ent = getattr(model, "log_ent_coef", None)
    return float(log_ent.detach().exp().cpu().item()) if log_ent is not None else float(model.ent_coef)


def _select_geometry_pools(split: Dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    easy_train = [row for row in split["pilot_train"] if row["tier"] == "Easy"]
    quotas = {"SCCS": 3, "SCSS": 3, "SCS": 2}
    selected: list[dict[str, Any]] = []
    for family, count in quotas.items():
        candidates = sorted(
            (row for row in easy_train if row["sequence"] == family),
            key=lambda row: (int(row["geometry_generation_seed"]), row["geometry_id"]),
        )
        if len(candidates) < count:
            raise ValueError(f"PILOT_TRAIN Easy sequence {family} has fewer than {count} geometries")
        selected.extend(candidates[:count])
    diagnostic = sorted(
        (row for row in split["pilot_diagnostic"] if row["tier"] == "Easy"),
        key=lambda row: (row["sequence"], int(row["geometry_generation_seed"]), row["geometry_id"]),
    )
    train_ids = {row["geometry_id"] for row in selected}
    diagnostic_ids = {row["geometry_id"] for row in diagnostic}
    if len(selected) != 8 or len(diagnostic) != 6 or train_ids & diagnostic_ids:
        raise ValueError("S5.4 geometry pool cardinality or isolation invariant failed")
    if any(len(row.get("geometry_sha256", "")) != 64 for row in selected + diagnostic):
        raise ValueError("Selected S5.4 geometry lacks verified SHA-256")
    return selected, diagnostic


def _version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _new_trainer(allowed_ids: List[str], transform_id: Optional[str]) -> SacBaselineTrainer:
    return SacBaselineTrainer(
        training_root_seed=101, project_root=ROOT, device="cpu",
        allowed_geometry_ids=allowed_ids,
        pilot_observation_transform=transform_id,
        config_overrides={"dataset_sampler": {"default_strategy": "ROUND_ROBIN"}},
    )


def _write_protocol(results_dir: Path, run_root: Path, run_id: str) -> Path:
    if run_root.exists():
        raise FileExistsError(f"Refusing to overwrite S5.4 run: {run_root}")
    if results_dir.exists():
        existing = {path.name for path in results_dir.iterdir()}
        if existing != {"mechanism_analysis.json"}:
            raise FileExistsError(f"Refusing to overwrite S5.4 result directory contents: {sorted(existing)}")
    registration_path = ROOT / "configs/experiments/stage5_s54_learnability_v1.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    split_path = ROOT / "results/stage5/pilot_split_v1.json"
    split = load_pilot_split(split_path)
    train_pool, diagnostic_pool = _select_geometry_pools(split)
    schedule = diagnostic_schedule(split, schedule_seed=PRIMARY_SCHEDULE_SEED)
    diag_map = {row["geometry_id"]: row for row in schedule}
    diagnostic_schedule_rows = [{**row, "environment_seed": diag_map[row["geometry_id"]]["environment_seed"]}
                                for row in diagnostic_pool]
    config = json.loads((ROOT / "configs/experiments/stage5_rl_baseline_v1.json").read_text(encoding="utf-8"))
    payload: Dict[str, Any] = {
        "protocol": registration["protocol"], "run_id": run_id, "status": "PRE_REGISTERED",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                         capture_output=True, text=True).stdout.strip(),
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "numpy": np.__version__, "torch": str(torch.__version__),
                    "stable_baselines3": _version("stable-baselines3"),
                    "gymnasium": _version("gymnasium"), "metadrive": _version("metadrive"),
                    "device": "cpu"},
        "config_path": str(registration_path.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": compute_file_sha256(registration_path),
        "mechanism_analysis_path": str((results_dir / "mechanism_analysis.json").relative_to(ROOT)).replace("\\", "/"),
        "mechanism_analysis_sha256": compute_file_sha256(results_dir / "mechanism_analysis.json"),
        "pilot_split_path": str(split_path.relative_to(ROOT)).replace("\\", "/"),
        "pilot_split_sha256": split["split_sha256"],
        "source_geometry_manifest": "results/audits/evaluation_protocol/geometry_split_manifest.csv",
        "source_geometry_manifest_sha256": compute_file_sha256(ROOT / "results/audits/evaluation_protocol/geometry_split_manifest.csv"),
        "training_root_seed": 101,
        "environment_seed_derivation": registration["sampler"]["environment_seed_rule"],
        "training_sampler_strategy": "ROUND_ROBIN",
        "training_geometry_selection_rule": registration["sampler"]["selection_rule"],
        "training_geometry_pool": train_pool,
        "diagnostic_schedule_seed": PRIMARY_SCHEDULE_SEED,
        "diagnostic_schedule": diagnostic_schedule_rows,
        "diagnostic_cases": diagnostic_schedule_rows,
        "diagnostic_geometry_pool_sha256": canonical_sha256(diagnostic_schedule_rows),
        "training_geometry_pool_sha256": canonical_sha256(train_pool),
        "training_steps_max_per_arm": MAX_STEPS,
        "checkpoint_steps": [0, *CHECKPOINT_STEPS],
        "fast_probe_decision_cap": 300,
        "full_episode_checkpoints": registration["diagnostic"]["full_episode_checkpoints"],
        "conditional_full_episode_rule_at_50k": registration["diagnostic"]["conditional_50000_full_episode_rule"],
        "early_transfer_rule": registration["early_transfer_rule"],
        "common_sac_config": {
            **config["sac_hyperparameters"], "architecture": "SB3 MlpPolicy, two 256-unit ReLU layers",
            "reward_contract": config["reward_contract"], "observation_dim": 409,
        },
        "arms": registration["arms"],
        "only_arm_difference": registration["only_registered_arm_difference"],
        "observation_transform_spec": GROUPWISE_PHYSICAL_V1,
        "observation_transform_sha256": canonical_sha256(GROUPWISE_PHYSICAL_V1),
        "checkpoint_initialization": "One fresh baseline initialization shared by model parameter state; no historical trained checkpoint used.",
        "resume_policy": registration["resume_policy"],
        "holdouts_accessed": False,
        "benchmark_eligibility": {"agent_id": "stage5_sac", "status": "CANDIDATE", "benchmark_eligible": False},
        "reproducibility_note": "Same source revision, root seed, fresh initial parameters, geometry/seed rules, and training budget. Simulator/CPU execution is not claimed bitwise reproducible.",
    }
    payload["protocol_sha256"] = canonical_sha256(payload)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True)
    protocol_path = results_dir / "protocol.json"
    protocol_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    initial_trainer = _new_trainer([row["geometry_id"] for row in train_pool], None)
    try:
        initial_model = run_root / "common_initial_model.zip"
        initial_trainer.model.save(str(initial_model))
        init = {
            "model_sha256": compute_file_sha256(initial_model),
            "policy_entropy_fingerprint": _policy_fingerprint(initial_trainer.model),
            "entropy_coefficient": _alpha(initial_trainer.model),
            "log_entropy_coefficient": float(initial_trainer.model.log_ent_coef.detach().cpu().item()),
            "num_timesteps": int(initial_trainer.model.num_timesteps),
            "optimizer_updates": int(getattr(initial_trainer.model, "_n_updates", 0)),
            "seed": 101,
        }
        (run_root / "common_initial_state.json").write_text(
            json.dumps(init, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        initial_trainer.close()
    return protocol_path


def _child(command: List[str], log_base: Path) -> None:
    log_base.parent.mkdir(parents=True, exist_ok=True)
    with log_base.with_suffix(".stdout.log").open("w", encoding="utf-8") as stdout, \
         log_base.with_suffix(".stderr.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
    if process.returncode:
        raise RuntimeError(f"Child process exited with {process.returncode}; see {log_base}.*.log")


def _run_evaluation(checkpoint: Path, output: Path, geometry_ids: List[str], schedule_seed: int,
                    max_steps: Optional[int]) -> Dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic result: {output}")
    command = [sys.executable, str(ROOT / "scripts/evaluate_stage5_pilot.py"),
               "--checkpoint", str(checkpoint), "--split", str(ROOT / "results/stage5/pilot_split_v1.json"),
               "--output", str(output), "--schedule-seed", str(schedule_seed),
               *sum((["--geometry-id", value] for value in geometry_ids), [])]
    if max_steps is not None:
        command += ["--max-steps-per-episode", str(max_steps)]
    _child(command, output / "evaluation")
    step = int(json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))["total_timesteps"])
    summary_path = output / f"step_{step:05d}_summary.json"
    result = json.loads(summary_path.read_text(encoding="utf-8"))
    result["records_sha256"] = compute_file_sha256(output / result["records_file"])
    return result


def _record_success_and_safety(result: Dict[str, Any], output_dir: Path) -> tuple[set[str], int]:
    rows_path = output_dir / result["records_file"]
    successes: set[str] = set()
    safety_failures = 0
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("clean_success"):
            successes.add(str(row["geometry_id"]))
        safety_failures += int(bool(row.get("safety_failure")))
    return successes, safety_failures


def _early_go(primary: Dict[str, Any], primary_dir: Path,
              confirmation: Dict[str, Any], confirmation_dir: Path) -> tuple[bool, Dict[str, Any]]:
    primary_success, primary_safety = _record_success_and_safety(primary, primary_dir)
    confirmation_success, confirmation_safety = _record_success_and_safety(confirmation, confirmation_dir)
    primary_summary = primary["aggregate"]
    confirmation_summary = confirmation["aggregate"]
    action_std = primary.get("actor_state_sensitivity", {}).get("deterministic_action_std_across_fixed_observations", [0, 0])
    responsive = bool(action_std and max(action_std) >= 0.01)
    passed = (
        len(primary_success) >= 2 and primary_safety <= len(primary_success)
        and (primary_summary.get("stationary_fraction") is not None)
        and primary_summary["stationary_fraction"] <= 0.70 and responsive
        and len(confirmation_success) >= 1 and confirmation_safety <= len(confirmation_success)
        and confirmation_summary.get("clean_success_rate", 0.0) > 0.0
    )
    decision = {
        "decision": "GO_MULTI_MAP" if passed else "CONTINUE_EASY_LEARNABILITY",
        "primary_clean_success_geometry_ids": sorted(primary_success),
        "primary_safety_failures": primary_safety,
        "primary_mean_stationary_fraction": primary_summary.get("stationary_fraction"),
        "actor_action_std": action_std,
        "actor_observation_responsive": responsive,
        "confirmation_clean_success_geometry_ids": sorted(confirmation_success),
        "confirmation_safety_failures": confirmation_safety,
        "confirmation_clean_success_rate": confirmation_summary.get("clean_success_rate"),
        "criteria_met": passed,
    }
    return passed, decision


class S54EvaluationCallback(BaseCallback):
    def __init__(self, arm_root: Path, arm: str, geometry_ids: List[str], target_steps: int,
                 checkpoints: List[int], verbose: int = 0) -> None:
        super().__init__(verbose)
        self.arm_root = arm_root
        self.arm = arm
        self.geometry_ids = geometry_ids
        self.target_steps = target_steps
        self.checkpoints = set(checkpoints)
        self.evaluations: List[Dict[str, Any]] = []
        self.stop_boundary: Optional[int] = None

    def _on_step(self) -> bool:
        step = int(self.model.num_timesteps)
        if step not in self.checkpoints:
            return True
        checkpoint = self.arm_root / "checkpoints" / f"step_{step:05d}"
        verify_checkpoint_integrity(checkpoint)
        diagnostics_root = self.arm_root / "diagnostics" / f"step_{step:05d}"
        quick_dir = diagnostics_root / "probe_5404"
        quick = _run_evaluation(checkpoint, quick_dir, self.geometry_ids,
                                PRIMARY_SCHEDULE_SEED, max_steps=300)
        entry: Dict[str, Any] = {"training_step": step, "probe": quick,
                                 "probe_output": str(quick_dir)}
        print(f"[S5.4 {self.arm} {step}] probe speed={quick['aggregate'].get('probe_mean_speed_km_h')} "
              f"progress={quick['aggregate'].get('probe_mean_route_progress_delta')} "
              f"stationary={quick['aggregate'].get('probe_stationary_fraction')} "
              f"censored={quick['aggregate'].get('probe_censored_count')}", flush=True)
        run_full = step in (30_000, 100_000)
        if step == 50_000:
            aggregate = quick["aggregate"]
            run_full = (
                aggregate.get("probe_mean_route_progress_delta") is not None
                and aggregate["probe_mean_route_progress_delta"] >= 0.02
                and aggregate.get("probe_stationary_fraction") is not None
                and aggregate["probe_stationary_fraction"] <= 0.90
                and aggregate.get("probe_safety_failure_count", 0) == 0
            )
        full: Optional[Dict[str, Any]] = None
        if run_full:
            full_dir = diagnostics_root / "full_5404"
            full = _run_evaluation(checkpoint, full_dir, self.geometry_ids,
                                   PRIMARY_SCHEDULE_SEED, max_steps=None)
            entry["full_episode"] = full
            entry["full_episode_output"] = str(full_dir)
            print(f"[S5.4 {self.arm} {step}] full clean_success="
                  f"{full['aggregate'].get('clean_success_rate')} safety="
                  f"{full['aggregate'].get('safety_failure_rate')} route="
                  f"{full['aggregate'].get('final_route_completion')}", flush=True)

            # Only spend on a second fixed seed if the primary full evaluation
            # already meets the clean-driving, safety and actor-responsiveness gate.
            successes, failures = _record_success_and_safety(full, full_dir)
            astd = full.get("actor_state_sensitivity", {}).get("deterministic_action_std_across_fixed_observations", [0, 0])
            primary_gate = (
                len(successes) >= 2 and failures <= len(successes)
                and full["aggregate"].get("stationary_fraction") is not None
                and full["aggregate"]["stationary_fraction"] <= 0.70
                and bool(astd and max(astd) >= 0.01)
            )
            if primary_gate:
                confirmation_dir = diagnostics_root / "full_5405_confirmation"
                confirmation = _run_evaluation(checkpoint, confirmation_dir, self.geometry_ids,
                                               CONFIRMATION_SCHEDULE_SEED, max_steps=None)
                passed, decision = _early_go(full, full_dir, confirmation, confirmation_dir)
                entry["confirmation_full_episode"] = confirmation
                entry["early_transfer_decision"] = decision
                if passed:
                    self.stop_boundary = step
                    (self.arm_root / "early_transfer_decision.json").write_text(
                        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    self.evaluations.append(entry)
                    return False
        self.evaluations.append(entry)
        return True


def _save_initial_checkpoint(trainer: SacBaselineTrainer, checkpoint_dir: Path,
                             protocol: Dict[str, Any], arm: str, initial: Dict[str, Any]) -> Dict[str, Any]:
    trainer.config["stage5_s54"] = {
        "protocol_sha256": protocol["protocol_sha256"], "arm": arm,
        "only_arm_difference": protocol["only_arm_difference"],
        "common_initial_model_sha256": initial["model_sha256"],
        "common_initial_policy_entropy_fingerprint": initial["policy_entropy_fingerprint"],
    }
    trainer.save_checkpoint(checkpoint_dir, save_replay_buffer=False)
    metadata = verify_checkpoint_integrity(checkpoint_dir,
                                          expected_encoder_fingerprint=trainer.encoder.get_schema_fingerprint())
    if int(metadata["total_timesteps"]) != 0:
        raise ValueError("Initial checkpoint is not at zero steps")
    if metadata.get("observation_transform") != (
        GROUPWISE_PHYSICAL_V1 if arm == "B" else None
    ):
        raise ValueError(f"Arm {arm} checkpoint observation transform provenance mismatch")
    return metadata


def _run_arm(arm: str, run_root: Path, protocol_path: Path, target_steps: int = MAX_STEPS) -> Dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if protocol.get("protocol_sha256") != canonical_sha256(unsigned):
        raise ValueError("Pre-registered protocol SHA-256 mismatch")
    if not 0 < target_steps <= MAX_STEPS:
        raise ValueError(f"Invalid arm training budget {target_steps}")
    arm_root = run_root / f"arm_{arm}"
    if arm_root.exists():
        raise FileExistsError(f"Refusing to overwrite arm directory {arm_root}")
    arm_root.mkdir(parents=True)
    status_path = arm_root / "run_status.json"
    status: Dict[str, Any] = {
        "status": "RUNNING", "arm": arm, "target_steps": target_steps,
        "start_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol["protocol_sha256"], "source_commit": protocol["source_commit"],
    }
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    allowed = [row["geometry_id"] for row in protocol["training_geometry_pool"]]
    transform = GROUPWISE_PHYSICAL_V1["transform_id"] if arm == "B" else None
    trainer = _new_trainer(allowed, transform)
    started = time.perf_counter()
    try:
        init_path = run_root / "common_initial_model.zip"
        initial = json.loads((run_root / "common_initial_state.json").read_text(encoding="utf-8"))
        if compute_file_sha256(init_path) != initial["model_sha256"]:
            raise ValueError("Shared fresh initialization archive SHA-256 mismatch")
        loaded = SAC.load(str(init_path), env=trainer.model.get_env(), device="cpu")
        trainer.model = loaded
        if _policy_fingerprint(trainer.model) != initial["policy_entropy_fingerprint"]:
            raise ValueError(f"Arm {arm} did not load the shared initial actor/critic/entropy state")
        if _alpha(trainer.model) != float(initial["entropy_coefficient"]):
            raise ValueError(f"Arm {arm} initial entropy coefficient differs from shared state")
        if int(trainer.model.num_timesteps) != 0 or int(getattr(trainer.model, "_n_updates", 0)) != 0:
            raise ValueError(f"Arm {arm} did not start at zero environment/optimizer steps")

        # Reset all host RNG roots after deserialization; sampler derivation is
        # episode-index based and geometry pool is fixed by the protocol.
        random.seed(101)
        np.random.seed(101)
        if torch is not None:
            torch.manual_seed(101)
        trainer.model.action_space.seed(101)
        trainer.config["stage5_s54"] = {
            "protocol_sha256": protocol["protocol_sha256"], "arm": arm,
            "only_arm_difference": protocol["only_arm_difference"],
            "common_initial_model_sha256": initial["model_sha256"],
            "common_initial_policy_entropy_fingerprint": initial["policy_entropy_fingerprint"],
        }
        checkpoints_dir = arm_root / "checkpoints"
        checkpoints_dir.mkdir()
        cp0 = checkpoints_dir / "step_00000"
        initial_meta = _save_initial_checkpoint(trainer, cp0, protocol, arm, initial)
        diagnostic_ids = [row["geometry_id"] for row in protocol["diagnostic_schedule"]]
        probe0_dir = arm_root / "diagnostics/step_00000/probe_5404"
        probe0 = _run_evaluation(cp0, probe0_dir, diagnostic_ids, PRIMARY_SCHEDULE_SEED, 300)
        status["step_0_checkpoint_sha256"] = initial_meta["model_sha256"]
        status["step_0_probe_summary"] = probe0["aggregate"]
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        positive_checkpoints = [step for step in CHECKPOINT_STEPS if step <= target_steps]
        checkpoint_callback = PilotCheckpointCallback(
            trainer, checkpoints_dir, checkpoints=positive_checkpoints,
            requested_timesteps=target_steps, stop_file=arm_root / "STOP_REQUESTED",
            verbose=0, write_step_log=False,
        )
        eval_callback = S54EvaluationCallback(arm_root, arm, diagnostic_ids, target_steps,
                                             positive_checkpoints, verbose=0)
        trainer.model.learn(total_timesteps=target_steps,
                            callback=CallbackList([trainer.callback, checkpoint_callback, eval_callback]),
                            reset_num_timesteps=True)
        actual_steps = int(trainer.model.num_timesteps)
        expected_saved = list(checkpoint_callback.saved_steps)
        valid_early = eval_callback.stop_boundary == actual_steps
        if actual_steps != target_steps and not valid_early:
            raise RuntimeError(f"Arm stopped at {actual_steps}/{target_steps} without registered early-transfer decision")
        if not valid_early and expected_saved != positive_checkpoints:
            raise RuntimeError(f"Checkpoint schedule incomplete: expected {positive_checkpoints}, saved {expected_saved}")
        summary = trainer.callback.get_summary_metrics()
        summary.update({
            "actual_environment_steps": actual_steps,
            "optimizer_updates": int(getattr(trainer.model, "_n_updates", 0)),
            "training_wall_time_s": round(time.perf_counter() - started, 3),
            "steps_per_second": round(actual_steps / max(1e-9, time.perf_counter() - started), 3),
            "episodes_completed": len(trainer.callback.episode_records),
            "training_geometry_ids_encountered": sorted(checkpoint_callback.geometry_step_counts),
            "training_geometry_step_counts": checkpoint_callback.geometry_step_counts,
            "saved_checkpoint_steps": [0, *expected_saved],
            "final_policy_entropy_fingerprint": _policy_fingerprint(trainer.model),
            "initial_entropy_coefficient": initial["entropy_coefficient"],
            "final_entropy_coefficient": _alpha(trainer.model),
            "callback_evaluation_index": eval_callback.evaluations,
        })
        checkpoint_rows = []
        for step in [0, *expected_saved]:
            cp = checkpoints_dir / f"step_{step:05d}"
            metadata = verify_checkpoint_integrity(cp, expected_encoder_fingerprint=trainer.encoder.get_schema_fingerprint())
            context_path = cp / "checkpoint_context.json"
            checkpoint_rows.append({
                "training_step": step, "model_sha256": metadata["model_sha256"],
                "metadata_sha256": compute_file_sha256(cp / "metadata.json"),
                "context_sha256": compute_file_sha256(context_path) if context_path.is_file() else None,
                "transform_fingerprint": metadata.get("observation_transform_fingerprint"),
                "checkpoint_dir": str(cp.relative_to(ROOT)).replace("\\", "/"),
            })
        final_status = "EARLY_GO_MULTI_MAP_CANDIDATE" if valid_early else "COMPLETE"
        status.update({
            "status": final_status, "finish_utc": datetime.now(timezone.utc).isoformat(),
            "training_summary": summary, "checkpoints": checkpoint_rows,
            "elapsed_training_s": round(time.perf_counter() - started, 3),
            "resource_samples_file": str((checkpoints_dir / "resource_samples.json").relative_to(ROOT)).replace("\\", "/"),
        })
    except BaseException as exc:
        status.update({"status": "FAILED_OR_INCOMPLETE", "finish_utc": datetime.now(timezone.utc).isoformat(),
                       "actual_environment_steps": int(getattr(trainer.model, "num_timesteps", 0)),
                       "optimizer_updates": int(getattr(trainer.model, "_n_updates", 0)),
                       "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    finally:
        trainer.close()
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status


def run(run_id: str) -> Path:
    run_root = (ROOT / "runs/stage5_learnability" / run_id).resolve()
    results_dir = (ROOT / "results/stage5" / run_id).resolve()
    protocol_path = _write_protocol(results_dir, run_root, run_id)
    arm_a = _run_arm("A", run_root, protocol_path)
    target_b = int(arm_a["training_summary"]["actual_environment_steps"])
    arm_b = _run_arm("B", run_root, protocol_path, target_steps=target_b)
    summary = {
        "protocol_sha256": json.loads(protocol_path.read_text(encoding="utf-8"))["protocol_sha256"],
        "arms": {"A": arm_a["status"], "B": arm_b["status"]},
        "actual_steps": {"A": arm_a["training_summary"]["actual_environment_steps"],
                         "B": arm_b["training_summary"]["actual_environment_steps"]},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_complete": all(arm["status"] in ("COMPLETE", "EARLY_GO_MULTI_MAP_CANDIDATE") for arm in (arm_a, arm_b)),
    }
    (results_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results_dir


def technical_smoke() -> Path:
    """Short, non-scientific end-to-end check of treatment, callback and inference."""
    split = load_pilot_split(ROOT / "results/stage5/pilot_split_v1.json")
    train_pool, diagnostic_pool = _select_geometry_pools(split)
    run_root = ROOT / "runs/stage5_learnability" / "s5_4_technical_smoke_20261119"
    if run_root.exists():
        raise FileExistsError(f"Refusing to overwrite technical smoke artifacts: {run_root}")
    run_root.mkdir(parents=True)
    trainer = _new_trainer([row["geometry_id"] for row in train_pool], GROUPWISE_PHYSICAL_V1["transform_id"])
    try:
        trainer.config["stage5_s54_smoke"] = {"scientific_result": False, "purpose": "technical_driver_verification"}
        cp_root = run_root / "checkpoints"
        cp_root.mkdir()
        cp0 = cp_root / "step_00000"
        trainer.save_checkpoint(cp0, save_replay_buffer=False)
        metadata0 = verify_checkpoint_integrity(cp0, expected_encoder_fingerprint=trainer.encoder.get_schema_fingerprint())
        if metadata0.get("observation_transform") != GROUPWISE_PHYSICAL_V1:
            raise AssertionError("Treatment transform was not written to checkpoint metadata")
        diag = diagnostic_pool[0]["geometry_id"]
        probe0 = _run_evaluation(cp0, run_root / "diagnostics/step_00000", [diag], PRIMARY_SCHEDULE_SEED, 20)
        callback = PilotCheckpointCallback(trainer, cp_root, checkpoints=[1000], requested_timesteps=1100,
                                          verbose=0, write_step_log=False)
        eval_callback = S54EvaluationCallback(run_root, "SMOKE", [diag], 1100, [1000], verbose=0)
        trainer.model.learn(total_timesteps=1100,
                            callback=CallbackList([trainer.callback, callback, eval_callback]),
                            reset_num_timesteps=True)
        if int(trainer.model.num_timesteps) != 1100 or callback.saved_steps != [1000]:
            raise AssertionError("Short smoke did not exercise continuous checkpoint callback at step 1k")
        cp1000 = cp_root / "step_01000"
        metadata = verify_checkpoint_integrity(cp1000, expected_encoder_fingerprint=trainer.encoder.get_schema_fingerprint())
        context = json.loads((cp1000 / "checkpoint_context.json").read_text(encoding="utf-8"))
        if context.get("action_fidelity") is not True:
            raise AssertionError(f"Action fidelity mismatch in checkpoint context: {context}")
        latest_model = trainer.model
        probe_rows = eval_callback.evaluations[0]["probe"]
        if not probe_rows["evaluation_state_unchanged"] or probe_rows["aggregate"]["probe_censored_count"] != 1:
            raise AssertionError("Snapshot evaluator mutated the model or misclassified the short probe")
        trainer.close()

        agent = Stage5SacAgent(checkpoint_path=cp1000, expected_sha256=metadata["model_sha256"])
        agent.reset(AgentPublicEpisodeContext())
        agent_input = AgentInputV1(
            profile_id=InputProfileId.STATE_DECISION_V1,
            core_observation=CoreObservationV1(np.zeros(259, dtype=np.float32)),
            traffic_context=TrafficContextV1.empty(), task_context=TaskContextV1.empty(), step_index=0,
        )
        decision = agent.act(agent_input)
        encoded = agent.encoder.encode(agent_input)
        from src.training.stage5.pilot_observation_transform import apply_observation_transform
        policy_input = apply_observation_transform(encoded, GROUPWISE_PHYSICAL_V1["transform_id"])
        inference_model = SAC.load(str(cp1000 / "model.zip"), device="cpu")
        expected_action, _ = inference_model.predict(policy_input, deterministic=True)
        np.testing.assert_allclose(decision.action_payload, expected_action, atol=1e-6)
        agent.close()
    finally:
        trainer.close()
    status = {
        "status": "TECHNICAL_SMOKE_PASS", "scientific_result": False,
        "training_steps": 1100, "optimizer_updates": int(getattr(latest_model, "_n_updates", 0)),
        "checkpoint_sha256": metadata["model_sha256"],
        "observation_transform": metadata["observation_transform"],
        "checkpoint_action_fidelity": context["action_fidelity"],
        "evaluation_read_only": probe_rows["evaluation_state_unchanged"],
        "probe_label_count": probe_rows["aggregate"]["probe_censored_count"],
        "agent_transform_inference_parity": True,
        "initial_probe_summary": probe0["aggregate"],
    }
    status_path = run_root / "technical_smoke_status.json"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)
    return status_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--arm", choices=("A", "B"), help=argparse.SUPPRESS)
    parser.add_argument("--run-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--protocol", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help=argparse.SUPPRESS)
    parser.add_argument("--technical-smoke", action="store_true",
                        help="Run a short non-scientific 1.1k-step treatment and evaluator integration check.")
    args = parser.parse_args()
    if args.arm:
        if args.run_root is None or args.protocol is None:
            parser.error("internal arm invocation requires --run-root and --protocol")
        _run_arm(args.arm, args.run_root.resolve(), args.protocol.resolve(), args.max_steps)
        return 0
    if args.technical_smoke:
        technical_smoke()
        return 0
    output = run(args.run_id)
    print(f"S5.4 run artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
