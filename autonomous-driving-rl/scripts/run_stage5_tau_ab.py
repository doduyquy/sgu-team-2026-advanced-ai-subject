"""Run the registered fresh SAC tau-only A/B pilot on internal TRAIN data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List

try:  # Load Torch before Panda3D on Windows.
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList

from src.training.stage5.checkpoint import compute_file_sha256, verify_checkpoint_integrity
from src.training.stage5.pilot_callbacks import PilotCheckpointCallback
from src.training.stage5.pilot_protocol import canonical_sha256, diagnostic_schedule, load_pilot_split
from src.training.stage5.observation_encoder import STAGE5_FULL_OBSERVATION_DIM
from src.training.stage5.sac_trainer import SacBaselineTrainer
from src.training.stage5.train_sampler import SamplerStrategy


RUN_ID = "s5_3m_tau_ab_seed101_20261119"
TAUS = {"A": 0.005, "B": 0.001}
STEPS = 10_000
CHECKPOINT_STEPS = (0, 5_000, 10_000)
DIAGNOSTIC_LIMIT = 300


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


def _entropy_coefficient(model: SAC) -> float:
    log_ent = getattr(model, "log_ent_coef", None)
    if log_ent is not None:
        return float(log_ent.detach().exp().cpu().item())
    return float(model.ent_coef)


def _baseline_config() -> Dict[str, Any]:
    return json.loads((ROOT / "configs/experiments/stage5_rl_baseline_v1.json").read_text(encoding="utf-8"))


def _protocol(run_id: str) -> Dict[str, Any]:
    registration_path = ROOT / "configs/experiments/stage5_tau_ab_v1.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    split_path = ROOT / "results/stage5/pilot_split_v1.json"
    split = load_pilot_split(split_path)
    schedule = diagnostic_schedule(split, schedule_seed=5302)
    tier_order = ("Easy", "Medium", "Hard", "Extreme")
    selected: List[Dict[str, Any]] = []
    for tier in tier_order:
        selected.extend([item for item in schedule if item["tier"] == tier][:2])
    if len(selected) != 8 or {tier: sum(row["tier"] == tier for row in selected) for tier in tier_order} != {tier: 2 for tier in tier_order}:
        raise ValueError("The locked internal diagnostic split must provide two geometries per tier")
    if [row["geometry_id"] for row in selected] != registration["diagnostic_geometry_ids"]:
        raise ValueError("Generated fixed diagnostic geometry IDs differ from the registered protocol")

    baseline = _baseline_config()
    if (registration["protocol"] != "stage5_tau_ab_v1" or
            int(registration["training_root_seed"]) != 101 or
            int(registration["training_steps_per_arm"]) != STEPS or
            tuple(registration["checkpoint_steps"]) != CHECKPOINT_STEPS or
            registration["sampler_strategy"] != "TIER_ROUND_ROBIN" or
            registration["arms"] != {"A": {"tau": TAUS["A"]}, "B": {"tau": TAUS["B"]}}):
        raise ValueError("Registration file changed from the supported tau-only pilot protocol")
    if STAGE5_FULL_OBSERVATION_DIM != 409:
        raise ValueError("The registered tau experiment requires the 409D full encoder")
    if float(baseline["sac_hyperparameters"]["gamma"]) != float(registration["common_parameters"]["gamma"]):
        raise ValueError("Baseline gamma differs from the registered tau-only protocol")
    if float(baseline["sac_hyperparameters"]["learning_rate"]) != float(registration["common_parameters"]["learning_rate"]):
        raise ValueError("Baseline learning rate differs from the registered tau-only protocol")
    protocol: Dict[str, Any] = {
        "protocol": "stage5_tau_ab_v1",
        "run_id": run_id,
        "status": "PRE_REGISTERED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       capture_output=True, check=True).stdout.strip(),
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "torch": str(torch.__version__) if torch is not None else None},
        "source_manifest": "results/audits/evaluation_protocol/geometry_split_manifest.csv",
        "pilot_split_path": str(split_path.relative_to(ROOT)).replace("\\", "/"),
        "pilot_split_sha256": split["split_sha256"],
        "registered_config_sha256": compute_file_sha256(registration_path),
        "pilot_train_geometry_ids_sha256": hashlib.sha256("\n".join(sorted(
            row["geometry_id"] for row in split["pilot_train"])).encode()).hexdigest(),
        "diagnostic_schedule_seed": int(registration["diagnostic_schedule_seed"]),
        "diagnostic_geometry_ids": [row["geometry_id"] for row in selected],
        "diagnostic_schedule": selected,
        "diagnostic_max_steps_per_probe": int(registration["diagnostic_max_steps_per_case"]),
        "diagnostic_interrupted_label": "PROBE_CENSORED",
        "training_root_seed": 101,
        "training_steps": int(registration["training_steps_per_arm"]),
        "checkpoint_steps": list(registration["checkpoint_steps"]),
        "sampler_strategy": registration["sampler_strategy"],
        "device": "cpu",
        "common_hyperparameters": {
            "algorithm": "SAC", "gamma": registration["common_parameters"]["gamma"],
            "learning_rate": registration["common_parameters"]["learning_rate"],
            "observation_dim": STAGE5_FULL_OBSERVATION_DIM, "architecture": registration["common_parameters"]["actor_critic"],
            "reward_contract": baseline["reward_contract"],
            "buffer_size": baseline["sac_hyperparameters"]["buffer_size"],
            "batch_size": baseline["sac_hyperparameters"]["batch_size"],
            "learning_starts": baseline["sac_hyperparameters"]["learning_starts"],
            "ent_coef": baseline["sac_hyperparameters"]["ent_coef"],
            "train_freq": baseline["sac_hyperparameters"]["train_freq"],
            "gradient_steps": baseline["sac_hyperparameters"]["gradient_steps"],
            "normalization": "none",
        },
        "arms": registration["arms"],
        "only_registered_arm_difference": "sac_hyperparameters.tau",
        "candidate_eligibility": {"agent_id": "stage5_sac", "status": "CANDIDATE",
                                  "benchmark_eligible": False},
        "holdouts_accessed": False,
        "historical_checkpoints_used_for_initialization": False,
        "reproducibility_note": "One shared initialization and root seed; MetaDrive and CPU runtime do not guarantee bitwise-identical trajectories.",
    }
    protocol["protocol_sha256"] = canonical_sha256(protocol)
    return protocol


def _trainer(tau: float, allowed_ids: List[str]) -> SacBaselineTrainer:
    overrides = {
        "dataset_sampler": {"default_strategy": "TIER_ROUND_ROBIN"},
        "sac_hyperparameters": {"gamma": 0.99, "learning_rate": 0.0003, "tau": float(tau)},
    }
    return SacBaselineTrainer(
        training_root_seed=101, project_root=ROOT, device="cpu",
        allowed_geometry_ids=allowed_ids, config_overrides=overrides,
    )


def _make_protocol_and_initial(run_root: Path, results_dir: Path, run_id: str) -> Path:
    if run_root.exists() or results_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run/results: {run_root} / {results_dir}")
    run_root.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    protocol = _protocol(run_id)
    protocol_path = results_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    split = load_pilot_split(ROOT / protocol["pilot_split_path"])
    allowed = [row["geometry_id"] for row in split["pilot_train"]]
    trainer = _trainer(TAUS["A"], allowed)
    try:
        initial_path = run_root / "common_initial_model.zip"
        trainer.model.save(str(initial_path))
        initial_state = {
            "model_sha256": compute_file_sha256(initial_path),
            "policy_and_entropy_fingerprint": _policy_fingerprint(trainer.model),
            "log_entropy_coefficient": float(trainer.model.log_ent_coef.detach().cpu().item())
                if getattr(trainer.model, "log_ent_coef", None) is not None else None,
            "entropy_coefficient": _entropy_coefficient(trainer.model),
            "tau_at_initialization": float(trainer.model.tau),
            "num_timesteps": int(trainer.model.num_timesteps),
            "optimizer_updates": int(getattr(trainer.model, "_n_updates", 0)),
            "seed": 101,
            "source_code_commit": protocol["code_commit"],
        }
        (run_root / "common_initial_state.json").write_text(
            json.dumps(initial_state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    finally:
        trainer.close()
    return protocol_path


def _run_arm(arm: str, run_root: Path, protocol_path: Path) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_sha256") != canonical_sha256({k: v for k, v in protocol.items() if k != "protocol_sha256"}):
        raise ValueError("Protocol SHA-256 mismatch")
    tau = TAUS[arm]
    arm_root = run_root / f"arm_{arm}"
    if arm_root.exists():
        raise FileExistsError(f"Refusing to overwrite arm directory: {arm_root}")
    arm_root.mkdir(parents=True)
    status_path = arm_root / "run_status.json"
    status: Dict[str, Any] = {"status": "RUNNING", "arm": arm, "tau": tau,
                              "start_utc": datetime.now(timezone.utc).isoformat(),
                              "code_commit": protocol["code_commit"],
                              "protocol_sha256": protocol["protocol_sha256"]}
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    split = load_pilot_split(ROOT / protocol["pilot_split_path"])
    allowed = [row["geometry_id"] for row in split["pilot_train"]]
    trainer = _trainer(tau, allowed)
    started = time.perf_counter()
    try:
        base_model = run_root / "common_initial_model.zip"
        if compute_file_sha256(base_model) != json.loads((run_root / "common_initial_state.json").read_text())["model_sha256"]:
            raise ValueError("Shared initialization model hash mismatch")
        loaded = SAC.load(str(base_model), env=trainer.env, device="cpu")
        loaded.tau = tau
        trainer.model = loaded
        expected = json.loads((run_root / "common_initial_state.json").read_text())
        actual_fingerprint = _policy_fingerprint(trainer.model)
        if actual_fingerprint != expected["policy_and_entropy_fingerprint"]:
            raise ValueError(f"Arm {arm} initial model/entropy differs from shared initialization")
        if int(trainer.model.num_timesteps) != 0 or int(getattr(trainer.model, "_n_updates", 0)) != 0:
            raise ValueError("Fresh A/B arm did not start at zero steps and zero optimizer updates")
        if _entropy_coefficient(trainer.model) != float(expected["entropy_coefficient"]):
            raise ValueError("Initial entropy coefficient differs from shared initialization")

        # Re-seed collection RNGs after loading, so both arms start their collectors from the same roots.
        random.seed(101)
        import numpy as np
        np.random.seed(101)
        if torch is not None:
            torch.manual_seed(101)
        trainer.model.action_space.seed(101)
        checkpoints_dir = arm_root / "checkpoints"
        checkpoints_dir.mkdir()
        trainer.config["pilot_tau_ab"] = {
            "protocol_sha256": protocol["protocol_sha256"], "arm": arm,
            "tau": tau, "common_initial_sha256": expected["model_sha256"],
            "common_initial_policy_entropy_fingerprint": expected["policy_and_entropy_fingerprint"],
        }
        initial_dir = checkpoints_dir / "step_00000"
        trainer.save_checkpoint(initial_dir, save_replay_buffer=False)
        callback = PilotCheckpointCallback(
            trainer, checkpoints_dir, checkpoints=(5_000, 10_000), requested_timesteps=STEPS,
            stop_file=arm_root / "STOP_REQUESTED", verbose=0,
        )
        trainer.model.learn(total_timesteps=STEPS,
                            callback=CallbackList([trainer.callback, callback]),
                            reset_num_timesteps=True)
        if int(trainer.model.num_timesteps) != STEPS or callback.saved_steps != [5_000, 10_000]:
            raise RuntimeError(f"Incomplete arm: steps={trainer.model.num_timesteps}, saved={callback.saved_steps}")
        summary = trainer.callback.get_summary_metrics()
        coverage_path = checkpoints_dir / "training_geometry_coverage.json"
        action_path = checkpoints_dir / "training_action_summary.json"
        resource_path = checkpoints_dir / "resource_samples.json"
        summary.update({
            "actual_num_timesteps": int(trainer.model.num_timesteps),
            "optimizer_updates": int(getattr(trainer.model, "_n_updates", 0)),
            "training_wall_time_s": round(time.perf_counter() - started, 3),
            "checkpoint_steps_saved": [0, *callback.saved_steps],
            "training_geometry_coverage": json.loads(coverage_path.read_text(encoding="utf-8")),
            "training_action_summary": json.loads(action_path.read_text(encoding="utf-8")),
            "resource_samples": json.loads(resource_path.read_text(encoding="utf-8")),
            "initial_policy_entropy_fingerprint": expected["policy_and_entropy_fingerprint"],
            "final_policy_entropy_fingerprint": _policy_fingerprint(trainer.model),
        })
        checkpoint_rows = []
        for step in CHECKPOINT_STEPS:
            cp = checkpoints_dir / f"step_{step:05d}"
            meta = verify_checkpoint_integrity(cp, expected_encoder_fingerprint=trainer.encoder.get_schema_fingerprint())
            if int(meta.get("total_timesteps", -1)) != step:
                raise ValueError(f"Checkpoint step mismatch in {cp}")
            checkpoint_rows.append({"training_step": step, "model_sha256": meta["model_sha256"],
                                    "metadata_sha256": compute_file_sha256(cp / "metadata.json"),
                                    "checkpoint_dir": str(cp.relative_to(ROOT)).replace("\\", "/")})
        status.update({"status": "COMPLETE", "finish_utc": datetime.now(timezone.utc).isoformat(),
                       "training_summary": summary, "checkpoints": checkpoint_rows,
                       "initial_policy_entropy_fingerprint": expected["policy_and_entropy_fingerprint"]})
    except BaseException as exc:
        status.update({"status": "FAILED_OR_INCOMPLETE", "finish_utc": datetime.now(timezone.utc).isoformat(),
                       "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    finally:
        trainer.close()
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _child(command: List[str], log_base: Path) -> None:
    log_base.parent.mkdir(parents=True, exist_ok=True)
    with log_base.with_suffix(".stdout.log").open("w", encoding="utf-8") as stdout, \
         log_base.with_suffix(".stderr.log").open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
    if proc.returncode:
        raise RuntimeError(f"Child process failed ({proc.returncode}); see {log_base}.*.log")


def _run_evaluations(run_root: Path, protocol_path: Path) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    geometry_ids = list(protocol["diagnostic_geometry_ids"])
    python_script = str(ROOT / "scripts/evaluate_stage5_pilot.py")
    anchor_dir = run_root / "anchor_observations"
    _child([sys.executable, python_script, "--checkpoint",
            str(run_root / "arm_A/checkpoints/step_00000"), "--split",
            str(ROOT / protocol["pilot_split_path"]), "--output", str(anchor_dir / "records"),
            "--schedule-seed", str(protocol["diagnostic_schedule_seed"]),
            "--max-steps-per-episode", "1", "--trace-dir", str(anchor_dir / "traces"),
            *sum((["--geometry-id", value] for value in geometry_ids), [])],
           anchor_dir / "anchor_eval")
    for arm in ("A", "B"):
        for step in CHECKPOINT_STEPS:
            cp = run_root / f"arm_{arm}/checkpoints/step_{step:05d}"
            output = run_root / f"arm_{arm}/diagnostics/step_{step:05d}"
            command = [sys.executable, python_script, "--checkpoint", str(cp), "--split",
                       str(ROOT / protocol["pilot_split_path"]), "--output", str(output),
                       "--schedule-seed", str(protocol["diagnostic_schedule_seed"]),
                       "--max-steps-per-episode", str(DIAGNOSTIC_LIMIT)]
            command += sum((["--geometry-id", value] for value in geometry_ids), [])
            log_base = output / "evaluation"
            _child(command, log_base)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--arm", choices=("A", "B"), help=argparse.SUPPRESS)
    parser.add_argument("--run-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--protocol", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--evaluate-only", action="store_true", help="Run only the pre-registered diagnostic schedule for completed arms.")
    args = parser.parse_args()
    if args.arm:
        if args.run_root is None or args.protocol is None:
            parser.error("internal arm invocation requires --run-root and --protocol")
        _run_arm(args.arm, args.run_root.resolve(), args.protocol.resolve())
        return 0

    run_root = (ROOT / "runs/stage5_tau_ab" / args.run_id).resolve()
    results_dir = ROOT / "results/stage5" / args.run_id
    if args.evaluate_only:
        protocol_path = results_dir / "protocol.json"
        if not protocol_path.is_file():
            parser.error(f"Missing preregistered protocol: {protocol_path}")
        _run_evaluations(run_root, protocol_path)
        return 0

    protocol_path = _make_protocol_and_initial(run_root, results_dir, args.run_id)
    try:
        for arm in ("A", "B"):
            _child([sys.executable, str(Path(__file__).resolve()), "--arm", arm,
                    "--run-root", str(run_root), "--protocol", str(protocol_path)],
                   run_root / f"arm_{arm}/training")
        _run_evaluations(run_root, protocol_path)
    except BaseException:
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
