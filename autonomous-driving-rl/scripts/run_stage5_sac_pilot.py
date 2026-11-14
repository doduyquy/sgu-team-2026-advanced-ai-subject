"""Run one continuous 10k-step Stage 5 SAC pilot with fixed internal diagnostics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

try:  # Load torch before Panda3D on Windows.
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3.common.callbacks import CallbackList

from src.training.stage5.checkpoint import verify_checkpoint_integrity
from src.training.stage5.pilot_callbacks import PilotCheckpointCallback
from src.training.stage5.pilot_protocol import load_pilot_split, save_pilot_split
from src.training.stage5.sac_trainer import SacBaselineTrainer


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_progress(path: Path, **updates: object) -> None:
    current = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
    current.update(updates)
    current["updated_at_epoch_s"] = time.time()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(current, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def evaluate(checkpoint: Path, run_dir: Path, split_path: Path, *, probe: bool = False) -> subprocess.CompletedProcess:
    command = [sys.executable, str(ROOT / "scripts/evaluate_stage5_pilot.py"),
               "--checkpoint", str(checkpoint), "--split", str(split_path),
               "--output", str(run_dir / ("technical_probe" if probe else "diagnostics"))]
    command += ["--progress-file", str(run_dir / "progress.json"),
                "--progress-phase", "TECHNICAL_PROBE" if probe else "DIAGNOSTICS"]
    if probe:
        command += ["--max-cases", "1", "--max-steps-per-episode", "20"]
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("s5_3a_%Y%m%dT%H%M%SZ"))
    parser.add_argument("--skip-full-diagnostics", action="store_true",
                        help="Stop after the pre-registered 10k training run; diagnostics can be resumed manually.")
    args = parser.parse_args()
    run_dir = ROOT / "runs" / "stage5_pilot" / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing pilot run: {run_dir}")
    run_dir.mkdir(parents=True)
    split_path = save_pilot_split(ROOT, split_seed=5301)
    split = load_pilot_split(split_path)
    config_path = ROOT / "configs/experiments/stage5_pilot_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["split_sha256"] = split["split_sha256"]
    config["baseline_config_sha256"] = sha256_file(ROOT / "configs/experiments/stage5_rl_baseline_v1.json")
    (run_dir / "pilot_protocol.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record = {
        "protocol": config["protocol"], "run_id": args.run_id, "status": "RUNNING",
        "training_root_seed": 101, "split_sha256": split["split_sha256"],
        "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       capture_output=True, check=True).stdout.strip(),
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__) if torch is not None else None,
        "start_utc": datetime.now(timezone.utc).isoformat(), "checkpoint_steps": [0, 5000, 10000],
    }
    status_path = run_dir / "run_status.json"
    status_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_progress(run_dir / "progress.json", status="RUNNING", phase="PREPARING", training_step=0,
                   requested_timesteps=10000, checkpoint_steps=[0, 5000, 10000])
    trainer = None
    try:
        trainer = SacBaselineTrainer(training_root_seed=101, project_root=ROOT,
                                     allowed_geometry_ids=[x["geometry_id"] for x in split["pilot_train"]],
                                     device="cpu")
        trainer.config["pilot_protocol"] = {
            "protocol": config["protocol"], "split_sha256": split["split_sha256"],
            "training_membership_count": 156, "diagnostic_membership_count": 24,
            "training_allowed_geometry_ids_sha256": hashlib.sha256(
                "\n".join(sorted(x["geometry_id"] for x in split["pilot_train"])).encode()).hexdigest(),
        }
        if trainer.sampler.total_eligible_geometries != 156:
            raise AssertionError("SAC training sampler did not restrict itself to 156 PILOT_TRAIN geometries")
        checkpoints_dir = run_dir / "checkpoints"
        checkpoint0 = trainer.save_checkpoint(checkpoints_dir / "step_00000", save_replay_buffer=False)
        meta0 = verify_checkpoint_integrity(checkpoint0,
                    expected_encoder_fingerprint=trainer.encoder.get_schema_fingerprint())
        if meta0["total_timesteps"] != 0 or meta0["training_root_seed"] != 101:
            raise AssertionError("UNTRAINED checkpoint provenance mismatch")

        # A separate process owns the simulator. This probe is explicitly interrupted and is
        # never counted as a timeout or as an empirical completed episode.
        probe = evaluate(checkpoint0, run_dir, split_path, probe=True)
        (run_dir / "technical_probe.stdout.txt").write_text(probe.stdout, encoding="utf-8")
        (run_dir / "technical_probe.stderr.txt").write_text(probe.stderr, encoding="utf-8")
        if probe.returncode != 0:
            raise RuntimeError(f"Evaluator technical probe failed with exit code {probe.returncode}")

        checkpoint_callback = PilotCheckpointCallback(
            trainer, checkpoints_dir, [5000, 10000], requested_timesteps=10000, verbose=1
        )
        started = time.perf_counter()
        trainer.model.learn(total_timesteps=10000,
                            callback=CallbackList([trainer.callback, checkpoint_callback]),
                            reset_num_timesteps=True)
        elapsed = time.perf_counter() - started
        summary = trainer.callback.get_summary_metrics()
        summary.update({
            "requested_timesteps": 10000, "actual_num_timesteps": int(trainer.model.num_timesteps),
            "optimizer_updates": int(getattr(trainer.model, "_n_updates", 0)),
            "unique_training_geometries": len({r["geometry_id"] for r in trainer.callback.episode_records}),
            "training_wall_time_s": elapsed,
            "training_steps_per_second": int(trainer.model.num_timesteps) / max(elapsed, 0.001),
            "checkpoint_steps_saved": [0] + checkpoint_callback.saved_steps,
            "training_action_summary": json.loads(
                (checkpoints_dir / "training_action_summary.json").read_text(encoding="utf-8")
            ),
        })
        (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if (run_dir / "STOP_REQUESTED").exists():
            actual_step = int(trainer.model.num_timesteps)
            partial_checkpoint = checkpoints_dir / f"step_{actual_step:05d}"
            if not partial_checkpoint.exists():
                trainer.save_checkpoint(partial_checkpoint, save_replay_buffer=False)
            metadata = verify_checkpoint_integrity(partial_checkpoint,
                         expected_encoder_fingerprint=trainer.encoder.get_schema_fingerprint())
            record.update({
                "status": "INTERRUPTED", "training_summary": summary,
                "stop_reason": "USER_REQUESTED_GRACEFUL_STOP",
                "partial_checkpoint_step": actual_step,
                "partial_checkpoint_sha256": metadata["model_sha256"],
                "finish_utc": datetime.now(timezone.utc).isoformat(),
            })
            status_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            write_progress(run_dir / "progress.json", status="INTERRUPTED", phase="STOPPED_BY_USER",
                           training_step=actual_step, requested_timesteps=10000,
                           partial_checkpoint_step=actual_step)
            return 0
        if int(trainer.model.num_timesteps) != 10000 or checkpoint_callback.saved_steps != [5000, 10000]:
            raise RuntimeError("Pilot stopped before exact 10,000 steps or a required checkpoint was missed")
        for step in (0, 5000, 10000):
            meta = verify_checkpoint_integrity(checkpoints_dir / f"step_{step:05d}",
                                               expected_encoder_fingerprint=trainer.encoder.get_schema_fingerprint())
            if meta["total_timesteps"] != step or meta["training_root_seed"] != 101:
                raise AssertionError(f"Checkpoint provenance mismatch at step {step}")
        record["status"] = "TRAINING_COMPLETE_DIAGNOSTICS_PENDING"
        record["training_summary"] = summary
        record["checkpoint_sha256"] = {
            str(step): verify_checkpoint_integrity(checkpoints_dir / f"step_{step:05d}")["model_sha256"]
            for step in (0, 5000, 10000)
        }
        status_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        trainer.close()
        trainer = None

        if not args.skip_full_diagnostics:
            for step in (0, 5000, 10000):
                write_progress(run_dir / "progress.json", status="RUNNING", phase="DIAGNOSTICS",
                               diagnostic_checkpoint_step=step, diagnostic_cases_completed=0,
                               diagnostic_cases_planned=24)
                result = evaluate(checkpoints_dir / f"step_{step:05d}", run_dir, split_path)
                (run_dir / f"diagnostics_step_{step:05d}.stdout.txt").write_text(result.stdout, encoding="utf-8")
                (run_dir / f"diagnostics_step_{step:05d}.stderr.txt").write_text(result.stderr, encoding="utf-8")
                if result.returncode != 0:
                    record["status"] = "DIAGNOSTICS_INCOMPLETE"
                    status_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    raise RuntimeError(f"Full diagnostic evaluator failed for step {step}: {result.returncode}")
            report_dir = ROOT / "results/stage5" / args.run_id
            summary_process = subprocess.run(
                [sys.executable, str(ROOT / "scripts/summarize_stage5_pilot.py"),
                 "--run-dir", str(run_dir), "--output-dir", str(report_dir)],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
            (run_dir / "summary_builder.stdout.txt").write_text(summary_process.stdout, encoding="utf-8")
            (run_dir / "summary_builder.stderr.txt").write_text(summary_process.stderr, encoding="utf-8")
            if summary_process.returncode != 0:
                raise RuntimeError(f"Empirical report builder failed with exit code {summary_process.returncode}")
            shutil.copy2(split_path, report_dir / "pilot_split_v1.json")
            shutil.copy2(config_path, report_dir / "pilot_protocol_v1.json")
            record["status"] = "COMPLETE"
            write_progress(run_dir / "progress.json", status="COMPLETE", phase="COMPLETE",
                           training_step=10000, requested_timesteps=10000,
                           diagnostic_checkpoints_complete=[0, 5000, 10000])
        else:
            record["status"] = "TRAINING_COMPLETE_DIAGNOSTICS_PENDING"
            write_progress(run_dir / "progress.json", status="TRAINING_COMPLETE_DIAGNOSTICS_PENDING",
                           phase="TRAINING_COMPLETE", training_step=10000, requested_timesteps=10000)
        record["finish_utc"] = datetime.now(timezone.utc).isoformat()
        status_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    except Exception as exc:
        record["status"] = "FAILED_OR_INCOMPLETE"
        record["error"] = repr(exc)
        record["traceback"] = traceback.format_exc()
        record["finish_utc"] = datetime.now(timezone.utc).isoformat()
        status_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_progress(run_dir / "progress.json", status="FAILED_OR_INCOMPLETE", phase="FAILED",
                       error=repr(exc), training_step=record.get("training_summary", {}).get("actual_num_timesteps", 0))
        return 1
    finally:
        if trainer is not None:
            trainer.close()


if __name__ == "__main__":
    raise SystemExit(main())
