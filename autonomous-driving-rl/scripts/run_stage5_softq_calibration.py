"""S5.3L L1: eight paired stochastic continuations per fixed K1 intervention."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "s5_3l_softq_calibration_seed101_20261115"
REPLICATES = 8
BASE_SUMMARY = ROOT / "results/stage5/s5_3k_action_intervention_seed101_20261115/summary.json"
BASE_RAW = ROOT / "runs/stage5_pilot/s5_3k_action_intervention_seed101_20261115/transition_records.jsonl"
CHECKPOINT_MODEL = ROOT / "runs/stage5_pilot/s5_3g_throttle_hold5_seed101_20261115/checkpoints/step_10000/model.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    base = json.loads(BASE_SUMMARY.read_text(encoding="utf-8"))
    if base.get("status") != "COMPLETE" or len(base.get("episodes", [])) != 12:
        raise RuntimeError("K1 baseline must be complete with exactly 12 episodes")
    if not all(row.get("reset_parity_verified") for row in base["episodes"]):
        raise RuntimeError("K1 baseline reset parity is incomplete")
    if not CHECKPOINT_MODEL.is_file() or sha256(CHECKPOINT_MODEL) != base["checkpoint_sha256"]:
        raise RuntimeError("Current checkpoint bytes do not match the K1 provenance SHA")
    if not BASE_RAW.is_file():
        raise FileNotFoundError(BASE_RAW)

    out_root = ROOT / "results/stage5" / RUN_ID
    raw_root = ROOT / "runs/stage5_pilot" / RUN_ID
    out_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    replicate_dirs = [out_root / f"replicate_{i:02d}" for i in range(1, REPLICATES + 1)]
    raw_dirs = [raw_root / f"replicate_{i:02d}" for i in range(1, REPLICATES + 1)]
    if any(p.exists() for p in replicate_dirs + raw_dirs):
        raise FileExistsError("L1 replicate output path already exists; refusing to overwrite prior data")

    baseline_runtime = float(base["wall_time_s"])
    expected_runtime = baseline_runtime * REPLICATES
    estimated_bytes = BASE_RAW.stat().st_size * REPLICATES
    manifest = {
        "protocol": "S5.3L-L1 soft-Q calibration repeated frozen rollouts v1",
        "status": "RUNNING", "replicates_planned": REPLICATES,
        "planned_rollouts": 12 * REPLICATES,
        "checkpoint_sha256": base["checkpoint_sha256"],
        "pilot_split_sha256": base["pilot_split_sha256"],
        "baseline_k1_summary_sha256": sha256(BASE_SUMMARY),
        "baseline_k1_raw_sha256": sha256(BASE_RAW),
        "estimated_wall_time_s": expected_runtime,
        "estimated_raw_trace_bytes": estimated_bytes,
        "budget_basis": "linear extrapolation from complete K1: 942.5 s for 12 rollouts, 93,472,992 raw trace bytes",
        "replicate_states": [],
        "prohibited_changes": ["weights", "optimizer", "replay buffer", "normalization", "reward", "gamma"],
        "holdouts_accessed": False,
    }
    manifest_path = out_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    overall_start = time.perf_counter()
    runner = ROOT / "scripts/run_stage5_action_intervention.py"
    for replicate_id in range(1, REPLICATES + 1):
        start = time.perf_counter()
        command = [sys.executable, "-u", str(runner), "--replicate-id", str(replicate_id), "--run-id", RUN_ID]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        summary_path = out_root / f"replicate_{replicate_id:02d}" / "summary.json"
        replicate_status = "COMPLETE" if completed.returncode == 0 and summary_path.is_file() else "FAILED"
        item = {
            "replicate_id": replicate_id,
            "status": replicate_status,
            "exit_code": completed.returncode,
            "wall_time_s": time.perf_counter() - start,
            "summary_sha256": sha256(summary_path) if summary_path.is_file() else None,
        }
        manifest["replicate_states"].append(item)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(item), flush=True)
        if replicate_status != "COMPLETE":
            manifest["status"] = "FAILED_PARTIAL"
            manifest["wall_time_s"] = time.perf_counter() - overall_start
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return completed.returncode or 1

    manifest["status"] = "COMPLETE"
    manifest["wall_time_s"] = time.perf_counter() - overall_start
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
