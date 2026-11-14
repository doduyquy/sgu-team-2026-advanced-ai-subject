"""Đo chi phí thu TRAIN và kiểm tra môi trường tensor; không huấn luyện Agent 4."""

import argparse
from datetime import datetime, timezone
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psutil
from scripts.audit_stage4_telemetry import git
from scripts.stage4 import fingerprints, seal_attempt, write_record
from src.imitation.audit import EvidenceError, load_json, project_path, sha256, source_hashes, train_geometries
from src.imitation.pilot import verify_seal
from src.imitation.resources import disk_inventory, project_costs, summarize_phases, summarize_samples
from src.launcher.models import validate_custom_run_id


EXTRA_FILES = ["scripts/profile_stage4_resources.py", "scripts/collect_stage4_pilot.py",
               "scripts/audit_stage4_telemetry.py", "scripts/stage4.py", "scripts/run_tests.py",
               "scripts/replay_rule_based_trace.py", "experiments/stage4/g1_resources_v1.json",
               "experiments/stage4/g1_pilot_v1.json", "experiments/stage4/plan_v1.json"]
SOURCE_DIRS = ["src", "tests", "configs/platform", "datasets/mapsuite_v1"]


def now():
    return datetime.now(timezone.utc).isoformat()


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def command_record(command):
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf8", errors="replace",
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return {"command": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def inspect_environment(output):
    memory = psutil.virtual_memory()
    hardware = {"platform": platform.platform(), "processor_identifier": platform.processor(),
                "physical_cpu_cores": psutil.cpu_count(logical=False), "logical_cpu_cores": psutil.cpu_count(),
                "physical_ram_bytes": memory.total, "available_ram_at_inspection_bytes": memory.available,
                "system_cpu_percent_before_workload": psutil.cpu_percent(interval=1.0),
                "disk_free_at_inspection_bytes": shutil.disk_usage(ROOT).free}
    cpu_query = command_record(["powershell", "-NoProfile", "-Command",
               "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress"])
    write_record(output / "cpu_query.json", cpu_query)
    if cpu_query["exit_code"] == 0:
        hardware["cpu_wmi"] = json.loads(cpu_query["stdout"])
    smi = shutil.which("nvidia-smi")
    gpu = command_record([smi, "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
                          "--format=csv,noheader,nounits"]) if smi else {"exit_code": None, "reason": "nvidia-smi unavailable"}
    write_record(output / "gpu_query.json", gpu)
    hardware["gpu_query"] = {k: v for k, v in gpu.items() if k != "command"}
    write_record(output / "hardware.json", hardware)
    versions = {name: package_version(name) for name in
                ("numpy", "panda3d", "metadrive-simulator", "wandb", "psutil", "torch")}
    pip_check = command_record([sys.executable, "-B", "-m", "pip", "check"])
    write_record(output / "pip_check.json", pip_check)
    freeze = command_record([sys.executable, "-B", "-m", "pip", "freeze", "--all"])
    write_record(output / "pip_freeze.json", freeze)
    tensor = {"status": "NOT_INSTALLED", "reason": "PyTorch is not installed in the selected MetaDrive environment",
              "optimizer_updates": 0, "stage4_training_updates": 0}
    if versions["torch"]:
        code = """import json, torch
devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
results = []
for device in devices:
    x = torch.tensor([1., 2.], device=device, requires_grad=True)
    (x * x).sum().backward()
    assert torch.isfinite(x.grad).all().item()
    results.append({'device': device, 'gradient': x.grad.cpu().tolist()})
print(json.dumps({'torch_version': torch.__version__, 'torch_cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'results': results, 'optimizer_updates': 0, 'stage4_training_updates': 0}))
"""
        smoke = command_record([sys.executable, "-B", "-X", "utf8", "-c", code])
        write_record(output / "tensor_smoke.json", smoke)
        tensor = {"status": "PASSED" if smoke["exit_code"] == 0 else "FAILED",
                  "optimizer_updates": 0, "stage4_training_updates": 0}
        if smoke["exit_code"] == 0:
            tensor["result"] = json.loads(smoke["stdout"].strip().splitlines()[-1])
    ready = tensor["status"] == "PASSED" and pip_check["exit_code"] == 0
    environment = {"python": sys.version, "executable": sys.executable, "packages": versions,
                   "pip_check_exit_code": pip_check["exit_code"], "tensor_smoke": tensor,
                   "training_tensor_environment_status": "SMOKE_PASSED" if ready else "NOT_READY",
                   "stage4_training_pipeline_validated": False, "dependency_installations": 0,
                   "training_updates": 0, "note": "Hardware/driver visibility alone does not verify PyTorch CUDA compatibility"}
    write_record(output / "environment_check.json", environment)
    return hardware, environment


def sample_tree(pid, elapsed, phase):
    rows, denied, vanished = [], 0, 0
    try:
        parent = psutil.Process(pid)
        candidates = [parent] + parent.children(recursive=True)
    except psutil.NoSuchProcess:
        candidates, vanished = [], 1
    except psutil.AccessDenied:
        candidates, denied = [], 1
    seen = set()
    for process in candidates:
        try:
            with process.oneshot():
                key = (process.pid, process.create_time())
                if key in seen:
                    continue
                memory = process.memory_info()
                rows.append({"pid": process.pid, "create_time": key[1], "rss_bytes": memory.rss,
                             "virtual_bytes": memory.vms, "name": process.name()})
                seen.add(key)
        except psutil.NoSuchProcess:
            vanished += 1
        except psutil.AccessDenied:
            denied += 1
    return {"at_utc": now(), "elapsed_s": elapsed, "episode_id": phase.get("episode_id"),
            "phase": phase.get("phase", "setup"), "processes": rows,
            "tree_rss_bytes": sum(p["rss_bytes"] for p in rows),
            "profiler_rss_bytes": psutil.Process().memory_info().rss,
            "system_available_bytes": psutil.virtual_memory().available,
            "access_denied": denied, "vanished": vanished}


def run_workload(output, command, interval):
    rows, events, phase, pending = [], [], {}, ""
    console_path = output / "collector_console.log"
    start = time.perf_counter()
    with console_path.open("xb") as console, gzip.open(output / "resource_samples.jsonl.gz", "xt", encoding="utf8") as trace:
        process = subprocess.Popen(command, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT,
                  env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"},
                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            with console_path.open(encoding="utf8", errors="replace") as reader:
                while True:
                    elapsed = time.perf_counter() - start
                    pending += reader.read()
                    lines = pending.split("\n")
                    pending = lines.pop()
                    for line in lines:
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if event.get("phase") in ("collecting", "verified"):
                            event["observed_elapsed_s"] = elapsed
                            events.append(event)
                            phase = event if event["phase"] == "collecting" else {"phase": "between_cases_or_finalization"}
                            print(json.dumps(event), flush=True)
                    row = sample_tree(process.pid, elapsed, phase)
                    rows.append(row)
                    trace.write(json.dumps(row, separators=(",", ":")) + "\n")
                    if process.poll() is not None:
                        # Drain any final complete lines after process exit.
                        rest = pending + reader.read()
                        for line in rest.splitlines():
                            try:
                                event = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if event.get("phase") in ("collecting", "verified"):
                                event["observed_elapsed_s"] = time.perf_counter() - start
                                events.append(event)
                        break
                    trace.flush()
                    time.sleep(interval)
        except BaseException:
            for child in psutil.Process(process.pid).children(recursive=True):
                child.terminate()
            process.terminate()
            process.wait()
            raise
    wall = time.perf_counter() - start
    write_record(output / "phase_events.json", events)
    if process.returncode:
        raise EvidenceError("Lần thu dùng để đo thất bại; xem collector_console.log")
    return rows, events, wall


def profile(run_id):
    validate_custom_run_id(run_id, ROOT / "runs")
    child_id = run_id + "_collect"
    validate_custom_run_id(child_id, ROOT / "runs")
    if (ROOT / "runs" / child_id).exists():
        raise FileExistsError("ID tiến trình thu đã tồn tại")
    output = ROOT / "runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    state = {"run_id": run_id, "status": "RUNNING", "started_at_utc": now(), "scope": "G1_STEP3_ONLY",
             "child_run_id": child_id, "training_updates": 0, "g1_complete": False}
    write_record(output / "run_state.json", state)
    try:
        manifest = load_json(ROOT / "experiments/stage4/g1_resources_v1.json")
        registry = load_json(ROOT / manifest["reference_registry"])
        reference = ROOT / "runs" / registry["source_run_id"]
        reference_seal = verify_seal(reference)
        if sha256(reference / "run_integrity.json") != registry["source_integrity_sha256"]:
            raise EvidenceError("Mã băm pilot tham chiếu khác sổ đăng ký")
        sources = source_hashes(ROOT, SOURCE_DIRS, EXTRA_FILES)
        for name in sources:
            target = output / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write((ROOT / name).read_bytes())
        command = [sys.executable, "-B", "-X", "utf8", "scripts/collect_stage4_pilot.py", "collect", "--run-id", child_id]
        write_record(output / "request.json", {"manifest": manifest, "source_sha256": sources, "command": command,
                     "reference_registry_sha256": sha256(ROOT / manifest["reference_registry"]),
                     "reference_integrity_sha256": registry["source_integrity_sha256"],
                     "project_commit": git(ROOT, "rev-parse", "HEAD"), "project_status": git(ROOT, "status", "--porcelain")})
        hardware, environment = inspect_environment(output)
        print(json.dumps({"phase": "environment_checked", "training_environment": environment["training_tensor_environment_status"]}), flush=True)
        rows, events, wall = run_workload(output, command, manifest["sample_interval_s"])
        child = ROOT / "runs" / child_id
        child_seal = verify_seal(child)
        summary = load_json(child / "pilot_summary.json")
        reference_summary = load_json(reference / "pilot_summary.json")
        ram = summarize_samples(rows)
        if ram["access_denied_observations"] or not ram["peak_tree_rss_bytes"]:
            raise EvidenceError("Có tiến trình không đo được RAM; số đo không đầy đủ")
        timing = summarize_phases(events, summary["episodes"], wall)
        disk = disk_inventory(child)
        reference_disk = disk_inventory(reference)
        train, _ = train_geometries(ROOT, "results/audits/evaluation_protocol/geometry_split_manifest.csv")
        projections = project_costs(summary["episodes"], timing, disk, list(train.values()),
                                   manifest["projection_environment_seed_counts"])
        if reference_seal["files"] != fingerprints(reference):
            raise EvidenceError("Dữ liệu pilot gốc thay đổi trong lúc đo")
        if sources != source_hashes(ROOT, SOURCE_DIRS, EXTRA_FILES):
            raise EvidenceError("Nguồn/cấu hình thay đổi trong lúc đo")
        paired = []
        for actual, prior in zip(summary["episodes"], reference_summary["episodes"]):
            paired.append({"case_id": actual["case"]["case_id"], "steps": actual["steps"],
                           "terminal_reason": actual["terminal_reason"],
                           "reference_case_outcome_steps_match": actual["case"] == prior["case"] and
                               actual["steps"] == prior["steps"] and actual["terminal_reason"] == prior["terminal_reason"]})
        result = {"schema_version": "stage4-resource-result/1.0.0", "run_id": run_id,
                  "workload_run_id": child_id, "workload_integrity_sha256": sha256(child / "run_integrity.json"),
                  "verified_workload_files": len(child_seal["files"]), "workload_wall_seconds": wall,
                  "workload_collector_reported_seconds": summary["collection_and_verification_wall_seconds"],
                  "sample_interval_requested_s": manifest["sample_interval_s"], "ram": ram, "timing": timing,
                  "disk": disk, "reference_disk": reference_disk, "reference_pilot_unchanged": True,
                  "workload_cases": paired, "raw_steps": summary["raw_sample_count"], "outcomes": summary["outcomes"],
                  "projections": projections, "environment_check": environment, "hardware": hardware,
                  "source_files_unchanged": len(sources), "training_updates": 0,
                  "quality_status": "UNREVIEWED", "bc_accepted_samples": 0, "g1_complete": False}
        write_record(output / "resource_summary.json", result)
        state.update(status="COMPLETE", finished_at_utc=now(), training_environment_status=environment["training_tensor_environment_status"])
    except BaseException as error:
        state.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                     finished_at_utc=now(), error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_record(output / "run_state.json", state, replace=True)
        seal_attempt(output, state)
    return result


def publish(run_id, destination_name):
    validate_custom_run_id(run_id, ROOT / "runs")
    source = ROOT / "runs" / run_id
    seal = verify_seal(source)
    summary = load_json(source / "resource_summary.json")
    child = ROOT / "runs" / summary["workload_run_id"]
    verify_seal(child)
    if sha256(child / "run_integrity.json") != summary["workload_integrity_sha256"]:
        raise EvidenceError("Bằng chứng lượt đo không khớp")
    destination = project_path(ROOT, destination_name)
    allowed = ROOT / "results/audits/stage4/g1"
    if destination == allowed or not destination.is_relative_to(allowed):
        raise EvidenceError("Chỉ xuất vào thư mục mới dưới results/audits/stage4/g1/")
    destination.mkdir(parents=True, exist_ok=False)
    clean = {**summary, "environment_check": {k: v for k, v in summary["environment_check"].items() if k != "executable"}}
    write_record(destination / "resource_summary.json", clean)
    request = load_json(source / "request.json")
    write_record(destination / "measurement_definition.json", request["manifest"])
    write_record(destination / "source_inventory.json", request["source_sha256"])
    write_record(destination / "evidence_index.json", {"schema_version": "stage4-resource-evidence/1.0.0",
                 "published_at_utc": now(), "source_run_id": run_id,
                 "source_integrity_sha256": sha256(source / "run_integrity.json"), "verified_raw_files": len(seal["files"]),
                 "workload_run_id": summary["workload_run_id"], "workload_integrity_sha256": summary["workload_integrity_sha256"],
                 "training_environment_status": summary["environment_check"]["training_tensor_environment_status"],
                 "training_updates": 0, "g1_complete": False, "files": fingerprints(destination),
                 "raw_required": "Raw process samples, console logs, dependency checks, snapshots and full measured collection remain under runs/<run_id> and runs/<workload_run_id>."})
    return {"published": destination.relative_to(ROOT).as_posix()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("profile")
    run.add_argument("--run-id", required=True)
    export = sub.add_parser("publish")
    export.add_argument("--run-id", required=True)
    export.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = profile(args.run_id) if args.command == "profile" else publish(args.run_id, args.output)
        print(json.dumps({k: result[k] for k in result if k in ("run_id", "workload_wall_seconds", "raw_steps", "outcomes", "published")}))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(json.dumps({"error_type": type(error).__name__, "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
