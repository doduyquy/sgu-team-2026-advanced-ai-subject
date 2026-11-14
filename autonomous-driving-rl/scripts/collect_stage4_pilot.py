"""Thu thử Stage 1 trên TRAIN, giữ mọi kết quả và kiểm tra ghép bước; chưa tạo BC."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_stage4_telemetry import git, map_metadata, measurement
from scripts.stage4 import fingerprints, seal_attempt, write_record
from src.agents.rule_based_agent import Stage1RuleBasedAgent
from src.imitation.audit import (EvidenceError, load_json, project_path, sha256, source_hashes,
                                train_geometries, validate_case)
from src.imitation.pilot import (encode_runtime_config, observation_digest, verify_episode,
                                verify_prerequisite, verify_seal)
from src.launcher.cases import make_jsonable, load_locked_geometry_block_sequence
from src.launcher.executor import ExperimentExecutor, build_metadrive_case_config
from src.launcher.models import LaunchRequestV1, LauncherMode, validate_custom_run_id
from src.launcher.resolver import resolve_experiment_plan
from src.logging.agent_experiment import save_agent_provenance
from src.platform import WandbMode
from metadrive import MetaDriveEnv
import metadrive


PROTECTED = ["src/agents", "src/platform", "src/launcher", "src/logging", "configs/platform",
             "datasets/mapsuite_v1", "experiments/rule_based/versions", "results/audits/stage4/g0",
             "results/audits/stage4/g1/telemetry_v1"]
SOURCE_DIRS = ["src", "tests", "configs/platform", "datasets/mapsuite_v1"]
SOURCE_FILES = ["scripts/collect_stage4_pilot.py", "scripts/audit_stage4_telemetry.py",
                "scripts/stage4.py", "scripts/replay_rule_based_trace.py", "scripts/run_tests.py",
                "experiments/stage4/g1_pilot_v1.json", "experiments/stage4/plan_v1.json"]


def now():
    return datetime.now(timezone.utc).isoformat()


def resolve(spec, agent_seed):
    return resolve_experiment_plan(LaunchRequestV1(
        mode=LauncherMode.SANDBOX, agent_id="stage1_rule_based", tier=spec["tier"],
        sequence=spec["sequence"], geometry_generation_seed=spec["geometry_seed"],
        environment_seed=spec["environment_seed"], agent_seed=agent_seed,
        render_mode="OFF", wandb_mode=WandbMode.DISABLED))


def write_gzip_json(path, value):
    with gzip.open(path, "xt", encoding="utf8", compresslevel=3) as stream:
        json.dump(make_jsonable(value), stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def worker(directory):
    request = load_json(directory / "request.json")
    plan = resolve(request["spec"], request["agent_seed"])
    if plan.to_dict() != request["resolved_plan"]:
        raise EvidenceError("Kế hoạch tiến trình con khác kế hoạch đã đóng băng")
    case = plan.resolved_cases[0]
    if case.split != "TRAIN":
        raise EvidenceError("Tiến trình con chỉ được chạy TRAIN")
    state = {"status": "RUNNING", "started_at_utc": now(), "case_id": case.case_id,
             "interventions": False, "quality_status": "UNREVIEWED"}
    write_record(directory / "episode_state.json", state)
    count, events = 0, []
    start = time.perf_counter()
    try:
        blocks, _, _ = load_locked_geometry_block_sequence(case.to_dict())
        config = build_metadrive_case_config(case, blocks, render_mode="OFF")
        write_record(directory / "environment.json", {"case": case.to_dict(), "requested_config": config,
                     "control_dt_s": plan.control_dt_s, "interventions": False})
        with gzip.open(directory / "transitions.jsonl.gz", "xt", encoding="utf8", compresslevel=3) as stream:
            class RecordingEnv(MetaDriveEnv):
                def reset(self, *args, **kwargs):
                    obs, info = super().reset(*args, **kwargs)
                    self.previous_observation = obs
                    write_gzip_json(directory / "map.json.gz", map_metadata(self))
                    write_record(directory / "reset.json", make_jsonable({"observation": obs, "info": info,
                                 "effective_config_format": "json-with-python-references/1.0.0",
                                 "effective_config": encode_runtime_config(self.config)}))
                    return obs, info

                def step(self, action):
                    nonlocal count
                    pre = measurement(self.agent)
                    submitted = [float(x) for x in action]
                    digest = observation_digest(self.previous_observation)
                    try:
                        obs, reward, terminated, truncated, info = super().step(action)
                    except BaseException as error:
                        write_record(directory / "step_error.json", {"step_index": count, "case_id": case.case_id,
                                     "canonical_action": submitted, "pre_vehicle": pre,
                                     "pre_observation_sha256": digest, "error_type": type(error).__name__,
                                     "error": str(error)})
                        raise
                    row = {"case_id": case.case_id, "step_index": count, "next_step_index": count + 1,
                           "simulation_time_before_s": count * plan.control_dt_s,
                           "simulation_time_after_s": (count + 1) * plan.control_dt_s,
                           "pre_observation_sha256": digest, "canonical_action": submitted,
                           "pre_vehicle": pre, "post_vehicle": measurement(self.agent),
                           "post_observation": obs, "env_reward": float(reward),
                           "env_terminated": bool(terminated), "env_truncated": bool(truncated), "step_info": info}
                    stream.write(json.dumps(make_jsonable(row), ensure_ascii=False, allow_nan=False,
                                            separators=(",", ":")) + "\n")
                    count += 1
                    if count % 50 == 0:
                        stream.flush()
                    self.previous_observation = obs
                    return obs, reward, terminated, truncated, info

            # Only the environment constructor is replaced, in this isolated worker.
            # The wrapper observes the unchanged launcher's reset/step without changing inputs or actions.
            with patch("src.launcher.executor.MetaDriveEnv", RecordingEnv):
                report = ExperimentExecutor(plan, runs_root=directory / "launcher", custom_run_id="stage1",
                            event_callback=lambda event: events.append(event.to_dict())).execute()
        write_record(directory / "execution_report.json", report.to_dict())
        if report.status != "COMPLETE" or report.completed_episodes != 1:
            if report.status == "INTERRUPTED":
                raise KeyboardInterrupt("Tiến trình con đã bị ngắt")
            raise EvidenceError("Launcher chưa hoàn tất lượt chạy")
        state.update(status="COMPLETE", finished_at_utc=now())
    except BaseException as error:
        state.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                     finished_at_utc=now(), error_type=type(error).__name__, error=str(error))
        raise
    finally:
        state.update(completed_physics_steps=count, worker_wall_seconds=time.perf_counter() - start)
        write_record(directory / "events.json", events)
        write_record(directory / "episode_state.json", state, replace=True)


def collect(run_id):
    validate_custom_run_id(run_id, ROOT / "runs")
    output = ROOT / "runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    state = {"run_id": run_id, "status": "RUNNING", "started_at_utc": now(),
             "scope": "G1_STEP2_ONLY", "canonical_run": False, "quality_status": "UNREVIEWED",
             "bc_accepted_samples": 0, "completed_episodes": 0, "failed_episodes": 0}
    write_record(output / "run_state.json", state)
    started = time.perf_counter()
    try:
        manifest = load_json(ROOT / "experiments/stage4/g1_pilot_v1.json")
        prerequisite = verify_prerequisite(ROOT, manifest["prerequisite"])
        plans = [resolve(spec, manifest["agent_seed"]) for spec in manifest["cases"]]
        train, _ = train_geometries(ROOT, "results/audits/evaluation_protocol/geometry_split_manifest.csv")
        for plan in plans:
            validate_case(plan.resolved_cases[0].to_dict(), train)
        if len({p.resolved_cases[0].case_id for p in plans}) != len(plans):
            raise EvidenceError("Lặp tình huống trong định nghĩa lần thu")
        protected = source_hashes(ROOT, PROTECTED)
        sources = source_hashes(ROOT, SOURCE_DIRS, SOURCE_FILES)
        for name in sources:
            target = output / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write((ROOT / name).read_bytes())
        write_record(output / "protected_before.json", protected)
        simulator = Path(metadrive.__file__).resolve().parent.parent
        runtime = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                   "packages": {p: importlib.metadata.version(p) for p in ("numpy", "panda3d", "metadrive-simulator", "wandb")},
                   "metadrive_path": str(simulator), "metadrive_commit": git(simulator, "rev-parse", "HEAD"),
                   "metadrive_dirty": git(simulator, "status", "--porcelain"),
                   "project_commit": git(ROOT, "rev-parse", "HEAD"), "project_status": git(ROOT, "status", "--porcelain")}
        write_record(output / "runtime.json", runtime)
        if (runtime["packages"]["metadrive-simulator"] != "0.4.3" or runtime["metadrive_dirty"] or
                runtime["metadrive_commit"] != "85e5dadc6c7436d324348f6e3d8f8e680c06b4db"):
            raise EvidenceError("Môi trường MetaDrive không khớp phiên bản/commit/cây nguồn sạch")
        simulator_hashes = {}
        for path in sorted((simulator / "metadrive").rglob("*.py")):
            name = path.relative_to(simulator).as_posix()
            target = output / "simulator_source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(path.read_bytes())
            simulator_hashes[name] = sha256(target)
        expert_dir = output / "expert"
        expert_dir.mkdir()
        agent = Stage1RuleBasedAgent()
        try:
            expert = save_agent_provenance(expert_dir, agent)
        finally:
            agent.close()
        if expert["descriptor"]["agent_version"] != "0.6.1":
            raise EvidenceError("Sai phiên bản tác nhân hướng dẫn")
        write_record(output / "request.json", {"manifest": manifest, "manifest_sha256": sha256(ROOT / SOURCE_FILES[-2]),
                     "prerequisite": prerequisite, "source_sha256": sources, "simulator_source_sha256": simulator_hashes,
                     "command": sys.argv, "worker_isolation": "one subprocess per case, passive environment wrapper"})
        write_record(output / "resolved_plans.json", [p.to_dict() for p in plans])
        reports = []
        for index, (spec, plan) in enumerate(zip(manifest["cases"], plans)):
            episode_id = f"episode_{index:03d}"
            directory = output / "episodes" / episode_id
            directory.mkdir(parents=True)
            write_record(directory / "request.json", {"spec": spec, "agent_seed": manifest["agent_seed"],
                         "resolved_plan": plan.to_dict()})
            command = [sys.executable, "-B", "-X", "utf8", "scripts/collect_stage4_pilot.py", "_worker",
                       "--episode-dir", directory.relative_to(ROOT).as_posix()]
            write_record(directory / "worker_command.json", {"command": command})
            print(json.dumps({"phase": "collecting", "episode_id": episode_id,
                              "case_id": plan.resolved_cases[0].case_id}), flush=True)
            try:
                with (directory / "console.log").open("xb") as console:
                    process = subprocess.run(command, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"})
                if process.returncode == 130:
                    raise KeyboardInterrupt("Tiến trình con đã bị ngắt")
                if process.returncode:
                    raise EvidenceError(f"Tiến trình con thất bại, mã thoát {process.returncode}")
                result = verify_episode(ROOT, directory, plan.to_dict(), expert)
                result.update(episode_id=episode_id, status="COMPLETE", raw_directory=directory.relative_to(ROOT).as_posix())
                state["completed_episodes"] += 1
            except Exception as error:
                result = {"episode_id": episode_id, "case": plan.resolved_cases[0].to_dict(), "status": "FAILED",
                          "error_type": type(error).__name__, "error": str(error),
                          "raw_directory": directory.relative_to(ROOT).as_posix(), "quality_status": "UNREVIEWED",
                          "bc_accepted_samples": 0}
                state["failed_episodes"] += 1
            write_record(directory / "verification.json", result)
            reports.append(result)
            write_record(output / "run_state.json", state, replace=True)
            print(json.dumps({"phase": "verified", "episode_id": episode_id, "status": result["status"],
                              "steps": result.get("steps"), "outcome": result.get("terminal_reason")}), flush=True)
        after = source_hashes(ROOT, PROTECTED)
        write_record(output / "protected_after.json", after)
        if protected != after or sources != source_hashes(ROOT, SOURCE_DIRS, SOURCE_FILES):
            raise EvidenceError("Nguồn/cấu hình hoặc bằng chứng được bảo vệ đã thay đổi trong lúc thu")
        complete = [r for r in reports if r["status"] == "COMPLETE"]
        with gzip.open(output / "raw_sample_index.jsonl.gz", "xt", encoding="utf8", compresslevel=3) as stream:
            for result in complete:
                for step in range(result["steps"]):
                    stream.write(json.dumps({"run_id": run_id, "episode_id": result["episode_id"], "step_index": step,
                                 "case_id": result["case"]["case_id"], "trace_sha256": result["trace_sha256"],
                                 "quality_status": "UNREVIEWED", "bc_accepted": False}, separators=(",", ":")) + "\n")
        summary = {"schema_version": "stage4-pilot-result/1.0.0", "run_id": run_id, "scope": "G1_STEP2_ONLY",
                   "expected_episodes": len(plans), "completed_episodes": len(complete),
                   "failed_episodes": state["failed_episodes"], "raw_sample_count": sum(r["steps"] for r in complete),
                   "replay_steps": sum(r["replay_steps"] for r in complete),
                   "unique_geometries": len({r["case"]["geometry_sha256"] for r in reports}),
                   "outcomes": dict(Counter(r["terminal_reason"] for r in complete)), "episodes": reports,
                   "protected_files_unchanged": len(protected), "source_files_unchanged": len(sources),
                   "collection_and_verification_wall_seconds": time.perf_counter() - started,
                   "quality_status": "UNREVIEWED", "bc_accepted_samples": 0, "g1_complete": False,
                   "partition_status": "UNASSIGNED_PENDING_STEP4", "all_outcomes_preserved": True}
        write_record(output / "pilot_summary.json", summary)
        if len(complete) != len(plans):
            raise EvidenceError("Có lượt lỗi; giữ mọi hồ sơ, lần thu chưa được chứng nhận hoàn tất")
        state.update(status="COMPLETE", finished_at_utc=now())
    except BaseException as error:
        state.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                     finished_at_utc=now(), error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_record(output / "run_state.json", state, replace=True)
        seal_attempt(output, state)
    return summary


def publish(run_id, destination_name):
    validate_custom_run_id(run_id, ROOT / "runs")
    source = ROOT / "runs" / run_id
    seal = verify_seal(source)
    summary = load_json(source / "pilot_summary.json")
    if summary["failed_episodes"] or summary["completed_episodes"] != summary["expected_episodes"]:
        raise EvidenceError("Không xuất lần thu thiếu hoặc có lỗi như bộ dữ liệu thô đầy đủ")
    destination = project_path(ROOT, destination_name)
    allowed = ROOT / "results/audits/stage4/g1"
    if destination == allowed or not destination.is_relative_to(allowed):
        raise EvidenceError("Chỉ xuất vào thư mục mới dưới results/audits/stage4/g1/")
    destination.mkdir(parents=True, exist_ok=False)
    write_record(destination / "pilot_summary.json", summary)
    request = load_json(source / "request.json")
    write_record(destination / "collection_definition.json", request["manifest"])
    write_record(destination / "source_inventory.json", {"source_sha256": request["source_sha256"],
                 "simulator_source_sha256": request["simulator_source_sha256"]})
    expert = load_json(source / "expert/agent_provenance.json")
    write_record(destination / "expert_provenance.json", expert)
    runtime = load_json(source / "runtime.json")
    write_record(destination / "evidence_index.json", {"schema_version": "stage4-pilot-evidence/1.0.0",
                 "published_at_utc": now(), "source_run_id": run_id, "source_integrity_sha256": sha256(source / "run_integrity.json"),
                 "verified_raw_files": len(seal["files"]), "prerequisite": request["prerequisite"],
                 "runtime": {k: runtime[k] for k in ("python", "platform", "packages", "metadrive_commit", "project_commit")},
                 "raw_sample_index_sha256": sha256(source / "raw_sample_index.jsonl.gz"),
                 "quality_status": "UNREVIEWED", "bc_accepted_samples": 0, "g1_complete": False,
                 "raw_required": "Source snapshots, maps, observations/actions/transitions, simulator info and console logs remain under runs/<source_run_id>/; summaries alone are insufficient.",
                 "files": fingerprints(destination)})
    return {"published": destination.relative_to(ROOT).as_posix(), "raw_samples": summary["raw_sample_count"],
            "bc_accepted_samples": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("collect", help="Thu thử và kiểm tra dữ liệu thô, chưa duyệt BC")
    run.add_argument("--run-id", required=True)
    export = sub.add_parser("publish", help="Kiểm tra mã băm rồi xuất tổng hợp vào thư mục mới")
    export.add_argument("--run-id", required=True)
    export.add_argument("--output", required=True)
    child = sub.add_parser("_worker", help=argparse.SUPPRESS)
    child.add_argument("--episode-dir", required=True)
    args = parser.parse_args()
    try:
        if args.command == "collect":
            result = collect(args.run_id)
            print(json.dumps({k: result[k] for k in ("run_id", "completed_episodes", "raw_sample_count", "outcomes")}))
        elif args.command == "publish":
            print(json.dumps(publish(args.run_id, args.output)))
        else:
            directory = project_path(ROOT, args.episode_dir)
            if not directory.is_relative_to(ROOT / "runs"):
                raise EvidenceError("Tiến trình con chỉ ghi trong runs/")
            worker(directory)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(json.dumps({"error_type": type(error).__name__, "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
