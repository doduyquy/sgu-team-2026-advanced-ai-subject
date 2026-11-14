"Kiểm định G0 của Stage 4 và xuất bằng chứng đã xác minh. Chưa thu dữ liệu hoặc huấn luyện."

import argparse
from collections import Counter
from dataclasses import fields
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.imitation.audit import (
    EvidenceError, audit_legacy_run, audit_snapshot, load_json, project_path,
    propose_partition, replay_historical_evidence, sha256, source_hashes,
    train_geometries, verify_gate7_run,
)
from src.agents.rule_based_agent import Stage1RuleBasedAgent
from src.platform.action_adapter import ContinuousBox2Adapter
from src.launcher.cases import LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256
from src.launcher.models import LaunchRequestV1, validate_custom_run_id
from src.launcher.registry import build_canonical_agent_registry
from src.logging.agent_experiment import save_agent_provenance
from src.platform.experiment_logging import (
    capture_environment_provenance, capture_git_provenance, sanitize_error_message,
)
from src.platform.protocol import canonical_csv_file_sha256


class VietnameseArgumentParser(argparse.ArgumentParser):
    """Việt hóa cách trình bày trợ giúp, giữ cú pháp và hành vi phân tích lệnh."""

    def format_help(self):
        text = super().format_help()
        for original, translated in [
            ("usage: ", "Cách dùng: "),
            ("positional arguments:", "Lệnh và đối số:"),
            ("optional arguments:", "Tùy chọn:"),
            ("options:", "Tùy chọn:"),
            ("show this help message and exit", "Hiển thị hướng dẫn này rồi thoát"),
        ]:
            text = text.replace(original, translated)
        return text

    def format_usage(self):
        return super().format_usage().replace("usage: ", "Cách dùng: ", 1)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_record(path, payload, replace=False):
    "Ghi bản ghi cục bộ theo cách nguyên tử; chỉ trạng thái của lần thử được cập nhật."
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError(path.name)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def fingerprints(directory):
    return {p.relative_to(directory).as_posix(): sha256(p)
            for p in sorted(directory.rglob("*")) if p.is_file() and p != directory / "run_integrity.json"}


def seal_attempt(output, state):
    "Chuyển khỏi trạng thái hoàn tất khi phát hiện lỗi ghi hồ sơ toàn vẹn."
    try:
        write_record(output / "run_integrity.json", {"run_id": state["run_id"], "status": state["status"],
                                                     "finalized_at_utc": now(), "files": fingerprints(output)})
    except BaseException as error:
        state.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                     finished_at_utc=now(), integrity_failure=sanitize_error_message(error))
        # Ổ đĩa đầy hoặc không truy cập được cũng có thể ngăn ghi trạng thái lỗi.
        # Lỗi ban đầu vẫn làm lần chạy thất bại; xuất báo cáo phải có hồ sơ toàn vẹn.
        try:
            write_record(output / "run_state.json", state, replace=True)
            if not (output / "failure.json").exists():
                write_record(output / "failure.json", {"status": state["status"], "at_utc": now(),
                                                       "type": "INTEGRITY_FINALIZATION_ERROR",
                                                       "message": state["integrity_failure"]})
        except OSError:
            pass
        raise


def integration_inventory():
    registry = build_canonical_agent_registry()
    registrations = [r.to_dict() for r in registry.list_all()
                     if r.agent_id in ("stage0_random_naive", "stage1_rule_based")]
    executor = (ROOT / "src/launcher/executor.py").read_text(encoding="utf8")
    package = importlib.util.find_spec("metadrive")
    installed_source = None
    if package and package.submodule_search_locations:
        installed_source = Path(next(iter(package.submodule_search_locations))) / "base_class/base_object.py"
    speed_source = installed_source.read_text(encoding="utf8") if installed_source and installed_source.is_file() else ""
    request_fields = [field.name for field in fields(LaunchRequestV1)]
    return {
        "registrations": registrations,
        "agent_contract_metadata": load_json(ROOT / "configs/platform/agent_contract_v1.json")["metadata"],
        "launch_request_fields": request_fields,
        "checkpoint_binding_in_launch_request": any("checkpoint" in f or "bundle" in f for f in request_fields),
        "trace_opt_in": "Bộ thực thi bật DecisionTrace khi tác nhân cung cấp phương thức export_config có thể gọi",
        "stage0_exports_configuration": callable(getattr(registry.get_factory("stage0_random_naive")(), "export_config", None)),
        "stage1_exports_configuration": callable(getattr(Stage1RuleBasedAgent(), "export_config", None)),
        "speed_telemetry": {
            "executor_uses_speed_kmh": "env.agent.speed_kmh" in executor,
            "installed_metadrive_defines_speed_km_h": "def speed_km_h(" in speed_source,
            "installed_source_sha256": sha256(installed_source) if speed_source else None,
            "status": "REQUIRES_FIX_AND_TRAIN_VERIFICATION",
            "historical_logs_edited": False,
        },
        "adapter_action_description": ContinuousBox2Adapter().action_space_description,
        "steering_note": "Mô tả bộ chuyển đổi ghi số dương là rẽ phải; mã nguồn và kiểm thử Stage 0/1 ghi số dương là rẽ trái. Giữ nguyên nhãn số của hành động chuẩn và xác minh ý nghĩa trước khi thu dữ liệu.",
        "missing_trace_fields": ["quan sát sau bước mô phỏng", "tiến độ dọc tuyến do bộ đánh giá đo ở từng bước",
                                 "cờ an toàn môi trường ở từng bước", "xác nhận riêng rằng bước điều khiển đã được thực thi"],
        "missing_fields_interpretation": "BC không yêu cầu dữ liệu chuyển trạng thái hoặc phần thưởng. Vì vết được ghi trước env.step, cần kiểm tra trạng thái hoàn tất và số dòng. Độ dịch chuyển quan sát được không phải tiến độ dọc tuyến hoặc bằng chứng đoạn chạy an toàn.",
    }


def run_audit(manifest_name, custom_run_id=None, skip_tests=False):
    run_id = custom_run_id or ("s4_g0_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:6])
    validate_custom_run_id(run_id, ROOT / "runs")
    output = ROOT / "runs" / run_id
    output.mkdir(exist_ok=False)
    state = {"run_id": run_id, "status": "INITIALIZING", "started_at_utc": now(),
             "scope": "Kiểm định S4-G0 ngoại tuyến; chưa có lượt lái nghiên cứu mới, chưa nghiệm thu dữ liệu hoặc huấn luyện",
             "canonical_run": False, "gate_approved": False}
    write_record(output / "run_state.json", state)
    try:
        manifest_path = project_path(ROOT, manifest_name)
        manifest = load_json(manifest_path)
        if manifest["schema_version"] != "stage4-g0-audit/1.0.0" or manifest["scope"] != "G0_ONLY":
            raise EvidenceError("Phiên bản tệp kê khai hoặc phạm vi kiểm định không được hỗ trợ")
        git = capture_git_provenance(ROOT)
        write_record(output / "request.json", {
            "run_id": run_id, "created_at_utc": state["started_at_utc"],
            "command": ["<audit-python>", "-B", "-X", "utf8", "scripts/stage4.py", "audit",
                        "--manifest", manifest_name, "--run-id", run_id] + (["--skip-tests"] if skip_tests else []),
            "manifest": manifest, "manifest_sha256": sha256(manifest_path),
            "git_provenance": git, "runtime": capture_environment_provenance(ROOT),
            "skip_tests": skip_tests,
        })
        if git["git_branch"] != manifest["expected_branch"]:
            raise EvidenceError("Kiểm định G0 phải chạy trên nhánh khoa đã khai báo")
        before = source_hashes(ROOT, manifest["protected_directories"])
        write_record(output / "protected_before.json", before)
        sources = source_hashes(ROOT, ["src", "tests", "configs/platform", "experiments/stage4"],
                                ["scripts/stage4.py", "scripts/run_tests.py", "scripts/replay_rule_based_trace.py"])
        source_output = output / "sources"
        source_output.mkdir()
        for name, expected in sources.items():
            target = source_output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(project_path(ROOT, name).read_bytes())
            if sha256(target) != expected:
                raise EvidenceError("Mã nguồn thay đổi trong lúc chụp bản kiểm định")
        write_record(output / "source_inventory.json", sources)
        state.update(status="RUNNING", updated_at_utc=now())
        write_record(output / "run_state.json", state, replace=True)

        geometry_path = project_path(ROOT, manifest["geometry_manifest"])
        if canonical_csv_file_sha256(geometry_path) != LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256:
            raise EvidenceError("Mã băm tệp phân chia hình học chính thức không khớp")
        train, split_counts = train_geometries(ROOT, manifest["geometry_manifest"])
        if split_counts != {"TRAIN": 180, "VALIDATION": 48, "TEST": 12}:
            raise EvidenceError("Số lượng hình học trong các tập không đúng dự kiến")
        write_record(output / "proposed_fit_dev.json", propose_partition(train))
        snapshot = audit_snapshot(ROOT, manifest["historical_snapshot"])
        write_record(output / "historical_snapshot_check.json", snapshot)
        candidate = output / "candidate_expert"
        candidate.mkdir()
        provenance = save_agent_provenance(candidate, Stage1RuleBasedAgent())
        historical_config = load_json(project_path(ROOT, manifest["historical_snapshot"]))["configuration_sha256"]
        if provenance["configuration_sha256"] != historical_config:
            raise EvidenceError("Cấu hình tác nhân hướng dẫn hiện tại khác bản phát hành bảo trì")
        write_record(candidate / "candidate_status.json", {
            "candidate_id": run_id + ":stage1_rule_based@0.6.1", "created_at_utc": now(),
            "status": "LIVE_CHECKOUT_CANDIDATE_NOT_RELEASE_OR_COLLECTION_APPROVAL",
            "provenance_sha256": sha256(candidate / "agent_provenance.json"),
            "historical_archive_modified": False, "configuration_matches_maintenance_release": True,
        })
        reference = load_json(project_path(ROOT, manifest["expert_source_reference"]))["source_sha256"]
        runs = []
        for relative in manifest["natural_train_runs"]:
            print(json.dumps({"phase": "audit_natural_train", "source": relative}), flush=True)
            runs.append(audit_legacy_run(ROOT, relative, train, reference))
        episodes = [episode for run in runs for episode in run["episodes"]]
        if len({e["case"]["case_id"] for e in episodes}) != len(episodes):
            raise EvidenceError("Trùng tình huống lái xe tự nhiên giữa các lượt kiểm định")
        steps = sum(e["steps"] for e in episodes)
        if len(episodes) != manifest["expected_natural_cases"] or steps != manifest["expected_natural_steps"]:
            raise EvidenceError("Phạm vi bằng chứng lái xe tự nhiên khác định nghĩa G0 đã cố định")
        behavior, events = Counter(), Counter()
        review = []
        for episode in episodes:
            behavior.update(episode["behaviors"])
            events.update(episode["events"])
            for segment in episode.pop("timeout_review_segments"):
                review.append({"case_id": episode["case"]["case_id"],
                               "source_trace": episode["source_directory"] + "/decisions.jsonl.gz",
                               "trace_sha256": episode["trace_sha256"], "expert_version": episode["expert_version"],
                               **segment})
        write_record(output / "natural_train_evidence.json", {"runs": runs})
        write_record(output / "quality_review_queue.json", {
            "status": "UNREVIEWED_NO_SEGMENTS_APPROVED", "segments": review,
            "boundary_rule": "Các nhãn hành vi liên tiếp chỉ hỗ trợ tra cứu vết; chưa xác định ranh giới chất lượng",
            "quality_thresholds": None, "sampling_ratios": None, "sample_weights": None,
            "caution": "Thông tin chẩn đoán của tác nhân hướng dẫn cần được đối chiếu; chưa tự chứng minh chất lượng. Độ dịch chuyển không phải tiến độ dọc tuyến. Mỗi đoạn ứng viên cần được xem cùng đầu vào công khai trước và sau đoạn.",
        })
        write_record(output / "integration_inventory.json", integration_inventory())
        smoke = verify_gate7_run(ROOT, manifest["gate7_smoke_run"])
        write_record(output / "gate7_smoke_integrity.json", smoke)
        replay = replay_historical_evidence(ROOT, manifest["replay_evidence"])
        if replay["steps"] != manifest["expected_replay_steps"]:
            raise EvidenceError("Phạm vi phát lại khác định nghĩa G0")
        write_record(output / "live_replay.json", replay)
        if skip_tests:
            tests = {"status": "NOT_RUN", "reason": "Đã chọn --skip-tests; bằng chứng chưa đủ để nghiệm thu kiểm thử hồi quy"}
        else:
            print(json.dumps({"phase": "isolated_backend_regressions"}), flush=True)
            command = [sys.executable, "-B", "-X", "utf8", "scripts/run_tests.py",
                       "--group", "backend", "--output-dir", str(output / "regressions")]
            with (output / "regression_runner.log").open("xb") as stream:
                completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"})
            tests = load_json(output / "regressions/summary.json")
            if completed.returncode or tests["status"] != "PASSED":
                raise EvidenceError("Kiểm thử hồi quy phần lõi không đạt; xem nhật ký đã lưu")
        write_record(output / "tests.json", tests)
        after = source_hashes(ROOT, manifest["protected_directories"])
        if before != after:
            raise EvidenceError("Mã tác nhân, nền tảng hoặc bằng chứng lịch sử được bảo vệ đã thay đổi trong lúc kiểm định")
        if sources != source_hashes(ROOT, ["src", "tests", "configs/platform", "experiments/stage4"],
                                    ["scripts/stage4.py", "scripts/run_tests.py", "scripts/replay_rule_based_trace.py"]):
            raise EvidenceError("Mã nguồn hoặc cấu hình kiểm định thay đổi trong lúc chạy")
        write_record(output / "protected_after.json", after)
        summary = {
            "run_id": run_id, "audit_execution_status": "COMPLETE", "completed_at_utc": now(),
            "gate_status": "AWAITING_REVIEW", "gate_approved": False, "canonical_run": False,
            "new_driving_episodes": 0, "training_updates": 0, "accepted_bc_segments": 0,
            "expert_candidate": provenance["descriptor"], "historical_expert_version": "0.6.0",
            "natural_cases": len(episodes), "natural_steps": steps,
            "unique_geometries": len({(e["case"]["sequence"], e["case"]["geometry_generation_seed"]) for e in episodes}),
            "outcomes": dict(Counter(e["terminal_reason"] for e in episodes)),
            "cases_by_tier": dict(Counter(e["case"]["tier"] for e in episodes)),
            "steps_by_behavior": dict(behavior), "descriptive_events": dict(events),
            "quality_review_segment_count": len(review), "snapshot_byte_check": snapshot["counts"],
            "replay_steps": replay["steps"], "tests_status": tests["status"],
            "tests_passed": tests.get("total_tests"), "protected_files_unchanged": len(before),
            "open_findings": ["Cần xem xét mã băm byte gốc không khớp trong bản chụp lịch sử; tương đương LF chưa phải chứng nhận",
                              "Ngưỡng chất lượng và toàn bộ nhãn đoạn TIMEOUT vẫn chưa được phê duyệt",
                              "Phạm vi bằng chứng của tác nhân hướng dẫn còn hạn chế, gồm ba tình huống Extreme hết thời gian",
                              "Tên trường đo vận tốc không khớp; cần sửa và kiểm chứng riêng ở bước sau",
                              "Mô tả dấu góc lái mâu thuẫn; giữ nguyên hành động số",
                              "Chưa triển khai ràng buộc bản lưu mô hình với lần chạy",
                              "Kiểm định AgentContract 1.3 và điều kiện tham gia đánh giá chuẩn còn chờ",
                              "Cần chạy thử quy mô nhỏ để xác định môi trường PyTorch và ngân sách tính toán trước huấn luyện"],
        }
        write_record(output / "audit_summary.json", summary)
        state.update(status="COMPLETE", finished_at_utc=now(), gate_status="AWAITING_REVIEW")
        write_record(output / "run_state.json", state, replace=True)
    except BaseException as error:
        state.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                     finished_at_utc=now(), error_type=type(error).__name__,
                     failure_message=sanitize_error_message(error))
        write_record(output / "run_state.json", state, replace=True)
        write_record(output / "failure.json", {"status": state["status"], "at_utc": now(),
                                              "type": type(error).__name__, "message": state["failure_message"]})
        raise
    finally:
        seal_attempt(output, state)
    return summary


def publish_audit(run_id, output_name):
    validate_custom_run_id(run_id, ROOT / "runs")
    source = project_path(ROOT, "runs/" + run_id)
    integrity = load_json(source / "run_integrity.json")
    if integrity["run_id"] != run_id or integrity["status"] != "COMPLETE":
        raise EvidenceError("Chỉ được xuất bằng chứng của lần kiểm định đã hoàn tất đầy đủ")
    if fingerprints(source) != integrity["files"]:
        raise EvidenceError("Bằng chứng kiểm định có mã băm không khớp")
    summary = load_json(source / "audit_summary.json")
    if summary["tests_status"] != "PASSED":
        raise EvidenceError("Chỉ xuất báo cáo khi có bằng chứng kiểm thử hồi quy")
    destination = project_path(ROOT, output_name)
    allowed = project_path(ROOT, "results/audits/stage4/g0")
    if not destination.is_relative_to(allowed) or destination == allowed:
        raise EvidenceError("Xuất vào thư mục mới results/audits/stage4/g0/<attempt>")
    destination.mkdir(parents=True, exist_ok=False)
    names = ["audit_summary.json", "historical_snapshot_check.json", "natural_train_evidence.json",
             "quality_review_queue.json", "integration_inventory.json", "gate7_smoke_integrity.json",
             "live_replay.json", "proposed_fit_dev.json", "source_inventory.json"]
    for name in names:
        with (destination / name).open("xb") as stream:
            stream.write((source / name).read_bytes())
    request = load_json(source / "request.json")
    write_record(destination / "evidence_index.json", {
        "source_run_id": run_id, "source_run_integrity_sha256": sha256(source / "run_integrity.json"),
        "published_at_utc": now(), "gate_status": "AWAITING_REVIEW", "gate_approved": False,
        "git_provenance": request["git_provenance"], "runtime": request["runtime"],
        "manifest_sha256": request["manifest_sha256"], "command": request["command"],
        "test_summary": {k: load_json(source / "tests.json").get(k) for k in ("status", "total_tests")},
        "raw_test_summary_sha256": sha256(source / "tests.json"),
        "candidate_expert_provenance": load_json(source / "candidate_expert/agent_provenance.json"),
        "files": {name: sha256(destination / name) for name in names},
        "raw_payloads": "Bản chụp mã nguồn và nhật ký kiểm thử gốc nằm trong runs/<source_run_id>; vết lịch sử giữ tại đường dẫn gốc. Bản xuất này chỉ cung cấp báo cáo tổng hợp.",
    })
    return {"published": destination.relative_to(ROOT).as_posix(), "gate_status": "AWAITING_REVIEW"}


def main(argv=None):
    parser = VietnameseArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="Kiểm tra bằng chứng TRAIN hiện có, không thu dữ liệu hoặc huấn luyện")
    audit.add_argument("--manifest", default="experiments/stage4/g0_audit_v1.json", metavar="TEP_KE_KHAI",
                       help="Tệp định nghĩa kiểm định; mặc định dùng kế hoạch G0 hiện có")
    audit.add_argument("--run-id", metavar="MA_LAN_CHAY", help="Mã mới để lưu riêng hồ sơ lần thử")
    audit.add_argument("--skip-tests", action="store_true", help="Bỏ qua kiểm thử hồi quy; chưa đủ bằng chứng để xuất báo cáo")
    publish = sub.add_parser("publish-audit", help="Xác minh và xuất báo cáo G0 đã hoàn tất vào thư mục bằng chứng mới")
    publish.add_argument("--run-id", required=True, metavar="MA_LAN_CHAY",
                         help="Mã lần kiểm định đã hoàn tất cần xuất bằng chứng")
    publish.add_argument("--output", required=True, metavar="THU_MUC_DICH",
                         help="Thư mục bằng chứng mới dưới results/audits/stage4/g0/")
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            result = run_audit(args.manifest, args.run_id, args.skip_tests)
        else:
            result = publish_audit(args.run_id, args.output)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(json.dumps({"error_type": type(error).__name__, "message": sanitize_error_message(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
