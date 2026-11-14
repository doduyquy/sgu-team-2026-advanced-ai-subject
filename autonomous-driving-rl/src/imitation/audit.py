"""Kiểm định G0 chỉ đọc các đầu vào công khai và bằng chứng TRAIN hiện có.

Không khởi tạo mô phỏng hoặc thực thi chính sách lịch sử. Nhãn chất lượng
vẫn chờ xem xét. COMPLETED của bản ghi cũ và COMPLETE của Gate 7 khác định dạng.
"""

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from scripts.replay_rule_based_trace import restore_input
from src.agents.rule_based_agent import Stage1RuleBasedAgent
from src.platform import agent_input as input_types
from src.platform.action_adapter import ContinuousBox2Adapter
from src.platform.agent import AgentDecision, AgentPublicEpisodeContext
from src.platform.protocol import canonical_csv_file_sha256, canonical_json_file_sha256


class EvidenceError(ValueError):
    "Bằng chứng không nhất quán, bị thiếu hoặc có đường dẫn không hợp lệ."


def project_path(root, relative):
    root = Path(root).resolve()
    relative = str(relative).replace("\\", "/")
    if not relative or ":" in relative or Path(relative).is_absolute():
        raise EvidenceError("Cần đường dẫn tương đối tính từ thư mục dự án")
    if ".." in Path(relative).parts:
        raise EvidenceError("Không cho phép truy cập ngược thư mục bằng dấu ..")
    result = (root / relative).resolve()
    if not result.is_relative_to(root) or result == root:
        raise EvidenceError("Đường dẫn đầu vào vượt ra ngoài thư mục dự án")
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value):
    raise EvidenceError(f"Hằng số JSON không hữu hạn: {value}")


def load_json(path):
    with Path(path).open(encoding="utf8") as stream:
        return json.load(stream, parse_constant=_reject_constant)


def source_hashes(root, directories, extra_files=()):
    paths = set()
    for name in directories:
        directory = project_path(root, name)
        if not directory.is_dir():
            raise EvidenceError(f"Thiếu thư mục mã nguồn: {name}")
        paths.update(p for p in directory.rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.suffix != ".pyc")
    paths.update(project_path(root, name) for name in extra_files)
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths)}


def audit_snapshot(root, relative):
    path = project_path(root, relative)
    record = load_json(path)
    files = []
    for name, expected in record["source_sha256"].items():
        source = project_path(path.parent, name)
        if not source.is_file():
            files.append({"path": name, "expected_sha256": expected, "status": "MISSING"})
            continue
        content = source.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        lf_hash = hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
        status = ("EXACT" if actual == expected else
                  "LF_CONVERSION_MATCH_ONLY" if lf_hash == expected else "MISMATCH")
        files.append({"path": name, "expected_sha256": expected,
                      "actual_sha256": actual, "lf_sha256": lf_hash, "status": status})
    return {"source": relative, "manifest_sha256": sha256(path),
            "version": record["version"], "files": files,
            "counts": dict(Counter(item["status"] for item in files)),
            "interpretation": "Tương đương sau khi đổi sang LF chỉ hỗ trợ chẩn đoán; kiểm tra mã băm byte gốc vẫn chưa đạt. Giữ nguyên bản lưu lịch sử và tách riêng bản chụp ứng viên hiện tại."}


def train_geometries(root, relative):
    with project_path(root, relative).open(encoding="utf8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    identities = [(r["sequence"], int(r["geometry_generation_seed"])) for r in rows]
    if len(set(identities)) != len(identities):
        raise EvidenceError("Trùng định danh hình học trong tệp phân chia tập chính thức")
    train = {key: row for key, row in zip(identities, rows) if row["split"] == "TRAIN"}
    return train, dict(Counter(row["split"] for row in rows))


def propose_partition(train):
    "Đề xuất chia tập theo hình học bản đồ; chưa tạo dữ liệu huấn luyện."
    families = defaultdict(list)
    for row in train.values():
        families[row["sequence"]].append(row)
    result = []
    for sequence, rows in sorted(families.items()):
        if len(rows) != 15:
            raise EvidenceError(f"Cần đúng 15 hình học TRAIN trong họ {sequence}")
        ranked = sorted(rows, key=lambda r: (
            hashlib.sha256(("stage4-fit-dev-v1|" + r["geometry_id"]).encode()).hexdigest(),
            r["geometry_id"]))
        for index, row in enumerate(ranked):
            result.append({"geometry_id": row["geometry_id"], "sequence": sequence,
                           "geometry_generation_seed": int(row["geometry_generation_seed"]),
                           "geometry_sha256": row["geometry_sha256"],
                           "partition": "DEV" if index < 3 else "FIT"})
    return {"status": "PROPOSED_NOT_APPROVED", "scope": "Chỉ các hình học bản đồ thuộc TRAIN",
            "ranking": "SHA256(UTF8('stage4-fit-dev-v1|' + geometry_id)), sau đó dùng geometry_id để phân xử khi trùng",
            "counts": dict(Counter(r["partition"] for r in result)), "geometries": result}


def validate_case(case, train):
    key = (case["sequence"], int(case["geometry_generation_seed"]))
    if case["split"] != "TRAIN" or key not in train:
        raise EvidenceError("Bằng chứng hành vi mẫu phải thuộc TRAIN theo tệp phân chia chính thức")
    geometry = train[key]
    if case["geometry_sha256"] != geometry["geometry_sha256"] or case["tier"] != geometry["tier"]:
        raise EvidenceError("Mã băm hình học hoặc mức khó của tình huống khác tệp kê khai chính thức")
    if case["environment_seed"] in (5101, 5102, 9101, 9102, 9103, 9104, 9105):
        raise EvidenceError("Bằng chứng hành vi mẫu dùng seed môi trường dành riêng cho đánh giá")


def validate_trace_row(row, case_id, step_index):
    "Kiểm tra hành động float32 thực tế gửi đến xe, có xét phép chuyển từ float Python."
    if type(row["step_index"]) is not int or row["case_id"] != case_id or row["step_index"] != step_index:
        raise EvidenceError("Sai tình huống hoặc các bước quyết định không liên tục")
    serialized = row["agent_input"]
    required = {"core_observation", "traffic_context", "task_context", "profile_id",
                "step_index", "ego_motion"}
    if set(serialized) != required or serialized["profile_id"] != "STATE_DECISION_V1":
        raise EvidenceError("Cấu trúc hoặc hồ sơ đầu vào công khai không đúng quy định")
    if type(serialized["step_index"]) is not int or serialized["step_index"] != step_index:
        raise EvidenceError("Chỉ số bước của đầu vào và quyết định không khớp")
    if not {"lane_context", "route_passage"}.issubset(serialized["task_context"]):
        raise EvidenceError("Thiếu trường ngữ cảnh nhiệm vụ của P4")
    if not {"motion_paths", "motion_paths_overflow_count"}.issubset(serialized["traffic_context"]):
        raise EvidenceError("Thiếu trường ngữ cảnh giao thông của P4")
    inp = restore_input(serialized, input_types)
    if inp.to_dict() != serialized:
        raise EvidenceError("Đầu vào công khai không khớp sau khi khôi phục và tuần tự hóa lại, kể cả trường lồng nhau")
    decision = AgentDecision(**row["decision"])
    expected = ContinuousBox2Adapter().to_canonical(decision.action_payload).to_numpy()
    canonical = row["canonical_action"]
    if np.shape(canonical) != (2,) or not np.array_equal(expected.astype(float), canonical):
        raise EvidenceError("Hành động chuẩn không khớp hành động của tác nhân hướng dẫn sau khi chuyển sang float32")
    return inp, decision


def _new_segment(start, behavior):
    return {"start_step": start, "end_step_exclusive": start,
            "expert_behavior": behavior, "review_status": "UNREVIEWED",
            "quality_class": None, "approved_for_bc": False,
            "speed_limit_reasons": Counter(), "observed_displacement_m": 0.0,
            "min_observed_speed_mps": None, "max_observed_speed_mps": None,
            "min_throttle_brake": 1.0, "max_throttle_brake": -1.0}


def audit_episode(directory, case, result):
    "Đọc lần lượt bằng chứng lái xe tự nhiên; chưa tự động chấp nhận dữ liệu mẫu."
    trace = Path(directory) / "decisions.jsonl.gz"
    count, events, behaviors, segments = 0, Counter(), Counter(), []
    max_rounding = 0.0
    with gzip.open(trace, "rt", encoding="utf8") as stream:
        for line in stream:
            row = json.loads(line, parse_constant=_reject_constant)
            inp, decision = validate_trace_row(row, case["case_id"], count)
            action = row["canonical_action"]
            max_rounding = max(max_rounding, max(abs(float(a) - float(b))
                               for a, b in zip(decision.action_payload, action)))
            diag = decision.diagnostics or {}
            behavior = str(diag.get("behavior_state", "UNSPECIFIED"))
            behaviors[behavior] += 1
            events["brake_lt_minus_0_1"] += int(action[1] < -0.1)
            events["turn_abs_gt_0_1"] += int(abs(action[0]) > 0.1)
            events["route_passage_present"] += int(inp.task_context.route_passage is not None)
            events["actors_present"] += int(inp.traffic_context.active_count > 0)
            events["motion_paths_present"] += int(bool(inp.traffic_context.motion_paths))
            # Các ngưỡng chỉ mô tả phân phối, chưa dùng để chấp nhận chất lượng.
            if result["terminal_reason"] == "TIMEOUT":
                if not segments or segments[-1]["expert_behavior"] != behavior:
                    segments.append(_new_segment(count, behavior))
                segment = segments[-1]
                segment["end_step_exclusive"] = count + 1
                segment["speed_limit_reasons"][str(diag.get("speed_limit_reason"))] += 1
                segment["min_throttle_brake"] = min(segment["min_throttle_brake"], action[1])
                segment["max_throttle_brake"] = max(segment["max_throttle_brake"], action[1])
                if inp.ego_motion is not None:
                    motion = inp.ego_motion
                    speed = math.hypot(motion.velocity_x_mps, motion.velocity_y_mps)
                    segment["observed_displacement_m"] += math.hypot(
                        motion.displacement_x_m, motion.displacement_y_m)
                    for name, fn in [("min_observed_speed_mps", min), ("max_observed_speed_mps", max)]:
                        segment[name] = speed if segment[name] is None else fn(segment[name], speed)
            count += 1
    if count == 0 or count != result["steps"] or count > case["horizon_steps"]:
        raise EvidenceError("Số bước trong vết quyết định không khớp kết quả lượt chạy hoặc giới hạn bước")
    if result["case_id"] != case["case_id"]:
        raise EvidenceError("Kết quả lượt chạy không khớp tình huống")
    if bool(result["clean_success"]) != (result["terminal_reason"] == "SUCCESS"):
        raise EvidenceError("Nhãn thành công và kết quả kết thúc không nhất quán")
    if result["clean_success"] and any(result.get(k) for k in ("crash", "out_of_road", "boundary_contact")):
        raise EvidenceError("Nhãn đến đích an toàn mâu thuẫn với sự kiện mất an toàn đã ghi")
    if result.get("technical_error"):
        raise EvidenceError("Lượt chạy có lỗi kỹ thuật")
    return {"case": case, "expert_version": result["agent_version"], "steps": count,
            "terminal_reason": result["terminal_reason"], "trace_sha256": sha256(trace),
            "result_sha256": sha256(Path(directory) / "result.json"),
            "environment_sha256": sha256(Path(directory) / "environment.json"),
            "recorded_safety": {k: result.get(k) for k in ("crash", "out_of_road", "boundary_contact")},
            "events": dict(events), "behaviors": dict(behaviors),
            "max_payload_float32_rounding_error": max_rounding,
            "trajectory_quality_status": "UNREVIEWED_NOT_A_DEMONSTRATION_DATASET",
            "timeout_review_segments": segments}


def verify_expert_sources(directory, recorded_hashes, expected_source_hashes):
    "Đối chiếu các phụ thuộc của tác nhân hướng dẫn với P4, tách các bản sao phụ do bộ ghi lưu kèm."
    bindings = {}
    required = {"policy.py", "agent_input.py", "action_adapter.py", "agent_experiment.py",
                "rule_based_traffic.py", "rule_based_memory.py", "rule_based_junction.py",
                "rule_based_lane_change.py"}
    matched = set()
    for name, expected in recorded_hashes.items():
        key = name.removeprefix("agents/")
        if key == "rule_based_agent.py":
            key = "policy.py"
        if key in expected_source_hashes:
            if expected != expected_source_hashes[key]:
                raise EvidenceError(f"Mã nguồn tác nhân hướng dẫn khác nguồn P4 đã đóng băng: {name}")
            matched.add(key)
            bindings[name] = {"role": "P4_REFERENCE_MATCH", "reference_key": key}
        elif name in {"agents/random_naive_agent.py", "agents/__init__.py"}:
            bindings[name] = {"role": "AUXILIARY_RECORDER_COPY_NOT_P4_REFERENCE"}
        else:
            raise EvidenceError(f"Mã nguồn chưa được phân loại trong hồ sơ lịch sử: {name}")
        if sha256(project_path(directory, name)) != expected:
            raise EvidenceError(f"Mã băm mã nguồn lịch sử không khớp: {name}")
    if not required.issubset(matched):
        raise EvidenceError("Bản chụp các phụ thuộc của tác nhân hướng dẫn lịch sử chưa đầy đủ")
    return bindings


def audit_legacy_run(root, relative, train, expected_source_hashes):
    run = project_path(root, relative)
    state = load_json(run / "run_state.json")
    if state["status"] != "COMPLETED" or state["technical_errors"] != 0:
        raise EvidenceError("Lượt chạy lịch sử chưa hoàn tất hoặc có lỗi kỹ thuật")
    resolved = load_json(run / "resolved_cases.json")
    if resolved["split"] != "TRAIN" or not resolved["noncanonical"]:
        raise EvidenceError("Cần bằng chứng phát triển TRAIN chưa thuộc đánh giá chuẩn")
    cases = resolved["cases"]
    if len({c["case_id"] for c in cases}) != len(cases):
        raise EvidenceError("Trùng tình huống trong lượt chạy lịch sử")
    for case in cases:
        validate_case(case, train)
    if state["expected_episodes"] != len(cases) or state["completed_episodes"] != len(cases):
        raise EvidenceError("Số lượt chạy con trong hồ sơ lịch sử chưa đầy đủ")
    variant = run / "current"
    provenance = load_json(variant / "agent_provenance.json")
    config_bytes = json.dumps(provenance["configuration"], sort_keys=True,
                              separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(config_bytes).hexdigest() != provenance["configuration_sha256"]:
        raise EvidenceError("Mã băm cấu hình tác nhân hướng dẫn không khớp")
    bindings = verify_expert_sources(variant / "sources", provenance["source_sha256"], expected_source_hashes)
    if provenance["descriptor"]["agent_version"] != "0.6.0":
        raise EvidenceError("Phiên bản tác nhân hướng dẫn lịch sử không đúng dự kiến")
    directories = sorted(variant.glob("episode_*"))
    if len(directories) != len(cases):
        raise EvidenceError("Thiếu thư mục lượt chạy con hoặc có thư mục ngoài dự kiến")
    episodes = []
    for directory, expected_case in zip(directories, cases):
        environment = load_json(directory / "environment.json")
        if environment["case"] != expected_case:
            raise EvidenceError("Môi trường không khớp tệp tình huống đã được xác định")
        episode = audit_episode(directory, expected_case, load_json(directory / "result.json"))
        if episode["expert_version"] != provenance["descriptor"]["agent_version"]:
            raise EvidenceError("Phiên bản tác nhân hướng dẫn trong lượt chạy và hồ sơ nguồn gốc không khớp")
        episode["source_directory"] = directory.relative_to(root).as_posix()
        episodes.append(episode)
    runtime = load_json(run / "runtime.json")
    if sha256(run / "runner.py") != runtime["runner_sha256"]:
        raise EvidenceError("Mã băm chương trình chạy lịch sử không khớp")
    return {"run_directory": relative, "status": state["status"],
            "runtime_sha256": sha256(run / "runtime.json"),
            "recorded_runtime": {k: runtime.get(k) for k in (
                "python", "packages", "git_head", "git_dirty", "metadrive_commit", "control_dt_s")},
            "state_sha256": sha256(run / "run_state.json"),
            "resolved_cases_sha256": sha256(run / "resolved_cases.json"),
            "provenance_sha256": sha256(variant / "agent_provenance.json"),
            "configuration_sha256": provenance["configuration_sha256"],
            "source_sha256": provenance["source_sha256"], "source_reference_bindings": bindings,
            "episodes": episodes}


def verify_gate7_run(root, relative):
    run = project_path(root, relative)
    state = load_json(run / "run_state.json")
    if state["status"] != "COMPLETE":
        raise EvidenceError("Lượt chạy Gate 7 chưa có trạng thái COMPLETE")
    integrity = load_json(run / "run_integrity.json")
    checks = {}
    for field, name, fn in [
        ("run_manifest_sha256", "run_manifest.json", canonical_json_file_sha256),
        ("run_state_sha256", "run_state.json", canonical_json_file_sha256),
        ("episodes_sha256", "episodes.csv", canonical_csv_file_sha256),
        ("summary_sha256", "summary.json", canonical_json_file_sha256),
        ("timing_sha256", "timing.csv", canonical_csv_file_sha256),
        ("wandb_sync_sha256", "wandb_sync.json", canonical_json_file_sha256),
    ]:
        checks[name] = fn(run / name) == integrity[field]
    failures = run / "technical_failures.jsonl"
    checks["technical_failures.jsonl"] = (
        sha256(failures) if failures.exists() else None) == integrity["technical_failures_sha256"]
    if not all(checks.values()):
        raise EvidenceError("Kiểm tra toàn vẹn theo nội dung của Gate 7 không đạt")
    with (run / "episodes.csv").open(encoding="utf8", newline="") as stream:
        episodes = list(csv.DictReader(stream))
    if len(episodes) != state["recorded_episode_count"] or len(episodes) != state["expected_episode_count"]:
        raise EvidenceError("Số lượt chạy con của Gate 7 không khớp")
    if any(row["split"] != "TRAIN" for row in episodes):
        raise EvidenceError("Cần bằng chứng kiểm tra nhanh thuộc TRAIN")
    return {"run_directory": relative, "status": state["status"], "checks": checks,
            "integrity_sha256": sha256(run / "run_integrity.json"),
            "episodes": episodes, "canonical_run": state["canonical_run"]}


def replay_historical_evidence(root, relative):
    evidence = load_json(project_path(root, relative))
    reports = []
    for entry in evidence["traces"]:
        trace = project_path(root, entry["source_trace"])
        if sha256(trace) != entry["source_trace_sha256"]:
            raise EvidenceError("Mã băm vết quyết định lịch sử dùng để phát lại không khớp")
        environment = load_json(trace.parent / "environment.json")
        result = load_json(trace.parent / "result.json")
        if environment["case"]["split"] != "TRAIN":
            raise EvidenceError("Chỉ được phát lại bằng chứng thuộc TRAIN")
        case_id = environment["case"]["case_id"]
        if "scenario" in result:
            case_id += "/" + result["scenario"]
        agent = Stage1RuleBasedAgent()
        agent.reset(AgentPublicEpisodeContext(horizon_steps=environment["case"]["horizon_steps"]))
        count = 0
        try:
            with gzip.open(trace, "rt", encoding="utf8") as stream:
                for line in stream:
                    row = json.loads(line, parse_constant=_reject_constant)
                    inp, expected = validate_trace_row(row, case_id, count)
                    decision = agent.act(inp)
                    if decision.to_dict() != expected.to_dict():
                        raise EvidenceError(f"Quyết định khi phát lại không khớp tại {entry['label']} bước {count}")
                    if ContinuousBox2Adapter().to_canonical(decision.action_payload).to_numpy().tolist() != row["canonical_action"]:
                        raise EvidenceError("Hành động khi phát lại không khớp")
                    count += 1
        finally:
            agent.close()
        if count != entry["steps"] or count != result["steps"]:
            raise EvidenceError("Phát lại bằng chứng lịch sử chưa đủ số bước")
        reports.append({"label": entry["label"], "source_trace": trace.relative_to(root).as_posix(),
                        "trace_sha256": sha256(trace), "steps": count,
                        "decision_mismatches": 0, "action_mismatches": 0})
    return {"scope": "Phát lại bằng chính sách hiện tại; các phép thử vật lý có kiểm soát được tách khỏi dữ liệu mẫu lái xe tự nhiên",
            "historical_evidence_sha256": sha256(project_path(root, relative)),
            "steps": sum(r["steps"] for r in reports), "traces": reports,
            "status": "PASS", "new_closed_loop_driving_score": False}
