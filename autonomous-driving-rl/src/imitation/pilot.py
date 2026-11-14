"""Kiểm tra tính toàn vẹn và ghép bước của dữ liệu thô; không duyệt chất lượng BC."""

from collections import Counter
import gzip
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path

import numpy as np

from src.imitation.audit import (EvidenceError, load_json, project_path, sha256,
                                validate_case, validate_trace_row, train_geometries,
                                verify_gate7_run)
from src.platform.action_adapter import ContinuousBox2Adapter
from src.platform.agent import AgentPublicEpisodeContext
from src.platform.episode import classify_episode_outcome
from src.platform.reward import RewardSpecV1


FLAGS = ("arrive_dest", "out_of_road", "crash_vehicle", "crash_object", "crash_building",
         "crash_human", "crash_sidewalk", "max_step")


def encode_runtime_config(value):
    """Giữ cấu trúc và tên import của lớp/hàm; không gọi chúng hoặc dùng repr tùy tiện."""
    if isinstance(value, type):
        return {"python_type": value.__module__ + ":" + value.__qualname__}
    if callable(value):
        return {"python_callable": value.__module__ + ":" + value.__qualname__}
    if hasattr(value, "get_dict"):
        return encode_runtime_config(value.get_dict())
    if isinstance(value, dict):
        return {str(k): encode_runtime_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_runtime_config(v) for v in value]
    if isinstance(value, np.ndarray):
        return encode_runtime_config(value.tolist())
    if isinstance(value, np.generic):
        return encode_runtime_config(value.item())
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("Không có mã hóa cấu hình cho " + type(value).__qualname__)


def observation_digest(observation):
    return hashlib.sha256(np.asarray(observation, dtype="<f4").tobytes()).hexdigest()


def configuration_digest(configuration):
    return hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def verify_seal(directory):
    directory = Path(directory)
    seal = load_json(directory / "run_integrity.json")
    actual = {p.relative_to(directory).as_posix(): sha256(p) for p in sorted(directory.rglob("*"))
              if p.is_file() and p != directory / "run_integrity.json"}
    if seal["status"] != "COMPLETE" or actual != seal["files"]:
        raise EvidenceError("Lần thu chưa hoàn tất hoặc danh mục toàn vẹn không khớp")
    return seal


def verify_prerequisite(root, index_name):
    index_path = project_path(root, index_name)
    index = load_json(index_path)
    if not index.get("step1_verified"):
        raise EvidenceError("Thiếu bằng chứng kiểm chứng vận tốc/dấu lái")
    for name, expected in index["files"].items():
        path = project_path(root, str(index_path.parent.relative_to(root) / name))
        if sha256(path) != expected:
            raise EvidenceError("Bằng chứng bước 1 có mã băm không khớp")
    sources = load_json(index_path.parent / "after_sources.json")["source_sha256"]
    for name, expected in sources.items():
        if name.startswith(("src/agents/", "src/platform/", "src/launcher/", "src/logging/", "configs/platform/")):
            if sha256(project_path(root, name)) != expected:
                raise EvidenceError("Nguồn tác nhân/nền tảng khác nguồn đã kiểm chứng ở bước 1: " + name)
    return {"index": index_name, "index_sha256": sha256(index_path), "step1_verified": True}


def check_pair(decision, transition, case, index, previous_observation):
    inp, _ = validate_trace_row(decision, case["case_id"], index)
    if (transition["case_id"] != case["case_id"] or type(transition["step_index"]) is not int
            or transition["step_index"] != index or type(transition["next_step_index"]) is not int
            or transition["next_step_index"] != index + 1):
        raise EvidenceError("Bản ghi chuyển bước sai tình huống hoặc lệch chỉ số")
    if transition["canonical_action"] != decision["canonical_action"]:
        raise EvidenceError("Hành động ghi trước bước khác hành động gửi vào simulator")
    features = inp.core_observation.features
    if (not np.array_equal(features, np.asarray(previous_observation, dtype=np.float32)) or
            transition["pre_observation_sha256"] != observation_digest(features)):
        raise EvidenceError("Quan sát công khai không đúng thời điểm trước hành động")
    next_obs = np.asarray(transition["post_observation"], dtype=np.float32)
    if next_obs.shape != (259,) or not np.all(np.isfinite(next_obs)):
        raise EvidenceError("Quan sát sau bước không hợp lệ")
    if (abs(transition["simulation_time_before_s"] - index * .1) > 1e-8 or
            abs(transition["simulation_time_after_s"] - (index + 1) * .1) > 1e-8):
        raise EvidenceError("Mốc thời gian mô phỏng không khớp điều khiển 10 Hz")
    post = transition["post_vehicle"]
    if abs(post["applied_steering"] - transition["canonical_action"][0]) > 1e-6:
        raise EvidenceError("Lệnh lái thực tế không khớp nhãn hành động")
    velocity = post["body_velocity_mps"]
    speed = 3.6 * math.hypot(*velocity)
    if not math.isfinite(speed) or not math.isfinite(post["speed_km_h"]) or abs(speed - post["speed_km_h"]) > 1e-5:
        raise EvidenceError("Vận tốc sau bước khác vận tốc vật lý")
    motion = inp.ego_motion
    pre_speed = 3.6 * math.hypot(*transition["pre_vehicle"]["body_velocity_mps"])
    if motion is None or abs(3.6 * math.hypot(motion.velocity_x_mps, motion.velocity_y_mps) - pre_speed) > 1e-5:
        raise EvidenceError("Vận tốc quan sát trước hành động không khớp")
    return inp, speed


def reject_nonfinite(value):
    raise EvidenceError("Hằng số JSON không hữu hạn trong vết: " + value)


def verify_episode(root, directory, plan, expert, replay=True):
    """Đọc tuần tự, kiểm tra cả SUCCESS/TIMEOUT/thất bại an toàn bằng cùng quy tắc."""
    from src.agents.rule_based_agent import Stage1RuleBasedAgent

    case = plan["resolved_cases"][0]
    train, _ = train_geometries(root, "results/audits/evaluation_protocol/geometry_split_manifest.csv")
    validate_case(case, train)
    launcher = directory / "launcher/stage1"
    gate7 = verify_gate7_run(root, launcher.relative_to(root).as_posix())
    if len(gate7["episodes"]) != 1 or gate7["canonical_run"]:
        raise EvidenceError("Cần đúng một lượt TRAIN không thuộc benchmark")
    episode = gate7["episodes"][0]
    provenance = load_json(launcher / "agent_provenance.json")
    if provenance != expert or provenance["configuration_sha256"] != configuration_digest(provenance["configuration"]):
        raise EvidenceError("Nguồn/cấu hình tác nhân không khớp bản chụp đầu lần thu")
    for name, expected in provenance["source_sha256"].items():
        if sha256(launcher / "sources" / name) != expected:
            raise EvidenceError("Bản chụp tác nhân bị thay đổi")
    reset = load_json(directory / "reset.json")
    previous = reset["observation"]
    environment = load_json(directory / "environment.json")
    with gzip.open(directory / "map.json.gz", "rt", encoding="utf8") as stream:
        map_data = json.load(stream)
    if (map_data["block_sequence"] != environment["requested_config"]["map_config"]["config"]
            or environment["case"] != case):
        raise EvidenceError("Bản đồ thực tế khác cấu hình hình học đã khóa")
    agent = Stage1RuleBasedAgent() if replay else None
    if agent:
        if agent.export_config() != expert["configuration"]:
            raise EvidenceError("Nguồn hiện tại không thể phát lại đúng cấu hình")
        agent.reset(AgentPublicEpisodeContext(horizon_steps=case["horizon_steps"],
                    action_adapter_id="continuous_box2_v1"), agent_seed=plan["agent_seed"])
    trace = launcher / "agent_decisions.jsonl.gz"
    transitions = directory / "transitions.jsonl.gz"
    flags, count, speeds, behaviors = {}, 0, [], Counter()
    last_route, max_route, episode_return, env_return = 0.0, 0.0, 0.0, 0.0
    outcome = None
    try:
        with gzip.open(trace, "rt", encoding="utf8") as decisions, gzip.open(transitions, "rt", encoding="utf8") as steps:
            for left, right in zip_longest(decisions, steps):
                if left is None or right is None or (outcome is not None and outcome.is_done):
                    raise EvidenceError("Thiếu bước, thừa bước hoặc còn bước sau kết thúc")
                decision = json.loads(left, parse_constant=reject_nonfinite)
                transition = json.loads(right, parse_constant=reject_nonfinite)
                inp, speed = check_pair(decision, transition, case, count, previous)
                if agent:
                    actual = agent.act(inp)
                    if (actual.to_dict() != decision["decision"] or
                            ContinuousBox2Adapter().to_canonical(actual.action_payload).to_numpy().tolist() != decision["canonical_action"]):
                        raise EvidenceError("Phát lại Stage 1 không khớp quyết định/hành động")
                info = transition["step_info"]
                for key in FLAGS:
                    if bool(info.get(key, False)):
                        flags[key] = True
                count += 1
                outcome = classify_episode_outcome(flags, transition["env_terminated"],
                           transition["env_truncated"] or count >= case["horizon_steps"])
                route = float(info.get("route_completion", 0.0))
                episode_return += RewardSpecV1().compute_step_reward(route - last_route, case["horizon_steps"], outcome).total_reward
                env_return += float(transition["env_reward"])
                last_route, max_route = route, max(max_route, route)
                speeds.append(speed)
                behaviors[str((decision["decision"].get("diagnostics") or {}).get("behavior_state", "UNSPECIFIED"))] += 1
                previous = transition["post_observation"]
    finally:
        if agent:
            agent.close()
    if not count or count != int(episode["episode_steps"]) or count > case["horizon_steps"] or not outcome.is_done:
        raise EvidenceError("Số bước hoặc trạng thái kết thúc không khớp")
    if outcome.primary_reason.value != episode["primary_terminal_reason"]:
        raise EvidenceError("Kết quả tính lại từ cờ từng bước khác kết quả launcher")
    if outcome.clean_success != (episode["clean_success"].lower() in ("true", "1")):
        raise EvidenceError("Cờ thành công sạch không khớp các cờ an toàn từng bước")
    events = load_json(directory / "events.json")
    if (sum(e["event_type"] == "EPISODE_STARTED" for e in events) != 1 or
            sum(e["event_type"] == "EPISODE_FINISHED" for e in events) != 1 or
            sum(e["event_type"] == "RUN_FINISHED" for e in events) != 1 or
            any(e["event_type"] in ("RUN_FAILED", "RUN_INTERRUPTED") for e in events)):
        raise EvidenceError("Thiếu sự kiện vòng đời hoặc có sự kiện lỗi")
    for event in events:
        if event["event_type"] == "EPISODE_PROGRESS":
            step = event["step_index"]
            if type(step) is not int or not (1 <= step <= count) or abs(event["speed_kmh"] - speeds[step - 1]) > 1e-5:
                raise EvidenceError("Vận tốc trong sự kiện tiến độ không khớp đúng bước")
    for name, value in (("mean_speed_kmh", sum(speeds) / count), ("max_speed_kmh", max(speeds)),
                        ("episode_return", episode_return), ("final_route_completion", last_route),
                        ("max_route_completion", max_route)):
        if abs(float(episode[name]) - value) > 1e-5:
            raise EvidenceError("Chỉ số tổng hợp không khớp bản ghi từng bước: " + name)
    return {"case": case, "steps": count, "terminal_reason": outcome.primary_reason.value,
            "clean_success": outcome.clean_success, "final_route_completion": last_route,
            "mean_speed_kmh": sum(speeds) / count, "max_speed_kmh": max(speeds),
            "platform_episode_return": episode_return, "simulator_episode_return": env_return,
            "behavior_steps": dict(behaviors), "pairing_passed": True, "replay_steps": count if replay else 0,
            "trace_sha256": sha256(trace), "transitions_sha256": sha256(transitions),
            "map_sha256": sha256(directory / "map.json.gz"), "launcher_integrity": gate7,
            "quality_status": "UNREVIEWED", "bc_accepted_samples": 0,
            "partition": "UNASSIGNED_PENDING_STEP4"}
