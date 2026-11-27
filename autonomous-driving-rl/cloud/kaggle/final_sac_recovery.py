"""Trace missing frozen SAC outcomes without modifying the evaluator or policy input."""
from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import csv
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import shutil
from unittest.mock import patch

from cloud.kaggle import final_sac_evaluation as evaluation
from cloud.kaggle.common import read, sha, verify
from cloud.kaggle.final_sac_report import REPORT, collect, generate, save_csv
from src.launcher.models import ResolvedCaseV1
from src.platform.episode import REASON_TO_RAW_FLAG, classify_episode_outcome
from src.platform.task_context_producer import TaskContextLocalizationError


OUTPUT = evaluation.WORK / "final-sac-evaluation-recovery"
BASE = evaluation.OUTPUT
FLAGS = (*REASON_TO_RAW_FLAG.values(), "max_step")
MARKER = "## Recovery F09"


def lane_index(lane):
    return getattr(lane, "index", None)


def route_diagnostics(ego, road_network):
    """Inspect the same lane identity and route seams checked by F09."""
    import numpy as np

    from src.platform.reference_station import SEAM_TOLERANCE_M, _locate

    nav = ego.navigation
    checkpoints = getattr(nav, "checkpoints", None)
    target = getattr(nav, "_target_checkpoints_index", None)
    current = getattr(nav, "current_lane", None)
    graph = road_network.graph
    edges = list(zip(checkpoints, checkpoints[1:])) if checkpoints is not None else []
    groups = [graph.get(start, {}).get(end, ()) for start, end in edges]
    checkpoint = int(target[0]) if target is not None and len(target) else None
    nearby = [] if checkpoint is None else list(range(max(0, checkpoint - 1), min(len(groups), checkpoint + 2)))
    matches = [i for i, group in enumerate(groups) if any(lane is current for lane in group)]
    anchors = [i for i in nearby if i in matches]
    seams = [None if not prev or not nxt else float(np.linalg.norm(
        np.asarray(prev[0].position(prev[0].length, 0)) - np.asarray(nxt[0].position(0, 0))))
        for prev, nxt in zip(groups, groups[1:])]
    locate = _locate(nav, graph, ego.position[:2])
    if current is None or not edges or checkpoint is None:
        reason = "NAVIGATION_STATE_MISSING"
    elif not 0 <= checkpoint < len(edges):
        reason = "TARGET_CHECKPOINT_OUTSIDE_ROUTE"
    elif not anchors:
        reason = "ROUTE_LANE_OUTSIDE_WINDOW" if matches else "CURRENT_LANE_NOT_ON_ROUTE"
    elif len(anchors) != 1:
        reason = "AMBIGUOUS_CURRENT_LANE_ANCHOR"
    elif any(not group for group in groups):
        reason = "ROUTE_EDGE_MISSING"
    elif any(not math.isfinite(group[0].length) or group[0].length <= 0 for group in groups):
        reason = "INVALID_REFERENCE_LANE_LENGTH"
    elif any(seam is not None and not seam <= SEAM_TOLERANCE_M for seam in seams):
        reason = "REFERENCE_LANE_SEAM"
    elif locate is None:
        reason = "NONFINITE_REFERENCE_STATION"
    else:
        reason = "NO_VALID_WAYPOINT_AFTER_LOCALIZATION"
    references = getattr(nav, "current_ref_lanes", None)
    ref = references[0] if references else None
    ego_lane = getattr(ego, "lane", None)
    local_s, local_offset = ego_lane.local_coordinates(ego.position) if ego_lane is not None else (None, None)
    ref_s, ref_offset = ref.local_coordinates(ego.position) if ref is not None else (None, None)
    network_road = getattr(nav, "current_road", None)
    return {"f09_reason": reason, "checkpoint_index": checkpoint, "route_checkpoints": list(checkpoints or ()),
            "navigation_road": [network_road.start_node, network_road.end_node] if network_road is not None else None,
            "lane_index": ego.lane_index, "navigation_current_lane": lane_index(current),
            "current_reference_lane": lane_index(ref), "reference_candidates": [lane_index(x) for x in references or ()],
            "route_lane_matches": matches, "nearby_checkpoint_indices": nearby,
            "nearby_edges": [edges[i] for i in nearby], "reference_seam_distances_m": seams,
            "route_station_m": locate[1] if locate is not None else None, "ego_lane_station_m": local_s,
            "ego_lane_offset_m": local_offset, "current_reference_local_s_m": ref_s,
            "current_reference_offset_m": ref_offset,
            "ego_position": list(map(float, ego.position)), "ego_heading_rad": float(ego.heading_theta),
            "ego_speed_kmh": float(ego.speed_km_h), "left_boundary_m": ego.dist_to_left_side,
            "right_boundary_m": ego.dist_to_right_side}


def trace_case(case, condition, digest, *, parity=False):
    """Attach read-only diagnostics to the original evaluation loop in this worker."""
    from metadrive.envs.metadrive_env import MetaDriveEnv
    from src.launcher import executor

    original_input = executor.build_launcher_agent_input
    original_step = MetaDriveEnv.step
    last_steps = deque(maxlen=5)
    accumulated = {}
    decision_trace = {}
    observed_geometry = None

    def observe_step(env, action):
        result = original_step(env, action)
        _, _, terminated, truncated, info = result
        flags = {key: bool(info.get(key)) for key in FLAGS}
        accumulated.update({key: True for key, value in flags.items() if value})
        last_steps.append({"native_terminated": bool(terminated), "native_truncated": bool(truncated),
                           "native_flags": flags, "route_completion": float(info["route_completion"]),
                           "action_executed": list(map(float, action))})
        return result

    def observe_input(env, raw, decision, profile, producer):
        nonlocal observed_geometry
        if parity and observed_geometry is None:
            from src.launcher.cases import make_jsonable
            from src.platform.protocol import canonical_json_sha256

            observed_geometry = canonical_json_sha256(make_jsonable(env.current_map.get_meta_data()["block_sequence"]))
        try:
            return original_input(env, raw, decision, profile, producer)
        except TaskContextLocalizationError as error:
            last = last_steps[-1] if last_steps else {}
            result = classify_episode_outcome(accumulated, last.get("native_terminated", False),
                                              last.get("native_truncated", False) or decision >= case.horizon_steps)
            # CoreObservation exposes the same boundary measurements without requiring F09.
            from src.platform.agent_input import extract_core_observation
            core = extract_core_observation(raw)
            decision_trace.update({**last, **route_diagnostics(env.agent, env.current_map.road_network),
                                   "failure_step": decision, "simulation_time_s": decision / 10,
                                   "primary_reason_at_failure": result.primary_reason.value,
                                   "episode_done_at_failure": result.is_done, "cumulative_flags": accumulated.copy(),
                                   "core_observation_head": core.to_flat()[:9].tolist() if hasattr(core, "to_flat") else raw[:9].tolist(),
                                   "previous_steps": list(last_steps), "exception": str(error)})
            raise

    with patch.object(MetaDriveEnv, "step", observe_step), patch.object(executor, "build_launcher_agent_input", observe_input):
        try:
            outcome = evaluation.episode(case, condition, digest, parity=parity)
        except TaskContextLocalizationError:
            if not decision_trace:
                raise
            return {"diagnostic": decision_trace, "outcome": None}
    return {"diagnostic": None, "outcome": outcome, "runtime_canonical_geometry_sha256": observed_geometry}


def worker(case, condition, digest):
    with open(os.devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
        return trace_case(case, condition, digest)


def verify_base(manifest):
    for filename, digest in manifest["files"].items():
        verify(BASE / filename, digest)
    for filename, digest in manifest["episode_receipts_sha256"].items():
        verify(BASE / "episodes" / filename, digest)


def run():
    from cloud.kaggle.smooth_traffic_eval import memory, worker_budget

    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "episodes").mkdir(exist_ok=True)
    (OUTPUT / "diagnostics").mkdir(exist_ok=True)
    manifest_sha = sha(BASE / "manifest.json")
    baseline = read(BASE / "manifest.json")
    verify_base(baseline)
    plan, _, missing = collect(BASE)
    digest = evaluation.pin()["model_sha256"]
    if digest != plan["checkpoint"]["model_sha256"]:
        raise ValueError("Recovery checkpoint differs from the baseline")
    old_errors = [json.loads(line) for line in (BASE / "infrastructure.jsonl").read_text(encoding="utf-8").splitlines()]
    jobs = [(ResolvedCaseV1(**{k: v for k, v in row.items() if k not in ("effective_density", "traffic")}), row["traffic"])
            for row in sorted(missing, key=lambda row: (row["split"] != "TEST", row["protocol_order_index"]))]
    initial_memory = memory()
    workers = worker_budget(initial_memory)
    identities = [dict(case.to_dict(), traffic=traffic) for case, traffic in jobs]
    recovery_plan = {"original_manifest_sha256": manifest_sha, "original_runner_sha256": plan["runner_sha256"],
                     "model_sha256": digest, "canonical_run": False, "training_replicates": 1,
                     "outcomes_recoverable_from_original_logs": 0,
                     "original_logs_limitation": "Exception messages and stack traces without simulator outcome flags or final-route telemetry",
                     "cases": identities, "workers": workers, "start_memory": initial_memory}
    verify(Path(evaluation.__file__), plan["runner_sha256"])
    evaluation.atomic_json(OUTPUT / "recovery-plan.json", recovery_plan)
    save_csv(OUTPUT / "missing-cases.csv", identities)
    receipts, recovered, diagnostics = [], [], []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=evaluation.initialize) as pool:
        for split in ("TEST", "VALIDATION"):
            pending = {}
            for case, traffic in jobs:
                if case.split != split:
                    continue
                path = OUTPUT / "diagnostics" / evaluation.receipt_path(OUTPUT, case, traffic).name
                if path.exists():
                    receipt = read(path)
                    if any(receipt.get(key) != value for key, value in {**case.to_dict(), "traffic": traffic, "model_sha256": digest}.items()):
                        raise ValueError(f"Recovery receipt identity mismatch: {path}")
                    receipts.append(receipt)
                    if receipt["diagnostic"] is not None:
                        diagnostics.append({**case.to_dict(), "traffic": traffic, **receipt["diagnostic"]})
                    elif receipt["eligible_to_merge"]:
                        recovered.append(receipt["outcome"])
                    continue
                pending[pool.submit(worker, case, traffic, digest)] = case, traffic
            for future in as_completed(pending):
                case, traffic = pending[future]
                result = future.result()
                trace = result["diagnostic"]
                previous = [error["detail"] for error in old_errors if (error["case_id"], error["traffic"]) == (case.case_id, traffic)]
                old_step = int(re.search(r"decision (\d+)", previous[0]).group(1))
                if trace is not None:
                    state = "UNRESOLVED_ACTIVE_LOCALIZATION" if not trace["episode_done_at_failure"] else "TERMINAL_EVIDENCE_REVIEW_REQUIRED"
                    diagnostics.append({**case.to_dict(), "traffic": traffic, **trace})
                else:
                    state = "RECOVERED_WITH_ORIGINAL_RUNNER"
                    recovered.append(result["outcome"])
                receipt = {**case.to_dict(), "traffic": traffic, "model_sha256": digest, "status": state,
                           "original_outcome": None, "original_errors": previous, "original_failure_step": old_step,
                           "method": "Replay frozen episode with passive diagnostics using original evaluator",
                           "observation_action_modified": False, "trajectory_intervention": False,
                           "reproduced_original_error": trace is not None and all(error == trace["exception"] for error in previous),
                           "eligible_to_merge": result["outcome"] is not None, **result}
                path = evaluation.receipt_path(OUTPUT, case, traffic)
                evaluation.atomic_json(OUTPUT / "diagnostics" / path.name, receipt)
                receipts.append(receipt)
                print(split, case.case_id, state, trace.get("f09_reason") if trace else result["outcome"]["primary_reason"], flush=True)
    finalize(plan, baseline, recovery_plan, receipts, recovered, diagnostics, memory())


def finalize(plan, baseline, recovery_plan, receipts, recovered, diagnostics, end_memory):
    from cloud.kaggle.final_sac_report import table

    original = read(BASE / "parity-proof.json")
    regression = {"status": "PENDING", "producer_changed": False, "original_evaluator_changed": False}
    evaluation.initialize()
    # Match against the original TRAIN proof, including the full Box2 action traces.
    from cloud.kaggle.runtime import ScheduledEnv
    env = ScheduledEnv("eval", 0, 0, fixed=("geom_medium_SCTCS_seed11", 33701))
    case = env.sampler.get_case_for_geometry_id("geom_medium_SCTCS_seed11", environment_seed=33701)
    compared = ("action_trace_sha256", "primary_reason", "episode_steps", "simulation_time_s", "final_route_completion",
                "terminal_preview_steps", "effective_density", "geometry_sha256")
    for condition in evaluation.CONDITIONS:
        reference = original[f"train_{condition.lower()}"]
        traced = trace_case(case, condition, recovery_plan["model_sha256"], parity=True)
        checked = traced["outcome"]
        if checked is None:
            raise ValueError(f"TRAIN parity {condition} did not produce an outcome")
        differences = {field: [reference[field], checked[field]] for field in compared if checked[field] != reference[field]}
        if traced["runtime_canonical_geometry_sha256"] != case.geometry_sha256:
            differences["runtime_canonical_geometry_sha256"] = [case.geometry_sha256, traced["runtime_canonical_geometry_sha256"]]
        if differences:
            raise ValueError(f"TRAIN parity {condition} differs: {differences}")
        regression[condition] = {**{field: checked[field] for field in compared},
                                 "runtime_canonical_geometry_sha256": traced["runtime_canonical_geometry_sha256"],
                                 "original_geometry_repr_sha256": reference["runtime_geometry_identity"],
                                 "recovery_geometry_repr_sha256": checked["runtime_geometry_identity"]}
    regression["status"] = "PASS"
    evaluation.atomic_json(OUTPUT / "regression-proof.json", regression)
    verify_base(baseline)
    verify(BASE / "manifest.json", recovery_plan["original_manifest_sha256"])
    verify(evaluation.BUNDLE / "model.zip", recovery_plan["model_sha256"])
    verify(evaluation.V2 / "stage-05000/model.zip", recovery_plan["model_sha256"])
    for filename in ("execution-plan.json", "parity-proof.json", "infrastructure.jsonl"):
        shutil.copyfile(BASE / filename, OUTPUT / filename)
    for filename in baseline["episode_receipts_sha256"]:
        shutil.copyfile(BASE / "episodes" / filename, OUTPUT / "episodes" / filename)
    for row in recovered:
        filename = f"{row['split'].lower()}-{row['protocol_order_index']:03d}-{row['traffic'].lower()}.json"
        path = OUTPUT / "episodes" / filename
        if path.exists():
            raise ValueError(f"Recovery would replace an original episode: {filename}")
        evaluation.atomic_json(path, row)
    save_csv(OUTPUT / "recovered-episodes.csv", recovered)
    if not recovered:
        with (BASE / "episodes.csv").open(encoding="utf-8-sig", newline="") as source, \
                (OUTPUT / "recovered-episodes.csv").open("w", encoding="utf-8-sig", newline="") as target:
            csv.writer(target).writerow(next(csv.reader(source)))
    save_csv(OUTPUT / "localization-diagnostics.csv", diagnostics)
    with (OUTPUT / "recovery-receipts.jsonl").open("w", encoding="utf-8") as stream:
        for receipt in receipts:
            stream.write(json.dumps(receipt, ensure_ascii=False) + "\n")
    generate(OUTPUT, OUTPUT / "recovered-summary.md")
    shutil.copyfile(OUTPUT / "episodes.csv", OUTPUT / "merged-episodes.csv")
    integrity = read(OUTPUT / "integrity-check.json")
    integrity.update(original_manifest_sha256=recovery_plan["original_manifest_sha256"], original_outcomes_preserved=baseline["valid_episodes"],
                     recovered_outcomes=len(recovered), recovery_attempts=len(receipts), regression="PASS")
    evaluation.atomic_json(OUTPUT / "integrity-check.json", integrity)
    manifest = read(OUTPUT / "manifest.json")
    counts = Counter(row["f09_reason"] for row in diagnostics)
    manifest.update(original_manifest_sha256=recovery_plan["original_manifest_sha256"], recovery_workers=recovery_plan["workers"],
                    recovery_end_memory=end_memory, recovery_source_sha256=sha(Path(__file__)), localization_causes=dict(counts),
                    files={path.name: sha(path) for path in sorted(OUTPUT.iterdir()) if path.is_file() and path.name != "manifest.json"})
    evaluation.atomic_json(OUTPUT / "manifest.json", manifest)
    by_cause = Counter((row["split"], row["f09_reason"], row["primary_reason_at_failure"]) for row in diagnostics)
    lines = [MARKER, "", f"Đã chạy lại {len(receipts)} ca thiếu theo thứ tự TEST rồi VALIDATION. "
             f"Log ban đầu chỉ có exception và stack trace, nên phục hồi trực tiếp từ log được 0 ca. "
             f"Recovery có thêm {len(recovered)} outcome hợp lệ. Tổng cuối **{integrity['valid']}/312**.", "",
             "Runner gốc phân loại outcome ngay sau `env.step` và dừng vòng lặp trước lần tạo observation kế tiếp nếu đã terminal. "
             "Instrumentation chỉ đọc flags và trạng thái khi F09 báo lỗi. Outcome vẫn do evaluator chung quyết định.", "",
             table(["Split", "Điều kiện F09 thất bại", "Outcome tại bước lỗi", "Số ca"],
                   [(split, reason, outcome, n) for (split, reason, outcome), n in sorted(by_cause.items())]), "",
             "Đối chứng TRAIN Medium ON/OFF cho cùng SHA-256 chuỗi action Box2, primary reason, final route completion và simulation time. "
             "Đã đối chiếu SHA-256 của manifest gốc, 296 receipt gốc và model. Không đổi producer hay evaluator. 0 SAC updates và 0 hàng nạp replay.", "",
             "Chạy lại chẩn đoán bằng `python -m cloud.kaggle.final_sac_recovery`. Dữ liệu gốc ở `runs/kaggle/final-sac-evaluation/`. "
             "Trace, receipt và bảng recovery ở `runs/kaggle/final-sac-evaluation-recovery/`.", ""]
    original_report = REPORT.read_text(encoding="utf-8").split(MARKER, 1)[0]
    REPORT.write_text(original_report.rstrip() + "\n\n" + "\n".join(lines), encoding="utf-8")
    print("Recovery complete:", integrity["valid"], "valid episodes", dict(counts), flush=True)


if __name__ == "__main__":
    run()
