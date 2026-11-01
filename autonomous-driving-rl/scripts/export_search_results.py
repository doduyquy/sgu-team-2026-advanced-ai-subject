"""Export a compact, verified summary of the completed eight-run search campaign.

Raw runs, logs and machine-specific paths stay local. Primary metric values are
copied from Gate-7 summaries after validating every run and the full paired suites.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
METHODS = {"stage2_astar": "A*", "stage2_bfs": "BFS", "stage2_dijkstra": "Dijkstra", "stage2_hybrid_astar": "Hybrid A*"}
METRICS = ("clean_success_rate", "safety_failure_rate", "mean_final_route_completion",
           "median_final_route_completion", "mean_time_to_clean_success_s")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_campaign(campaign_path, runs_root, output):
    # Load Qt before simulator-linked platform imports on Windows.
    from PySide6.QtWidgets import QApplication
    from src.launcher.cases import resolve_validation_cases, resolve_test_cases
    from src.platform import PINNED_METADRIVE_COMMIT, PINNED_METADRIVE_VERSION
    from src.workbench.results_repository import IntegrityDisplayStatus, RunArtifactRepository

    campaign_path, runs_root, output = Path(campaign_path), Path(runs_root), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory; published evidence is never overwritten")
    state = json.loads(campaign_path.read_text(encoding="utf-8"))
    if state.get("status") != "COMPLETE" or state.get("completed_episodes") != 624 or len(state.get("jobs", [])) != 8:
        raise ValueError("The complete 624-episode, eight-run campaign is required")
    jobs = state["jobs"]
    if {(j["mode"], j["agent_id"]) for j in jobs} != {(mode, method) for mode in ("VALIDATION", "TEST") for method in METHODS}:
        raise ValueError("Campaign methods or splits differ from the fixed comparison")
    repository = RunArtifactRepository(runs_root)
    rows, common_identity, run_ids, snapshots = [], None, set(), []
    expected_cases = {"VALIDATION": resolve_validation_cases(ROOT), "TEST": resolve_test_cases(ROOT)}
    for job in jobs:
        if job.get("status") != "COMPLETE" or job["run_id"] in run_ids:
            raise ValueError("Incomplete or duplicate run in the cohort")
        run_ids.add(job["run_id"])
        snapshot = repository.load_run_snapshot(job["run_id"])
        if snapshot is None or snapshot.status != "COMPLETE" or snapshot.integrity_status != IntegrityDisplayStatus.VERIFIED:
            raise ValueError(f"Unverified run: {job['run_id']}")
        if (not snapshot.canonical_run or snapshot.git_worktree_dirty is not False or snapshot.dirty_override
                or snapshot.unverified_env_override or snapshot.environment_verification_status != "VERIFIED"
                or snapshot.metadrive_version != PINNED_METADRIVE_VERSION or snapshot.metadrive_commit != PINNED_METADRIVE_COMMIT):
            raise ValueError(f"Run provenance is not canonical: {job['run_id']}")
        if (snapshot.agent_id != job["agent_id"] or snapshot.agent_seed is not None
                or snapshot.git_commit_sha != state["frozen_commit"]
                or snapshot.input_profile_id != "STATE_DECISION_V1" or snapshot.action_adapter_id != "discrete9_lowbranch_v1"):
            raise ValueError("Agent, source or interface identity differs from the frozen campaign")
        cases = expected_cases[job["mode"]]
        if snapshot.recorded_episode_count != len(cases) or len(snapshot.episode_rows) != len(cases):
            raise ValueError("Not all fixed cases were recorded")
        for index, (episode, case) in enumerate(zip(snapshot.episode_rows, cases), 1):
            expected = dict(case_id=case.case_id, split=case.split, episode_index=index,
                            protocol_order_index=case.protocol_order_index, geometry_generation_seed=case.geometry_generation_seed,
                            geometry_sha256=case.geometry_sha256, environment_seed=case.environment_seed, horizon_steps=case.horizon_steps)
            # The Gate-7 CSV has no geometry_sha256 column; that identity is verified by the locked case manifests.
            expected.pop("geometry_sha256")
            if any(str(episode.get(key)) != str(value) for key, value in expected.items()):
                raise ValueError("Episode content/order differs from the locked evaluation manifest")
        if snapshot.summary_payload.get("technical_failure_count") != 0 or snapshot.technical_failures_payload:
            raise ValueError("Technical failures prevent cohort publication")
        identity = (snapshot.git_commit_sha, tuple(sorted(snapshot.platform_contracts.items())))
        if common_identity is not None and identity != common_identity:
            raise ValueError("Runs differ in source or frozen contracts")
        common_identity = identity
        run_dir = runs_root / snapshot.run_id
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        planner = manifest["config"]["algorithm_hyperparameters"]["planner"]
        rows.append(dict(split=job["mode"], agent_id=snapshot.agent_id, agent_version=snapshot.agent_version,
                         run_id=snapshot.run_id, episodes=len(cases), integrity="VERIFIED", canonical_run=True,
                         started_at_utc=snapshot.started_at_utc, finished_at_utc=snapshot.finished_at_utc,
                         metrics={key: snapshot.summary_payload["overall_metrics"][key] for key in METRICS},
                         planner=planner, evidence_sha256={name: sha256(run_dir/name) for name in (
                             "run_manifest.json", "run_state.json", "episodes.csv", "timing.csv", "summary.json", "run_integrity.json")}))
        snapshots.append(snapshot)
    last_validation = max(datetime.fromisoformat(r["finished_at_utc"]) for r in rows if r["split"] == "VALIDATION")
    first_test = min(datetime.fromisoformat(r["started_at_utc"]) for r in rows if r["split"] == "TEST")
    if first_test < last_validation:
        raise ValueError("TEST started before all VALIDATION runs completed")
    report = dict(format_version="search_evaluation_summary_v1", status="COMPLETE", total_episodes=624,
                  recorded_at_utc=state["finished_at_utc"], campaign_progress_sha256=sha256(campaign_path),
                  frozen_commit=state["frozen_commit"], frozen_tree=state["frozen_tree"],
                  input_profile_id="STATE_DECISION_V1", action_adapter_id="discrete9_lowbranch_v1",
                  platform_contracts=snapshots[0].platform_contracts,
                  metadrive=dict(version=PINNED_METADRIVE_VERSION, commit=PINNED_METADRIVE_COMMIT),
                  raw_artifacts_included=False, runs=rows)
    source_snapshot = campaign_path.parent / "snapshot.json"
    if source_snapshot.exists():
        frozen_project = Path(json.loads(source_snapshot.read_text(encoding="utf-8"))["frozen_project"])
        source_files = ("src/agents/graph_search.py", "src/agents/planning_graph.py", "src/agents/astar_agent.py",
                        "src/agents/graph_search_agents.py", "src/agents/hybrid_astar.py", "src/agents/hybrid_astar_agent.py",
                        "src/agents/path_following.py", "src/launcher/registry.py", "src/launcher/executor.py",
                        "configs/experiments/astar_learning_v1.json", "configs/experiments/search_planners_v1.json")
        report["frozen_source_sha256_lf"] = {name: hashlib.sha256((frozen_project/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                            for name in source_files}
    lines = ["# Kết quả đánh giá nhóm tìm kiếm Stage 2", "",
             "Đã hoàn tất 624 episode: bốn phương pháp × (96 VALIDATION + 60 TEST).",
             "Cả tám run COMPLETE / VERIFIED, không có lỗi kỹ thuật. TEST chỉ chạy sau khi toàn bộ VALIDATION hoàn tất; không chỉnh phương pháp giữa hai bộ.", "",
             "| Bộ | Phương pháp | Episode | Đến đích an toàn | Lỗi an toàn | Mean route | Median route |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        metrics = row["metrics"]
        percentages = " | ".join(f"{100 * metrics[key]:.2f}%" for key in METRICS[:4])
        lines.append(f"| {row['split']} | {METHODS[row['agent_id']]} | {row['episodes']} | {percentages} |")
    test_rows = [row for row in rows if row["split"] == "TEST"]
    if all(row["metrics"]["clean_success_rate"] == 0 for row in test_rows):
        lines += ["", "Cả bốn phương pháp đều chưa có lần đến đích an toàn trên TEST trong lượt đánh giá này."]
    lines += ["", "Các chênh lệch trong bảng mô tả bốn bản triển khai này, không chứng minh một thuật toán luôn tốt hơn thuật toán khác.", "",
              "## Phạm vi của kết quả", "",
              "Đây là kết quả của các bản triển khai cục bộ và bộ điều khiển hiện tại. A*/Dijkstra cho cùng metric; cần điều tra đồ thị, chọn đường và bộ bám đường trong nghiên cứu tiếp theo.",
              "Hybrid A* dùng không gian trạng thái và chi phí riêng. Mỗi phương pháp deterministic chỉ chạy một lượt trên mỗi bộ; không suy diễn ý nghĩa thống kê từ các lượt này.", "",
              f"Nguồn thực thi đóng băng: commit `{state['frozen_commit']}`, tree `{state['frozen_tree']}`.",
              "Commit chạy này nằm trên nhánh local `run/search-full-20261101`; không phải ref đã được công bố trên remote.",
              "Nhánh triển khai hiện tại tích hợp các cập nhật HUD mới hơn và vẫn giữ trạng thái candidate. Kết quả này không được trình bày như một lần chạy lại trên source mới.", "",
              "## Bằng chứng và kiểm tra lại", "",
              "[metrics.json](metrics.json) giữ metric lấy trực tiếp từ summary Gate-7, Run ID, cấu hình từng planner, contract hash, source hash và SHA-256 của các artifact đã xác minh.",
              "Raw runs/log và đường dẫn máy cá nhân không nằm trong bản công bố. Các bản gốc vẫn giữ tại `runs/<run_id>/` của workspace dùng chạy thí nghiệm.",
              "Từ `autonomous-driving-rl/`, dùng script xuất lại từ bản gốc vào thư mục trống:", "", "```powershell",
              "& '..\\.venv\\metadrive\\python.exe' -B scripts/export_search_results.py --campaign artifacts/search_full_20261101/progress.json --runs-root runs --output artifacts/search_report_check",
              "```", "", "Script kiểm tra đủ cả tám run, integrity, source/provenance, thứ tự/số lượng case và việc VALIDATION hoàn tất trước TEST; không chạy simulator hoặc tính lại metric chính thức."]
    output.mkdir(parents=True, exist_ok=True)
    (output/"metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (output/"README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = export_campaign(args.campaign, args.runs_root, args.output)
    print(f"Exported {len(report['runs'])} VERIFIED runs / {report['total_episodes']} episodes to {args.output}")


if __name__ == "__main__":
    main()
