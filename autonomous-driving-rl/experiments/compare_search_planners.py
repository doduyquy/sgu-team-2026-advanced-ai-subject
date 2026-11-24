"""Compare all four fixed Stage 2 planners on the same non-benchmark TRAIN case."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Execute the four TRAIN runs; otherwise only show the fixed schedule")
    args = parser.parse_args(argv)
    # Keep the project's Windows Qt/MetaDrive DLL import order.
    from PySide6.QtWidgets import QApplication
    from src.launcher.executor import ExperimentExecutor
    from src.launcher.preflight import run_preflight
    from src.launcher.registry import build_canonical_agent_registry
    from src.launcher.resolver import resolve_experiment_plan
    from src.training.runtime import local_training_runtime
    from src.workbench.benchmark_campaign import build_evaluation_campaign, inspect_campaign_results, load_evaluation_profiles
    from src.workbench.results_repository import RunArtifactRepository

    registry = build_canonical_agent_registry()
    methods = ("stage2_bfs", "stage2_dijkstra", "stage2_astar", "stage2_hybrid_astar")
    profile = next(p for p in load_evaluation_profiles(ROOT) if p.profile_id == "train_demo_v1")
    campaign = build_evaluation_campaign(profile, [registry.get(a) for a in methods], show_demo=False)
    if campaign.blocking_reasons:
        raise ValueError(campaign.blocking_reasons)
    requests = [j.request.to_dict() for j in campaign.jobs]
    if not args.run:
        print(json.dumps(dict(scope="TRAIN_DEMO_NON_BENCHMARK", requests=requests), indent=2))
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "artifacts/search_comparison" / f"{stamp}_{uuid4().hex[:6]}"
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status="RUNNING", scope="TRAIN_DEMO_NON_BENCHMARK", fingerprint=campaign.fingerprint,
                 requests=requests, reports=[])

    def save():
        temporary = output / "comparison.tmp"
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output / "comparison.json")

    try:
        with local_training_runtime(ROOT):
            plans = [resolve_experiment_plan(j.request, project_root=ROOT) for j in campaign.jobs]
            state["preflight"] = [run_preflight(plan, ROOT).to_dict() for plan in plans]
            save()
            if not all(report["can_execute"] for report in state["preflight"]):
                raise ValueError("A planner failed preflight; inspect comparison.json")
            for index, plan in enumerate(plans, 1):
                agent_id = plan.agent_registration.agent_id
                print(f"Running {index}/4: {agent_id}", flush=True)
                with (output / f"{agent_id}.log").open("w", encoding="utf-8") as log:
                    with redirect_stdout(log), redirect_stderr(log):
                        report = ExperimentExecutor(plan, project_root=ROOT).execute()
                state["reports"].append(report.to_dict())
                save()
                if report.status != "COMPLETE":
                    raise ValueError(f"Incomplete run: {report.run_id}")
            state["metrics"] = inspect_campaign_results(campaign, state["reports"], ROOT)
            repository = RunArtifactRepository(ROOT / "runs")
            state["episodes"] = []
            lines = ["# Stage 2 · So sánh tìm kiếm trên TRAIN", "",
                     "Cùng case Easy/SCS, geometry 0, environment 0; chạy headless. Không phải benchmark.", "",
                     "| Phương pháp | Hoàn thành tuyến | Số bước | Kết thúc | Integrity |",
                     "|---|---:|---:|---|---|"]
            for metric in state["metrics"]:
                snapshot = repository.load_run_snapshot(metric["run_id"])
                row = snapshot.episode_rows[0]
                state["episodes"].append(dict(run_id=snapshot.run_id, integrity=snapshot.integrity_status.value, episode=row))
                lines.append(f"| {snapshot.agent_id} | {100 * metric['mean_final_route_completion']:.2f}% | "
                             f"{row['episode_steps']} | {row['primary_terminal_reason']} | {snapshot.integrity_status.value} |")
            lines += ["", "Một case TRAIN chỉ kiểm chứng tích hợp, không kết luận phương pháp thắng.", "",
                      "BFS/Dijkstra/A* dùng chung đồ thị. Hybrid A* dùng không gian trạng thái và chi phí chuyển động riêng.", "",
                      "## Run ID", ""]
            lines.extend(f"- `{metric['agent_id']}`: `{metric['run_id']}`" for metric in state["metrics"])
            lines += ["", "[Cấu hình, preflight, metric và episode](comparison.json)."]
            (output / "KET_QUA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            state.update(status="COMPLETE", finished_at_utc=datetime.now(timezone.utc).isoformat())
            save()
            print(json.dumps(state["metrics"], indent=2), flush=True)
            print(f"Results: {output / 'KET_QUA.md'}", flush=True)
        return 0
    except BaseException as error:
        state.update(status="FAILED", error=str(error))
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
