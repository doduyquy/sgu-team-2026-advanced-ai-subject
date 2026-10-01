"""
Command-Line Interface for Research Platform V1 Launcher (Gate 7.5A).

This module implements a thin CLI adapter over the Launcher Core:
- commands: agents, cases, plan, run.
- --json flag for structured machine consumption by future Gate 7.5B GUI or automation.
- Zero duplicate scientific or resolution logic; delegates strictly to models, resolver, preflight, and executor.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from src.launcher.cases import (
    LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256,
    LOCKED_TEST_CASE_MANIFEST_SHA256,
    LOCKED_VALIDATION_CASE_MANIFEST_SHA256,
    get_default_project_root,
    load_manifest_csv,
)
from src.launcher.executor import ExperimentExecutor
from src.launcher.models import (
    LaunchRequestV1,
    LauncherMode,
    describe_plan,
)
from src.launcher.preflight import run_preflight
from src.launcher.registry import build_default_agent_registry
from src.launcher.resolver import resolve_experiment_plan
from src.platform import WandbMode


def create_parser() -> argparse.ArgumentParser:
    """Constructs the CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m src.launcher.cli",
        description="Research Platform V1 Autonomous Driving Launcher Core CLI (Gate 7.5A)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Launcher command to execute")

    # Command: agents
    p_agents = subparsers.add_parser("agents", help="List registered agents and capabilities")
    p_agents.add_argument("--json", action="store_true", help="Output structured JSON")

    # Command: cases
    p_cases = subparsers.add_parser("cases", help="Inspect Gate-5 benchmark and training cases")
    p_cases.add_argument("--split", choices=["TRAIN", "VALIDATION", "TEST"], default="TRAIN", help="Split to query")
    p_cases.add_argument("--tier", choices=["Easy", "Medium", "Hard", "Extreme", "easy", "medium", "hard", "extreme"], default=None, help="Difficulty tier filter")
    p_cases.add_argument("--sequence", default=None, help="Sequence family filter (e.g. SCS, SCXCS)")
    p_cases.add_argument("--limit", type=int, default=20, help="Maximum cases to display")
    p_cases.add_argument("--json", action="store_true", help="Output structured JSON")

    # Command: plan
    p_plan = subparsers.add_parser("plan", help="Preview resolved experiment plan and run preflight checks (zero side-effects)")
    _add_plan_arguments(p_plan)

    # Command: run
    p_run = subparsers.add_parser("run", help="Resolve plan, run preflight, and execute experiment suite")
    _add_plan_arguments(p_run)

    return parser


def _add_plan_arguments(subparser: argparse.ArgumentParser) -> None:
    """Adds common plan/run configuration arguments."""
    subparser.add_argument("--mode", choices=["SANDBOX", "VALIDATION", "TEST", "AUDIT"], default="SANDBOX", help="Launcher mode")
    subparser.add_argument("--agent", default="fixture_seeded_random", help="Registered agent_id")
    subparser.add_argument("--tier", choices=["Easy", "Medium", "Hard", "Extreme", "easy", "medium", "hard", "extreme"], default=None, help="Difficulty tier")
    subparser.add_argument("--sequence", default=None, help="Sequence family")
    subparser.add_argument("--geom-seed", type=int, default=None, dest="geom_seed", help="Geometry generation seed")
    subparser.add_argument("--env-seed", type=int, default=None, dest="env_seed", help="Environment seed")
    subparser.add_argument("--agent-seed", type=int, default=None, dest="agent_seed", help="Agent stochasticity seed (101, 202, 303 for benchmark)")
    subparser.add_argument("--render", choices=["OFF", "NATIVE", "off", "native"], default="OFF", help="Rendering mode")
    subparser.add_argument("--wandb", choices=["AUTO", "DISABLED", "OFFLINE", "ONLINE", "auto", "disabled", "offline", "online"], default="AUTO", help="W&B tracking mode (AUTO defaults to OFFLINE for benchmarks, DISABLED for sandbox)")
    subparser.add_argument("--runs-root", default=None, help="Custom runs storage directory override")
    subparser.add_argument("--run-id", "--custom-run-id", default=None, dest="custom_run_id", help="Custom run_id identifier")
    subparser.add_argument("--json", action="store_true", help="Output structured JSON")


def handle_agents(args: argparse.Namespace) -> int:
    registry = build_default_agent_registry()
    snapshot = registry.to_snapshot()

    if args.json:
        print(json.dumps(snapshot, indent=2))
        return 0

    print("============================================================")
    print("REGISTERED AGENTS IN PLATFORM V1")
    print("============================================================")
    for a in snapshot:
        elig = []
        if a["sandbox_eligible"]: elig.append("SANDBOX")
        if a["benchmark_eligible"]: elig.append("BENCHMARK")
        if a["audit_eligible"]: elig.append("AUDIT")
        print(f"[{a['agent_id']}] v{a['agent_version']} ({a['method_family']})")
        print(f"  Purpose:     {a['purpose']} | Stage: {a['stage_label'] or 'None (Fixture)'}")
        print(f"  Profile:     {a['input_profile_id']} -> {a['action_adapter_id']}")
        print(f"  Inference:   {a['inference_stochasticity']} (Stateful={a['stateful_within_episode']})")
        print(f"  Eligible:    {', '.join(elig)}")
        print(f"  Description: {a['description']}")
        print("")
    return 0


def handle_cases(args: argparse.Namespace) -> int:
    root = getattr(args, "project_root", None) or get_default_project_root()
    split = args.split.upper()

    if split == "TRAIN":
        p = root / "results" / "audits" / "evaluation_protocol" / "geometry_split_manifest.csv"
        records = [r for r in load_manifest_csv(p, expected_sha256=LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256) if r["split"].upper() == "TRAIN"]
    elif split == "VALIDATION":
        p = root / "results" / "audits" / "evaluation_protocol" / "validation_case_manifest.csv"
        records = load_manifest_csv(p, expected_sha256=LOCKED_VALIDATION_CASE_MANIFEST_SHA256)
    else:
        p = root / "results" / "audits" / "evaluation_protocol" / "test_case_manifest.csv"
        records = load_manifest_csv(p, expected_sha256=LOCKED_TEST_CASE_MANIFEST_SHA256)

    if args.tier:
        records = [r for r in records if r["tier"].lower() == args.tier.lower()]
    if args.sequence:
        records = [r for r in records if r["sequence"] == args.sequence]

    displayed = records[:args.limit]

    if args.json:
        print(json.dumps({
            "split": split,
            "total_matched": len(records),
            "displayed_count": len(displayed),
            "cases": displayed
        }, indent=2))
        return 0

    print(f"=== {split} CASES (Showing {len(displayed)} of {len(records)}) ===")
    for r in displayed:
        c_id = r.get("case_id", r.get("geometry_id"))
        env_s = r.get("environment_seed", "N/A")
        print(f"  Case: {c_id:30s} | Tier: {r['tier']:7s} | Seq: {r['sequence']:6s} | GeomSeed: {r['geometry_generation_seed']:4s} | EnvSeed: {env_s}")
    return 0


def build_launch_request_from_args(args: argparse.Namespace) -> LaunchRequestV1:
    """Builds LaunchRequestV1 from parsed CLI arguments."""
    mode = LauncherMode(args.mode.upper())
    wandb_arg = args.wandb.upper()
    wandb_mode = None if wandb_arg == "AUTO" else WandbMode(wandb_arg)
    render_mode = args.render.upper()
    tier = args.tier.capitalize() if args.tier else None
    runs_root = Path(args.runs_root) if args.runs_root else None
    custom_run_id = getattr(args, "custom_run_id", None)

    return LaunchRequestV1(
        mode=mode,
        agent_id=args.agent,
        tier=tier,
        sequence=args.sequence,
        geometry_generation_seed=args.geom_seed,
        environment_seed=args.env_seed,
        agent_seed=args.agent_seed,
        render_mode=render_mode,
        wandb_mode=wandb_mode,
        runs_root=runs_root,
        custom_run_id=custom_run_id
    )


def handle_plan(args: argparse.Namespace) -> int:
    request = build_launch_request_from_args(args)
    registry = build_default_agent_registry()

    try:
        plan = resolve_experiment_plan(request, registry)
    except Exception as e:
        print(f"[ERROR] Failed to resolve experiment plan: {e}", file=sys.stderr)
        return 1

    preflight = run_preflight(plan)

    if args.json:
        out = {
            "request": request.to_dict(),
            "plan": plan.to_dict(),
            "preflight": preflight.to_dict()
        }
        print(json.dumps(out, indent=2))
        return 0 if preflight.can_execute else 2

    # Human-readable display
    print(describe_plan(plan))
    print("")
    print("--- Preflight Checks ---")
    for c in preflight.checks:
        tag = f"[{c.status}]"
        print(f"  {tag:9s} ({c.category:12s}) {c.message}")
    print("")
    print(f"Preflight Verdict: {preflight.summary_verdict} (Can Execute: {preflight.can_execute})")
    return 0 if preflight.can_execute else 2


def handle_run(args: argparse.Namespace) -> int:
    request = build_launch_request_from_args(args)
    registry = build_default_agent_registry()

    try:
        plan = resolve_experiment_plan(request, registry)
    except Exception as e:
        print(f"[ERROR] Failed to resolve experiment plan: {e}", file=sys.stderr)
        return 1

    preflight = run_preflight(plan, registry=registry)
    if not preflight.can_execute:
        print(f"[ERROR] Preflight validation failed with {preflight.fail_count} blocking error(s):", file=sys.stderr)
        for c in preflight.checks:
            if c.status == "FAIL":
                print(f"  [FAIL] ({c.category}) {c.message}", file=sys.stderr)
        return 2

    executor = ExperimentExecutor(
        plan=plan,
        registry=registry,
        runs_root=request.runs_root,
        custom_run_id=request.custom_run_id
    )

    try:
        report = executor.execute()
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print("============================================================")
            print(f"EXPERIMENT COMPLETED: {report.status}")
            print(f"Run ID:    {report.run_id}")
            print(f"Directory: {report.run_dir}")
            print(f"Episodes:  {report.completed_episodes} / {report.total_cases}")
            if report.summary_payload and "overall_metrics" in report.summary_payload:
                ov = report.summary_payload["overall_metrics"]
                print(f"Clean Success Rate: {ov.get('clean_success_rate')}")
                print(f"Mean Route Completion: {ov.get('mean_final_route_completion'):.3f}")
            print("============================================================")
        return 0
    except Exception as e:
        print(f"[ERROR] Execution failed: {e}", file=sys.stderr)
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "agents":
        return handle_agents(args)
    elif args.command == "cases":
        return handle_cases(args)
    elif args.command == "plan":
        return handle_plan(args)
    elif args.command == "run":
        return handle_run(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
