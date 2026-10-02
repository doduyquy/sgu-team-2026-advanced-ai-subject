"""
Workbench Worker Subprocess Entrypoint (Gate 7.5B).

This script runs in an isolated Python subprocess managed by the Workbench GUI:
- Parses serialized LaunchRequestV1 from stdin or argument.
- Dispatches either "PLAN" or "RUN" operation.
- In "PLAN" mode:
  1. Resolves plan via resolve_experiment_plan().
  2. Runs preflight via run_preflight().
  3. Emits PLAN_RESOLVED and PREFLIGHT_REPORT.
  4. Emits WORKER_DONE and exits cleanly.
- In "RUN" mode:
  1. Resolves plan via resolve_experiment_plan().
  2. Runs preflight via run_preflight().
  3. If preflight blocked: emits PLAN_RESOLVED, PREFLIGHT_REPORT, WORKER_DONE(blocked=True).
  4. If preflight allowed: runs ExperimentExecutor with callback streaming LAUNCHER_EVENT.
  5. Emits EXECUTION_REPORT and WORKER_DONE.
- All structured messages use the @@WORKBENCH@@ sentinel.
- Non-sentinel output (from Panda3D, MetaDrive, etc.) passes through safely.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.launcher.events import LauncherEventV1
from src.launcher.executor import ExperimentExecutor
from src.launcher.models import LauncherMode, PreflightBlockedError
from src.launcher.preflight import run_preflight
from src.launcher.resolver import resolve_experiment_plan
from src.workbench.contracts import WORKBENCH_PROTOCOL_VERSION
from src.workbench.protocol import (
    WorkbenchMessageType,
    WorkbenchMessageV1,
    deserialize_launch_request,
)


def emit(msg_type: WorkbenchMessageType, payload: Dict[str, Any]) -> None:
    """Emits a structured sentinel line to worker stdout and flushes immediately."""
    msg = WorkbenchMessageV1.create(msg_type, payload)
    sys.stdout.write(msg.to_line() + "\n")
    sys.stdout.flush()


def run_worker_main() -> int:
    parser = argparse.ArgumentParser(description="Research Workbench Worker Subprocess")
    parser.add_argument("--operation", choices=["PLAN", "RUN"], required=True, help="Workbench operation mode")
    parser.add_argument("--request-json", type=str, default=None, help="Inline serialized LaunchRequestV1 JSON")
    parser.add_argument("--request-file", type=str, default=None, help="Path to file containing LaunchRequestV1 JSON")
    args = parser.parse_args()

    emit(WorkbenchMessageType.WORKER_READY, {
        "operation": args.operation,
        "pid": os.getpid(),
        "python_executable": sys.executable,
        "protocol_version": WORKBENCH_PROTOCOL_VERSION,
    })

    try:
        raw_json_str = ""
        if args.request_file:
            with open(args.request_file, "r", encoding="utf-8") as f:
                raw_json_str = f.read()
        elif args.request_json:
            raw_json_str = args.request_json
        else:
            # Fall back to stdin if neither argument provided
            raw_json_str = sys.stdin.read()

        if not raw_json_str.strip():
            raise ValueError("No launch request payload provided to worker.")

        req_dict = json.loads(raw_json_str)
        launch_request = deserialize_launch_request(req_dict)

        # Canonical authority routing (Specification Section 12):
        # For VALIDATION / TEST: do NOT provide an external registry to resolver, preflight, or executor.
        # For SANDBOX / AUDIT: use normal default development registry as allowed by Gate 7.5A.
        # Notice we pass registry=None, allowing Gate 7.5A to enforce its canonical rules.
        resolved_plan = resolve_experiment_plan(launch_request)
        plan_dict = resolved_plan.to_dict()

        emit(WorkbenchMessageType.PLAN_RESOLVED, {
            "plan": plan_dict,
            "resolved_plan_sha256": resolved_plan.resolved_plan_sha256,
            "case_count": len(resolved_plan.resolved_cases),
            "canonical_run": resolved_plan.canonical_run,
        })

        preflight_report = run_preflight(resolved_plan)
        emit(WorkbenchMessageType.PREFLIGHT_REPORT, {
            "report": preflight_report.to_dict(),
            "can_execute": preflight_report.can_execute,
            "summary_verdict": preflight_report.summary_verdict,
            "fail_count": preflight_report.fail_count,
            "warning_count": preflight_report.warning_count,
            "pass_count": preflight_report.pass_count,
        })

        if args.operation == "PLAN":
            emit(WorkbenchMessageType.WORKER_DONE, {
                "operation": "PLAN",
                "success": True,
                "can_execute": preflight_report.can_execute,
            })
            return 0

        # Operation is "RUN"
        if not preflight_report.can_execute:
            emit(WorkbenchMessageType.WORKER_DONE, {
                "operation": "RUN",
                "success": False,
                "blocked": True,
                "reason": "Preflight validation failed",
            })
            return 1

        # Preflight passed, proceed with ExperimentExecutor
        def on_launcher_event(event: LauncherEventV1) -> None:
            emit(WorkbenchMessageType.LAUNCHER_EVENT, {
                "event": event.to_dict(),
            })

        executor = ExperimentExecutor(
            plan=resolved_plan,
            runs_root=launch_request.runs_root,
            custom_run_id=launch_request.custom_run_id,
            event_callback=on_launcher_event,
        )

        exec_report = executor.execute()
        report_dict = exec_report.to_dict()

        emit(WorkbenchMessageType.EXECUTION_REPORT, {
            "execution_report": report_dict,
        })

        emit(WorkbenchMessageType.WORKER_DONE, {
            "operation": "RUN",
            "success": (report_dict.get("status") == "COMPLETED"),
            "blocked": False,
            "run_id": report_dict.get("run_id"),
        })
        return 0

    except PreflightBlockedError as pbe:
        emit(WorkbenchMessageType.WORKER_ERROR, {
            "error_type": "PreflightBlockedError",
            "message": str(pbe),
        })
        emit(WorkbenchMessageType.WORKER_DONE, {
            "operation": args.operation,
            "success": False,
            "blocked": True,
            "reason": str(pbe),
        })
        return 1

    except Exception as e:
        emit(WorkbenchMessageType.WORKER_ERROR, {
            "error_type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        })
        emit(WorkbenchMessageType.WORKER_DONE, {
            "operation": args.operation,
            "success": False,
            "blocked": False,
            "reason": str(e),
        })
        return 1


if __name__ == "__main__":
    sys.exit(run_worker_main())
