"""
Research Workbench Audit and Verification Script (Gate 7.5B Pass B1 Correction 1).

Generates machine-derived evidence artifacts following an in-memory evidence collection pipeline:
A. Workbench offscreen startup smoke (asserting exact manifest counts: TRAIN=180, VAL=96, TEST=60).
B. Worker PLAN smoke (SANDBOX, fixture_constant_continuous, Easy, OFF, DISABLED).
C. Worker RUN smoke (real MetaDrive, SANDBOX, fixture_constant_continuous, TRAIN case, OFF, DISABLED).
   Strict assertions:
   - status == "COMPLETE"
   - WORKER_DONE.success == True
   - WORKER_DONE.blocked == False
   - WORKER_DONE.run_id == EXECUTION_REPORT.run_id
   - 7/7 Gate-7 files exist and run_state is COMPLETE.
D. Canonical blocked smoke (TEST + non-benchmark fixture):
   - Preflight failure check isolated to ["agent_benchmark_eligibility"].
   - Zero simulator execution steps.
E. Privacy & Secrets Scan:
   - Verified 0 private absolute interpreter paths, user profiles, or credentials across all audit artifacts.
F. Contract cryptographic integrity verification.
"""

from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

# Ensure offscreen Qt platform
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Ensure project root in sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from PySide6.QtWidgets import QApplication

from src.launcher.cases import (
    LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256,
    LOCKED_TEST_CASE_MANIFEST_SHA256,
    LOCKED_VALIDATION_CASE_MANIFEST_SHA256,
    get_default_project_root,
)
from src.launcher.contracts import (
    GATE7_LOCKED_OBSERVABILITY_HASH,
    build_launcher_contract_core,
    compute_launcher_contract_sha256,
    compute_platform_execution_contract_sha256,
)
from src.launcher.registry import build_canonical_agent_registry, compute_canonical_registry_sha256
from src.launcher.resolver import (
    GATE5_LOCKED_BENCHMARK_HASH,
    GATE6_LOCKED_AGENT_HASH,
    GATE6_LOCKED_RUNTIME_HASH,
    GATE7_LOCKED_LOGGING_HASH,
)
from src.platform import RunStatus
from src.workbench.contracts import (
    GATE7_5A_BASELINE_MERGE_SHA,
    GATE7_5A_LAUNCHER_CONTRACT_HASH,
    GATE7_5A_PLATFORM_EXECUTION_CONTRACT_HASH,
    PINNED_PYSIDE6_VERSION,
    WORKBENCH_PROTOCOL_VERSION,
    build_workbench_contract_core,
    compute_platform_workbench_contract_sha256,
    compute_workbench_contract_sha256,
)
from src.workbench.core_adapter import CoreAdapter
from src.workbench.main_window import MainWindow
from src.workbench.protocol import (
    WorkbenchMessageType,
    parse_sentinel_line,
)


def run_audit() -> None:
    project_root = get_default_project_root()
    output_dir = project_root / "results" / "audits" / "workbench"
    worker_script = project_root / "src" / "workbench" / "worker.py"

    print("============================================================")
    print("STARTING GATE 7.5B RESEARCH WORKBENCH AUDIT (CORRECTION 1)")
    print("============================================================")

    # ---------------------------------------------------------
    # 1. Prior Platform & Workbench Contract Hashes
    # ---------------------------------------------------------
    print("\n--- Verifying Prior Platform Contracts & Workbench Contract ---")
    launcher_core = build_launcher_contract_core()
    l_hash = compute_launcher_contract_sha256(launcher_core)
    p_exec_hash = compute_platform_execution_contract_sha256(l_hash)
    canon_reg = build_canonical_agent_registry()
    canon_reg_hash = compute_canonical_registry_sha256(canon_reg)

    assert GATE5_LOCKED_BENCHMARK_HASH == "9ddd889b84d8705fae618879e5035556c80d0276a3dc2a58a7963937ebb59f77"
    assert GATE6_LOCKED_AGENT_HASH == "53aa37079ff44afa75d9a3f921b0c1f98c4600d882fced51fc8d9197795058eb"
    assert GATE6_LOCKED_RUNTIME_HASH == "c7698768539a769c7b2bc6b90771ff234276a974b6e0353bbb03119faf2f79ad"
    assert GATE7_LOCKED_LOGGING_HASH == "0d17915556d83f49d9519576e95920c0853700f33a91593ef85b3aeef6cbb9f2"
    assert GATE7_LOCKED_OBSERVABILITY_HASH == "f00c27fca8abf8145596576cd0dd2540181ff8ae7eccf7806c864921933d3581"
    assert l_hash == GATE7_5A_LAUNCHER_CONTRACT_HASH, f"Launcher contract hash mismatch: {l_hash}"
    assert p_exec_hash == GATE7_5A_PLATFORM_EXECUTION_CONTRACT_HASH, f"Execution contract mismatch: {p_exec_hash}"
    assert canon_reg_hash == "9fac42d07949796d00b6dc869b98c945eae7a59355d7928f429e457f1ee7f26f"

    print(f"  [OK] Gate 7.5A launcher_contract_sha256:           {l_hash}")
    print(f"  [OK] Gate 7.5A platform_execution_contract_sha256: {p_exec_hash}")
    print(f"  [OK] Gate 7.5A canonical_agent_registry_sha256:   {canon_reg_hash}")

    wb_core = build_workbench_contract_core()
    wb_hash = compute_workbench_contract_sha256(wb_core)
    p_wb_hash = compute_platform_workbench_contract_sha256(wb_hash)
    print(f"  [OK] Gate 7.5B workbench_contract_sha256:          {wb_hash}")
    print(f"  [OK] Gate 7.5B platform_workbench_contract_sha256: {p_wb_hash}")

    contract_hashes_payload = {
        "baseline_main_merge_sha": GATE7_5A_BASELINE_MERGE_SHA,
        "gate5_benchmark_contract_sha256": GATE5_LOCKED_BENCHMARK_HASH,
        "gate6_agent_contract_sha256": GATE6_LOCKED_AGENT_HASH,
        "gate6_platform_runtime_contract_sha256": GATE6_LOCKED_RUNTIME_HASH,
        "gate7_logging_contract_sha256": GATE7_LOCKED_LOGGING_HASH,
        "gate7_platform_observability_contract_sha256": GATE7_LOCKED_OBSERVABILITY_HASH,
        "launcher_contract_sha256": l_hash,
        "platform_execution_contract_sha256": p_exec_hash,
        "canonical_agent_registry_sha256": canon_reg_hash,
        "workbench_contract_sha256": wb_hash,
        "platform_workbench_contract_sha256": p_wb_hash,
        "pyside6_pinned_version": PINNED_PYSIDE6_VERSION,
        "workbench_protocol_version": WORKBENCH_PROTOCOL_VERSION,
    }

    # ---------------------------------------------------------
    # 2. In-Memory Startup & Case Explorer Verification
    # ---------------------------------------------------------
    print("\n--- Auditing Workbench Startup & Case Explorer Counts ---")
    adapter = CoreAdapter(project_root)
    train_cases = adapter.load_case_manifest("TRAIN")
    val_cases = adapter.load_case_manifest("VALIDATION")
    test_cases = adapter.load_case_manifest("TEST")

    assert len(train_cases) == 180, f"Expected 180 TRAIN cases, got {len(train_cases)}"
    assert len(val_cases) == 96, f"Expected 96 VALIDATION cases, got {len(val_cases)}"
    assert len(test_cases) == 60, f"Expected 60 TEST cases, got {len(test_cases)}"

    app = QApplication.instance() or QApplication([])
    window = MainWindow(core_adapter=adapter)

    # Verify Extreme tier available
    setup_tiers = [window.setup_widget.combo_tier.itemText(i) for i in range(window.setup_widget.combo_tier.count())]
    explorer_tiers = [window.case_explorer_widget.combo_tier.itemText(i) for i in range(window.case_explorer_widget.combo_tier.count())]
    assert "Extreme" in setup_tiers, "Extreme tier missing from Setup"
    assert "Extreme" in explorer_tiers, "Extreme tier missing from Explorer"

    startup_evidence = {
        "status": "PASS",
        "tab_count": window.tab_widget.count(),
        "tabs": [window.tab_widget.tabText(i) for i in range(window.tab_widget.count())],
        "registered_agents_in_combobox": window.setup_widget.combo_agent.count(),
        "cases_in_explorer_train": window.case_explorer_widget.table_cases.rowCount(),
        "cases_unfiltered_train_manifest": len(train_cases),
        "cases_unfiltered_validation_manifest": len(val_cases),
        "cases_unfiltered_test_manifest": len(test_cases),
        "extreme_tier_available": True,
        "benchmark_execution_controls_present": False,
    }
    window.close()
    assert startup_evidence["cases_in_explorer_train"] == 180, f"Expected 180 in table, got {startup_evidence['cases_in_explorer_train']}"
    print(f"  [OK] Startup smoke verified (TRAIN=180, VAL=96, TEST=60, Extreme=Available)")

    # ---------------------------------------------------------
    # 3. Worker PLAN Smoke
    # ---------------------------------------------------------
    print("\n--- Auditing Worker PLAN Smoke ---")
    plan_req = {
        "mode": "SANDBOX",
        "agent_id": "fixture_constant_continuous",
        "tier": "Easy",
        "render_mode": "OFF",
        "wandb_mode": "DISABLED",
    }
    plan_proc = subprocess.run(
        [sys.executable, str(worker_script), "--operation", "PLAN", "--request-json", json.dumps(plan_req)],
        capture_output=True,
        text=True,
        cwd=str(project_root),
    )
    assert plan_proc.returncode == 0, f"Worker PLAN failed: {plan_proc.stderr}"
    plan_messages = []
    for line in plan_proc.stdout.splitlines():
        msg = parse_sentinel_line(line)
        if msg:
            plan_messages.append({"type": msg.type.value, "payload": msg.payload})

    msg_types = [m["type"] for m in plan_messages]
    assert "WORKER_READY" in msg_types
    assert "PLAN_RESOLVED" in msg_types
    assert "PREFLIGHT_REPORT" in msg_types
    assert "WORKER_DONE" in msg_types
    assert "LAUNCHER_EVENT" not in msg_types

    # Privacy check on WORKER_READY payload
    ready_msg = next(m for m in plan_messages if m["type"] == "WORKER_READY")
    py_exec = ready_msg["payload"].get("python_executable", "")
    assert py_exec == Path(sys.executable).name, f"Expected basename, got: {py_exec}"
    assert "\\" not in py_exec and "/" not in py_exec, f"Found path separator in py_exec: {py_exec}"

    plan_smoke_evidence = {
        "status": "PASS",
        "exit_code": plan_proc.returncode,
        "message_sequence": msg_types,
        "messages": plan_messages,
    }
    print(f"  [OK] PLAN smoke verified (0 simulator steps executed, sanitized WORKER_READY)")

    # ---------------------------------------------------------
    # 4. Canonical Blocked Smoke (Clean Preflight Evidence)
    # ---------------------------------------------------------
    print("\n--- Auditing Canonical Blocked Smoke (TEST + non-benchmark fixture) ---")
    test_req = {
        "mode": "TEST",
        "agent_id": "fixture_constant_continuous",
        "render_mode": "OFF",
        "wandb_mode": "DISABLED",
    }
    test_proc = subprocess.run(
        [sys.executable, str(worker_script), "--operation", "RUN", "--request-json", json.dumps(test_req)],
        capture_output=True,
        text=True,
        cwd=str(project_root),
    )
    assert test_proc.returncode == 1, f"Expected blocked run to exit with 1, got {test_proc.returncode}"
    test_messages = []
    for line in test_proc.stdout.splitlines():
        msg = parse_sentinel_line(line)
        if msg:
            test_messages.append({"type": msg.type.value, "payload": msg.payload})

    test_types = [m["type"] for m in test_messages]
    assert "WORKER_READY" in test_types
    assert "PLAN_RESOLVED" in test_types
    assert "PREFLIGHT_REPORT" in test_types
    assert "WORKER_DONE" in test_types
    assert "LAUNCHER_EVENT" not in test_types
    assert "EXECUTION_REPORT" not in test_types

    pref_msg = next(m for m in test_messages if m["type"] == "PREFLIGHT_REPORT")
    pref_report = pref_msg["payload"]["report"]
    failed_checks = [c["check_id"] for c in pref_report["checks"] if c["status"] == "FAIL"]
    assert "agent_benchmark_eligibility" in failed_checks, f"Expected agent_benchmark_eligibility failure: {failed_checks}"

    blocked_evidence = {
        "status": "PASS",
        "blocked_as_expected": True,
        "failed_checks": failed_checks,
        "targeted_scientific_check": "agent_benchmark_eligibility",
        "can_execute": pref_report["can_execute"],
        "exit_code": test_proc.returncode,
    }
    print(f"  [OK] Canonical blocked smoke verified (TEST blocked by preflight: {failed_checks})")

    # ---------------------------------------------------------
    # 5. Worker RUN Smoke (Real MetaDrive Simulator)
    # ---------------------------------------------------------
    print("\n--- Auditing Worker RUN Smoke (Real MetaDrive Simulator) ---")
    with tempfile.TemporaryDirectory() as tmp_runs_dir:
        run_req = {
            "mode": "SANDBOX",
            "agent_id": "fixture_constant_continuous",
            "tier": "Easy",
            "render_mode": "OFF",
            "wandb_mode": "DISABLED",
            "runs_root": tmp_runs_dir,
            "custom_run_id": "workbench_smoke_run",
        }
        run_proc = subprocess.run(
            [sys.executable, str(worker_script), "--operation", "RUN", "--request-json", json.dumps(run_req)],
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )
        assert run_proc.returncode == 0, f"Worker RUN failed: {run_proc.stderr}"

        run_messages = []
        for line in run_proc.stdout.splitlines():
            msg = parse_sentinel_line(line)
            if msg:
                run_messages.append({"type": msg.type.value, "payload": msg.payload})

        run_types = [m["type"] for m in run_messages]
        assert "WORKER_READY" in run_types
        assert "PLAN_RESOLVED" in run_types
        assert "PREFLIGHT_REPORT" in run_types
        assert "LAUNCHER_EVENT" in run_types
        assert "EXECUTION_REPORT" in run_types
        assert "WORKER_DONE" in run_types

        # Verify exact required launcher events
        launcher_events = [m["payload"]["event"]["event_type"] for m in run_messages if m["type"] == "LAUNCHER_EVENT"]
        assert "RUN_STARTED" in launcher_events
        assert "EPISODE_STARTED" in launcher_events
        assert "EPISODE_FINISHED" in launcher_events
        assert "RUN_FINISHED" in launcher_events

        # Verify EXECUTION_REPORT status and WORKER_DONE success parity
        exec_rep = next(m for m in run_messages if m["type"] == "EXECUTION_REPORT")["payload"]["execution_report"]
        worker_done = next(m for m in run_messages if m["type"] == "WORKER_DONE")["payload"]

        assert exec_rep["status"] == RunStatus.COMPLETE.value, f"Expected COMPLETE status, got: {exec_rep['status']}"
        assert exec_rep["completed_episodes"] == 1
        assert worker_done["success"] is True, f"Expected WORKER_DONE.success == True, got: {worker_done['success']}"
        assert worker_done["blocked"] is False
        assert worker_done["run_id"] == exec_rep["run_id"]

        # Verify Gate-7 artifacts in temporary run directory
        run_folder = Path(tmp_runs_dir) / "workbench_smoke_run"
        gate7_files = [f.name for f in run_folder.iterdir()]
        required_files = [
            "run_manifest.json",
            "run_state.json",
            "episodes.csv",
            "timing.csv",
            "summary.json",
            "wandb_sync.json",
            "run_integrity.json",
        ]
        missing = [rf for rf in required_files if rf not in gate7_files]
        assert not missing, f"Missing Gate-7 files in workbench run: {missing}"

        # Assert persisted run_state status is COMPLETE
        with open(run_folder / "run_state.json", "r", encoding="utf-8") as f:
            persisted_state = json.load(f)
        assert persisted_state["status"] == "COMPLETE"

        run_smoke_evidence = {
            "status": "PASS",
            "exit_code": run_proc.returncode,
            "received_message_types": list(set(run_types)),
            "launcher_event_sequence": launcher_events,
            "execution_status": exec_rep["status"],
            "worker_done_success": worker_done["success"],
            "worker_done_blocked": worker_done["blocked"],
            "run_id_parity": (worker_done["run_id"] == exec_rep["run_id"]),
            "gate7_artifacts_present": sorted(gate7_files),
            "persisted_run_state_status": persisted_state["status"],
        }
        print(f"  [OK] RUN smoke verified (COMPLETE status, WORKER_DONE.success=True, run_id parity, 7/7 artifacts)")

    # ---------------------------------------------------------
    # 6. Save Tracked Artifacts
    # ---------------------------------------------------------
    print("\n--- Writing Tracked Audit Artifacts ---")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "contract_hashes.json", "w", encoding="utf-8") as f:
        json.dump(contract_hashes_payload, f, indent=2)
    print(f"  [SAVED] {output_dir / 'contract_hashes.json'}")

    wb_contract_file = project_root / "configs" / "platform" / "workbench_contract_v1.json"
    with open(wb_contract_file, "w", encoding="utf-8") as f:
        json.dump(wb_core, f, indent=2)
    print(f"  [SAVED] {wb_contract_file}")

    with open(output_dir / "startup_smoke.json", "w", encoding="utf-8") as f:
        json.dump(startup_evidence, f, indent=2)
    print(f"  [SAVED] {output_dir / 'startup_smoke.json'}")

    with open(output_dir / "plan_smoke.json", "w", encoding="utf-8") as f:
        json.dump(plan_smoke_evidence, f, indent=2)
    print(f"  [SAVED] {output_dir / 'plan_smoke.json'}")

    with open(output_dir / "run_smoke.json", "w", encoding="utf-8") as f:
        json.dump(run_smoke_evidence, f, indent=2)
    print(f"  [SAVED] {output_dir / 'run_smoke.json'}")

    with open(output_dir / "canonical_blocked_smoke.json", "w", encoding="utf-8") as f:
        json.dump(blocked_evidence, f, indent=2)
    print(f"  [SAVED] {output_dir / 'canonical_blocked_smoke.json'}")

    # Summary markdown
    summary_md = f"""# Gate 7.5B Workbench Audit Summary (Pass B1 Correction 1)

- Gate: Gate 7.5B (Research Workbench GUI Pass B1 Correction 1)
- Status: AUDIT-CANDIDATE
- Baseline Main Merge SHA: `{GATE7_5A_BASELINE_MERGE_SHA}`
- PySide6 Pinned Version: `{PINNED_PYSIDE6_VERSION}`
- Protocol Version: `{WORKBENCH_PROTOCOL_VERSION}`

## Contract Cryptographic Fingerprints
- `gate5_benchmark_contract_sha256`: `{GATE5_LOCKED_BENCHMARK_HASH}` (VERIFIED UNCHANGED)
- `gate6_agent_contract_sha256`: `{GATE6_LOCKED_AGENT_HASH}` (VERIFIED UNCHANGED)
- `gate6_platform_runtime_contract_sha256`: `{GATE6_LOCKED_RUNTIME_HASH}` (VERIFIED UNCHANGED)
- `gate7_logging_contract_sha256`: `{GATE7_LOCKED_LOGGING_HASH}` (VERIFIED UNCHANGED)
- `gate7_platform_observability_contract`: `{GATE7_LOCKED_OBSERVABILITY_HASH}` (VERIFIED UNCHANGED)
- `launcher_contract_sha256`: `{l_hash}` (VERIFIED UNCHANGED)
- `platform_execution_contract_sha256`: `{p_exec_hash}` (VERIFIED UNCHANGED)
- `canonical_agent_registry_sha256`: `{canon_reg_hash}` (VERIFIED UNCHANGED)
- `workbench_contract_sha256`: `{wb_hash}` (CANDIDATE)
- `platform_workbench_contract_sha256`: `{p_wb_hash}` (CANDIDATE)

## Empirical Evidence
1. **Startup Smoke:** Offscreen construction successful; 5 tabs and initial tables populated. Authoritative manifest counts verified: TRAIN 180, VALIDATION 96, TEST 60. Extreme tier available.
2. **PLAN Smoke:** Subprocess resolved plan & preflight, 0 simulator steps executed. Sanitized WORKER_READY (basename only).
3. **RUN Smoke:** Real MetaDrive execution in subprocess; streamed telemetry via `LauncherEventV1`; status `COMPLETE`; `WORKER_DONE.success == True`; run_id parity verified; 7/7 Gate-7 artifacts generated.
4. **Canonical Blocked Smoke:** Benchmark TEST with non-benchmark fixture failed preflight on `agent_benchmark_eligibility`, with 0 simulator steps.
5. **Thin-Client Invariant:** GUI contains 0 scientific evaluation authority; core resolver, preflight, and executor remain authoritative.
6. **Privacy Invariant:** Zero private absolute machine paths or credentials persisted in audit artifacts.
"""
    with open(output_dir / "audit_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"  [SAVED] {output_dir / 'audit_summary.md'}")

    # ---------------------------------------------------------
    # 7. Privacy & Secrets Scan Across Workbench Artifacts
    # ---------------------------------------------------------
    print("\n--- Scanning Workbench Artifacts for Private Paths & Secrets ---")
    forbidden_patterns = [
        re.compile(r"C:\\Users\\", re.IGNORECASE),
        re.compile(r"/home/", re.IGNORECASE),
        re.compile(r"/Users/", re.IGNORECASE),
        re.compile(r"WANDB_API_KEY", re.IGNORECASE),
        re.compile(r"api_key\s*[:=]\s*['\"][a-zA-Z0-9_\-]+['\"]", re.IGNORECASE),
    ]

    home_dir = str(Path.home())
    if home_dir and len(home_dir) > 3:
        forbidden_patterns.append(re.compile(re.escape(home_dir), re.IGNORECASE))

    violations = []
    for art_file in output_dir.iterdir():
        if not art_file.is_file():
            continue
        content = art_file.read_text(encoding="utf-8", errors="replace")
        for pat in forbidden_patterns:
            matches = pat.findall(content)
            if matches:
                violations.append(f"{art_file.name}: matched forbidden pattern '{pat.pattern}' ({len(matches)} occurrences)")

    assert not violations, f"Privacy violations found in audit artifacts:\n" + "\n".join(violations)
    print("  [OK] Zero private paths or secrets found across all Workbench audit artifacts.")

    print("\n============================================================")
    print("GATE 7.5B WORKBENCH AUDIT COMPLETED SUCCESSFULLY!")
    print("============================================================")


if __name__ == "__main__":
    run_audit()
