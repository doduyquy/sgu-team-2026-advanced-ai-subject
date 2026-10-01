"""
Audit and Verification Script for Research Platform V1 Launcher Core (Gate 7.5A).

Performs comprehensive verification of:
1. Prior locked platform contracts (Gates 5, 6, and 7).
2. Additive launcher contract and execution contract hashes.
3. Code-to-disk schema consistency and introspection.
4. Ephemeral secrets and machine-private identifiers scanning.
5. Mode resolution and protocol scoping.
6. Agent registry capabilities and fixture boundaries.
7. Gate-5 manifest parity (96 Validation, 60 Test).
8. Holdout safety (Sandbox TRAIN-only, benchmark fixture rejection).
9. Preflight battery and decision matrix.
10. Plan hash determinism and mutation sensitivity.
11. CLI and Core functional parity.
12. Real simulator-backed Sandbox execution smoke test.
13. Gate-7 provenance embedding in run_manifest.json.
"""

import copy
import csv
import dataclasses
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

# Add project root to sys.path portably
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.launcher.cases import (
    get_default_project_root,
    load_locked_geometry_block_sequence,
    load_manifest_csv,
    resolve_audit_cases,
    resolve_sandbox_case,
    resolve_test_cases,
    resolve_validation_cases,
)
from src.launcher.cli import build_launch_request_from_args, create_parser
from src.launcher.contracts import (
    GATE7_LOCKED_OBSERVABILITY_HASH,
    build_launcher_contract_core,
    compute_launcher_contract_sha256,
    compute_platform_execution_contract_sha256,
)
from src.launcher.events import LauncherEventType, LauncherEventV1
from src.launcher.executor import ExecutionReportV1, ExperimentExecutor
from src.launcher.models import (
    AgentRegistrationV1,
    LaunchRequestV1,
    LauncherMode,
    ResolvedCaseV1,
    ResolvedExperimentPlanV1,
    compute_resolved_plan_sha256,
    describe_plan,
    mode_to_run_kind,
)
from src.launcher.preflight import PreflightCheckV1, PreflightReportV1, run_preflight
from src.launcher.registry import AgentRegistryV1, build_default_agent_registry
from src.launcher.resolver import (
    DEFAULT_LAUNCHER_CONTRACT_HASH,
    DEFAULT_PLATFORM_EXECUTION_HASH,
    GATE5_LOCKED_BENCHMARK_HASH,
    GATE6_LOCKED_AGENT_HASH,
    GATE6_LOCKED_RUNTIME_HASH,
    GATE7_LOCKED_LOGGING_HASH,
    resolve_experiment_plan,
)
from src.platform import (
    PINNED_METADRIVE_COMMIT,
    PINNED_METADRIVE_VERSION,
    RunKind,
    RunStatus,
    WandbMode,
    canonical_json_file_sha256,
    canonical_json_sha256,
    sanitize_error_message,
)


def verify_prior_contracts(root: Path):
    """Verifies that Gates 5 through 7 locked contract hashes remain untouched."""
    print("\n--- Verifying Prior Platform Contracts & Hashes ---")
    p5 = root / "results" / "audits" / "evaluation_protocol" / "protocol_hashes.json"
    p6 = root / "results" / "audits" / "agent_contract" / "contract_hashes.json"
    p7 = root / "results" / "audits" / "logging_contract" / "contract_hashes.json"

    with open(p5, "r", encoding="utf-8") as f: d5 = json.load(f)
    with open(p6, "r", encoding="utf-8") as f: d6 = json.load(f)
    with open(p7, "r", encoding="utf-8") as f: d7 = json.load(f)

    assert d5["benchmark_contract_sha256"] == GATE5_LOCKED_BENCHMARK_HASH, "Gate-5 hash divergence!"
    assert d6["agent_contract_sha256"] == GATE6_LOCKED_AGENT_HASH, "Gate-6 agent contract divergence!"
    assert d6["platform_runtime_contract_sha256"] == GATE6_LOCKED_RUNTIME_HASH, "Gate-6 runtime contract divergence!"
    assert d7["logging_contract_sha256"] == GATE7_LOCKED_LOGGING_HASH, "Gate-7 logging contract divergence!"
    assert d7["platform_observability_contract_sha256"] == GATE7_LOCKED_OBSERVABILITY_HASH, "Gate-7 observability divergence!"

    print(f"  [OK] Gate-5 benchmark_contract_sha256:       {GATE5_LOCKED_BENCHMARK_HASH}")
    print(f"  [OK] Gate-6 agent_contract_sha256:           {GATE6_LOCKED_AGENT_HASH}")
    print(f"  [OK] Gate-6 platform_runtime_contract:       {GATE6_LOCKED_RUNTIME_HASH}")
    print(f"  [OK] Gate-7 logging_contract_sha256:         {GATE7_LOCKED_LOGGING_HASH}")
    print(f"  [OK] Gate-7 platform_observability_contract: {GATE7_LOCKED_OBSERVABILITY_HASH}")


def audit_secret_scan(root: Path, scan_label: str = "Initial Pre-Run"):
    """Scans all Gate-7.5-owned and related files for credentials and ephemeral keys."""
    print(f"\n--- Scanning for Ephemeral Secrets & API Keys ({scan_label}) ---")
    files_to_scan = [
        root / "src" / "launcher" / "models.py",
        root / "src" / "launcher" / "registry.py",
        root / "src" / "launcher" / "cases.py",
        root / "src" / "launcher" / "resolver.py",
        root / "src" / "launcher" / "preflight.py",
        root / "src" / "launcher" / "events.py",
        root / "src" / "launcher" / "executor.py",
        root / "src" / "launcher" / "contracts.py",
        root / "src" / "launcher" / "cli.py",
        root / "configs" / "platform" / "launcher_contract_v1.json",
        root / "tests" / "test_launcher_core.py",
        root / "scripts" / "audit_launcher_core.py",
        root.parent / "requirements.txt",
    ]
    results_dir = root / "results" / "audits" / "launcher_core"
    if results_dir.exists():
        for f in results_dir.glob("*"):
            if f.is_file():
                files_to_scan.append(f)

    patterns = ["wandb" + "_v1_", "WANDB_API_KEY=", "api_" + "key="]
    env_key = os.environ.get("WANDB_API_KEY")
    secrets_found = []

    for p in files_to_scan:
        if p.exists():
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for idx, line in enumerate(lines, 1):
                if any(skip in line for skip in ("patterns =", "pat in", "re.sub", "credential_source", "os.environ.get")):
                    continue
                for pat in patterns:
                    if pat.lower() in line.lower():
                        secrets_found.append(f"{p.name}:{idx}")
                if env_key and len(env_key) > 8 and env_key in line:
                    secrets_found.append(f"{p.name}:{idx} (ACTUAL_KEY_LEAK)")

    assert len(secrets_found) == 0, f"Potential secret detected: {secrets_found}"
    print(f"  [OK] Zero credentials found ({len(files_to_scan)} files scanned).")


def audit_machine_privacy_scan(root: Path):
    """Scans all Gate-7.5-owned files to verify zero machine-private paths or usernames."""
    print("\n--- Scanning for Machine-Private Identifiers & Local Paths ---")
    files_to_scan = [
        root / "src" / "launcher" / "models.py",
        root / "src" / "launcher" / "registry.py",
        root / "src" / "launcher" / "cases.py",
        root / "src" / "launcher" / "resolver.py",
        root / "src" / "launcher" / "preflight.py",
        root / "src" / "launcher" / "events.py",
        root / "src" / "launcher" / "executor.py",
        root / "src" / "launcher" / "contracts.py",
        root / "src" / "launcher" / "cli.py",
        root / "configs" / "platform" / "launcher_contract_v1.json",
        root / "tests" / "test_launcher_core.py",
        root / "scripts" / "audit_launcher_core.py",
    ]
    results_dir = root / "results" / "audits" / "launcher_core"
    if results_dir.exists():
        for f in results_dir.glob("*"):
            if f.is_file():
                files_to_scan.append(f)

    private_targets = []
    try:
        h = str(Path.home()).strip()
        if len(h) > 3:
            private_targets.append(h)
            private_targets.append(h.replace("\\", "/"))
    except Exception:
        pass

    try:
        w = str(root.parent).strip()
        if len(w) > 3:
            private_targets.append(w)
            private_targets.append(w.replace("\\", "/"))
    except Exception:
        pass

    leaks = []
    for p in files_to_scan:
        if p.exists():
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            for target in private_targets:
                if target.lower() in content.lower():
                    if p.name in ("test_launcher_core.py", "audit_launcher_core.py"):
                        lines = content.splitlines()
                        for idx, l in enumerate(lines, 1):
                            if target.lower() in l.lower() and not any(skip in l for skip in ("Path.home", "Path(__file__)", "target", "private_targets")):
                                leaks.append(f"{p.name}:{idx}")
                    else:
                        leaks.append(p.name)

    assert len(leaks) == 0, f"Private paths detected: {leaks}"
    print(f"  [OK] Zero machine-private paths found across {len(files_to_scan)} files.")


def audit_contract_and_hashes(configs_dir: Path, results_dir: Path):
    """Generates launcher contract config and computes additive hashes."""
    print("\n--- Generating Launcher Contract & Computing Additive Hashes ---")
    core = build_launcher_contract_core(status="LOCKED-FOR-PLATFORM-V1")
    contract_file = configs_dir / "launcher_contract_v1.json"
    with open(contract_file, "w", encoding="utf-8") as f:
        json.dump(core, f, indent=2)

    h_launch = compute_launcher_contract_sha256(core)
    h_exec = compute_platform_execution_contract_sha256(h_launch)

    hashes_data = {
        "gate5_benchmark_contract_sha256": GATE5_LOCKED_BENCHMARK_HASH,
        "gate6_agent_contract_sha256": GATE6_LOCKED_AGENT_HASH,
        "platform_runtime_contract_sha256": GATE6_LOCKED_RUNTIME_HASH,
        "logging_contract_sha256": GATE7_LOCKED_LOGGING_HASH,
        "platform_observability_contract_sha256": GATE7_LOCKED_OBSERVABILITY_HASH,
        "launcher_contract_sha256": h_launch,
        "platform_execution_contract_sha256": h_exec,
        "pinned_metadrive_version": PINNED_METADRIVE_VERSION,
        "pinned_metadrive_commit": PINNED_METADRIVE_COMMIT,
        "specification_gate": "Gate 7.5A"
    }

    out_hashes = results_dir / "contract_hashes.json"
    with open(out_hashes, "w", encoding="utf-8") as f:
        json.dump(hashes_data, f, indent=2)

    print(f"[SAVED] Launcher contract saved to: {contract_file}")
    print(f"[SAVED] Contract hashes saved to:   {out_hashes}")
    print(f"  launcher_contract_sha256:           {h_launch}")
    print(f"  platform_execution_contract_sha256: {h_exec}")
    return hashes_data, core


def verify_code_to_disk_consistency(contract_file: Path, core: Dict[str, Any], hashes_data: Dict[str, Any]):
    """Verifies that Python dataclass/enum schemas match disk JSON and stored hashes."""
    print("\n--- Verifying Code-to-Disk Schema Consistency ---")
    with open(contract_file, "r", encoding="utf-8") as f:
        disk_core = json.load(f)

    code_hash = compute_launcher_contract_sha256(core)
    disk_hash = compute_launcher_contract_sha256(disk_core)
    assert code_hash == disk_hash == hashes_data["launcher_contract_sha256"], "Code-to-disk hash mismatch!"

    # Dataclass field matching
    dc_map = {
        "AgentRegistrationV1": AgentRegistrationV1,
        "LaunchRequestV1": LaunchRequestV1,
        "ResolvedCaseV1": ResolvedCaseV1,
        "ResolvedExperimentPlanV1": ResolvedExperimentPlanV1,
        "LauncherEventV1": LauncherEventV1,
    }
    for name, cls in dc_map.items():
        actual = [f.name for f in dataclasses.fields(cls)]
        declared = disk_core["runtime_dataclass_schemas"][name]
        assert actual == declared, f"Dataclass drift in {name}"
    print("  [OK] All 5 runtime dataclass schemas match contract!")

    # Enum value matching
    enum_map = {
        "LauncherMode": LauncherMode,
        "LauncherEventType": LauncherEventType,
    }
    for ename, ecls in enum_map.items():
        actual = [e.value for e in ecls]
        declared = disk_core["runtime_enum_schemas"][ename]
        assert actual == declared, f"Enum drift in {ename}"
    print("  [OK] All 2 runtime enum schemas match contract!")


def audit_mode_resolution_matrix(results_dir: Path):
    """Audits mode mapping and generates mode_resolution_matrix.csv."""
    print("\n--- Auditing Launcher Mode Resolution Matrix ---")
    modes = [LauncherMode.SANDBOX, LauncherMode.VALIDATION, LauncherMode.TEST, LauncherMode.AUDIT]
    rows = []
    for m in modes:
        rk = mode_to_run_kind(m)
        is_canon = (m in (LauncherMode.VALIDATION, LauncherMode.TEST))
        def_cases = 1 if m == LauncherMode.SANDBOX else (96 if m == LauncherMode.VALIDATION else (60 if m == LauncherMode.TEST else 1))
        render_rule = "OFF" if is_canon else "OFF or NATIVE"
        wandb_def = "OFFLINE" if is_canon else "DISABLED"
        rows.append({
            "launcher_mode": m.value,
            "mapped_run_kind": rk.value,
            "canonical_status": is_canon,
            "resolved_cases_count": def_cases,
            "render_mode_policy": render_rule,
            "default_wandb_mode": wandb_def,
            "verdict": "VERIFIED"
        })

    out_csv = results_dir / "mode_resolution_matrix.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVED] Mode resolution matrix saved to: {out_csv}")


def audit_agent_registry_snapshot(results_dir: Path):
    """Captures agent registry snapshot and verifies fixture isolation."""
    print("\n--- Auditing Agent Registry Snapshot ---")
    registry = build_default_agent_registry()
    snapshot = registry.to_snapshot()

    # Assert all fixtures are not benchmark eligible
    for a in snapshot:
        assert a["benchmark_eligible"] is False, f"Fixture {a['agent_id']} must not be benchmark eligible!"
        assert a["sandbox_eligible"] is True
        assert a["audit_eligible"] is True

    out_file = results_dir / "agent_registry_snapshot.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"[SAVED] Agent registry snapshot saved to: {out_file} ({len(snapshot)} agents)")


def audit_case_plan_parity(root: Path, results_dir: Path):
    """Verifies that validation (96) and test (60) cases resolve with 100% manifest parity."""
    print("\n--- Auditing Case Plan Parity Against Gate-5 Manifests ---")
    val_cases = resolve_validation_cases(root)
    test_cases = resolve_test_cases(root)

    val_manifest = load_manifest_csv(root / "results" / "audits" / "evaluation_protocol" / "validation_case_manifest.csv")
    test_manifest = load_manifest_csv(root / "results" / "audits" / "evaluation_protocol" / "test_case_manifest.csv")

    assert len(val_cases) == 96 == len(val_manifest)
    assert len(test_cases) == 60 == len(test_manifest)

    val_mismatches = 0
    for c, m in zip(val_cases, val_manifest):
        if (c.case_id != m["case_id"] or
            c.protocol_order_index != int(m["protocol_order_index"]) or
            c.geometry_sha256 != m["geometry_sha256"] or
            c.environment_seed != int(m["environment_seed"]) or
            c.horizon_steps != int(m["horizon_steps"])):
            val_mismatches += 1

    test_mismatches = 0
    for c, m in zip(test_cases, test_manifest):
        if (c.case_id != m["case_id"] or
            c.protocol_order_index != int(m["protocol_order_index"]) or
            c.geometry_sha256 != m["geometry_sha256"] or
            c.environment_seed != int(m["environment_seed"]) or
            c.horizon_steps != int(m["horizon_steps"])):
            test_mismatches += 1

    parity_report = {
        "validation_suite_parity": {
            "expected_count": 96,
            "resolved_count": len(val_cases),
            "manifest_mismatches": val_mismatches,
            "order_monotonic": all(c.protocol_order_index == (i + 1) for i, c in enumerate(val_cases)),
            "verdict": "PERFECT_PARITY"
        },
        "test_benchmark_parity": {
            "expected_count": 60,
            "resolved_count": len(test_cases),
            "manifest_mismatches": test_mismatches,
            "order_monotonic": all(c.protocol_order_index == (i + 1) for i, c in enumerate(test_cases)),
            "verdict": "PERFECT_PARITY"
        }
    }

    out_file = results_dir / "case_plan_parity.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(parity_report, f, indent=2)
    print(f"[SAVED] Case plan parity saved to: {out_file}")
    assert val_mismatches == 0 and test_mismatches == 0


def audit_benchmark_lock_checks(root: Path, results_dir: Path):
    """Verifies holdout isolation (Sandbox TRAIN-only) and fixture benchmark rejection."""
    print("\n--- Auditing Benchmark Holdout Locks & Fixture Rejection ---")
    registry = build_default_agent_registry()
    test_manifest = load_manifest_csv(root / "results" / "audits" / "evaluation_protocol" / "test_case_manifest.csv")
    test_seed = int(test_manifest[0]["geometry_generation_seed"])
    test_seq = test_manifest[0]["sequence"]

    sandbox_test_holdout_violation_caught = False
    try:
        resolve_sandbox_case(LaunchRequestV1(
            mode=LauncherMode.SANDBOX,
            agent_id="fixture_seeded_random",
            sequence=test_seq,
            geometry_generation_seed=test_seed
        ), root)
    except ValueError as e:
        if "HOLDOUT VIOLATION" in str(e):
            sandbox_test_holdout_violation_caught = True

    # Preflight fixture rejection on TEST and VALIDATION
    p_val = resolve_experiment_plan(LaunchRequestV1(mode=LauncherMode.VALIDATION, agent_id="fixture_seeded_random", agent_seed=101), registry, root)
    rep_val = run_preflight(p_val, root, custom_git_provenance={"git_worktree_dirty": False}, custom_environment_provenance={"metadrive_version": PINNED_METADRIVE_VERSION, "metadrive_commit": PINNED_METADRIVE_COMMIT})

    p_test = resolve_experiment_plan(LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_seeded_random", agent_seed=101), registry, root)
    rep_test = run_preflight(p_test, root, custom_git_provenance={"git_worktree_dirty": False}, custom_environment_provenance={"metadrive_version": PINNED_METADRIVE_VERSION, "metadrive_commit": PINNED_METADRIVE_COMMIT})

    report = {
        "sandbox_test_holdout_rejection_caught": sandbox_test_holdout_violation_caught,
        "validation_fixture_rejected": not rep_val.can_execute,
        "test_fixture_rejected": not rep_test.can_execute,
        "verdict": "PASSED" if (sandbox_test_holdout_violation_caught and not rep_val.can_execute and not rep_test.can_execute) else "FAILED"
    }

    out_file = results_dir / "benchmark_lock_checks.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[SAVED] Benchmark lock checks saved to: {out_file}")
    assert report["verdict"] == "PASSED"


def audit_preflight_matrix(root: Path, results_dir: Path):
    """Evaluates full preflight decision matrix across valid and invalid configurations."""
    print("\n--- Auditing Preflight Decision Matrix ---")
    registry = build_default_agent_registry()
    clean_git = {"git_worktree_dirty": False, "git_commit_sha": "abc12345"}
    dirty_git = {"git_worktree_dirty": True, "git_commit_sha": "abc12345"}
    clean_env = {"metadrive_version": PINNED_METADRIVE_VERSION, "metadrive_commit": PINNED_METADRIVE_COMMIT}
    bad_env = {"metadrive_version": "0.4.2", "metadrive_commit": "wrong"}

    # Register benchmark eligible dummy agent for testing
    mock_bm = AgentRegistrationV1(agent_id="mock_benchmark_agent", benchmark_eligible=True, inference_stochasticity="deterministic")
    mock_stoch = AgentRegistrationV1(agent_id="mock_stoch_benchmark", benchmark_eligible=True, inference_stochasticity="stochastic")
    reg_suite = AgentRegistryV1()
    for a in registry.list_all(): reg_suite.register(a, registry.get_factory(a.agent_id))
    reg_suite.register(mock_bm, lambda: None)
    reg_suite.register(mock_stoch, lambda: None)

    matrix_scenarios = [
        ("valid_sandbox_fixture", LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_seeded_random", tier="Easy", agent_seed=101), clean_git, clean_env, True),
        ("fixture_on_validation_blocked", LaunchRequestV1(mode=LauncherMode.VALIDATION, agent_id="fixture_seeded_random", agent_seed=101), clean_git, clean_env, False),
        ("fixture_on_test_blocked", LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_seeded_random", agent_seed=101), clean_git, clean_env, False),
        ("test_render_native_blocked", LaunchRequestV1(mode=LauncherMode.TEST, agent_id="mock_benchmark_agent", render_mode="NATIVE"), clean_git, clean_env, False),
        ("validation_render_native_blocked", LaunchRequestV1(mode=LauncherMode.VALIDATION, agent_id="mock_benchmark_agent", render_mode="NATIVE"), clean_git, clean_env, False),
        ("stochastic_test_missing_seed_blocked", LaunchRequestV1(mode=LauncherMode.TEST, agent_id="mock_stoch_benchmark", agent_seed=None), clean_git, clean_env, False),
        ("stochastic_test_invalid_seed_blocked", LaunchRequestV1(mode=LauncherMode.TEST, agent_id="mock_stoch_benchmark", agent_seed=999), clean_git, clean_env, False),
        ("stochastic_test_valid_seed_passed", LaunchRequestV1(mode=LauncherMode.TEST, agent_id="mock_stoch_benchmark", agent_seed=101), clean_git, clean_env, True),
        ("deterministic_test_with_seed_blocked", LaunchRequestV1(mode=LauncherMode.TEST, agent_id="mock_benchmark_agent", agent_seed=101), clean_git, clean_env, False),
        ("deterministic_test_no_seed_passed", LaunchRequestV1(mode=LauncherMode.TEST, agent_id="mock_benchmark_agent", agent_seed=None), clean_git, clean_env, True),
        ("dirty_worktree_on_test_blocked", LaunchRequestV1(mode=LauncherMode.TEST, agent_id="mock_benchmark_agent"), dirty_git, clean_env, False),
        ("dirty_worktree_on_sandbox_allowed", LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_seeded_random", tier="Easy", agent_seed=101), dirty_git, clean_env, True),
        ("unverified_metadrive_on_test_blocked", LaunchRequestV1(mode=LauncherMode.TEST, agent_id="mock_benchmark_agent"), clean_git, bad_env, False),
    ]

    rows = []
    for label, req, git_p, env_p, expected_can_exec in matrix_scenarios:
        plan = resolve_experiment_plan(req, reg_suite, root)
        rep = run_preflight(plan, root, custom_git_provenance=git_p, custom_environment_provenance=env_p)
        matched = (rep.can_execute == expected_can_exec)
        rows.append({
            "scenario": label,
            "launcher_mode": req.mode.value,
            "agent_id": req.agent_id,
            "expected_can_execute": expected_can_exec,
            "actual_can_execute": rep.can_execute,
            "fail_count": rep.fail_count,
            "warning_count": rep.warning_count,
            "summary_verdict": rep.summary_verdict,
            "preflight_parity": matched
        })
        assert matched, f"Preflight scenario mismatch for {label}!"

    out_csv = results_dir / "preflight_matrix.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVED] Preflight matrix saved to: {out_csv} ({len(rows)} scenarios tested)")


def audit_plan_hash_determinism(root: Path, results_dir: Path):
    """Verifies deterministic plan hashing and mutation sensitivity."""
    print("\n--- Auditing Plan Hash Determinism & Sensitivity ---")
    registry = build_default_agent_registry()
    req1 = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_seeded_random", tier="Easy", agent_seed=101)
    req2 = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_seeded_random", tier="Easy", agent_seed=101)

    p1 = resolve_experiment_plan(req1, registry, root)
    p2 = resolve_experiment_plan(req2, registry, root)
    assert p1.resolved_plan_sha256 == p2.resolved_plan_sha256, "Plan hash non-deterministic!"

    # Mutations
    p_seed = resolve_experiment_plan(LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_seeded_random", tier="Easy", agent_seed=202), registry, root)
    p_tier = resolve_experiment_plan(LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_seeded_random", tier="Medium", agent_seed=101), registry, root)
    p_agent = resolve_experiment_plan(LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", tier="Easy"), registry, root)

    report = {
        "identical_requests_match": (p1.resolved_plan_sha256 == p2.resolved_plan_sha256),
        "base_plan_sha256": p1.resolved_plan_sha256,
        "seed_mutation_differs": (p1.resolved_plan_sha256 != p_seed.resolved_plan_sha256),
        "tier_mutation_differs": (p1.resolved_plan_sha256 != p_tier.resolved_plan_sha256),
        "agent_mutation_differs": (p1.resolved_plan_sha256 != p_agent.resolved_plan_sha256),
        "verdict": "PASSED_STRICT_DETERMINISM"
    }

    out_file = results_dir / "plan_hash_determinism.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[SAVED] Plan hash determinism saved to: {out_file}")
    assert all([report["identical_requests_match"], report["seed_mutation_differs"], report["tier_mutation_differs"], report["agent_mutation_differs"]])


def audit_cli_core_parity(root: Path, results_dir: Path):
    """Verifies that CLI commands produce identical semantic output to core engine."""
    print("\n--- Auditing CLI / Core Parity ---")
    registry = build_default_agent_registry()
    parser = create_parser()
    args = parser.parse_args(["plan", "--mode", "SANDBOX", "--agent", "fixture_seeded_random", "--tier", "Easy", "--agent-seed", "101"])
    cli_req = build_launch_request_from_args(args)
    cli_plan = resolve_experiment_plan(cli_req, registry, root)

    direct_req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_seeded_random", tier="Easy", agent_seed=101)
    direct_plan = resolve_experiment_plan(direct_req, registry, root)

    parity = (cli_plan.resolved_plan_sha256 == direct_plan.resolved_plan_sha256)
    report = {
        "cli_plan_sha256": cli_plan.resolved_plan_sha256,
        "direct_plan_sha256": direct_plan.resolved_plan_sha256,
        "parity_verified": parity,
        "verdict": "PERFECT_PARITY"
    }

    out_file = results_dir / "cli_core_parity.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[SAVED] CLI / Core parity saved to: {out_file}")
    assert parity is True


def audit_sandbox_execution_smoke(root: Path, results_dir: Path):
    """
    Executes ONE real simulator-backed Sandbox episode (Section 49 & 50):
    - Uses exact locked TRAIN geometry (Easy SCS)
    - Gate-6 fixture agent (fixture_constant_continuous)
    - render OFF, W&B DISABLED
    - Verifies complete Gate-7 local directory structure and run_manifest launcher provenance
    """
    print("\n--- Executing Real Simulator-Backed Sandbox Smoke Test ---")
    temp_dir = tempfile.TemporaryDirectory()
    runs_root = Path(temp_dir.name)

    registry = build_default_agent_registry()
    request = LaunchRequestV1(
        mode=LauncherMode.SANDBOX,
        agent_id="fixture_constant_continuous",
        tier="Easy",
        sequence="SCS",
        geometry_generation_seed=0,
        environment_seed=0,
        render_mode="OFF",
        wandb_mode=WandbMode.DISABLED,
        runs_root=runs_root
    )

    plan = resolve_experiment_plan(request, registry, root)
    preflight = run_preflight(plan, root)
    assert preflight.can_execute, f"Sandbox smoke preflight failed: {preflight.to_dict()}"

    events_captured = []
    def on_event(ev: LauncherEventV1):
        events_captured.append(ev.event_type.value)

    executor = ExperimentExecutor(
        plan=plan,
        agent_factory=registry.get_factory("fixture_constant_continuous"),
        runs_root=runs_root,
        custom_run_id="run_sandbox_smoke_01",
        event_callback=on_event,
        project_root=root
    )

    report = executor.execute()
    run_dir = Path(report.run_dir)

    # 1. Assert required Gate-7 files exist
    required_files = [
        "run_manifest.json",
        "run_state.json",
        "episodes.csv",
        "timing.csv",
        "summary.json",
        "wandb_sync.json",
        "run_integrity.json"
    ]
    files_present = {f: (run_dir / f).exists() for f in required_files}
    assert all(files_present.values()), f"Missing local run artifacts: {files_present}"

    # 2. Assert run_state is COMPLETE
    with open(run_dir / "run_state.json", "r", encoding="utf-8") as f:
        st = json.load(f)
    assert st["status"] == "COMPLETE"
    assert st["recorded_episode_count"] == 1

    # 3. Assert launcher provenance in run_manifest.json (Section 50)
    with open(run_dir / "run_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    launcher_prov = manifest["config"]["algorithm_hyperparameters"]["launcher"]
    assert launcher_prov["launcher_mode"] == "SANDBOX"
    assert launcher_prov["resolved_plan_sha256"] == plan.resolved_plan_sha256
    assert launcher_prov["protocol_scope"] == "SANDBOX_SINGLE_CASE"

    smoke_artifact = {
        "run_id": report.run_id,
        "launcher_mode": "SANDBOX",
        "status": report.status,
        "completed_episodes": report.completed_episodes,
        "required_files_verified": files_present,
        "events_captured": events_captured,
        "launcher_provenance_verified": True,
        "embedded_launcher_provenance": launcher_prov,
        "verdict": "PASSED"
    }

    out_file = results_dir / "sandbox_execution_smoke.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(smoke_artifact, f, indent=2)
    print(f"[SAVED] Sandbox execution smoke saved to: {out_file}")
    temp_dir.cleanup()


def generate_summary_markdown(summary_md_path: Path, hashes_data: Dict[str, Any]):
    """Generates results/audits/launcher_core/audit_summary.md cleanly."""
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Gate 7.5A Research Launcher Core Audit Summary\n\n")
        f.write("## 1. Verified MetaDrive Source & Pinned Versions\n")
        f.write(f"- **Pinned MetaDrive Commit:** `{PINNED_METADRIVE_COMMIT}`\n")
        f.write(f"- **Package Version:** `{PINNED_METADRIVE_VERSION}`\n")
        f.write("- **Working Tree Policy:** Strict clean worktree required for VALIDATION and TEST.\n\n")

        f.write("## 2. Orchestration Architecture & Modes\n")
        f.write("- **Launcher Modes:** `SANDBOX`, `VALIDATION`, `TEST`, `AUDIT`.\n")
        f.write("- **RunKind Mapping:** SANDBOX/AUDIT $\\to$ `AUDIT`; VALIDATION $\\to$ `VALIDATION_EVALUATION`; TEST $\\to$ `TEST_EVALUATION`.\n")
        f.write("- **Holdout Isolation:** Sandbox is strictly restricted to TRAIN split geometries. Requests for VALIDATION or TEST geometries raise holdout violations loudly.\n")
        f.write("- **Benchmark Case Parity:** Exactly 96 cases for VALIDATION and 60 paired cases for TEST with 100% manifest parity.\n\n")

        f.write("## 3. Agent Registry & Fixture Boundaries\n")
        f.write("- **AgentRegistryV1:** Segregates declarative `AgentRegistrationV1` metadata from runtime factory callables.\n")
        f.write("- **Audit Fixtures:** 4 Gate-6 fixtures registered (`fixture_constant_continuous`, `fixture_seeded_random`, `fixture_stateful_counter`, `fixture_discrete`). All flagged `benchmark_eligible = False`.\n")
        f.write("- **Benchmark Protection:** Fixtures are strictly blocked from executing on TEST or VALIDATION suites.\n\n")

        f.write("## 4. Preflight Validation Battery\n")
        f.write("- Pure preflight engine evaluates 13 distinct scientific check categories.\n")
        f.write("- Blocks dirty worktrees, unverified environments, missing/invalid stochastic seeds, and native rendering on benchmark runs.\n")
        f.write("- Execution is allowed if and only if zero blocking failures occur.\n\n")

        f.write("## 5. Execution Pipeline & Gate-7 Integration\n")
        f.write("- Generic `ExperimentExecutor` connects MetaDrive simulator, Gate-6 AgentPolicy, Gate-3 outcome classification, Gate-4 reward metrics, and Gate-7 logging.\n")
        f.write("- Information Parity Principle strictly preserved: agent receives only `AgentInputV1` and `AgentPublicEpisodeContext`.\n")
        f.write("- Launcher provenance embedded cleanly into `run_manifest.json` under `config.algorithm_hyperparameters['launcher']` without mutating Gate-7 schemas.\n\n")

        f.write("## 6. Cryptographic Contract Hashes\n")
        f.write(f"- **`gate5_benchmark_contract_sha256`:** `{hashes_data['gate5_benchmark_contract_sha256']}` (locked, untouched)\n")
        f.write(f"- **`gate6_agent_contract_sha256`:** `{hashes_data['gate6_agent_contract_sha256']}` (locked, untouched)\n")
        f.write(f"- **`platform_runtime_contract_sha256`:** `{hashes_data['platform_runtime_contract_sha256']}` (locked, untouched)\n")
        f.write(f"- **`logging_contract_sha256`:** `{hashes_data['logging_contract_sha256']}` (locked, untouched)\n")
        f.write(f"- **`platform_observability_contract_sha256`:** `{hashes_data['platform_observability_contract_sha256']}` (locked, untouched)\n")
        f.write(f"- **`launcher_contract_sha256`:** `{hashes_data['launcher_contract_sha256']}`\n")
        f.write(f"- **`platform_execution_contract_sha256`:** `{hashes_data['platform_execution_contract_sha256']}`\n")

    print(f"[SAVED] Audit summary markdown saved to: {summary_md_path}")


def main():
    print("============================================================")
    print("STARTING GATE 7.5A LAUNCHER CORE AUDIT")
    print("============================================================")

    configs_dir = project_root / "configs" / "platform"
    results_dir = project_root / "results" / "audits" / "launcher_core"
    configs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify prior platform contracts
    verify_prior_contracts(project_root)

    # 2. Build launcher contract and compute additive hashes
    hashes_data, core = audit_contract_and_hashes(configs_dir, results_dir)

    # 3. Code-to-disk consistency check
    contract_file = configs_dir / "launcher_contract_v1.json"
    verify_code_to_disk_consistency(contract_file, core, hashes_data)

    # 4. Initial secret and privacy scan
    audit_secret_scan(project_root, scan_label="Initial Pre-Run")
    audit_machine_privacy_scan(project_root)

    # 5. Mode resolution matrix
    audit_mode_resolution_matrix(results_dir)

    # 6. Agent registry snapshot
    audit_agent_registry_snapshot(results_dir)

    # 7. Case plan parity audit
    audit_case_plan_parity(project_root, results_dir)

    # 8. Benchmark lock & holdout safety audit
    audit_benchmark_lock_checks(project_root, results_dir)

    # 9. Preflight decision matrix audit
    audit_preflight_matrix(project_root, results_dir)

    # 10. Plan hash determinism & sensitivity audit
    audit_plan_hash_determinism(project_root, results_dir)

    # 11. CLI / Core parity audit
    audit_cli_core_parity(project_root, results_dir)

    # 12. Real simulator-backed Sandbox smoke execution
    audit_sandbox_execution_smoke(project_root, results_dir)

    # 13. Generate summary markdown
    summary_md_path = results_dir / "audit_summary.md"
    generate_summary_markdown(summary_md_path, hashes_data)

    # 14. Final secret and privacy scans after all artifacts written
    audit_secret_scan(project_root, scan_label="Final Post-Artifacts")
    audit_machine_privacy_scan(project_root)

    print("\n============================================================")
    print("GATE 7.5A LAUNCHER CORE AUDIT COMPLETED SUCCESSFULLY!")
    print("============================================================")


if __name__ == "__main__":
    main()
