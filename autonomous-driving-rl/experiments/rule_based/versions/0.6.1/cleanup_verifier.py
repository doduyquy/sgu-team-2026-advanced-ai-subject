"""Verify maintenance evidence and freeze 0.6.1 without changing old records."""
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.agents.rule_based_agent import Stage1RuleBasedAgent
from src.launcher.registry import build_default_agent_registry
from src.platform.agent import build_agent_contract_core
from src.logging.agent_experiment import save_agent_provenance, sha256_file, write_json

RUN = Path(__file__).parent
OUT = ROOT / "results/audits/source_cleanup/v0.6.1"
VERSION = ROOT / "experiments/rule_based/versions/0.6.1"
read = lambda p: json.loads(p.read_text(encoding="utf8"))
if OUT.exists() or VERSION.exists():
    raise FileExistsError("Maintenance evidence/snapshot already exists")
checks = read(RUN / "final_checks/summary.json")
assert checks["status"] == "PASSED" and checks["total_tests"] == 547
request = read(RUN / "final_checks/request.json")
assert all(sha256_file(ROOT / name) == expected for name, expected in request["source_sha256"].items())
assert sha256_file(ROOT / "scripts/run_tests.py") == request["runner_sha256"]
replay = read(RUN / "replay_01/state.json")
replay_rows = read(RUN / "replay_01/results.json")
assert replay["status"] == "PASSED" and replay["total_steps"] == 1992
assert all(not r["decision_mismatches"] and not r["action_mismatches"] for r in replay_rows)
protected = read(RUN / "protected_snapshots.json")
assert all(sha256_file(ROOT / path) == expected for path, expected in protected.items())
for name in protected:
    snapshot = ROOT / name
    record = read(snapshot)
    hashes = record.get("source_sha256", record.get("source_origin", {}).get("policy_input_helper_sha256", {}))
    assert all(sha256_file(snapshot.parent / path) == expected for path, expected in hashes.items())
retired = read(RUN / "retired_paths.json")
assert all(not (ROOT / row["original_path"]).exists() and sha256_file(ROOT / row["archive_path"]) == row["sha256"] for row in retired)
cache = read(RUN / "cache_cleanup.json")
assert cache["status"] == "COMPLETED"
assert all(not list((ROOT / folder).rglob("__pycache__")) for folder in ("src", "tests", "scripts"))
assert not (ROOT / "src/visualization/assets").exists()
parent = read(ROOT / "experiments/rule_based/versions/0.6.0/snapshot.json")
agent = Stage1RuleBasedAgent()
assert agent.descriptor.agent_version == "0.6.1"
assert agent.export_config() == parent["configuration"]
assert build_default_agent_registry().get("stage1_rule_based").agent_version == "0.6.1"
assert build_agent_contract_core() == read(ROOT / "configs/platform/agent_contract_v1.json")
assert sha256_file(ROOT / "src/platform/action_adapter.py") == parent["source_sha256"]["action_adapter.py"]
smoke_dir = ROOT / "runs/rule_based_v061_cleanup_smoke_20261031"
smoke = read(smoke_dir / "summary.json")
assert read(smoke_dir / "run_state.json")["status"] == "COMPLETE"
assert smoke["overall_metrics"]["clean_success_rate"] == 1.0 and smoke["technical_failure_count"] == 0
with gzip.open(smoke_dir / "agent_decisions.jsonl.gz", "rt", encoding="utf8") as stream:
    smoke_steps = sum(1 for line in stream)
assert smoke_steps == smoke["overall_metrics"]["mean_episode_steps"]
before = read(RUN / "source_before.json")
paths = sorted(set(ROOT.glob("*.py")) | set((ROOT / "src").rglob("*.py")) | set((ROOT / "scripts").glob("*.py")))
after = {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}
diff = {"removed": sorted(set(before)-set(after)), "added": sorted(set(after)-set(before)),
        "modified": sorted(name for name in before.keys() & after.keys() if before[name] != after[name])}
planned = read(RUN / "unused_imports.json")
actual = [row for row in planned if Path(row["path"]).name != "action_adapter.py"]
assert sum(len(row["removed"]) for row in actual) == 79
OUT.mkdir(parents=True, exist_ok=False)
write_json(OUT / "source_before.json", before, exclusive=True)
write_json(OUT / "source_after.json", after, exclusive=True)
write_json(OUT / "source_diff.json", diff, exclusive=True)
write_json(OUT / "removed_imports.json", actual, exclusive=True)
write_json(OUT / "retired_sources.json", retired, exclusive=True)
write_json(OUT / "cache_cleanup.json", cache, exclusive=True)
write_json(OUT / "decision_replay.json", {"scope": replay["scope"], "state": replay, "traces": replay_rows}, exclusive=True)
write_json(OUT / "test_verification.json", {"summary": checks, "request": request}, exclusive=True)
attempts = [{"attempt": p.parent.name, "state": read(p)} for p in sorted(RUN.glob("checks_*/state.json"))]
write_json(OUT / "test_attempts.json", attempts, exclusive=True)
summary = {"version": "0.6.1", "kind": "SOURCE_MAINTENANCE_NO_NEW_BENCHMARK_SCORE", "status": "VERIFIED",
           "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "retired_scripts": len(retired),
           "active_python_files_before": len(before), "active_python_files_after": len(after),
           "unused_imports_removed": 79, "import_cleanup_files": len({row["path"] for row in actual}),
           "generated_textures_removed_from_src": cache["removed_textures"], "cache_directories_removed": cache["removed_cache_directories"],
           "tests_passed": checks["total_tests"], "decision_replay_steps": replay["total_steps"],
           "decision_or_action_mismatches": 0, "cli_smoke_steps": smoke_steps, "cli_clean_success": True,
           "protected_prior_snapshots": len(protected), "policy_configuration_unchanged": True,
           "contract_json_unchanged": True, "pinned_adapter_bytes_unchanged": True,
           "raw_cleanup_run": str(RUN.relative_to(ROOT)), "cli_run": smoke_dir.name,
           "limits": "P4/P5/benchmark status unchanged; 16-case driving evidence belongs to 0.6.0, not a newly scored 0.6.1 sweep"}
write_json(OUT / "summary.json", summary, exclusive=True)
after_provenance = RUN / "after"
after_provenance.mkdir(exist_ok=False)
save_agent_provenance(after_provenance, agent)
files = {"policy.py": ROOT / "src/agents/rule_based_agent.py", "agent_input.py": ROOT / "src/platform/agent_input.py",
         "platform_agent.py": ROOT / "src/platform/agent.py", "platform_init.py": ROOT / "src/platform/__init__.py",
         "agent_registry.py": ROOT / "src/launcher/registry.py", "action_adapter.py": ROOT / "src/platform/action_adapter.py",
         "agent_contract_v1.json": ROOT / "configs/platform/agent_contract_v1.json",
         "agent_experiment.py": ROOT / "src/logging/agent_experiment.py", "test_runner.py": ROOT / "scripts/run_tests.py",
         "verified_runner.py": ROOT / "scripts/run_rule_based_experiment.py", "replay_runner.py": ROOT / "scripts/replay_rule_based_trace.py",
         "cleanup_verifier.py": Path(__file__)}
for name in ("traffic", "memory", "junction", "lane_change"):
    files[f"rule_based_{name}.py"] = ROOT / f"src/agents/rule_based_{name}.py"
flat = set(files.values())
for name in diff["modified"]:
    source = ROOT / name
    if source not in flat:
        files["maintenance_sources/" + source.relative_to(ROOT).as_posix()] = source
for name in ("test_finish_visual", "test_source_layout", "test_workbench", "test_workbench_hardening", "test_workbench_results"):
    files[f"maintenance_sources/tests/{name}.py"] = ROOT / f"tests/{name}.py"
VERSION.mkdir(exist_ok=False)
hashes = {}
for name, source in files.items():
    destination = VERSION / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(source.read_bytes())
    hashes[name] = sha256_file(destination)
write_json(VERSION / "snapshot.json", {"version": "0.6.1", "parent_version": "0.6.0",
    "status": "MAINTENANCE_RELEASE_SANDBOX_AUDIT_ONLY", "snapshot_recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "descriptor": agent.descriptor.to_dict(), "configuration": agent.export_config(),
    "configuration_sha256": parent["configuration_sha256"], "source_sha256": hashes,
    "tests_passed": checks["total_tests"], "replay_steps": replay["total_steps"], "replay_exact_match": True,
    "evidence_summary": str((OUT / "summary.json").relative_to(ROOT)), "verified_runs": [RUN.name, smoke_dir.name],
    "full_agent_plan_complete": False, "benchmark_eligible": False, "contract_spec_version": "1.3.0",
    "contract_status": "SANDBOX-EXTENSION-PENDING-AUDIT", "prior_driving_evidence_version": "0.6.0",
    "known_limits": parent["known_limits"], "note": summary["limits"]}, exclusive=True)
state = read(RUN / "run_state.json")
state.update(status="COMPLETED", finished_at_utc=datetime.now(timezone.utc).isoformat(), version_after="0.6.1",
             evidence_summary=str((OUT / "summary.json").relative_to(ROOT)))
write_json(RUN / "source_after.json", after, exclusive=True)
write_json(RUN / "run_state.json", state)
print(json.dumps(summary, indent=2))
