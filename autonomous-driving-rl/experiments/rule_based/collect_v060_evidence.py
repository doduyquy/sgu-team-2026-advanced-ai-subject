"""Verify immutable P4 evidence, curate results, optionally freeze 0.6.0.

Run from project root after the named attempts have completed. Existing output
and version folders are never overwritten. No driving run is launched here.
"""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.summarize_rule_based_experiment import main as compare
from src.logging.agent_experiment import sha256_file, write_json
from src.agents.rule_based_agent import Stage1RuleBasedAgent

GROUPS = ("roads9", "extreme_cr", "extreme_sctx", "extreme_scxo", "env1")
PROBE = "rule_based_v060_probe10_20261030"
CLI = "rule_based_v060_cli_smoke02_20261030"
SCENARIOS = ("safe_overtake", "fast_rear", "mandatory_route", "mandatory_left", "precommit_abort", "postcommit_front_block")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    runs = {group: ROOT / "runs" / f"rule_based_v060_accept04_{group}_20261030" for group in GROUPS}
    if any(read(folder / "run_state.json")["status"] != "COMPLETED" for folder in runs.values()):
        raise ValueError("Final driving runs are not all COMPLETED")
    if read(ROOT / "runs" / PROBE / "run_state.json")["status"] != "COMPLETED":
        raise ValueError("Physical probe is not complete")
    if read(ROOT / "runs" / CLI / "run_state.json")["status"] != "COMPLETE":
        raise ValueError("Final official launcher smoke is not complete")
    cli_summary = read(ROOT / "runs" / CLI / "summary.json")
    if cli_summary["overall_metrics"]["clean_success_rate"] != 1.0 or cli_summary["technical_failure_count"] != 0:
        raise ValueError("Final official launcher smoke did not succeed cleanly")
    physical = read(ROOT / "runs" / PROBE / "summary.json")
    if len(physical) != 6 or not all(row["passed"] for row in physical):
        raise ValueError("Physical scenario expectations failed")
    checks = ROOT / "results/audits/rule_based/v0.6.0_checks"
    replays = [read(checks / f"accept04_replay_{name}.json") for name in SCENARIOS]
    if any(not row["completed_replay"] or row["decision_mismatches"] or row["action_mismatches"] for row in replays):
        raise ValueError("Physical replay failed")
    for name, replay in zip(SCENARIOS, replays):
        scenario_dir = ROOT / "runs" / PROBE / name
        if Path(replay["variant_dir"]).resolve() != scenario_dir or replay["steps"] != read(scenario_dir / "result.json")["steps"]:
            raise ValueError("Physical replay belongs to a different source attempt")
        if replay["trace_sha256"] != sha256_file(scenario_dir / "decisions.jsonl.gz") or replay["source_sha256"] != read(scenario_dir / "agent_provenance.json")["source_sha256"]:
            raise ValueError("Physical replay/source hash mismatch")
    test_log = ROOT / "runs/v060_integrated_checks_05.log"
    test_bytes = test_log.read_bytes()
    test_encoding = "utf-16" if test_bytes.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    test_text = test_bytes.decode(test_encoding, errors="replace")
    match = re.search(r"Ran (\d+) tests", test_text)
    if match is None:
        raise ValueError("Cannot parse unittest result from console artifact")
    count = int(match[1])
    if count != 279 or not test_text.rstrip().endswith("OK"):
        raise ValueError("Final integrated checks are not 279/279 passing")
    files = {"policy.py": ROOT / "src/agents/rule_based_agent.py",
             "agent_input.py": ROOT / "src/platform/agent_input.py",
             "platform_agent.py": ROOT / "src/platform/agent.py",
             "platform_init.py": ROOT / "src/platform/__init__.py",
             "agent_registry.py": ROOT / "src/launcher/registry.py",
             "agent_contract_v1.json": ROOT / "configs/platform/agent_contract_v1.json",
             "action_adapter.py": ROOT / "src/platform/action_adapter.py",
             "agent_experiment.py": ROOT / "src/logging/agent_experiment.py",
             "verified_runner.py": ROOT / "scripts/run_rule_based_experiment.py",
             "physical_probe_runner.py": ROOT / "scripts/probe_rule_based_lane_change.py",
             "replay_runner.py": ROOT / "scripts/replay_rule_based_trace.py",
             "summarize_runner.py": ROOT / "scripts/summarize_rule_based_experiment.py",
             "evidence_collector.py": Path(__file__)}
    for name in ("traffic", "memory", "junction", "lane_change"):
        files[f"rule_based_{name}.py"] = ROOT / f"src/agents/rule_based_{name}.py"
    hashes = {name: sha256_file(path) for name, path in files.items()}
    expected_configuration = Stage1RuleBasedAgent().export_config()
    expected_config_hash = hashlib.sha256(json.dumps(expected_configuration, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    sources_to_verify = list(runs.values()) + [ROOT / "runs" / PROBE / name for name in SCENARIOS] + [ROOT / "runs" / CLI]
    for folder in sources_to_verify:
        provenance_path = folder / "current/agent_provenance.json" if folder in runs.values() else folder / "agent_provenance.json"
        provenance = read(provenance_path)
        if provenance["configuration"] != expected_configuration or provenance["descriptor"]["agent_version"] != "0.6.0":
            raise ValueError("Final policy version/configuration mismatch: " + str(folder))
        for name in ("policy.py", "agent_input.py", "rule_based_traffic.py", "rule_based_memory.py", "rule_based_junction.py", "rule_based_lane_change.py"):
            key = name if name in provenance["source_sha256"] else "agents/" + name
            if hashes[name] != provenance["source_sha256"][key]:
                raise ValueError(f"Live/final-run source mismatch: {folder}, {name}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    comparisons = []
    rows, lane_events = [], []
    for group, folder in runs.items():
        baseline = ROOT / "runs" / ("rule_based_v050_acceptance_env1_20261030" if group == "env1" else "rule_based_v050_acceptance_family12_20261030")
        destination = args.output_dir / group
        compare(["--baseline-run", str(baseline), "--baseline-variant", "current",
                 "--candidate-run", str(folder), "--output-dir", str(destination)])
        comparison = read(destination / "comparison.json")
        comparisons.append(comparison)
        with (folder / "episodes.csv").open(encoding="utf8", newline="") as stream:
            rows.extend(list(csv.DictReader(stream)))
        for episode in sorted((folder / "current").glob("episode_*")):
            previous, starts, completions, veto = "KEEP_LANE", [], 0, Counter()
            phase_counts = Counter()
            with gzip.open(episode / "decisions.jsonl.gz", "rt", encoding="utf8") as stream:
                for line in stream:
                    trace = json.loads(line)
                    lane = trace["decision"]["diagnostics"].get("lane_change", {})
                    phase = lane.get("phase", "KEEP_LANE")
                    phase_counts[phase] += 1
                    veto.update(lane.get("veto", ()))
                    if previous == "KEEP_LANE" and phase in {"PREPARE_CHANGE", "CHANGE_LANE"}:
                        starts.append({"step": trace["step_index"], "reason": lane["reason"],
                                       "source_id": lane["source_id"], "target_id": lane["target_id"]})
                    completions += bool(lane.get("completed"))
                    previous = phase
            lane_events.append({"case_id": read(episode / "result.json")["case_id"], "starts": starts,
                                "completions": completions, "phase_counts": dict(phase_counts), "veto_counts": dict(veto)})
    if len(rows) != 16 or len({row["case_id"] for row in rows}) != 16:
        raise ValueError("Final coverage must contain exactly 16 unique cases")
    summary = {"noncanonical": True, "split": "TRAIN", "unique_cases": len(rows),
               "baseline_successes": sum(item["baseline_successes"] for item in comparisons),
               "candidate_successes": sum(item["candidate_successes"] for item in comparisons),
               "candidate_crashes": sum(row["crash"] == "True" for row in rows),
               "candidate_out_of_road": sum(row["out_of_road"] == "True" for row in rows),
               "candidate_solid_boundary_contacts": sum(row["boundary_contact"] == "True" for row in rows),
               "timeouts": [row["case_id"] for row in rows if row["terminal_reason"] == "TIMEOUT"],
               "lost_successes": sum((item["lost_success_cases"] for item in comparisons), []),
               "new_successes": sum((item["new_success_cases"] for item in comparisons), []),
               "physical_scenarios_passed": len(physical), "physical_replay_steps": sum(row["steps"] for row in replays),
               "tests_passed": count, "source_sha256": hashes,
               "note": "Policy + public extractor change; same locked physical cases/config; reuse original verified v050 current-variant baseline; no canonical benchmark claim"}
    write_json(args.output_dir / "combined_summary.json", summary, exclusive=True)
    write_json(args.output_dir / "lane_change_events.json", lane_events, exclusive=True)
    write_json(args.output_dir / "physical_scenarios.json", physical, exclusive=True)
    write_json(args.output_dir / "test_verification.json", {"tests": count, "passed": True,
        "console_artifact": str(test_log.relative_to(ROOT)), "console_sha256": sha256_file(test_log), "console_encoding": test_encoding,
        "source_sha256_at_collection": hashes, "test_scope": "11 named modules in work log; conda metadrive python, -B -X utf8 -m unittest -q"}, exclusive=True)
    with (args.output_dir / "episodes.csv").open("x", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    index = []
    source_groups = {}
    acceptance_names = {p.name for p in runs.values()} | {PROBE, CLI}
    for folder in sorted((ROOT / "runs").glob("rule_based_v060_*")):
        if not folder.is_dir() or not (folder / "run_state.json").exists():
            continue
        state = read(folder / "run_state.json")
        if state["status"] == "RUNNING":
            raise ValueError("Unclosed historical attempt: " + folder.name)
        provenance_paths = list(folder.glob("*/agent_provenance.json"))
        if (folder / "agent_provenance.json").exists():
            provenance_paths.append(folder / "agent_provenance.json")
        entries = [{"path": str(p.relative_to(ROOT)), "sha256": sha256_file(p), "provenance": read(p)} for p in provenance_paths]
        for entry in entries:
            provenance = entry["provenance"]
            if provenance["descriptor"]["agent_version"] != "0.6.0":
                continue
            selected = {Path(k).name: v for k, v in provenance["source_sha256"].items()
                        if Path(k).name in {"policy.py", "agent_input.py", "rule_based_traffic.py", "rule_based_memory.py", "rule_based_junction.py", "rule_based_lane_change.py"}}
            identity = hashlib.sha256(json.dumps({"source_sha256": selected, "configuration_sha256": provenance["configuration_sha256"]}, sort_keys=True).encode()).hexdigest()
            group = source_groups.setdefault(identity, {"identity": identity, "provenance_path": entry["path"],
                         "policy_input_helper_sha256": selected, "configuration_sha256": provenance["configuration_sha256"], "runs": []})
            if folder.name not in group["runs"]:
                group["runs"].append(folder.name)
        index.append({"run_id": folder.name, "state": state, "state_sha256": sha256_file(folder / "run_state.json"),
                      "artifact_role": "FINAL_ACCEPTANCE" if folder.name in acceptance_names else "DRAFT_NOT_RELEASE",
                      "provenance": entries, "runner_sha256": sha256_file(folder / "runner.py") if (folder / "runner.py").exists() else None})
    write_json(args.output_dir / "experiment_index.json", index, exclusive=True)
    if summary["candidate_crashes"] or summary["candidate_out_of_road"] or summary["candidate_solid_boundary_contacts"] or summary["lost_successes"]:
        raise ValueError("Acceptance regression: evidence preserved; release not frozen")
    if args.freeze:
        draft_index = []
        final_identity = {name: hashes[name] for name in ("policy.py", "agent_input.py", "rule_based_traffic.py", "rule_based_memory.py", "rule_based_junction.py", "rule_based_lane_change.py")}
        for group in source_groups.values():
            if group["policy_input_helper_sha256"] == final_identity and group["configuration_sha256"] == expected_config_hash:
                group["archive"] = "0.6.0"
                draft_index.append(group)
                continue
            label = f"dev060_{len([g for g in draft_index if g['archive'] != '0.6.0']) + 1:02d}"
            draft = ROOT / "experiments/rule_based/versions" / label
            draft.mkdir(exist_ok=False)
            original = ROOT / group["provenance_path"]
            provenance = read(original)
            for key, expected in provenance["source_sha256"].items():
                name = Path(key).name
                if name not in set(group["policy_input_helper_sha256"]) | {"action_adapter.py", "agent_experiment.py"}:
                    continue
                source = original.parent / "sources" / key
                if sha256_file(source) != expected:
                    raise ValueError("Modified draft snapshot: " + str(source))
                with (draft / name).open("xb") as stream:
                    stream.write(source.read_bytes())
            write_json(draft / "snapshot.json", {"status": "DEVELOPMENT_DRAFT_NOT_RELEASE", "retrospective_archive": True,
                "archived_at_utc": datetime.now(timezone.utc).isoformat(), "source_origin": group,
                "configuration": provenance["configuration"], "descriptor": provenance["descriptor"],
                "note": "Copied verbatim from original raw run snapshots; source-group numbering is not chronological; missing original platform API sources are not reconstructed"}, exclusive=True)
            group["archive"] = label
            draft_index.append(group)
        write_json(args.output_dir / "development_snapshot_index.json", draft_index, exclusive=True)
        version = ROOT / "experiments/rule_based/versions/0.6.0"
        version.mkdir(exist_ok=False)
        for name, path in files.items():
            with (version / name).open("xb") as stream:
                stream.write(path.read_bytes())
        agent = Stage1RuleBasedAgent()
        configuration = agent.export_config()
        record = {"version": "0.6.0", "status": "DEVELOPMENT_RELEASE_SANDBOX_AUDIT_ONLY",
                  "p4_status": "IMPLEMENTED_WITH_SCOPED_TRAIN_VALIDATION", "full_agent_plan_complete": False,
                  "snapshot_recorded_at_utc": datetime.now(timezone.utc).isoformat(), "descriptor": agent.descriptor.to_dict(),
                  "configuration": configuration, "configuration_sha256": hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                  "source_sha256": hashes, "verified_runs": [p.name for p in runs.values()] + [PROBE, CLI],
                  "paired_reports": [str((args.output_dir / group / "comparison.json").relative_to(ROOT)) for group in GROUPS],
                  "tests_passed": count, "contract_spec_version": "1.3.0", "contract_status": "SANDBOX-EXTENSION-PENDING-AUDIT",
                  "benchmark_eligible": False, "known_limits": ["Adjacent nearly straight same-direction lanes only; no changes inside junction/roundabout", "No legal signal/sign priority source or future-intent oracle", "Bounded 50m observations/4s predictions, postcommit stop is conservative", "Full 108 TRAIN, canonical VALIDATION/TEST and contract audit remain P5", "Dense traffic latency exceeds aspirational 10ms; measurements taken during parallel development runs"],
                  "timeouts": summary["timeouts"], "shared_platform_api_note": "Recorded platform_agent and adapter are shared execution dependencies; flat frozen policy/helpers/input used for baselines"}
        write_json(version / "snapshot.json", record, exclusive=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
