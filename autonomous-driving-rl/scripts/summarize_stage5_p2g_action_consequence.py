"""Build compact, auditable P2-G summaries from the frozen evidence artifact."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/stage5/p2g_action_consequence_20261121"
RUN = ROOT / "runs/stage5_p2d/s5_p2d_actor_activation_seed101_20261121"
P2F = ROOT / "runs/stage5_p2f/s5_p2f_start_distance_seed101_20261121_retry3"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()

def ranks(x):
    order = np.argsort(x, kind="mergesort")
    result = np.empty(len(x), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and x[order[j]] == x[order[i]]:
            j += 1
        result[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return result

def corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(ranks(np.asarray(a)), ranks(np.asarray(b)))[0, 1])

def main():
    evidence_path = OUT / "evidence.json"
    data = json.loads(evidence_path.read_text(encoding="utf-8"))
    physical = data["counterfactuals"]
    model_rows = data["critic_actor_diagnostics"]["state_model_diagnostics"]
    model_map = {(r["state_label"], r["model"]): r for r in model_rows}
    physical_summary = []
    for label in dict.fromkeys(r["label"] for r in data["states"]):
        rows = [r for r in physical if r["state_label"] == label]
        qrank = {}
        for model in data["checkpoint_provenance"]:
            q = model_map[(label, model)]["action_grid_q"]
            by_action = {(round(x["steering"], 3), round(x["throttle_brake"], 3)): x for x in q}
            qvalues = [by_action[(round(r["action"][0], 3), round(r["action"][1], 3))]["min_q"] for r in rows]
            best_q = max(q, key=lambda x: x["min_q"])
            qrank[model] = {
                "q_min_range": [float(min(qvalues)), float(max(qvalues))],
                "q_min_action_grid_spearman_vs_short_v1_return": corr(qvalues, [r["reward_v1_sum"] for r in rows]),
                "q_min_action_grid_spearman_vs_route_delta_m": corr(qvalues, [r["route_delta_m"] for r in rows]),
                "q_preferred_action": [best_q["steering"], best_q["throttle_brake"]],
                "actor_action": model_map[(label, model)]["deterministic_action"],
                "actor_entropy_q_gradient": model_map[(label, model)]["actor_objective_gradient"],
                "action_q_gradient_norm_mean": float(np.mean([np.linalg.norm(x["min_q_gradient"]) for x in q])),
            }
        physical_summary.append({
            "state_label": label, "prefix_decisions": next(s["prefix_decisions"] for s in data["states"] if s["label"] == label),
            "trials": len(rows), "safety_failures": sum(bool(r["terminal_outcome"] and r["terminal_outcome"].get("terminated")) for r in rows),
            "probe_censored": sum(bool(r["probe_censored"]) for r in rows),
            "action_results": [{"action": r["action"], "route_delta_m": r["route_delta_m"],
                                "reward_v1_sum": r["reward_v1_sum"], "mean_speed_kmh": r["mean_speed_kmh"],
                                "min_estimated_clearance_m": r["min_estimated_clearance_m"],
                                "outcome": r["terminal_outcome"]["primary_reason"] if r["terminal_outcome"] else "PROBE_CENSORED"}
                               for r in rows], "critic_actor": qrank,
        })
    summary = {
        "schema": "stage5_p2g_summary_v1", "evidence_sha256": sha(evidence_path),
        "protocol_sha256": data["protocol_sha256"], "state_count": len(data["states"]),
        "counterfactual_count": len(physical), "action_combinations_per_state": 9,
        "max_post_branch_decisions": 20, "physical_replay_validation_passed": all(x["passed"] for x in data["state_replay_validation"]),
        "state_replay_validation": data["state_replay_validation"], "states": physical_summary,
        "limitations": ["one source geometry/environment seed", "one source training trajectory", "20-decision fixed-action horizon",
                        "short return is not SAC soft-Q", "estimated lateral body clearance is descriptive, not MetaDrive safety oracle",
                        "action support is a local empirical envelope, not trajectory density"],
        "validation_test_accessed": False, "scientific_training": False, "benchmark_eligible": False,
    }
    summary_path = OUT / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    checks = OUT / "CHECKSUMS.sha256"
    entries = {}
    if checks.exists():
        for line in checks.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, name = line.split("  ", 1); entries[name] = digest
    entries["results/stage5/p2g_action_consequence_20261121/analysis_summary.json"] = sha(summary_path)
    entries["results/stage5/p2g_action_consequence_20261121/evidence.json"] = sha(evidence_path)
    for name in ("physical_counterfactuals.json", "pre_amendment_repeatability_probe.json", "protocol.json"):
        path = OUT / name
        if path.exists():
            entries[f"results/stage5/p2g_action_consequence_20261121/{name}"] = sha(path)
    entries["runs/stage5_p2d/s5_p2d_actor_activation_seed101_20261121/arm_A/replay_action_trace.jsonl"] = sha(
        ROOT / "runs/stage5_p2d/s5_p2d_actor_activation_seed101_20261121/arm_A/replay_action_trace.jsonl")
    for arm, run in (("P2D_A", RUN / "arm_A"), ("P2D_B", RUN / "arm_B"), ("P2F_A", P2F / "arm_A"), ("P2F_B", P2F / "arm_B")):
        checkpoint = run / "checkpoints/step_010000"
        for filename in ("model.zip", "metadata.json"):
            entries[f"{checkpoint.relative_to(ROOT).as_posix()}/{filename}"] = sha(checkpoint / filename)
    for name in ("scripts/run_stage5_p2g_action_consequence.py", "scripts/summarize_stage5_p2g_action_consequence.py",
                 "tests/test_stage5_p2g_action_consequence.py", "docs/stage5/S5_P2G_ACTION_CONSEQUENCE_PROTOCOL.md",
                 "docs/stage5/S5_P2G_ACTION_CONSEQUENCE_RESULTS.md", "docs/stage5/S5_P2G_NEXT_DECISION.md"):
        entries[name] = sha(ROOT / name)
    checks.write_text("".join(f"{v}  {k}\n" for k, v in sorted(entries.items())), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "summary_sha256": sha(summary_path), "states": len(data["states"]),
                      "counterfactuals": len(physical), "checksums": str(checks)}, indent=2))

if __name__ == "__main__":
    main()
