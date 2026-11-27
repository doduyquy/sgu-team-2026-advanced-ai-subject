"""Replay fixed TRAIN-only reference/stationary/stress traces through S5.5 rewards.

This script never creates training data for SAC.  Its outputs are solely reward
integrity evidence and use the authoritative Stage5TrainingEnv step semantics.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.random_naive_agent import Stage0RandomNaiveAgent
from src.platform import AgentPublicEpisodeContext, get_action_adapter
from src.training.stage5.checkpoint import verify_checkpoint_integrity
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.reward_experiments import (
    RewardScale,
    Stage5ExperimentalRewardWrapper,
    compute_s5_5_reward_variants,
)
from src.training.stage5.train_sampler import SamplerStrategy, TrainCaseSampler


RUN_ID = "s5_5_meter_reward_seed101_20261120"
OUT = ROOT / "runs" / "stage5_meter_reward" / RUN_ID / "preflight"
RESULTS = ROOT / "results" / "stage5" / RUN_ID
GEOMETRY_ID = "geom_easy_SCS_seed0"
ENVIRONMENT_SEED = 0
TRAIN_IDS = [
    "geom_easy_SCCS_seed0", "geom_easy_SCCS_seed2", "geom_easy_SCCS_seed3",
    "geom_easy_SCSS_seed0", "geom_easy_SCSS_seed3", "geom_easy_SCSS_seed4",
    "geom_easy_SCS_seed0", "geom_easy_SCS_seed1",
]
SAFETY_AGENT_SEEDS = [42, 101]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_actions(path: Path) -> list[list[float]]:
    actions: list[list[float]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            action = row.get("canonical_action")
            if action is None or len(action) != 2:
                raise ValueError(f"Invalid canonical action in reference trace row {len(actions)}")
            actions.append([float(action[0]), float(action[1])])
    if not actions:
        raise ValueError("Reference trajectory contains no decisions")
    return actions


def effective_road_key(base: Stage5TrainingEnv) -> str | None:
    nav = base._env.agent.navigation
    road = getattr(nav, "current_road", None)
    if road is None:
        return None
    block = getattr(road, "block", None)
    for obj, attr in ((road, "block_ID"), (road, "block_id"), (block, "ID"), (block, "id")):
        value = getattr(obj, attr, None) if obj is not None else None
        if value is not None:
            if callable(value):
                value = value()
            return str(value)
    # Lane/road repr is a stable-enough within-run transition marker, not identity provenance.
    return f"{type(road).__name__}:{getattr(road, 'index', None)}"


def make_env() -> tuple[Stage5ExperimentalRewardWrapper, Any]:
    sampler = TrainCaseSampler(
        training_root_seed=101,
        strategy=SamplerStrategy.FIXED,
        tier_filter="Easy",
        sequence_filter="SCS",
        fixed_geometry_seed=0,
        allowed_geometry_ids=[GEOMETRY_ID],
        project_root=ROOT,
    )
    case = sampler.get_case_for_geometry_id(GEOMETRY_ID, environment_seed=ENVIRONMENT_SEED)
    base = Stage5TrainingEnv(sampler=sampler, training_root_seed=101, project_root=ROOT)
    # Placeholder scale is replaced by protocol-derived exact k in main().
    return Stage5ExperimentalRewardWrapper(base, "V1", RewardScale(1.0, 1.0)), case


def run_episode(
    *,
    trace_id: str,
    policy_kind: str,
    scale: RewardScale,
    reference_actions: list[list[float]] | None = None,
    model: SAC | None = None,
    random_seed: int | None = None,
) -> dict[str, Any]:
    env, case = make_env()
    env.scale = scale
    env.variant = "V1"
    rows: list[dict[str, Any]] = []
    raw = env.unwrapped
    policy: Any = None
    if policy_kind == "stage0_random":
        policy = Stage0RandomNaiveAgent()
        policy.reset(AgentPublicEpisodeContext(
            horizon_steps=case.horizon_steps,
            action_adapter_id="discrete9_lowbranch_v1",
            mode="INFERENCE",
        ), agent_seed=random_seed)
    try:
        obs, reset_info = env.reset(options={"case": case})
        nav = raw._env.agent.navigation
        length_m = float(nav.total_length)
        p0 = float(reset_info["initial_route_completion"])
        if not math.isclose(float(nav.route_completion), p0, rel_tol=0.0, abs_tol=1e-8):
            raise RuntimeError("Reset route completion differs between Stage5 info and MetaDrive navigation")
        reset_road = effective_road_key(raw)
        reference_index = 0
        ended = False
        previous_route = p0
        previous_road = reset_road
        boundary_events: list[dict[str, Any]] = []
        while not ended:
            if policy_kind == "stage1_reference":
                if reference_actions is None or reference_index >= len(reference_actions):
                    raise RuntimeError("Reference action trace ended before simulator episode termination")
                action = np.asarray(reference_actions[reference_index], dtype=np.float32)
            elif policy_kind == "sac_deterministic":
                if model is None:
                    raise ValueError("SAC model is required")
                action, _ = model.predict(obs, deterministic=True)
                action = np.asarray(action, dtype=np.float32)
            elif policy_kind == "stage0_random":
                decision = policy.act(None)  # Stage0RandomNaiveAgent intentionally ignores AgentInputV1.
                canonical = get_action_adapter("discrete9_lowbranch_v1").to_canonical(decision.action_payload)
                action = np.asarray(canonical.to_numpy(), dtype=np.float32)
            else:
                raise ValueError(f"Unknown policy_kind {policy_kind}")

            obs, reward_v1, terminated, truncated, info = env.step(action)
            actual_action = [float(info["action_steering"]), float(info["action_throttle_brake"])]
            if not np.allclose(action, actual_action, rtol=0.0, atol=2e-7):
                raise RuntimeError(f"Action fidelity failure at trace step {reference_index}")
            breakdown = info["reward_breakdown"]
            values = info["s5_5_rewards"]
            p = float(info["route_completion"])
            road_key = effective_road_key(raw)
            delta_p = float(info["delta_route"])
            if not math.isclose(delta_p, p - previous_route, rel_tol=0.0, abs_tol=1e-8):
                raise RuntimeError(f"Signed progress delta reset/mismatch at step {info['step_index']}")
            if road_key != previous_road:
                boundary_events.append({
                    "step_index": int(info["step_index"]),
                    "from_road_key": previous_road,
                    "to_road_key": road_key,
                    "route_completion_before": previous_route,
                    "route_completion_after": p,
                    "delta_route": delta_p,
                    "delta_s_m": length_m * delta_p,
                })
            task = np.asarray(obs[325:385], dtype=np.float32).reshape(20, 3)
            task_masks = np.asarray(obs[385:405], dtype=np.float32) > 0.5
            curve_heading = np.abs(task[task_masks, 2]) if np.any(task_masks) else np.asarray([], dtype=np.float32)
            rows.append({
                "step_index": int(info["step_index"]),
                "action_executed": actual_action,
                "speed_km_h": float(info["speed_km_h"]),
                "route_completion": p,
                "delta_p": float(info["delta_route"]),
                "delta_s_m": float(values["delta_s_m"]),
                "L_effective_m": length_m,
                "reward_v1_env": float(reward_v1),
                "reward_v1_recomputed": float(values["reward_v1"]),
                "reward_scale": float(values["reward_scale"]),
                "reward_meter": float(values["reward_meter"]),
                "reward_used": float(values["reward_used"]),
                "progress_reward": float(breakdown["progress_reward"]),
                "time_cost": float(breakdown["time_cost"]),
                "terminal_reward": float(breakdown["terminal_reward"]),
                "clipping_applied": bool(breakdown["clipping_applied"]),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "primary_reason": info.get("primary_reason"),
                "road_key": road_key,
                "max_abs_task_waypoint_heading_rad": float(np.max(curve_heading)) if curve_heading.size else None,
            })
            previous_route = p
            previous_road = road_key
            reference_index += 1
            ended = bool(terminated or truncated)

        result = rows[-1]
        terminal_nonzero = sum(abs(float(row["terminal_reward"])) > 0.0 for row in rows)
        speeds = [float(row["speed_km_h"]) for row in rows]
        route_deltas = [float(row["delta_p"]) for row in rows]
        first_curve = next((i for i, row in enumerate(rows)
                            if (row["max_abs_task_waypoint_heading_rad"] or 0.0) >= 0.20), None)
        segment_end = first_curve if first_curve is not None else len(rows)
        segment_start = max(0, segment_end - 50)
        summary = {
            "trace_id": trace_id,
            "policy_kind": policy_kind,
            "random_agent_seed": random_seed,
            "case_id": case.case_id,
            "split": case.split,
            "geometry_id": GEOMETRY_ID,
            "geometry_sha256": case.geometry_sha256,
            "geometry_generation_seed": case.geometry_generation_seed,
            "environment_seed": case.environment_seed,
            "horizon_steps": case.horizon_steps,
            "steps": len(rows),
            "initial_route_completion": p0,
            "final_route_completion": float(rows[-1]["route_completion"]),
            "route_progress_delta": float(rows[-1]["route_completion"] - p0),
            "manifest_route_length_m": float(case_manifest_route_length()),
            "effective_route_length_m": length_m,
            "effective_vs_manifest_delta_m": length_m - float(case_manifest_route_length()),
            "max_route_progress_delta_m": length_m * (max(row["route_completion"] for row in rows) - p0),
            "mean_speed_km_h": statistics.fmean(speeds),
            "peak_speed_km_h": max(speeds),
            "stationary_fraction_speed_lt_1_km_h": sum(v < 1.0 for v in speeds) / len(speeds),
            "negative_progress_steps": sum(delta < 0.0 for delta in route_deltas),
            "negative_progress_total_m": length_m * sum(delta for delta in route_deltas if delta < 0.0),
            "terminal_nonzero_reward_steps": terminal_nonzero,
            "terminal_reward_sum": sum(float(row["terminal_reward"]) for row in rows),
            "outcome": info.get("primary_reason"),
            "clean_success": bool(info.get("clean_success", False)),
            "safety_failure": str(info.get("primary_reason", "")).startswith((
                "OUT_OF_ROAD", "CRASH_")),
            "timeout": info.get("primary_reason") == "TIMEOUT",
            "terminal_transition_count": int(bool(terminated or truncated)),
            "road_block_or_road_key_transitions": sum(
                rows[i]["road_key"] != rows[i - 1]["road_key"] for i in range(1, len(rows))
            ),
            "road_key_boundary_events": boundary_events,
            "max_abs_delta_route_at_road_key_transition": max(
                (abs(float(event["delta_route"])) for event in boundary_events), default=0.0
            ),
            "reset_road_key": reset_road,
            "first_curve_step_from_public_task_waypoints": first_curve,
            "pre_curve_or_initial_50step_returns": {
                "start_step": segment_start,
                "end_step_exclusive": segment_end,
                "V1_undiscounted": sum(float(r["reward_v1_env"]) for r in rows[segment_start:segment_end]),
                "Scale_undiscounted": sum(float(r["reward_scale"]) for r in rows[segment_start:segment_end]),
                "Meter_undiscounted": sum(float(r["reward_meter"]) for r in rows[segment_start:segment_end]),
            },
            "returns": {},
        }
        for key, reward_key in (("V1", "reward_v1_env"), ("Scale", "reward_scale"), ("Meter", "reward_meter")):
            vals = [float(row[reward_key]) for row in rows]
            summary["returns"][key] = {
                "undiscounted": sum(vals),
                "discounted_gamma_0_99": sum((0.99 ** i) * value for i, value in enumerate(vals)),
                "reward_positive_fraction": sum(value > 0.0 for value in vals) / len(vals),
                "reward_mean": statistics.fmean(vals),
            }
        summary["all_v1_steps_match_environment"] = all(
            math.isclose(float(row["reward_v1_env"]), float(row["reward_v1_recomputed"]), rel_tol=0.0, abs_tol=1e-8)
            and not row["clipping_applied"] for row in rows
        )
        summary["terminal_penalty_applied_exactly_once"] = terminal_nonzero == int(
            abs(float(rows[-1]["terminal_reward"])) > 0.0
        )
        return {"summary": summary, "rows": rows}
    finally:
        if policy is not None:
            policy.close()
        env.close()


def case_manifest_route_length() -> float:
    import csv
    manifest = ROOT / "results" / "audits" / "evaluation_protocol" / "geometry_split_manifest.csv"
    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["geometry_id"] == GEOMETRY_ID:
                if row["split"].upper() != "TRAIN":
                    raise RuntimeError("Reward sanity case is not TRAIN")
                return float(row["route_length_m"])
    raise KeyError(GEOMETRY_ID)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=RESULTS / "protocol.json")
    parser.add_argument("--reference-trace", type=Path, default=(
        ROOT.parent.parent / f"{ROOT.parent.name}-khoa-s55-readonly" / "autonomous-driving-rl" /
        "runs" / "s55_trace_ref_scs0_20261120" / "current" / "episode_000" / "decisions.jsonl.gz"
    ))
    parser.add_argument("--reference-result", type=Path, default=(
        ROOT.parent.parent / f"{ROOT.parent.name}-khoa-s55-readonly" / "autonomous-driving-rl" /
        "runs" / "s55_trace_ref_scs0_20261120" / "current" / "episode_000" / "result.json"
    ))
    parser.add_argument("--stationary-checkpoint", type=Path, default=(
        ROOT / "runs" / "stage5_learnability" / "s5_4_learnability_seed101_20261119" /
        "arm_A" / "checkpoints" / "step_100000"
    ))
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    scale = RewardScale(
        median_route_length_m=float(protocol["reward_variants"]["median_training_manifest_route_length_m"]),
        scale_k=float(protocol["reward_variants"]["k"]),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not args.reference_trace.is_file() or not args.reference_result.is_file():
        raise FileNotFoundError("Stage 1 TRAIN reference trace/result is missing")
    reference_sha = file_sha256(args.reference_trace)
    copied_trace = OUT / "stage1_reference_decisions.jsonl.gz"
    if not copied_trace.exists():
        shutil.copy2(args.reference_trace, copied_trace)
    reference_result = json.loads(args.reference_result.read_text(encoding="utf-8"))
    actions = load_actions(copied_trace)

    checkpoint = verify_checkpoint_integrity(args.stationary_checkpoint)
    stationary_model = SAC.load(str(args.stationary_checkpoint / "model.zip"), device="cpu")

    traces: list[dict[str, Any]] = []
    traces.append(run_episode(
        trace_id="stage1_reference_clean", policy_kind="stage1_reference", scale=scale,
        reference_actions=actions,
    ))
    traces.append(run_episode(
        trace_id="s54_a_100k_deterministic", policy_kind="sac_deterministic", scale=scale,
        model=stationary_model,
    ))
    # Both fixed Stage0 seeds are run; no post-hoc seed search is performed.
    for seed in SAFETY_AGENT_SEEDS:
        traces.append(run_episode(
            trace_id=f"stage0_random_stress_seed{seed}", policy_kind="stage0_random",
            scale=scale, random_seed=seed,
        ))

    compact = {
        "protocol_sha256": protocol["protocol_sha256"],
        "source_commit": protocol["source_commit"],
        "trace_source": {
            "path": str(args.reference_trace), "sha256": reference_sha,
            "stage1_result": reference_result,
        },
        "stationary_checkpoint": checkpoint,
        "effective_route_length_validation": {
            "geometry_id": GEOMETRY_ID,
            "manifest_route_length_m": case_manifest_route_length(),
            "metadrive_node_network_navigation_total_length_m": traces[0]["summary"]["effective_route_length_m"],
            "matches_within_1e_5_m": abs(traces[0]["summary"]["effective_vs_manifest_delta_m"]) < 1e-5,
            "source_semantics": "MetaDrive 0.4.3 NodeNetworkNavigation.route_completion = travelled_length / total_length",
        },
        "trace_summaries": [item["summary"] for item in traces],
        "preflight": {"status": "PENDING_CHECKS"},
    }
    clean = traces[0]["summary"]
    stationary = traces[1]["summary"]
    random_runs = [x["summary"] for x in traces[2:]]
    unsafe = any(x["safety_failure"] and x["mean_speed_km_h"] > 1.0 for x in random_runs)
    clean_valid = (
        clean["clean_success"] and clean["steps"] == int(reference_result["steps"])
        and abs(clean["final_route_completion"] - float(reference_result["route_completion"])) < 0.02
        and clean["all_v1_steps_match_environment"]
    )
    stationary_valid = stationary["timeout"] and stationary["stationary_fraction_speed_lt_1_km_h"] >= 0.95
    route_match = compact["effective_route_length_validation"]["matches_within_1e_5_m"]
    reward_pass = all(x["all_v1_steps_match_environment"] for x in compact["trace_summaries"])
    terminal_pass = all(x["terminal_penalty_applied_exactly_once"] for x in compact["trace_summaries"])
    checks = {
        "stage1_reference_replay_matches_real_clean_episode": clean_valid,
        "real_near_stationary_episode": stationary_valid,
        "real_moving_safety_failure_episode": unsafe,
        "effective_route_length_matches_locked_manifest": route_match,
        "v1_reward_parity_all_steps_all_traces": reward_pass,
        "terminal_reward_exactly_once": terminal_pass,
        "all_traces_full_episode": all(x["terminal_transition_count"] == 1 for x in compact["trace_summaries"]),
    }
    compact["preflight"] = {
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "checks": checks,
        "scientific_training_authorized_by_preflight": all(checks.values()),
        "limitations": [
            "No negative route-completion step observed" if not any(x["negative_progress_steps"] for x in compact["trace_summaries"]) else "",
            "Stage0 safety stress trajectories are reward-integrity traces only, never SAC training data",
        ],
    }
    compact["reward_return_rankings"] = {
        item["summary"]["trace_id"]: {
            "undiscounted_return_descending": sorted(
                item["summary"]["returns"],
                key=lambda name: item["summary"]["returns"][name]["undiscounted"],
                reverse=True,
            ),
            "discounted_gamma_0_99_descending": sorted(
                item["summary"]["returns"],
                key=lambda name: item["summary"]["returns"][name]["discounted_gamma_0_99"],
                reverse=True,
            ),
        }
        for item in traces
    }

    raw_path = OUT / "reward_traces.jsonl.gz"
    with gzip.open(raw_path, "wt", encoding="utf-8", compresslevel=6) as f:
        for item in traces:
            for row in item["rows"]:
                f.write(json.dumps({"trace_id": item["summary"]["trace_id"], **row}, separators=(",", ":")) + "\n")
    compact["raw_trace_path"] = str(raw_path.relative_to(ROOT)).replace("\\", "/")
    compact["raw_trace_sha256"] = file_sha256(raw_path)
    compact["raw_trace_size_bytes"] = raw_path.stat().st_size
    result_path = RESULTS / "reward_preflight.json"
    result_path.write_text(json.dumps(compact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"result_path": str(result_path), "status": compact["preflight"]["status"],
                      "checks": checks, "summaries": compact["trace_summaries"]}, indent=2))
    if compact["preflight"]["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
