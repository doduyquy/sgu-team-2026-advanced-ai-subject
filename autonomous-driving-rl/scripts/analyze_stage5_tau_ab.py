"""Summarize paired bounded diagnostics and fixed-anchor SAC critic probes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, Iterable, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.stage5.checkpoint import compute_file_sha256, load_sac_checkpoint, verify_checkpoint_integrity
from src.training.stage5.observation_encoder import Stage5ObservationEncoder


STEPS = (0, 5_000, 10_000)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mean(values: Iterable[float]) -> float | None:
    values = [float(x) for x in values if x is not None]
    return statistics.mean(values) if values else None


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    return {
        "cases": n,
        "probe_censored": sum(row["outcome"] == "PROBE_CENSORED" for row in rows),
        "episodes_terminated": sum(row["outcome"] != "PROBE_CENSORED" for row in rows),
        "mean_route_progress_delta": _mean(row.get("route_progress_delta") for row in rows),
        "median_route_progress_delta": statistics.median(
            [float(row["route_progress_delta"]) for row in rows if row.get("route_progress_delta") is not None]
        ) if any(row.get("route_progress_delta") is not None for row in rows) else None,
        "mean_max_route_progress_delta": _mean(row.get("max_route_progress_delta") for row in rows),
        "mean_speed_km_h": _mean(row.get("mean_speed_km_h") for row in rows),
        "mean_episode_return": _mean(row.get("episode_return") for row in rows),
        "stationary_fraction": _mean(row.get("stationary_fraction") for row in rows),
        "mean_throttle": _mean(row.get("mean_throttle") for row in rows),
        "mean_brake": _mean(row.get("mean_brake") for row in rows),
        "mean_steering": _mean(row.get("mean_steering") for row in rows),
        "mean_inference_latency_ms": _mean(row.get("mean_inference_latency_ms") for row in rows),
        "mean_probe_duration_s": _mean(row.get("duration_s") for row in rows),
        "clean_success_count": sum(bool(row.get("clean_success")) for row in rows),
        "safety_failure_count": sum(bool(row.get("safety_failure")) for row in rows),
        "official_timeout_count": sum(bool(row.get("timeout")) for row in rows),
        "outcomes": dict(sorted((outcome, sum(row["outcome"] == outcome for row in rows))
                                 for outcome in {row["outcome"] for row in rows})),
    }


def _load_anchor_observations(run_root: Path, protocol: Dict[str, Any]) -> Dict[str, np.ndarray]:
    traces = run_root / "anchor_observations/traces"
    observations: Dict[str, np.ndarray] = {}
    for item in protocol["diagnostic_schedule"]:
        path = traces / f"step_00000_{item['geometry_id']}.jsonl"
        rows = _read_jsonl(path)
        if len(rows) != 1 or rows[0]["environment_seed"] != item["environment_seed"]:
            raise ValueError(f"Invalid fixed anchor trace for {item['geometry_id']}")
        if rows[0]["geometry_id"] != item["geometry_id"]:
            raise ValueError(f"Anchor geometry mismatch in {path}")
        observation = np.asarray(rows[0]["policy_observation"], dtype=np.float32)
        if observation.shape != (409,) or not np.isfinite(observation).all():
            raise ValueError(f"Invalid 409D finite anchor observation: {item['geometry_id']}")
        observations[item["geometry_id"]] = observation
    return observations


def _critic_rows(checkpoint: Path, arm: str, step: int, anchors: Dict[str, np.ndarray],
                 anchor_meta: Dict[str, Dict[str, Any]], samples: int = 64) -> List[Dict[str, Any]]:
    encoder = Stage5ObservationEncoder(core_only=False)
    model, metadata = load_sac_checkpoint(checkpoint, device="cpu",
                                          expected_encoder_fingerprint=encoder.get_schema_fingerprint())
    rows: List[Dict[str, Any]] = []
    generator_seed = 531000 + step + (0 if arm == "A" else 1)
    torch.manual_seed(generator_seed)
    model.policy.set_training_mode(False)
    with torch.no_grad():
        for geometry_id, observation in anchors.items():
            obs_tensor, _ = model.policy.obs_to_tensor(observation.reshape(1, -1))
            deterministic, _ = model.predict(observation, deterministic=True)
            det_tensor = torch.as_tensor(np.asarray(deterministic, dtype=np.float32).reshape(1, -1))
            det_q = model.policy.critic(obs_tensor, det_tensor)
            det_target = model.policy.critic_target(obs_tensor, det_tensor)

            sampled_obs = obs_tensor.repeat(samples, 1)
            sampled_actions, sampled_log_prob = model.policy.actor.action_log_prob(sampled_obs)
            sampled_q = model.policy.critic(sampled_obs, sampled_actions)
            sampled_target = model.policy.critic_target(sampled_obs, sampled_actions)
            throttle = sampled_actions[:, 1].detach().cpu().numpy()
            q1 = sampled_q[0].reshape(-1).detach().cpu().numpy()
            q2 = sampled_q[1].reshape(-1).detach().cpu().numpy()
            tq1 = sampled_target[0].reshape(-1).detach().cpu().numpy()
            tq2 = sampled_target[1].reshape(-1).detach().cpu().numpy()
            logp = sampled_log_prob.reshape(-1).detach().cpu().numpy()
            rows.append({
                "arm": arm, "training_step": step, "checkpoint_sha256": metadata["model_sha256"],
                "geometry_id": geometry_id, "tier": anchor_meta[geometry_id]["tier"],
                "environment_seed": anchor_meta[geometry_id]["environment_seed"],
                "anchor_observation_sha256": anchor_meta[geometry_id]["observation_sha256"],
                "deterministic_action": np.asarray(deterministic, dtype=float).reshape(-1).tolist(),
                "deterministic_q1": float(det_q[0].item()), "deterministic_q2": float(det_q[1].item()),
                "deterministic_min_q": float(torch.minimum(det_q[0], det_q[1]).item()),
                "deterministic_target_q1": float(det_target[0].item()),
                "deterministic_target_q2": float(det_target[1].item()),
                "stochastic_samples": samples,
                "stochastic_throttle_mean": float(np.mean(throttle)),
                "stochastic_throttle_std": float(np.std(throttle)),
                "stochastic_throttle_quantiles": {str(q): float(np.quantile(throttle, q)) for q in (0, .1, .5, .9, 1)},
                "stochastic_positive_throttle_fraction": float(np.mean(throttle > 0.05)),
                "stochastic_q1_mean": float(np.mean(q1)), "stochastic_q2_mean": float(np.mean(q2)),
                "stochastic_min_q_mean": float(np.mean(np.minimum(q1, q2))),
                "stochastic_target_min_q_mean": float(np.mean(np.minimum(tq1, tq2))),
                "stochastic_log_prob_mean": float(np.mean(logp)),
                "entropy_coefficient": float(model.log_ent_coef.detach().exp().cpu().item())
                    if getattr(model, "log_ent_coef", None) is not None else float(model.ent_coef),
                "diagnostic_label": "critic evaluations on fixed reset observations; not empirical returns",
            })
    return rows


def analyze(run_root: Path, results_dir: Path, protocol_path: Path, *, overwrite: bool = False) -> Dict[str, Any]:
    protected = [results_dir / name for name in (
        "anchor_observations.json", "q_anchor_diagnostics.json", "analysis.json", "artifact_manifest.json"
    )]
    existing = [path for path in protected if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite Stage 5 analysis artifacts: {existing}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    schedule = {row["geometry_id"]: row for row in protocol["diagnostic_schedule"]}
    rows: List[Dict[str, Any]] = []
    for arm in ("A", "B"):
        for step in STEPS:
            path = run_root / f"arm_{arm}/diagnostics/step_{step:05d}" / f"step_{step:05d}_episodes.jsonl"
            current = _read_jsonl(path)
            if len(current) != 8 or {row["geometry_id"] for row in current} != set(schedule):
                raise ValueError(f"Arm {arm} step {step} does not contain the complete fixed 8-case schedule")
            for row in current:
                expected = schedule[row["geometry_id"]]
                if (row["environment_seed"] != expected["environment_seed"] or
                        row["geometry_sha256"] != expected["geometry_sha256"]):
                    raise ValueError(f"Paired case/seed/hash mismatch for {row['geometry_id']}")
                if row.get("interrupted") and row["outcome"] != "PROBE_CENSORED":
                    raise ValueError("Interrupted probe was not explicitly labeled PROBE_CENSORED")
                if row["outcome"] == "PROBE_CENSORED" and row.get("timeout"):
                    raise ValueError("Censored probe must not be counted as official timeout")
                row["arm"] = arm
                rows.append(row)

    grouped: Dict[str, Any] = {}
    for arm in ("A", "B"):
        grouped[arm] = {}
        for step in STEPS:
            subset = [r for r in rows if r["arm"] == arm and r["training_step"] == step]
            grouped[arm][str(step)] = {"overall": _summary(subset), "per_tier": {
                tier: _summary([row for row in subset if row["tier"] == tier])
                for tier in ("Easy", "Medium", "Hard", "Extreme")}}

    by_key = {(row["arm"], row["training_step"], row["geometry_id"]): row for row in rows}
    paired: Dict[str, Any] = {}
    for step in STEPS:
        pairs = [(by_key[("B", step, geom)], by_key[("A", step, geom)]) for geom in sorted(schedule)]
        metric_names = ("route_progress_delta", "max_route_progress_delta", "mean_speed_km_h",
                        "stationary_fraction", "mean_episode_return", "mean_throttle", "mean_brake",
                        "mean_steering", "mean_inference_latency_ms")
        contrasts: Dict[str, Any] = {}
        for metric in metric_names:
            differences = [float(b[metric]) - float(a[metric]) for b, a in pairs
                           if b.get(metric) is not None and a.get(metric) is not None]
            contrasts[metric + "_B_minus_A"] = {
                "n_pairs": len(differences), "mean": _mean(differences),
                "median": statistics.median(differences) if differences else None,
                "positive_pairs": sum(value > 0 for value in differences),
                "negative_pairs": sum(value < 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
                "case_differences": differences,
            }
        paired[str(step)] = contrasts

    anchors = _load_anchor_observations(run_root, protocol)
    anchor_meta: Dict[str, Dict[str, Any]] = {}
    for geometry_id, observation in anchors.items():
        row = schedule[geometry_id]
        anchor_meta[geometry_id] = {
            "tier": row["tier"], "environment_seed": row["environment_seed"],
            "geometry_sha256": row["geometry_sha256"],
            "observation_sha256": hashlib.sha256(observation.tobytes()).hexdigest(),
        }
    anchor_path = results_dir / "anchor_observations.json"
    anchor_path.write_text(json.dumps({
        "protocol_sha256": protocol["protocol_sha256"], "observation_dim": 409,
        "anchors": {key: {**anchor_meta[key], "observation": value.tolist()} for key, value in anchors.items()},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    q_rows = []
    for arm in ("A", "B"):
        for step in STEPS:
            cp = run_root / f"arm_{arm}/checkpoints/step_{step:05d}"
            q_rows.extend(_critic_rows(cp, arm, step, anchors, anchor_meta))
    q_result = {
        "protocol_sha256": protocol["protocol_sha256"],
        "definition": "online and target critic Q on the exact same fixed reset observations; stochastic actor action samples; no environment rollouts in this analysis",
        "records": q_rows,
    }
    q_path = results_dir / "q_anchor_diagnostics.json"
    q_path.write_text(json.dumps(q_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = {
        "protocol_sha256": protocol["protocol_sha256"], "arms": grouped, "paired_B_minus_A": paired,
        "diagnostic_records": rows, "q_anchor_diagnostics_path": "q_anchor_diagnostics.json",
        "holdouts_accessed": False, "candidate_selection_split": "internal PILOT_DIAGNOSTIC only",
    }
    analysis_path = results_dir / "analysis.json"
    analysis_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    initial_path = run_root / "common_initial_model.zip"
    initial_state = json.loads((run_root / "common_initial_state.json").read_text(encoding="utf-8"))
    checkpoint_manifest: Dict[str, Any] = {}
    diagnostic_manifest: Dict[str, Any] = {}
    encoder_fingerprint = Stage5ObservationEncoder(core_only=False).get_schema_fingerprint()
    for arm in ("A", "B"):
        checkpoint_manifest[arm] = []
        diagnostic_manifest[arm] = []
        for step in STEPS:
            checkpoint = run_root / f"arm_{arm}/checkpoints/step_{step:05d}"
            metadata = verify_checkpoint_integrity(
                checkpoint, expected_encoder_fingerprint=encoder_fingerprint
            )
            records_path = run_root / f"arm_{arm}/diagnostics/step_{step:05d}/step_{step:05d}_episodes.jsonl"
            checkpoint_manifest[arm].append({
                "training_step": step,
                "model_sha256": metadata["model_sha256"],
                "metadata_sha256": compute_file_sha256(checkpoint / "metadata.json"),
                "checksums_sha256": compute_file_sha256(checkpoint / "CHECKSUMS.sha256"),
                "verified": True,
            })
            diagnostic_manifest[arm].append({
                "training_step": step,
                "path": str(records_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": compute_file_sha256(records_path),
                "bytes": records_path.stat().st_size,
                "episodes": len(_read_jsonl(records_path)),
            })
    run_manifest = {
        "protocol": "stage5_tau_ab_v1",
        "run_id": protocol["run_id"],
        "protocol_sha256": protocol["protocol_sha256"],
        "training_source_commit": protocol["code_commit"],
        "pilot_split_sha256": protocol["pilot_split_sha256"],
        "registered_config_sha256": protocol["registered_config_sha256"],
        "initial_model": {
            "path": str(initial_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": compute_file_sha256(initial_path),
            "policy_entropy_fingerprint": initial_state["policy_and_entropy_fingerprint"],
        },
        "checkpoints": checkpoint_manifest,
        "diagnostic_records": diagnostic_manifest,
        "arm_run_status": {
            arm: {
                "status": json.loads((run_root / f"arm_{arm}/run_status.json").read_text(encoding="utf-8"))["status"],
                "sha256": compute_file_sha256(run_root / f"arm_{arm}/run_status.json"),
                "training_summary": json.loads((run_root / f"arm_{arm}/run_status.json").read_text(encoding="utf-8"))["training_summary"],
            } for arm in ("A", "B")
        },
        "anchors_sha256": compute_file_sha256(anchor_path),
        "q_diagnostics_sha256": compute_file_sha256(q_path),
        "analysis_sha256": compute_file_sha256(analysis_path),
        "postprocessing_source_sha256": {
            "scripts/analyze_stage5_tau_ab.py": compute_file_sha256(ROOT / "scripts/analyze_stage5_tau_ab.py"),
            "scripts/evaluate_stage5_pilot.py": compute_file_sha256(ROOT / "scripts/evaluate_stage5_pilot.py"),
        },
        "raw_artifacts_root": str(run_root.relative_to(ROOT)).replace("\\", "/"),
        "raw_artifacts_gitignored": True,
        "holdouts_accessed": False,
    }
    (results_dir / "artifact_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="s5_3m_tau_ab_seed101_20261119")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace compact derived analysis files; raw runs/checkpoints are never modified.")
    args = parser.parse_args()
    run_root = ROOT / "runs/stage5_tau_ab" / args.run_id
    results_dir = ROOT / "results/stage5" / args.run_id
    result = analyze(run_root, results_dir, results_dir / "protocol.json", overwrite=args.overwrite)
    print(json.dumps({"protocol_sha256": result["protocol_sha256"], "paired_B_minus_A": result["paired_B_minus_A"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
