"""Bounded TRAIN-only physical action consequences for frozen Stage 5 SAC checkpoints."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER
from src.training.stage5.checkpoint import load_sac_checkpoint, verify_checkpoint_integrity
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.training.stage5.pilot_observation_transform import GROUPWISE_PHYSICAL_V1, GroupwisePhysicalObservation
from src.training.stage5.train_sampler import SamplerStrategy, TrainCaseSampler

RUN = ROOT / "runs/stage5_p2d/s5_p2d_actor_activation_seed101_20261121"
P2F = ROOT / "runs/stage5_p2f/s5_p2f_start_distance_seed101_20261121_retry3"
OUT = ROOT / "results/stage5/p2g_action_consequence_20261121"
PROTOCOL_PATH = OUT / "protocol.json"
GEOMETRY_ID = "geom_easy_SCSS_seed3"
ENV_SEED = 976051
SOURCE_EPISODE = 6
MAX_HOLD = 20
REPETITIONS = 3
TOL = {
    "position_m": 0.02, "heading_rad": 0.001, "speed_kmh": 0.05,
    "route_completion": 1e-6, "historical_reward": 1e-6,
    "observation": 1e-4,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_env() -> tuple[GroupwisePhysicalObservation, Stage5TrainingEnv, TrainCaseSampler, Any]:
    sampler = TrainCaseSampler(training_root_seed=101, strategy=SamplerStrategy.ROUND_ROBIN,
                               allowed_geometry_ids=[GEOMETRY_ID], project_root=ROOT)
    case = sampler.get_case_for_geometry_id(GEOMETRY_ID, ENV_SEED)
    base = Stage5TrainingEnv(sampler=sampler, training_root_seed=101, project_root=ROOT,
                             simulator_render_mode="OFF", task_context_producer_id=REFERENCE_TASK_PRODUCER)
    return GroupwisePhysicalObservation(base), base, sampler, case


def wrap_angle(x: float) -> float:
    return float((x + math.pi) % (2 * math.pi) - math.pi)


def physical_signature(env: GroupwisePhysicalObservation, base: Stage5TrainingEnv,
                       obs: np.ndarray, info: dict[str, Any]) -> dict[str, Any]:
    sim = base._env
    vehicle = sim.agent
    nav = vehicle.navigation
    lane = nav.current_ref_lanes[0]
    pos = np.asarray(vehicle.position, dtype=np.float64)[:2]
    station, lateral = lane.local_coordinates(pos)
    lane_heading = float(lane.heading_theta_at(float(station)))
    width = float(nav.get_current_lane_width())
    vehicle_width = float(getattr(vehicle, "WIDTH", getattr(vehicle, "width", 1.8)))
    margin = width / 2.0 - abs(float(lateral)) - vehicle_width / 2.0
    task_headings = np.asarray(obs, dtype=np.float64)[327:385:3] * math.pi
    task_valid = np.asarray(obs, dtype=np.float64)[385:405] > 0.5
    valid_headings = task_headings[task_valid[:len(task_headings)]] if task_valid.any() else np.asarray([])
    return {
        "position_xy": pos.tolist(), "heading_rad": float(vehicle.heading_theta),
        "lane_heading_rad": lane_heading, "heading_error_rad": wrap_angle(float(vehicle.heading_theta) - lane_heading),
        "speed_kmh": float(vehicle.speed_km_h), "route_completion": float(info.get("route_completion", info.get("initial_route_completion", 0.0))),
        "reference_station_m": float(station), "lateral_error_m": float(lateral),
        "lane_width_m": width, "vehicle_width_m": vehicle_width,
        "navigation_total_length_m": float(nav.total_length),
        "estimated_body_edge_clearance_m": margin, "on_lane": bool(getattr(vehicle, "on_lane", False)),
        "task_valid_waypoints": int(task_valid.sum()),
        "max_abs_valid_waypoint_heading_rad": float(np.max(np.abs(valid_headings))) if len(valid_headings) else 0.0,
        "observation": np.asarray(obs, dtype=np.float32).copy(),
    }


def run_prefix(env: GroupwisePhysicalObservation, base: Stage5TrainingEnv, case: Any,
               prefix: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obs, reset_info = env.reset(options={"case": case})
    obs = np.asarray(obs, dtype=np.float32)
    previous_route = float(reset_info["initial_route_completion"])
    rows: list[dict[str, Any]] = []
    for expected in prefix:
        action = np.asarray(expected["executed_action"], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        obs = np.asarray(obs, dtype=np.float32)
        signature = physical_signature(env, base, obs, info)
        route = float(info["route_completion"])
        rows.append({"signature": signature, "reward": float(reward), "delta_route": float(info["delta_route"]),
                     "route_completion": route, "terminated": bool(terminated), "truncated": bool(truncated),
                     "info": info})
        if abs(float(info["speed_km_h"]) - float(expected["speed_km_h"])) > TOL["speed_kmh"]:
            raise RuntimeError(f"historical speed replay mismatch at decision {expected['training_step']}")
        if abs(float(info["delta_route"]) - float(expected["delta_route"])) > TOL["route_completion"]:
            raise RuntimeError(f"historical route replay mismatch at decision {expected['training_step']}")
        if abs(float(reward) - float(expected["reward"])) > TOL["historical_reward"]:
            raise RuntimeError(f"historical reward replay mismatch at decision {expected['training_step']}")
        if terminated or truncated:
            if expected is not prefix[-1]:
                raise RuntimeError("historical source episode terminated before selected state")
            break
        previous_route = route
    if rows and (rows[-1]["terminated"] or rows[-1]["truncated"]):
        raise RuntimeError("selected state is terminal; counterfactual starts must be pre-terminal")
    if prefix:
        signature = rows[-1]["signature"]
    else:
        signature = physical_signature(env, base, obs, reset_info)
    return signature, rows


def state_from_scan(sig: dict[str, Any], prefix_len: int, episode_delta: float, label: str) -> dict[str, Any]:
    return {"label": label, "prefix_decisions": int(prefix_len), "episode_route_completion_delta": float(episode_delta),
            **{k: v for k, v in sig.items() if k != "observation"},
            "observation": sig["observation"]}


def select_states(trace: list[dict[str, Any]], env: GroupwisePhysicalObservation,
                  base: Stage5TrainingEnv, case: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ep = [r for r in trace if int(r["episode_index"]) == SOURCE_EPISODE]
    if not ep or any(r["geometry_id"] != GEOMETRY_ID or int(r["environment_seed"]) != ENV_SEED for r in ep):
        raise RuntimeError("P2-D source episode identity does not match frozen protocol")
    obs, reset_info = env.reset(options={"case": case})
    reset_sig = physical_signature(env, base, np.asarray(obs, dtype=np.float32), reset_info)
    initial = float(reset_info["initial_route_completion"])
    candidates: list[tuple[str, int, dict[str, Any], float]] = []
    cumulative = 0.0
    scan_rows: list[dict[str, Any]] = []
    previous_route = initial
    for idx, expected in enumerate(ep, start=1):
        obs, reward, term, trunc, info = env.step(np.asarray(expected["executed_action"], dtype=np.float32))
        obs = np.asarray(obs, dtype=np.float32)
        sig = physical_signature(env, base, obs, info)
        if abs(sig["speed_kmh"] - float(expected["speed_km_h"])) > TOL["speed_kmh"]:
            raise RuntimeError(f"source scan speed mismatch at history step {expected['training_step']}")
        if abs(float(info["delta_route"]) - float(expected["delta_route"])) > TOL["route_completion"]:
            raise RuntimeError(f"source scan progress mismatch at history step {expected['training_step']}")
        if abs(float(reward) - float(expected["reward"])) > TOL["historical_reward"]:
            raise RuntimeError(f"source scan reward mismatch at history step {expected['training_step']}")
        cumulative += float(info["delta_route"])
        scan_rows.append({"prefix": idx, "history_step": expected["training_step"], "signature": sig,
                          "route_delta": float(info["delta_route"]), "reward": float(reward),
                          "terminal": bool(term or trunc)})
        if term or trunc:
            if idx != len(ep):
                raise RuntimeError("source trace continued after a terminal transition")
            break
    if len(scan_rows) != len(ep):
        raise RuntimeError("source trace was not replayed completely")

    candidates.append(("reset_stationary", 0, reset_sig, 0.0))
    moving = next((x for x in scan_rows if x["signature"]["speed_kmh"] >= 1.0 and x["route_delta"] > 0), None)
    if moving:
        candidates.append(("moving_positive_progress", moving["prefix"], moving["signature"], moving["signature"]["route_completion"] - initial))
    progressed = next((x for x in scan_rows if x["signature"]["route_completion"] - initial >= 0.01 and not x["terminal"]), None)
    if progressed:
        candidates.append(("already_progressed", progressed["prefix"], progressed["signature"], progressed["signature"]["route_completion"] - initial))
    valid_curves = [x for x in scan_rows if not x["terminal"] and x["signature"]["task_valid_waypoints"] >= 3]
    curve = max(valid_curves, key=lambda x: x["signature"]["max_abs_valid_waypoint_heading_rad"], default=None)
    if curve and curve["signature"]["max_abs_valid_waypoint_heading_rad"] >= 0.05:
        candidates.append(("near_curve", curve["prefix"], curve["signature"], curve["signature"]["route_completion"] - initial))
    preterminal = [x for x in scan_rows if not x["terminal"] and x["signature"]["speed_kmh"] >= 1.0]
    lateral = max(preterminal, key=lambda x: abs(x["signature"]["lateral_error_m"]), default=None)
    if lateral:
        candidates.append(("lateral_heading_error", lateral["prefix"], lateral["signature"], lateral["signature"]["route_completion"] - initial))
    margin = min(preterminal, key=lambda x: x["signature"]["estimated_body_edge_clearance_m"], default=None)
    if margin:
        candidates.append(("reduced_safety_margin", margin["prefix"], margin["signature"], margin["signature"]["route_completion"] - initial))

    # Preserve distinct physical decision states; merge labels only when selectors coincide.
    merged: dict[int, dict[str, Any]] = {}
    for label, prefix, sig, delta in candidates:
        if prefix not in merged:
            merged[prefix] = state_from_scan(sig, prefix, delta, label)
        else:
            merged[prefix]["label"] += "+" + label
    states = list(merged.values())[:6]
    return states, {"source_episode_transition_count": len(ep), "source_terminal": ep[-1].get("outcome"),
                    "source_initial_route_completion": initial, "scan_rows": scan_rows,
                    "selected_state_count": len(states), "selected_labels": [x["label"] for x in states]}


def signature_error(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    return {
        "position_m": float(np.linalg.norm(np.asarray(a["position_xy"]) - np.asarray(b["position_xy"]))),
        "heading_rad": abs(wrap_angle(float(a["heading_rad"]) - float(b["heading_rad"]))),
        "speed_kmh": abs(float(a["speed_kmh"]) - float(b["speed_kmh"])),
        "route_completion": abs(float(a["route_completion"]) - float(b["route_completion"])),
        "observation": float(np.max(np.abs(np.asarray(a["observation"]) - np.asarray(b["observation"])))),
    }


def eval_models(states: list[dict[str, Any]], model_specs: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    encoder = Stage5ObservationEncoder(core_only=False)
    models: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    replay_roots = {"P2D_A_10000": RUN / "arm_A", "P2D_B_10000": RUN / "arm_B",
                    "P2F_A_10000": P2F / "arm_A", "P2F_B_10000": P2F / "arm_B"}
    replay_support: dict[str, Any] = {}
    for name, path in model_specs.items():
        meta = verify_checkpoint_integrity(path, expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
                                          task_context_producer_id=REFERENCE_TASK_PRODUCER)
        if meta.get("observation_transform") != GROUPWISE_PHYSICAL_V1:
            raise RuntimeError(f"Checkpoint {name} transform differs from P2D/P2F contract")
        model, _ = load_sac_checkpoint(path, device="cpu", load_replay_buffer=True,
                                       task_context_producer_id=REFERENCE_TASK_PRODUCER,
                                       expected_actor_activation_id=meta.get("actor_activation"))
        trace = jsonl(replay_roots[name] / "replay_action_trace.jsonl")
        n = model.replay_buffer.size()
        replay_obs = np.asarray(model.replay_buffer.observations[:n]).reshape(n, -1)
        trace = trace[:n]
        support_rows = []
        for state in states:
            same = np.asarray([r["geometry_id"] == GEOMETRY_ID for r in trace], dtype=bool)
            distances = np.linalg.norm(replay_obs[same] - np.asarray(state["observation"])[None, :], axis=1)
            same_rows = [r for r, keep in zip(trace, same) if keep]
            nearest_idx = np.argsort(distances)[:min(20, len(distances))]
            local_actions = np.asarray([same_rows[int(i)]["replay_action"] for i in nearest_idx], dtype=np.float64)
            query_actions = [[s, t] for s in (-0.35, 0.0, 0.35) for t in (-0.5, 0.0, 0.5)]
            support_rows.append({"state_label": state["label"], "same_geometry_transition_count": len(same_rows),
                                 "nearest_state_distance_mean": float(np.mean(distances[nearest_idx])) if len(nearest_idx) else None,
                                 "nearest_state_distance_min": float(np.min(distances[nearest_idx])) if len(nearest_idx) else None,
                                 "nearest_action_mean": local_actions.mean(axis=0).tolist() if len(local_actions) else None,
                                 "nearest_action_std": local_actions.std(axis=0).tolist() if len(local_actions) else None,
                                 "actions_inside_axis_envelope": [bool(np.all(np.asarray(a) >= local_actions.min(axis=0)) and
                                                                         np.all(np.asarray(a) <= local_actions.max(axis=0)))
                                                                   for a in query_actions] if len(local_actions) else [],
                                 "interpretation": "local empirical action envelope only; not a density or trajectory-support test"})
        replay_support[name] = support_rows
        model.replay_buffer = None
        models[name] = model
        provenance[name] = {"path": str(path.relative_to(ROOT)), "model_sha256": meta["model_sha256"],
                            "step": int(meta["total_timesteps"]), "actor_activation": meta.get("actor_activation"),
                            "producer": meta.get("task_context_producer"),
                            "transform_fingerprint": meta.get("observation_transform_fingerprint"),
                            "replay_buffer_sha256": meta.get("replay_buffer_sha256")}

    rows: list[dict[str, Any]] = []
    for state in states:
        obs = np.asarray(state["observation"], dtype=np.float32)
        for name, model in models.items():
            tensor, _ = model.policy.obs_to_tensor(obs.reshape(1, -1))
            actor = model.policy.actor
            with torch.no_grad():
                deterministic = actor(tensor, deterministic=True)
                mean_actions, log_std, _ = actor.get_action_dist_params(tensor)
            sampled_action, log_prob = actor.action_log_prob(tensor)
            actions = torch.tensor([[s, t] for s in (-0.35, 0.0, 0.35) for t in (-0.5, 0.0, 0.5)],
                                   dtype=torch.float32, requires_grad=True)
            tiled = tensor.repeat(9, 1)
            q1, q2 = model.policy.critic(tiled, actions)
            qmin = torch.minimum(q1, q2)
            q_grad = torch.autograd.grad(qmin.sum(), actions, retain_graph=False)[0]
            alpha = (float(model.log_ent_coef.detach().exp().cpu().item())
                     if getattr(model, "log_ent_coef", None) is not None else float(model.ent_coef))
            entropy_loss = alpha * log_prob.mean()
            sample_q1, sample_q2 = model.policy.critic(tensor, sampled_action)
            q_loss = -torch.minimum(sample_q1, sample_q2).mean()
            actor_params = [p for p in actor.parameters() if p.requires_grad]
            ge = torch.autograd.grad(entropy_loss, actor_params, retain_graph=True, allow_unused=True)
            gq = torch.autograd.grad(q_loss, actor_params, retain_graph=False, allow_unused=True)
            ve = torch.cat([g.detach().reshape(-1) for g in ge if g is not None])
            vq = torch.cat([g.detach().reshape(-1) for g in gq if g is not None])
            denom = float(ve.norm() * vq.norm())
            rows.append({"state_label": state["label"], "model": name,
                         "deterministic_action": deterministic.detach().cpu().numpy()[0].astype(float).tolist(),
                         "policy_mean_action": mean_actions.detach().cpu().numpy()[0].astype(float).tolist(),
                         "log_std": log_std.detach().cpu().numpy()[0].astype(float).tolist(),
                         "alpha": alpha, "sampled_log_prob": float(log_prob.detach().cpu().numpy()[0]),
                         "actor_objective_gradient": {"entropy_norm": float(ve.norm()), "negative_q_norm": float(vq.norm()),
                                                       "cosine": float(torch.dot(ve, vq) / denom) if denom else None},
                         "action_grid_q": [{"steering": float(actions[i, 0].detach()), "throttle_brake": float(actions[i, 1].detach()),
                                            "q1": float(q1[i].detach()), "q2": float(q2[i].detach()), "min_q": float(qmin[i].detach()),
                                            "min_q_gradient": q_grad[i].detach().cpu().numpy().astype(float).tolist()}
                                           for i in range(9)]})
    return provenance, {"state_model_diagnostics": rows, "replay_action_support": replay_support}


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol_hash = canonical_sha(protocol)
    source = RUN / "arm_A/replay_action_trace.jsonl"
    if not source.exists():
        raise FileNotFoundError(source)
    trace = jsonl(source)
    episode = [r for r in trace if int(r["episode_index"]) == SOURCE_EPISODE]
    if len(episode) == 0 or episode[-1].get("outcome") is None:
        raise RuntimeError("P2-D source episode has no recorded terminal outcome")
    env, base, sampler, case = make_env()
    try:
        states, scan = select_states(trace, env, base, case)
        # Validate each selected state by independently replaying its exact action prefix three times.
        validations = []
        for state in states:
            prefix = episode[:int(state["prefix_decisions"])]
            signatures = []
            for _ in range(REPETITIONS):
                sig, _ = run_prefix(env, base, case, prefix)
                signatures.append(sig)
            errors = [signature_error(signatures[0], item) for item in signatures[1:]]
            replay_keys = ("position_m", "heading_rad", "speed_kmh", "route_completion", "observation")
            maxima = {key: max((e[key] for e in errors), default=0.0) for key in replay_keys}
            replay_tolerances = {key: TOL[key] for key in replay_keys}
            passed = all(maxima[k] <= replay_tolerances[k] for k in maxima)
            validation = {"label": state["label"], "prefix_decisions": state["prefix_decisions"],
                          "repetitions": REPETITIONS, "max_pairwise_errors": maxima, "tolerances": replay_tolerances,
                          "passed": passed, "reference_signature": {k: v for k, v in signatures[0].items() if k != "observation"}}
            validations.append(validation)
            state["replay_validated"] = passed
            state["reference_signature"] = signatures[0]
        if not all(x["passed"] for x in validations):
            (OUT / "state_replay_validation_failed.json").write_text(
                json.dumps({"protocol_sha256": protocol_hash, "validations": validations,
                            "state_labels": [x["label"] for x in states]}, indent=2, allow_nan=False) + "\n",
                encoding="utf-8")
            print(json.dumps(validations, indent=2))
            raise RuntimeError("One or more physical states failed repeatability; action interventions aborted")

        actions = [[float(s), float(t)] for s in (-0.35, 0.0, 0.35) for t in (-0.5, 0.0, 0.5)]
        counterfactuals = []
        for state in states:
            prefix = episode[:int(state["prefix_decisions"])]
            reference = state["reference_signature"]
            for action in actions:
                actual_start, _ = run_prefix(env, base, case, prefix)
                start_error = signature_error(reference, actual_start)
                if any(start_error[k] > TOL[k] for k in ("position_m", "heading_rad", "speed_kmh", "route_completion", "observation")):
                    raise RuntimeError(f"Counterfactual did not start from matched state: {state['label']} {action} {start_error}")
                obs = np.asarray(actual_start["observation"], dtype=np.float32)
                step_rows = []
                for decision in range(MAX_HOLD):
                    obs, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
                    obs = np.asarray(obs, dtype=np.float32)
                    sig = physical_signature(env, base, obs, info)
                    step_rows.append({"decision": decision + 1, "action": action, "reward_v1": float(reward),
                                      "reward_breakdown": info["reward_breakdown"], "route_completion": sig["route_completion"],
                                      "delta_route": float(info["delta_route"]),
                                      "route_delta_m": float(info["delta_route"]) * float(base._env.agent.navigation.total_length),
                                      "speed_kmh": sig["speed_kmh"], "position_xy": sig["position_xy"],
                                      "heading_rad": sig["heading_rad"], "heading_error_rad": sig["heading_error_rad"],
                                      "lateral_error_m": sig["lateral_error_m"], "lane_width_m": sig["lane_width_m"],
                                      "estimated_body_edge_clearance_m": sig["estimated_body_edge_clearance_m"],
                                      "on_lane": sig["on_lane"], "terminated": bool(terminated), "truncated": bool(truncated),
                                      "outcome": info.get("outcome")})
                    if terminated or truncated:
                        break
                counterfactuals.append({"state_label": state["label"], "prefix_decisions": state["prefix_decisions"],
                                        "initial_state_error": start_error, "action": action, "steps": step_rows,
                                        "terminal_outcome": step_rows[-1]["outcome"] if step_rows else None,
                                        "probe_censored": bool(step_rows and not step_rows[-1]["terminated"] and not step_rows[-1]["truncated"]),
                                        "route_delta_m": float(sum(x["route_delta_m"] for x in step_rows)),
                                        "mean_speed_kmh": float(np.mean([x["speed_kmh"] for x in step_rows])) if step_rows else 0.0,
                                        "min_estimated_clearance_m": float(min(x["estimated_body_edge_clearance_m"] for x in step_rows)) if step_rows else None,
                                        "reward_v1_sum": float(sum(x["reward_v1"] for x in step_rows))})

        physical_artifact = {"protocol_sha256": protocol_hash, "source_trace_sha256": sha256(source),
                             "state_replay_validation": validations,
                             "states": [{"label": s["label"], "prefix_decisions": s["prefix_decisions"],
                                        "reference_signature": {k: v for k, v in s["reference_signature"].items() if k != "observation"}}
                                       for s in states],
                             "counterfactual_count": len(counterfactuals), "counterfactuals": counterfactuals}
        (OUT / "physical_counterfactuals.json").write_text(
            json.dumps(physical_artifact, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

        model_specs = {
            "P2D_A_10000": RUN / "arm_A/checkpoints/step_010000",
            "P2D_B_10000": RUN / "arm_B/checkpoints/step_010000",
            "P2F_A_10000": P2F / "arm_A/checkpoints/step_010000",
            "P2F_B_10000": P2F / "arm_B/checkpoints/step_010000",
        }
        checkpoint_provenance, critic_actor = eval_models(states, model_specs)
        payload = {
            "schema": "stage5_p2g_action_consequence_evidence_v1", "status": "COMPLETE",
            "protocol_sha256": protocol_hash, "protocol": protocol,
            "source_revision": "08c00e3b42ffe635a4d8170cb70fcb4397d8736c",
            "source_trace_sha256": sha256(source),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "runtime": {"python": sys.version.split()[0], "numpy": importlib.metadata.version("numpy"),
                        "torch": torch.__version__, "gymnasium": importlib.metadata.version("gymnasium"),
                        "stable_baselines3": importlib.metadata.version("stable-baselines3"),
                        "metadrive": importlib.metadata.version("metadrive-simulator") if any(d.metadata["Name"].lower() == "metadrive-simulator" for d in importlib.metadata.distributions()) else "runtime MetaDrive 0.4.3"},
            "wall_time_seconds": time.perf_counter() - started,
            "source_geometry": {"geometry_id": GEOMETRY_ID, "environment_seed": ENV_SEED,
                                 "geometry_generation_seed": 3, "split": "TRAIN/PILOT_TRAIN"},
            "source_episode": {k: v for k, v in scan.items() if k != "scan_rows"},
            "state_replay_validation": validations,
            "states": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in state.items() if k != "reference_signature"}
                       | {"reference_signature": {k: v for k, v in state["reference_signature"].items() if k != "observation"}}
                       for state in states],
            "checkpoint_provenance": checkpoint_provenance,
            "critic_actor_diagnostics": critic_actor,
            "counterfactual_count": len(counterfactuals), "counterfactuals": counterfactuals,
            "limits": {"short_horizon_decisions": MAX_HOLD, "counterfactuals_are_not_training_data": True,
                       "short_environment_return_is_not_soft_q": True, "estimated_clearance_is_not_MetaDrive_on_lane_contract": True,
                       "teacher_or_external_controller_used": False, "validation_test_accessed": False,
                       "benchmark_eligible": False},
        }
        out_path = OUT / "evidence.json"
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        checks = {"results/stage5/p2g_action_consequence_20261121/protocol.json": sha256(PROTOCOL_PATH),
                  "runs/stage5_p2d/s5_p2d_actor_activation_seed101_20261121/arm_A/replay_action_trace.jsonl": sha256(source),
                  "results/stage5/p2g_action_consequence_20261121/evidence.json": sha256(out_path)}
        for name, path in model_specs.items():
            relative = path.relative_to(ROOT).as_posix()
            checks[f"{relative}/model.zip"] = sha256(path / "model.zip")
            checks[f"{relative}/metadata.json"] = sha256(path / "metadata.json")
        (OUT / "CHECKSUMS.sha256").write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(checks.items())), encoding="utf-8")
        print(json.dumps({"status": "COMPLETE", "protocol_sha256": protocol_hash,
                          "state_count": len(states), "counterfactuals": len(counterfactuals),
                          "evidence": str(out_path), "evidence_sha256": sha256(out_path)}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
