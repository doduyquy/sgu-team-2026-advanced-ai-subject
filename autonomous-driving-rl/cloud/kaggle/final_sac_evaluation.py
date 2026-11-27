"""Paired held-out evaluation of the locked SAC candidate outside canonical launcher execution."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
import multiprocessing
import os
from time import monotonic

from cloud.kaggle.common import ROOT, WORK, read, sha, verify, write
from src.launcher.cases import resolve_sandbox_case, resolve_test_cases, resolve_validation_cases
from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER


OUTPUT = WORK / "final-sac-evaluation"
V2 = WORK / "first-window-v2"
BUNDLE = V2 / "best-v2-extension-10000-bundle"
LOCK = WORK / "postwindow-product/cleanup-locked-v2/delivery-lock.json"
AUDIT = ROOT / "results/audits/evaluation_protocol"
KIND = "SAC_V2_PAIRED_TRAFFIC_EVALUATION"
CONTRACT = {"canonical_run": False, "evaluation_kind": KIND, "training_replicates": 1,
            "physics_dt_s": .02, "decision_repeat": 5, "control_frequency_hz": 10,
            "observation_dim": 409, "action_dim": 2, "traffic_context_m": 50, "lidar_m": 50,
            "task_context_producer": REFERENCE_TASK_PRODUCER, "terminal_preview": True,
            "observation_transform": "pilot_groupwise_physical_v1", "traffic_mode": "trigger",
            "deterministic": True, "action_intervention": False, "reward": "native_metadrive",
            "gradient_updates": 0, "replay_rows_admitted": 0}
SPLITS = {"VALIDATION": (resolve_validation_cases, 96, {5101, 5102}),
          "TEST": (resolve_test_cases, 60, set(range(9101, 9106)))}
CONDITIONS = ("ON", "OFF")


def atomic_json(path, document):
    temporary = path.with_suffix(".tmp")
    write(temporary, document)
    temporary.replace(path)


def pin():
    from src.launcher.registry import make_sac_agent_factory

    lock = read(LOCK)
    source = read(V2 / "extension-10000-result.json")["best"]
    if source["state"] != "stage-05000" or type(V2)(lock["full_state"]).name != source["state"]:
        raise ValueError("Selection and delivery lock must identify V2–5.000")
    digest = lock["model_sha256"]
    for path in (BUNDLE / "model.zip", V2 / "stage-05000/model.zip"):
        verify(path, digest)
    factory = make_sac_agent_factory(BUNDLE, task_context_producer_id=REFERENCE_TASK_PRODUCER, terminal_preview=True)
    provenance = factory.experiment_parameters()
    if provenance["training_state_sha256"] != lock["full_state_manifest_sha256"]:
        raise ValueError("Inference bundle does not bind the locked full state")
    return {**provenance, "selection": source["state"], "selection_manifest_sha256": sha(V2 / "extension-10000-result.json"),
            "delivery_lock_sha256": sha(LOCK), "bundle_metadata_sha256": sha(BUNDLE / "metadata.json")}


def density(case, traffic):
    return max(.08, case.traffic_density) if traffic == "ON" else 0.


def jobs_for(split):
    resolver, total, seeds = SPLITS[split]
    rows = sorted(resolver(ROOT), key=lambda case: case.protocol_order_index)
    if (len({case.case_id for case in rows}) != total or {case.environment_seed for case in rows} != seeds
            or [case.protocol_order_index for case in rows] != list(range(1, total + 1))
            or len({(case.geometry_sha256, case.environment_seed) for case in rows}) != total):
        raise ValueError(f"Unexpected case or seed identity in {split}")
    return [(case, traffic) for case in rows for traffic in CONDITIONS]


def initialize():
    from cloud.kaggle.local_workers import initialize as configure_worker

    configure_worker()
    os.environ["WANDB_MODE"] = "disabled"


@lru_cache(maxsize=1)
def agent_for(digest):
    from src.agents.stage5_terminal_preview_v1 import Stage5TerminalPreviewSacAgent

    return Stage5TerminalPreviewSacAgent(BUNDLE, expected_sha256=digest)


@lru_cache(maxsize=64)
def geometry_blocks(sequence, seed, digest):
    from src.launcher.cases import load_locked_geometry_block_sequence

    blocks, _, _ = load_locked_geometry_block_sequence({"sequence": sequence, "geometry_generation_seed": seed,
                                                       "geometry_sha256": digest}, ROOT)
    return blocks


def episode(case, traffic, digest, *, parity=False):
    import numpy as np
    import psutil
    from metadrive.envs.metadrive_env import MetaDriveEnv
    from src.launcher.executor import build_launcher_agent_input, build_metadrive_case_config, read_vehicle_speed_kmh
    from src.platform import AgentPublicEpisodeContext, get_action_adapter
    from src.platform.episode import REASON_TO_RAW_FLAG, classify_episode_outcome
    from src.platform.metrics import EpisodeRecord

    started = monotonic()
    agent = agent_for(digest)
    agent.reset(AgentPublicEpisodeContext(10, .1, case.horizon_steps, "STATE_DECISION_V1", "continuous_box2_v1", "INFERENCE"))
    blocks = geometry_blocks(case.sequence, case.geometry_generation_seed, case.geometry_sha256)
    effective_density = case.traffic_density if traffic == "DEFAULT" else density(case, traffic)
    env = MetaDriveEnv(build_metadrive_case_config(replace(case, traffic_density=effective_density), blocks))
    try:
        obs, initial = env.reset(seed=case.environment_seed)
        lidar = env.agent.config["lidar"]
        actual = (env.config["traffic_density"], env.current_seed, env.config["horizon"],
                  env.config["physics_world_step_size"], env.config["decision_repeat"], lidar["distance"], lidar["num_lasers"])
        if actual != (effective_density, case.environment_seed, case.horizon_steps, .02, 5, 50, 240):
            raise ValueError(f"Simulator config differs from the locked case: {actual}")
        geometry_identity = sha256(repr(env.current_map.get_meta_data()["block_sequence"]).encode()).hexdigest() if parity else None
        max_progress = float(initial.get("route_completion", 0.0))
        speeds, flags = [], {}
        native_return, preview_steps, elapsed_inference = 0., 0, 0.
        box = get_action_adapter("continuous_box2_v1")
        action_trace = sha256()
        npc_count = len(env.engine.traffic_manager.traffic_vehicles) + sum(
            len(group.vehicles) for group in env.engine.traffic_manager.block_triggered_vehicles)
        for step in range(1, case.horizon_steps + 1):
            state = build_launcher_agent_input(env, obs, step - 1, "STATE_DECISION_V1", REFERENCE_TASK_PRODUCER)
            tick = monotonic()
            decision = agent.act(state, task_context_producer_id=REFERENCE_TASK_PRODUCER)
            elapsed_inference += monotonic() - tick
            preview_steps += bool(decision.diagnostics["terminal_preview_activated"])
            action = box.to_canonical(decision.action_payload).to_numpy()
            if parity:
                np.testing.assert_array_equal(action, np.asarray(decision.action_payload, dtype=np.float32))
                action_trace.update(action.tobytes())
                if state.traffic_context.radius_m != 50:
                    raise ValueError("TrafficContext radius changed")
            obs, reward, terminated, truncated, info = env.step(action)
            flags.update({key: True for key in (*REASON_TO_RAW_FLAG.values(), "max_step") if info.get(key)})
            outcome = classify_episode_outcome(flags, terminated, truncated or step == case.horizon_steps)
            progress = float(info["route_completion"])
            max_progress = max(max_progress, progress)
            speeds.append(read_vehicle_speed_kmh(env.agent))
            native_return += float(reward)
            if outcome.is_done:
                break
        record = EpisodeRecord(case.tier, case.sequence, case.environment_seed, outcome.terminated, outcome.truncated,
                               outcome.primary_reason, bool(flags.get("arrive_dest")), outcome.clean_success,
                               progress, max_progress, *(bool(flags.get(key)) for key in
                               ("crash_vehicle", "crash_object", "crash_building", "crash_human", "crash_sidewalk", "out_of_road")),
                               step, step / 10, sum(speeds) / step, max(speeds), native_return,
                               step / 10 if outcome.clean_success else None)
        return {**case.to_dict(), **record.to_dict(), "traffic": traffic, "effective_density": effective_density,
                "model_sha256": digest, "terminal_preview_steps": preview_steps, "initial_npcs": npc_count,
                "mean_local_inference_ms": elapsed_inference * 1000 / step, "wall_seconds": monotonic() - started,
                "worker_pid": os.getpid(), "worker_rss_bytes": psutil.Process().memory_info().rss,
                **({"action_trace_sha256": action_trace.hexdigest(), "runtime_geometry_identity": geometry_identity} if parity else {})}
    finally:
        env.close()


def worker(case, traffic, digest):
    import traceback

    with open(os.devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
        try:
            return episode(case, traffic, digest)
        except Exception as error:
            return {"case_id": case.case_id, "traffic": traffic, "infrastructure_error": type(error).__name__,
                    "detail": str(error), "traceback": traceback.format_exc()}


def smoke(digest):
    from cloud.kaggle.runtime import ScheduledEnv, load_policy

    env = ScheduledEnv("eval", 0, 0, fixed=("geom_medium_SCTCS_seed11", 33701))
    case = env.sampler.get_case_for_geometry_id("geom_medium_SCTCS_seed11", environment_seed=33701)
    actor = load_policy(BUNDLE / "model.zip")
    actions = sha256()
    try:
        obs, _ = env.reset()
        for step in range(1, case.horizon_steps + 1):
            action, _ = actor.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            actions.update(action.tobytes())
            if terminated or truncated:
                break
    finally:
        env.close()
    on, off = (episode(case, condition, digest, parity=True) for condition in CONDITIONS)
    if (not on["clean_success"] or abs(on["final_route_completion"] - .9922) > .0002
            or on["simulation_time_s"] != 63.9 or step != on["episode_steps"]
            or info["primary_reason"] != on["primary_reason"] or info["route_completion"] != on["final_route_completion"]
            or actions.hexdigest() != on["action_trace_sha256"]
            or on["initial_npcs"] <= 0 or off["initial_npcs"] != 0
            or on["runtime_geometry_identity"] != off["runtime_geometry_identity"]):
        raise ValueError(f"Medium parity failed: {on}, {off}; reference {step}, {info}, {actions.hexdigest()}")
    return {"status": "PASS", "train_on": on, "train_off": off, "reference_action_trace_sha256": actions.hexdigest()}


def receipt_path(directory, case, traffic):
    return directory / "episodes" / f"{case.split.lower()}-{case.protocol_order_index:03d}-{traffic.lower()}.json"


def run_phase(pool, directory, jobs, digest):
    pending = {pool.submit(worker, case, traffic, digest): (case, traffic) for case, traffic in jobs
               if not receipt_path(directory, case, traffic).exists()}
    errors = []
    for index, future in enumerate(as_completed(pending), 1):
        case, traffic = pending[future]
        row = future.result()
        if "infrastructure_error" in row:
            errors.append(row)
            with (directory / "infrastructure.jsonl").open("a", encoding="utf-8") as log:
                import json
                log.write(json.dumps(row) + "\n")
        else:
            atomic_json(receipt_path(directory, case, traffic), row)
        if index % 12 == 0 or index == len(pending):
            import psutil
            print(case.split, f"attempted {index}/{len(pending)}", f"errors {len(errors)}",
                  f"free RAM {psutil.virtual_memory().available / 1024**3:.1f} GiB", flush=True)
    return errors


def main():
    from cloud.kaggle.smooth_traffic_eval import memory, worker_budget

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=type(OUTPUT), default=OUTPUT)
    parser.add_argument("--test-first", action="store_true")
    parser.add_argument("--workers", type=int, choices=(4, 6, 8))
    args = parser.parse_args()
    directory = args.output
    (directory / "episodes").mkdir(parents=True, exist_ok=True)
    initialize()
    initial_memory = memory()
    workers = args.workers or worker_budget(initial_memory)
    provenance = pin()
    digest = provenance["model_sha256"]
    jobs = {split: jobs_for(split) for split in SPLITS}
    plan = {**CONTRACT, "checkpoint": provenance, "runner_sha256": sha(type(OUTPUT)(__file__)),
            "locked_manifests": {split: sha(AUDIT / f"{split.lower()}_case_manifest.csv") for split in SPLITS},
            "cases": [{**case.to_dict(), "traffic": traffic, "effective_density": density(case, traffic)}
                      for split_jobs in jobs.values() for case, traffic in split_jobs]}
    execution_plan = directory / "execution-plan.json"
    if execution_plan.exists():
        if read(execution_plan) != plan:
            raise ValueError("Existing execution plan differs from frozen evaluation")
    else:
        atomic_json(execution_plan, plan)
    parity = directory / "parity-proof.json"
    if not parity.exists():
        atomic_json(parity, smoke(digest))
    if read(parity)["train_on"]["model_sha256"] != digest:
        raise ValueError("Parity checkpoint differs from evaluation checkpoint")
    print(f"Parity PASS. Starting {workers} workers", flush=True)
    started = monotonic()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=initialize) as pool:
        ordering = ("TEST", "VALIDATION") if args.test_first else tuple(SPLITS)
        for split in ordering:
            run_phase(pool, directory, jobs[split], digest)
        # Retry only missing outcomes. A committed crash, off-road, or timeout is final.
        for split in ordering:
            run_phase(pool, directory, jobs[split], digest)
    verify(BUNDLE / "model.zip", digest)
    verify(V2 / "stage-05000/model.zip", digest)
    atomic_json(directory / "execution-runtime.json", {"workers": workers, "seconds": monotonic() - started,
                "start_memory": initial_memory, "end_memory": memory(), "model_sha256_after": digest})


if __name__ == "__main__":
    main()
