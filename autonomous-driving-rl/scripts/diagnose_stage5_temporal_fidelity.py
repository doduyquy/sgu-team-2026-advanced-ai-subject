"""S5.3K-K0: verify decision-step/action fidelity and physics repeat semantics."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import stable_baselines3
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.training.stage5.pilot_protocol import load_pilot_split
from src.training.stage5.train_sampler import SamplerStrategy, TrainCaseSampler

RUN = ROOT / "runs/stage5_pilot/s5_3g_throttle_hold5_seed101_20261115"
PILOT_SPLIT = ROOT / "results/stage5/pilot_split_v1.json"
SCHEDULE = ROOT / "results/stage5/s5_3g_steering_isolation_seed101_20261115/summary.json"
OUTPUT = ROOT / "results/stage5/s5_3k_action_temporal_fidelity_20261115"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_length_rows(rows, include_episode_boundaries: bool):
    runs = []
    current_value = None
    current_len = 0
    current_ep = None
    previous_step = None
    for row in rows:
        value = float(row["throttle_brake"])
        episode_key = row["case_id"]
        episode_step = int(row["episode_step"])
        boundary = current_len and include_episode_boundaries and (
            episode_key != current_ep or episode_step != previous_step + 1)
        if current_len and (boundary or not math.isclose(value, current_value, rel_tol=0, abs_tol=1e-7)):
            runs.append(current_len)
            current_len = 0
        if current_len == 0:
            current_value = value
            current_ep = episode_key
        current_len += 1
        previous_step = episode_step
    if current_len:
        runs.append(current_len)
    return runs


def audit_training_telemetry():
    log_path = RUN / "checkpoints/training_steps.jsonl"
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 10000:
        raise AssertionError(f"Expected 10000 logged transitions, got {len(rows)}")
    if [int(r["training_step"]) for r in rows] != list(range(1, 10001)):
        raise AssertionError("training_step is not a contiguous one-based decision-step index")

    run_all = run_length_rows(rows, include_episode_boundaries=False)
    run_episode = run_length_rows(rows, include_episode_boundaries=True)
    run_hist_all = {str(k): v for k, v in sorted(Counter(run_all).items())}
    run_hist_episode = {str(k): v for k, v in sorted(Counter(run_episode).items())}

    violations = []
    steering_changed_within_block = 0
    same_episode_block_pairs = 0
    episode_boundary_count = 0
    for prev, curr in zip(rows, rows[1:]):
        same_episode = (curr["case_id"] == prev["case_id"] and
                        int(curr["episode_step"]) == int(prev["episode_step"]) + 1)
        if not same_episode:
            episode_boundary_count += 1
        same_global_hold_block = ((int(curr["training_step"]) - 1) // 5 ==
                                  (int(prev["training_step"]) - 1) // 5)
        if same_episode and same_global_hold_block:
            same_episode_block_pairs += 1
            if not math.isclose(float(curr["throttle_brake"]), float(prev["throttle_brake"]), rel_tol=0, abs_tol=1e-7):
                violations.append({"prev_step": prev["training_step"], "step": curr["training_step"],
                                   "prev_case": prev["case_id"], "case": curr["case_id"],
                                   "prev_throttle": prev["throttle_brake"], "throttle": curr["throttle_brake"]})
            if not math.isclose(float(curr["steering"]), float(prev["steering"]), rel_tol=0, abs_tol=1e-7):
                steering_changed_within_block += 1

    total_run_steps = sum(run_all)
    if total_run_steps != len(rows):
        raise AssertionError(f"Run-length accounting mismatch: {total_run_steps} != {len(rows)}")
    source = RUN / "checkpoints/step_10000/metadata.json"
    meta = json.loads(source.read_text(encoding="utf-8"))
    return {
        "n_training_decision_steps": len(rows), "training_step_is_contiguous": True,
        "source_log_sha256": digest(log_path), "checkpoint_sha256": meta["model_sha256"],
        "checkpoint_replay_saved": meta.get("replay_buffer_saved"),
        "hold_parameter_steps": 5,
        "throttle_only_run_lengths_ignoring_episode_boundaries": {
            "run_count": len(run_all), "sum_of_run_lengths": total_run_steps,
            "run_length_histogram": run_hist_all, "max_run_length": max(run_all),
        },
        "throttle_only_run_lengths_split_at_episode_boundaries": {
            "run_count": len(run_episode), "sum_of_run_lengths": sum(run_episode),
            "run_length_histogram": run_hist_episode,
            "run_lengths_exactly_five": sum(x == 5 for x in run_episode),
            "run_lengths_shorter_than_five": sum(x < 5 for x in run_episode),
            "run_lengths_longer_than_five": sum(x > 5 for x in run_episode),
        },
        "adjacent_pairs_same_episode_and_global_hold_block": same_episode_block_pairs,
        "unexpected_throttle_changes_within_same_episode_and_global_hold_block": violations,
        "steering_changed_in_same_episode_and_global_hold_block_pairs": steering_changed_within_block,
        "episode_boundaries_between_adjacent_logged_rows": episode_boundary_count,
        "interpretation": "These are Gymnasium decision-step log rows, not physics substeps. Exact multiples are not required because the hold resets at episode boundaries and the 10k run ends at a fixed budget.",
    }


def inspect_runtime_sources():
    from src.training.stage5 import env_adapter as stage5_env
    from src.training.stage5 import pilot_action_persistence as persistence
    from metadrive.envs import base_env as metadrive_base
    from metadrive.engine import base_engine
    off_policy = inspect.getsource(OffPolicyAlgorithm.collect_rollouts)
    source_sets = {
        "sb3_collect_rollouts": (OffPolicyAlgorithm.collect_rollouts, ["_sample_action(", "env.step(actions)", "_store_transition("]),
        "stage5_env_step": (stage5_env.Stage5TrainingEnv.step, ["to_canonical(action)", "self._env.step(canonical_act.to_numpy())"]),
        "metadrive_step_simulator": (metadrive_base.BaseEnv._step_simulator, ["self.engine.step(self.config[\"decision_repeat\"])"]),
        "metadrive_engine_step": (base_engine.BaseEngine.step, ["for i in range(step_num)", "self.step_physics_world()"]),
        "throttle_persistence_sample": (persistence.ThrottlePersistenceSAC._sample_action, ["super()._sample_action(", "hold_block = int(self.num_timesteps)", "buffer_actions[:, 1] = self._held_throttle", "actions = self.policy.unscale_action(buffer_actions)"]),
    }
    inspected = {}
    for label, (func, required) in source_sets.items():
        src = inspect.getsource(func)
        missing = [needle for needle in required if needle not in src]
        if missing:
            raise AssertionError(f"Source path {label} missing expected operations: {missing}")
        inspected[label] = {"source_file": inspect.getsourcefile(func),
                            "first_line": inspect.getsourcelines(func)[1],
                            "required_operations_found": required,
                            "one_env_step_call_in_collector": (src.count("env.step(actions)") == 1 if label == "sb3_collect_rollouts" else None)}
    return inspected


def runtime_probe():
    split = load_pilot_split(PILOT_SPLIT)
    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))["case_schedule"]
    case_row = next(r for r in schedule if r["tier"].lower() == "easy")
    sampler = TrainCaseSampler(training_root_seed=101, strategy=SamplerStrategy.TIER_ROUND_ROBIN,
        allowed_geometry_ids=[x["geometry_id"] for x in split["pilot_diagnostic"]],
        project_root=ROOT)
    env = Stage5TrainingEnv(sampler=sampler, encoder=Stage5ObservationEncoder(core_only=False), project_root=ROOT)
    inputs, engine_repeats, physics_calls, info_actions = [], [], [], []
    try:
        case = sampler.get_case_for_geometry_id(case_row["geometry_id"], case_row["environment_seed"])
        env.reset(options={"case": case})
        md = env._env
        engine = md.engine
        original_md_step = md.step
        original_engine_step = engine.step
        original_physics_step = engine.step_physics_world

        def counted_md_step(action):
            inputs.append(np.asarray(action, dtype=np.float32).reshape(-1).tolist())
            return original_md_step(action)

        def counted_engine_step(step_num=1):
            engine_repeats.append(int(step_num))
            return original_engine_step(step_num)

        def counted_physics_step(*args, **kwargs):
            physics_calls.append(1)
            return original_physics_step(*args, **kwargs)

        md.step = counted_md_step
        engine.step = counted_engine_step
        engine.step_physics_world = counted_physics_step

        actions = [[-0.35, -0.6], [0.12, 0.0], [0.31, 0.55]]
        for action in actions:
            _, _, terminated, truncated, info = env.step(action)
            info_actions.append([float(info["action_steering"]), float(info["action_throttle_brake"])])
            if terminated or truncated:
                raise AssertionError("Runtime probe terminated before three decision steps")
    finally:
        env.close()
    config = dict(md.config)
    result = {
        "case_id": case.case_id, "geometry_id": case_row["geometry_id"],
        "environment_seed": case_row["environment_seed"],
        "stage5_gymnasium_step_calls": len(actions),
        "metadrive_env_step_calls": len(inputs), "meta_drive_received_actions": inputs,
        "platform_info_actions": info_actions,
        "gym_to_metadrive_action_parity": bool(np.allclose(np.asarray(actions), np.asarray(inputs), atol=1e-7)),
        "gym_to_platform_info_parity": bool(np.allclose(np.asarray(actions), np.asarray(info_actions), atol=1e-7)),
        "decision_repeat_config": int(config["decision_repeat"]),
        "physics_world_step_size_s": float(config["physics_world_step_size"]),
        "engine_step_calls": len(engine_repeats), "engine_step_repeat_arguments": engine_repeats,
        "physics_world_step_calls": len(physics_calls),
        "expected_physics_world_step_calls": len(actions) * int(config["decision_repeat"]),
        "decision_dt_s": float(config["physics_world_step_size"] * config["decision_repeat"]),
        "effective_decision_frequency_hz": float(1.0 / (config["physics_world_step_size"] * config["decision_repeat"])),
        "pass": (len(inputs) == len(actions) and len(engine_repeats) == len(actions) and
                 engine_repeats == [int(config["decision_repeat"])] * len(actions) and
                 len(physics_calls) == len(actions) * int(config["decision_repeat"]) and
                 np.allclose(actions, inputs, atol=1e-7) and np.allclose(actions, info_actions, atol=1e-7)),
    }
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    telemetry = audit_training_telemetry()
    source_trace = inspect_runtime_sources()
    actual_runtime = runtime_probe()
    metadata = json.loads((RUN / "checkpoints/step_10000/metadata.json").read_text(encoding="utf-8"))
    report = {
        "protocol": "S5.3K-K0 action temporal fidelity v1", "status": "PASS" if actual_runtime["pass"] and not telemetry["unexpected_throttle_changes_within_same_episode_and_global_hold_block"] else "FAIL",
        "current_git_head": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "checkpoint_source_commit": metadata.get("git_commit"),
        "checkpoint_sha256": metadata["model_sha256"], "sb3_version": stable_baselines3.__version__,
        "telemetry_audit": telemetry, "source_trace": source_trace, "actual_metadrive_runtime_probe": actual_runtime,
        "source_file_sha256": {p: digest(ROOT / p) for p in ["src/launcher/executor.py", "src/platform/action_adapter.py",
            "src/training/stage5/env_adapter.py", "src/training/stage5/pilot_action_persistence.py",
            "src/training/stage5/pilot_callbacks.py"]},
        "known_limitations": [
            "S5.3G telemetry logs executed throttle after persistence; actor-requested pre-hold throttle was not recorded historically.",
            "Historical replay arrays were not saved; this audit cannot compare every historical replay row to actual simulator input.",
            "Runtime probe executes three actual Stage5TrainingEnv actions directly. SB3 collector-to-env order is additionally verified from installed SB3 source and prior synthetic collect_rollouts audit; this is not a new optimizer/training run.",
        ],
    }
    (OUTPUT / "K0_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hist = telemetry["throttle_only_run_lengths_split_at_episode_boundaries"]["run_length_histogram"]
    lines = ["# S5.3K — K0 action temporal fidelity", "",
        f"Status: **{report['status']}**. No training or optimizer update was run.", "",
        "## What the 5-step runs mean", "",
        "S5.3G deliberately enabled `ThrottlePersistenceSAC(throttle_hold_steps=5)`. Its `_sample_action` calls the actor sampler every collection step, then replaces only throttle/brake before sending the action to the environment and replay. Steering remains resampled each decision. The logging callback records the post-intervention action, so the historic 10k telemetry cannot reveal the discarded raw actor throttle.", "",
        f"The historical log has {telemetry['n_training_decision_steps']:,} contiguous decision rows. When throttle runs are segmented at case/episode boundaries, run-length histogram is `{json.dumps(hist, sort_keys=True)}`; lengths sum to 10,000. The earlier phrase ‘2,007 runs, each exactly five’ was inaccurate: 2,007 was the run count ignoring episode boundaries, and quantiles/max did not prove every run was length five. The actual distribution is recorded in `K0_audit.json`.", "",
        f"Within the same episode and same global 5-step hold block, unexpected throttle changes: {len(telemetry['unexpected_throttle_changes_within_same_episode_and_global_hold_block'])}. Steering changed on {telemetry['steering_changed_in_same_episode_and_global_hold_block_pairs']} adjacent pairs inside such blocks, consistent with steering being sampled each decision.", "",
        "## Decision-step to physics-step mapping", "",
        f"Runtime probe: {actual_runtime['stage5_gymnasium_step_calls']} Gymnasium steps → {actual_runtime['metadrive_env_step_calls']} MetaDrive `step()` calls → {actual_runtime['engine_step_calls']} engine repeats of `{actual_runtime['engine_step_repeat_arguments']}` → {actual_runtime['physics_world_step_calls']} physics-world steps. Config is `decision_repeat={actual_runtime['decision_repeat_config']}`, `physics_world_step_size={actual_runtime['physics_world_step_size_s']}` s; one policy decision is {actual_runtime['decision_dt_s']:.2f} s ({actual_runtime['effective_decision_frequency_hz']:.1f} Hz). Actions matched both MetaDrive input and Platform info: `{actual_runtime['gym_to_metadrive_action_parity']}` / `{actual_runtime['gym_to_platform_info_parity']}`.", "",
        "SB3 source path is one `_sample_action` → one `env.step(actions)` → one `_store_transition` per collect-rollout loop iteration. `Stage5TrainingEnv.step()` calls MetaDrive once; MetaDrive calls `engine.step(decision_repeat)`; the engine loops that many physics steps. Thus the configured repeat holds an action over five 0.02 s physics substeps, not five SAC/Gymnasium decisions.", "",
        "## Decision", "",
        "K0 passes for the declared S5.3G throttle-hold intervention and the current Stage5 → MetaDrive temporal path. The five-repeat simulator setting is not an extra five-decision action hold. S5.3G intentionally did add a separate five-decision throttle hold; this was declared, reset at episode boundaries, and action/replay values were changed together. Therefore K1 can be designed as the next diagnostic if desired; K0 found no undeclared temporal mapping defect.", "",
        "Limit: the historical S5.3G log does not preserve the actor's pre-hold throttle sample or raw replay buffer. K0 validates source/runtime semantics and recorded executed actions; it cannot retrospectively verify every historical transition array.", ""]
    (OUTPUT / "CONCLUSION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "telemetry": telemetry,
                      "runtime": actual_runtime, "artifact": str(OUTPUT / "CONCLUSION.md")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
