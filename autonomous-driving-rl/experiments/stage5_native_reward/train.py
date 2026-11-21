"""Train a fresh Stage 5 SAC policy with MetaDrive's native reward."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.native_metadrive_sac.train import DiskBackedReplayBuffer
from src.agents.stage5_sac_agent import Stage5SacAgent
from src.platform.agent import AgentPublicEpisodeContext
from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER, task_producer_identity
from src.training.stage5.checkpoint import save_sac_checkpoint, verify_checkpoint_integrity
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.native_reward_training import (
    NATIVE_REWARD_TRAINING_MODE_ID,
    NativeMetaDriveRewardTrainingEnv,
)
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.training.stage5.pilot_observation_transform import (
    GROUPWISE_PHYSICAL_V1,
    GroupwisePhysicalObservation,
    apply_observation_transform,
    transform_fingerprint,
)
from src.training.stage5.train_sampler import SamplerStrategy, TrainCaseSampler


CONFIG_PATH = ROOT / "configs/experiments/stage5_native_reward_integration_v1.json"
GEOMETRY_ID = "geom_easy_SCS_seed0"
PRODUCER_ID = REFERENCE_TASK_PRODUCER
TRANSFORM_ID = GROUPWISE_PHYSICAL_V1["transform_id"]
CHECKPOINT_STEPS = [10000, 25000, 50000, 75000, 100000]
EVAL_SEEDS = [10101, 10102, 10103, 10104, 10105]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def runtime_info() -> dict[str, str]:
    import gymnasium
    import metadrive
    import stable_baselines3
    return {
        "python": sys.version,
        "metadrive_version": str(getattr(metadrive, "__version__", "0.4.3")),
        "stable_baselines3": stable_baselines3.__version__,
        "gymnasium": gymnasium.__version__,
        "torch": torch.__version__,
        "device": "cpu",
    }


def make_sampler(seed: int) -> TrainCaseSampler:
    return TrainCaseSampler(
        training_root_seed=seed,
        strategy=SamplerStrategy.FIXED,
        tier_filter="Easy",
        sequence_filter="SCS",
        fixed_geometry_seed=0,
        project_root=ROOT,
    )


def make_env(seed: int, *, explicit_case_seed: int | None = None):
    sampler = make_sampler(seed)
    base = Stage5TrainingEnv(
        sampler=sampler,
        project_root=ROOT,
        training_root_seed=seed,
        task_context_producer_id=PRODUCER_ID,
        simulator_render_mode="OFF",
    )
    reward_env = NativeMetaDriveRewardTrainingEnv(base)
    env = GroupwisePhysicalObservation(reward_env)
    if explicit_case_seed is None:
        observation, info = env.reset(seed=seed)
        case = base._current_case
    else:
        case = sampler.get_case_for_geometry_id(GEOMETRY_ID, environment_seed=explicit_case_seed)
        observation, info = env.reset(seed=explicit_case_seed, options={"case": case})
    if case is None:
        env.close()
        raise RuntimeError("Stage5 sampler did not resolve a case")
    return env, base, case, observation, info


def verify_preflight() -> dict[str, Any]:
    env, base, case, obs, reset_info = make_env(101)
    try:
        if tuple(obs.shape) != (409,) or not env.observation_space.contains(obs):
            raise RuntimeError(f"Stage5 observation contract mismatch: {obs.shape}")
        if base.task_context_producer_id != PRODUCER_ID:
            raise RuntimeError("Corrected F09 TaskContext producer is not active")
        if case.case_id.split("/")[1:4] != ["Easy", "SCS", "geom-0"]:
            raise RuntimeError(f"Unexpected selected case identity: {case.case_id}")
        if case.split != "TRAIN" or case.traffic_density != 0.0 or reset_info.get("traffic_density") != 0.0:
            raise RuntimeError("Selected geometry is not the registered traffic-free TRAIN case")
        if int(reset_info["horizon_steps"]) <= 0:
            raise RuntimeError("Invalid route-aware horizon")
        meta = env.unwrapped._env.config
        if int(meta["decision_repeat"]) != 5 or abs(float(meta["physics_world_step_size"]) - 0.02) > 1e-9:
            raise RuntimeError("MetaDrive control cadence differs from the 10 Hz / five-substep contract")
        action = np.asarray([0.17, 0.23], dtype=np.float32)
        next_obs, returned_reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            raise RuntimeError("Preflight action unexpectedly terminated the case")
        if not np.all(np.isfinite(next_obs)) or not env.observation_space.contains(next_obs):
            raise RuntimeError("Stage5 encoder produced invalid 409D observation")
        if abs(float(info["action_steering"]) - float(action[0])) > 1e-6:
            raise RuntimeError("Steering action fidelity failed")
        if abs(float(info["action_throttle_brake"]) - float(action[1])) > 1e-6:
            raise RuntimeError("Throttle/brake action fidelity failed")
        if info.get("training_reward_mode_id") != NATIVE_REWARD_TRAINING_MODE_ID:
            raise RuntimeError("Native reward mode ID is missing")
        if abs(float(returned_reward) - float(info["native_training_reward"])) > 1e-8:
            raise RuntimeError("Native reward wrapper changed MetaDrive reward")
        if abs(float(info["reward_spec_v1_comparison"]) - float(info["reward_breakdown"]["total_reward"])) > 1e-8:
            raise RuntimeError("RewardSpecV1 comparison telemetry mismatch")
        if not np.allclose(next_obs, apply_observation_transform(
            Stage5ObservationEncoder().encode(base.current_agent_input), TRANSFORM_ID
        ), rtol=0.0, atol=1e-6):
            raise RuntimeError("Observation transform path is inconsistent")
        return {
            "passed": True,
            "case_id": case.case_id,
            "geometry_id": GEOMETRY_ID,
            "geometry_sha256": case.geometry_sha256,
            "split": case.split,
            "traffic_density": case.traffic_density,
            "observation_shape": list(obs.shape),
            "task_context_producer_id": PRODUCER_ID,
            "native_reward_one_step": float(returned_reward),
            "reward_spec_v1_one_step": float(info["reward_spec_v1_comparison"]),
            "action_requested_and_executed": action.tolist(),
            "horizon_steps": int(reset_info["horizon_steps"]),
            "control_hz": 10,
            "decision_repeat": int(meta["decision_repeat"]),
            "physics_world_step_size": float(meta["physics_world_step_size"]),
            "native_reward_mode": env.env.reward_mode_identity,
        }
    finally:
        env.close()


class IntegrationCallback(BaseCallback):
    def __init__(self, run_dir: Path, training_config: dict[str, Any], protocol_sha: str, verbose: int = 0):
        super().__init__(verbose)
        self.run_dir = run_dir
        self.training_config = training_config
        self.protocol_sha = protocol_sha
        self.steps_file = None
        self.episode_file = run_dir / "training_episodes.jsonl"
        self.last_checkpoint = 0
        self.passed_checkpoint: str | None = None
        self.episodes = 0
        self.action_fidelity_checks = 0
        self.step_count = 0

    def _on_training_start(self) -> None:
        self.steps_file = (self.run_dir / "training_steps.csv").open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.steps_file, fieldnames=[
            "global_step", "case_id", "geometry_id", "environment_seed", "episode_step",
            "steering", "throttle_brake", "speed_km_h", "route_completion", "delta_route",
            "native_training_reward", "reward_spec_v1_comparison", "progress_reward", "time_cost",
            "terminal_reward", "terminated", "truncated", "clean_success", "primary_reason",
        ])
        self.csv_writer.writeheader()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        actions = np.asarray(self.locals.get("actions", []))
        rewards = np.asarray(self.locals.get("rewards", []))
        dones = np.asarray(self.locals.get("dones", []))
        if not infos:
            raise RuntimeError("SB3 callback did not receive Stage5 step info")
        info = infos[0]
        action = np.asarray(actions).reshape(-1, 2)[0]
        if abs(float(action[0]) - float(info["action_steering"])) > 1e-5 or abs(float(action[1]) - float(info["action_throttle_brake"])) > 1e-5:
            raise RuntimeError("Training action differs from action executed by Stage5TrainingEnv")
        self.action_fidelity_checks += 1
        self.step_count += 1
        breakdown = info["reward_breakdown"]
        training_reward = float(rewards.reshape(-1)[0])
        if not np.isclose(training_reward, float(info["native_training_reward"]), atol=1e-6, rtol=0.0):
            raise RuntimeError("SB3 replay reward differs from native MetaDrive reward")
        self.csv_writer.writerow({
            "global_step": int(self.model.num_timesteps),
            "case_id": info.get("case_id"),
            "geometry_id": info.get("geometry_id"),
            "environment_seed": info.get("environment_seed"),
            "episode_step": info.get("step_index"),
            "steering": float(action[0]),
            "throttle_brake": float(action[1]),
            "speed_km_h": info.get("speed_km_h"),
            "route_completion": info.get("route_completion"),
            "delta_route": info.get("delta_route"),
            "native_training_reward": float(info["native_training_reward"]),
            "reward_spec_v1_comparison": float(info["reward_spec_v1_comparison"]),
            "progress_reward": breakdown.get("progress_reward"),
            "time_cost": breakdown.get("time_cost"),
            "terminal_reward": breakdown.get("terminal_reward"),
            "terminated": bool(info.get("outcome", {}).get("terminated", False)),
            "truncated": bool(info.get("outcome", {}).get("truncated", False)),
            "clean_success": info.get("clean_success"),
            "primary_reason": info.get("primary_reason"),
        })
        if self.step_count % 500 == 0:
            self.steps_file.flush()
        if dones.size and bool(dones.reshape(-1)[0]):
            row = {
                "global_step": int(self.model.num_timesteps),
                "episode_index": self.episodes,
                "case_id": info.get("case_id"),
                "geometry_id": info.get("geometry_id"),
                "geometry_sha256": self.training_config["geometry_sha256"],
                "environment_seed": info.get("environment_seed"),
                "outcome": info.get("primary_reason"),
                "clean_success": bool(info.get("clean_success", False)),
                "route_progress_delta": (
                    float(info["route_completion"]) - float(info["initial_route_completion"])
                    if info.get("route_completion") is not None and info.get("initial_route_completion") is not None
                    else None
                ),
                "native_training_reward": info.get("native_training_reward"),
                "reward_spec_v1_comparison": info.get("reward_spec_v1_comparison"),
                "episode_native_return": info.get("episode", {}).get("r"),
                "episode_length": info.get("episode", {}).get("l"),
            }
            with self.episode_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            self.episodes += 1

        due = int(self.model.num_timesteps)
        if due in CHECKPOINT_STEPS and due != self.last_checkpoint:
            checkpoint = self.save_checkpoint(due)
            self.last_checkpoint = due
            eval_json = self.run_dir / "evaluations" / f"eval_{due:06d}.json"
            command = [
                sys.executable, str(Path(__file__).with_name("evaluate.py")),
                "--checkpoint", str(checkpoint), "--out", str(eval_json),
                "--seeds", ",".join(map(str, EVAL_SEEDS)),
            ]
            with eval_json.with_suffix(".log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            if completed.returncode != 0 or not eval_json.exists():
                raise RuntimeError(f"Stage5 evaluation failed at {due}: rc={completed.returncode}; see {eval_json.with_suffix('.log')}")
            result = json.loads(eval_json.read_text(encoding="utf-8"))
            clean = int(result["summary"]["clean_success"])
            print(f"STAGE5_EVAL step={due} clean={clean}/5 outcomes={result['summary']['outcomes']}", flush=True)
            if clean >= 4:
                self.passed_checkpoint = str(checkpoint)
                return False
        return True

    def save_checkpoint(self, step: int) -> Path:
        target = self.run_dir / "checkpoints" / f"step_{step:06d}"
        save_sac_checkpoint(
            model=self.model,
            checkpoint_dir=target,
            encoder=Stage5ObservationEncoder(),
            observation_transform=TRANSFORM_ID,
            total_timesteps=step,
            training_root_seed=101,
            training_config=self.training_config,
            training_summary={
                "status": "CHECKPOINT",
                "actual_environment_steps": step,
                "optimizer_updates": int(self.model._n_updates),
                "completed_training_episodes": self.episodes,
            },
            save_replay_buffer=False,
            project_root=ROOT,
            task_context_producer_id=PRODUCER_ID,
        )
        metadata = verify_checkpoint_integrity(
            target,
            expected_encoder_fingerprint=Stage5ObservationEncoder().get_schema_fingerprint(),
            task_context_producer_id=PRODUCER_ID,
        )
        if metadata["training_config"]["training_reward_mode"]["fingerprint"] != self.training_config["training_reward_mode"]["fingerprint"]:
            raise RuntimeError("Checkpoint reward-mode fingerprint mismatch")
        with (self.run_dir / "checkpoint_manifest.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "step": step,
                "model_sha256": metadata["model_sha256"],
                "encoder_fingerprint": metadata["encoder_fingerprint"],
                "policy_input_fingerprint": metadata["policy_input_fingerprint"],
                "task_context_producer": metadata["task_context_producer"],
                "reward_mode": self.training_config["training_reward_mode"],
                "protocol_sha256": self.protocol_sha,
                "optimizer_updates": int(self.model._n_updates),
            }, sort_keys=True) + "\n")
        return target

    def _on_training_end(self) -> None:
        if self.steps_file is not None:
            self.steps_file.flush()
            self.steps_file.close()
            self.steps_file = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty Stage5 integration run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["experiment_id"] != "stage5_native_reward_integration_v1":
        raise RuntimeError("Unexpected Stage5 reward mode configuration")
    config_sha = sha256_file(CONFIG_PATH)
    preflight = verify_preflight()
    reward_identity = preflight["native_reward_mode"]
    transform_spec = GROUPWISE_PHYSICAL_V1
    training_config = {
        "experiment_id": config["experiment_id"],
        "training_reward_mode": reward_identity,
        "task_context_producer_id": PRODUCER_ID,
        "task_context_producer": task_producer_identity(PRODUCER_ID),
        "observation_encoder_id": Stage5ObservationEncoder().encoder_id,
        "observation_encoder_fingerprint": Stage5ObservationEncoder().get_schema_fingerprint(),
        "observation_transform": transform_spec,
        "observation_transform_fingerprint": transform_fingerprint(transform_spec),
        "action_adapter_id": "continuous_box2_v1",
        "geometry_id": GEOMETRY_ID,
        "geometry_sha256": preflight["geometry_sha256"],
        "geometry_split": "TRAIN",
        "traffic_density": 0.0,
        "control_frequency_hz": 10,
        "sac_hyperparameters": config["sac"],
        "training_seeds": {"root_seed": 101, "environment_seed_rule": "TrainCaseSampler deterministic fixed-geometry derivation"},
        "config_sha256": config_sha,
    }
    protocol = {
        "protocol_id": config["experiment_id"],
        "status": "REGISTERED",
        "source_commit": git_sha(),
        "source_files": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (Path(__file__).resolve(), Path(__file__).with_name("evaluate.py").resolve(),
                         ROOT / "src/training/stage5/env_adapter.py", ROOT / "src/training/stage5/native_reward_training.py",
                         ROOT / "src/training/stage5/checkpoint.py", CONFIG_PATH)
        },
        "runtime": runtime_info(),
        "config_sha256": config_sha,
        "preflight": preflight,
        "training_config": training_config,
        "training_step_limit": 100000,
        "evaluation_steps": CHECKPOINT_STEPS,
        "evaluation_environment_seeds": EVAL_SEEDS,
        "pass_threshold": "At least 4/5 CLEAN_SUCCESS full episodes; stop at the first passing checkpoint.",
        "prohibitions": ["no native checkpoint weights", "no teacher", "no Stage 3", "no P2-H sampler", "no VALIDATION/TEST"],
    }
    protocol["protocol_sha256"] = canonical_sha(protocol)
    (run_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    torch.set_num_threads(int(args.torch_threads))
    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)

    env = DummyVecEnv([lambda: Monitor(GroupwisePhysicalObservation(
        NativeMetaDriveRewardTrainingEnv(Stage5TrainingEnv(
            sampler=make_sampler(101),
            training_root_seed=101,
            project_root=ROOT,
            task_context_producer_id=PRODUCER_ID,
            simulator_render_mode="OFF",
        ))
    ))])
    obs = env.reset()
    if obs.shape != (1, 409):
        env.close()
        raise RuntimeError(f"Stage5 SAC initial observation must be 409D, got {obs.shape}")
    replay_dir = run_dir / "replay_memmap"
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_entropy="auto",
        use_sde=False,
        policy_kwargs={"net_arch": [256, 256], "activation_fn": torch.nn.ReLU},
        replay_buffer_class=DiskBackedReplayBuffer,
        replay_buffer_kwargs={"storage_dir": str(replay_dir)},
        seed=101,
        device="cpu",
        verbose=0,
    )
    model.set_logger(configure(str(run_dir / "sb3_logs"), ["stdout", "csv"]))
    initial_dir = run_dir / "checkpoints" / "step_000000"
    save_sac_checkpoint(
        model=model,
        checkpoint_dir=initial_dir,
        encoder=Stage5ObservationEncoder(),
        observation_transform=TRANSFORM_ID,
        total_timesteps=0,
        training_root_seed=101,
        training_config=training_config,
        training_summary={"status": "INITIALIZATION", "actual_environment_steps": 0},
        save_replay_buffer=False,
        project_root=ROOT,
        task_context_producer_id=PRODUCER_ID,
    )
    preflight_agent = Stage5SacAgent(
        checkpoint_path=initial_dir,
        task_context_producer_id=PRODUCER_ID,
    )
    try:
        raw_env = env.envs[0].unwrapped
        current_input = raw_env.current_agent_input
        if current_input is None:
            raise RuntimeError("No real AgentInputV1 available for checkpoint reload preflight")
        preflight_agent.reset(AgentPublicEpisodeContext(), agent_seed=101)
        predicted = preflight_agent.act(current_input, task_context_producer_id=PRODUCER_ID).action_payload
        expected = model.predict(obs[0], deterministic=True)[0]
        if not np.allclose(np.asarray(predicted), np.asarray(expected), rtol=0.0, atol=1e-6):
            raise RuntimeError("Stage5 checkpoint reload changed deterministic actor action")
    finally:
        preflight_agent.close()
    # The actual training starts from a clean episode; preflight data is never
    # inserted into replay.
    obs = env.reset()
    callback = IntegrationCallback(run_dir, training_config, protocol["protocol_sha256"])
    started = time.perf_counter()
    failure = None
    try:
        model.learn(total_timesteps=100000, callback=callback, log_interval=100, progress_bar=False)
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        env.close()
        for name in ("observations", "next_observations", "actions", "rewards", "dones", "timeouts"):
            array = getattr(model.replay_buffer, name, None)
            if hasattr(array, "flush"):
                array.flush()
        final_dir = run_dir / "checkpoints" / f"step_{model.num_timesteps:06d}_final"
        save_sac_checkpoint(
            model=model,
            checkpoint_dir=final_dir,
            encoder=Stage5ObservationEncoder(),
            observation_transform=TRANSFORM_ID,
            total_timesteps=int(model.num_timesteps),
            training_root_seed=101,
            training_config=training_config,
            training_summary={"status": "FINAL", "actual_environment_steps": int(model.num_timesteps),
                              "optimizer_updates": int(model._n_updates), "failure": failure},
            save_replay_buffer=False,
            project_root=ROOT,
            task_context_producer_id=PRODUCER_ID,
        )
    elapsed = time.perf_counter() - started
    eval_files = sorted((run_dir / "evaluations").glob("eval_*.json"))
    evaluations = [json.loads(path.read_text(encoding="utf-8")) for path in eval_files]
    passing = [item for item in evaluations if item["summary"]["clean_success"] >= 4]
    decision = "PASS" if passing else ("FAIL" if model.num_timesteps >= 100000 else "INCOMPLETE")
    summary = {
        "status": "PASS_EARLY_STOP" if decision == "PASS" and model.num_timesteps < 100000 else decision,
        "decision": decision,
        "actual_environment_steps": int(model.num_timesteps),
        "optimizer_updates": int(model._n_updates),
        "completed_training_episodes": callback.episodes,
        "action_fidelity_checks": callback.action_fidelity_checks,
        "runtime_seconds": elapsed,
        "protocol_sha256": protocol["protocol_sha256"],
        "training_geometry_id": GEOMETRY_ID,
        "training_geometry_sha256": preflight["geometry_sha256"],
        "task_context_producer_id": PRODUCER_ID,
        "training_reward_mode": reward_identity,
        "selected_checkpoint": passing[0]["checkpoint"] if passing else None,
        "evaluations": [item["summary"] | {"checkpoint": item["checkpoint"], "step": item["checkpoint_timesteps"]} for item in evaluations],
        "failure": failure,
    }
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# Stage 5 SAC with native MetaDrive reward",
        "",
        f"Decision: **{decision}**",
        "",
        f"Environment steps: {summary['actual_environment_steps']}; optimizer updates: {summary['optimizer_updates']}; runtime: {elapsed/60:.1f} minutes.",
        "",
        "The learner used only fresh Stage 5 SAC initialization, 409D Stage5ObservationEncoder observations, corrected F09 TaskContext, ContinuousBox2 actions, and the versioned native MetaDrive training reward. RewardSpecV1 was computed and logged in parallel, not sent to SAC.",
        "",
        "| Checkpoint | Clean | Safety failures | Timeout | Mean route progress delta | Mean speed km/h | Peak speed km/h | Stationary fraction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in evaluations:
        s = item["summary"]
        report.append(f"| {item['checkpoint_timesteps']} | {s['clean_success']}/5 | {s['safety_failure']} | {s['timeout']} | {s['mean_route_progress_delta']:.4f} | {s['mean_speed_km_h']:.2f} | {s['mean_peak_speed_km_h']:.2f} | {s['mean_stationary_fraction']:.3f} |")
    report.extend(["", "RewardSpecV1 remains unchanged as an official contract. This run is development training only and does not alter benchmark eligibility.", ""])
    (run_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
