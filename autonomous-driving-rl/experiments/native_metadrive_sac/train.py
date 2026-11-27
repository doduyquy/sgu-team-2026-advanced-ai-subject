from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import Wrapper
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.version import VERSION as METADRIVE_VERSION


EXPERIMENT_ID = "native_metadrive_sac_single_map_v1"
FIXED_ENV_CONFIG = {
    "use_render": False,
    "traffic_density": 0.0,
    # Replaced before training by a frozen PGMap block sequence captured once.
    "map": 3,
    "random_spawn_lane_index": False,
    "random_lane_width": False,
    "random_lane_num": False,
    "accident_prob": 0.0,
    "horizon": 1000,
    "truncate_as_terminate": False,
    "start_seed": 101,
    "num_scenarios": 1,
}
TRAINING_SEED = 101
EVAL_SEEDS = [10101, 10102, 10103, 10104, 10105]
CHECKPOINT_STEPS = [100_000, 300_000, 500_000, 1_000_000]
MAX_STEPS = 1_000_000
ACTIVE_ENV_CONFIG: dict[str, Any] = dict(FIXED_ENV_CONFIG)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def geometry_fingerprint(env) -> str:
    network = env.unwrapped.current_map.road_network
    rows = []
    for source in sorted(network.graph, key=str):
        for destination in sorted(network.graph[source], key=str):
            for lane in network.graph[source][destination]:
                length = float(lane.length)
                positions = [lane.position(float(s), 0.0).tolist() for s in (0.0, length / 2.0, length)]
                rows.append({
                    "source": str(source),
                    "destination": str(destination),
                    "lane_index": int(lane.index[-1]),
                    "length_m": round(length, 6),
                    "centerline_samples_m": [[round(float(v), 6) for v in p] for p in positions],
                })
    return sha256_json(rows)


def repository_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


class NativeTelemetry(Wrapper):
    """Adds diagnostics to info without changing observation, action, reward, or done flags."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["native_telemetry"] = self.snapshot()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["native_telemetry"] = self.snapshot()
        return obs, reward, terminated, truncated, info

    def snapshot(self) -> dict[str, Any]:
        vehicle = self.unwrapped.agent
        lane = getattr(vehicle, "lane", None)
        lateral = None
        longitudinal = None
        if lane is not None:
            longitudinal, lateral = lane.local_coordinates(vehicle.position)
        return {
            "x": float(vehicle.position[0]),
            "y": float(vehicle.position[1]),
            "heading": float(vehicle.heading_theta),
            "speed_km_h": float(vehicle.speed_km_h),
            "lateral_error_m": None if lateral is None else float(lateral),
            "lane_longitudinal_m": None if longitudinal is None else float(longitudinal),
            "route_completion": float(vehicle.navigation.route_completion),
        }


class DiskBackedReplayBuffer:
    """Factory returning an SB3 ReplayBuffer-compatible disk-backed allocation.

    The returned object is an SB3 ReplayBuffer instance whose arrays are replaced
    before collection with memmaps. SB3 add/sample and timeout handling are unchanged.
    """

    def __new__(
        cls,
        buffer_size,
        observation_space,
        action_space,
        device="auto",
        n_envs=1,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
        *,
        storage_dir: str,
        **kwargs,
    ):
        if optimize_memory_usage:
            raise ValueError("This run uses the full SB3 replay layout to preserve timeout bootstrap semantics")
        if kwargs:
            raise TypeError(f"Unexpected ReplayBuffer arguments: {sorted(kwargs)}")
        from stable_baselines3.common.buffers import ReplayBuffer
        from stable_baselines3.common.buffers import BaseBuffer

        obj = object.__new__(ReplayBuffer)
        BaseBuffer.__init__(obj, buffer_size, observation_space, action_space, device, n_envs=n_envs)
        obj.buffer_size = max(buffer_size // n_envs, 1)
        obj.optimize_memory_usage = False
        obj.handle_timeout_termination = handle_timeout_termination
        path = Path(storage_dir)
        path.mkdir(parents=True, exist_ok=True)
        shapes_and_dtypes = {
            "observations": ((obj.buffer_size, obj.n_envs, *obj.obs_shape), observation_space.dtype),
            "next_observations": ((obj.buffer_size, obj.n_envs, *obj.obs_shape), observation_space.dtype),
            "actions": ((obj.buffer_size, obj.n_envs, obj.action_dim), ReplayBuffer._maybe_cast_dtype(action_space.dtype)),
            "rewards": ((obj.buffer_size, obj.n_envs), np.float32),
            "dones": ((obj.buffer_size, obj.n_envs), np.float32),
            "timeouts": ((obj.buffer_size, obj.n_envs), np.float32),
        }
        for name, (shape, dtype) in shapes_and_dtypes.items():
            target = np.lib.format.open_memmap(
                path / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
            )
            setattr(obj, name, target)
        obj.memmap_storage_dir = str(path.resolve())
        return obj


def make_raw_env(config: dict[str, Any] | None = None):
    return NativeTelemetry(MetaDriveEnv(copy.deepcopy(ACTIVE_ENV_CONFIG if config is None else config)))


def freeze_training_geometry() -> list[dict[str, Any]]:
    """Resolve MetaDrive's S block once, then freeze its sampled block length."""
    probe_config = dict(FIXED_ENV_CONFIG)
    probe_config["map"] = "S"
    probe_config["start_seed"] = TRAINING_SEED
    probe_config["num_scenarios"] = 1
    probe = NativeTelemetry(MetaDriveEnv(copy.deepcopy(probe_config)))
    try:
        probe.reset(seed=TRAINING_SEED)
        sequence = probe.unwrapped.current_map.get_meta_data()["block_sequence"]
        return json.loads(json.dumps(sequence))
    finally:
        probe.close()


def configure_frozen_geometry(sequence: list[dict[str, Any]]) -> None:
    global ACTIVE_ENV_CONFIG
    ACTIVE_ENV_CONFIG = dict(FIXED_ENV_CONFIG)
    ACTIVE_ENV_CONFIG["map"] = 3
    ACTIVE_ENV_CONFIG["map_config"] = {
        "type": "pg_map_file",
        "config": json.loads(json.dumps(sequence)),
        "lane_width": 3.5,
        "lane_num": 3,
        "exit_length": 50,
        "start_position": [0, 0],
    }


class CheckpointAndEvalCallback(BaseCallback):
    def __init__(
        self,
        run_dir: Path,
        eval_script: Path,
        eval_seeds: list[int],
        geometry_path: Path,
        protocol_sha256: str,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.run_dir = run_dir
        self.eval_script = eval_script
        self.eval_seeds = eval_seeds
        self.geometry_path = geometry_path
        self.protocol_sha256 = protocol_sha256
        self.next_index = 0
        self.episode_file = run_dir / "training_episodes.jsonl"
        self.episode_rows = 0
        self.passing_checkpoint: str | None = None
        self.episode_index = 0
        self.episode_speed: list[float] = []
        self.episode_route: list[float] = []
        self.episode_actions: list[list[float]] = []
        self.episode_start_route: float | None = None

    def _on_training_start(self) -> None:
        self.episode_file.parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        actions = self.locals.get("actions", None)
        if infos:
            telem = infos[0].get("native_telemetry", {})
            if telem:
                speed = float(telem.get("speed_km_h", 0.0))
                route = float(telem.get("route_completion", 0.0))
                if self.episode_start_route is None:
                    self.episode_start_route = route
                self.episode_speed.append(speed)
                self.episode_route.append(route)
            if actions is not None:
                action = np.asarray(actions).reshape(-1, 2)[0]
                self.episode_actions.append([float(action[0]), float(action[1])])
        if len(infos) and len(dones) and bool(dones[0]):
            info = dict(infos[0])
            outcome = "UNKNOWN_TERMINAL"
            if info.get("arrive_dest") and not info.get("out_of_road") and not info.get("crash"):
                outcome = "CLEAN_SUCCESS"
            elif info.get("out_of_road"):
                outcome = "OUT_OF_ROAD"
            elif any(info.get(key) for key in ("crash", "crash_vehicle", "crash_object", "crash_building", "crash_sidewalk", "crash_human")):
                outcome = "CRASH"
            elif info.get("max_step") or info.get("TimeLimit.truncated"):
                outcome = "TIMEOUT"
            speeds = np.asarray(self.episode_speed, dtype=np.float64)
            route_delta = None
            if len(self.episode_route) and self.episode_start_route is not None:
                route_delta = float(self.episode_route[-1] - self.episode_start_route)
            actions_array = np.asarray(self.episode_actions, dtype=np.float64)
            throttle = actions_array[:, 1] if len(actions_array) else np.asarray([], dtype=np.float64)
            positive_runs = []
            run = 0
            for value in throttle:
                if value > 0.05:
                    run += 1
                elif run:
                    positive_runs.append(run)
                    run = 0
            if run:
                positive_runs.append(run)
            row = {
                "step": int(self.num_timesteps),
                "episode_index": self.episode_index,
                "outcome": outcome,
                "episode": info.get("episode", {}),
                "route_progress_delta": route_delta,
                "mean_speed_km_h": float(speeds.mean()) if len(speeds) else None,
                "peak_speed_km_h": float(speeds.max()) if len(speeds) else None,
                "stationary_fraction_lt_1km_h": float(np.mean(speeds < 1.0)) if len(speeds) else None,
                "mean_steering": float(actions_array[:, 0].mean()) if len(actions_array) else None,
                "mean_throttle_brake": float(throttle.mean()) if len(throttle) else None,
                "positive_throttle_fraction": float(np.mean(throttle > 0.05)) if len(throttle) else None,
                "longest_positive_throttle_run": max(positive_runs, default=0),
                "outcome_flags": {
                    key: info.get(key) for key in (
                        "arrive_dest", "out_of_road", "crash", "crash_vehicle", "crash_object",
                        "crash_building", "crash_sidewalk", "crash_human", "max_step", "TimeLimit.truncated"
                    ) if key in info
                },
                "terminal_info": info.get("terminal_info", None),
            }
            with self.episode_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            self.episode_rows += 1
            self.episode_index += 1
            self.episode_speed.clear()
            self.episode_route.clear()
            self.episode_actions.clear()
            self.episode_start_route = None

        while self.next_index < len(CHECKPOINT_STEPS) and self.num_timesteps >= CHECKPOINT_STEPS[self.next_index]:
            requested_step = CHECKPOINT_STEPS[self.next_index]
            if self.num_timesteps != requested_step:
                raise RuntimeError(f"checkpoint callback overshot: expected={requested_step} actual={self.num_timesteps}")
            self._save_and_evaluate(requested_step)
            self.next_index += 1
        return self.passing_checkpoint is None

    def _save_and_evaluate(self, step: int) -> None:
        checkpoint = self.run_dir / "checkpoints" / f"sac_{step:07d}"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(checkpoint))
        zip_path = checkpoint.with_suffix(".zip")
        checkpoint_sha = sha256_file(zip_path)
        record = {
            "step": step,
            "optimizer_updates": int(self.model._n_updates),
            "checkpoint_path": str(zip_path),
            "checkpoint_sha256": checkpoint_sha,
            "protocol_sha256": self.protocol_sha256,
            "model_class": type(self.model).__qualname__,
            "policy_class": type(self.model.policy).__qualname__,
            "source_commit": repository_sha(),
        }
        with (self.run_dir / "checkpoint_manifest.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"CHECKPOINT step={step} sha256={checkpoint_sha} updates={self.model._n_updates}", flush=True)
        eval_json = self.run_dir / "evaluations" / f"eval_{step:07d}.json"
        eval_json.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(self.eval_script), "--checkpoint", str(zip_path), "--out", str(eval_json),
            "--step", str(step), "--seeds", ",".join(map(str, self.eval_seeds)),
            "--geometry", str(self.geometry_path),
        ]
        log_path = eval_json.with_suffix(".log")
        with log_path.open("w", encoding="utf-8") as stdout:
            completed = subprocess.run(command, stdout=stdout, stderr=subprocess.STDOUT, check=False)
        if completed.returncode != 0 or not eval_json.exists():
            raise RuntimeError(f"evaluation process failed at {step}: rc={completed.returncode}; see {log_path}")
        result = json.loads(eval_json.read_text(encoding="utf-8"))
        clean = int(result["summary"]["clean_success"])
        print(f"EVALUATION step={step} clean={clean}/5 outcomes={result['summary']['outcomes']}", flush=True)
        if clean >= 4:
            self.passing_checkpoint = str(result["checkpoint"])


def build_protocol(run_dir: Path, smoke_steps: int | None, geometry_sequence: list[dict[str, Any]]) -> dict[str, Any]:
    import gymnasium
    import stable_baselines3
    import metadrive
    from metadrive.envs.metadrive_env import METADRIVE_DEFAULT_CONFIG

    default = MetaDriveEnv.default_config()
    native_reward = {key: default[key] for key in (
        "success_reward", "out_of_road_penalty", "crash_vehicle_penalty", "crash_object_penalty",
        "crash_sidewalk_penalty", "driving_reward", "speed_reward", "use_lateral_reward",
    )}
    protocol = {
        "experiment_id": EXPERIMENT_ID,
        "status": "SMOKE_ONLY" if smoke_steps else "REGISTERED",
        "source_commit": repository_sha(),
        "python": sys.version,
        "python_executable": sys.executable,
        "metadrive_version": METADRIVE_VERSION,
        "metadrive_source": str(Path(metadrive.__file__).resolve()),
        "metadrive_source_revision": subprocess.check_output(
            ["git", "-C", str(Path(metadrive.__file__).resolve().parent.parent), "rev-parse", "HEAD"], text=True
        ).strip(),
        "metadrive_config_source_sha256": sha256_file(Path(metadrive.__file__).parent / "envs" / "metadrive_env.py"),
        "experiment_source_sha256": {
            "train.py": sha256_file(Path(__file__).resolve()),
            "evaluate.py": sha256_file(Path(__file__).with_name("evaluate.py").resolve()),
        },
        "stable_baselines3": stable_baselines3.__version__,
        "gymnasium": gymnasium.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
        "native_default_reward": native_reward,
        "environment_overrides": ACTIVE_ENV_CONFIG,
        "native_observation_expected_shape": [259],
        "native_action_expected_shape": [2],
        "fixed_geometry": "Frozen MetaDrive PGMap block sequence resolved once from S at training seed 101; same exact block IDs and length are replayed for every train/evaluation episode.",
        "geometry_block_sequence": geometry_sequence,
        "geometry_block_sequence_sha256": sha256_json(geometry_sequence),
        "training_seed": TRAINING_SEED,
        "evaluation_seeds": EVAL_SEEDS,
        "max_environment_steps": smoke_steps or MAX_STEPS,
        "scheduled_checkpoints": CHECKPOINT_STEPS if not smoke_steps else [],
        "eval_episodes_per_checkpoint": 5,
        "pass_rule": "At least 4/5 CLEAN_SUCCESS full deterministic episodes at any scheduled checkpoint.",
        "training": {
            "algorithm": "Stable-Baselines3 SAC",
            "policy": "MlpPolicy",
            "net_arch": [256, 256],
            "activation": "SB3 default ReLU",
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "learning_starts": 10_000,
            "buffer_size": 1_000_000,
            "batch_size": 256,
            "tau": 0.005,
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": "auto",
            "target_entropy": "auto (-action_dim)",
            "use_sde": False,
            "replay_storage": "SB3 ReplayBuffer API and sampling; full arrays disk-backed memmaps to fit RAM",
        },
        "paper_comparison": {
            "haarnoja_2018_applications_learning_rate": 3e-4,
            "requested_learning_rate": 1e-4,
            "learning_rate_is_paper_exact": False,
            "replay_size_paper": 1_000_000,
            "batch_size_paper": 256,
            "target_smoothing_tau_paper": 0.005,
            "gamma_paper": 0.99,
            "warmup_note": "10k random-action steps is common in reference implementations (e.g. Spinning Up start_steps), not claimed as the Applications paper Table 1 setting.",
            "sb3_notes": [
                "SB3 2.9.2 default learning_starts is 100; this run overrides it to 10000.",
                "SB3 MlpPolicy defaults to two 256-unit ReLU layers for pi and qf when net_arch=[256,256] is given.",
                "SB3 auto entropy coefficient learns log(alpha), with automatic target entropy -action_dim.",
                "SB3 uses squashed Gaussian continuous actions, twin critics/target critics, train_freq=1 and gradient_steps=1 here.",
                "SB3 updates occur after each collected step once num_timesteps exceeds learning_starts; warm-up actions are uniform random.",
            ],
        },
        "prohibited_sources": ["Stage 3", "teacher", "demonstration", "pretrained checkpoint", "Platform V1", "Stage 5"],
    }
    protocol["protocol_sha256"] = sha256_json(protocol)
    return protocol


def validate_runtime() -> dict[str, Any]:
    env = make_raw_env()
    try:
        obs, info = env.reset(seed=TRAINING_SEED)
        action_space = env.action_space
        observation_space = env.observation_space
        if METADRIVE_VERSION != "0.4.3":
            raise RuntimeError(f"Expected MetaDrive 0.4.3, found {METADRIVE_VERSION}")
        if tuple(obs.shape) != (259,) or not observation_space.contains(obs):
            raise RuntimeError(f"Unexpected native observation: shape={obs.shape} space={observation_space}")
        if tuple(action_space.shape) != (2,) or not np.allclose(action_space.low, -1) or not np.allclose(action_space.high, 1):
            raise RuntimeError(f"Unexpected native action space: {action_space}")
        if float(env.unwrapped.config["traffic_density"]) != 0.0:
            raise RuntimeError("traffic is not disabled")
        telemetry = info.get("native_telemetry", {})
        return {
            "observation_shape": list(obs.shape),
            "observation_dtype": str(obs.dtype),
            "observation_space": repr(observation_space),
            "action_space": repr(action_space),
            "reset_route_completion": telemetry.get("route_completion"),
            "resolved_map_config": env.unwrapped.config["map_config"].get_serializable_dict(),
            "geometry_sha256": geometry_fingerprint(env),
            "reward_config": {k: env.unwrapped.config[k] for k in (
                "success_reward", "out_of_road_penalty", "crash_vehicle_penalty", "crash_object_penalty",
                "crash_sidewalk_penalty", "driving_reward", "speed_reward", "use_lateral_reward",
            )},
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--smoke-steps", type=int, default=None, help="Technical smoke only; never a scientific result")
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)

    geometry_sequence = freeze_training_geometry()
    configure_frozen_geometry(geometry_sequence)
    geometry_path = run_dir / "geometry.json"
    geometry_path.write_text(json.dumps({"block_sequence": geometry_sequence}, indent=2), encoding="utf-8")
    runtime = validate_runtime()
    protocol = build_protocol(run_dir, args.smoke_steps, geometry_sequence)
    protocol["runtime_probe"] = runtime
    protocol["protocol_sha256"] = sha256_json({k: v for k, v in protocol.items() if k != "protocol_sha256"})
    (run_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")

    env = DummyVecEnv([lambda: Monitor(make_raw_env())])
    obs = env.reset()
    if obs.shape != (1, 259):
        env.close()
        raise RuntimeError(f"SB3 vector environment changed native observation shape to {obs.shape}")

    if args.smoke_steps is not None and args.smoke_steps < 1:
        raise ValueError("--smoke-steps must be positive")
    steps = args.smoke_steps if args.smoke_steps is not None else MAX_STEPS
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
        policy_kwargs={"net_arch": [256, 256]},
        replay_buffer_class=DiskBackedReplayBuffer,
        replay_buffer_kwargs={"storage_dir": str(replay_dir)},
        tensorboard_log=None,
        verbose=0,
        seed=TRAINING_SEED,
        device="cpu",
    )
    model.set_logger(configure(str(run_dir / "sb3_logs"), ["stdout", "csv"]))
    initial_path = run_dir / "checkpoints" / "sac_0000000.zip"
    initial_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(initial_path.with_suffix("")))
    initial_sha = sha256_file(initial_path)
    protocol["initial_checkpoint_sha256"] = initial_sha
    protocol["protocol_sha256"] = sha256_json({k: v for k, v in protocol.items() if k != "protocol_sha256"})
    (run_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "initial_checkpoint.json").write_text(json.dumps({"path": str(initial_path), "sha256": initial_sha}, indent=2), encoding="utf-8")
    with (run_dir / "checkpoint_manifest.jsonl").open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "step": 0,
            "optimizer_updates": 0,
            "checkpoint_path": str(initial_path),
            "checkpoint_sha256": initial_sha,
            "protocol_sha256": protocol["protocol_sha256"],
            "model_class": type(model).__qualname__,
            "policy_class": type(model.policy).__qualname__,
            "source_commit": repository_sha(),
        }, sort_keys=True) + "\n")
    eval_script = Path(__file__).with_name("evaluate.py").resolve()
    callback = CheckpointAndEvalCallback(
        run_dir, eval_script, EVAL_SEEDS, geometry_path, protocol["protocol_sha256"]
    )

    started = time.time()
    failure = None
    try:
        model.learn(total_timesteps=steps, callback=callback, log_interval=100, progress_bar=False)
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        env.close()
        for arr_name in ("observations", "next_observations", "actions", "rewards", "dones", "timeouts"):
            arr = getattr(model.replay_buffer, arr_name, None)
            if hasattr(arr, "flush"):
                arr.flush()
        model.save(str(run_dir / "checkpoints" / f"sac_final_{model.num_timesteps:07d}"))

    elapsed = time.time() - started
    summary = {
        "status": "SMOKE_ONLY" if args.smoke_steps else (
            "INCOMPLETE" if failure else ("PASS_EARLY_STOP" if callback.passing_checkpoint else "TRAINING_COMPLETE")
        ),
        "failure": failure,
        "actual_environment_steps": int(model.num_timesteps),
        "optimizer_updates": int(model._n_updates),
        "completed_training_episodes_logged": callback.episode_rows,
        "geometry": "MetaDrive map block sequence S",
        "training_seed": TRAINING_SEED,
        "runtime_seconds": elapsed,
        "runtime_hours": elapsed / 3600,
        "replay_size": int(model.replay_buffer.size()),
        "replay_storage": str(replay_dir),
        "protocol_sha256": protocol["protocol_sha256"],
        "initial_checkpoint_sha256": initial_sha,
        "last_checkpoint_sha256": sha256_file(run_dir / "checkpoints" / f"sac_final_{model.num_timesteps:07d}.zip"),
    }
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if not args.smoke_steps and not failure:
        evals = []
        for step in CHECKPOINT_STEPS:
            path = run_dir / "evaluations" / f"eval_{step:07d}.json"
            if path.exists():
                evals.append(json.loads(path.read_text(encoding="utf-8")))
        passing = [e for e in evals if e["summary"]["clean_success"] >= 4]
        summary["decision"] = "PASS" if passing else ("FAIL" if model.num_timesteps == MAX_STEPS else "INCOMPLETE")
        summary["selected_checkpoint"] = None if not passing else passing[0]["checkpoint"]
        summary["early_stop_on_pass"] = callback.passing_checkpoint is not None
        (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        report = [
            "# Native MetaDrive SAC baseline result",
            "",
            f"Decision: **{summary['decision']}**",
            "",
            f"Actual training steps: {summary['actual_environment_steps']:,}; optimizer updates: {summary['optimizer_updates']:,}.",
            "PASS requires at least 4/5 clean full episodes at a scheduled evaluation checkpoint. The selected checkpoint is the earliest passing checkpoint. Training stops at the first passing checkpoint; otherwise it continues through the 1M-step budget.",
            "",
            "| Step | Clean | OUT_OF_ROAD | Crash | Timeout | Mean route progress delta | Mean speed km/h | Stationary fraction |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for e in evals:
            s = e["summary"]
            report.append(f"| {e['step']:,} | {s['clean_success']}/5 | {s['out_of_road']} | {s['crash']} | {s['timeout']} | {s['mean_route_progress_delta']:.4f} | {s['mean_speed_km_h']:.2f} | {s['mean_stationary_fraction']:.3f} |")
        report += ["", f"Protocol SHA-256: `{summary['protocol_sha256']}`", "", "A PASS establishes only this fixed native MetaDrive task and evaluation schedule; it does not promote the policy to Platform V1 or Stage 5."]
        (run_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
