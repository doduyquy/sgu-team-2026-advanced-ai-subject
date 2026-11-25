"""Bounded Stage 5 SAC preflight, reset-boundary resume and frozen evaluation."""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from functools import partial
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time

for thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[thread_var] = "1"

import gymnasium as gym
import numpy as np
import psutil
import stable_baselines3
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from cloud.kaggle.common import ROOT, SOURCE_SHA, fingerprint, read, sha, verify, write
from src.agents.stage5_terminal_preview_v1 import Stage5TerminalPreviewAdapterV1
from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.native_reward_training import NativeMetaDriveRewardTrainingEnv
from src.training.stage5.pilot_observation_transform import GROUPWISE_PHYSICAL_V1, apply_observation_transform
from src.training.stage5.train_sampler import TrainCaseSampler

RECIPE = read(Path(__file__).with_name("recipe.json"))
SOURCE_RECIPE = read(ROOT / "configs/experiments/stage5_native_reward_integration_v1.json")
SPLIT_PATH = ROOT / "results/stage5/pilot_split_v1.json"
PARTS = ("model.zip", "replay.pkl", "rng.pt", "metrics.json")

def pool(phase):
    split = read(SPLIT_PATH)
    train, diag = split["pilot_train"], split["pilot_diagnostic"]
    if len(train) != 156 or len(diag) != 24 or len({row["geometry_sha256"] for row in train + diag}) != 180:
        raise ValueError("Invalid TRAIN development split")
    ids = [row["geometry_id"] for row in train + diag]
    sampler = TrainCaseSampler(project_root=ROOT, allowed_geometry_ids=ids)
    for row in train + diag:
        case = sampler.get_case_for_geometry_id(row["geometry_id"])
        if (case.geometry_sha256, case.tier, case.sequence) != (row["geometry_sha256"], row["tier"], row["sequence"]):
            raise ValueError("Development split differs from locked TRAIN geometry")
    if phase == "smoke":
        lookup = {row["geometry_id"]: row for row in train}
        return [lookup[key] for key in RECIPE["smoke_geometries"]]
    if phase == "eval":
        return diag
    return [row for row in train if row["tier"] in RECIPE["phase_weights"][phase]]

def seed_for(*parts):
    return int(fingerprint(parts)[:8], 16) & 0x7fffffff

def schedule(rows, phase, worker, episode):
    key = (RECIPE["lineage"], RECIPE["seed"])
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["tier"]][row["sequence"]].append(row["geometry_id"])
    rng = random.Random(seed_for(*key, "order", worker, episode // 3))
    tiers = sorted(groups)
    weights = RECIPE["phase_weights"].get(phase, {"Easy": 100})
    tier = rng.choices(tiers, [weights[t] for t in tiers])[0]
    families = sorted(groups[tier])
    rng.shuffle(families)
    family = families[episode % len(families)]
    choices = sorted(groups[tier][family])
    geometry = choices[(worker + episode) % len(choices)]
    if phase == "smoke":
        geometry = rows[(worker + episode) % len(rows)]["geometry_id"]
    return geometry, seed_for(*key, "sim", worker, episode)

class ScheduledEnv(gym.Wrapper):
    def __init__(self, phase, worker, cursor, fixed=None):
        self.rows = pool(phase)
        self.phase, self.worker, self.cursor, self.fixed = phase, worker, cursor, fixed
        sampler = TrainCaseSampler(project_root=ROOT, allowed_geometry_ids=[r["geometry_id"] for r in self.rows])
        base = Stage5TrainingEnv(sampler=sampler, project_root=ROOT, training_root_seed=RECIPE["seed"],
                                 simulator_render_mode="OFF", task_context_producer_id=REFERENCE_TASK_PRODUCER)
        super().__init__(NativeMetaDriveRewardTrainingEnv(base))
        self.sampler = sampler
        self.adapter = Stage5TerminalPreviewAdapterV1()
        self.reset_times = deque(maxlen=128)
        self.active = None
        self.triggers = 0
        self.speed = 0.0

    def transform(self, obs):
        result = self.adapter.apply(obs)
        self.triggers += int(result.activated)
        return apply_observation_transform(result.observation, GROUPWISE_PHYSICAL_V1["transform_id"])

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("Cases are controlled by the locked schedule")
        geometry, env_seed = self.fixed or schedule(self.rows, self.phase, self.worker, self.cursor)
        case = self.sampler.get_case_for_geometry_id(geometry, environment_seed=env_seed)
        self.active = {"worker": self.worker, "episode": self.cursor, "case": case.case_id,
                       "geometry_sha256": case.geometry_sha256, "environment_seed": env_seed}
        start = time.monotonic()
        self.triggers = 0
        self.speed = 0.0
        obs, info = self.env.reset(options={"case": case})
        self.cursor += 1
        self.reset_times.append(time.monotonic() - start)
        return self.transform(obs), {**info, **self.active}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        transformed = self.transform(obs)
        if not np.allclose(action, [info["action_steering"], info["action_throttle_brake"]], atol=1e-6, rtol=0):
            raise RuntimeError("Box2 action fidelity failed")
        self.speed += float(info.get("speed_km_h") or 0)
        return transformed, reward, terminated, truncated, {
            **info, **self.active, "preview_triggers": self.triggers,
            "episode_mean_speed_km_h": self.speed / int(info["step_index"]),
        }

    def boundary(self):
        return {"next_cursor": self.cursor, "interrupted": self.active,
                "reset_seconds": list(self.reset_times)}

def make_vec(workers, phase, cursors):
    factories = [partial(ScheduledEnv, phase, i, cursors[i]) for i in range(workers)]
    if workers == 1:
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories, start_method="spawn")

def recipe_identity(phase, workers, capacity, learning_starts, warm):
    paths = [Path(__file__), SPLIT_PATH, ROOT / "results/audits/evaluation_protocol/geometry_split_manifest.csv",
             ROOT / "src/agents/stage5_terminal_preview_v1.py",
             ROOT / "src/platform/task_context_producer.py",
             ROOT / "src/training/stage5/native_reward_training.py",
             ROOT / "src/training/stage5/env_adapter.py",
             ROOT / "src/training/stage5/pilot_observation_transform.py",
             ROOT / "src/training/stage5/observation_encoder.py"]
    return {"phase": phase, "workers": workers, "capacity": capacity, "learning_starts": learning_starts,
            "warm_source": SOURCE_SHA if warm else None, "recipe": RECIPE, "source_recipe": SOURCE_RECIPE,
            "files": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
            "sb3": stable_baselines3.__version__}

def model_for(env, device, capacity, learning_starts, source=None):
    config = SOURCE_RECIPE["sac"]
    model = SAC("MlpPolicy", env, device=device, seed=RECIPE["seed"], buffer_size=capacity,
                learning_starts=learning_starts, gradient_steps=-1, train_freq=1,
                batch_size=config["batch_size"], learning_rate=config["learning_rate"], gamma=config["gamma"],
                tau=config["tau"], ent_coef=config["ent_coef"], target_entropy=config["target_entropy"],
                policy_kwargs={"net_arch": config["actor_critic_architecture"]})
    if source:
        verify(source, SOURCE_SHA)
        parent = load_policy(source, device)
        for attr in ("actor", "critic", "critic_target"):
            getattr(model, attr).load_state_dict(getattr(parent, attr).state_dict())
        if parent.num_timesteps != 100000:
            raise ValueError("Warm source counter differs from registered 100k")
    return model

def load_policy(path, device="cpu"):
    # Replace serialized values before deserialization or model construction.
    # The 100k source contains a disk-backed replay factory and a Windows path.
    return SAC.load(path, device=device, custom_objects={
        "replay_buffer_class": ReplayBuffer, "replay_buffer_kwargs": {},
        "buffer_size": 1, "optimize_memory_usage": False, "n_envs": 1,
    })

class Telemetry(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episodes = deque(maxlen=512)
        self.geometries = defaultdict(int)
        self.steady_start = None
        self.rss_peak = 0
        self.process = psutil.Process()

    def _on_step(self):
        if self.steady_start is None and self.num_timesteps > self.model.learning_starts:
            self.steady_start = time.monotonic()
        if self.n_calls == 1 or self.n_calls % 32 == 0:
            processes = [self.process, *self.process.children(recursive=True)]
            rss = 0
            for process in processes:
                try:
                    rss += process.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            self.rss_peak = max(rss, self.rss_peak)
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            self.geometries[info["geometry_id"]] += 1
            if done:
                self.episodes.append({k: info.get(k) for k in (
                    "case_id", "geometry_sha256", "environment_seed", "tier", "sequence", "primary_reason",
                    "clean_success", "route_completion", "episode_mean_speed_km_h", "preview_triggers", "step_index")})
        return True

def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}

def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and device == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])

def check_state(path, expected):
    verify(path / "COMPLETE.json", expected)
    manifest = read(path / "COMPLETE.json")
    if manifest["schema"] != 1 or set(manifest["files"]) != set(PARTS):
        raise ValueError("Invalid checkpoint manifest file list")
    for name in PARTS:
        verify(path / name, manifest["files"][name])
    return manifest

def save_state(out, model, manifest, metrics):
    if out.exists():
        raise FileExistsError(out)
    stage = out.with_name(out.name + ".staging")
    stage.mkdir(parents=True)
    start = time.monotonic()
    state = rng_state()
    model.save(stage / "model.zip")
    model.save_replay_buffer(stage / "replay.pkl")
    torch.save(state, stage / "rng.pt")
    metrics["serialization_seconds"] = time.monotonic() - start
    write(stage / "metrics.json", metrics)
    write(stage / "COMPLETE.json", {**manifest, "schema": 1, "files": {name: sha(stage / name) for name in PARTS}})
    stage.rename(out)

def train(args):
    t0 = time.monotonic()
    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable")
    if args.transitions % args.workers or args.capacity % args.workers or args.transitions <= 0:
        raise ValueError("Positive transitions and capacity must align with workers")
    parent = check_state(args.resume, args.parent_sha) if args.resume else None
    identity = recipe_identity(args.phase, args.workers, args.capacity, args.learning_starts,
                               bool(args.source or (parent and parent["identity"]["warm_source"])))
    if parent and parent["identity"] != identity:
        raise ValueError("Resume recipe, source, capacity or worker mismatch")
    cursors = [item["next_cursor"] for item in parent["workers"]] if parent else [0] * args.workers
    env = make_vec(args.workers, args.phase, cursors)
    try:
        if parent:
            model = SAC.load(args.resume / "model.zip", env=env, device=args.device)
            model.load_replay_buffer(args.resume / "replay.pkl")
            if (model.num_timesteps, model._n_updates) != (parent["transitions"], parent["updates"]):
                raise ValueError("Resume model counters differ from manifest")
            if model.replay_buffer.n_envs != args.workers or model.replay_buffer.buffer_size * args.workers != args.capacity:
                raise ValueError("Resume replay shape differs from recipe")
            restore_rng(torch.load(args.resume / "rng.pt", map_location="cpu", weights_only=False), args.device)
        else:
            model = model_for(env, args.device, args.capacity, args.learning_starts, args.source)
        before, updates = model.num_timesteps, model._n_updates
        end = min(t0 + args.max_seconds, args.stop_at) if args.stop_at else t0 + args.max_seconds
        callback = Telemetry()
        start = time.monotonic()
        while model.num_timesteps - before < args.transitions and time.monotonic() < end:
            block = min(args.transitions - (model.num_timesteps - before), 32 * args.workers)
            model.learn(block, reset_num_timesteps=False, callback=callback, log_interval=None)
        boundaries = env.env_method("boundary") if model.num_timesteps > before else []
        elapsed = time.monotonic() - start
        local = model.num_timesteps - before
        delta = model._n_updates - updates
        if not local:
            raise RuntimeError("Deadline expired before collection")
        resets = [seconds for worker in boundaries for seconds in worker["reset_seconds"]]
        losses = {key: float(value) for key, value in model.logger.name_to_value.items()
                  if key in {"train/actor_loss", "train/critic_loss", "train/ent_coef"}}
        metrics = {"seconds": elapsed, "transitions": local, "updates": delta, "utd": delta / local,
                   "sample_rows_per_transition": delta * model.batch_size / local,
                   "decisions_per_second": local / elapsed, "updates_per_second": delta / elapsed,
                   "post_warmup_seconds": time.monotonic() - callback.steady_start if callback.steady_start else None,
                   "rss_peak_sampled_bytes": callback.rss_peak, "reset_p50_seconds": float(np.percentile(resets, 50)),
                   "reset_p95_seconds": float(np.percentile(resets, 95)), "replay_size": model.replay_buffer.size() * args.workers,
                   "device": str(model.device), "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
                   "affinity_cores": len(psutil.Process().cpu_affinity()), "system_ram": psutil.virtual_memory().total,
                   "free_disk": shutil.disk_usage(args.out.parent).free, "platform": platform.platform(),
                   "python": sys.version, "torch": torch.__version__, "sb3": stable_baselines3.__version__,
                   "gymnasium": gym.__version__, "numpy": np.__version__, "losses": losses,
                   "episodes": list(callback.episodes), "decisions_by_geometry": dict(callback.geometries)}
        manifest = {"identity": identity, "job": args.job, "parent_sha": args.parent_sha,
                    "transitions": model.num_timesteps, "updates": model._n_updates, "workers": boundaries,
                    "ledger": parent.get("ledger", {}) if parent else {},
                    "external_admitted": parent.get("external_admitted", 0) if parent else 0,
                    "result": "COMPLETE" if local == args.transitions and delta else "PARTIAL"}
    finally:
        env.close()
    save_state(args.out, model, manifest, metrics)
    print(f"{manifest['result']} transitions={local} updates={delta} state={sha(args.out / 'COMPLETE.json')}")

def evaluate(args):
    torch.set_num_threads(1)
    verify(args.model, args.model_sha)
    model = load_policy(args.model)
    results = []
    for row in pool("eval"):
        if row["tier"] != args.tier:
            continue
        for seed in RECIPE["evaluation_seeds"]:
            env = ScheduledEnv("eval", 0, 0, fixed=(row["geometry_id"], seed))
            try:
                obs, reset = env.reset()
                for step in range(1, reset["horizon_steps"] + 1):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                results.append({"geometry": row["geometry_id"], "family": row["sequence"], "seed": seed,
                                "clean": info.get("clean_success", False), "outcome": info.get("primary_reason"),
                                "censored": not (terminated or truncated),
                                "progress": info["route_completion"], "speed_km_h": info["episode_mean_speed_km_h"],
                                "success_seconds": step / 10 if info.get("clean_success") else None,
                                "adapter_triggers": info["preview_triggers"]})
            finally:
                env.close()
    if args.out.exists():
        raise FileExistsError(args.out)
    write(args.out, {"model_sha": args.model_sha, "tier": args.tier, "cases": results})

def parity(args):
    torch.set_num_threads(1)
    verify(args.source, SOURCE_SHA)
    print("Parity: loading verified source checkpoint", flush=True)
    model = load_policy(args.source)
    anchors = []
    for i, row in enumerate(pool("smoke")):
        print(f"Parity: resetting {row['geometry_id']}", flush=True)
        env = ScheduledEnv("smoke", i, 0, fixed=(row["geometry_id"], 10101))
        try:
            obs, info = env.reset()
            raw = env.unwrapped.encoder.encode(env.unwrapped.current_agent_input)
            action, _ = model.predict(obs, deterministic=True)
            baseline, _ = model.predict(apply_observation_transform(
                raw, GROUPWISE_PHYSICAL_V1["transform_id"]), deterministic=True)
            if env.triggers or not np.array_equal(action, baseline):
                raise RuntimeError("Warm source action parity failed before adapter activation")
            _, reward, term, trunc, stepped = env.step(action)
            cfg = env.unwrapped._env.config
            if (cfg["decision_repeat"], cfg["physics_world_step_size"]) != (5, 0.02):
                raise RuntimeError("Platform cadence drift")
            anchors.append({"geometry": row["geometry_id"], "geometry_sha256": row["geometry_sha256"],
                            "seed": info["environment_seed"], "horizon": info["horizon_steps"],
                            "reward_identity": info["training_reward_mode_fingerprint"],
                            "obs": obs.tolist(), "action": action.tolist(), "reward": reward,
                            "terminated": term, "truncated": trunc, "outcome": stepped.get("primary_reason")})
        finally:
            env.close()
    if args.reference:
        reference = read(args.reference)
        for actual, expected in zip(anchors, reference, strict=True):
            for key, value in actual.items():
                if key in ("obs", "action", "reward"):
                    matches = np.allclose(value, expected[key], rtol=1e-4, atol=1e-4)
                else:
                    matches = value == expected[key]
                if not matches:
                    raise RuntimeError(f"Cloud parity drift: {actual['geometry']} {key}")
    write(args.out, anchors)

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("train")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--job", default="local")
    run.add_argument("--workers", type=int, choices=(1, 2, 4), default=1)
    run.add_argument("--phase", choices=("smoke", *RECIPE["phase_weights"]), default="smoke")
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    for name in ("transitions", "capacity", "learning_starts", "max_seconds"):
        run.add_argument("--" + name.replace("_", "-"), type=int, default=RECIPE["preflight"][name])
    run.add_argument("--stop-at", type=float)
    group = run.add_mutually_exclusive_group()
    group.add_argument("--source", type=Path)
    group.add_argument("--resume", type=Path)
    run.add_argument("--parent-sha")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--model", type=Path, required=True)
    ev.add_argument("--model-sha", required=True)
    ev.add_argument("--tier", choices=("Easy", "Medium", "Hard", "Extreme"), default="Easy")
    ev.add_argument("--out", type=Path, required=True)
    anchor = sub.add_parser("parity")
    anchor.add_argument("--source", type=Path, required=True)
    anchor.add_argument("--out", type=Path, required=True)
    anchor.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        if bool(args.resume) != bool(args.parent_sha):
            parser.error("--resume and --parent-sha must be supplied together")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        parity(args)

if __name__ == "__main__":
    main()

