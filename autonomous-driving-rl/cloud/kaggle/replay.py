"""B1 batch replay admission for SB3 learners with one or more environments."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from cloud.kaggle.common import SOURCE_SHA, fingerprint, read, sha, verify, write
from cloud.kaggle.runtime import (
    RECIPE, ScheduledEnv, check_state, load_policy, make_vec, model_for, pool, recipe_identity,
    restore_rng, save_state,
)

MAX_ROWS = 5000
DTYPES = {
    "obs": (np.float32, (409,)), "next_obs": (np.float32, (409,)),
    "action": (np.float32, (2,)), "reward": (np.float32, ()),
    "terminated": (np.bool_, ()), "truncated": (np.bool_, ()),
    "episode": (np.int32, ()), "step": (np.int32, ()),
}
GUIDED_DTYPES = {**DTYPES, "progress": (np.float32, ()), "speed_km_h": (np.float32, ())}

def publish(model_path, expected, identity, version, out):
    verify(model_path, expected)
    out.mkdir(parents=True)
    actor_path = out / "policy.zip"
    actor_path.write_bytes(model_path.read_bytes())
    snapshot = {"schema": 1, "identity": identity, "policy_sha": expected, "policy_version": version,
                "lineage": RECIPE["lineage"]}
    write(out / "SNAPSHOT.json", snapshot)
    return sha(out / "SNAPSHOT.json")

def collect(args):
    torch.set_num_threads(1)
    verify(args.snapshot / "SNAPSHOT.json", args.snapshot_sha)
    snapshot = read(args.snapshot / "SNAPSHOT.json")
    verify(args.snapshot / "policy.zip", snapshot["policy_sha"])
    if snapshot["lineage"] != RECIPE["lineage"] or snapshot["schema"] != 1:
        raise ValueError("Unexpected snapshot lineage or schema")
    model = load_policy(args.snapshot / "policy.zip")
    torch.manual_seed(int(fingerprint((args.stream, args.sequence))[:8], 16))
    geometry = getattr(args, "geometry", None)
    collection_phase = getattr(args, "collection_phase", None)
    if collection_phase is not None and (not geometry or collection_phase not in RECIPE["phase_weights"]):
        raise ValueError("An explicit collection phase requires a fixed TRAIN geometry")
    phase = collection_phase or ("easy" if geometry else "smoke")
    rows = pool(phase)
    if geometry and (geometry not in {r["geometry_id"] for r in rows}
                     or (collection_phase is None and snapshot["identity"]["phase"] != "easy")):
        raise ValueError("Guided collection requires a geometry in the locked TRAIN pool")
    dtypes = GUIDED_DTYPES if geometry else DTYPES
    start = time.monotonic()
    data = {key: [] for key in dtypes}
    episodes = []
    fixed = (geometry, args.seed) if geometry else None
    env = ScheduledEnv(phase, args.stream, args.sequence * MAX_ROWS, fixed=fixed)
    outcome = None
    try:
        obs, info = env.reset()
        episodes.append(info)
        for _ in range(args.rows):
            action, _ = model.predict(obs, deterministic=bool(geometry))
            successor, reward, terminated, truncated, step_info = env.step(action)
            row = (obs, successor, action, reward, terminated, truncated,
                   len(episodes) - 1, step_info["step_index"])
            if geometry:
                executed = np.asarray([step_info["action_steering"], step_info["action_throttle_brake"]], dtype=np.float32)
                row = (obs, successor, executed, reward, terminated, truncated, 0, step_info["step_index"],
                       step_info["route_completion"], step_info["speed_km_h"])
            for key, value in zip(dtypes, row, strict=True):
                data[key].append(value)
            obs = successor
            if terminated or truncated:
                if geometry:
                    outcome = step_info["primary_reason"]
                    break
                obs, info = env.reset()
                episodes.append(info)
        if data["episode"][-1] != len(episodes) - 1:
            episodes.pop()
    finally:
        env.close()
    if geometry and not outcome:
        raise ValueError("Guided collection requires the full episode through a real terminal or timeout")
    args.out.mkdir(parents=True)
    payload = args.out / "rows.npz"
    np.savez(payload, **{key: np.asarray(values, dtype=dtypes[key][0]) for key, values in data.items()})
    meta = {"schema": 1, "count": len(data["reward"]), "stream": args.stream, "sequence": args.sequence,
            "policy_version": snapshot["policy_version"], "snapshot_sha": args.snapshot_sha,
            "identity": snapshot["identity"], "geometry_pool": fingerprint(rows),
            "payload_sha": sha(payload), "episodes": [{k: info[k] for k in (
                "episode", "geometry_id", "geometry_sha256", "environment_seed", "horizon_steps",
                "traffic_density", "training_reward_mode_fingerprint")} for info in episodes],
            "seconds": time.monotonic() - start}
    if geometry:
        meta["guided"] = {"geometry": geometry, "seed": args.seed, "behavior": "frozen_deterministic", "outcome": outcome}
    if collection_phase is not None:
        meta["collection_phase"] = collection_phase
    write(args.out / "CHUNK.json", meta)
    print(sha(args.out / "CHUNK.json"))

def check_snapshot(path, digest, identity):
    verify(path / "SNAPSHOT.json", digest)
    snapshot = read(path / "SNAPSHOT.json")
    if (snapshot["schema"] != 1 or snapshot["lineage"] != RECIPE["lineage"]
            or snapshot["identity"] != identity):
        raise ValueError("Replay snapshot lineage or learner identity mismatch")
    verify(path / "policy.zip", snapshot["policy_sha"])
    return snapshot["policy_version"]


def validate(chunk, expected, identity, ledger, policy_version, snapshot=None, *, guided=False, collection_phase=None):
    marker = chunk / "CHUNK.json"
    if marker.stat().st_size > 4_000_000:
        raise ValueError("Chunk manifest is too large")
    verify(marker, expected)
    meta = read(marker)
    n = meta["count"]
    if meta["schema"] != 1 or not 0 < n <= MAX_ROWS or meta["identity"] != identity:
        raise ValueError("Incompatible replay chunk")
    if snapshot is not None and (meta["snapshot_sha"], meta["policy_version"]) != snapshot:
        raise ValueError("Chunk does not match the authorized actor snapshot")
    key = f"{meta['stream']}:{meta['sequence']}"
    if key in ledger:
        if ledger[key] != expected:
            raise ValueError("Chunk identity reused with different content")
        return None, key
    if not 0 <= policy_version - meta["policy_version"] <= 1:
        raise ValueError("Future or stale actor snapshot")
    if not 100 <= meta["stream"] <= 999 or not 0 <= meta["sequence"] <= 1000:
        raise ValueError("Collector namespace outside preflight range")
    guidance = meta.get("guided")
    if bool(guidance) != guided:
        raise ValueError("Replay collection mode mismatch")
    if meta.get("collection_phase") != collection_phase or (
            collection_phase is not None and (not guided or collection_phase not in RECIPE["phase_weights"])):
        raise ValueError("Replay collection phase mismatch")
    rows = pool(collection_phase or ("easy" if guided else "smoke"))
    dtypes = GUIDED_DTYPES if guided else DTYPES
    if meta["geometry_pool"] != fingerprint(rows):
        raise ValueError("Replay geometry pool mismatch")
    lookup = {r["geometry_id"]: r["geometry_sha256"] for r in rows}
    payload = chunk / "rows.npz"
    if payload.stat().st_size > 20_000_000:
        raise ValueError("Replay payload exceeds preflight size limit")
    verify(payload, meta["payload_sha"])
    with zipfile.ZipFile(payload) as archive:
        entries = archive.infolist()
        if len(entries) != len(dtypes) or {x.filename for x in entries} != {f"{k}.npy" for k in dtypes}:
            raise ValueError("Unexpected payload members")
        if sum(x.file_size for x in entries) > 20_000_000:
            raise ValueError("Expanded replay exceeds preflight size limit")
        # Inspect headers before NumPy allocates arrays from the payload.
        for name, (dtype, tail) in dtypes.items():
            entry = archive.getinfo(f"{name}.npy")
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                if version != (1, 0):
                    raise ValueError("Unsupported NumPy header")
                shape, _, actual_dtype = np.lib.format.read_array_header_1_0(stream)
                if shape != (n, *tail) or actual_dtype != dtype:
                    raise ValueError(f"Malformed replay header for {name}")
                if entry.file_size != stream.tell() + int(np.prod(shape)) * actual_dtype.itemsize:
                    raise ValueError(f"Unexpected byte size for {name}")
    with np.load(payload, allow_pickle=False) as arrays:
        data = {key: arrays[key] for key in dtypes}
    for name, (dtype, tail) in dtypes.items():
        value = data[name]
        if value.shape != (n, *tail) or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f"Malformed replay field {name}")
    if np.abs(data["action"]).max() > 1:
        raise ValueError("Replay action outside Box2")
    episode_index, decision = 0, 1
    episodes = meta["episodes"]
    if guided:
        done = data["terminated"] | data["truncated"]
        if (len(episodes) != 1 or not done[-1] or done[:-1].any()
                or guidance["behavior"] != "frozen_deterministic" or guidance["geometry"] not in lookup
                or type(guidance["seed"]) is not int or not 0 <= guidance["seed"] < 2**31
                or not isinstance(guidance["outcome"], str) or not guidance["outcome"]
                or (guidance["outcome"] == "TIMEOUT") != bool(data["truncated"][-1] and not data["terminated"][-1])
                or (guidance["outcome"] != "TIMEOUT" and not data["terminated"][-1])):
            raise ValueError("Invalid full-episode guided replay")
    from cloud.kaggle.runtime import schedule
    from src.training.stage5.train_sampler import TrainCaseSampler
    sampler = TrainCaseSampler(allowed_geometry_ids=list(lookup))
    for i in range(n):
        if episode_index >= len(episodes):
            raise ValueError("Missing replay episode metadata")
        episode = episodes[episode_index]
        cursor = meta["sequence"] * MAX_ROWS + episode_index
        geom, seed = ((guidance["geometry"], guidance["seed"]) if guided
                      else schedule(rows, "smoke", meta["stream"], cursor))
        case = sampler.get_case_for_geometry_id(geom, environment_seed=seed)
        if (episode["episode"], episode["geometry_id"], episode["geometry_sha256"],
            episode["environment_seed"], episode["horizon_steps"], episode["traffic_density"]) != (
                cursor, geom, case.geometry_sha256, seed, case.horizon_steps, case.traffic_density):
            raise ValueError("Replay episode identity differs from locked TRAIN schedule")
        if data["episode"][i] != episode_index or data["step"][i] != decision or decision > case.horizon_steps:
            raise ValueError("Overlapping or discontinuous replay decision IDs")
        done = data["terminated"][i] or data["truncated"][i]
        if done and i < n - 1:
            episode_index += 1
            decision = 1
        elif i < n - 1:
            if not np.array_equal(data["next_obs"][i], data["obs"][i + 1]):
                raise ValueError("Replay continuation observation mismatch")
            decision += 1
    if len(episodes) != episode_index + 1:
        raise ValueError("Unexpected replay episode metadata")
    return data, key

def admit(buffer, data):
    workers = buffer.n_envs
    count = len(data["reward"])
    if buffer.optimize_memory_usage or not buffer.handle_timeout_termination:
        raise ValueError("B1 requires a non-aliased timeout-aware replay buffer")
    if count == 0 or count % workers:
        raise ValueError("B1 chunk must contain complete vector replay rows")
    for i in range(0, count, workers):
        rows = slice(i, i + workers)
        term, trunc = data["terminated"][rows], data["truncated"][rows]
        buffer.add(data["obs"][rows], data["next_obs"][rows], data["action"][rows],
                   data["reward"][rows], term | trunc,
                   [{"TimeLimit.truncated": bool(value)} for value in trunc & ~term])

def import_and_update(args):
    torch.set_num_threads(1)
    if bool(args.parent) != bool(args.parent_sha):
        raise ValueError("Parent path and digest are paired")
    parent = check_state(args.parent, args.parent_sha) if args.parent else None
    if args.capacity <= 0 or args.capacity % args.workers:
        raise ValueError("Replay capacity must align with learner workers")
    identity = recipe_identity("smoke", args.workers, args.capacity, args.learning_starts, True)
    if parent and parent["identity"] != identity:
        raise ValueError("Replay parent has an incompatible recipe")
    ledger = dict(parent.get("ledger", {})) if parent else {}
    policy_version = args.policy_version if args.policy_version is not None else 0
    authorized = None
    if args.snapshot:
        version = check_snapshot(args.snapshot, args.snapshot_sha, identity)
        authorized = (args.snapshot_sha, version)
        policy_version = parent["updates"] if parent else 0
        if args.policy_version is not None and args.policy_version != policy_version:
            raise ValueError("Policy version differs from the verified learner")
    data, key = validate(args.chunk, args.chunk_sha, identity, ledger, policy_version, authorized)
    if data is None:
        print("Duplicate chunk; existing committed replay state retained")
        return
    n = len(data["reward"])
    if n > args.capacity or n % args.workers:
        raise ValueError("Replay chunk must fit capacity and align with learner workers")
    cursors = parent["workers"] if parent else [{"next_cursor": 0, "interrupted": None} for _ in range(args.workers)]
    env = make_vec(args.workers, "smoke", [cursor["next_cursor"] for cursor in cursors])
    try:
        if parent:
            model = SAC.load(args.parent / "model.zip", env=env, device="cpu")
            model.load_replay_buffer(args.parent / "replay.pkl")
            if (model.num_timesteps, model._n_updates) != (parent["transitions"], parent["updates"]):
                raise ValueError("Learner checkpoint counters differ from the parent manifest")
            restore_rng(torch.load(args.parent / "rng.pt", map_location="cpu", weights_only=False), "cpu")
        else:
            model = model_for(env, "cpu", args.capacity, args.learning_starts, args.source)
        buffer = model.replay_buffer
        if buffer.n_envs != args.workers or buffer.buffer_size * args.workers != args.capacity:
            raise ValueError("Learner replay shape differs from the configured workers or capacity")
        if buffer.size() * args.workers + n < model.batch_size:
            raise ValueError("Replay has fewer than one minibatch")
        admit(model.replay_buffer, data)
        ledger[key] = args.chunk_sha
        model.set_logger(configure(format_strings=[]))
        networks = (model.actor, model.critic, model.critic_target)
        before = [[p.detach().clone() for p in net.parameters()] for net in networks]
        previous_admitted = parent.get("external_admitted", 0) if parent else 0
        start = time.monotonic()
        budget = min(n, args.max_updates, model.num_timesteps + previous_admitted + n - model._n_updates)
        if budget <= 0:
            raise ValueError("No remaining gradient budget")
        model.train(gradient_steps=budget, batch_size=model.batch_size)
        changes = [max(float((p.detach() - old).abs().max()) for p, old in zip(net.parameters(), weights))
                   for net, weights in zip(networks, before)]
        if not all(np.isfinite(changes)) or min(changes) <= 0:
            raise RuntimeError("Replay did not update actor, critics and targets")
        metrics = {"updates": budget, "admitted": n, "utd_external_window": budget / n,
                   "snapshot_sha": args.snapshot_sha, "learner_policy_version": policy_version,
                   "sample_rows": budget * model.batch_size, "seconds": time.monotonic() - start,
                   "max_parameter_delta": dict(zip(("actor", "critic", "target"), changes)),
                   "losses": {k: float(v) for k, v in model.logger.name_to_value.items() if k.startswith("train/")}}
        save_state(args.out, model, {"identity": identity, "parent_sha": args.parent_sha, "job": "external-import",
                   "result": "COMPLETE", "workers": cursors, "ledger": ledger,
                   "transitions": model.num_timesteps, "external_admitted": previous_admitted + n,
                   "updates": model._n_updates}, metrics)
    finally:
        env.close()

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--out", type=Path, required=True)
    snapshot.add_argument("--state", type=Path, required=True)
    snapshot.add_argument("--state-sha", required=True)
    initial = sub.add_parser("source-snapshot")
    initial.add_argument("--out", type=Path, required=True)
    initial.add_argument("--source", type=Path, required=True)
    initial.add_argument("--workers", type=int, choices=(1, 2, 4), default=1)
    collection = sub.add_parser("collect")
    collection.add_argument("--out", type=Path, required=True)
    collection.add_argument("--snapshot", type=Path, required=True)
    collection.add_argument("--snapshot-sha", required=True)
    collection.add_argument("--stream", type=int, default=100)
    collection.add_argument("--sequence", type=int, default=0)
    collection.add_argument("--rows", type=int, default=1024)
    collection.add_argument("--geometry")
    collection.add_argument("--seed", type=int, default=30811)
    learner = sub.add_parser("import")
    learner.add_argument("--out", type=Path, required=True)
    learner.add_argument("--source", type=Path, required=True)
    learner.add_argument("--chunk", type=Path, required=True)
    learner.add_argument("--chunk-sha", required=True)
    learner.add_argument("--parent", type=Path)
    learner.add_argument("--parent-sha")
    learner.add_argument("--capacity", type=int, default=8192)
    learner.add_argument("--workers", type=int, choices=(1, 2, 4), default=1)
    learner.add_argument("--learning-starts", type=int, default=1024)
    learner.add_argument("--policy-version", type=int)
    learner.add_argument("--snapshot", type=Path)
    learner.add_argument("--snapshot-sha")
    learner.add_argument("--max-updates", type=int, default=8)
    args = parser.parse_args()
    if args.command == "snapshot":
        state = check_state(args.state, args.state_sha)
        print(publish(args.state / "model.zip", state["files"]["model.zip"], state["identity"], state["updates"], args.out))
    elif args.command == "source-snapshot":
        caps = RECIPE["preflight"]
        identity = recipe_identity("smoke", args.workers, caps["capacity"], caps["learning_starts"], True)
        print(publish(args.source, SOURCE_SHA, identity, 0, args.out))
    elif args.command == "collect":
        if not 0 < args.rows <= MAX_ROWS or not 100 <= args.stream <= 999 or not 0 <= args.sequence <= 1000:
            parser.error("Collector rows or namespace outside preflight range")
        collect(args)
    else:
        if bool(args.snapshot) != bool(args.snapshot_sha):
            parser.error("--snapshot and --snapshot-sha must be supplied together")
        verify(args.source, SOURCE_SHA)
        import_and_update(args)

if __name__ == "__main__":
    main()

