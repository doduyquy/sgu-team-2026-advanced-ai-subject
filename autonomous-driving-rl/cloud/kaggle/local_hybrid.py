"""Resume S01 with independent base, donor, and online replay for local SAC."""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.logger import configure
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from cloud.kaggle import replay, runtime
from cloud.kaggle.common import WORK, fingerprint, read, sha, verify
from cloud.kaggle.consolidated_replay import LOCAL_EXTERNAL_ROWS, assert_contract, training_replay
from cloud.kaggle.guided import GuidedSAC, prefix_sha
from cloud.kaggle.local_workers import PARENT, PARENT_SHA


RECIPE = {"actor_lr": 2.5e-5, "critic_lr": 1e-4, "batch_size": 256, "gamma": 0.99, "tau": 0.005,
          "utd": 0.5, "base_external_online": [128, 64, 64], "external_easy_medium": [22, 42],
          "online_capacity": 32768, "external_capacity_per_tier": 180000,
          "online_behavior": "deterministic current-policy snapshot per wave"}
V1_BATCHES = (
    ("first-window-product/recollected-train-01", "b2454b19afd436d6b1c1b1fabb7f84d2895d4734c0137e6d984aa83635b156cf"),
    ("first-window-local/batch-01", "4e4a15d3c5506d95febb42e3073bebcf40a692a32ca51d7c8f637f07dee5cf2d"),
    ("first-window-local/batch-02", "3449627164f76a4008fd5062ed18deda9a2f640cf3e73833f910ce84f1efbae8"),
    ("first-window-local/batch-03", "79444004206a24d3c31fd8cf7016354fc9b7088ed1282e306e60ee6da45db085"),
)


class HybridSAC(GuidedSAC):
    def _update_learning_rate(self, optimizers):
        super()._update_learning_rate(optimizers)
        for optimizer, rate in ((self.actor.optimizer, RECIPE["actor_lr"]),
                                (self.critic.optimizer, RECIPE["critic_lr"])):
            for group in optimizer.param_groups:
                group["lr"] = rate


def row_ids(data, episode, seen):
    episode_key = fingerprint((episode["geometry_sha256"], episode["environment_seed"])).encode()
    fields = ("obs", "action", "reward", "next_obs", "terminated", "truncated")
    admitted = []
    for j, step in enumerate(data["step"]):
        parts = [episode_key, step.tobytes(), *(data[field][j].tobytes() for field in fields)]
        identity = hashlib.sha256(b"".join(parts)).digest()
        if identity not in seen:
            seen.add(identity)
            admitted.append(j)
    return admitted


class ThreeWayReplay:
    def __init__(self, model, base, parent):
        def ring(capacity):
            return ReplayBuffer(capacity, model.observation_space, model.action_space, device="cpu",
                                n_envs=1, handle_timeout_termination=True)

        self.base = base
        self.online = ring(RECIPE["online_capacity"])
        self.external = {tier: ring(RECIPE["external_capacity_per_tier"]) for tier in ("Easy", "Medium")}
        self.identity = parent["identity"]
        env = runtime.ScheduledEnv("mixed", 0, 0)
        try:
            self.reward = env.env.reward_mode_identity["fingerprint"]
        finally:
            env.close()
        self.tiers = {row["geometry_id"]: row["tier"] for row in runtime.pool("mixed")}
        self.draws = Counter()
        self.online_rows = 0
        self.datasets = []
        self.audit = []
        self.seen = set()
        old, self.audit, self.ledger = training_replay(
            [(WORK / path, digest) for path, digest in V1_BATCHES], parent, PARENT_SHA, row_limit=LOCAL_EXTERNAL_ROWS)
        for parts in old.values():
            for tier, values in parts.items():
                if tier in self.external:
                    replay.admit(self.external[tier], values)
        # V1 episode identities are disjoint from newly allocated seed ranges.
        self.old_cases = {(row["geometry"], row["environment_seed"]) for row in self.audit}

    def admit(self, folder, digest, *, online_limit=None):
        verify(folder / "CONSOLIDATED.json", digest)
        manifest = read(folder / "CONSOLIDATED.json")
        if manifest["schema"] != 2 or manifest["parent_sha"] != PARENT_SHA:
            raise ValueError("Hybrid replay has a different schema or source parent")
        if digest in [spec["sha"] for spec in self.datasets]:
            raise ValueError("Duplicate hybrid dataset")
        online = manifest["online"]
        if online != (online_limit is not None) or (online and online_limit <= 0):
            raise ValueError("Only online snapshots have an explicit UTD admission budget")
        remaining_budget = online_limit
        remaining_rows = sum(record.get("rows", 0) for record in manifest["records"])
        sources = manifest["sources"]
        for source in sources.values():
            snapshot = Path(source["snapshot"])
            verify(snapshot.parent / "SOURCE.json", source["source_state"])
            pin = read(snapshot.parent / "SOURCE.json")
            if (pin["identity"], pin["files"]["model.zip"], pin["updates"]) != (
                    source["identity"], source["policy_sha"], source["policy_version"]):
                raise ValueError("Snapshot provenance disagrees with checkpoint manifest")
            if online and pin.get("update_mode") != "hybrid_local_v2" and source["source_state"] != PARENT_SHA:
                raise ValueError("Expected online transitions from S01 or a V2 continuation")
            assert_contract(source["identity"], self.identity)
            version = replay.check_snapshot(snapshot, source["snapshot_sha"], source["identity"])
            if version != source["policy_version"] or read(snapshot / "SNAPSHOT.json")["policy_sha"] != source["policy_sha"]:
                raise ValueError("Unexpected actor identity for collection")
        for index, record in enumerate(manifest["records"]):
            if "discard" in record:
                continue
            source = sources[record["label"]]
            chunk = folder / record["chunk"]
            rows, key = replay.validate(chunk, record["sha"], source["identity"], self.ledger, source["policy_version"],
                                        (source["snapshot_sha"], source["policy_version"]),
                                        guided=True, collection_phase="mixed")
            if rows is None or key != f"{manifest['stream']}:{index}":
                raise ValueError("Hybrid dataset repeated a chunk or used an invalid namespace")
            meta = read(chunk / "CHUNK.json")
            episode, = meta["episodes"]
            geometry, seed = episode["geometry_id"], episode["environment_seed"]
            tier = self.tiers[geometry]
            if ((geometry, seed) != (record["geometry"], record["seed"]) or (geometry, seed) in self.old_cases
                    or tier != record["tier"] or tier not in self.external
                    or episode["training_reward_mode_fingerprint"] != self.reward):
                raise ValueError("Hybrid collection differs from its TRAIN plan or reward contract")
            ids = row_ids(rows, episode, self.seen)
            eligible = len(ids)
            if online:
                if record["rows"] != meta["count"]:
                    raise ValueError("Online decision count disagrees with the validated chunk")
                quota = min(eligible, (remaining_budget * record["rows"] + remaining_rows - 1) // remaining_rows)
                if quota < eligible:
                    ids = [ids[i] for i in np.linspace(0, eligible - 1, num=quota, dtype=int)]
                remaining_budget -= quota
                remaining_rows -= record["rows"]
            target = self.online if online else self.external[tier]
            if not online and target.size() + len(ids) > target.buffer_size:
                raise ValueError("External replay exceeded its declared memory bound")
            if ids:
                replay.admit(target, {name: values[ids] for name, values in rows.items()})
            self.ledger[key] = record["sha"]
            if online:
                self.online_rows += len(ids)
            self.audit.append({"source": record["label"], "tier": tier, "geometry": geometry,
                               "environment_seed": seed, "collected": meta["count"], "admitted": len(ids),
                               "duplicate": meta["count"] - eligible, "held_out": eligible - len(ids),
                               "outcome": meta["guided"]["outcome"],
                               "pool": "online" if online else "external"})
        self.datasets.append({"path": folder.name, "sha": digest, "online_limit": online_limit})

    def sample(self, batch_size, env=None):
        if batch_size != RECIPE["batch_size"] or any(b.size() < 64 for b in self.external.values()):
            raise ValueError("Hybrid batch or external replay is undersized")
        online_n = 64 if self.online.size() >= 256 else 0
        groups = (("base", self.base, 192 - online_n), ("external_easy", self.external["Easy"], 22),
                  ("external_medium", self.external["Medium"], 42), ("online", self.online, online_n))
        tensors = []
        for name, buffer, n in groups:
            if n:
                tensors.append(buffer.sample(n, env))
                self.draws[name] += n
        if any(batch.discounts is not None for batch in tensors):
            raise ValueError("Expected one-step hybrid replay")
        return ReplayBufferSamples(*(torch.cat(values, dim=0) for values in zip(*(b[:5] for b in tensors), strict=True)))

    def proof(self):
        state = runtime.rng_state()
        draws = self.draws.copy()
        try:
            sample = self.sample(RECIPE["batch_size"])
            return hashlib.sha256(b"".join(x.numpy().tobytes() for x in sample[:5])).hexdigest()
        finally:
            runtime.restore_rng(state, "cpu")
            self.draws = draws


def optimizer_steps(model):
    optimizers = (model.actor.optimizer, model.critic.optimizer, model.ent_coef_optimizer)
    steps = [{int(record["step"].item()) for record in optimizer.state.values() if "step" in record}
             for optimizer in optimizers]
    if any(len(counts) != 1 for counts in steps):
        raise ValueError("Optimizer continuation counters are ambiguous")
    return [next(iter(counts)) for counts in steps]


def restore(root, state=PARENT, state_sha=PARENT_SHA):
    parent = runtime.check_state(PARENT, PARENT_SHA)
    marker = parent if state == PARENT and state_sha == PARENT_SHA else runtime.check_state(state, state_sha)
    model = HybridSAC.load(state / "model.zip", device="cpu", custom_objects={
        "replay_buffer_class": ReplayBuffer, "replay_buffer_kwargs": {}, "buffer_size": 1,
        "optimize_memory_usage": False, "n_envs": 1,
    })
    model.load_replay_buffer(state / "replay.pkl")
    ring = model.replay_buffer
    frozen = read(PARENT / "metrics.json")
    if (model.gamma, model.tau, model.batch_size) != (RECIPE["gamma"], RECIPE["tau"], RECIPE["batch_size"]):
        raise ValueError("SAC configuration differs from the frozen recipe")
    if (marker["files"]["replay.pkl"] != parent["files"]["replay.pkl"]
            or ring.size() * ring.n_envs != frozen["replay_size"]
            or prefix_sha(ring, parent["external_admitted"] // ring.n_envs) != frozen["protected_replay_sha"]):
        raise ValueError("S01 replay is not intact")
    if ((model.num_timesteps, model._n_updates) != (marker["transitions"], marker["updates"])
            or optimizer_steps(model) != [marker["updates"]] * 3):
        raise ValueError("State did not restore all training counters")
    model.set_logger(configure(format_strings=[]))
    model._vec_normalize_env = None
    mix = ThreeWayReplay(model, ring, parent)
    config = marker.get("hybrid")
    if config:
        if config["recipe"] != RECIPE or config["anchor"] != PARENT_SHA:
            raise ValueError("Unexpected V2 resume recipe")
        for record in config["datasets"]:
            mix.admit(root / record["path"], record["sha"], online_limit=record["online_limit"])
        mix.draws.update(config["sampling"])
    runtime.restore_rng(torch.load(state / "rng.pt", map_location="cpu", weights_only=False), "cpu")
    if config and (mix.online_rows != config["online_rows"] or mix.proof() != config["sample_resume_sha"]):
        raise ValueError("Replay/RNG did not reconstruct the exact next minibatch")
    return model, mix, marker


def update(model, mix, parent, target):
    previous = model._n_updates - parent["updates"]
    if not 0 < target - previous <= mix.online_rows // 2 - previous:
        raise ValueError("SAC update would exceed the UTD budget")
    before = optimizer_steps(model)
    before_draws = mix.draws.copy()
    started = monotonic()
    model.num_timesteps = parent["transitions"] + mix.online_rows
    frozen = model.replay_buffer
    model.replay_buffer = mix
    try:
        model.train(gradient_steps=target - previous, batch_size=RECIPE["batch_size"])
    finally:
        model.replay_buffer = frozen
    delta = target - previous
    draws = mix.draws - before_draws
    if (optimizer_steps(model) != [step + delta for step in before]
            or model._n_updates != parent["updates"] + target
            or sum(draws.values()) != delta * RECIPE["batch_size"]):
        raise ValueError("SAC optimizer or replay counters diverged")
    actual = {name: [group["lr"] for group in optim.param_groups] for name, optim in (
        ("actor", model.actor.optimizer), ("critic", model.critic.optimizer), ("alpha", model.ent_coef_optimizer))}
    if set(actual["actor"]) != {RECIPE["actor_lr"]} or set(actual["critic"]) != {RECIPE["critic_lr"]}:
        raise ValueError("SB3 overwrote the effective actor or critic learning rates")
    logs = {key: float(value) for key, value in model.logger.name_to_value.items()
            if key in ("train/actor_loss", "train/critic_loss", "train/ent_coef", "train/ent_coef_loss")}
    with torch.no_grad():
        rng = runtime.rng_state()
        sample = mix.base.sample(256)
        runtime.restore_rng(rng, "cpu")
        q1, q2 = model.critic(sample.observations, sample.actions)
        q = torch.minimum(q1, q2)
        logs.update(q_mean=float(q.mean()), q_p95=float(q.quantile(0.95)), q_abs_max=float(q.abs().max()))
    if not all(np.isfinite(value) for value in logs.values()):
        raise ValueError("SAC training emitted non-finite losses or Q values")
    elapsed = monotonic() - started
    return {"updates": delta, "cumulative_updates": target, "optimizer_steps": optimizer_steps(model),
            "new_draws": dict(draws), "learning_rates": actual, "seconds": elapsed,
            "updates_per_second": delta / elapsed, "log": logs}


def checkpoint(model, mix, parent, out, learning, cursor):
    pins = {"recipe": RECIPE, "anchor": PARENT_SHA, "online_rows": mix.online_rows, "datasets": mix.datasets,
            "sampling": dict(mix.draws), "sample_resume_sha": mix.proof(), "cursor": cursor}
    metrics = {**read(PARENT / "metrics.json"), "updates": learning["updates"], "hybrid": learning,
               "consolidation_sources": mix.audit, "online_live_rows_v2": mix.online.size()}
    marker = {**parent, "job": "first-window-local-hybrid-v2", "update_mode": "hybrid_local_v2", "hybrid": pins,
              "parent_sha": PARENT_SHA, "updates": model._n_updates, "transitions": model.num_timesteps}
    runtime.save_state(out, model, marker, metrics)
    return sha(out / "COMPLETE.json")
