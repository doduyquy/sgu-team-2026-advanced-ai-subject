"""Collect and validate versioned TRAIN replay from first-window policies."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

from cloud.kaggle import replay, runtime
from cloud.kaggle.common import fingerprint, read, sha, verify, write
from src.platform.task_context_producer import TaskContextLocalizationError

MAX_EXTERNAL_ROWS = 12_000
LOCAL_EXTERNAL_ROWS = 75_000


def assert_contract(identity, parent):
    for field in ("files", "recipe", "source_recipe", "sb3"):
        if identity[field] != parent[field]:
            raise ValueError(f"Replay policy has incompatible {field}")
    expected = runtime.recipe_identity(parent["phase"], parent["workers"], parent["capacity"],
                                       parent["learning_starts"], bool(parent["warm_source"]))
    for field in ("files", "recipe", "source_recipe", "sb3"):
        if expected[field] != parent[field]:
            raise ValueError(f"Collector or update runtime has incompatible {field}")


def collect(plan, out):
    parent = runtime.check_state(Path(plan["parent"]), plan["parent_sha"])
    if not 2 <= len(plan["sources"]) <= 4 or not 1 <= len(plan["cases"]) <= 12:
        raise ValueError("Use two to four first-window sources and at most 12 cases each")
    stream_base = plan.get("stream_base", 801)
    if not isinstance(stream_base, int) or not 100 <= stream_base <= 996:
        raise ValueError("Collector namespace is outside the reserved stream range")
    out.mkdir(parents=True)
    manifest = {"schema": 1, "parent_sha": plan["parent_sha"], "sources": [], "discarded": []}
    if stream_base != 801:
        manifest["stream_base"] = stream_base
    rows = runtime.pool("mixed")
    families = defaultdict(list)
    for row in rows:
        families[row["tier"], row["sequence"]].append(row["geometry_id"])
    plan_cases = []
    for i, case in enumerate(plan["cases"]):
        family = sorted(families[case["tier"], case["family"]])
        geometry = family[case["geometry_offset"]]
        seed = case["seed"]
        plan_cases.append((geometry, seed, case["tier"]))
    if len(set(plan_cases)) != len(plan_cases):
        raise ValueError("Collection plan repeats a case")

    for source_index, source in enumerate(plan["sources"]):
        state = Path(source["state"])
        marker = runtime.check_state(state, source["state_sha"])
        assert_contract(marker["identity"], parent["identity"])
        folder = out / f"source-{source_index:02d}"
        snap = folder / "snapshot"
        snapshot_sha = replay.publish(state / "model.zip", marker["files"]["model.zip"],
                                      marker["identity"], marker["updates"], snap)
        (folder / "SOURCE.json").write_bytes((state / "COMPLETE.json").read_bytes())
        source_record = {"label": source["label"], "source_state": source["state_sha"], "model_sha": marker["files"]["model.zip"],
                         "snapshot_sha": snapshot_sha, "policy_version": marker["updates"], "chunks": []}
        for i, (geometry, seed, tier) in enumerate(plan_cases):
            chunk = folder / f"chunk-{i:02d}"
            try:
                replay.collect(SimpleNamespace(snapshot=snap, snapshot_sha=snapshot_sha, rows=replay.MAX_ROWS,
                                               stream=stream_base + source_index, sequence=i, geometry=geometry,
                                               collection_phase="mixed", seed=seed, out=chunk))
            except TaskContextLocalizationError:
                manifest["discarded"].append({"source": source["label"], "geometry": geometry, "seed": seed,
                                              "reason": "LOCALIZATION"})
                continue
            source_record["chunks"].append({"sequence": i, "sha": sha(chunk / "CHUNK.json"), "tier": tier})
        manifest["sources"].append(source_record)
    write(out / "CONSOLIDATED.json", manifest)
    print(sha(out / "CONSOLIDATED.json"), flush=True)


def validate_dataset(dataset, digest, parent, parent_sha, *, seen=None, previous_ledger=None, row_limit=MAX_EXTERNAL_ROWS):
    verify(dataset / "CONSOLIDATED.json", digest)
    manifest = read(dataset / "CONSOLIDATED.json")
    if manifest["schema"] != 1 or manifest["parent_sha"] != parent_sha:
        raise ValueError("Dataset belongs to a different learner parent")
    sources = manifest["sources"]
    if not 2 <= len(sources) <= 4 or len({source["label"] for source in sources}) != len(sources):
        raise ValueError("Consolidation requires two or more distinct sources")
    if len({source["model_sha"] for source in sources}) != len(sources):
        raise ValueError("Consolidation cannot count a policy twice")
    assert_contract(parent["identity"], parent["identity"])
    env = runtime.ScheduledEnv("mixed", 0, 0)
    try:
        native_reward = env.env.reward_mode_identity["fingerprint"]
    finally:
        env.close()
    stream_base = manifest.get("stream_base", 801)
    if not isinstance(stream_base, int) or not 100 <= stream_base <= 996:
        raise ValueError("Dataset uses an invalid collector namespace")
    seen = set() if seen is None else seen
    ledger = dict(parent["ledger"] if previous_ledger is None else previous_ledger)
    admitted = {}
    audit = []
    geometry_tiers = {r["geometry_id"]: r["tier"] for r in runtime.pool("mixed")}
    for i, source in enumerate(sources):
        folder = dataset / f"source-{i:02d}"
        snapshot = folder / "snapshot"
        verify(snapshot / "SNAPSHOT.json", source["snapshot_sha"])
        snap = read(snapshot / "SNAPSHOT.json")
        verify(folder / "SOURCE.json", source["source_state"])
        original = read(folder / "SOURCE.json")
        if (snap["policy_sha"], snap["identity"], snap["policy_version"]) != (
                original["files"]["model.zip"], original["identity"], original["updates"]):
            raise ValueError("Snapshot differs from the verified source checkpoint")
        if snap["policy_sha"] != source["model_sha"]:
            raise ValueError("Source model digest disagrees with snapshot")
        assert_contract(snap["identity"], parent["identity"])
        version = replay.check_snapshot(snapshot, source["snapshot_sha"], snap["identity"])
        if version != source["policy_version"]:
            raise ValueError("Source optimizer version differs from snapshot")
        chunks = source["chunks"]
        if not 0 < len(chunks) <= 12:
            raise ValueError("A source has no complete eligible episodes or exceeds the pilot bound")
        pieces = defaultdict(list)
        for spec in chunks:
            chunk = folder / f"chunk-{spec['sequence']:02d}"
            data, key = replay.validate(chunk, spec["sha"], snap["identity"], ledger, version,
                                        (source["snapshot_sha"], version), guided=True, collection_phase="mixed")
            if data is None:
                continue
            meta = read(chunk / "CHUNK.json")
            episode, = meta["episodes"]
            tier = geometry_tiers[episode["geometry_id"]]
            if (episode["training_reward_mode_fingerprint"] != native_reward or tier != spec["tier"]
                    or key != f"{stream_base + i}:{spec['sequence']}"):
                raise ValueError("Wrong native reward, tier or collector namespace")
            episode_key = fingerprint((episode["geometry_sha256"], episode["environment_seed"])).encode()
            ids = []
            for j, step in enumerate(data["step"]):
                parts = [episode_key, step.tobytes()]
                parts.extend(data[field][j].tobytes() for field in
                             ("obs", "action", "reward", "next_obs", "terminated", "truncated"))
                identity = hashlib.sha256(b"".join(parts)).digest()
                if identity not in seen:
                    seen.add(identity)
                    ids.append(j)
            if len(seen) > row_limit:
                raise ValueError("Unique external rows exceed the consolidation pilot budget")
            if ids:
                pieces[tier].append({name: values[ids] for name, values in data.items()})
            ledger[key] = spec["sha"]
            audit.append({"source": source["label"], "tier": tier, "outcome": meta["guided"]["outcome"],
                          "collected": meta["count"], "admitted": len(ids), "duplicate": meta["count"] - len(ids),
                          "geometry": episode["geometry_id"], "environment_seed": episode["environment_seed"]})
        if pieces:
            admitted[source["label"]] = {
                tier: {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
                for tier, parts in pieces.items()
            }
    if len(admitted) < 2:
        raise ValueError("Consolidation requires eligible rows from multiple distinct source policies")
    return admitted, audit, ledger


class MixedReplay:
    """SAC minibatch view over the unchanged parent ring and bounded external pools."""

    def __init__(self, parent, sources):
        self.parent = parent
        self.sources = sources
        self.draws = Counter()

    def sample(self, batch_size, env=None):
        import torch
        from stable_baselines3.common.type_aliases import ReplayBufferSamples

        external_budget = batch_size // 4
        if external_budget < sum(len(tiers) for tiers in self.sources.values()):
            raise ValueError("Minibatch is too small to sample all external strata")
        parent_rows = batch_size - external_budget
        batches = [self.parent.sample(parent_rows, env)]
        self.draws["parent"] += parent_rows
        for i, (source, tiers) in enumerate(sorted(self.sources.items())):
            quota, remainder = divmod(external_budget, len(self.sources))
            count = quota + int(i < remainder)
            tier_quota, tier_remainder = divmod(count, len(tiers))
            if not tier_quota:
                raise ValueError("External source budget is too small to sample all its tiers")
            for j, (tier, buffer) in enumerate(sorted(tiers.items())):
                n = tier_quota + int(j < tier_remainder)
                batches.append(buffer.sample(n, env))
                self.draws[f"{source}/{tier}"] += n
        if any(batch.discounts is not None for batch in batches):
            raise ValueError("Consolidation currently requires one-step SAC replay")
        return ReplayBufferSamples(*(torch.cat(field, dim=0) for field in zip(*(b[:5] for b in batches), strict=True)))


def build_mixed_replay(model, data):
    from stable_baselines3.common.buffers import ReplayBuffer

    external = {}
    for source, tiers in data.items():
        external[source] = {}
        for tier, rows in tiers.items():
            buffer = ReplayBuffer(len(rows["reward"]), model.observation_space, model.action_space,
                                  device=model.device, n_envs=1, handle_timeout_termination=True)
            replay.admit(buffer, rows)
            external[source][tier] = buffer
    return MixedReplay(model.replay_buffer, external)


def training_replay(datasets, parent, anchor, *, row_limit):
    if len({digest for _, digest in datasets}) != len(datasets):
        raise ValueError("A replay dataset cannot be included twice")
    if len(datasets) == 1:
        return validate_dataset(*datasets[0], parent, anchor, row_limit=row_limit)
    seen, ledger = set(), dict(parent["ledger"])
    grouped = defaultdict(lambda: defaultdict(list))
    audit = []
    model_by_source = {}
    for dataset, digest in datasets:
        parts, records, ledger = validate_dataset(
            dataset, digest, parent, anchor, seen=seen, previous_ledger=ledger, row_limit=row_limit)
        manifest = read(dataset / "CONSOLIDATED.json")
        for source in manifest["sources"]:
            label, model = source["label"], source["model_sha"]
            if model_by_source.setdefault(label, model) != model:
                raise ValueError("One source label identifies different models across datasets")
        audit.extend(records)
        for source, tiers in parts.items():
            for tier, rows in tiers.items():
                grouped[source][tier].append(rows)
    if len(set(model_by_source.values())) != len(model_by_source):
        raise ValueError("A model cannot receive separate weights under different source labels")
    return {
        source: {tier: {field: np.concatenate([part[field] for part in split]) for field in split[0]}
                 for tier, split in tiers.items()}
        for source, tiers in grouped.items()
    }, audit, ledger


def cloud_update(args):
    import torch
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.logger import configure
    from cloud.kaggle.guided import GuidedSAC, prefix_sha

    working = Path("/kaggle/working")
    local = getattr(args, "local", False)
    if local:
        from cloud.kaggle.common import WORK
        if not args.out.resolve().is_relative_to(WORK.resolve()):
            raise ValueError("Local consolidation output must stay in the project runs workspace")
    elif sys.platform != "linux" or not working.is_dir() or not args.out.resolve().is_relative_to(working):
        raise RuntimeError("Updates run only in the Kaggle working directory unless local mode is explicitly enabled")
    if not 1 <= args.updates <= (10000 if local else 512):
        raise ValueError("Update budget is outside the consolidation limit")
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(getattr(args, "torch_threads", 1))
    parent = runtime.check_state(args.parent, args.parent_sha)
    identity = parent["identity"]
    if not identity["bounded_guided_replay"] or not identity["localization_guard"]:
        raise ValueError("Consolidation requires the verified bounded S01 parent recipe")
    extras = getattr(args, "extra_dataset", ())
    if extras and not local:
        raise ValueError("Multiple replay batches require an explicitly local update")
    datasets = [(args.dataset, args.dataset_sha), *[(Path(path), digest) for path, digest in extras]]
    anchor = args.parent_sha
    if local and parent.get("update_mode") == "offline_consolidated_v1":
        anchor = parent.get("consolidation_anchor_sha256", parent["parent_sha"])
    data, audit, ledger = training_replay(
        datasets, parent, anchor, row_limit=LOCAL_EXTERNAL_ROWS if local else MAX_EXTERNAL_ROWS)
    model = GuidedSAC.load(args.parent / "model.zip", device=args.device, custom_objects={
        "replay_buffer_class": ReplayBuffer, "replay_buffer_kwargs": {},
    })
    model.load_replay_buffer(args.parent / "replay.pkl")
    model.set_logger(configure(format_strings=[]))
    model._vec_normalize_env = None
    ring = model.replay_buffer
    original = (ring.pos, ring.full, ring.size())
    parent_metrics = read(args.parent / "metrics.json")
    if (model.num_timesteps, model._n_updates, ring.n_envs, ring.buffer_size * ring.n_envs) != (
            parent["transitions"], parent["updates"], identity["workers"], identity["capacity"]):
        raise ValueError("Parent model or replay counters differ from the verified checkpoint")
    protected = parent["external_admitted"] // ring.n_envs
    if prefix_sha(ring, protected) != parent_metrics["protected_replay_sha"]:
        raise ValueError("S01 protected guided replay digest changed")
    if ring.size() * ring.n_envs != parent_metrics["replay_size"]:
        raise ValueError("S01 live row count differs from checkpoint")
    mix = build_mixed_replay(model, data)

    def adam_steps(optimizer):
        return {int(state["step"].item()) for state in optimizer.state.values() if "step" in state}

    optimizers = {"actor": model.actor.optimizer, "critic": model.critic.optimizer,
                  "entropy": model.ent_coef_optimizer}
    before = {name: adam_steps(optimizer) for name, optimizer in optimizers.items()}
    if any(len(steps) != 1 for steps in before.values()):
        raise ValueError("Parent optimizer lacks unambiguous continuation counters")
    runtime.restore_rng(torch.load(args.parent / "rng.pt", map_location="cpu", weights_only=False), args.device)
    model.replay_buffer = mix
    started = time.monotonic()
    try:
        model.train(gradient_steps=args.updates, batch_size=model.batch_size)
    finally:
        model.replay_buffer = ring
    updates = model._n_updates - parent["updates"]
    after = {name: adam_steps(optimizer) for name, optimizer in optimizers.items()}
    if updates != args.updates or sum(mix.draws.values()) != updates * model.batch_size:
        raise RuntimeError("Minibatch and update counts disagree")
    if any(after[name] != {next(iter(steps)) + updates} for name, steps in before.items()):
        raise RuntimeError("Adam counters did not continue for every optimizer")
    if (ring.pos, ring.full, ring.size()) != original or prefix_sha(ring, protected) != parent_metrics["protected_replay_sha"]:
        raise RuntimeError("Offline update changed the source replay")
    evidence = {"dataset_sha256": args.dataset_sha, "external_rows": sum(row["admitted"] for row in audit),
                "sampling_draws": dict(mix.draws), "gradient_updates": updates,
                "steps_before": {name: next(iter(steps)) for name, steps in before.items()},
                "steps_after": {name: next(iter(steps)) for name, steps in after.items()},
                "sources": audit, "ledger": ledger, "seconds": time.monotonic() - started}
    if local:
        evidence["datasets"] = [{"path": str(path), "sha256": digest} for path, digest in datasets]
        evidence["device"] = args.device
    # Preserve the original replay and ledger for normal GuidedSAC full-state resume.
    # The pinned external dataset remains separately reusable through this update mode.
    manifest = {**parent, "job": "first-window-consolidation", "parent_sha": args.parent_sha,
                "updates": model._n_updates, "update_mode": "offline_consolidated_v1",
                "consolidation_dataset_sha256": args.dataset_sha}
    retained_fields = ("replay_size", "protected_replay_sha", "guided_prefill", "online_behavior",
                       "guided_live_rows", "online_live_rows", "online_evicted_rows", "localization_invalid_rows")
    metrics = {key: parent_metrics[key] for key in retained_fields if key in parent_metrics}
    if local:
        manifest["consolidation_anchor_sha256"] = anchor
        manifest["consolidation_datasets"] = [digest for _, digest in datasets]
    metrics.update(transitions=0, updates=updates, consolidation=evidence)
    runtime.save_state(args.out, model, manifest, metrics)
    print(sha(args.out / "COMPLETE.json"), flush=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    collection = commands.add_parser("collect")
    collection.add_argument("--plan", type=Path, required=True)
    collection.add_argument("--out", type=Path, required=True)
    validation = commands.add_parser("verify")
    update = commands.add_parser("update")
    for command in (validation, update):
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--dataset-sha", required=True)
        command.add_argument("--parent", type=Path, required=True)
        command.add_argument("--parent-sha", required=True)
    update.add_argument("--out", type=Path, required=True)
    update.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    update.add_argument("--updates", type=int, default=128)
    update.add_argument("--local", action="store_true")
    update.add_argument("--torch-threads", type=int, choices=(1, 2, 3, 4), default=1)
    update.add_argument("--extra-dataset", nargs=2, action="append", metavar=("PATH", "MANIFEST_SHA"), default=[])
    args = parser.parse_args()
    if args.command == "collect":
        collect(read(args.plan), args.out)
    elif args.command == "update":
        cloud_update(args)
    else:
        _, audit, _ = validate_dataset(args.dataset, args.dataset_sha,
                                       runtime.check_state(args.parent, args.parent_sha), args.parent_sha)
        for row in audit:
            print(row, flush=True)


if __name__ == "__main__":
    main()
