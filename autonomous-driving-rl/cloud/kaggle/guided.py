"""Champion-guided Easy replay prefill and bounded online exploration."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from cloud.kaggle import learning_hooks, runtime
from cloud.kaggle.common import read, sha
from cloud.kaggle.replay import DTYPES, admit, check_snapshot, validate

NOISE_STD = 0.04
BEHAVIOR = "champion_guided_v1"
BEHAVIOR_CONFIG = {"kind": "learner deterministic plus clipped Gaussian Box2 noise", "std": NOISE_STD}


def progress_band(progress):
    return "early" if progress < 0.25 else "middle" if progress < 0.75 else "late"


def prefill(root, identity):
    manifest = read(root / "GUIDED.json")
    snapshot = manifest["snapshot_sha"]
    version = check_snapshot(root / "snapshot", snapshot, identity)
    if version != 0 or read(root / "snapshot/SNAPSHOT.json")["policy_sha"] != identity["warm_source"]:
        raise ValueError("Guided replay must come from the approved initial champion")
    ledger, arrays, counts, seen = {}, [], Counter(), set()
    families = {row["geometry_id"]: row["sequence"] for row in runtime.pool("easy")}
    outcomes = Counter()
    for index, digest in enumerate(manifest["chunks"]):
        path = root / f"chunk-{index:02}"
        data, key = validate(path, digest, identity, ledger, version, (snapshot, version), guided=True)
        meta = read(path / "CHUNK.json")["guided"]
        case = (meta["geometry"], meta["seed"])
        if data is None or case in seen:
            raise ValueError("Duplicate guided episode")
        seen.add(case)
        ledger[key] = digest
        arrays.append(data)
        counts.update(progress_band(p) for p in data["progress"])
        outcomes[meta["outcome"]] += 1
    if not arrays:
        raise ValueError("No approved guided episodes")
    data = {key: np.concatenate([row[key] for row in arrays]) for key in DTYPES}
    count = len(data["reward"])
    dropped = count % identity["workers"]
    if dropped:
        # Remove only nonterminal transitions so the complete episodes retain terminal support.
        indices = np.flatnonzero(~(data["terminated"] | data["truncated"]))[:dropped]
        if len(indices) != dropped:
            raise ValueError("Cannot align guided buffer without dropping terminal support")
        progress = np.concatenate([row["progress"] for row in arrays])
        for index in indices:
            counts[progress_band(progress[index])] -= 1
        data = {key: np.delete(values, indices, axis=0) for key, values in data.items()}
    if count - dropped < 1024 or len({families[g] for g, _ in seen}) != 3 or not counts["late"] or not outcomes["SUCCESS"]:
        raise ValueError("Guided prefill needs all three families and real late-route and success support")
    return data, {"manifest_sha256": sha(root / "GUIDED.json"), "snapshot_sha256": snapshot,
                  "rows": count - dropped, "alignment_dropped": dropped, "progress_bands": dict(counts),
                  "cases": sorted(seen), "outcomes": dict(outcomes), "ledger": ledger,
                  "behavior": "frozen deterministic champion"}


class GuidedSAC(runtime.TimedSAC):
    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        action, _ = self.predict(self._last_obs, deterministic=True)
        scaled = self.policy.scale_action(action)
        scaled = np.clip(scaled + np.random.normal(0, NOISE_STD, scaled.shape), -1, 1).astype(np.float32)
        return self.policy.unscale_action(scaled), scaled


def grow_replay(model, capacity, live_rows):
    old = model.replay_buffer
    if (type(old) is not runtime.ReplayBuffer or old.full or old.optimize_memory_usage
            or not old.handle_timeout_termination or old.pos * old.n_envs != live_rows
            or capacity % old.n_envs or capacity < old.buffer_size * old.n_envs):
        raise ValueError("Guided resume requires intact replay slots and compatible capacity")
    if capacity == old.buffer_size * old.n_envs:
        return
    new = runtime.ReplayBuffer(capacity, old.observation_space, old.action_space, device=model.device,
                               n_envs=old.n_envs, handle_timeout_termination=True)
    for field in ("observations", "next_observations", "actions", "rewards", "dones", "timeouts"):
        getattr(new, field)[:old.pos] = getattr(old, field)[:old.pos]
    new.pos = old.pos
    model.buffer_size, model.replay_buffer = capacity, new


def train(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--guided-prefill", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--parent-sha")
    parser.add_argument("--capacity", type=int, default=runtime.RECIPE["preflight"]["capacity"])
    parser.add_argument("--transitions", required=True, type=int)
    config, learner_args = parser.parse_known_args(args)
    if bool(config.guided_prefill) == bool(config.resume) or bool(config.resume) != bool(config.parent_sha):
        parser.error("Specify guided prefill or a verified guided resume")
    if "--measure-optimizer" not in learner_args:
        parser.error("Guided training requires the metered GuidedSAC learner")
    learner_args.extend(["--capacity", str(config.capacity), "--transitions", str(config.transitions)])
    parent = None
    if config.resume:
        learner_args.extend(["--resume", str(config.resume), "--parent-sha", config.parent_sha])
        parent = runtime.check_state(config.resume, config.parent_sha)
        if (parent.get("learning_mode") != BEHAVIOR or parent["identity"].get("online_behavior") != BEHAVIOR
                or read(config.resume / "metrics.json")["online_behavior"] != BEHAVIOR_CONFIG
                or not parent["ledger"] or parent["external_admitted"] <= 0
                or parent["external_admitted"] + parent["transitions"] + config.transitions > config.capacity):
            raise ValueError("Resume must retain guided behavior and all prior replay throughout the budget")
    model_for, base_sac, telemetry, save = runtime.model_for, runtime.TimedSAC, runtime.Telemetry, runtime.save_state
    recipe_identity = runtime.recipe_identity
    online = Counter()

    def parent_identity(phase, workers, capacity, learning_starts, warm):
        identity = recipe_identity(phase, workers, parent["identity"]["capacity"], learning_starts, warm)
        return {**identity, "online_behavior": BEHAVIOR}

    class ResumedGuidedSAC(GuidedSAC):
        def load_replay_buffer(self, path, truncate_last_traj=True):
            super().load_replay_buffer(path, truncate_last_traj=truncate_last_traj)
            evidence = self.guided_prefill
            if (self.buffer_size != parent["identity"]["capacity"]
                    or self.replay_buffer.buffer_size * self.replay_buffer.n_envs != self.buffer_size
                    or self.gradient_steps != -1 or evidence["rows"] != parent["external_admitted"]
                    or evidence["ledger"] != parent["ledger"]):
                raise ValueError("Guided state differs from parent identity, behavior or provenance")
            grow_replay(self, config.capacity, evidence["rows"] + parent["transitions"])

    def seeded(env, device, capacity, learning_starts, source=None, *, warm_model=None, warm_sha=None, metered=False):
        if learning_starts != 0 or not warm_model or not metered or source:
            raise ValueError("Guided prefill requires a fresh metered champion warm-start with zero random steps")
        identity = runtime.recipe_identity("easy", env.num_envs, capacity, 0, True)
        identity["warm_source"] = warm_sha
        data, evidence = prefill(config.guided_prefill, identity)
        if evidence["rows"] + config.transitions > capacity:
            raise ValueError("Pilot capacity would overwrite verified guided replay")
        model = model_for(env, device, capacity, learning_starts, warm_model=warm_model, warm_sha=warm_sha, metered=True)
        admit(model.replay_buffer, data)
        model.guided_prefill = evidence
        return model

    class GuidedTelemetry(telemetry):
        def _on_step(self):
            online.update(progress_band(info["route_completion"]) for info in self.locals["infos"])
            return super()._on_step()

    def save_guided(out, model, manifest, metrics):
        evidence = model.guided_prefill
        if model.replay_buffer.size() * model.n_envs != evidence["rows"] + model.num_timesteps:
            raise ValueError("Guided replay was overwritten or replay accounting diverged")
        manifest.update(external_admitted=evidence["rows"], ledger=evidence["ledger"], learning_mode=BEHAVIOR)
        manifest["identity"] = {**manifest["identity"], "online_behavior": BEHAVIOR, "capacity": config.capacity}
        metrics.update(guided_prefill=evidence, online_progress_bands=dict(online),
                       online_behavior=BEHAVIOR_CONFIG,
                       guided_live_rows=evidence["rows"], online_live_rows=model.num_timesteps,
                       updates_per_total_real_transition=model._n_updates / (evidence["rows"] + model.num_timesteps))
        save(out, model, manifest, metrics)

    try:
        runtime.TimedSAC = ResumedGuidedSAC if parent else GuidedSAC
        runtime.model_for, runtime.Telemetry, runtime.save_state = seeded, GuidedTelemetry, save_guided
        if parent:
            runtime.recipe_identity = parent_identity
        learning_hooks.train(learner_args)
    finally:
        runtime.TimedSAC, runtime.model_for, runtime.Telemetry, runtime.save_state = base_sac, model_for, telemetry, save
        runtime.recipe_identity = recipe_identity


def main(argv=None):
    import sys

    args = sys.argv[1:] if argv is None else argv
    train(args[1:] if args[:1] == ["train"] else args)


if __name__ == "__main__":
    main()
