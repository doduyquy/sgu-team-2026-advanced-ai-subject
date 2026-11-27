"""Local Stage 5 risk experiment with a frozen V2 replay mixture."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import os
from pathlib import Path

import numpy as np

from cloud.kaggle.common import WORK, read, sha, verify, write
from cloud.kaggle.local_workers import Simulators, assess, stage_source
from cloud.kaggle.risk_guidance import PROFILE

EXPERIMENT_ROOT = WORK / "first-window-risk"
ROOT = EXPERIMENT_ROOT
V2_ROOT = WORK / "first-window-v2"
LOCK = WORK / "postwindow-product/cleanup-locked-v2/delivery-lock.json"
TIERS = ("Easy", "Medium", "Hard", "Extreme")
MILESTONES = (128, 512, 1000, 2000, 5000)
MIX = {"base": 64, "external_easy": 11, "external_medium": 21, "v2_online": 32,
       "risk": 64, "online": 64}
RETENTION_MIX = {"base": 96, "external_easy": 16, "external_medium": 32, "v2_online": 48,
                 "risk": 32, "online": 32}


def v2_parent():
    lock = read(LOCK)
    return Path(lock["full_state"]), lock["full_state_manifest_sha256"]


def collect_wave(sim, state, digest, pool_name, wave):
    from cloud.kaggle import replay, runtime
    from cloud.kaggle.consolidated_replay import assert_contract

    risk = pool_name == "risk"
    folder = ROOT / f"{pool_name}-{wave:02d}"
    if (folder / "WAVE.json").is_file():
        saved = read(folder / "WAVE.json")
        if (saved["source_state"], saved["kind"], saved["wave"]) != (digest, pool_name, wave):
            raise ValueError("Existing wave belongs to a different parent")
        return folder
    folder.mkdir(parents=True)
    donor = stage_source(state / "model.zip", state / "COMPLETE.json", digest, folder / "donor")
    anchor, _ = v2_parent()
    assert_contract(donor["identity"], read(anchor / "COMPLETE.json")["identity"])
    rows = runtime.pool("mixed")
    tiers = (("Medium",) * 4 + ("Hard",) * 4 + ("Extreme",) * 4) if risk else (
        "Easy", "Medium", "Hard", "Medium", "Easy", "Medium", "Extreme", "Hard")
    stream = (740 if risk else 780) + wave
    jobs = []
    planned = []
    for j, tier in enumerate(tiers):
        geometry_pool = [row for row in rows if row["tier"] == tier]
        row = geometry_pool[(wave * 13 + j * 11) % len(geometry_pool)]
        geometry = row["geometry_id"]
        seed = 57700 + (0 if risk else 1000) + wave * 20 + j
        jobs.append({"kind": "collect", "snapshot": donor["snapshot"], "snapshot_sha": donor["snapshot_sha"],
                     "policy": donor["snapshot"] / "policy.zip", "policy_sha": donor["policy_sha"],
                     "geometry": geometry, "seed": seed, "stream": stream, "sequence": j,
                     "risk_guidance": risk, "out": folder / f"chunk-{j:03d}"})
        planned.append({"tier": tier, "geometry": geometry, "seed": seed, "chunk": jobs[-1]["out"].name})
    outputs, performance = sim.run(jobs)
    records = [{**case, **out} for case, out in zip(planned, outputs, strict=True)]
    write(folder / "WAVE.json", {"schema": 1, "kind": pool_name, "wave": wave, "stream": stream,
          "source_state": digest, "snapshot": {**donor, "snapshot": str(donor["snapshot"])},
          "records": records, "performance": performance})
    print("COLLECT", folder.name, performance["decisions"], round(performance["decisions_per_second"], 1), flush=True)
    return folder


class RiskReplay:
    def __init__(self, model, frozen, mixture=None, *, experiment_root=None, donor_guidance=True):
        from stable_baselines3.common.buffers import ReplayBuffer

        self.frozen = frozen
        self.experiment_root = EXPERIMENT_ROOT if experiment_root is None else experiment_root
        self.donor_guidance = donor_guidance
        self.mixture = dict(mixture or MIX)
        self.base = frozen.base
        self.online_rows = 0
        self.risk_rows = 0
        self.draws = Counter()
        self.ledger = dict(frozen.ledger)
        self.seen = set() if donor_guidance else set(frozen.seen)
        self.waves = []
        self.audit = []
        donor_capacity = 48000 if donor_guidance else 80000
        self.rings = {name: ReplayBuffer(donor_capacity if name == "risk" else 32000, model.observation_space,
                      model.action_space, device="cpu", n_envs=1, handle_timeout_termination=True)
                      for name in ("risk", "online")}
        self.protected_cases = frozen.old_cases | {(row["geometry"], row["environment_seed"]) for row in frozen.audit}

    def admit(self, folder, expected, online_limit=None):
        from cloud.kaggle import replay
        from cloud.kaggle.local_hybrid import row_ids
        from cloud.kaggle.consolidated_replay import assert_contract

        verify(folder / "WAVE.json", expected)
        batch = read(folder / "WAVE.json")
        kind = batch["kind"]
        relative = folder.relative_to(self.experiment_root).as_posix()
        if batch["schema"] != 1 or kind not in self.rings or any(p["path"] == relative for p in self.waves):
            raise ValueError("Unknown or already admitted replay wave")
        if (kind == "online") != (online_limit is not None) or (online_limit is not None and online_limit <= 0):
            raise ValueError("Online waves require an explicit UTD budget")
        remaining_rows = sum(spec.get("rows", 0) for spec in batch["records"])
        remaining_budget = online_limit
        snap = batch["snapshot"]
        path = Path(snap["snapshot"])
        if path.resolve() != (folder / "donor/snapshot").resolve():
            raise ValueError("Wave snapshot is outside its verified directory")
        verify(path.parent / "SOURCE.json", batch["source_state"])
        record = read(path.parent / "SOURCE.json")
        if (record["files"]["model.zip"], record["updates"], record["identity"]) != (
                snap["policy_sha"], snap["policy_version"], snap["identity"]):
            raise ValueError("Snapshot provenance disagrees with original policy")
        assert_contract(snap["identity"], self.frozen.identity)
        version = replay.check_snapshot(path, snap["snapshot_sha"], snap["identity"])
        if version != snap["policy_version"] or read(path / "SNAPSHOT.json")["policy_sha"] != snap["policy_sha"]:
            raise ValueError("Risk replay policy version mismatch")
        for j, spec in enumerate(batch["records"]):
            if spec["chunk"] != f"chunk-{j:03d}":
                raise ValueError("Unexpected trajectory path in wave")
            if "discard" in spec:
                self.audit.append({"kind": kind, **spec})
                continue
            chunk = folder / spec["chunk"]
            rows, key = replay.validate(chunk, spec["sha"], snap["identity"], self.ledger, version,
                (snap["snapshot_sha"], version), guided=True, collection_phase="mixed",
                behavior=PROFILE if kind == "risk" and self.donor_guidance else "frozen_deterministic")
            if rows is None or key != f"{batch['stream']}:{j}":
                raise ValueError("Replay chunk is duplicate or outside its namespace")
            meta = read(chunk / "CHUNK.json")
            episode, = meta["episodes"]
            case = (episode["geometry_id"], episode["environment_seed"])
            if (case in self.protected_cases or case != (spec["geometry"], spec["seed"])
                    or self.frozen.tiers[case[0]] != spec["tier"] or self.frozen.reward != episode["training_reward_mode_fingerprint"]):
                raise ValueError("Replay case differs from the authorized TRAIN schedule")
            ids = row_ids(rows, episode, self.seen)
            if kind == "risk" and not self.donor_guidance:
                quota = min(len(ids), spec["admit_limit"])
                if quota <= 0:
                    raise ValueError("Donor admission requires a positive row cap")
                ids = [ids[i] for i in np.linspace(0, len(ids) - 1, num=quota, dtype=int)]
            if kind == "online":
                if spec["rows"] != meta["count"]:
                    raise ValueError("Invalid online decision count")
                quota = min(len(ids), (remaining_budget * spec["rows"] + remaining_rows - 1) // remaining_rows)
                if quota < len(ids):
                    ids = [ids[index] for index in np.linspace(0, len(ids) - 1, num=quota, dtype=int)]
                remaining_budget -= quota
                remaining_rows -= spec["rows"]
            if ids:
                if not self.donor_guidance and self.rings[kind].size() + len(ids) > self.rings[kind].buffer_size:
                    raise ValueError("V3 admission would evict previously admitted experience")
                replay.admit(self.rings[kind], {name: data[ids] for name, data in rows.items()})
            self.ledger[key] = spec["sha"]
            self.protected_cases.add(case)
            if kind == "online":
                self.online_rows += len(ids)
            else:
                self.risk_rows += len(ids)
            events = Counter()
            if kind == "risk" and self.donor_guidance:
                events.update(row["reason"] for row in read(chunk / "DECISIONS.json") if row["reason"] != "none")
            self.audit.append({"kind": kind, **spec, "admitted": len(ids),
                               "outcome": meta["guided"]["outcome"], "interventions": dict(events)})
            if not self.donor_guidance:
                self.audit[-1]["wave"] = batch["label"]
        self.waves.append({"path": relative, "sha": expected, "online_limit": online_limit})

    def sample(self, batch_size, env=None):
        import torch
        from stable_baselines3.common.type_aliases import ReplayBufferSamples

        if batch_size != 256:
            raise ValueError("Risk branch requires the frozen V2 batch size")
        buffers = {"base": self.base, "external_easy": self.frozen.external["Easy"],
                   "external_medium": self.frozen.external["Medium"], "v2_online": self.frozen.online, **self.rings}
        order = tuple(MIX)
        if not self.donor_guidance:
            legacy = {f"v1_{tier.lower()}": buf for tier, buf in self.frozen.legacy_tiers.items()}
            buffers.update(legacy)
            order = (*order[:4], *legacy, *order[4:])
        if any(buffers[name].size() < n for name, n in self.mixture.items()):
            raise ValueError("Replay groups are too small for the declared mixture")
        batches = []
        # Manifest JSON sorts keys. Pool order must survive serialization.
        for name in order:
            n = self.mixture[name]
            batches.append(buffers[name].sample(n, env))
            self.draws[name] += n
        if any(batch.discounts is not None for batch in batches):
            raise ValueError("Risk replay expects one-step SAC samples")
        if not self.donor_guidance:
            self.retention_obs = torch.cat([x.observations for x in batches[:-2]], dim=0)
        return ReplayBufferSamples(*(torch.cat(part, dim=0) for part in zip(*(batch[:5] for batch in batches), strict=True)))

    def proof(self):
        from cloud.kaggle import runtime

        state = runtime.rng_state()
        draws = self.draws.copy()
        try:
            data = self.sample(256)
            return hashlib.sha256(b"".join(x.numpy().tobytes() for x in data[:5])).hexdigest()
        finally:
            runtime.restore_rng(state, "cpu")
            self.draws = draws


def save_model(model, replay, parent, source_sha, where, learning):
    import torch
    from cloud.kaggle import runtime

    stage = where.with_name(where.name + ".stage")
    stage.mkdir(parents=True)
    # No replay is written into the immutable anchor. The new checkpoint has
    # its own name for the byte-identical core ring.
    anchor, _ = v2_parent()
    os.link(anchor / "replay.pkl", stage / "replay.pkl")
    model.save(stage / "model.zip")
    torch.save(runtime.rng_state(), stage / "rng.pt")
    metrics = {**read(anchor / "metrics.json"), "risk_learning": learning, "risk_replay": replay.audit}
    write(stage / "metrics.json", metrics)
    marker = {**parent, "job": "stage5-local-risk", "update_mode": PROFILE, "parent_sha": source_sha,
              "updates": model._n_updates, "transitions": model.num_timesteps, "risk": {
                  "profile": PROFILE, "sampling": dict(replay.draws), "waves": replay.waves,
                  "next_sample": replay.proof(), "online_rows": replay.online_rows,
                  "risk_rows": replay.risk_rows, "mixture": replay.mixture},
              "files": {name: sha(stage / name) for name in runtime.PARTS}}
    write(stage / "COMPLETE.json", marker)
    stage.rename(where)
    return sha(where / "COMPLETE.json")


def restore_risk(state, digest, mixture=None):
    import torch
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.logger import configure
    from cloud.kaggle import runtime
    from cloud.kaggle.local_hybrid import HybridSAC, restore, optimizer_steps

    anchor, anchor_sha = v2_parent()
    model, frozen, base = restore(V2_ROOT, anchor, anchor_sha)
    if state == anchor and digest == anchor_sha:
        return model, RiskReplay(model, frozen, mixture), base
    marker = runtime.check_state(state, digest)
    pin = marker["risk"]["mixture"]
    if marker["parent_sha"] != anchor_sha or marker["update_mode"] != PROFILE or pin not in (MIX, RETENTION_MIX):
        raise ValueError("Checkpoint has a different risk recipe or parent")
    del model
    model = HybridSAC.load(state / "model.zip", device="cpu", custom_objects={
        "replay_buffer_class": ReplayBuffer, "replay_buffer_kwargs": {}, "buffer_size": 1,
        "optimize_memory_usage": False, "n_envs": 1,
    })
    if marker["files"]["replay.pkl"] != base["files"]["replay.pkl"]:
        raise ValueError("The risk checkpoint core replay is not the frozen V2 ring")
    model.replay_buffer = frozen.base
    model.set_logger(configure(format_strings=[]))
    replay = RiskReplay(model, frozen, pin)
    for wave in marker["risk"]["waves"]:
        folder = (EXPERIMENT_ROOT / wave["path"]).resolve()
        if not folder.is_relative_to(EXPERIMENT_ROOT.resolve()):
            raise ValueError("Wave is outside the risk experiment")
        replay.admit(folder, wave["sha"], wave["online_limit"])
    replay.draws.update(marker["risk"]["sampling"])
    runtime.restore_rng(torch.load(state / "rng.pt", map_location="cpu", weights_only=False), "cpu")
    if (optimizer_steps(model) != [marker["updates"]] * 3 or model._n_updates != marker["updates"]
            or model.num_timesteps != marker["transitions"] or replay.online_rows != marker["risk"]["online_rows"] or replay.risk_rows != marker["risk"]["risk_rows"]
            or replay.proof() != marker["risk"]["next_sample"]):
        raise ValueError("Risk resume did not reconstruct optimizer, replay and next minibatch")
    return model, replay, marker


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-only", action="store_true")
    parser.add_argument("--behavior-only", action="store_true")
    parser.add_argument("--verify-state", type=Path)
    parser.add_argument("--state-sha")
    parser.add_argument("--retention", action="store_true")
    parser.add_argument("--deadline", type=datetime.fromisoformat, default="2026-11-27T19:30:00+07:00")
    args = parser.parse_args()
    if args.retention:
        ROOT = EXPERIMENT_ROOT / "retention"
    parent, digest = v2_parent()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.verify_state:
        if not args.state_sha:
            parser.error("--verify-state requires the recorded --state-sha")
        import torch

        torch.set_num_threads(2)
        _, mix, manifest = restore_risk(args.verify_state.resolve(), args.state_sha)
        print(f"Verified optimizer step {manifest['updates']} and next minibatch {mix.proof()}")
        return
    if args.prefill_only:
        with Simulators() as sim:
            collect_wave(sim, parent, digest, "risk", 1)
        return
    if args.behavior_only:
        measure_behavior(parent)
        return
    train(args.deadline, parent, digest, retention=args.retention)


def measure_behavior(parent):
    from cloud.kaggle.final_evaluation import summarize

    reference = read(V2_ROOT / "confirmation-05000.json")["cases"]
    result = read(ROOT / "result.json")
    best = result["best_experimental"]
    if best is None:
        raise ValueError("A fully evaluated candidate is required for matched behavior metrics")
    diagnostic = "geom_extreme_SCTXORyCCS_seed2"
    cases = {}
    for tier, doc in reference.items():
        rows = doc["cases"]
        if tier == "Extreme":
            rows = [case for case in rows if case["geometry"] == diagnostic and case["seed"] == 33702] + [
                case for case in rows if case["geometry"] != diagnostic]
        elif tier == "Hard":
            rows = [case for case in rows if case["clean"]] + [case for case in rows if not case["clean"]]
        cases[tier] = {**doc, "cases": rows[:2]}
    state = ROOT / best["state"]
    earlier = EXPERIMENT_ROOT / "behavior-comparison.json"
    outputs = {"V2-5000": read(earlier)["V2-5000"]} if ROOT != EXPERIMENT_ROOT and earlier.is_file() else {}
    with Simulators() as sim:
        for name, folder in (("V2-5000", parent), (best["state"], state)):
            if name in outputs:
                continue
            digest = read(folder / "COMPLETE.json")["files"]["model.zip"]
            panels, performance = assess(sim, folder / "model.zip", digest, cases, TIERS, risk_metrics=True)
            outputs[name] = {"cases": panels, "performance": performance,
                             "summary": {tier: summarize(panel) for tier, panel in panels.items()}}
            write(ROOT / "behavior-comparison.json", outputs)

def train(deadline, parent, digest, *, retention=False):
    import gc
    import psutil
    import torch

    from cloud.kaggle.delivery import export
    from cloud.kaggle.final_evaluation import compare_panels, summarize
    from cloud.kaggle.local_campaign import screen_panel
    from cloud.kaggle.local_hybrid import update
    from cloud.kaggle.runtime import check_state

    torch.set_num_threads(2)
    if os.name == "nt":
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    baseline = read(V2_ROOT / "confirmation-05000.json")["cases"]
    short = screen_panel(baseline)
    for tier in ("Hard", "Extreme"):
        doc = baseline[tier]
        positive = [row for row in doc["cases"] if row["clean"]]
        failing = [row for row in doc["cases"] if not row["clean"]]
        short[tier] = {**doc, "cases": (positive[:1] + failing[:1]) if positive else failing[:2]}
    write(ROOT / "screen-plan.json", short)
    base_short = {tier: summarize(doc) for tier, doc in short.items()}
    model, mix, marker = restore_risk(parent, digest, RETENTION_MIX if retention else MIX)
    state, state_sha = parent, digest
    print("RESTORED V2", marker["updates"], "base_online", mix.frozen.online.size(), flush=True)

    def seconds_left():
        return (deadline - datetime.now(deadline.tzinfo)).total_seconds()

    def add(sim, kind, wave, limit=None):
        original = EXPERIMENT_ROOT / f"{kind}-01"
        if retention and wave == 1:
            folder = original
            if read(folder / "WAVE.json")["source_state"] != digest:
                raise ValueError("Shared initial wave does not belong to the frozen V2 parent")
        else:
            folder = collect_wave(sim, state, state_sha, kind, wave + (30 if retention else 0))
        mix.admit(folder, sha(folder / "WAVE.json"), limit)

    reports = []
    milestones = (128, 512, 1000) if retention else MILESTONES
    with Simulators() as sim:
        add(sim, "risk", 1)
        online_wave = 0
        for index, target in enumerate(milestones):
            if seconds_left() < 19 * 60:
                break
            while mix.online_rows < 2 * target:
                online_wave += 1
                add(sim, "online", online_wave, 2 * target - mix.online_rows)
                if online_wave > 24:
                    raise RuntimeError("Online budget did not converge")
            telemetry = update(model, mix, marker, target)
            state = ROOT / f"checkpoint-{target:05d}"
            state_sha = save_model(model, mix, marker, digest, state, telemetry)
            model_sha = read(state / "COMPLETE.json")["files"]["model.zip"]
            panels, perf = assess(sim, state / "model.zip", model_sha, short, TIERS)
            summary = {tier: summarize(document) for tier, document in panels.items()}
            result = {"updates": target, "state": state.name, "state_sha": state_sha, "screen": summary,
                      "cases": panels, "performance": perf, "learning": telemetry, "risk_rows": mix.risk_rows,
                      "online_rows": mix.online_rows}
            write(ROOT / f"screen-{target:05d}.json", result)
            reports.append(result)
            print("SCREEN", target, {t:(p["clean_successes"],p["safety_failures"]) for t,p in summary.items()},
                  "optimizer", telemetry["optimizer_steps"], flush=True)
            severe = any(summary[t]["clean_successes"] < base_short[t]["clean_successes"] - 1
                         or summary[t]["safety_failures"] > base_short[t]["safety_failures"] + 1
                         or summary[t]["technical_failures"] for t in TIERS)
            if severe or target == milestones[-1]:
                break
            if index < 3:
                add(sim, "risk", index + 2)

        def rank(result):
            screen = result["screen"]
            return (sum(row["technical_failures"] for row in screen.values()),
                    sum(base_short[t]["clean_successes"] - screen[t]["clean_successes"] for t in ("Easy", "Medium")),
                    sum(screen[t]["safety_failures"] for t in TIERS),
                    -sum(screen[t]["clean_successes"] for t in TIERS),
                    -sum(screen[t]["mean_progress"] or 0.0 for t in ("Hard", "Extreme")),
                    result["updates"])

        full = []
        for result in sorted(reports, key=rank)[:1 if retention else 2]:
            if seconds_left() < 480:
                break
            where = ROOT / result["state"]
            pin = check_state(where, result["state_sha"])
            cases, performance = assess(sim, where / "model.zip", pin["files"]["model.zip"], baseline, TIERS)
            pairs = {tier: compare_panels(baseline[tier], cases[tier]) for tier in TIERS}
            keep = all(pair["candidate"]["clean_successes"] >= pair["reference"]["clean_successes"]
                       and pair["candidate"]["safety_failures"] <= pair["reference"]["safety_failures"]
                       and pair["candidate"]["technical_failures"] == 0 for pair in pairs.values())
            gain = any(pair["candidate"]["clean_successes"] > pair["reference"]["clean_successes"]
                       or pair["candidate"]["safety_failures"] < pair["reference"]["safety_failures"]
                       for tier, pair in pairs.items() if tier in ("Hard", "Extreme"))
            decision = {**result, "pairs": pairs, "full_cases": cases,
                        "full_performance": performance, "eligible": keep and gain}
            write(ROOT / f"confirmation-{result['updates']:05d}.json", decision)
            full.append(decision)
            print("CONFIRMED", result["updates"], keep and gain,
                  {t:(p["candidate"]["clean_successes"],p["candidate"]["safety_failures"]) for t,p in pairs.items()},
                  flush=True)
    best = min(full, key=lambda row: (
        not row["eligible"], sum(max(0, pair["reference"]["clean_successes"] - pair["candidate"]["clean_successes"])
                                 for pair in row["pairs"].values()),
        sum(pair["candidate"]["safety_failures"] for pair in row["pairs"].values()),
        -sum(pair["candidate"]["clean_successes"] for pair in row["pairs"].values()),
        -sum(pair["candidate"]["mean_progress"] or 0.0 for pair in row["pairs"].values()))) if full else None
    result = {"selected": best["state"] if best and best["eligible"] else "V2-5000", "best_experimental": best,
              "screens": reports, "confirmations": full, "replay": mix.audit, "waves": mix.waves}
    write(ROOT / "result.json", result)
    del model, mix
    gc.collect()
    if best:
        where = ROOT / best["state"]
        best_sha = best["state_sha"]
        model, mix, manifest = restore_risk(where, best_sha)
        write(ROOT / "resume-proof.json", {"state": best["state"], "state_sha": best_sha, "passed": True,
            "next_sample": mix.proof(), "updates": manifest["updates"], "sampling": dict(mix.draws)})
        export(where, best_sha, ROOT / "experiment-bundle")
    print("SELECTED",result["selected"],flush=True)


if __name__ == "__main__":
    main()
