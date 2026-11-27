"""SAC Consolidation V3 with verified second-window experience."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import gc
import os
from pathlib import Path
import shutil

import numpy as np

from cloud.kaggle.common import read, sha, verify, write
from cloud.kaggle.local_risk import RiskReplay, V2_ROOT, v2_parent
from cloud.kaggle.local_workers import Simulators, assess
from cloud.kaggle.observation_capture import CASES, OUTPUT
from cloud.kaggle.second_window_inventory import ROOT, probes, sensitivity
from cloud.kaggle.smooth_traffic_eval import memory, worker_budget
from cloud.kaggle.v3_collect import TIERS, collect_waves, prepare_wave, train_cases

PILOT_MIX = {"base": 96, "external_easy": 16, "external_medium": 32, "v2_online": 48, "risk": 38, "online": 26}
MIXTURE = {**PILOT_MIX, "base": 80, "v1_hard": 8, "v1_extreme": 8}
MILESTONES = (16, 32, 64, 128, 512, 1000, 2000, 5000)
RECIPE = "v3_frozen_lrn_retention_75_15_10"
RETAINED_ACTOR_LR = 2.5e-6
STEER_RETENTION = 4.


def restored(state=None, digest=None, *, final=True):
    import torch
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.logger import configure
    from cloud.kaggle import runtime
    from cloud.kaggle.local_hybrid import HybridSAC, optimizer_steps, restore

    anchor, anchor_sha = v2_parent()
    metadata = runtime.check_state(state, digest) if state is not None else None
    mixture = MIXTURE if final else PILOT_MIX
    if metadata is not None:
        mixture = metadata["v3"]["mixture"]
    if mixture not in (PILOT_MIX, MIXTURE):
        raise ValueError("Unexpected V3 replay mixture")
    model, frozen, parent = restore(V2_ROOT, anchor, anchor_sha, preserve_legacy_tiers="v1_hard" in mixture)
    if state is None:
        return model, RiskReplay(model, frozen, mixture, experiment_root=ROOT, donor_guidance=False), parent
    pin = metadata["v3"]
    if (metadata["parent_sha"], pin["recipe"], metadata["files"]["replay.pkl"]) != (
            anchor_sha, RECIPE, parent["files"]["replay.pkl"]):
        raise ValueError("Unexpected V3 retention or anchor lineage")
    del model
    model = HybridSAC.load(state / "model.zip", device="cpu", custom_objects={
        "replay_buffer_class": ReplayBuffer, "replay_buffer_kwargs": {}, "buffer_size": 1,
        "optimize_memory_usage": False, "n_envs": 1})
    model.replay_buffer = frozen.base
    model.set_logger(configure(format_strings=[]))
    mix = RiskReplay(model, frozen, mixture, experiment_root=ROOT, donor_guidance=False)
    for wave in pin["waves"]:
        folder = (ROOT / wave["path"]).resolve()
        if not folder.is_relative_to(ROOT.resolve()):
            raise ValueError("Wave is outside the V3 experiment")
        mix.admit(folder, wave["sha"], wave["online_limit"])
    mix.draws.update(pin["sampling"])
    runtime.restore_rng(torch.load(state / "rng.pt", map_location="cpu", weights_only=False), "cpu")
    if (optimizer_steps(model), model._n_updates, model.num_timesteps, mix.risk_rows, mix.online_rows, mix.proof()) != (
            [metadata["updates"]] * 3, metadata["updates"], metadata["transitions"],
            pin["donor_rows"], pin["online_rows"], pin["next_sample"]):
        raise ValueError("V3 file matched but optimizer, replay or next minibatch did not reproduce")
    if pin.get("actor_lr") is not None and getattr(model, "local_actor_lr", None) != pin["actor_lr"]:
        raise ValueError("V3 actor learning rate did not resume")
    return model, mix, metadata


def snapshot(model, mix, parent, folder, learning):
    import torch
    from cloud.kaggle import runtime

    anchor, digest = v2_parent()
    core = ROOT / "immutable-v3-core.pkl"
    if not core.exists():
        temp = core.with_suffix(".staging")
        shutil.copyfile(anchor / "replay.pkl", temp)
        verify(temp, parent["files"]["replay.pkl"])
        temp.rename(core)
    stage = folder.with_name(folder.name + ".stage")
    stage.mkdir(parents=True)
    # All V3 milestones refer to one separate, immutable copy of the core.
    os.link(core, stage / "replay.pkl")
    model.save(stage / "model.zip")
    torch.save(runtime.rng_state(), stage / "rng.pt")
    write(stage / "metrics.json", {"learning": learning, "wave_accounting": mix.audit})
    marker = {**parent, "job": "stage5-local-consolidation-v3", "update_mode": RECIPE, "parent_sha": digest,
              "updates": model._n_updates, "transitions": model.num_timesteps,
              "v3": {"recipe": RECIPE, "mixture": mix.mixture, "waves": mix.waves, "sampling": dict(mix.draws),
                     "next_sample": mix.proof(), "donor_rows": mix.risk_rows, "online_rows": mix.online_rows,
                     "actor_lr": getattr(model, "local_actor_lr", None),
                     "steering_retention": STEER_RETENTION if getattr(model, "local_actor_lr", None) else 0},
              "files": {name: sha(stage / name) for name in runtime.PARTS}}
    if marker["files"]["replay.pkl"] != parent["files"]["replay.pkl"]:
        raise ValueError("V3 core replay no longer matches the frozen parent")
    write(stage / "COMPLETE.json", marker)
    stage.rename(folder)
    return sha(folder / "COMPLETE.json")


def drift(model, observations, baseline):
    action, _ = model.predict(observations, deterministic=True)
    delta = np.abs(action - baseline)
    signs = (abs(action[:, 0]) > .05) & (abs(baseline[:, 0]) > .05)
    return {"steering_mae": float(delta[:, 0].mean()), "steering_p95": float(np.quantile(delta[:, 0], .95)),
            "throttle_mae": float(delta[:, 1].mean()), "throttle_p95": float(np.quantile(delta[:, 1], .95)),
            "steering_reversals": int(np.count_nonzero(signs & (action[:, 0] * baseline[:, 0] < 0))),
            "observations": len(action)}


def screen_fixtures(baseline):
    from cloud.kaggle.local_campaign import screen_panel

    result = screen_panel(baseline)
    for tier in ("Hard", "Extreme"):
        doc = baseline[tier]
        good = [r for r in doc["cases"] if r["clean"]]
        bad = [r for r in doc["cases"] if not r["clean"]]
        result[tier] = {**doc, "cases": good[:2] + bad[:4-len(good[:2])]}
    return result


def screen_accept(before, after):
    return all(not item["technical_failures"] and item["safety_failures"] <= before[tier]["safety_failures"] and (
        tier not in ("Easy", "Medium") or item["clean_successes"] >= before[tier]["clean_successes"])
        for tier, item in after.items())


def retain_steering(model, mix):
    """Apply a V2 steering reference only on the protected V2 minibatch pools."""
    import torch

    reference = deepcopy(model.actor).eval()
    reference.requires_grad_(False)
    model.local_actor_lr = RETAINED_ACTOR_LR
    reports = []

    def regularize(_, _args, _kwargs):
        obs = mix.retention_obs
        with torch.no_grad():
            pinned = reference(obs, deterministic=True)[:, 0]
        steering = model.actor(obs, deterministic=True)[:, 0]
        loss = (steering - pinned).square().mean()
        (STEER_RETENTION * loss).backward()
        reports.append(float(loss.detach()))

    return model.actor.optimizer.register_step_pre_hook(regularize), reports


def campaign(deadline, limit, run):
    import torch
    from cloud.kaggle.final_evaluation import summarize
    from cloud.kaggle.local_campaign import confirm, confirmation_rank
    from cloud.kaggle.local_hybrid import update

    torch.set_num_threads(2)
    run.mkdir(parents=True)
    model, mix, parent = restored(final=run.name == "v3")
    hook, retention_losses = retain_steering(model, mix) if run.name != "pilot" else (None, [])
    observations = np.concatenate([
        np.load(OUTPUT / "capture" / f"{name}.npz", allow_pickle=False)["policy"][::3]
        for name, *_ in CASES])
    fixed_actions, _ = model.predict(observations, deterministic=True)
    sensor = probes()
    sensitivity_base = sensitivity(model, sensor)
    for marker in sorted(ROOT.glob("donor-*/WAVE.json")):
        mix.admit(marker.parent, sha(marker))
    donors = {record["wave"] for record in mix.audit if record.get("admitted", 0)}
    if len(donors) < 2 or mix.risk_rows < 2500:
        raise ValueError("V3 requires multiple verified donor policies and at least 2,500 new rows")
    baseline = read(V2_ROOT / "confirmation-05000.json")["cases"]
    screen = screen_fixtures(baseline)
    reference = {tier: summarize(data) for tier, data in screen.items()}
    report = {"status": "WARMUP", "donors": sorted(donors), "milestones": [], "selected": "V2-5000"}
    write(run / "result.json", report)
    state, state_sha = v2_parent()
    online_wave = 0
    prior = None

    def left():
        return (deadline - datetime.now(deadline.tzinfo)).total_seconds()

    for target in MILESTONES:
        if target > limit or left() < 30 * 60 or shutil.disk_usage(run).free < 2 * 1024**3:
            break
        minimum = max(4096, target * 2, (target * 26 + 9) // 10)
        while mix.online_rows < minimum:
            online_wave += 1
            if online_wave > 12 or left() < 30 * 60:
                raise RuntimeError("Online diversity or time budget exhausted")
            label = f"online-{online_wave:02d}" if online_wave == 1 else f"{run.name}-online-{online_wave:02d}"
            marker = ROOT / label / "WAVE.json"
            if not marker.exists():
                tiers = ("Easy", "Medium", "Hard", "Extreme", "Medium", "Hard", "Extreme", "Medium")
                collect_waves([prepare_wave(label, state, state_sha, "online", 639 + online_wave,
                                            train_cases(99 + online_wave, tiers))])
            data = read(marker)
            rows = sum(row.get("rows", 0) for row in data["records"])
            if rows:
                mix.admit(marker.parent, sha(marker), min(rows, 30000 - mix.online_rows))
            print("ONLINE", mix.online_rows, len([r for r in mix.audit if r["kind"] == "online"]), flush=True)
        online = [record for record in mix.audit if record["kind"] == "online" and record.get("admitted")]
        if len({r["geometry"] for r in online}) < 6 or len({r["tier"] for r in online}) < 3:
            raise ValueError("Online samples do not cover enough independent geometry and tiers")
        learning = update(model, mix, parent, target)
        if hook is not None:
            if len(retention_losses) != target:
                raise ValueError("Every actor update must apply the frozen-reference retention loss")
            learning["steer_reference_loss"] = float(np.mean(retention_losses[-learning["updates"]:]))
        change = drift(model, observations, fixed_actions)
        steering_bad = change["steering_p95"] > .35 or change["steering_reversals"] > .10 * change["observations"]
        pedal_bad = change["throttle_p95"] > .60
        state = run / f"stage-{target:05d}"
        state_sha = snapshot(model, mix, parent, state, learning)
        new = {"updates": target, "state": state.name, "sha": state_sha, "learning": learning, "drift": change,
               "sensitivity": sensitivity(model, sensor), "donor_rows": mix.risk_rows, "online_rows": mix.online_rows,
               "drift_accepted": not (steering_bad or pedal_bad)}
        report["milestones"].append(new)
        report.update(status="PILOT", sensitivity_v2=sensitivity_base, audit=mix.audit, sampling=dict(mix.draws),
                      wave_paths=mix.waves, memory=memory(), free_disk_bytes=shutil.disk_usage(run).free)
        write(run / "result.json", report)
        print("MILESTONE", target, change, "safe", new["drift_accepted"], flush=True)
        if not new["drift_accepted"]:
            break
        prior = new
        if target >= 128:
            profile = memory()
            policy_sha = read(state / "COMPLETE.json")["files"]["model.zip"]
            with Simulators(worker_budget(profile)) as sim:
                results, throughput = assess(sim, state / "model.zip", policy_sha, screen, TIERS, risk_metrics=True)
            summary = {tier: summarize(value) for tier, value in results.items()}
            new.update(summary=summary, screening_cases=results, performance=throughput,
                       screen_accepted=screen_accept(reference, summary))
            write(run / "result.json", report)
            print("SCREEN", target, new["screen_accepted"],
                  {tier: (v["clean_successes"], v["safety_failures"]) for tier,v in summary.items()}, flush=True)
            if not new["screen_accepted"]:
                break
    report["status"] = "POST_PILOT"
    write(run / "result.json", report)
    safe = [r for r in report["milestones"] if r.get("screen_accepted")]
    if not safe:
        safe = [r for r in report["milestones"] if r["drift_accepted"] and "summary" not in r][-1:]
    safe.sort(key=lambda x: (-(sum(t["clean_successes"] for t in x["summary"].values()) if "summary" in x else 0),
                             sum(t["safety_failures"] for t in x.get("summary", {}).values())))
    if hook is not None:
        hook.remove()
    del model, mix
    gc.collect()
    evaluated = []
    for candidate in safe[:1]:
        if left() > 14 * 60:
            with Simulators(worker_budget(memory())) as sim:
                evaluated.append(confirm(sim, run, candidate, baseline))
    report["confirmations"] = evaluated
    if evaluated:
        best = min(evaluated, key=confirmation_rank)
        if best["eligible"]:
            report["selected"] = best["state"]
    proof = min(evaluated, key=confirmation_rank) if evaluated else prior
    if proof:
        model, replay, manifest = restored(run / proof["state"], proof["sha"])
        write(run / "resume-proof.json", {"state": proof["state"], "manifest_sha256": proof["sha"],
              "passed": True, "updates": manifest["updates"], "next_sample_sha256": replay.proof()})
    report["status"] = "COMPLETE"
    write(run / "result.json", report)
    print("SELECTED", report["selected"], flush=True)


def review_stage(run, target, deadline):
    """Recheck a previously saved, drift-accepted checkpoint without training."""
    import torch
    from cloud.kaggle.final_evaluation import summarize
    from cloud.kaggle.local_campaign import confirm

    if (deadline - datetime.now(deadline.tzinfo)).total_seconds() < 20 * 60:
        raise ValueError("Insufficient time for paired review")
    torch.set_num_threads(1)
    report = read(run / "result.json")
    if report["status"] != "COMPLETE" or report["selected"] != "V2-5000":
        raise ValueError("Stage review requires a completed run still using the V2 fallback")
    record = next(row for row in report["milestones"] if row["updates"] == target)
    if not record["drift_accepted"] or record.get("summary"):
        raise ValueError("Review requires a held checkpoint that passed offline drift only")
    baseline = read(V2_ROOT / "confirmation-05000.json")["cases"]
    fixtures = screen_fixtures(baseline)
    source = run / record["state"]
    pin = read(source / "COMPLETE.json")
    with Simulators(worker_budget(memory())) as sim:
        data, throughput = assess(sim, source / "model.zip", pin["files"]["model.zip"], fixtures, TIERS)
        summary = {tier: summarize(value) for tier, value in data.items()}
        record.update(summary=summary, performance=throughput, screening_cases=data,
                      screen_accepted=screen_accept({tier: summarize(value) for tier, value in fixtures.items()}, summary))
        if record["screen_accepted"]:
            check = confirm(sim, run, record, baseline)
            report["confirmations"].append(check)
            if check["eligible"]:
                _, mix, manifest = restored(source, record["sha"])
                write(run / "resume-proof.json", {"state": record["state"], "manifest_sha256": record["sha"],
                      "passed": True, "updates": manifest["updates"], "next_sample_sha256": mix.proof()})
                report["selected"] = record["state"]
    write(run / "result.json", report)
    print("REVIEW", target, record["screen_accepted"], report["selected"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", type=datetime.fromisoformat, required=True)
    parser.add_argument("--max-updates", type=int, choices=MILESTONES, default=2000)
    parser.add_argument("--run", choices=("pilot", "tuned-pilot", "v3"), default="v3")
    parser.add_argument("--review-update", type=int, choices=MILESTONES)
    args = parser.parse_args()
    if args.review_update:
        review_stage(ROOT / args.run, args.review_update, args.deadline)
    else:
        campaign(args.deadline, args.max_updates, ROOT / args.run)


if __name__ == "__main__":
    main()
