"""One-shot Kaggle preflight packaging, submission and artifact harvest."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import mkdtemp
import zipfile

from cloud.kaggle.common import METADRIVE_PIN, ROOT, SOURCE, SOURCE_SHA, WORK, fingerprint, read, sha, verify, write
from cloud.kaggle.checkpoint import restore_checkpoint
from cloud.kaggle.credentials import (
    SLOTS, accelerator_quota, cli, current_private_kernel_version, download_kernel_output, operator_slot,
    private_input, slot, verify_owner,
)

RECIPE = read(Path(__file__).with_name("recipe.json"))
BUNDLE_PATHS = (
    "cloud/kaggle/common.py", "cloud/kaggle/runtime.py", "cloud/kaggle/replay.py", "cloud/kaggle/checkpoint.py",
    "cloud/kaggle/learning_hooks.py", "cloud/kaggle/guided.py", "cloud/kaggle/campaign_tasks.py",
    "cloud/kaggle/traffic_guidance.py",
    "cloud/kaggle/recipe.json", "cloud/kaggle/linux-lock.txt", "cloud/kaggle/kernel_template.py.txt",
    "configs/experiments/stage5_native_reward_integration_v1.json",
    "results/audits/evaluation_protocol/geometry_split_manifest.csv",
    "results/stage5/pilot_split_v1.json", "src/__init__.py", "src/training/__init__.py",
    "src/training/stage5/__init__.py", "src/training/stage5/env_adapter.py",
    "src/training/stage5/native_reward_training.py", "src/training/stage5/observation_encoder.py",
    "src/training/stage5/pilot_observation_transform.py", "src/training/stage5/train_sampler.py",
    *("src/agents/" + name + ".py" for name in (
        "__init__", "MCTS_agent", "astar_agent", "graph_search", "graph_search_agents", "heuristic_model",
        "hybrid_astar", "hybrid_astar_agent", "mcts_model", "path_following", "planning_graph",
        "random_naive_agent", "stage5_sac_agent", "stage5_terminal_preview_v1")),
    *("src/launcher/" + name + ".py" for name in (
        "__init__", "cases", "contracts", "events", "executor", "models", "preflight", "registry", "resolver")),
    *("src/platform/" + name + ".py" for name in (
        "__init__", "action_adapter", "agent", "agent_input", "episode", "experiment_logging", "metrics",
        "protocol", "reference_station", "reward", "specs", "task_context_producer", "wandb_backend")),
    *("datasets/mapsuite_v1/" + name for name in (
        "dataset_manifest.json", "generation_config.json", "geometries.csv", "scenario_families.csv",
        "splits/train_geometries.csv", "splits/validation_geometries.csv", "splits/test_geometries.csv")),
)
STATE_FILES = {"model.zip", "replay.pkl", "rng.pt", "metrics.json", "COMPLETE.json"}
OUTPUT_FILES = STATE_FILES | {"runtime_versions.txt", "execution.json", "parity.json"}
LONG_PARENT_LIMIT = 512_000_000
SECRET = re.compile(rb"(?i)(?:KGAT_[a-zA-Z0-9_-]{12,}|gh[pousr]_[a-zA-Z0-9_]{20,}|github_pat_[a-zA-Z0-9_]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|AIza[a-zA-Z0-9_-]{30,})")
ACTIVE = {"SUBMITTING", "SUBMIT_UNKNOWN", "SUBMITTED", "QUEUED", "RUNNING", "STATUS_UNKNOWN"}
REPAIR_FILES = {
    "PYTHON_ABI": {"cloud/kaggle/kernel_template.py.txt"},
    "DEPENDENCY_RESOLUTION": {"cloud/kaggle/linux-lock.txt", "cloud/kaggle/kernel_template.py.txt"},
    "RUNTIME_IMPORT": {"cloud/kaggle/linux-lock.txt", "cloud/kaggle/kernel_template.py.txt", "cloud/kaggle/runtime.py"},
    "NATIVE_RUNTIME_CRASH": {"cloud/kaggle/linux-lock.txt", "cloud/kaggle/kernel_template.py.txt", "cloud/kaggle/runtime.py"},
    "METADRIVE_ASSETS": {"cloud/kaggle/kernel_template.py.txt", "cloud/kaggle/linux-lock.txt"},
    "PRIVATE_INPUT": {"cloud/kaggle/kernel_template.py.txt"},
}
KERNEL_STATES = {"complete": "CLOUD_SUCCEEDED", "error": "FAILED", "cancelled": "CANCELLED",
                 "cancel_acknowledged": "CANCELLED", "cancel_requested": "RUNNING",
                 "new_script": "QUEUED", "running": "RUNNING", "queued": "QUEUED"}
CAPACITY_ERROR = re.compile(
    r"\b(?:too many (?:active |concurrent |running )?(?:notebook |cpu )?(?:sessions|kernels|notebooks)"
    r"|(?:no|zero) (?:free |available )?(?:cpu |notebook )?sessions? (?:available|remaining|left)"
    r"|(?:session|concurrency|concurrent sessions?) (?:limit|quota) (?:reached|exceeded)"
    r"|(?:maximum|max) number of (?:concurrent |active |running )?(?:notebook |cpu )?sessions"
    r"|maximum (?:batch )?(?:cpu |notebook )?session count of \d+ (?:reached|exceeded))\b",
    re.I,
)

def now():
    stamp = datetime.now(timezone.utc)
    return {"utc": stamp.isoformat(), "ict": stamp.astimezone(timezone(timedelta(hours=7))).isoformat()}

@contextmanager
def locked():
    WORK.mkdir(parents=True, exist_ok=True)
    lock = WORK / "registry.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError("Registry locked. Check for a live launcher and reconcile pending receipts before manual recovery.") from exc
    try:
        yield
    finally:
        lock.rmdir()

def archive(target, entries):
    with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as out:
        for name, data in sorted(entries.items()):
            member = zipfile.ZipInfo(name, (2026, 11, 25, 0, 0, 0))
            out.writestr(member, data, compress_type=zipfile.ZIP_DEFLATED)

def package(base_job=None):
    source = SOURCE
    if base_job:
        entry = registry()["jobs"][base_job]
        if entry["status"] != "HARVESTED_VERIFIED" or entry.get("imported_evidence"):
            raise ValueError("Package inputs must come from a locally harvested learner")
        previous = entry["planned"]
        inputs = (ROOT / previous["runtime_path"]).resolve()
        if not inputs.is_relative_to(WORK.resolve()):
            raise ValueError("Base runtime path escapes the campaign workspace")
        verify(inputs / "runtime.zip", previous["runtime_sha"])
        with zipfile.ZipFile(inputs / "runtime.zip") as bundle:
            anchor = bundle.read("parity.json")
        source = inputs / "model.zip"
    else:
        path = WORK / "parity.json"
        if not path.is_file():
            raise RuntimeError("Run the local parity command first")
        anchor = path.read_bytes()
    verify(source, SOURCE_SHA)
    files = {name: (ROOT / name).read_bytes() for name in BUNDLE_PATHS}
    files["parity.json"] = anchor
    if any(SECRET.search(data) for data in files.values()):
        raise RuntimeError("Secret-pattern audit failed for allowlisted runtime")
    identifiers = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    inventory = {"git_head_at_packaging": base, "runtime_id": fingerprint(identifiers), "files": identifiers,
                 "source_model_sha256": SOURCE_SHA, "metadrive_pin": METADRIVE_PIN}
    files["source_inventory.json"] = json.dumps(inventory, sort_keys=True).encode()
    staging = WORK / ("package-" + fingerprint(inventory)[:20])
    if not staging.exists():
        staging.mkdir()
        archive(staging / "runtime.zip", files)
        shutil.copyfile(source, staging / "model.zip")
        write(staging / "source_inventory.json", inventory)
    verify(staging / "model.zip", SOURCE_SHA)
    with zipfile.ZipFile(staging / "runtime.zip") as bundle:
        if set(bundle.namelist()) != set(files) or any(bundle.read(name) != data for name, data in files.items()):
            raise ValueError("Existing immutable package differs. Preserve it and investigate.")
    result = {"runtime_sha": sha(staging / "runtime.zip"), "source_sha": SOURCE_SHA,
              "runtime_bytes": (staging / "runtime.zip").stat().st_size, "source_bytes": source.stat().st_size,
              "runtime_id": inventory["runtime_id"], "path": str(staging.relative_to(ROOT)),
              "template_sha": identifiers["cloud/kaggle/kernel_template.py.txt"]}
    write(WORK / "bundle.json", result)
    return result

def registry():
    path = WORK / "job_registry.json"
    return read(path) if path.exists() else {"schema": 1, "jobs": {}, "uploads": {}}

def persist(state):
    write(WORK / "job_registry.json", state)

class DatasetPending(RuntimeError):
    pass

def audit_zip(path, label):
    with zipfile.ZipFile(path) as bundle:
        names = bundle.namelist()
        limit = LONG_PARENT_LIMIT if label == "long-parent" else 120_000_000
        if len(set(names)) != len(names) or sum(x.file_size for x in bundle.infolist()) > limit:
            raise ValueError("Unexpected upload archive size or duplicate members")
        if label == "runtime":
            current = set(BUNDLE_PATHS) | {"parity.json", "source_inventory.json"}
            pre_traffic = current - {"cloud/kaggle/traffic_guidance.py"}
            baseline = pre_traffic - {"cloud/kaggle/campaign_tasks.py"}
            previous = baseline - {"cloud/kaggle/guided.py"}
            older = previous - {"cloud/kaggle/learning_hooks.py"}
            oldest = older - {"cloud/kaggle/checkpoint.py"}
            if set(names) not in (current, pre_traffic, baseline, previous, older, oldest):
                raise ValueError("Runtime upload member allowlist mismatch")
        if label in {"parent", "long-parent"} and set(names) != STATE_FILES:
            raise ValueError("Parent upload member allowlist mismatch")
        if label == "snapshot" and set(names) != {"policy.zip", "SNAPSHOT.json"}:
            raise ValueError("B1 snapshot upload member allowlist mismatch")
        if label == "replay" and set(names) != {"CHUNK.json", "rows.npz"}:
            raise ValueError("B1 replay upload member allowlist mismatch")
        if label == "guided":
            manifest = json.loads(bundle.read("GUIDED.json"))
            allowed = {"GUIDED.json", "snapshot/SNAPSHOT.json", "snapshot/policy.zip"}
            allowed.update(f"chunk-{i:02}/{name}" for i in range(len(manifest["chunks"])) for name in ("CHUNK.json", "rows.npz"))
            if not 1 <= len(manifest["chunks"]) <= 39 or set(names) != allowed:
                raise ValueError("Unexpected guided replay archive members")
        for name in names:
            with bundle.open(name) as stream:
                overlap = b""
                while block := stream.read(1024 * 1024):
                    if SECRET.search(overlap + block):
                        raise ValueError("Secret-pattern audit failed inside upload archive")
                    overlap = block[-128:]
            if (label, name) in {("parent", "model.zip"), ("long-parent", "model.zip"), ("snapshot", "policy.zip"),
                                 ("guided", "snapshot/policy.zip")}:
                from io import BytesIO
                if bundle.getinfo(name).file_size > 32_000_000:
                    raise ValueError("Unexpected compressed parent model archive size")
                with zipfile.ZipFile(BytesIO(bundle.read(name))) as model:
                    if sum(x.file_size for x in model.infolist()) > 32_000_000:
                        raise ValueError("Unexpected parent model archive size")
                    if any(SECRET.search(model.read(member)) for member in model.infolist()):
                        raise ValueError("Secret-pattern audit failed inside parent model")

def identity_part(text):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,17}", text):
        raise ValueError("Campaign and round identifiers require 1–18 lowercase slug characters")

def plan(args, state):
    identity_part(args.campaign)
    identity_part(args.round)
    capacity_after = getattr(args, "capacity_after", None)
    anchor_id = args.parent or capacity_after
    anchor = state["jobs"].get(anchor_id)
    budget = RECIPE["preflight"]
    if anchor_id:
        if not anchor or anchor["status"] != "HARVESTED_VERIFIED":
            raise ValueError("Follow-up requires a verified harvested job")
        if anchor["execution"]["campaign"] != args.campaign:
            raise ValueError("Follow-up requires a job from the same campaign")
        if capacity_after and anchor.get("gates", {}).get("G2") != "PASS":
            raise ValueError("Capacity plan requires a verified G2 artifact")
        pinned = anchor["planned"]
        bundle = {key: pinned[key] for key in ("runtime_sha", "source_sha", "template_sha")}
        bundle["path"] = pinned["runtime_path"]
        budget = {key: pinned[key] for key in budget}
    else:
        bundle = read(WORK / "bundle.json")
    plan_id = f"{args.campaign}-{args.round}"
    assignments = dict(item.split("=") for item in getattr(args, "assign", []))
    allowed_jobs = {"j07"} if args.parent else {"j02", "j03", "j04"}
    if not anchor_id:
        allowed_jobs.add("j01")
    if set(assignments) - allowed_jobs or set(assignments.values()) - set(SLOTS):
        raise ValueError("Job assignment format is j01=account_01")
    jobs = {}
    for label, role, workers, device in (
        ("j01", "parity", 1, "cpu"), ("j02", "cpu1", 1, "cpu"),
        ("j03", "cpu2", 2, "cpu"), ("j04", "gpu2", 2, "cuda"),
    ):
        if label not in allowed_jobs:
            continue
        short_slug = "s5-" + fingerprint([args.campaign, args.round])[:12] + "-" + label + "-a1"
        jobs[label] = {"job": short_slug, "campaign": args.campaign, "round": args.round, "attempt": 1,
                       "role": role, "account": assignments.get(label, args.account), "workers": workers, "device": device,
                       "runtime_sha": bundle["runtime_sha"], "source_sha": bundle["source_sha"],
                       "runtime_path": bundle["path"], "template_sha": bundle["template_sha"], **budget}
    deferred = {"j05": "Local B1 collector proof only; cloud snapshot sharing unverified",
                "j06": "Depends on a verified private collector artifact",
                "j07": "Depends on a harvested COMPLETE online state"}
    if args.parent:
        parent = anchor["execution"]
        job = {"job": "s5-" + fingerprint([args.campaign, args.round])[:12] + "-j07-a1",
               "campaign": args.campaign, "round": args.round, "attempt": 1, "role": "resume",
               "account": assignments.get("j07", args.account), "workers": parent["workers"], "device": parent["device"],
               "runtime_sha": bundle["runtime_sha"], "source_sha": bundle["source_sha"],
               "runtime_path": bundle["path"], "template_sha": bundle["template_sha"], "parent_job": args.parent,
               "parent_sha": anchor["state_sha"], **budget}
        jobs = {"j07": job}
        deferred.pop("j07")
    save_plan(plan_id, jobs, deferred, state)


def save_plan(plan_id, jobs, deferred, state):
    body = {"jobs": jobs, "deferred": deferred}
    digest = fingerprint(body)
    path = WORK / f"plan-{plan_id}.json"
    doc = {"fingerprint": digest, **body}
    if path.exists() and read(path) != doc:
        raise ValueError("Immutable plan already exists. Use a new round ID.")
    write(path, doc)
    for job in jobs.values():
        state["jobs"].setdefault(job["job"], {"status": "PLANNED", "planned": job, "at": now()})
    persist(state)
    print(f"{path.relative_to(ROOT)} SHA={digest}")


def plan_pilot(args, state):
    for value in (args.campaign, args.round, args.capacity_round):
        identity_part(value)
    evidence = read(WORK / f"{args.capacity_round}-capacity-report.json")
    if evidence["G4"] != "PASS" or set(evidence["samples"]) != {"j02", "j03", "j04"}:
        raise RuntimeError("Easy pilot requires verified CPU1, CPU2 and GPU2 capacity samples")
    samples = evidence["samples"]
    reference = getattr(args, "reference_job", None)
    guided = getattr(args, "guided_prefill", None)
    for row in samples.values():
        receipt = state["jobs"][row["job"]]
        if receipt["status"] != "HARVESTED_VERIFIED" or receipt["output_version"] != row["version"]:
            raise RuntimeError("Capacity sample does not match its current harvested receipt")
    cpu = max((samples["j02"], samples["j03"]), key=lambda row: (
        row["end_to_end_transitions_per_second"], row["post_warmup_transitions_per_second"]
    ))
    gpu = samples["j04"]
    selected = gpu if (gpu["end_to_end_transitions_per_second"] > cpu["end_to_end_transitions_per_second"]
                       and gpu["post_warmup_transitions_per_second"] > cpu["post_warmup_transitions_per_second"]) else cpu
    if reference:
        selected = samples["j03"] if guided else gpu
    validated = state["jobs"][selected["job"]]["planned"]
    bundle = read(WORK / "bundle.json")
    allowed_changes = {"cloud/kaggle/runtime.py", "cloud/kaggle/kernel_template.py.txt",
                       "cloud/kaggle/checkpoint.py", "cloud/kaggle/replay.py", "cloud/kaggle/learning_hooks.py",
                       "cloud/kaggle/guided.py"}
    inventories = []
    for path, digest in ((validated["runtime_path"], validated["runtime_sha"]), (bundle["path"], bundle["runtime_sha"])):
        source = (ROOT / path / "runtime.zip").resolve()
        if not source.is_relative_to(WORK.resolve()):
            raise ValueError("Pilot bundle path escapes campaign workspace")
        verify(source, digest)
        with zipfile.ZipFile(source) as archive:
            inventories.append(json.loads(archive.read("source_inventory.json"))["files"])
    before, after = inventories
    if ("cloud/kaggle/checkpoint.py" not in after or bundle["source_sha"] != validated["source_sha"]
            or {k: v for k, v in before.items() if k not in allowed_changes}
            != {k: v for k, v in after.items() if k not in allowed_changes}):
        raise RuntimeError("Pilot bundle must match the verified physics, dependencies and parity anchors")
    label = "j11" if guided else "j09" if reference else "j08"
    job = {"job": "s5-" + fingerprint([args.campaign, args.round])[:12] + "-" + label + "-a1",
           "campaign": args.campaign, "round": args.round, "attempt": 1, "role": "easy_pilot",
           "account": args.account, "workers": validated["workers"], "device": validated["device"],
           "runtime_sha": bundle["runtime_sha"], "source_sha": bundle["source_sha"], "runtime_path": bundle["path"],
           "template_sha": bundle["template_sha"], "validated_runtime_sha": validated["runtime_sha"],
           "capacity_sample_job": selected["job"], "phase": "easy", "checkpoint_every": 4096,
           **{key: validated[key] for key in RECIPE["preflight"]}}
    job["transitions"] = 8192
    if reference:
        previous = state["jobs"][reference]
        if previous["status"] != "HARVESTED_VERIFIED" or previous["planned"]["role"] != "easy_pilot":
            raise RuntimeError("Longer pilot requires the verified Easy pilot as an evaluation reference")
        report_path = args.evaluation_report.resolve()
        if not report_path.is_relative_to(WORK.resolve()):
            raise ValueError("Evaluation report must be stored in the campaign workspace")
        evaluation = read(report_path)
        confirmed = set()
        for step in (4096, 8192):
            _, manifest = pilot_checkpoint(state, reference, step)
            confirmed.add(manifest["files"]["model.zip"])
        if (evaluation["baseline_clean_successes"] != 8 or evaluation["baseline_cases"] != 12
                or {row["model_sha"] for row in evaluation["milestones"]} != confirmed):
            raise ValueError("Evaluation does not match the two verified Easy pilot checkpoints")
        job.update(role="easy_long", transitions=args.transitions, capacity=65536, max_seconds=2400,
                   export_reserve=480, checkpoint_every=12500, sharded_output=True,
                   measure_optimizer=True, evaluate_easy=True, evaluation_sha=sha(report_path),
                   reference_job=reference, accelerator="NvidiaTeslaT4")
        if "cloud/kaggle/learning_hooks.py" in after:
            job["policy_every"] = 2500
            job["warm_start_cursor"] = next_easy_cursor(state, args.campaign)
        if args.checkpoint != "source":
            step = int(args.checkpoint)
            _, manifest = pilot_checkpoint(state, reference, step)
            job.update(warm_step=step, warm_model_sha=manifest["files"]["model.zip"])
        if guided:
            archive = guided.resolve(strict=True)
            if not archive.is_relative_to(WORK.resolve()) or args.transitions != 5000 or args.checkpoint != "4096":
                raise ValueError("Guided pilot requires the approved champion and a 5,000-step budget")
            audit_zip(archive, "guided")
            with zipfile.ZipFile(archive) as replay:
                snapshot = json.loads(replay.read("snapshot/SNAPSHOT.json"))
            if snapshot["policy_sha"] != job["warm_model_sha"]:
                raise ValueError("Guided replay is tied to a different champion")
            job.pop("accelerator")
            job.update(role="easy_guided", capacity=16384, max_seconds=1200, export_reserve=300,
                       checkpoint_every=5000, learning_starts=0, guided_sha=sha(archive),
                       guided_path=str(archive.relative_to(ROOT)))
        elif args.transitions == 5000:
            raise ValueError("The 5,000-step profile requires approved guided replay")
    require_gate(state, job)
    save_plan(f"{args.campaign}-{args.round}", {label: job}, {}, state)


def pilot_checkpoint(state, job, step):
    receipt = state["jobs"][job]
    if step == receipt["planned"]["transitions"]:
        path = WORK / "harvest" / job
        digest = receipt["state_sha"]
    else:
        path = WORK / "periodic" / job / f"step-{step:012d}"
        digest = receipt["periodic"][str(step)]["state_sha"]
    verify(path / "COMPLETE.json", digest)
    manifest = read(path / "COMPLETE.json")
    if manifest["job"] != job or manifest["transitions"] != step:
        raise ValueError("Unexpected Easy pilot checkpoint milestone")
    verify(path / "model.zip", manifest["files"]["model.zip"])
    return path, manifest


def next_easy_cursor(state, campaign):
    cursors = []
    for entry in state["jobs"].values():
        planned = entry["planned"]
        if entry["status"] != "HARVESTED_VERIFIED" or planned["campaign"] != campaign or planned.get("phase") != "easy":
            continue
        marker = WORK / "harvest" / planned["job"] / "COMPLETE.json"
        verify(marker, entry["state_sha"])
        cursors.extend(worker["next_cursor"] for worker in read(marker)["workers"])
    return max(cursors, default=0)


def parent_checkpoint(state, parent_id, step=None):
    receipt = state["jobs"][parent_id]
    if step is not None:
        milestone = receipt.get("periodic", {}).get(str(step))
        if not milestone or not receipt.get("version") or milestone.get("version") != receipt["version"]:
            raise ValueError("Resume requires a verified periodic checkpoint from the recorded version")
        return WORK / "periodic" / parent_id / f"step-{step:012d}", milestone["state_sha"]
    if receipt["status"] != "HARVESTED_VERIFIED":
        raise ValueError("Resume requires a verified harvested final checkpoint")
    return WORK / "harvest" / parent_id, receipt["state_sha"]


def plan_learning_resume(args, state):
    identity_part(args.campaign)
    identity_part(args.round)
    previous = state["jobs"][args.parent]
    planned = previous["planned"]
    guided = planned["role"] == "easy_guided" or planned.get("guided_resume", False)
    if planned["role"] not in {"easy_long", "easy_resume", "easy_guided"} or planned["campaign"] != args.campaign:
        raise RuntimeError("Full-state continuation requires a harvested long Easy checkpoint from this campaign")
    compare_utd = getattr(args, "reduce_guided_utd_after", None)
    actor_lr_control = getattr(args, "dampen_guided_actor_after", None)
    comparison_id = compare_utd or actor_lr_control
    deadline = getattr(args, "campaign_deadline", None)
    if deadline is not None and (not guided or comparison_id or not
            1800 <= deadline - datetime.now(timezone.utc).timestamp() <= 43200):
        raise ValueError("Timeboxed GuidedSAC requires 30 minutes to 12 hours remaining")
    budgets = (5000,) if comparison_id else ((10000, 12500, 25000, 50000, 100000) if guided else (12500, 25000, 50000))
    if args.transitions not in ((4_000_000,) if deadline else budgets):
        raise ValueError("Unsupported continuation budget for this learner")
    step = getattr(args, "parent_step", None)
    parent, state_sha = parent_checkpoint(state, args.parent, step)
    state_path = parent / "COMPLETE.json"
    verify(state_path, state_sha)
    checkpoint = read(state_path)
    device = getattr(args, "device", None)
    seconds = getattr(args, "max_seconds", None)
    next_phase = getattr(args, "phase", None)
    traffic_guidance = getattr(args, "traffic_guidance", False)
    teacher = getattr(args, "traffic_teacher", False)
    autonomous = getattr(args, "autonomous_traffic", False)
    if (device or seconds or next_phase or traffic_guidance or teacher or autonomous) and (not guided or comparison_id):
        raise ValueError("Production options require a Guided SAC continuation")
    if autonomous and (traffic_guidance or teacher):
        raise ValueError("Autonomous collection and traffic guidance cannot be combined")
    if seconds is not None and not 1800 <= seconds <= (43200 if deadline else 14400):
        raise ValueError("Production time budget must be 1,800–14,400 seconds")
    if comparison_id:
        comparison = state["jobs"][comparison_id]
        execution = comparison.get("execution", {})
        if (not guided or checkpoint["identity"].get("guided_gradient_steps", -1) != -1
                or comparison["status"] != "HARVESTED_VERIFIED"
                or execution.get("parent_sha") != state_sha or comparison["planned"]["account"] != args.account
                or not execution.get("guided_resume") or execution.get("guided_actor_lr_scale")
                or execution.get("guided_gradient_steps") != (1 if actor_lr_control else None)
                or execution["workers"] != 2 or execution["device"] != "cpu"):
            raise ValueError("The matched profile requires a verified two-environment CPU control from the same parent")
        if actor_lr_control:
            utd_control = state["jobs"].get(execution.get("utd_control"), {})
            if (execution["transitions"] != args.transitions or utd_control.get("status") != "HARVESTED_VERIFIED"
                    or utd_control["execution"].get("parent_sha") != state_sha):
                raise ValueError("The actor profile requires both harvested UTD controls from the same parent")
    bundle = read(WORK / "bundle.json")
    allowed = {"cloud/kaggle/kernel_template.py.txt", "cloud/kaggle/learning_hooks.py", "cloud/kaggle/campaign_tasks.py",
               "cloud/kaggle/traffic_guidance.py",
               "src/launcher/registry.py", "src/launcher/executor.py"}
    if guided:
        if (checkpoint.get("learning_mode") != "champion_guided_v1"
                or checkpoint["identity"].get("online_behavior") != "champion_guided_v1"
                or checkpoint["external_admitted"] <= 0 or not checkpoint["ledger"]):
            raise ValueError("Guided continuation requires the verified behavior and replay provenance")
        if checkpoint["identity"].get("guided_rollout_axis"):
            raise ValueError("Continuation of an anchored collector requires its original frozen behavior anchor")
        allowed.add("cloud/kaggle/guided.py")
    inventories = []
    for archive_path, expected in ((planned["runtime_path"], planned["runtime_sha"]), (bundle["path"], bundle["runtime_sha"])):
        archive = (ROOT / archive_path / "runtime.zip").resolve()
        if not archive.is_relative_to(WORK.resolve()):
            raise ValueError("Resume bundle path escapes campaign workspace")
        verify(archive, expected)
        with zipfile.ZipFile(archive) as runtime:
            files = json.loads(runtime.read("source_inventory.json"))["files"]
        inventories.append({name: digest for name, digest in files.items() if name not in allowed})
    policy_hooks = "cloud/kaggle/learning_hooks.py" in files
    if inventories[0] != inventories[1] or planned["source_sha"] != bundle["source_sha"]:
        raise RuntimeError("Full-state continuation requires the original learner, dependencies, and TRAIN schedule")
    job = {key: value for key, value in planned.items()
           if key not in {"warm_step", "warm_start_cursor", "parent_step", "policy_every", "guided_path", "guided_sha",
                          "utd_control", "actor_lr_control", "guided_curriculum_from", "traffic_guidance",
                          "traffic_teacher_sha", "traffic_teacher_job", "autonomous_traffic"}}
    job.update(job="s5-" + fingerprint([args.campaign, args.round])[:12] + "-j10-a1",
               round=args.round, role="easy_resume", account=args.account, transitions=args.transitions,
               start_transition=checkpoint["transitions"], parent_job=args.parent, parent_sha=state_sha,
               parent_archive_limit=LONG_PARENT_LIMIT, runtime_path=bundle["path"], runtime_sha=bundle["runtime_sha"],
               template_sha=bundle["template_sha"])
    if step is not None:
        job["parent_step"] = step
    if policy_hooks:
        job["policy_every"] = 2500
    for key in ("campaign_deadline", "bounded_guided_replay"):
        job.pop(key, None)
    if guided:
        live_rows = checkpoint["external_admitted"] + checkpoint["transitions"]
        parent_bounded = checkpoint["identity"].get("bounded_guided_replay", False)
        if parent_bounded and not deadline:
            raise ValueError("Bounded replay requires explicit timeboxed continuation")
        capacity = checkpoint["identity"]["capacity"] if parent_bounded else (
            max(checkpoint["identity"]["capacity"], 122880) if deadline else
            max(checkpoint["identity"]["capacity"], math.ceil((live_rows + args.transitions) / 4096) * 4096))
        if capacity * 3300 + 32_000_000 > LONG_PARENT_LIMIT:
            raise ValueError("Projected Guided replay exceeds the verified long-parent transport budget")
        job.update(guided_resume=True, capacity=capacity, max_seconds=seconds or 2400, export_reserve=480,
                   checkpoint_every=25000 if args.transitions >= 50000 else 10000)
        parent_phase = checkpoint["identity"].get("phase", planned.get("phase", "easy"))
        job["phase"] = parent_phase
        if traffic_guidance and ((parent_phase, next_phase) not in {("medium", None), ("easy", "medium")} or comparison_id):
            raise ValueError("Traffic guidance requires a verified Medium parent or an Easy-to-Medium transition")
        prior_traffic = checkpoint["identity"].get("traffic_guidance")
        if checkpoint["identity"].get("traffic_teacher_sha") and not (teacher or autonomous):
            raise ValueError("Choose the teacher again or retire traffic guidance explicitly")
        if autonomous:
            if not prior_traffic:
                raise ValueError("Only a traffic-guided parent can retire its guidance")
            job["autonomous_traffic"] = True
        elif traffic_guidance or prior_traffic or teacher:
            job["traffic_guidance"] = "traffic_follow_v1"
            if parent_phase == "medium" and next_phase:
                raise ValueError("A traffic-guided state requires same-phase continuation")
        if teacher:
            if parent_phase != "medium" or not prior_traffic or next_phase:
                raise ValueError("Teacher mixture requires a traffic-guided Medium parent")
            from cloud.kaggle.parallel import CHAMPION, MODEL, STATE
            teacher_path, teacher_state = parent_checkpoint(state, CHAMPION)
            if teacher_state != STATE:
                raise ValueError("Unexpected frozen teacher state")
            verify(teacher_path / "model.zip", MODEL)
            job.update(traffic_teacher_sha=MODEL, traffic_teacher_job=CHAMPION,
                       parallel_workload="traffic-teacher-medium")
        if traffic_guidance:
            job["parallel_workload"] = "traffic-follow-medium"
            job["guided_gradient_steps"] = 1
        if next_phase:
            if {"easy": "medium", "medium": "hard", "hard": "extreme"}.get(parent_phase) != next_phase:
                raise ValueError("Curriculum may advance only to the next approved phase")
            job.update(phase=next_phase, guided_curriculum_from=parent_phase)
        if device:
            job["device"] = device
        job.pop("accelerator", None)
        if job.get("device") == "cuda":
            job["accelerator"] = "NvidiaTeslaT4"
        if policy_hooks and args.transitions >= 25000:
            job["policy_every"] = 5000
        if not comparison_id and (device or seconds or next_phase or traffic_guidance or teacher or autonomous or args.transitions >= 50000):
            job["production_continuation"] = True
        if deadline:
            if not seconds or not policy_hooks:
                raise ValueError("Timeboxed training requires an explicit runtime and screening hooks")
            lr = getattr(args, "campaign_actor_lr", None)
            if lr not in (None, 0.25):
                raise ValueError("Unknown campaign actor update profile")
            if lr is not None:
                job["guided_actor_lr_scale"] = lr
            job.update(campaign_deadline=deadline, bounded_guided_replay=True, policy_every=50000,
                       guided_gradient_steps=1,
                       checkpoint_every=250000, export_reserve=900, production_continuation=True,
                       parallel_workload=f"window-{args.round}")
        if comparison_id:
            if execution["capacity"] < capacity:
                raise ValueError("The matched control capacity cannot retain the experiment replay")
            job.update(guided_gradient_steps=1, utd_control=compare_utd or execution["utd_control"],
                       capacity=execution["capacity"], workers=2, device="cpu")
            if actor_lr_control:
                job.update(actor_lr_control=actor_lr_control, guided_actor_lr_scale=0.25)
    require_gate(state, job)
    save_plan(f"{args.campaign}-{args.round}", {"j10": job}, {}, state)

def upload(state, alias, owner, label, name, source, digest, *, journal=None):
    verify(source, digest)
    audit_zip(source, label)
    # Kaggle expands archives uploaded with a .zip extension, even in private datasets.
    # Keep the exact bytes as an opaque file with a distinct immutable dataset slug.
    transport_name = name.removesuffix(".zip") + ".bin"
    slug = f"s5-{label}-{digest[:20]}-bin"
    ref = f"{owner}/{slug}"
    if state["uploads"].get(ref, {}).get("status") == "ACCEPTED":
        return dataset_ready(state, alias, ref, journal=journal)
    if state["uploads"].get(ref, {}).get("status") == "WINDOW_INPUT_STAGING":
        raise DatasetPending("Private inputs reserved for window staging")
    if ref in state["uploads"]:
        raise RuntimeError("Prior dataset upload is uncertain. Inspect the private dataset before any recovery.")
    staging = WORK / "uploads" / alias / slug
    staging.mkdir(parents=True, exist_ok=True)
    dest = staging / transport_name
    if dest.exists():
        verify(dest, digest)
    else:
        shutil.copyfile(source, dest)
    write(staging / "dataset-metadata.json", {"id": ref, "title": slug, "licenses": [{"name": "other"}]})
    if {p.name for p in staging.iterdir()} != {transport_name, "dataset-metadata.json"} or any(
        p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(staging.resolve())
        for p in staging.iterdir()
    ):
        raise ValueError("Upload staging must contain only the audited payload and generated metadata")
    state["uploads"][ref] = {"status": "UPLOADING", "sha": digest, "alias": alias,
                             "transport_name": transport_name, "at": now()}
    if label == "long-parent":
        state["uploads"][ref]["byte_limit"] = LONG_PARENT_LIMIT + 1_000_000
    persist(state) if journal is None else write(journal, state)
    code, output = cli(alias, ["datasets", "create", "-p", str(staging), "-q", "-t"],
                       timeout=600 if label == "long-parent" else 240)
    accepted = code == 0 and f"/datasets/{ref}" in output
    state["uploads"][ref].update(status="ACCEPTED" if accepted else "UPLOAD_UNKNOWN",
                                 at=now(), cli_exit=code)
    persist(state) if journal is None else write(journal, state)
    if not accepted:
        raise RuntimeError("Dataset creation acceptance is uncertain. See registry; automatic retry is disabled.")
    return dataset_ready(state, alias, ref, journal=journal)

def dataset_ready(state, alias, ref, *, journal=None):
    code, output = cli(alias, ["datasets", "status", ref, "--format", "json"])
    response = dataset_status_response(code, output)
    if response.get("status") != "ready" or response.get("current_version_number") != 1:
        raise DatasetPending("Private input version 1 is not yet verified ready.")
    if not private_input(alias, ref):
        raise RuntimeError("Dataset API did not verify private ACL for version 1")
    row = state["uploads"][ref]
    if row.get("transport_name") and not row.get("remote_sha_verified_at"):
        if row.get("byte_limit"):
            verify_remote_input(alias, ref, row["transport_name"], row["sha"], byte_limit=row["byte_limit"])
        else:
            verify_remote_input(alias, ref, row["transport_name"], row["sha"])
        row["remote_sha_verified_at"] = now()
    row.update(version=1, private_verified_at=now())
    persist(state) if journal is None else write(journal, state)
    return ref

def verify_remote_input(alias, ref, name, digest, *, byte_limit=120_000_000):
    from tempfile import TemporaryDirectory
    from hashlib import sha256

    with TemporaryDirectory(prefix="stage5-kaggle-download-") as directory:
        code, _ = cli(alias, ["datasets", "download", ref + "/1", "-p", directory, "-q"],
                      timeout=600 if byte_limit > 120_000_000 else 90)
        files = list(Path(directory).iterdir())
        if code or len(files) != 1 or not files[0].is_file() or files[0].stat().st_size > byte_limit + 5_000_000:
            raise RuntimeError("Private version 1 could not be retrieved for byte verification")
        with zipfile.ZipFile(files[0]) as archive:
            if archive.namelist() != [name] or archive.getinfo(name).file_size > byte_limit:
                raise RuntimeError("Private dataset layout differs from the exact transport file")
            digest_remote = sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest_remote.update(block)
            if digest_remote.hexdigest() != digest:
                raise RuntimeError("Private dataset content hash mismatch")

def dataset_status_response(code, output):
    if code:
        return {}
    # Installed CLI can prepend an SDK message even when JSON output was requested.
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if not line.lstrip().startswith("{"):
            continue
        try:
            result = json.loads("\n".join(lines[index:]))
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and set(result) == {"status", "current_version_number"}:
            return result
    return {}


def publish_initial_snapshot(args, state):
    snapshot = args.snapshot.resolve(strict=True)
    if not snapshot.is_relative_to(WORK.resolve()):
        raise ValueError("B1 snapshot must be stored in the campaign workspace")
    marker = snapshot / "SNAPSHOT.json"
    verify(marker, args.snapshot_sha)
    meta = read(marker)
    identity = meta["identity"]
    if (meta["schema"], meta["lineage"], meta["policy_version"], meta["policy_sha"], identity["phase"],
            identity["warm_source"]) != (1, RECIPE["lineage"], 0, SOURCE_SHA, "smoke", SOURCE_SHA):
        raise ValueError("Expected the registered initial B1 policy, lineage and smoke recipe")
    verify(snapshot / "policy.zip", SOURCE_SHA)
    runtime = read(WORK / "bundle.json")
    bundle = ROOT / runtime["path"] / "runtime.zip"
    verify(bundle, runtime["runtime_sha"])
    with zipfile.ZipFile(bundle) as payload:
        runtime_files = json.loads(payload.read("source_inventory.json"))["files"]
    if identity["recipe"] != RECIPE or not identity["files"] or any(
        runtime_files.get(name) != digest for name, digest in identity["files"].items()
    ):
        raise ValueError("Snapshot learner recipe differs from the packaged runtime")
    target = WORK / "b1" / args.snapshot_sha
    target.mkdir(parents=True, exist_ok=True)
    parcel = target / "snapshot.zip"
    contents = {name: (snapshot / name).read_bytes() for name in ("policy.zip", "SNAPSHOT.json")}
    if not parcel.exists():
        archive(parcel, contents)
    with zipfile.ZipFile(parcel) as stored:
        if set(stored.namelist()) != set(contents) or any(stored.read(name) != data for name, data in contents.items()):
            raise ValueError("Existing B1 snapshot archive differs from verified source")
    alias, creds = account_for({"account": operator_slot()})
    digest = sha(parcel)
    ref = upload(state, alias, creds["owner"], "snapshot", "snapshot.zip", parcel, digest)
    receipt = {"dataset": ref, "version": 1, "snapshot_sha256": args.snapshot_sha, "archive_sha256": digest,
               "policy_version": 0, "learner_workers": identity["workers"], "runtime_sha": runtime["runtime_sha"]}
    write(target / "receipt.json", receipt)
    print(receipt)


def publish_verified_replay_chunk(args, state):
    chunk = args.chunk.resolve(strict=True)
    learner = args.learner.resolve(strict=True)
    if any(not path.is_relative_to(WORK.resolve()) for path in (chunk, learner)):
        raise ValueError("B1 artifacts must be stored in the campaign workspace")
    verify(chunk / "CHUNK.json", args.chunk_sha)
    verify(learner / "COMPLETE.json", args.learner_sha)
    meta = read(chunk / "CHUNK.json")
    trained = read(learner / "COMPLETE.json")
    if not re.fullmatch(r"[0-9a-f]{64}", meta["snapshot_sha"]):
        raise ValueError("Malformed B1 snapshot digest")
    key = f"{meta['stream']}:{meta['sequence']}"
    if (trained["job"] != "external-import" or trained["ledger"].get(key) != args.chunk_sha
            or meta["identity"] != trained["identity"] or trained["external_admitted"] < meta["count"]):
        raise ValueError("Replay chunk has no matching verified learner admission")
    verify(learner / "metrics.json", trained["files"]["metrics.json"])
    metrics = read(learner / "metrics.json")
    if metrics["snapshot_sha"] != meta["snapshot_sha"] or metrics["admitted"] != meta["count"] or metrics["updates"] <= 0:
        raise ValueError("Learner metrics do not attest the replay snapshot, accounting and updates")
    snapshot = read(WORK / "b1" / meta["snapshot_sha"] / "receipt.json")
    source = state["uploads"][snapshot["dataset"]]
    if (not source.get("private_verified_at") or not source.get("remote_sha_verified_at")
            or snapshot["snapshot_sha256"] != meta["snapshot_sha"]
            or snapshot["policy_version"] != meta["policy_version"]):
        raise ValueError("B1 snapshot version or private transfer is unverified")
    verify(chunk / "rows.npz", meta["payload_sha"])
    if (chunk / "rows.npz").stat().st_size > 20_000_000:
        raise ValueError("Replay chunk exceeds the verified B1 transport budget")
    target = WORK / "b1" / args.chunk_sha
    target.mkdir(parents=True, exist_ok=True)
    parcel = target / "replay.zip"
    contents = {name: (chunk / name).read_bytes() for name in ("CHUNK.json", "rows.npz")}
    if not parcel.exists():
        archive(parcel, contents)
    with zipfile.ZipFile(parcel) as stored:
        if stored.namelist() != sorted(contents) or any(stored.read(name) != data for name, data in contents.items()):
            raise ValueError("Existing replay archive differs from verified chunk")
    alias, creds = account_for({"account": operator_slot()})
    digest = sha(parcel)
    ref = upload(state, alias, creds["owner"], "replay", "replay.zip", parcel, digest)
    receipt = {"dataset": ref, "version": 1, "chunk_sha256": args.chunk_sha, "archive_sha256": digest,
               "snapshot_sha256": meta["snapshot_sha"], "policy_version": meta["policy_version"],
               "admitted_rows": metrics["admitted"], "learner_sha256": args.learner_sha}
    write(target / "receipt.json", receipt)
    print(receipt)

def planned_job(args, state):
    path = WORK / f"plan-{args.campaign}-{args.round}.json"
    identity_part(args.campaign)
    identity_part(args.round)
    plan_doc = read(path)
    body = {k: plan_doc[k] for k in ("jobs", "deferred")}
    if fingerprint(body) != args.fingerprint or plan_doc["fingerprint"] != args.fingerprint:
        raise ValueError("Plan fingerprint mismatch")
    job = body["jobs"][args.job]
    entry = state["jobs"][job["job"]]
    if entry["planned"] != job:
        raise ValueError("Registry plan differs from approved plan")
    return job, entry

def account_for(job):
    alias = job["account"]
    _, creds = slot(alias)
    if not creds or not verify_owner(alias):
        raise RuntimeError("READY_FOR_CREDENTIALS: account identity unverified")
    return alias, creds

def require_gate(state, job):
    if job["role"] == "parity":
        return
    validated_runtime = job.get("validated_runtime_sha", job["runtime_sha"])
    if not any(entry["status"] == "HARVESTED_VERIFIED" and entry.get("gates", {}).get("G2") == "PASS"
               and entry["execution"]["campaign"] == job["campaign"]
               and entry["execution"]["runtime_sha"] == validated_runtime for entry in state["jobs"].values()):
        raise RuntimeError("G2 requires a verified J01 cloud artifact for this campaign and runtime")
    if job["role"] in {"easy_pilot", "easy_long", "easy_resume", "easy_guided"} and not any(
        entry["status"] == "HARVESTED_VERIFIED" and entry.get("gates", {}).get("G3") == "PASS"
        and entry["execution"]["campaign"] == job["campaign"]
        and entry["execution"]["runtime_sha"] == validated_runtime for entry in state["jobs"].values()
    ):
        raise RuntimeError("Easy pilot requires verified cloud resume")
    if job.get("parent_job"):
        _, digest = parent_checkpoint(state, job["parent_job"], job.get("parent_step"))
        if digest != job["parent_sha"]:
            raise ValueError("Resume parent is not a matching harvested state")

def inputs_for(job, state, alias, owner):
    runtime_dir = (ROOT / job["runtime_path"]).resolve()
    if not runtime_dir.is_relative_to(WORK.resolve()):
        raise ValueError("Plan runtime path is outside campaign workspace")
    inputs = {}
    for label, name, key in (("runtime", "runtime.zip", "runtime_sha"), ("source", "model.zip", "source_sha")):
        try:
            inputs[label + "_ref"] = upload(state, alias, owner, label, name, runtime_dir / name, job[key])
        except DatasetPending:
            continue
    if len(inputs) != 2:
        raise DatasetPending("Dataset processing pending; repeat preparation to check version 1")
    return inputs

def kernel_state(code, output):
    match = re.search(r'has status "\s*(?:KernelWorkerStatus\.)?([a-z_]+)\s*"', output, re.I)
    if code or not match:
        return "STATUS_UNKNOWN"
    label = KERNEL_STATES.get(match[1].lower(), "STATUS_UNKNOWN")
    if label == "FAILED" and CAPACITY_ERROR.search(output):
        return "CAPACITY_REJECTED"
    return label

def probe_inventory(alias, owner, target, known_active=()):
    # Kaggle lists saved notebooks, and its status endpoint describes their latest run.
    # Interactive sessions and available CPU slots are not exposed by these calls.
    arguments = ["kernels", "list", "--mine", "--page-size", "100", "--sort-by", "dateRun", "--format", "json"]
    rows, tokens = [], set()
    page = arguments
    while True:
        code, output = cli(alias, page, timeout=25)
        match = re.search(r"^\[", output, re.M)
        if code or (not match and "Not found" not in output.splitlines()):
            raise RuntimeError("Kaggle notebook inventory unavailable")
        if match:
            result = json.loads(output[match.start():])
            if not isinstance(result, list):
                raise ValueError("Unexpected Kaggle notebook inventory")
            rows.extend(result)
        token = re.search(r"^Next Page Token = (\S+)\s*$", output, re.M)
        if not token:
            break
        if token[1] in tokens:
            raise RuntimeError("Kaggle notebook pagination repeated a token")
        tokens.add(token[1])
        page = [*arguments, "--page-token", token[1]]
    owned = [row for row in rows if re.fullmatch(re.escape(owner) + r"/[a-z0-9_-]+", row["ref"], re.I)]
    refs = [row["ref"].lower() for row in owned]
    if target in refs:
        raise RuntimeError("Target notebook already exists on Kaggle and requires reconciliation")
    # Direct GetKernel can return 403 for an unused slug. Search the authenticated
    # owner's inventory by the exact slug to cover the server's listing limit.
    query = ["kernels", "list", "--mine", "--search", target.split("/")[1], "--format", "json"]
    code, output = cli(alias, query, timeout=25)
    if code or output.splitlines()[-1:] != ["Not found"]:
        raise RuntimeError("Kaggle target slug search requires reconciliation")
    cutoff = (datetime.now(timezone.utc) - timedelta(days=1)).replace(tzinfo=None).isoformat()
    recent = dict.fromkeys(
        row["ref"] for index, row in enumerate(owned)
        if index < 5 or (row.get("lastRunTime") or "") >= cutoff
    )
    no_visible_session = 0
    for ref in recent:
        code, output = cli(alias, ["kernels", "status", ref], timeout=25, include_stderr=True)
        label = kernel_state(code, output)
        if label in {"QUEUED", "RUNNING"} and ref not in known_active:
            raise RuntimeError("A recent Kaggle notebook is queued or running")
        no_visible_session += label == "STATUS_UNKNOWN"
    return {"notebooks_listed": len(refs), "unaddressable_or_unowned": len(rows) - len(owned),
            "latest_runs_checked": len(recent), "latest_run_status_unknown": no_visible_session}

def push_result(code, output, ref):
    match = re.search(
        r"^Kernel version ([1-9]\d*) successfully pushed\.\s+Please check progress at "
        r"https://www\.kaggle\.com/code/" + re.escape(ref) + r"(?:[/?#][^\s]*)?\s*$",
        output, re.I | re.M,
    )
    if code == 0 and match:
        return "SUBMITTED", int(match[1])
    rejection = re.search(r"^(?:Kernel push error:|(?:400|403|409|429) Client Error:).*$", output, re.I | re.M)
    if rejection and CAPACITY_ERROR.search(rejection[0]):
        return "CAPACITY_REJECTED", None
    return "SUBMIT_UNKNOWN", None

def probe_retry(state, job, retry_of, owner):
    prior = {key: row for key, row in state["jobs"].items()
             if row.get("admission_probe") and row["planned"]["campaign"] == job["campaign"]}
    if not prior and retry_of is None:
        return {}
    if retry_of not in prior:
        raise RuntimeError("A prior admission probe requires an explicit retry of its failed attempt")
    latest = max(prior, key=lambda key: prior[key].get("submitted", {}).get("utc", ""))
    previous = prior[retry_of]
    if retry_of != latest or previous["status"] != "FAILED" or not previous.get("version"):
        raise RuntimeError("Retry requires the latest admission probe to have a confirmed failed version")
    if previous.get("owner") != owner or previous["planned"]["account"] != job["account"]:
        raise RuntimeError("Admission retry must use the account of the failed attempt")
    if job["runtime_sha"] in {row["planned"]["runtime_sha"] for row in prior.values()}:
        raise RuntimeError("Admission retry requires a new runtime fingerprint")
    categories = previous.get("log_diagnostic", {}).get("categories", [])
    repair_files = set().union(*(REPAIR_FILES.get(category, set()) for category in categories))
    if not repair_files or any(category not in REPAIR_FILES for category in categories):
        raise RuntimeError("Admission retry requires a diagnosed and repaired runtime failure")
    inventories = []
    for attempt in (previous["planned"], job):
        path = (ROOT / attempt["runtime_path"] / "runtime.zip").resolve()
        if not path.is_relative_to(WORK.resolve()):
            raise ValueError("Plan runtime path is outside campaign workspace")
        verify(path, attempt["runtime_sha"])
        with zipfile.ZipFile(path) as bundle:
            inventories.append(json.loads(bundle.read("source_inventory.json"))["files"])
    old, new = inventories
    changed = sorted(name for name in repair_files if name in old and name in new and old[name] != new[name])
    if not changed or any(not REPAIR_FILES[category].intersection(changed) for category in categories):
        raise RuntimeError("Runtime fingerprint does not contain a change for the diagnosed failure")
    alias = job["account"]
    ref = previous["ref"]
    if current_private_kernel_version(alias, ref) != previous["version"]:
        raise RuntimeError("Kaggle could not confirm the failed notebook version")
    code, output = cli(alias, ["kernels", "status", f"{ref}/{previous['version']}"], timeout=25, include_stderr=True)
    if kernel_state(code, output) != "FAILED":
        raise RuntimeError("Kaggle no longer confirms the previous attempt as terminal FAILED")
    return {"retry_of": retry_of, "previous_version": previous["version"], "previous_terminal_at": now(),
            "failure_categories": categories, "changed_runtime_files": changed}

def launch(args, state):
    job, entry = planned_job(args, state)
    if entry["status"] != "PLANNED":
        print(f"{job['job']} {entry['status']}")
        return
    require_gate(state, job)
    if any(other_id != job["job"] and other["status"] in ACTIVE
           and other["planned"]["campaign"] == job["campaign"]
           and other["planned"]["role"] == job["role"]
           and other["planned"]["account"] == job["account"]
           and other["planned"].get("parallel_workload") == job.get("parallel_workload")
           and other["planned"]["runtime_sha"] == job["runtime_sha"]
           and other["planned"].get("parent_sha") == job.get("parent_sha")
           for other_id, other in state["jobs"].items()):
        raise RuntimeError("Matching campaign work is active or awaits submission reconciliation")
    alias, creds = account_for(job)
    owner = creds["owner"]
    prior_continuations = {key: row for key, row in state["jobs"].items() if job.get("parent_sha")
                           and row.get("execution", {}).get("parent_sha") == job["parent_sha"]}
    controls = {job[key] for key in ("utd_control", "actor_lr_control") if key in job}
    if job.get("parallel_workload"):
        if any(row["status"] in {"SUBMITTING", "SUBMIT_UNKNOWN", "STATUS_UNKNOWN"}
               or row["planned"].get("parallel_workload") == job["parallel_workload"]
               for row in prior_continuations.values() if row["planned"]["account"] == job["account"]):
            raise RuntimeError("A matching parallel workload or unresolved parent receipt already exists")
    elif prior_continuations and not (job.get("guided_gradient_steps") == 1 and prior_continuations.keys() == controls
                                    and all(row["status"] == "HARVESTED_VERIFIED" for row in prior_continuations.values())):
        raise RuntimeError("This parent already has a submitted or uncertain continuation")
    age = datetime.now(timezone.utc) - datetime.fromisoformat(creds["observed_at"])
    sessions = creds["concurrency_observed"] if 0 <= age.total_seconds() <= 86400 else None
    active = [x for x in state["jobs"].values() if x.get("owner") == owner and x["status"] in ACTIVE]
    admission_probe = getattr(args, "admission_probe", False)
    server_admission = getattr(args, "server_admission", False)
    retry_of = getattr(args, "retry_of", None)
    if retry_of and not admission_probe:
        raise ValueError("--retry-of requires --admission-probe")
    repair = {}
    if admission_probe:
        if (sessions is not None or args.job != "j01" or job["role"] != "parity" or job["device"] != "cpu"
                or job["workers"] != 1 or job.get("parent_job") or job["max_seconds"] != RECIPE["preflight"]["max_seconds"]):
            raise RuntimeError("Admission probe requires unknown capacity and the planned J01 CPU preflight")
        if active:
            raise RuntimeError("An active receipt blocks this submission")
        repair = probe_retry(state, job, retry_of, owner)
    elif server_admission:
        roles = {"j02": "cpu1", "j03": "cpu2", "j04": "gpu2", "j07": "resume", "j08": "easy_pilot",
                 "j09": "easy_long", "j10": "easy_resume", "j11": "easy_guided"}
        budget = 2400 if job["role"] in {"easy_long", "easy_resume"} else RECIPE["preflight"]["max_seconds"]
        production = (job.get("production_continuation") and job.get("guided_resume") and job.get("parent_sha")
                      and 1800 <= job["max_seconds"] <= (43200 if job.get("campaign_deadline") else 14400))
        if (sessions is not None or roles.get(args.job) != job["role"]
                or (job["max_seconds"] != budget and not production)):
            raise RuntimeError("Server admission requires unknown capacity and a gated Stage 5 follow-up preflight")
        known_active = ()
        if job.get("campaign_deadline") and len(active) == 1:
            sibling = active[0]
            if (sibling["status"] in {"QUEUED", "RUNNING"} and sibling.get("version")
                    and sibling["planned"].get("campaign_deadline") == job["campaign_deadline"]
                    and sibling["planned"]["round"] != job["round"]
                    and sibling["planned"]["round"].rsplit("-", 1)[0] == job["round"].rsplit("-", 1)[0]):
                if current_private_kernel_version(alias, sibling["ref"]) == sibling["version"]:
                    known_active = (sibling["ref"],)
        if (active and not known_active) or any(
            row["planned"]["campaign"] == job["campaign"]
            and ((row.get("submitted") and row["planned"]["role"] == job["role"]
                  and row["planned"]["account"] == job["account"]
                  and row["planned"].get("parallel_workload") == job.get("parallel_workload")
                  and (not production or row["planned"].get("parent_sha") == job["parent_sha"])
                  and row["planned"]["runtime_sha"] == job["runtime_sha"])
                 or (row.get("owner") == owner and row["status"] == "CAPACITY_REJECTED"))
            for row in state["jobs"].values()
        ):
            raise RuntimeError("An active receipt or prior follow-up admission blocks this submission")
    elif sessions is None:
        raise RuntimeError("Fresh owner observation of available concurrency is required")
    elif len(active) >= sessions:
        raise RuntimeError("Account concurrency budget is occupied")
    observed_gpu_jobs = [x for x in state["jobs"].values() if x.get("owner") == owner
                         and x.get("execution", {}).get("device") == "cuda"
                         and x.get("submitted", {}).get("utc", "") >= creds["observed_at"]]
    gpu_minutes = sum(x["execution"]["max_seconds"] / 60 for x in observed_gpu_jobs)
    if job["device"] == "cuda":
        available = accelerator_quota(alias).get("GPU")
        owner_limit = creds["gpu_minutes_observed"]
        if available is not None and owner_limit is not None:
            available = min(available, owner_limit)
        if available is None or gpu_minutes + job["max_seconds"] / 60 > available:
            raise RuntimeError("Verified GPU budget is insufficient or unavailable")
    runtime = (ROOT / job["runtime_path"] / "runtime.zip").resolve()
    if not runtime.is_relative_to(WORK.resolve()):
        raise ValueError("Plan runtime path is outside campaign workspace")
    verify(runtime, job["runtime_sha"])
    with zipfile.ZipFile(runtime) as bundle:
        template = bundle.read("cloud/kaggle/kernel_template.py.txt")
    if hashlib.sha256(template).hexdigest() != job["template_sha"]:
        raise ValueError("Kernel template differs from the pinned runtime")
    actual_job = {k: v for k, v in job.items() if k not in ("runtime_path", "account", "guided_path")}
    actual_job.update(inputs_for(job, state, alias, owner))
    if job.get("guided_sha"):
        archive = (ROOT / job["guided_path"]).resolve()
        if not archive.is_relative_to(WORK.resolve()):
            raise ValueError("Guided replay escapes campaign workspace")
        actual_job["guided_ref"] = upload(state, alias, owner, "guided", "guided.zip", archive, job["guided_sha"])
    if job.get("warm_step"):
        path, manifest = pilot_checkpoint(state, job["reference_job"], job["warm_step"])
        if manifest["files"]["model.zip"] != job["warm_model_sha"]:
            raise ValueError("Warm-start checkpoint differs from the approved plan")
        actual_job["warm_ref"] = upload(state, alias, owner, "warm", "warm.zip", path / "model.zip", job["warm_model_sha"])
    if job.get("traffic_teacher_sha"):
        teacher, _ = parent_checkpoint(state, job["traffic_teacher_job"])
        actual_job["teacher_ref"] = upload(state, alias, owner, "teacher", "teacher.zip",
                                          teacher / "model.zip", job["traffic_teacher_sha"])
    if job.get("parent_job"):
        parent, _ = parent_checkpoint(state, job["parent_job"], job.get("parent_step"))
        verify(parent / "COMPLETE.json", job["parent_sha"])
        parent_archive = parent / "parent.zip"
        manifest = read(parent / "COMPLETE.json")
        if set(manifest["files"]) != STATE_FILES - {"COMPLETE.json"}:
            raise ValueError("Unexpected parent checkpoint members")
        for name, expected in manifest["files"].items():
            verify(parent / name, expected)
        if not parent_archive.exists():
            with zipfile.ZipFile(parent_archive.with_suffix(".tmp"), "x") as output:
                for name in STATE_FILES:
                    output.write(parent / name, name)
            parent_archive.with_suffix(".tmp").rename(parent_archive)
        with zipfile.ZipFile(parent_archive) as archived:
            if len(archived.namelist()) != len(STATE_FILES) or set(archived.namelist()) != STATE_FILES:
                raise ValueError("Parent archive differs from verified state")
            for name, expected in {**manifest["files"], "COMPLETE.json": job["parent_sha"]}.items():
                with archived.open(name) as stream:
                    digest = hashlib.sha256()
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != expected:
                    raise ValueError("Parent archive differs from verified state")
        actual_job["parent_zip_sha"] = sha(parent_archive)
        label = "long-parent" if job.get("parent_archive_limit") == LONG_PARENT_LIMIT else "parent"
        actual_job["parent_ref"] = upload(state, alias, owner, label, "parent.zip",
                                         parent_archive, actual_job["parent_zip_sha"])
    staging = WORK / "kernels" / job["job"]
    staging.mkdir(parents=True, exist_ok=True)
    template_text = template.decode("utf-8")
    refs = [actual_job["runtime_ref"], actual_job["source_ref"]]
    if actual_job.get("guided_ref"):
        refs.append(actual_job["guided_ref"])
    if actual_job.get("warm_ref"):
        refs.append(actual_job["warm_ref"])
    if actual_job.get("parent_ref"):
        refs.append(actual_job["parent_ref"])
    if actual_job.get("teacher_ref"):
        refs.append(actual_job["teacher_ref"])
    actual_job["input_versions"] = {key: state["uploads"].get(key, {}).get("version", 1) for key in refs}
    ref = f"{owner}/{job['job']}"
    (staging / "run.py").write_text(template_text.replace("__JOB__", repr(actual_job)), encoding="utf-8")
    write(staging / "kernel-metadata.json", {"id": ref, "title": job["job"], "code_file": "run.py",
          "language": "python", "kernel_type": "script", "is_private": True, "enable_internet": True,
          "enable_gpu": job["device"] == "cuda", "dataset_sources": refs,
          "competition_sources": [], "kernel_sources": [], "model_sources": []})
    if {p.name for p in staging.iterdir()} != {"run.py", "kernel-metadata.json"} or any(
        p.is_symlink() or SECRET.search(p.read_bytes()) for p in staging.iterdir()
    ):
        raise ValueError("Kernel upload staging failed the allowlist or secret audit")
    push = ["kernels", "push", "-p", str(staging), "-t", str(job["max_seconds"])]
    if job.get("accelerator"):
        push.extend(["--accelerator", job["accelerator"]])
    if admission_probe:
        entry["admission_probe"] = {"capacity": "UNKNOWN", **repair, **probe_inventory(alias, owner, ref)}
    if server_admission:
        entry["server_admission"] = {"capacity": "UNKNOWN", "reconciled_active": len(known_active),
                                     **(probe_inventory(alias, owner, ref, known_active) if known_active
                                        else probe_inventory(alias, owner, ref))}
    if job.get("campaign_deadline") and job["campaign_deadline"] - datetime.now(timezone.utc).timestamp() < 1800:
        raise RuntimeError("Campaign deadline leaves less than 30 minutes before submission")
    entry.update(status="SUBMITTING", owner=owner, ref=ref, execution=actual_job, submitted=now())
    persist(state)
    try:
        code, output = cli(alias, push, timeout=90, include_stderr=True)
    except (OSError, RuntimeError):
        code, output = -1, ""
    outcome, version = push_result(code, output, ref)
    entry.update(status=outcome, version=version, cli_exit=code, at=now())
    if version is not None:
        entry["accepted_at"] = entry["at"]
    persist(state)
    print(job["job"], entry["status"], entry["version"])

def status(state, job):
    entry = state["jobs"][job]
    if not entry.get("version") or entry["status"] == "HARVESTED_VERIFIED":
        print(job, entry["status"])
        return
    alias = entry["planned"]["account"]
    if not verify_owner(alias):
        raise RuntimeError("Account identity unverified")
    code, output = cli(alias, ["kernels", "status", f"{entry['ref']}/{entry['version']}"])
    entry.update(status=kernel_state(code, output), at=now())
    persist(state)
    print(job, entry["status"])

def diagnose(state, job):
    entry = state["jobs"][job]
    if not entry.get("version"):
        raise ValueError("A confirmed kernel version is required to inspect logs")
    alias = entry["planned"]["account"]
    if not verify_owner(alias):
        raise RuntimeError("Account identity unverified")
    if current_private_kernel_version(alias, entry["ref"]) != entry["version"]:
        raise RuntimeError("Kaggle log CLI reports the latest run and its version differs from the receipt")
    code, output = cli(alias, ["kernels", "logs", f"{entry['ref']}/{entry['version']}"], timeout=30)
    patterns = {
        "DEPENDENCY_RESOLUTION": r"No matching distribution|ResolutionImpossible|Could not find a version",
        "PYTHON_ABI": r"(?:Unvalidated|Unsupported) Kaggle Python ABI",
        "RUNTIME_IMPORT": r"ModuleNotFoundError|ImportError",
        "NATIVE_RUNTIME_CRASH": r"SIGSEGV|Fatal Python error: Segmentation fault",
        "METADRIVE_ASSETS": r"Assets folder does not exist|Assets version misses|pull_asset.*returned non-zero",
        "PRIVATE_INPUT": r"Missing, ambiguous or mismatched private input",
        "CLOUD_PARITY": r"Cloud parity drift|GEOMETRY DRIFT",
        "TIMEOUT": r"TimeoutExpired|Preflight time budget exhausted",
        "DISK_CAPACITY": r"No space left on device",
    }
    labels = [label for label, pattern in patterns.items() if code == 0 and re.search(pattern, output, re.I)]
    entry["log_diagnostic"] = {"at": now(), "exit": code, "categories": labels}
    persist(state)
    print(job, "LOG_CATEGORIES", labels)

def inspect_artifact(source, out, expected):
    if source.stat().st_size > 120_000_000:
        raise ValueError("Artifact ZIP too large")
    with zipfile.ZipFile(source) as bundle:
        names = bundle.namelist()
        if len(names) != len(OUTPUT_FILES) or set(names) != OUTPUT_FILES:
            raise ValueError("Unexpected artifact members")
        if sum(row.file_size for row in bundle.infolist()) > 120_000_000:
            raise ValueError("Expanded artifact too large")
        if json.loads(bundle.read("execution.json")) != expected:
            raise ValueError("Wrong campaign, attempt or input identity")
        manifest = json.loads(bundle.read("COMPLETE.json"))
        out.mkdir(parents=True)
        for name in names:
            with bundle.open(name) as stream, (out / name).open("xb") as target:
                shutil.copyfileobj(stream, target)
    return verify_state_artifact(manifest, out, expected)


def verify_state_artifact(manifest, out, expected):
    if (manifest["job"] != expected["job"] or manifest["parent_sha"] != expected.get("parent_sha")
            or set(manifest["files"]) != STATE_FILES - {"COMPLETE.json"}
            or manifest["identity"]["workers"] != expected["workers"]
            or manifest["identity"]["warm_source"] != expected.get("warm_model_sha", SOURCE_SHA)):
        raise ValueError("Wrong checkpoint identity")
    for name, digest in manifest["files"].items():
        verify(out / name, digest)
    metrics = read(out / "metrics.json")
    fingerprint(metrics)
    if manifest["result"] == "COMPLETE":
        transitions = metrics["transitions"]
        valid_count = transitions == expected["transitions"]
        if expected.get("campaign_deadline"):
            valid_count = (0 < transitions <= expected["transitions"] and transitions % expected["workers"] == 0
                           and metrics["termination_budget"] == ("transitions" if valid_count else "time"))
        if not valid_count or metrics["updates"] <= 0:
            raise ValueError("Complete artifact does not show the planned collection and updates")
        if manifest["files"]["model.zip"] == SOURCE_SHA:
            raise ValueError("The historical warm-start model cannot be a resume checkpoint")
        with zipfile.ZipFile(out / "model.zip") as model:
            if model.getinfo("data").file_size > 4_000_000:
                raise ValueError("SAC checkpoint metadata exceeds the limit")
            metadata = json.loads(model.read("data"))
        if (metadata.get("num_timesteps"), metadata.get("_n_updates")) != (manifest["transitions"], manifest["updates"]):
            raise ValueError("Saved SAC counters differ from the checkpoint manifest")
    return manifest


def fetch_output(entry, prefix, local, filename):
    alias = entry["planned"]["account"]
    if not download_kernel_output(alias, entry["ref"], entry["version"], local / filename, f"{prefix}/{filename}"):
        raise RuntimeError("Expected version-specific Kaggle output unavailable")


def fetch_state_shards(entry, prefix, local, restored, index_sha):
    local.mkdir(parents=True, exist_ok=True)
    fetch_output(entry, prefix, local, "INDEX.json")
    verify(local / "INDEX.json", index_sha)
    index = read(local / "INDEX.json")
    if index["schema"] != 1 or set(index["files"]) != STATE_FILES:
        raise ValueError("Unexpected sharded checkpoint index")
    for filename, shards in index["files"].items():
        for ordinal in range(len(shards)):
            fetch_output(entry, prefix, local, f"{filename}.{ordinal:06d}.bin")
    return restore_checkpoint(local, index_sha, restored)


def harvest_sharded(entry, staging, temp):
    expected = entry["execution"]
    job = expected["job"]
    fetch_output(entry, job, staging, "execution.json")
    envelope = read(staging / "execution.json")
    if envelope["job"] != expected:
        raise ValueError("Wrong campaign, attempt or input identity for sharded output")
    index_sha = envelope["index_sha256"]
    fetch_state_shards(entry, f"{job}/final", staging / "final-package", temp, index_sha)
    fetch_output(entry, job, temp, "runtime_versions.txt")
    fetch_output(entry, job, temp, "parity.json")
    fetch_output(entry, job, temp, "timing.json")
    shutil.copyfile(staging / "execution.json", temp / "execution.json")
    if read(temp / "timing.json")["evaluation_status"] == "COMPLETE":
        fetch_output(entry, job, temp, "panel.json")
        for tier in RECIPE["phase_weights"].get(expected.get("phase"), {}):
            if tier != "Easy":
                fetch_output(entry, job, temp, f"panel-{tier.lower()}.json")
    return verify_state_artifact(read(temp / "COMPLETE.json"), temp, expected), index_sha

def harvest(state, job):
    entry = state["jobs"][job]
    if entry["status"] in {"HARVESTED_VERIFIED", "PARTIAL"}:
        print(job, entry["status"])
        return
    status(state, job)
    if entry["status"] != "CLOUD_SUCCEEDED":
        return
    downloads = WORK / "downloads" / job
    downloads.mkdir(parents=True, exist_ok=True)
    staging = Path(mkdtemp(prefix=f"v{entry['version']}-", dir=downloads))
    alias = entry["planned"]["account"]
    filename = job + ".zip"
    final = WORK / "harvest" / job
    final.parent.mkdir(parents=True, exist_ok=True)
    temp = staging / "verified-state"
    try:
        expected = entry["execution"]
        if expected.get("sharded_output"):
            manifest, output_sha = harvest_sharded(entry, staging, temp)
        else:
            if not download_kernel_output(alias, entry["ref"], entry["version"], staging / filename):
                raise RuntimeError("Expected version-specific Kaggle output unavailable")
            manifest = inspect_artifact(staging / filename, temp, expected)
            output_sha = sha(staging / filename)
        gates = verify_cloud_evidence(temp, manifest, entry, state) if manifest["result"] == "COMPLETE" else {}
        temp.rename(final)
    except (ValueError, KeyError, zipfile.BadZipFile):
        entry.update(status="ARTIFACT_INVALID", at=now())
        persist(state)
        raise
    entry.update(status="HARVESTED_VERIFIED" if manifest["result"] == "COMPLETE" else "PARTIAL",
                 state_sha=sha(final / "COMPLETE.json"), output_sha=output_sha,
                 output_format="shards" if expected.get("sharded_output") else "zip",
                 output_version=entry["version"], gates=gates, at=now())
    persist(state)
    print(job, entry["status"], entry["state_sha"])


def checkpoint_log_digest(entry, pattern):
    alias, _ = account_for(entry["planned"])
    version = entry["version"]
    if current_private_kernel_version(alias, entry["ref"]) != version:
        raise RuntimeError("Kaggle log version differs from the checkpoint receipt")
    code, output = cli(alias, ["kernels", "logs", f"{entry['ref']}/{version}"], timeout=30)
    hashes = set(re.findall(pattern, output))
    if code or len(hashes) != 1:
        raise RuntimeError("Expected checkpoint hash is unavailable in the cloud log")
    return hashes.pop()


def harvest_periodic(state, job, step):
    entry = state["jobs"][job]
    planned = entry["planned"]
    interval = planned.get("checkpoint_every")
    start = planned.get("start_transition", 0)
    if not interval or not start < step < start + planned["transitions"] or step % interval or not entry.get("version"):
        raise ValueError("Requested step is outside the job's periodic checkpoint schedule")
    name = f"step-{step:012d}"
    root = WORK / "periodic" / job
    final = root / name
    previous = entry.get("periodic", {}).get(str(step))
    if previous:
        verify(final / "COMPLETE.json", previous["state_sha"])
        print(job, name, "HARVESTED_VERIFIED", previous["state_sha"])
        return
    version = entry["version"]
    index_sha = checkpoint_log_digest(entry, fr"Periodic checkpoint {name} index_sha256=([0-9a-f]{{64}})")
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(mkdtemp(prefix=f"v{version}-", dir=root))
    restored = staging / "state"
    state_sha = fetch_state_shards(entry, f"{job}/periodic/{name}", staging, restored, index_sha)
    manifest = read(restored / "COMPLETE.json")
    if (manifest["job"] != job or manifest["transitions"] != step
            or manifest["identity"]["phase"] != planned["phase"]):
        raise ValueError("Periodic checkpoint differs from the planned milestone")
    restored.rename(final)
    entry.setdefault("periodic", {})[str(step)] = {
        "state_sha": state_sha, "index_sha": index_sha, "version": version, "at": now(),
    }
    persist(state)
    print(job, name, "HARVESTED_VERIFIED", state_sha)


def harvest_policy(state, job, step):
    entry = state["jobs"][job]
    planned = entry["planned"]
    interval, state_interval = planned.get("policy_every"), planned.get("checkpoint_every")
    start = planned.get("start_transition", 0)
    if (not interval or not start < step <= start + planned["transitions"] or step % interval
            or (state_interval and step % state_interval == 0) or not entry.get("version")):
        raise ValueError("Requested step is outside the screening checkpoint schedule")
    name = f"step-{step:012d}"
    root = WORK / "screening" / job
    final = root / name
    recorded = entry.get("screening", {}).get(str(step))
    if recorded:
        verify(final / "POLICY.json", recorded["marker_sha"])
        verify(final / "policy.zip", recorded["model_sha"])
        print(job, name, "SCREENING_ONLY", recorded["model_sha"])
        return
    digest = checkpoint_log_digest(entry, fr"Screening policy {name} marker_sha256=([0-9a-f]{{64}})")
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(mkdtemp(prefix=f"v{entry['version']}-", dir=root))
    prefix = f"{job}/screening/{name}"
    fetch_output(entry, prefix, staging, "POLICY.json")
    verify(staging / "POLICY.json", digest)
    marker = read(staging / "POLICY.json")
    if (marker["schema"], marker["transitions"], marker["purpose"]) != (1, step, "SCREENING_ONLY"):
        raise ValueError("Screening checkpoint identity differs from the plan")
    fetch_output(entry, prefix, staging, "policy.zip")
    verify(staging / "policy.zip", marker["model_sha256"])
    staging.rename(final)
    entry.setdefault("screening", {})[str(step)] = {"marker_sha": digest, "model_sha": marker["model_sha256"],
                                                    "version": entry["version"], "at": now()}
    persist(state)
    print(job, name, "SCREENING_ONLY", marker["model_sha256"])

def verify_cloud_evidence(path, manifest, entry, state):
    job = entry["execution"]
    bundle = ROOT / entry["planned"]["runtime_path"] / "runtime.zip"
    verify(bundle, job["runtime_sha"])
    with zipfile.ZipFile(bundle) as runtime:
        anchors = json.loads(runtime.read("parity.json"))
        inventory = json.loads(runtime.read("source_inventory.json"))
    versions = (path / "runtime_versions.txt").read_text(encoding="utf-8").splitlines()
    if f"metadrive-commit={inventory['metadrive_pin']}" not in versions or "stable_baselines3==2.9.2" not in {
        line.lower().replace("-", "_") for line in versions
    }:
        raise ValueError("Runtime artifact did not confirm the expected MetaDrive and SB3 versions")
    actual = read(path / "parity.json")
    if len(anchors) != 2 or len(actual) != 2:
        raise ValueError("Both locked parity geometries are required")
    for expected, observed in zip(anchors, actual, strict=True):
        if expected.keys() != observed.keys():
            raise ValueError("Parity field mismatch")
        for name, value in expected.items():
            other = observed[name]
            if name in {"obs", "action", "reward"}:
                a = value if isinstance(value, list) else [value]
                b = other if isinstance(other, list) else [other]
                match = len(a) == len(b) and all(
                    abs(x - y) <= 1e-4 + 1e-4 * abs(x) and math.isfinite(y) for x, y in zip(a, b)
                )
            else:
                match = value == other
            if not match:
                raise ValueError(f"Cloud parity mismatch in {name}")
    metrics = read(path / "metrics.json")
    if not str(metrics["device"]).startswith(job["device"]):
        raise ValueError("Cloud run used a different compute device")
    if job.get("accelerator") == "NvidiaTeslaT4" and "T4" not in (metrics.get("gpu") or ""):
        raise ValueError("Cloud learner did not run on the requested T4")
    if any(metrics[k] <= 0 or not math.isfinite(metrics[k]) for k in (
        "seconds", "decisions_per_second", "updates_per_second", "post_warmup_seconds", "rss_peak_sampled_bytes"
    )):
        raise ValueError("Cloud run did not report finite positive capacity samples")
    if job.get("parent_job"):
        parent_path, digest = parent_checkpoint(state, job["parent_job"], job.get("parent_step"))
        verify(parent_path / "COMPLETE.json", digest)
        parent = read(parent_path / "COMPLETE.json")
        if (manifest["transitions"] - parent["transitions"] != metrics["transitions"]
            or manifest["updates"] - parent["updates"] != metrics["updates"]):
            raise ValueError("Cloud resume counters do not extend the parent")
        if job.get("guided_resume"):
            identity = {**parent["identity"], "capacity": job["capacity"]}
            guided_rows = parent["external_admitted"]
            parent_metrics_path = parent_path / "metrics.json"
            verify(parent_metrics_path, parent["files"]["metrics.json"])
            parent_metrics = read(parent_metrics_path)
            behavior = dict(parent_metrics["online_behavior"])
            if job.get("autonomous_traffic"):
                for key in ("traffic_guidance", "traffic_teacher_sha"):
                    identity.pop(key, None)
                    behavior.pop(key, None)
            if job.get("traffic_guidance"):
                if not 0 < metrics["traffic_guided_decisions"] <= metrics["transitions"]:
                    raise ValueError("Traffic guidance did not affect the collected actions")
                identity["traffic_guidance"] = job["traffic_guidance"]
                behavior["traffic_guidance"] = job["traffic_guidance"]
            if job.get("traffic_teacher_sha"):
                identity["traffic_teacher_sha"] = job["traffic_teacher_sha"]
                behavior["traffic_teacher_sha"] = job["traffic_teacher_sha"]
                mix = metrics["traffic_mixture"]
                guided, free = mix.get("teacher_decisions", 0), mix.get("autonomous_decisions", 0)
                if (not 0 < metrics["traffic_guided_decisions"] <= guided or free <= 0
                        or guided + free != metrics["online_tier_decisions"]["Medium"]):
                    raise ValueError("Teacher and autonomous Medium decisions do not reconcile")
            if job.get("guided_curriculum_from"):
                tiers = metrics["online_tier_decisions"]
                if (parent["identity"]["phase"] != job["guided_curriculum_from"]
                        or metrics["curriculum_from"] != job["guided_curriculum_from"]
                        or set(tiers) != set(RECIPE["phase_weights"][job["phase"]]) or min(tiers.values()) <= 0
                        or sum(tiers.values()) != metrics["transitions"]):
                    raise ValueError("Curriculum evidence must contain both Easy retention and Medium collection")
                identity["phase"] = job["phase"]
            if job.get("guided_rollout_axis"):
                axis = job["guided_rollout_axis"]
                anchor = parent["files"]["model.zip"]
                identity["guided_rollout_axis"] = axis
                behavior.update(anchored_axis=axis, anchor_model_sha=anchor)
                if metrics["guided_rollout_anchor_sha"] != anchor:
                    raise ValueError("Guided rollout anchor differs from the verified parent")
            gradient_steps = job.get("guided_gradient_steps", identity.get("guided_gradient_steps", -1))
            expected_updates = metrics["transitions"]
            if gradient_steps == 1:
                identity["guided_gradient_steps"] = 1
                expected_updates //= job["workers"]
            actor_lr_scale = job.get("guided_actor_lr_scale", identity.get("guided_actor_lr_scale"))
            if actor_lr_scale is not None:
                identity["guided_actor_lr_scale"] = actor_lr_scale
                base_lr = identity["source_recipe"]["sac"]["learning_rate"]
                if not math.isclose(metrics["guided_actor_lr"], base_lr * actor_lr_scale):
                    raise ValueError("Cloud guided actor learning rate differs from the plan")
            live_rows = guided_rows + manifest["transitions"]
            if job.get("bounded_guided_replay"):
                identity["bounded_guided_replay"] = True
                live_rows = min(live_rows, job["capacity"])
                digest = metrics["protected_replay_sha"]
                if (not re.fullmatch("[0-9a-f]{64}", digest)
                        or parent_metrics.get("protected_replay_sha", digest) != digest
                        or metrics["online_evicted_rows"] != guided_rows + manifest["transitions"] - live_rows):
                    raise ValueError("Bounded replay did not preserve the guided prefix or reconcile evictions")
            if (manifest["identity"] != identity or manifest["learning_mode"] != parent["learning_mode"]
                    or manifest["ledger"] != parent["ledger"] or manifest["external_admitted"] != guided_rows
                    or metrics["online_behavior"] != behavior
                    or metrics["guided_prefill"] != parent_metrics["guided_prefill"]
                    or metrics["guided_prefill"]["ledger"] != parent["ledger"]
                    or metrics["guided_live_rows"] != guided_rows or metrics["online_live_rows"] != live_rows - guided_rows
                    or metrics["replay_size"] != live_rows
                    or sum(metrics["online_progress_bands"].values()) != metrics["transitions"]
                    or metrics["updates"] != expected_updates):
                raise ValueError("Guided continuation did not retain replay provenance, behavior or accounting")
        return {"G3": "PASS"}
    if (manifest["transitions"], manifest["updates"]) != (metrics["transitions"], metrics["updates"]):
        raise ValueError("Cloud checkpoint counters do not match its measured work")
    if job["role"] in {"easy_pilot", "easy_long", "easy_guided"}:
        if manifest["identity"]["phase"] != "easy" or len(metrics["decisions_by_geometry"]) < 2:
            raise ValueError("Easy pilot did not train on multiple geometries in the planned phase")
        if job["role"] == "easy_guided":
            admitted = metrics["guided_live_rows"]
            if (manifest["learning_mode"] != "champion_guided_v1" or admitted != manifest["external_admitted"]
                    or admitted != sum(metrics["guided_prefill"]["progress_bands"].values())
                    or metrics["online_live_rows"] != metrics["transitions"]
                    or admitted + metrics["online_live_rows"] != metrics["replay_size"]
                    or metrics["updates"] != metrics["transitions"]):
                raise ValueError("Guided pilot replay composition or update counters differ from the plan")
        return {"pilot_training": "PASS"}
    return {"G2": "PASS"} if job["role"] == "parity" else {"capacity_sample": "PASS"}

def doctor():
    exe = shutil.which("kaggle")
    print("Kaggle CLI:", "AVAILABLE" if exe else "MISSING")
    alias = operator_slot()
    _, record = slot(alias)
    if record is None:
        print(alias, "MISSING")
    elif not verify_owner(alias):
        print(alias, "IDENTITY_UNVERIFIED")
    else:
        print(alias, "IDENTITY_VERIFIED", "accelerator_minutes", accelerator_quota(alias),
              "owner-observed sessions", record["concurrency_observed"])

def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    commands.add_parser("package").add_argument("--from-job")
    pilot = commands.add_parser("pilot-plan")
    pilot.add_argument("--campaign", required=True)
    pilot.add_argument("--round", required=True)
    pilot.add_argument("--capacity-round", required=True)
    pilot.add_argument("--account", choices=SLOTS, default=operator_slot())
    longer = commands.add_parser("learning-plan")
    longer.add_argument("--campaign", required=True)
    longer.add_argument("--round", required=True)
    longer.add_argument("--capacity-round", required=True)
    longer.add_argument("--account", choices=SLOTS, default=operator_slot())
    longer.add_argument("--reference-job", required=True)
    longer.add_argument("--evaluation-report", required=True, type=Path)
    longer.add_argument("--checkpoint", choices=("source", "4096", "8192"), required=True)
    longer.add_argument("--transitions", choices=(5000, 25000, 50000), required=True, type=int)
    longer.add_argument("--guided-prefill", type=Path)
    resume = commands.add_parser("learning-resume-plan")
    resume.add_argument("--campaign", required=True)
    resume.add_argument("--round", required=True)
    resume.add_argument("--account", choices=SLOTS, default=operator_slot())
    resume.add_argument("--parent", required=True)
    resume.add_argument("--parent-step", type=int)
    resume.add_argument("--transitions", choices=(5000, 10000, 12500, 25000, 50000, 100000), required=True, type=int)
    resume.add_argument("--device", choices=("cpu", "cuda"))
    resume.add_argument("--max-seconds", type=int)
    resume.add_argument("--phase", choices=("medium", "hard", "extreme"))
    resume.add_argument("--traffic-guidance", action="store_true")
    resume.add_argument("--traffic-teacher", action="store_true")
    resume.add_argument("--autonomous-traffic", action="store_true")
    profile = resume.add_mutually_exclusive_group()
    profile.add_argument("--reduce-guided-utd-after", metavar="HARVESTED_CONTROL_JOB")
    profile.add_argument("--dampen-guided-actor-after", metavar="HARVESTED_UTD_05_JOB")
    setup = commands.add_parser("setup")
    setup.add_argument("--account", choices=SLOTS, required=True)
    setup.add_argument("--owner", required=True)
    setup.add_argument("--gpu-minutes", type=int)
    setup.add_argument("--concurrency", type=int)
    for name in ("plan", "launch"):
        item = commands.add_parser(name)
        item.add_argument("--campaign", required=True)
        item.add_argument("--round", required=True)
        if name == "plan":
            item.add_argument("--account", choices=SLOTS, default=operator_slot())
            anchor = item.add_mutually_exclusive_group()
            anchor.add_argument("--parent")
            anchor.add_argument("--capacity-after", metavar="VERIFIED_J01_ID")
            item.add_argument("--assign", action="append", default=[], metavar="JOB=SLOT")
        else:
            item.add_argument("--job", choices=("j01", "j02", "j03", "j04", "j07", "j08", "j09", "j10", "j11"), required=True)
            item.add_argument("--fingerprint", required=True)
            admission = item.add_mutually_exclusive_group()
            admission.add_argument("--admission-probe", action="store_true")
            admission.add_argument("--server-admission", action="store_true")
            item.add_argument("--retry-of", metavar="FAILED_JOB_ID")
    for name in ("status", "harvest", "analyze", "logs"):
        commands.add_parser(name).add_argument("--job", required=True)
    periodic = commands.add_parser("harvest-periodic")
    periodic.add_argument("--job", required=True)
    periodic.add_argument("--step", required=True, type=int)
    policy = commands.add_parser("harvest-policy")
    policy.add_argument("--job", required=True)
    policy.add_argument("--step", required=True, type=int)
    snapshot = commands.add_parser("publish-b1-snapshot")
    snapshot.add_argument("--snapshot", required=True, type=Path)
    snapshot.add_argument("--snapshot-sha", required=True)
    replay_chunk = commands.add_parser("publish-b1-chunk")
    replay_chunk.add_argument("--chunk", required=True, type=Path)
    replay_chunk.add_argument("--chunk-sha", required=True)
    replay_chunk.add_argument("--learner", required=True, type=Path)
    replay_chunk.add_argument("--learner-sha", required=True)
    prepare = commands.add_parser("prepare-inputs")
    prepare.add_argument("--campaign", required=True)
    prepare.add_argument("--round", required=True)
    prepare.add_argument("--job", choices=("j01", "j02", "j03", "j04", "j07", "j08", "j09", "j10", "j11"), required=True)
    prepare.add_argument("--fingerprint", required=True)
    observe = commands.add_parser("observe")
    observe.add_argument("--account", choices=SLOTS, required=True)
    observe.add_argument("--available-sessions", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    if args.command == "setup":
        from cloud.kaggle.credentials import setup as enter
        enter(args.account, args.owner, args.gpu_minutes, args.concurrency)
    elif args.command == "observe":
        directory, record = slot(args.account)
        if not record or not verify_owner(args.account):
            raise ValueError("Account identity is unverified")
        record.update(concurrency_observed=args.available_sessions, observed_at=now()["utc"])
        write(directory / "slot.json", record)
    elif args.command == "doctor":
        doctor()
    else:
        with locked():
            state = registry()
            if args.command == "package":
                print(package(args.from_job))
            elif args.command == "plan":
                plan(args, state)
            elif args.command in {"pilot-plan", "learning-plan"}:
                plan_pilot(args, state)
            elif args.command == "learning-resume-plan":
                plan_learning_resume(args, state)
            elif args.command == "launch":
                launch(args, state)
            elif args.command == "prepare-inputs":
                job, entry = planned_job(args, state)
                if entry["status"] != "PLANNED":
                    raise ValueError("Prepare inputs only for a planned job")
                require_gate(state, job)
                alias, creds = account_for(job)
                print(inputs_for(job, state, alias, creds["owner"]))
            elif args.command == "status":
                status(state, args.job)
            elif args.command == "harvest":
                harvest(state, args.job)
            elif args.command == "harvest-periodic":
                harvest_periodic(state, args.job, args.step)
            elif args.command == "harvest-policy":
                harvest_policy(state, args.job, args.step)
            elif args.command == "publish-b1-snapshot":
                publish_initial_snapshot(args, state)
            elif args.command == "publish-b1-chunk":
                publish_verified_replay_chunk(args, state)
            elif args.command == "logs":
                diagnose(state, args.job)
            elif args.command == "analyze":
                entry = state["jobs"][args.job]
                if entry["status"] not in {"HARVESTED_VERIFIED", "PARTIAL"}:
                    raise ValueError("Analyze requires a harvested artifact")
                final = WORK / "harvest" / args.job
                verify(final / "metrics.json", read(final / "COMPLETE.json")["files"]["metrics.json"])
                print({k: v for k, v in read(final / "metrics.json").items() if k not in ("episodes", "decisions_by_geometry")})

if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, KeyError) as error:
        sys.exit(str(error))

