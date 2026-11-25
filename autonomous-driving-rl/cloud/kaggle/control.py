"""One-shot Kaggle preflight packaging, submission and artifact harvest."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

from cloud.kaggle.common import METADRIVE_PIN, ROOT, SOURCE, SOURCE_SHA, WORK, fingerprint, read, sha, verify, write
from cloud.kaggle.credentials import SLOTS, cli, slot, verify_owner

RECIPE = read(Path(__file__).with_name("recipe.json"))
BUNDLE_PATHS = (
    "cloud/kaggle/common.py", "cloud/kaggle/runtime.py", "cloud/kaggle/replay.py",
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
SECRET = re.compile(rb"(?i)(?:KGAT_[a-zA-Z0-9_-]{12,}|gh[pousr]_[a-zA-Z0-9_]{20,}|github_pat_[a-zA-Z0-9_]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|AIza[a-zA-Z0-9_-]{30,})")
ACTIVE = {"SUBMITTING", "SUBMIT_UNKNOWN", "SUBMITTED", "QUEUED", "RUNNING", "STATUS_UNKNOWN"}

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

def package():
    verify(SOURCE, SOURCE_SHA)
    anchor = WORK / "parity.json"
    if not anchor.is_file():
        raise RuntimeError("Run the local parity command first")
    files = {name: (ROOT / name).read_bytes() for name in BUNDLE_PATHS}
    files["parity.json"] = anchor.read_bytes()
    if any(SECRET.search(data) for data in files.values()):
        raise RuntimeError("Secret-pattern audit failed for allowlisted runtime")
    import hashlib
    identifiers = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    inventory = {"git_head_at_packaging": base, "runtime_id": fingerprint(identifiers), "files": identifiers,
                 "source_model_sha256": SOURCE_SHA, "metadrive_pin": METADRIVE_PIN}
    files["source_inventory.json"] = json.dumps(inventory, sort_keys=True).encode()
    staging = WORK / ("package-" + fingerprint(inventory)[:20])
    if not staging.exists():
        staging.mkdir()
        archive(staging / "runtime.zip", files)
        shutil.copyfile(SOURCE, staging / "model.zip")
        write(staging / "source_inventory.json", inventory)
    verify(staging / "model.zip", SOURCE_SHA)
    with zipfile.ZipFile(staging / "runtime.zip") as bundle:
        if set(bundle.namelist()) != set(files) or any(bundle.read(name) != data for name, data in files.items()):
            raise ValueError("Existing immutable package differs. Preserve it and investigate.")
    result = {"runtime_sha": sha(staging / "runtime.zip"), "source_sha": SOURCE_SHA,
              "runtime_bytes": (staging / "runtime.zip").stat().st_size, "source_bytes": SOURCE.stat().st_size,
              "runtime_id": inventory["runtime_id"], "path": str(staging.relative_to(ROOT)),
              "template_sha": identifiers["cloud/kaggle/kernel_template.py.txt"]}
    write(WORK / "bundle.json", result)
    return result

def registry():
    path = WORK / "job_registry.json"
    return read(path) if path.exists() else {"schema": 1, "jobs": {}, "uploads": {}}

def persist(state):
    write(WORK / "job_registry.json", state)

def identity_part(text):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,17}", text):
        raise ValueError("Campaign and round identifiers require 1–18 lowercase slug characters")

def plan(args, state):
    identity_part(args.campaign)
    identity_part(args.round)
    bundle = read(WORK / "bundle.json")
    plan_id = f"{args.campaign}-{args.round}"
    jobs = {}
    for label, role, workers, device in (
        ("j01", "parity", 1, "cpu"), ("j02", "cpu1", 1, "cpu"),
        ("j03", "cpu2", 2, "cpu"), ("j04", "gpu2", 2, "cuda"),
    ):
        short_slug = "s5-" + fingerprint([args.campaign, args.round])[:12] + "-" + label + "-a1"
        jobs[label] = {"job": short_slug, "campaign": args.campaign, "round": args.round, "attempt": 1,
                       "role": role, "account": args.account, "workers": workers, "device": device,
                       "runtime_sha": bundle["runtime_sha"], "source_sha": bundle["source_sha"],
                       "runtime_path": bundle["path"], "template_sha": bundle["template_sha"], **RECIPE["preflight"]}
    deferred = {"j05": "Local B1 collector proof only; cloud snapshot sharing unverified",
                "j06": "Depends on a verified private collector artifact",
                "j07": "Depends on a harvested COMPLETE online state"}
    if args.parent:
        record = state["jobs"].get(args.parent)
        if not record or record["status"] != "HARVESTED_VERIFIED":
            raise ValueError("Resume needs a verified harvested parent")
        parent = record["execution"]
        if parent["runtime_sha"] != bundle["runtime_sha"]:
            raise ValueError("Resume requires the original runtime bundle")
        job = {"job": "s5-" + fingerprint([args.campaign, args.round])[:12] + "-j07-a1",
               "campaign": args.campaign, "round": args.round, "attempt": 1, "role": "resume",
               "account": args.account, "workers": parent["workers"], "device": parent["device"],
               "runtime_sha": bundle["runtime_sha"], "source_sha": bundle["source_sha"],
               "runtime_path": bundle["path"], "template_sha": bundle["template_sha"], "parent_job": args.parent,
               "parent_sha": record["state_sha"], **RECIPE["preflight"]}
        jobs = {"j07": job}
        deferred.pop("j07")
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

def upload(state, alias, owner, label, name, source, digest):
    verify(source, digest)
    slug = f"s5-{label}-{digest[:20]}"
    ref = f"{owner}/{slug}"
    if state["uploads"].get(ref, {}).get("status") == "ACCEPTED":
        return dataset_ready(state, alias, ref)
    if ref in state["uploads"]:
        raise RuntimeError("Prior dataset upload is uncertain. Inspect the private dataset before any recovery.")
    staging = WORK / "uploads" / alias / slug
    staging.mkdir(parents=True, exist_ok=True)
    dest = staging / name
    if dest.exists():
        verify(dest, digest)
    else:
        shutil.copyfile(source, dest)
    write(staging / "dataset-metadata.json", {"id": ref, "title": slug, "licenses": [{"name": "other"}]})
    state["uploads"][ref] = {"status": "UPLOADING", "sha": digest, "alias": alias, "at": now()}
    persist(state)
    code, output = cli(alias, ["datasets", "create", "-p", str(staging), "-q", "-t"], timeout=240)
    accepted = code == 0 and f"/datasets/{ref}" in output
    state["uploads"][ref] = {"status": "ACCEPTED" if accepted else "UPLOAD_UNKNOWN",
                             "sha": digest, "alias": alias, "at": now(), "cli_exit": code}
    persist(state)
    if not accepted:
        raise RuntimeError("Dataset creation acceptance is uncertain. See registry; automatic retry is disabled.")
    return dataset_ready(state, alias, ref)

def dataset_ready(state, alias, ref):
    code, output = cli(alias, ["datasets", "status", ref, "--format", "json"])
    try:
        response = json.loads(output) if code == 0 else {}
    except json.JSONDecodeError:
        response = {}
    if response.get("status") != "ready" or response.get("current_version_number") != 1:
        raise RuntimeError("Private input version 1 is not yet verified ready. Check status and repeat launch later.")
    state["uploads"][ref]["version"] = 1
    persist(state)
    return ref

def launch(args, state):
    path = WORK / f"plan-{args.campaign}-{args.round}.json"
    identity_part(args.campaign)
    identity_part(args.round)
    plan_doc = read(path)
    body = {k: plan_doc[k] for k in ("jobs", "deferred")}
    if fingerprint(body) != args.fingerprint or plan_doc["fingerprint"] != args.fingerprint:
        raise ValueError("Plan fingerprint mismatch")
    job = body["jobs"][args.job]
    entry = state["jobs"][job["job"]]
    if entry["status"] != "PLANNED":
        print(f"{job['job']} {entry['status']}")
        return
    alias = job["account"]
    _, creds = slot(alias)
    if not creds or not verify_owner(alias):
        raise RuntimeError("READY_FOR_CREDENTIALS: account identity unverified")
    owner = creds["owner"]
    if job.get("parent_sha") and any(x.get("execution", {}).get("parent_sha") == job["parent_sha"]
                                    for x in state["jobs"].values()):
        raise RuntimeError("This parent already has a submitted or uncertain continuation")
    age = datetime.now(timezone.utc) - datetime.fromisoformat(creds["observed_at"])
    if not 0 <= age.total_seconds() <= 86400 or creds["concurrency_observed"] < 1:
        raise RuntimeError("Fresh owner observation of available concurrency is required")
    active = [x for x in state["jobs"].values() if x.get("owner") == owner and x["status"] in ACTIVE]
    if len(active) >= creds["concurrency_observed"]:
        raise RuntimeError("Account concurrency budget is occupied")
    observed_gpu_jobs = [x for x in state["jobs"].values() if x.get("owner") == owner
                         and x.get("execution", {}).get("device") == "cuda"
                         and x.get("submitted", {}).get("utc", "") >= creds["observed_at"]]
    gpu_minutes = sum(x["execution"]["max_seconds"] / 60 for x in observed_gpu_jobs)
    if job["device"] == "cuda" and gpu_minutes + job["max_seconds"] / 60 > creds["gpu_minutes_observed"]:
        raise RuntimeError("Observed GPU budget is insufficient")
    runtime_dir = (ROOT / job["runtime_path"]).resolve()
    if not runtime_dir.is_relative_to(WORK.resolve()):
        raise ValueError("Plan runtime path is outside campaign workspace")
    template = Path(__file__).with_name("kernel_template.py.txt")
    verify(template, job["template_sha"])
    actual_job = {k: v for k, v in job.items() if k not in ("runtime_path", "account")}
    actual_job["runtime_ref"] = upload(state, alias, owner, "runtime", "runtime.zip",
                                       runtime_dir / "runtime.zip", job["runtime_sha"])
    actual_job["source_ref"] = upload(state, alias, owner, "source", "model.zip",
                                      runtime_dir / "model.zip", job["source_sha"])
    if job.get("parent_job"):
        parent = WORK / "harvest" / job["parent_job"]
        verify(parent / "COMPLETE.json", job["parent_sha"])
        parent_archive = parent / "parent.zip"
        manifest = read(parent / "COMPLETE.json")
        for name, expected in manifest["files"].items():
            verify(parent / name, expected)
        if not parent_archive.exists():
            with zipfile.ZipFile(parent_archive.with_suffix(".tmp"), "x") as output:
                for name in STATE_FILES:
                    output.write(parent / name, name)
            parent_archive.with_suffix(".tmp").rename(parent_archive)
        with zipfile.ZipFile(parent_archive) as archived:
            if set(archived.namelist()) != STATE_FILES or any(archived.read(name) != (parent / name).read_bytes() for name in STATE_FILES):
                raise ValueError("Parent archive differs from verified state")
        actual_job["parent_zip_sha"] = sha(parent_archive)
        actual_job["parent_ref"] = upload(state, alias, owner, "parent", "parent.zip",
                                         parent_archive, actual_job["parent_zip_sha"])
    staging = WORK / "kernels" / job["job"]
    staging.mkdir(parents=True, exist_ok=True)
    template_text = template.read_text(encoding="utf-8")
    refs = [actual_job["runtime_ref"], actual_job["source_ref"]]
    if actual_job.get("parent_ref"):
        refs.append(actual_job["parent_ref"])
    actual_job["input_versions"] = {key: state["uploads"].get(key, {}).get("version", 1) for key in refs}
    ref = f"{owner}/{job['job']}"
    (staging / "run.py").write_text(template_text.replace("__JOB__", repr(actual_job)), encoding="utf-8")
    write(staging / "kernel-metadata.json", {"id": ref, "title": job["job"], "code_file": "run.py",
          "language": "python", "kernel_type": "script", "is_private": True, "enable_internet": True,
          "enable_gpu": job["device"] == "cuda", "dataset_sources": refs,
          "competition_sources": [], "kernel_sources": [], "model_sources": []})
    push = ["kernels", "push", "-p", str(staging), "-t", str(job["max_seconds"])]
    entry.update(status="SUBMITTING", owner=owner, ref=ref, execution=actual_job, submitted=now())
    persist(state)
    code, output = cli(alias, push, timeout=90)
    match = re.search(r"\bversion\s+(\d+)\b", output, re.I)
    entry.update(status="SUBMITTED" if code == 0 and match and ref in output else "SUBMIT_UNKNOWN",
                 version=int(match[1]) if match else None, cli_exit=code, at=now())
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
    match = re.search(r'has status "\s*(?:KernelWorkerStatus\.)?(complete|running|queued|error|cancelled)\s*"', output, re.I)
    mapping = {"complete": "CLOUD_SUCCEEDED", "error": "FAILED", "cancelled": "CANCELLED"}
    label = match[1].lower() if code == 0 and match else ""
    entry.update(status=mapping.get(label, label.upper() or "STATUS_UNKNOWN"), at=now())
    persist(state)
    print(job, entry["status"])

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
        if (manifest["job"] != expected["job"] or manifest["parent_sha"] != expected.get("parent_sha")
                or set(manifest["files"]) != STATE_FILES - {"COMPLETE.json"}
                or manifest["identity"]["workers"] != expected["workers"]
                or manifest["identity"]["warm_source"] != SOURCE_SHA):
            raise ValueError("Wrong checkpoint identity")
        out.mkdir(parents=True)
        for name in names:
            with bundle.open(name) as stream, (out / name).open("xb") as target:
                shutil.copyfileobj(stream, target)
    for name, digest in manifest["files"].items():
        verify(out / name, digest)
    metrics = read(out / "metrics.json")
    fingerprint(metrics)
    if manifest["result"] == "COMPLETE" and (metrics["transitions"] != expected["transitions"] or metrics["updates"] <= 0):
        raise ValueError("Complete artifact does not show the planned collection and updates")
    return manifest

def harvest(state, job):
    entry = state["jobs"][job]
    if entry["status"] in {"HARVESTED_VERIFIED", "PARTIAL"}:
        print(job, entry["status"])
        return
    status(state, job)
    if entry["status"] != "CLOUD_SUCCEEDED":
        return
    staging = WORK / "downloads" / job
    staging.mkdir(parents=True, exist_ok=True)
    alias = entry["planned"]["account"]
    filename = job + ".zip"
    code, _ = cli(alias, ["kernels", "output", f"{entry['ref']}/{entry['version']}", "-p", str(staging),
                          "--file-pattern", f"^{re.escape(filename)}$", "-q"], timeout=180)
    if code != 0 or not (staging / filename).is_file():
        raise RuntimeError("Expected version-specific Kaggle output unavailable")
    final = WORK / "harvest" / job
    temp = final.with_name(job + ".staging")
    try:
        manifest = inspect_artifact(staging / filename, temp, entry["execution"])
        temp.rename(final)
    except (ValueError, KeyError, zipfile.BadZipFile):
        entry.update(status="ARTIFACT_INVALID", at=now())
        persist(state)
        raise
    entry.update(status="HARVESTED_VERIFIED" if manifest["result"] == "COMPLETE" else "PARTIAL",
                 state_sha=sha(final / "COMPLETE.json"), at=now())
    persist(state)
    print(job, entry["status"], entry["state_sha"])

def doctor():
    exe = shutil.which("kaggle")
    print("Kaggle CLI:", "AVAILABLE" if exe else "MISSING")
    for alias in SLOTS:
        _, record = slot(alias)
        if record is None:
            print(alias, "MISSING")
        elif not verify_owner(alias):
            print(alias, "IDENTITY_UNVERIFIED")
        else:
            print(alias, "IDENTITY_VERIFIED", "owner-observed quota", record["observed_at"])

def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    commands.add_parser("package")
    setup = commands.add_parser("setup")
    setup.add_argument("--account", choices=SLOTS, required=True)
    setup.add_argument("--owner", required=True)
    setup.add_argument("--gpu-minutes", type=int, default=0)
    setup.add_argument("--concurrency", type=int, default=0)
    for name in ("plan", "launch"):
        item = commands.add_parser(name)
        item.add_argument("--campaign", required=True)
        item.add_argument("--round", required=True)
        if name == "plan":
            item.add_argument("--account", choices=SLOTS, default=SLOTS[0])
            item.add_argument("--parent")
        else:
            item.add_argument("--job", choices=("j01", "j02", "j03", "j04", "j07"), required=True)
            item.add_argument("--fingerprint", required=True)
    for name in ("status", "harvest", "analyze"):
        commands.add_parser(name).add_argument("--job", required=True)
    args = parser.parse_args()
    if args.command == "setup":
        from cloud.kaggle.credentials import setup as enter
        enter(args.account, args.owner, args.gpu_minutes, args.concurrency)
    elif args.command == "doctor":
        doctor()
    else:
        with locked():
            state = registry()
            if args.command == "package":
                print(package())
            elif args.command == "plan":
                plan(args, state)
            elif args.command == "launch":
                launch(args, state)
            elif args.command == "status":
                status(state, args.job)
            elif args.command == "harvest":
                harvest(state, args.job)
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

