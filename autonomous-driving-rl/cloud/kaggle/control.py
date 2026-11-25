"""One-shot Kaggle preflight packaging, submission and artifact harvest."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

from cloud.kaggle.common import METADRIVE_PIN, ROOT, SOURCE, SOURCE_SHA, WORK, fingerprint, read, sha, verify, write
from cloud.kaggle.credentials import SLOTS, accelerator_quota, cli, private_input, slot, verify_owner

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

class DatasetPending(RuntimeError):
    pass

def audit_zip(path, label):
    with zipfile.ZipFile(path) as bundle:
        names = bundle.namelist()
        if len(set(names)) != len(names) or sum(x.file_size for x in bundle.infolist()) > 120_000_000:
            raise ValueError("Unexpected upload archive size or duplicate members")
        if label == "runtime" and set(names) != set(BUNDLE_PATHS) | {"parity.json", "source_inventory.json"}:
            raise ValueError("Runtime upload member allowlist mismatch")
        if label == "parent" and set(names) != STATE_FILES:
            raise ValueError("Parent upload member allowlist mismatch")
        for name in names:
            data = bundle.read(name)
            if SECRET.search(data):
                raise ValueError("Secret-pattern audit failed inside upload archive")
            if label == "parent" and name == "model.zip":
                from io import BytesIO
                with zipfile.ZipFile(BytesIO(data)) as model:
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
    bundle = read(WORK / "bundle.json")
    plan_id = f"{args.campaign}-{args.round}"
    assignments = dict(item.split("=") for item in getattr(args, "assign", []))
    if set(assignments) - ({"j07"} if args.parent else {"j01", "j02", "j03", "j04"}) or set(assignments.values()) - set(SLOTS):
        raise ValueError("Job assignment format is j01=account_01")
    jobs = {}
    for label, role, workers, device in (
        ("j01", "parity", 1, "cpu"), ("j02", "cpu1", 1, "cpu"),
        ("j03", "cpu2", 2, "cpu"), ("j04", "gpu2", 2, "cuda"),
    ):
        short_slug = "s5-" + fingerprint([args.campaign, args.round])[:12] + "-" + label + "-a1"
        jobs[label] = {"job": short_slug, "campaign": args.campaign, "round": args.round, "attempt": 1,
                       "role": role, "account": assignments.get(label, args.account), "workers": workers, "device": device,
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
        if parent["campaign"] != args.campaign or parent["runtime_sha"] != bundle["runtime_sha"]:
            raise ValueError("Resume requires the original runtime bundle")
        job = {"job": "s5-" + fingerprint([args.campaign, args.round])[:12] + "-j07-a1",
               "campaign": args.campaign, "round": args.round, "attempt": 1, "role": "resume",
               "account": assignments.get("j07", args.account), "workers": parent["workers"], "device": parent["device"],
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
    audit_zip(source, label)
    # Kaggle expands archives uploaded with a .zip extension, even in private datasets.
    # Keep the exact bytes as an opaque file with a distinct immutable dataset slug.
    transport_name = name.removesuffix(".zip") + ".bin"
    slug = f"s5-{label}-{digest[:20]}-bin"
    ref = f"{owner}/{slug}"
    if state["uploads"].get(ref, {}).get("status") == "ACCEPTED":
        return dataset_ready(state, alias, ref)
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
    persist(state)
    code, output = cli(alias, ["datasets", "create", "-p", str(staging), "-q", "-t"], timeout=240)
    accepted = code == 0 and f"/datasets/{ref}" in output
    state["uploads"][ref].update(status="ACCEPTED" if accepted else "UPLOAD_UNKNOWN",
                                 at=now(), cli_exit=code)
    persist(state)
    if not accepted:
        raise RuntimeError("Dataset creation acceptance is uncertain. See registry; automatic retry is disabled.")
    return dataset_ready(state, alias, ref)

def dataset_ready(state, alias, ref):
    code, output = cli(alias, ["datasets", "status", ref, "--format", "json"])
    response = dataset_status_response(code, output)
    if response.get("status") != "ready" or response.get("current_version_number") != 1:
        raise DatasetPending("Private input version 1 is not yet verified ready.")
    if not private_input(alias, ref):
        raise RuntimeError("Dataset API did not verify private ACL for version 1")
    row = state["uploads"][ref]
    if row.get("transport_name") and not row.get("remote_sha_verified_at"):
        verify_remote_input(alias, ref, row["transport_name"], row["sha"])
        row["remote_sha_verified_at"] = now()
    row.update(version=1, private_verified_at=now())
    persist(state)
    return ref

def verify_remote_input(alias, ref, name, digest):
    from tempfile import TemporaryDirectory
    from hashlib import sha256

    with TemporaryDirectory(prefix="stage5-kaggle-download-") as directory:
        code, _ = cli(alias, ["datasets", "download", ref + "/1", "-p", directory, "-q"], timeout=90)
        files = list(Path(directory).iterdir())
        if code or len(files) != 1 or not files[0].is_file() or files[0].stat().st_size > 125_000_000:
            raise RuntimeError("Private version 1 could not be retrieved for byte verification")
        with zipfile.ZipFile(files[0]) as archive:
            if archive.namelist() != [name] or archive.getinfo(name).file_size > 120_000_000:
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
    if not any(entry["status"] == "HARVESTED_VERIFIED" and entry.get("gates", {}).get("G2") == "PASS"
               and entry["execution"]["campaign"] == job["campaign"]
               and entry["execution"]["runtime_sha"] == job["runtime_sha"] for entry in state["jobs"].values()):
        raise RuntimeError("G2 requires a verified J01 cloud artifact for this campaign and runtime")
    if job.get("parent_job"):
        parent = state["jobs"].get(job["parent_job"])
        if not parent or parent["status"] != "HARVESTED_VERIFIED" or parent.get("state_sha") != job["parent_sha"]:
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

def probe_inventory(alias, owner, target):
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
        raise RuntimeError("J01 notebook already exists on Kaggle and requires reconciliation")
    # Direct GetKernel can return 403 for an unused slug. Search the authenticated
    # owner's inventory by the exact slug to cover the server's listing limit.
    query = ["kernels", "list", "--mine", "--search", target.split("/")[1], "--format", "json"]
    code, output = cli(alias, query, timeout=25)
    if code or output.splitlines()[-1:] != ["Not found"]:
        raise RuntimeError("Kaggle J01 slug search requires reconciliation")
    cutoff = (datetime.now(timezone.utc) - timedelta(days=1)).replace(tzinfo=None).isoformat()
    recent = dict.fromkeys(
        row["ref"] for index, row in enumerate(owned)
        if index < 5 or (row.get("lastRunTime") or "") >= cutoff
    )
    no_visible_session = 0
    for ref in recent:
        code, output = cli(alias, ["kernels", "status", ref], timeout=25, include_stderr=True)
        label = kernel_state(code, output)
        if label in {"QUEUED", "RUNNING"}:
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

def launch(args, state):
    job, entry = planned_job(args, state)
    if entry["status"] != "PLANNED":
        print(f"{job['job']} {entry['status']}")
        return
    require_gate(state, job)
    if any(other_id != job["job"] and other["status"] in ACTIVE
           and other["planned"]["campaign"] == job["campaign"]
           and other["planned"]["role"] == job["role"]
           and other["planned"]["runtime_sha"] == job["runtime_sha"]
           and other["planned"].get("parent_sha") == job.get("parent_sha")
           for other_id, other in state["jobs"].items()):
        raise RuntimeError("Matching campaign work is active or awaits submission reconciliation")
    alias, creds = account_for(job)
    owner = creds["owner"]
    if job.get("parent_sha") and any(x.get("execution", {}).get("parent_sha") == job["parent_sha"]
                                    for x in state["jobs"].values()):
        raise RuntimeError("This parent already has a submitted or uncertain continuation")
    age = datetime.now(timezone.utc) - datetime.fromisoformat(creds["observed_at"])
    sessions = creds["concurrency_observed"] if 0 <= age.total_seconds() <= 86400 else None
    active = [x for x in state["jobs"].values() if x.get("owner") == owner and x["status"] in ACTIVE]
    admission_probe = getattr(args, "admission_probe", False)
    if admission_probe:
        if (sessions is not None or args.job != "j01" or job["role"] != "parity" or job["device"] != "cpu"
                or job["workers"] != 1 or job.get("parent_job") or job["max_seconds"] != RECIPE["preflight"]["max_seconds"]):
            raise RuntimeError("Admission probe requires unknown capacity and the planned J01 CPU preflight")
        if active or any(x.get("admission_probe") and x["planned"]["campaign"] == job["campaign"]
                         for x in state["jobs"].values()):
            raise RuntimeError("An active receipt or prior admission probe blocks this submission")
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
    template = Path(__file__).with_name("kernel_template.py.txt")
    verify(template, job["template_sha"])
    actual_job = {k: v for k, v in job.items() if k not in ("runtime_path", "account")}
    actual_job.update(inputs_for(job, state, alias, owner))
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
    if {p.name for p in staging.iterdir()} != {"run.py", "kernel-metadata.json"} or any(
        p.is_symlink() or SECRET.search(p.read_bytes()) for p in staging.iterdir()
    ):
        raise ValueError("Kernel upload staging failed the allowlist or secret audit")
    push = ["kernels", "push", "-p", str(staging), "-t", str(job["max_seconds"])]
    if admission_probe:
        entry["admission_probe"] = {"capacity": "UNKNOWN", **probe_inventory(alias, owner, ref)}
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
    code, output = cli(alias, ["kernels", "logs", f"{entry['ref']}/{entry['version']}"], timeout=30)
    patterns = {
        "DEPENDENCY_RESOLUTION": r"No matching distribution|ResolutionImpossible|Could not find a version",
        "PYTHON_ABI": r"(?:Unvalidated|Unsupported) Kaggle Python ABI",
        "RUNTIME_IMPORT": r"ModuleNotFoundError|ImportError",
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
    if manifest["result"] == "COMPLETE":
        if metrics["transitions"] != expected["transitions"] or metrics["updates"] <= 0:
            raise ValueError("Complete artifact does not show the planned collection and updates")
        if manifest["files"]["model.zip"] == SOURCE_SHA:
            raise ValueError("The historical warm-start model cannot be a resume checkpoint")
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
        expected = entry["execution"]
        manifest = inspect_artifact(staging / filename, temp, expected)
        gates = verify_cloud_evidence(temp, manifest, entry, state) if manifest["result"] == "COMPLETE" else {}
        temp.rename(final)
    except (ValueError, KeyError, zipfile.BadZipFile):
        entry.update(status="ARTIFACT_INVALID", at=now())
        persist(state)
        raise
    entry.update(status="HARVESTED_VERIFIED" if manifest["result"] == "COMPLETE" else "PARTIAL",
                 state_sha=sha(final / "COMPLETE.json"), gates=gates, at=now())
    persist(state)
    print(job, entry["status"], entry["state_sha"])

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
    if any(metrics[k] <= 0 or not math.isfinite(metrics[k]) for k in (
        "seconds", "decisions_per_second", "updates_per_second", "post_warmup_seconds", "rss_peak_sampled_bytes"
    )):
        raise ValueError("Cloud run did not report finite positive capacity samples")
    if job.get("parent_job"):
        parent = read(WORK / "harvest" / job["parent_job"] / "COMPLETE.json")
        if (manifest["transitions"] - parent["transitions"] != metrics["transitions"]
            or manifest["updates"] - parent["updates"] != metrics["updates"]):
            raise ValueError("Cloud resume counters do not extend the parent")
        return {"G3": "PASS"}
    if (manifest["transitions"], manifest["updates"]) != (metrics["transitions"], metrics["updates"]):
        raise ValueError("Cloud checkpoint counters do not match its measured work")
    return {"G2": "PASS"} if job["role"] == "parity" else {"capacity_sample": "PASS"}

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
            print(alias, "IDENTITY_VERIFIED", "accelerator_minutes", accelerator_quota(alias),
                  "owner-observed sessions", record["concurrency_observed"])

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
            item.add_argument("--assign", action="append", default=[], metavar="JOB=SLOT")
        else:
            item.add_argument("--job", choices=("j01", "j02", "j03", "j04", "j07"), required=True)
            item.add_argument("--fingerprint", required=True)
            item.add_argument("--admission-probe", action="store_true")
    for name in ("status", "harvest", "analyze", "logs"):
        commands.add_parser(name).add_argument("--job", required=True)
    prepare = commands.add_parser("prepare-inputs")
    prepare.add_argument("--campaign", required=True)
    prepare.add_argument("--round", required=True)
    prepare.add_argument("--job", choices=("j01", "j02", "j03", "j04", "j07"), required=True)
    prepare.add_argument("--fingerprint", required=True)
    import_slots = commands.add_parser("import-slots")
    import_slots.add_argument("--file", type=Path, required=True)
    observe = commands.add_parser("observe")
    observe.add_argument("--account", choices=SLOTS, required=True)
    observe.add_argument("--available-sessions", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    if args.command == "setup":
        from cloud.kaggle.credentials import setup as enter
        enter(args.account, args.owner, args.gpu_minutes, args.concurrency)
    elif args.command == "import-slots":
        from cloud.kaggle.credentials import import_file
        import_file(args.file)
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
                print(package())
            elif args.command == "plan":
                plan(args, state)
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

