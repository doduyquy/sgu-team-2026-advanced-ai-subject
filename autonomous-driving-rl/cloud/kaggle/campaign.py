"""Credential-isolated Stage 5 campaign queue and receipt transport."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from tempfile import mkdtemp
import time
from types import SimpleNamespace
import zipfile

from cloud.kaggle import control as c, parallel
from cloud.kaggle.common import WORK, fingerprint, read, sha, verify, write
from cloud.kaggle.credentials import operator_slot, slot, verify_owner
from cloud.kaggle.pilot import panel, summarize

GROUPS = (
    ("easy-t4", "medium-t4"), ("main", "main-long"), ("steering", "steer-long"),
    ("throttle", "brake-long"), ("retention", "medium-cpu"), ("guided-a", "guided-b"),
    ("eval-easy", "eval-medium"), ("easy-stable", "mixed"),
)
TERMINAL = {"WINDOW_ENDED", "HARVESTED_VERIFIED", "ARTIFACT_INVALID", "PARTIAL", "FAILED", "CAPACITY_REJECTED", "CANCELLED"}
UNCERTAIN = {"SUBMITTING", "SUBMIT_UNKNOWN", "STATUS_UNKNOWN"}


def sessions():
    rows = []
    for account, workloads in zip(c.SLOTS, GROUPS, strict=True):
        for workload in workloads:
            if workload == "mixed":
                budget, profile, job = 25000, {"device": "cuda"}, None
            else:
                assigned, budget, profile = parallel.WORKLOADS[workload]
                if assigned != account:
                    raise ValueError("Campaign session owner differs from the workload owner")
                job = parallel.job_id(workload)
            unit = {"collect": "episodes", "evaluate": "cases"}.get(profile.get("parallel_task"), "transitions")
            rows.append({"session": f"S{len(rows) + 1:02d}", "account": account, "workload": workload,
                         "job": job, "device": profile.get("device", "cpu"), "budget": f"{budget} {unit}"})
    return rows


WINDOW_PROFILES = (
    ("Medium teacher to autonomous", "cuda", "traffic", None, True, False, None),
    ("Medium autonomous retention", "cpu", "traffic", None, False, True, 0.25),
    ("Easy stability", "cuda", "easy", None, False, False, 0.25),
    ("Medium longitudinal teacher", "cpu", "traffic", None, True, False, 0.25),
    ("Medium autonomous, reduced actor LR", "cuda", "traffic", None, False, True, 0.25),
    ("Easy coverage", "cpu", "easy", None, False, False, None),
    ("Medium capped car following", "cuda", "easy", "medium", False, False, 0.25),
    ("Medium teacher replicate", "cpu", "traffic", None, True, False, None),
    ("Mixed Easy/Medium autonomous", "cuda", "easy", "medium", False, False, None),
    ("Easy retention, reduced actor LR", "cpu", "easy", None, False, False, 0.25),
    ("Medium traffic-guided collector?learner", "cuda", "easy", "medium", False, False, None),
    ("Medium teacher traffic experience", "cpu", "traffic", None, True, False, 0.25),
    ("Medium traffic teacher, reduced actor LR", "cuda", "traffic", None, True, False, 0.25),
    ("Mixed Easy/Medium retention", "cpu", "easy", "medium", False, False, 0.25),
    ("Medium learned traffic behavior", "cuda", "traffic", None, False, True, None),
    ("Medium traffic experience replicate", "cpu", "easy", "medium", False, False, None),
)


def prepare_window(root, package, source):
    clock = read(WORK / "window12.json")
    deadline = clock["deadline_epoch"]
    if (root / "campaign.json").exists():
        raise FileExistsError("Campaign dispatch is immutable")
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    with c.locked():
        registry = c.registry()
        c.package(source)
        for i, (workload, device, parent_kind, phase, teacher, autonomous, lr) in enumerate(WINDOW_PROFILES):
            session = f"S{i + 1:02d}"
            account = c.SLOTS[i // 2]
            round_id = f"{clock.get('round_prefix', 'win12')}-{session.lower()}"
            parent = source if parent_kind == "traffic" else parallel.CHAMPION
            guidance = phase == "medium" and ("guided" in workload or "capped" in workload or "experience" in workload)
            c.plan_learning_resume(SimpleNamespace(campaign="stage5", round=round_id, account=account,
                parent=parent, transitions=4_000_000, device=device, max_seconds=39600, phase=phase,
                traffic_guidance=guidance, traffic_teacher=teacher, autonomous_traffic=autonomous,
                campaign_deadline=deadline, campaign_actor_lr=lr), registry)
            doc = read(WORK / f"plan-stage5-{round_id}.json")
            job = doc["jobs"]["j10"]
            rows.append(dict(session=session, account=account, workload=workload, job=job["job"], device=device,
                             budget="39,600 s or campaign deadline", round=round_id,
                             plan_fingerprint=doc["fingerprint"], parent=parent, parent_sha=job["parent_sha"],
                             checkpoint="50k policy / 250k full state / final sharded"))
    data = {"target": 16, "per_account_target": 2, "parent_sha": parallel.STATE,
            "package_sha": sha(package), "sessions": rows, "window": clock}
    data["fingerprint"] = fingerprint(data)
    write(root / "campaign.json", data)
    print(data["fingerprint"])


def manifest(root):
    document = read(root / "campaign.json")
    fields = {k: v for k, v in document.items() if k != "fingerprint"}
    if (document["fingerprint"] != fingerprint(fields) or (document["sessions"] != sessions() if "window" not in document else
                [(row["session"], row["account"]) for row in document["sessions"]] !=
                [(f"S{i + 1:02d}", c.SLOTS[i // 2]) for i in range(16)])
            or document["target"] != 16 or document["per_account_target"] != 2
            or document["parent_sha"] != parallel.STATE):
        raise ValueError("Campaign definition differs from the pinned workload queue")
    return document


def init(root, package):
    root.mkdir(parents=True, exist_ok=True)
    data = {"target": 16, "per_account_target": 2, "parent_sha": parallel.STATE,
            "package_sha": sha(package), "sessions": sessions()}
    data["fingerprint"] = fingerprint(data)
    path = root / "campaign.json"
    if path.exists() and read(path) != data:
        raise ValueError("Campaign already exists with another deployment package")
    write(path, data)
    print(data["fingerprint"])


@contextmanager
def agent_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / ".agent.lock"
    lock.mkdir()
    try:
        yield
    finally:
        lock.rmdir()


def enqueue_continuation(root, document, args):
    account = operator_slot()
    with c.locked():
        registry = c.registry()
        parent = registry["jobs"][args.parent]
        if parent.get("imported_evidence") or parent["planned"]["account"] != account:
            raise ValueError("Continuation must be queued from the verified parent on its owning host")
        c.package(args.parent)
        c.plan_learning_resume(SimpleNamespace(campaign="stage5", round=args.round, account=account,
                                               parent=args.parent, parent_step=args.parent_step,
                                               transitions=args.transitions, device=args.device,
                                               max_seconds=args.max_seconds, phase=args.phase,
                                               traffic_guidance=args.traffic_guidance, traffic_teacher=args.traffic_teacher,
                                               autonomous_traffic=args.autonomous_traffic,
                                               campaign_deadline=getattr(args, "campaign_deadline", None)), registry)
        planned = read(WORK / f"plan-stage5-{args.round}.json")
        job = planned["jobs"]["j10"]
    folder = root / "rounds" / account
    folder.mkdir(parents=True, exist_ok=True)
    item = {"session": f"R-{args.round}", "account": account, "workload": job.get("parallel_workload", "continuation"),
            "job": job["job"], "device": job["device"], "budget": f"{job['transitions']} transitions",
            "round": args.round, "plan_fingerprint": planned["fingerprint"], "manifest": document["fingerprint"]}
    path = folder / f"{args.round}.json"
    if path.exists() and read(path) != item:
        raise ValueError("Continuation dispatch is immutable")
    write(path, item)
    print(job["job"])


def repair_window_localization(root, document, session):
    account = operator_slot()
    rows = [row for row in document["sessions"] if row["session"] == session and row["account"] == account]
    if "window" not in document or len(rows) != 1 or not verify_owner(account):
        raise ValueError("A localization repair requires the matching timeboxed session and identity")
    row = rows[0]
    entry = c.registry()["jobs"][row["job"]]
    if entry["status"] != "FAILED" or not entry.get("version"):
        raise ValueError("A repair requires a failed versioned submission")
    code, status = c.cli(account, ["kernels", "status", f"{entry['ref']}/{entry['version']}"], timeout=25)
    if c.kernel_state(code, status) != "FAILED" or c.current_private_kernel_version(account, entry["ref"]) != entry["version"]:
        raise ValueError("The original notebook version is not confirmed failed")
    code, logs = c.cli(account, ["kernels", "logs", f"{entry['ref']}/{entry['version']}"], timeout=25)
    if code:
        raise RuntimeError("Versioned localization logs are temporarily unavailable")
    if "TaskContextLocalizationError" not in logs:
        raise ValueError("Cloud logs do not establish a route localization failure")
    plan = entry["planned"]
    round_id = row["round"] + "g"
    with c.locked():
        registry = c.registry()
        c.package(plan["parent_job"])
        c.plan_learning_resume(SimpleNamespace(campaign="stage5", account=account, round=round_id,
            parent=plan["parent_job"], transitions=4_000_000, max_seconds=plan["max_seconds"],
            campaign_deadline=plan["campaign_deadline"], campaign_actor_lr=plan.get("guided_actor_lr_scale"),
            device=plan["device"], phase=plan["phase"] if plan.get("guided_curriculum_from") else None,
            traffic_guidance=bool(plan.get("traffic_guidance")) and not plan.get("traffic_teacher_sha"),
            traffic_teacher=bool(plan.get("traffic_teacher_sha")), autonomous_traffic=plan.get("autonomous_traffic", False),
            localization_guard=True), registry)
        document_plan = read(WORK / f"plan-stage5-{round_id}.json")
        job = document_plan["jobs"]["j10"]
        folder = root / "rounds" / account
        folder.mkdir(parents=True, exist_ok=True)
        dispatch = {**row, "session": f"R-{round_id}", "round": round_id, "job": job["job"],
                    "plan_fingerprint": document_plan["fingerprint"], "manifest": document["fingerprint"],
                    "supersedes": row["job"], "repair": "exclude_unlocalized_vectors_from_replay"}
        path = folder / f"{round_id}.json"
        if path.exists() and read(path) != dispatch:
            raise ValueError("Existing repair dispatch is immutable")
        write(path, dispatch)
    print(job["job"], flush=True)


def watch_localization_repairs(root, package, document, account):
    while time.time() < document["window"]["deadline_epoch"] + 3600:
        registry = c.registry()
        watched = [row for row in document["sessions"] if row["account"] == account]
        pending = False
        for row in watched:
            entry = registry["jobs"][row["job"]]
            repair_round = row["round"] + "g"
            dispatch = root / "rounds" / account / f"{repair_round}.json"
            if (entry["status"] == "FAILED" and not dispatch.is_file()
                    and time.time() < document["window"]["deadline_epoch"] - 1800):
                try:
                    repair_window_localization(root, document, row["session"])
                except (ValueError, RuntimeError) as error:
                    # Retry authentication or status reads. A verified non-F09 error is left terminal.
                    if not str(error).startswith("Cloud logs do not establish"):
                        pending = True
            if dispatch.is_file():
                try:
                    pending |= run_once(root, package, document, account, f"R-{repair_round}")
                except RuntimeError as exc:
                    if not str(exc).startswith("Registry locked."):
                        raise
                    pending = True
            pending |= entry["status"] not in TERMINAL
        if not pending:
            return
        time.sleep(90 + c.SLOTS.index(account))


def job_queue(root, document):
    queue = list(document["sessions"])
    known = {row["job"] for row in queue}
    for account in c.SLOTS:
        for path in sorted((root / "rounds" / account).glob("*.json")):
            data = read(path)
            c.identity_part(path.stem)
            if (data["manifest"] != document["fingerprint"] or data["account"] != account
                    or data["round"] != path.stem or data["session"] != f"R-{path.stem}"
                    or not data["job"] or data["job"] in known):
                raise ValueError("Invalid continuation dispatch")
            known.add(data["job"])
            queue.append(data)
    return queue


def harvest_aux(state, job):
    entry = state["jobs"][job]
    plan = entry["execution"]
    mode = plan["parallel_task"]
    out = WORK / "harvest" / job
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    stage = Path(mkdtemp(prefix=f"{job}-", dir=out.parent))
    try:
        c.fetch_output(entry, job, stage, "AUX.json")
        proof = read(stage / "AUX.json")
        file = "guided.zip" if mode == "collect" else "panel.json"
        if (proof["kind"], proof["parent_sha"], proof["model_sha"], proof["file"]) != (
                mode, parallel.STATE, parallel.MODEL, file):
            raise ValueError("Companion output differs from the approved request")
        c.fetch_output(entry, job, stage, file)
        verify(stage / file, proof["sha256"])
        if mode == "evaluate":
            result = read(stage / file)
            if result["model_sha"] != parallel.MODEL:
                raise ValueError("Evaluator used a different model")
            panel(result, tier=plan["evaluation_tier"])
        else:
            c.audit_zip(stage / file, "guided")
            with zipfile.ZipFile(stage / file) as z:
                payload = json.loads(z.read("GUIDED.json"))
                snapshot = z.read("snapshot/SNAPSHOT.json")
                if (hashlib.sha256(snapshot).hexdigest() != payload["snapshot_sha"]
                        or json.loads(snapshot)["policy_sha"] != parallel.MODEL
                        or hashlib.sha256(z.read("snapshot/policy.zip")).hexdigest() != parallel.MODEL
                        or len(payload["chunks"]) != 3):
                    raise ValueError("Wrong collector snapshot or episode count")
                for i, digest in enumerate(payload["chunks"]):
                    raw = z.read(f"chunk-{i:02}/CHUNK.json")
                    chunk = json.loads(raw)
                    rows = z.read(f"chunk-{i:02}/rows.npz")
                    if (hashlib.sha256(raw).hexdigest() != digest or chunk["stream"] != plan["collector_stream"]
                            or chunk["snapshot_sha"] != payload["snapshot_sha"]
                            or hashlib.sha256(rows).hexdigest() != chunk["payload_sha"]):
                        raise ValueError("Collector chunk provenance failed")
    except (ValueError, KeyError, zipfile.BadZipFile):
        entry.update(status="ARTIFACT_INVALID", at=c.now())
        c.persist(state)
        raise
    stage.rename(out)
    entry.update(status="HARVESTED_VERIFIED", output_version=entry["version"], output_sha=proof["sha256"], at=c.now())
    c.persist(state)


def candidate(job, entry):
    if entry["status"] != "HARVESTED_VERIFIED" or not entry.get("state_sha"):
        return None
    root = WORK / "harvest" / job
    verify(root / "COMPLETE.json", entry["state_sha"])
    marker = read(root / "COMPLETE.json")
    verify(root / "model.zip", marker["files"]["model.zip"])
    result = {"state_sha": entry["state_sha"], "model_sha": marker["files"]["model.zip"],
              "transitions": marker["transitions"], "updates": marker["updates"], "panels": {}}
    for tier, file in (("Easy", "panel.json"), ("Medium", "panel-medium.json")):
        if (root / file).is_file():
            data = read(root / file)
            if data["model_sha"] != result["model_sha"]:
                raise ValueError("Candidate panel model differs from checkpoint")
            result["panels"][tier] = summarize(panel(data, tier=tier))
    return result


def training_evidence(account, entry, prior):
    # Preserve earlier evidence when cloud logs are briefly unavailable.
    evidence = prior.get("execution_evidence", {})
    if entry["status"] == "HARVESTED_VERIFIED":
        metrics = read(WORK / "harvest" / entry["planned"]["job"] / "metrics.json")
        return {"source": "verified artifact", "observed_at": c.now(),
                "transitions": metrics["transitions"], "updates": metrics["updates"]}
    if entry["status"] != "RUNNING" or not entry.get("version"):
        return evidence
    checked = evidence.get("observed_at", {}).get("utc")
    if checked and (datetime.now(timezone.utc) - datetime.fromisoformat(checked)).total_seconds() < 240:
        return evidence
    if c.current_private_kernel_version(account, entry["ref"]) != entry["version"]:
        return evidence
    code, logs = c.cli(account, ["kernels", "logs", f"{entry['ref']}/{entry['version']}"], timeout=20)
    if code:
        return evidence
    transitions, updates = 0, 0
    for steps, gradients in re.findall(r"\b(?:TRAINING_ACTIVE|COMPLETE|PARTIAL) transitions=(\d+) updates=(\d+)", logs, re.M):
        transitions, updates = max(transitions, int(steps)), max(updates, int(gradients))
    for step in re.findall(r"\bPeriodic checkpoint step-(\d+) index_sha256=", logs, re.M):
        transitions = max(transitions, int(step) - entry["planned"]["start_transition"])
    return {"source": "versioned Kaggle log", "observed_at": c.now(),
            "transitions": max(transitions, evidence.get("transitions", 0)),
            "updates": max(updates, evidence.get("updates", 0))}


def run_once(root, package, document, account, session=None):
    verify(package, document["package_sha"])
    _, own = slot(account)
    authorized = bool(own and verify_owner(account))
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(own["observed_at"])).total_seconds() if authorized else None
    capacity = own["concurrency_observed"] if age is not None and 0 <= age <= 86400 else None
    folder = root / "receipts" / account
    pending = False
    for item in job_queue(root, document):
        if item["account"] != account or (session and item["session"] != session):
            continue
        receipt = folder / f"{item['session']}.json"
        status = {**item, "manifest": document["fingerprint"], "status": "PLANNED", "checked_at": c.now(),
                  "verified_concurrency": capacity, "api_status": None, "version": None, "submitted": False}
        if not authorized:
            status["blocker"] = "IDENTITY_UNVERIFIED"
            if receipt.exists():
                prior = read(receipt)
                if prior.get("manifest") == document["fingerprint"] and prior.get("submitted"):
                    status = {**prior, "blocker": "IDENTITY_RECHECK_UNAVAILABLE", "api_status_stale": True,
                              "api_observed_at": prior.get("api_observed_at", prior["checked_at"]),
                              "checked_at": c.now(), "verified_concurrency": None}
        elif document.get("window") and not receipt.exists() and time.time() >= document["window"]["deadline_epoch"]:
            status.update(status="WINDOW_ENDED", blocker="CAMPAIGN_DEADLINE")
        elif not item["job"]:
            status["blocker"] = "MIXED_CANDIDATE_REQUIRED"
        else:
            job = item["job"]
            prior = read(receipt) if receipt.exists() else {}
            with c.locked():
                registry = c.registry()
                row = registry["jobs"].get(job, {})
                try:
                    if row.get("version") and row["status"] not in TERMINAL:
                        c.status(registry, job)
                    if row.get("status") == "CLOUD_SUCCEEDED":
                        if row["planned"].get("parallel_task"):
                            harvest_aux(registry, job)
                        else:
                            c.harvest(registry, job)
                    active = [r for r in registry["jobs"].values() if r.get("owner") == own["owner"] and r["status"] in c.ACTIVE]
                    if not row or row.get("status") == "PLANNED":
                        if document.get("window") and time.time() > document["window"]["deadline_epoch"] - 1800:
                            status.update(status="WINDOW_ENDED", blocker="CAMPAIGN_DEADLINE")
                        elif any(r["status"] in UNCERTAIN for r in active):
                            status["blocker"] = "LAUNCHER_UNRESOLVED_RECEIPT"
                        elif len(active) >= document["per_account_target"]:
                            status["blocker"] = "CAMPAIGN_ACCOUNT_TARGET_REACHED"
                        elif capacity is None and active and (
                                not document.get("window") or active[0]["status"] not in {"QUEUED", "RUNNING"}
                                or active[0]["planned"].get("campaign_deadline") != document["window"]["deadline_epoch"]
                                or active[0]["planned"]["round"].rsplit("-", 1)[0] != item["round"].rsplit("-", 1)[0]):
                            status["blocker"] = "LAUNCHER_UNKNOWN_CAPACITY_WITH_ACTIVE_RECEIPT"
                        elif capacity is not None and len(active) >= capacity:
                            status["blocker"] = "LAUNCHER_OBSERVED_CAPACITY_OCCUPIED"
                        else:
                            if item.get("round"):
                                request = SimpleNamespace(job="j10", campaign="stage5", round=item["round"],
                                                          fingerprint=item["plan_fingerprint"], server_admission=capacity is None)
                                plan, _ = c.planned_job(request, registry)
                                if plan["job"] != job or plan["account"] != account:
                                    raise ValueError("Approved continuation differs from dispatch")
                            else:
                                plan = parallel.plan(package, document["package_sha"], item["workload"], registry)
                                doc = read(WORK / f"plan-stage5-{plan['round']}.json")
                                request = SimpleNamespace(job="j10", campaign="stage5", round=plan["round"],
                                                          fingerprint=doc["fingerprint"], server_admission=capacity is None)
                            c.launch(request, registry)
                except c.DatasetPending:
                    status["blocker"] = "API_PRIVATE_INPUT_PROCESSING"
                except (RuntimeError, ValueError, OSError, KeyError) as error:
                    # Error text can include environment paths or upstream responses.
                    if isinstance(error, RuntimeError) and str(error).startswith("A recent Kaggle notebook"):
                        status["blocker"] = "API_INVENTORY_FOUND_ACTIVE_NOTEBOOK"
                    elif isinstance(error, RuntimeError) and str(error).startswith("Verified GPU budget"):
                        status["blocker"] = "LAUNCHER_GPU_QUOTA_UNAVAILABLE_OR_INSUFFICIENT"
                    else:
                        status["blocker"] = "EXECUTION_ERROR_REQUIRES_OPERATOR_REVIEW"
                    status["error_class"] = type(error).__name__
                row = registry["jobs"].get(job, {})
                if row:
                    status.update(status=row["status"], version=row.get("version"), submitted=bool(row.get("version")),
                                  submission_attempted=bool(row.get("submitted")))
                    if row.get("submitted"):
                        status["api_status"] = {"HARVESTED_VERIFIED": "COMPLETE", "CLOUD_SUCCEEDED": "COMPLETE",
                                                "ARTIFACT_INVALID": "COMPLETE", "PARTIAL": "COMPLETE"}.get(row["status"], row["status"])
                    if row.get("execution"):
                        status["job_fingerprint"] = fingerprint(row["execution"])
                    if row.get("output_sha"):
                        status["output_sha"] = row["output_sha"]
                    try:
                        result = candidate(job, row)
                        if result:
                            status["candidate"] = result
                    except (ValueError, OSError, KeyError):
                        status["blocker"] = "ARTIFACT_REVIEW_REQUIRED"
            if document.get("window") and row.get("version"):
                status["execution_evidence"] = training_evidence(account, row, prior)
        write(receipt, status)
        print(item["session"], status["status"], status.get("blocker", ""), flush=True)
        if authorized and item["job"] and status["status"] not in TERMINAL:
            pending = True
    return pending


def admit_window_sibling(root, document, account):
    if "window" not in document or not verify_owner(account):
        raise RuntimeError("Window admission requires the matching verified execution identity")
    pair = [row for row in document["sessions"] if row["account"] == account]
    while time.time() < document["window"]["deadline_epoch"] - 1800:
        try:
            with c.locked():
                registry = c.registry()
                first, second = sorted(pair, key=lambda row: registry["jobs"][row["job"]]["status"] == "PLANNED")
                primary, existing = (registry["jobs"][row["job"]] for row in (first, second))
                if existing["status"] != "PLANNED":
                    print(second["session"], existing["status"], existing.get("version"), flush=True)
                    return
                if primary.get("version") and primary["status"] in c.ACTIVE:
                    c.status(registry, first["job"])
                if primary["status"] in {"QUEUED", "RUNNING", "HARVESTED_VERIFIED", "CLOUD_SUCCEEDED"}:
                    _, record = slot(account)
                    age = (datetime.now(timezone.utc) - datetime.fromisoformat(record["observed_at"])).total_seconds()
                    capacity = record["concurrency_observed"] if 0 <= age <= 86400 else None
                    if capacity is not None and capacity < 2 and primary["status"] in c.ACTIVE:
                        print(second["session"], "LAUNCHER_OBSERVED_CAPACITY_OCCUPIED", flush=True)
                        return
                    request = SimpleNamespace(job="j10", campaign="stage5", round=second["round"],
                                              fingerprint=second["plan_fingerprint"], server_admission=capacity is None)
                    c.launch(request, registry)
                    return
                if primary["status"] in TERMINAL or primary["status"] == "SUBMIT_UNKNOWN":
                    print(second["session"], "FIRST_SESSION_RECONCILIATION_REQUIRED", flush=True)
                    return
        except c.DatasetPending:
            print(second["session"], "API_PRIVATE_INPUT_PROCESSING", flush=True)
        except RuntimeError as exc:
            if not str(exc).startswith("Registry locked."):
                raise
        time.sleep(31 + c.SLOTS.index(account))


def window_assets(document, registry):
    for row in document["sessions"]:
        account = row["account"]
        _, own = slot(account)
        if not own or not own.get("authorized"):
            continue
        job = registry["jobs"][row["job"]]["planned"]
        base = c.ROOT / job["runtime_path"]
        parent, _ = c.parent_checkpoint(registry, job["parent_job"])
        verify(parent / "COMPLETE.json", job["parent_sha"])
        archive = parent / "parent.zip"
        if not archive.is_file():
            raise FileNotFoundError("Parallel staging requires the existing verified parent transport archive")
        # The same archive will be checked member-by-member again before notebook upload.
        assets = [
            ("runtime", "runtime.zip", base / "runtime.zip", job["runtime_sha"]),
            ("source", "model.zip", base / "model.zip", job["source_sha"]),
            ("long-parent", "parent.zip", archive, sha(archive)),
        ]
        if job.get("traffic_teacher_sha"):
            teacher, _ = c.parent_checkpoint(registry, job["traffic_teacher_job"])
            assets.append(("teacher", "teacher.zip", teacher / "model.zip", job["traffic_teacher_sha"]))
        for label, name, source, digest in assets:
            ref = f"{own['owner']}/s5-{label}-{digest[:20]}-bin"
            yield account, ref, label, name, source, digest


def reserve_window_inputs(root, document):
    if "window" not in document:
        raise ValueError("Parallel input staging requires a timeboxed campaign")
    jobs = {}
    with c.locked():
        registry = c.registry()
        for account, ref, label, name, source, digest in window_assets(document, registry):
            existing = registry["uploads"].get(ref)
            if existing and existing.get("stage_manifest") != document["fingerprint"]:
                continue
            jobs.setdefault(account, {})[ref] = dict(label=label, name=name, source=str(source), sha=digest)
            registry["uploads"].setdefault(ref, {"status": "WINDOW_INPUT_STAGING", "alias": account,
                "sha": digest, "stage_manifest": document["fingerprint"], "at": c.now()})
        c.persist(registry)
        for account, assets in jobs.items():
            path = root / "input-staging" / account
            path.mkdir(parents=True, exist_ok=True)
            write(path / "assets.json", {"manifest": document["fingerprint"], "assets": assets})
    print({account: len(assets) for account, assets in jobs.items()}, flush=True)


def stage_window_inputs(root, document, account):
    if not verify_owner(account):
        raise RuntimeError("Parallel input staging requires the matching execution identity")
    folder = root / "input-staging" / account
    journal = folder / "journal.json"
    approved = read(folder / "assets.json")
    if approved["manifest"] != document["fingerprint"]:
        raise ValueError("Input staging belongs to a different campaign")
    receipts = read(journal) if journal.exists() else {"uploads": {}}
    _, own = slot(account)
    for ref, asset in approved["assets"].items():
        row = c.registry()["uploads"][ref]
        if row["status"] == "ACCEPTED":
            continue
        if (row.get("stage_manifest"), row["alias"], row["sha"], ref.split("/")[0]) != (
                document["fingerprint"], account, asset["sha"], own["owner"]):
            raise ValueError("Input staging no longer owns the upload reservation")
        source = Path(asset["source"]).resolve()
        if not source.is_relative_to(WORK.resolve()):
            raise ValueError("Input staging file escapes the campaign workspace")
        while time.time() < document["window"]["deadline_epoch"] - 1800:
            try:
                c.upload(receipts, account, own["owner"], asset["label"], asset["name"], source, asset["sha"], journal=journal)
                break
            except c.DatasetPending:
                time.sleep(15)
            except RuntimeError as exc:
                if not str(exc).startswith(("Private version 1 could not be retrieved",
                                            "Dataset API did not verify private ACL for version 1")):
                    raise
                if receipts["uploads"].get(ref, {}).get("status") != "ACCEPTED":
                    raise
                # Retry the read of the same accepted private version, never its upload.
                time.sleep(60)
        completed = receipts["uploads"].get(ref, {})
        if completed.get("version") != 1 or not completed.get("remote_sha_verified_at"):
            print(asset["label"], "INPUT_STAGING_INCOMPLETE", flush=True)
            continue
        while True:
            try:
                with c.locked():
                    registry = c.registry()
                    prior = registry["uploads"][ref]
                    if prior.get("stage_manifest") != document["fingerprint"] or prior["sha"] != completed["sha"]:
                        raise ValueError("Input reservation changed before verified publication")
                    registry["uploads"][ref] = completed
                    c.persist(registry)
                break
            except RuntimeError as exc:
                if not str(exc).startswith("Registry locked."):
                    raise
                time.sleep(3 + c.SLOTS.index(account))
        print(asset["label"], "PRIVATE_VERSION_1_SHA_VERIFIED", flush=True)


def observe_window(root, document, account, watch):
    if "window" not in document or not verify_owner(account):
        raise RuntimeError("Window observation requires the matching verified execution identity")
    folder = root / "observations" / account
    folder.mkdir(parents=True, exist_ok=True)
    while True:
        pending = False
        registry = c.registry()
        for row in document["sessions"]:
            if row["account"] != account:
                continue
            entry = registry["jobs"][row["job"]]
            if not entry.get("version"):
                pending |= entry["status"] == "PLANNED" or entry["status"] in UNCERTAIN
                continue
            path = folder / f"{row['session']}.json"
            prior = read(path) if path.exists() else {}
            version = entry["version"]
            status = entry["status"]
            if status in c.ACTIVE:
                code, response = c.cli(account, ["kernels", "status", f"{entry['ref']}/{version}"], timeout=20)
                status = c.kernel_state(code, response)
            pending |= status in c.ACTIVE
            record = {"manifest": document["fingerprint"], "session": row["session"], "account": account,
                      "job": row["job"], "version": version, "checked_at": c.now()}
            entry = {**entry, "status": status}
            record.update(api_status="COMPLETE" if status in {"HARVESTED_VERIFIED", "CLOUD_SUCCEEDED", "PARTIAL", "ARTIFACT_INVALID"} else status,
                          execution_evidence=training_evidence(account, entry, prior))
            write(path, record)
            print(row["session"], version, record["api_status"], record["execution_evidence"], flush=True)
        if not watch or not pending or time.time() > document["window"]["deadline_epoch"] + 900:
            return
        time.sleep(120 + c.SLOTS.index(account))


def report(root, document):
    rows = []
    for row in job_queue(root, document):
        receipt = root / "receipts" / row["account"] / f"{row['session']}.json"
        status = read(receipt) if receipt.exists() else {**row, "status": "PLANNED", "api_status": None,
                    "submitted": False, "blocker": "OPERATOR_RECEIPT_MISSING" if row["job"] else
                    "OPERATOR_RECEIPT_MISSING_AND_MIXED_CANDIDATE_REQUIRED"}
        if receipt.exists() and (any(status[key] != value for key, value in row.items())
                                 or status["manifest"] != document["fingerprint"]):
            raise ValueError("Receipt does not belong to the pinned session")
        observation = root / "observations" / row["account"] / f"{row['session']}.json"
        if observation.is_file():
            observed = read(observation)
            if (observed["manifest"] != document["fingerprint"]
                    or any(observed[k] != row[k] for k in ("job", "account", "session"))
                    or status.get("version") not in (None, observed["version"])):
                raise ValueError("Observation does not match the pinned session and version")
            if observed["checked_at"]["utc"] >= status.get("checked_at", {}).get("utc", ""):
                status.update(version=observed["version"], api_status=observed["api_status"], api_status_stale=False,
                              submitted=True, submission_attempted=True, blocker=None,
                              api_observed_at=observed["checked_at"])
            if observed["execution_evidence"].get("transitions", 0) >= status.get("execution_evidence", {}).get("transitions", 0):
                status["execution_evidence"] = observed["execution_evidence"]
        rows.append(status)
    totals = {"submitted": sum(row["submitted"] for row in rows),
              "submission_attempts": sum(row.get("submission_attempted", False) for row in rows),
              "queued": sum(row["api_status"] == "QUEUED" and not row.get("api_status_stale") for row in rows),
              "running": sum(row["api_status"] == "RUNNING" and not row.get("api_status_stale") for row in rows),
              "stale_active": sum(row["api_status"] in c.ACTIVE and row.get("api_status_stale", False) for row in rows),
              "complete": sum(row["api_status"] == "COMPLETE" for row in rows),
              "collection_started": sum(row.get("execution_evidence", {}).get("transitions", 0) > 0 for row in rows),
              "updates_started": sum(row.get("execution_evidence", {}).get("updates", 0) > 0 for row in rows),
              "not_deployed": sum(not row["submitted"] for row in rows)}
    candidates = [{**r["candidate"], "account": r["account"], "job": r["job"]} for r in rows if r.get("candidate")]
    eligible = [r for r in candidates if r["panels"].get("Medium", {}).get("clean_successes", 0) > 0
                and r["panels"].get("Medium", {}).get("safety_failures", 1) == 0
                and r["panels"].get("Easy", {}).get("safety_failures", 1) == 0]
    best = max(eligible, key=lambda r: (r["panels"]["Medium"]["clean_successes"], r["panels"]["Easy"]["clean_successes"]), default=None)
    continuation = None if best is None else {"parent_job": best["job"], "state_sha": best["state_sha"],
                                             "account": best["account"], "status": "READY_FOR_OPERATOR_SELECTION"}
    write(root / "status.json", {"sessions": rows, "totals": totals, "multitier_shortlist": best,
                                 "continuation": continuation, "checked_at": c.now()})
    stage = root / "status.csv.tmp"
    with stage.open("w", encoding="utf-8", newline="") as f:
        table = csv.DictWriter(f, fieldnames=("session", "account", "workload", "job", "version", "parent",
                                             "device", "budget", "api_status", "evidence_source", "transitions",
                                             "updates", "checkpoint", "blocker"), extrasaction="ignore")
        table.writeheader()
        for row in rows:
            evidence = row.get("execution_evidence", {})
            table.writerow({**row, "evidence_source": evidence.get("source"),
                            "transitions": evidence.get("transitions") or None,
                            "updates": evidence.get("updates") or None})
    stage.replace(root / "status.csv")
    print(totals)
    return totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("init", "agent", "report", "enqueue", "prepare-window", "admit-window-sibling", "observe-window", "reserve-window-inputs", "stage-window-inputs", "repair-localization"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--parent")
    parser.add_argument("--session")
    parser.add_argument("--parent-step", type=int)
    parser.add_argument("--round")
    parser.add_argument("--transitions", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--max-seconds", type=int)
    parser.add_argument("--phase", choices=("medium", "hard", "extreme"))
    parser.add_argument("--traffic-guidance", action="store_true")
    parser.add_argument("--traffic-teacher", action="store_true")
    parser.add_argument("--autonomous-traffic", action="store_true")
    parser.add_argument("--campaign-deadline", type=int)
    args = parser.parse_args()
    if args.command in {"init", "agent", "prepare-window"} and not args.package:
        parser.error("--package is required")
    if args.command == "init":
        init(args.root, args.package)
        return
    if args.command == "prepare-window":
        if not args.parent:
            parser.error("prepare-window requires a verified Medium parent")
        prepare_window(args.root, args.package, args.parent)
        return
    document = manifest(args.root)
    if args.command == "enqueue":
        if not (args.parent and args.round and args.transitions):
            parser.error("enqueue requires --parent, --round and --transitions")
        enqueue_continuation(args.root, document, args)
        return
    if args.command == "report":
        while True:
            report(args.root, document)
            if not args.watch or "window" not in document or time.time() >= document["window"]["deadline_epoch"] + 900:
                return
            time.sleep(120)
    if args.command == "repair-localization":
        if args.watch:
            if not args.package:
                parser.error("Repair watch requires the deployment package")
            account = operator_slot()
            with agent_lock(args.root / "repairs" / account):
                watch_localization_repairs(args.root, args.package, document, account)
        else:
            repair_window_localization(args.root, document, args.session)
        return
    if args.command == "reserve-window-inputs":
        reserve_window_inputs(args.root, document)
        return
    account = operator_slot()
    if args.command == "stage-window-inputs":
        with agent_lock(args.root / "input-staging" / account):
            stage_window_inputs(args.root, document, account)
        return
    if args.command == "observe-window":
        with agent_lock(args.root / "observations" / account):
            observe_window(args.root, document, account, args.watch)
        return
    if args.command == "admit-window-sibling":
        with agent_lock(args.root / "dispatch" / account):
            admit_window_sibling(args.root, document, account)
        return
    with agent_lock(args.root / "receipts" / account):
        while True:
            if "window" in document and time.time() > document["window"]["deadline_epoch"] + 3600:
                return
            try:
                pending = run_once(args.root, args.package, document, account, args.session)
            except RuntimeError as exc:
                if not args.watch or not str(exc).startswith("Registry locked."):
                    raise
                time.sleep(17 + c.SLOTS.index(account))
                continue
            if not args.watch or not pending:
                break
            time.sleep(60)


if __name__ == "__main__":
    main()
