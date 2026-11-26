"""Credential-isolated Stage 5 campaign queue and receipt transport."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
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
            round_id = f"win12-{session.lower()}"
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


def run_once(root, package, document, account):
    verify(package, document["package_sha"])
    _, own = slot(account)
    authorized = bool(own and verify_owner(account))
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(own["observed_at"])).total_seconds() if authorized else None
    capacity = own["concurrency_observed"] if age is not None and 0 <= age <= 86400 else None
    folder = root / "receipts" / account
    pending = False
    for item in job_queue(root, document):
        if item["account"] != account:
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
                        elif capacity is None and active:
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
        write(receipt, status)
        print(item["session"], status["status"], status.get("blocker", ""), flush=True)
        if authorized and item["job"] and status["status"] not in TERMINAL:
            pending = True
    return pending


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
        rows.append(status)
    totals = {"submitted": sum(row["submitted"] for row in rows),
              "submission_attempts": sum(row.get("submission_attempted", False) for row in rows),
              "queued": sum(row["api_status"] == "QUEUED" and not row.get("api_status_stale") for row in rows),
              "running": sum(row["api_status"] == "RUNNING" and not row.get("api_status_stale") for row in rows),
              "stale_active": sum(row["api_status"] in c.ACTIVE and row.get("api_status_stale", False) for row in rows),
              "complete": sum(row["api_status"] == "COMPLETE" for row in rows),
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
        table = csv.DictWriter(f, fieldnames=("session", "account", "workload", "job", "device", "budget",
                                             "api_status", "blocker"), extrasaction="ignore")
        table.writeheader()
        table.writerows(rows)
    stage.replace(root / "status.csv")
    print(totals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("init", "agent", "report", "enqueue", "prepare-window"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--parent")
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
        report(args.root, document)
        return
    account = operator_slot()
    with agent_lock(args.root / "receipts" / account):
        while True:
            try:
                pending = run_once(args.root, args.package, document, account)
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
