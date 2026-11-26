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
TERMINAL = {"HARVESTED_VERIFIED", "ARTIFACT_INVALID", "PARTIAL", "FAILED", "CAPACITY_REJECTED", "CANCELLED"}
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


def manifest(root):
    document = read(root / "campaign.json")
    fields = {k: v for k, v in document.items() if k != "fingerprint"}
    if (document["fingerprint"] != fingerprint(fields) or document["sessions"] != sessions()
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
    for item in document["sessions"]:
        if item["account"] != account:
            continue
        status = {**item, "manifest": document["fingerprint"], "status": "PLANNED", "checked_at": c.now(),
                  "verified_concurrency": capacity, "api_status": None, "version": None, "submitted": False}
        if not authorized:
            status["blocker"] = "IDENTITY_UNVERIFIED"
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
                        if any(r["status"] in UNCERTAIN for r in active):
                            status["blocker"] = "LAUNCHER_UNRESOLVED_RECEIPT"
                        elif len(active) >= document["per_account_target"]:
                            status["blocker"] = "CAMPAIGN_ACCOUNT_TARGET_REACHED"
                        elif capacity is None and active:
                            status["blocker"] = "LAUNCHER_UNKNOWN_CAPACITY_WITH_ACTIVE_RECEIPT"
                        elif capacity is not None and len(active) >= capacity:
                            status["blocker"] = "LAUNCHER_OBSERVED_CAPACITY_OCCUPIED"
                        else:
                            plan = parallel.plan(package, document["package_sha"], item["workload"], registry)
                            doc = read(WORK / f"plan-stage5-{plan['round']}.json")
                            c.launch(SimpleNamespace(job="j10", campaign="stage5", round=plan["round"],
                                                     fingerprint=doc["fingerprint"], server_admission=capacity is None), registry)
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
        write(folder / f"{item['session']}.json", status)
        print(item["session"], status["status"], status.get("blocker", ""), flush=True)
        if authorized and item["job"] and status["status"] not in TERMINAL:
            pending = True
    return pending


def report(root, document):
    rows = []
    for row in document["sessions"]:
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
              "queued": sum(row["api_status"] == "QUEUED" for row in rows),
              "running": sum(row["api_status"] == "RUNNING" for row in rows),
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
    parser.add_argument("command", choices=("init", "agent", "report"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    if args.command in {"init", "agent"} and not args.package:
        parser.error("--package is required")
    if args.command == "init":
        init(args.root, args.package)
        return
    document = manifest(args.root)
    if args.command == "report":
        report(args.root, document)
        return
    account = operator_slot()
    with agent_lock(args.root / "receipts" / account):
        while True:
            pending = run_once(args.root, args.package, document, account)
            if not args.watch or not pending:
                break
            time.sleep(60)


if __name__ == "__main__":
    main()
