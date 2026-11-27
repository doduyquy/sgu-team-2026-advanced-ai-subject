"""Read-only, version-pinned Stage 5 harvest with small artifacts first."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from cloud.kaggle import campaign, control as c
from cloud.kaggle.checkpoint import STATE_FILES
from cloud.kaggle.common import WORK, read, sha, verify, write
from cloud.kaggle.credentials import SLOTS, operator_slot

PERIODIC = r"Periodic checkpoint (step-d+) index_sha256=([0-9a-f]{64})"
SCREENING = r"Screening policy (step-d+) marker_sha256=([0-9a-f]{64})"


class ArtifactUnavailable(RuntimeError):
    """Expected artifact absent from versioned transport."""


def archive_root():
    return Path(os.environ["LOCALAPPDATA"]) / "XeTuHanh" / "Stage5CampaignHarvest"


def file_catalog(account, ref, version):
    code, raw = c.cli(account, ["kernels", "files", "--format", "csv", "--page-size", "200", f"{ref}/{version}"],
                      timeout=30)
    lines = raw.splitlines()
    start = next((i for i, line in enumerate(lines) if line.lower().startswith("name,")), None)
    if code or start is None:
        return None
    names = [Path(row["name"]).name for row in csv.DictReader(lines[start:]) if row.get("name")]
    return {"listed_files": len(names), "may_be_paginated": len(names) >= 200,
            "screening_markers": names.count("POLICY.json"), "state_indexes": names.count("INDEX.json"),
            "execution_envelope": "execution.json" in names}


def fetch_checkpoint(entry, remote, target, expected, *, full=False):
    with TemporaryDirectory(dir=target.parent, prefix="v3-parts-") as interim:
        staging = Path(interim)
        def fetch(name):
            try:
                c.fetch_output(entry, remote, staging, name)
            except RuntimeError:
                raise ArtifactUnavailable(name) from None

        fetch("INDEX.json")
        verify(staging / "INDEX.json", expected)
        index = read(staging / "INDEX.json")
        if index["schema"] != 1 or set(index["files"]) != set(STATE_FILES):
            raise ValueError("Unexpected checkpoint index")
        result = staging / "verified"
        result.mkdir()
        names = STATE_FILES if full else ("COMPLETE.json", "model.zip", "metrics.json")
        for name in names:
            chunks = index["files"][name]
            limit_mb = 700 if name == "replay.pkl" else 100
            if not chunks or sum(shard["bytes"] for shard in chunks) > limit_mb * 1024**2:
                raise ValueError("Unexpected artifact size")
            with (result / name).open("xb") as out:
                for i, shard in enumerate(chunks):
                    part = f"{name}.{i:06d}.bin"
                    fetch(part)
                    path = staging / part
                    verify(path, shard["sha256"])
                    if path.stat().st_size != shard["bytes"]:
                        raise ValueError("Size mismatch")
                    with path.open("rb") as source:
                        shutil.copyfileobj(source, out, 1024**2)
                    path.unlink()
        digest = index["state_sha256"]
        verify(result / "COMPLETE.json", digest)
        marker = read(result / "COMPLETE.json")
        if marker["schema"] != 1 or marker["job"] != entry["planned"]["job"]:
            raise ValueError("Checkpoint job differs from receipt")
        for name in names:
            if name == "COMPLETE.json":
                continue
            verify(result / name, marker["files"][name])
        shutil.copyfile(staging / "INDEX.json", result / "INDEX.json")
        result.rename(target)
    return {"path": str(target), "sha256": digest, "policy_only": not full, "result": marker["result"]}


def one(receipt, row, entry, mode):
    job, version = row["job"], entry.get("version")
    journal = receipt / "receipt.json"
    saved = read(journal) if journal.is_file() else {
        "session": row["session"], "job": job, "version": version, "artifacts": {}}
    final = saved["artifacts"].get("final/state", {})
    if final.get("path") and (mode == "small" or not final.get("policy_only")):
        return
    saved.pop(f"{mode}_error", None)
    if not version:
        saved["api_status"] = "NO_VERSION"
        write(journal, saved)
        return
    account = row["account"]
    code, value = c.cli(account, ["kernels", "status", f'{entry["ref"]}/{version}'], timeout=30)
    status = c.kernel_state(code, value)
    saved.update(api_status=status, at=c.now(), scan_complete=False)
    write(journal, saved)
    if status in c.ACTIVE:
        return
    remote_version = c.current_private_kernel_version(account, entry["ref"])
    if remote_version != version:
        saved["version_check"] = "VERSION_UNVERIFIED" if remote_version is None else "VERSION_CHANGED"
        write(journal, saved)
        return
    code, logs = c.cli(account, ["kernels", "logs", f'{entry["ref"]}/{version}'], timeout=35)
    if code == 0:
        saved["registered_markers"] = {"periodic": dict(re.findall(PERIODIC, logs)),
                                       "screening": dict(re.findall(SCREENING, logs))}
        marks = re.findall(r"(?:TRAINING_ACTIVE|COMPLETE|PARTIAL) transitions=(d+) updates=(d+)", logs)
        saved["last_log_steps"] = [max((int(x[i]) for x in marks), default=0) for i in (0, 1)]
    marks = saved.get("registered_markers", {})
    if not marks and not saved.get("file_catalog"):
        saved["file_catalog"] = file_catalog(account, entry["ref"], version)
    write(journal, saved)
    specs = []
    if status in ("CLOUD_SUCCEEDED", "HARVESTED_VERIFIED", "PARTIAL", "ARTIFACT_INVALID"):
        if mode == "small":
            try:
                if not (receipt / "execution.json").is_file():
                    c.fetch_output(entry, job, receipt, "execution.json")
                envelope = read(receipt / "execution.json")
                if envelope["job"] != entry["execution"]:
                    raise ValueError("Execution envelope differs from registered job")
                specs.append(("final/state", f"{job}/final", envelope["index_sha256"]))
            except (RuntimeError, OSError, ValueError, KeyError) as exc:
                saved["final_index_error"] = type(exc).__name__
        else:
            existing = saved["artifacts"].get("final/state", {})
            if existing.get("path") and existing.get("policy_only"):
                specs.append(("final/state", f"{job}/final", sha(Path(existing["path"]) / "INDEX.json")))
    for name, digest in sorted(marks.get("periodic", {}).items(), reverse=True)[:1]:
        specs.append(("periodic/" + name, f"{job}/periodic/{name}", digest))
    for key, remote, digest in specs:
        previous = saved["artifacts"].get(key, {})
        if previous.get("path") and (mode == "small" or not previous.get("policy_only")):
            continue
        marker = receipt / key.replace("/", "-")
        try:
            if mode == "small":
                item = fetch_checkpoint(entry, remote, marker, digest)
            else:
                if shutil.disk_usage(receipt).free < 11 * 1024**3:
                    saved["full_state_blocker"] = "WINDOWS_DISK_RESERVE"
                    continue
                target = marker.with_name(marker.name + "-complete")
                item = fetch_checkpoint(entry, remote, target, digest, full=True)
        except (RuntimeError, OSError, ValueError, KeyError) as exc:
            saved["artifacts"][key] = {**previous, "status": "UNAVAILABLE_OR_UNVERIFIED", "failure": type(exc).__name__}
            if isinstance(exc, ArtifactUnavailable):
                saved["artifacts"][key]["failed_file"] = exc.args[0]
            write(journal, saved)
            continue
        saved["artifacts"][key] = item
        write(journal, saved)
        if key.startswith("final/"):
            break
    saved["scan_complete"] = True
    write(journal, saved)


def own(mode):
    account = operator_slot()
    registry = c.registry()
    for extension in ("", "-extension"):
        root = WORK / f"central12-second{extension}"
        doc = campaign.manifest(root)
        base = archive_root() / f'{doc["window"]["deadline_epoch"]}{extension}'
        destination = base.with_name(base.name + ("-catalog" if mode == "catalog" else "-selective"))
        eligible = [row for row in campaign.job_queue(root, doc) if row["account"] == account]
        for row in eligible:
            entry = registry["jobs"][row["job"]]
            version = entry.get("version")
            key = f"{row['job']}-v{version}"
            original = base / account / key / "receipt.json"
            if mode != "catalog" and original.exists() and any(
                item.get("path") and name.startswith("final/")
                for name, item in read(original)["artifacts"].items()):
                continue
            receipt = destination / account / key
            receipt.mkdir(exist_ok=True, parents=True)
            try:
                if mode == "catalog":
                    visible = None
                    if version and c.current_private_kernel_version(account, entry["ref"]) == version:
                        visible = file_catalog(account, entry["ref"], version)
                    write(receipt / "receipt.json", {"session": row["session"], "job": row["job"],
                          "version": version, "at": c.now(), "artifacts": {}, "file_catalog": visible})
                else:
                    one(receipt, row, entry, mode)
            except (RuntimeError, ValueError, OSError, KeyError) as exc:
                # Capture only an error class. CLI or SDK exception text is not a receipt field.
                journal = receipt / "receipt.json"
                current = read(journal) if journal.exists() else {
                    "session": row["session"], "job": row["job"], "version": version, "artifacts": {}}
                current[f"{mode}_error"] = type(exc).__name__
                write(journal, current)
            print(account, row["session"], mode, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("small", "full", "catalog"), default="small")
    parser.add_argument("--only-slot", action="store_true")
    parser.add_argument("--slots", nargs="*", choices=SLOTS, default=list(SLOTS))
    args = parser.parse_args()
    if args.only_slot:
        own(args.mode)
        return

    def launch(slot):
        env = {**os.environ, "STAGE5_KAGGLE_OPERATOR_SLOT": slot}
        result = subprocess.run([sys.executable, "-m", __spec__.name, "--only-slot", "--mode", args.mode],
                                env=env, check=False)
        return slot, result.returncode

    with ThreadPoolExecutor(max_workers=3 if args.mode == "small" else 2) as slots:
        print(list(slots.map(launch, args.slots)), flush=True)


if __name__ == "__main__":
    main()
