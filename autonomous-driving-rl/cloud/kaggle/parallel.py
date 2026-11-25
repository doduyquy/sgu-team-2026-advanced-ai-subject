"""Ship verified Stage 5 inputs and plan independent operator-owned learners."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import zipfile

from cloud.kaggle import control as c
from cloud.kaggle.common import ROOT, SOURCE_SHA, WORK, fingerprint, read, sha, verify, write
from cloud.kaggle.credentials import operator_slot

CHAMPION = "s5-e2938ff0e622-j11-a1"
STATE = "253fe755507ca860f93a0a881e09d795d9c4d8920d5bd621e69f08947b9de591"
MODEL = "f82ca94d27ebd9fc3468081e20631536c959990a708398f5474bebee986e6d07"
WORKLOADS = {
    "medium": ("account_01", 10000, {"phase": "medium", "guided_curriculum_from": "easy"}),
    "main": ("account_02", 10000, {}),
    "steering": ("account_03", 5000, {"guided_rollout_axis": "steering"}),
    "throttle": ("account_04", 5000, {"guided_rollout_axis": "throttle_brake"}),
    "retention": ("account_05", 10000, {}),
    "easy-stable": ("account_08", 10000, {"guided_actor_lr_scale": 0.25}),
}
INPUTS = {"runtime.zip", "model.zip", "parent.zip"}


def export(out, registry):
    parent, digest = c.parent_checkpoint(registry, CHAMPION)
    if digest != STATE:
        raise ValueError("The registered champion is immutable")
    verify(parent / "COMPLETE.json", STATE)
    marker = read(parent / "COMPLETE.json")
    verify(parent / "model.zip", MODEL)
    bundle = c.package()
    root = ROOT / bundle["path"]
    anchor_runtime = registry["jobs"][CHAMPION]["planned"]["validated_runtime_sha"]
    gates = {}
    for gate in ("G2", "G3"):
        job, entry = next((key, row) for key, row in registry["jobs"].items()
                          if row["status"] == "HARVESTED_VERIFIED" and row.get("gates", {}).get(gate) == "PASS"
                          and row["execution"]["campaign"] == "stage5" and row["execution"]["runtime_sha"] == anchor_runtime)
        gates[gate] = {"job": job, "state_sha": entry["state_sha"], "version": entry["output_version"]}
    with TemporaryDirectory(dir=WORK) as temp:
        archive = Path(temp) / "parent.zip"
        with zipfile.ZipFile(archive, "x") as outzip:
            for name in sorted(c.STATE_FILES):
                verify(parent / name, STATE if name == "COMPLETE.json" else marker["files"][name])
                outzip.write(parent / name, name)
        paths = {"runtime.zip": root / "runtime.zip", "model.zip": root / "model.zip", "parent.zip": archive}
        for name, label in (("runtime.zip", "runtime"), ("parent.zip", "long-parent")):
            c.audit_zip(paths[name], label)
        c.audit_zip(paths["model.zip"], "source")
        manifest = {"schema": 1, "parent_sha": STATE, "model_sha": MODEL, "gates": gates,
                    "validated_runtime_sha": anchor_runtime, "template_sha": bundle["template_sha"],
                    "files": {name: sha(path) for name, path in paths.items()}}
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "x") as transfer:
            transfer.writestr("MANIFEST.json", json.dumps(manifest, sort_keys=True))
            for name, path in paths.items():
                transfer.write(path, name)
    print(sha(out))


def extract(directory, bundle, names):
    for name in names:
        with bundle.open(name) as source, (directory / name).open("xb") as dest:
            shutil.copyfileobj(source, dest)


def install(package, digest, registry):
    verify(package, digest)
    path = WORK / f"parallel-{digest[:20]}"
    with zipfile.ZipFile(package) as bundle, TemporaryDirectory(dir=WORK) as scratch:
        if set(bundle.namelist()) != INPUTS | {"MANIFEST.json"} or len(bundle.infolist()) != 4:
            raise ValueError("Unexpected deployment package members")
        if sum(item.file_size for item in bundle.infolist()) > c.LONG_PARENT_LIMIT + 40_000_000:
            raise ValueError("Deployment package exceeds the input limit")
        manifest = json.loads(bundle.read("MANIFEST.json"))
        if (manifest["schema"] != 1 or manifest["parent_sha"] != STATE or manifest["model_sha"] != MODEL
                or manifest["files"]["model.zip"] != SOURCE_SHA or set(manifest["files"]) != INPUTS
                or set(manifest["gates"]) != {"G2", "G3"}):
            raise ValueError("Deployment package does not match the registered champion or gates")
        staging = Path(scratch)
        extract(staging, bundle, INPUTS)
        for name, digest in manifest["files"].items():
            verify(staging / name, digest)
        c.audit_zip(staging / "runtime.zip", "runtime")
        c.audit_zip(staging / "parent.zip", "long-parent")
        c.audit_zip(staging / "model.zip", "source")
        with zipfile.ZipFile(staging / "runtime.zip") as runtime:
            if hashlib.sha256(runtime.read("cloud/kaggle/kernel_template.py.txt")).hexdigest() != manifest["template_sha"]:
                raise ValueError("Runtime differs from the pinned notebook template")
        parent = WORK / "harvest" / CHAMPION
        parent_staging = staging / "state"
        parent_staging.mkdir()
        with zipfile.ZipFile(staging / "parent.zip") as archive:
            extract(parent_staging, archive, c.STATE_FILES)
        verify(parent_staging / "COMPLETE.json", STATE)
        marker = read(parent_staging / "COMPLETE.json")
        if marker["files"]["model.zip"] != MODEL or set(marker["files"]) != c.STATE_FILES - {"COMPLETE.json"}:
            raise ValueError("Incorrect champion state members")
        for name, digest in marker["files"].items():
            verify(parent_staging / name, digest)
            if parent.exists():
                verify(parent / name, digest)
        if parent.exists():
            verify(parent / "COMPLETE.json", STATE)
        else:
            parent.parent.mkdir(parents=True, exist_ok=True)
            parent_staging.rename(parent)
        if not path.exists():
            path.mkdir()
            for name in ("model.zip", "runtime.zip"):
                (staging / name).replace(path / name)
        for name in ("model.zip", "runtime.zip"):
            verify(path / name, manifest["files"][name])
    registry["jobs"].setdefault(CHAMPION, {"status": "HARVESTED_VERIFIED", "state_sha": STATE,
                                           "planned": {"account": "account_01", "job": CHAMPION, "campaign": "stage5"},
                                           "imported_evidence": True})
    # The copied receipts attest historical gates. They grant no account access or cloud job ownership.
    for gate, evidence in manifest["gates"].items():
        registry["jobs"].setdefault(evidence["job"], {"status": "HARVESTED_VERIFIED", "gates": {gate: "PASS"},
            "state_sha": evidence["state_sha"], "output_version": evidence["version"], "imported_evidence": True,
            "planned": {"account": "account_01", "job": evidence["job"], "campaign": "stage5"},
            "execution": {"campaign": "stage5", "runtime_sha": manifest["validated_runtime_sha"]}})
    return manifest, path, marker


def plan(package, digest, workload, registry):
    account, budget, profile = WORKLOADS[workload]
    if operator_slot() != account:
        raise ValueError("This workload belongs to a different independent operator")
    bundle, path, champion = install(package, digest, registry)
    round_id = f"par1-{workload}"
    runtime_sha = bundle["files"]["runtime.zip"]
    job = {"job": "s5-" + fingerprint(["stage5", round_id, account])[:12] + "-j10-a1",
           "campaign": "stage5", "round": round_id, "parallel_workload": workload,
           "account": account, "role": "easy_resume", "device": "cpu", "workers": 2,
           "capacity": 24576 if budget == 10000 else 20480, "phase": "easy", "transitions": budget,
           "parent_job": CHAMPION, "parent_sha": STATE, "start_transition": 5000,
           "parent_archive_limit": c.LONG_PARENT_LIMIT, "warm_model_sha": champion["identity"]["warm_source"],
           "source_sha": SOURCE_SHA, "runtime_sha": runtime_sha, "runtime_path": str(path.relative_to(ROOT)),
           "template_sha": bundle["template_sha"], "validated_runtime_sha": bundle["validated_runtime_sha"],
           "sharded_output": True, "evaluate_easy": True, "policy_every": 2500, "checkpoint_every": 10000,
           "learning_starts": 0, "measure_optimizer": True, "max_seconds": 2400, "export_reserve": 480, **profile}
    if workload != "retention":
        job["guided_gradient_steps"] = 1
    job["guided_resume"] = True
    c.require_gate(registry, job)
    c.save_plan(f"stage5-{round_id}", {"j10": job}, {}, registry)
    return job


def run_companion(args, registry):
    from cloud.kaggle import replay, runtime

    expected = "account_06" if args.command == "collect" else "account_07"
    if operator_slot() != expected:
        raise ValueError("This workload belongs to a different independent operator")
    _, _, state = install(args.package, args.sha, registry)
    parent = WORK / "harvest" / CHAMPION
    if args.command == "evaluate":
        verify(args.model, args.model_sha)
        runtime.evaluate(args)
        return
    marker = args.out / "GUIDED.json"
    if not marker.exists():
        args.out.mkdir(parents=True, exist_ok=False)
        snapshot = args.out / "snapshot"
        digest = replay.publish(parent / "model.zip", MODEL, state["identity"], 0, snapshot)
        by_family = {row["sequence"]: row["geometry_id"] for row in runtime.pool("easy")}
        chunks = []
        for index, geometry in enumerate(sorted(by_family.values())):
            target = args.out / f"chunk-{index:02}"
            replay.collect(SimpleNamespace(snapshot=snapshot, snapshot_sha=digest, out=target, geometry=geometry,
                                           seed=31101 + index, rows=replay.MAX_ROWS, stream=106, sequence=index))
            chunks.append(sha(target / "CHUNK.json"))
        write(marker, {"snapshot_sha": digest, "chunks": chunks})
    if args.publish:
        if not args.out.resolve().is_relative_to(WORK.resolve()):
            raise ValueError("Private replay publishing requires campaign workspace output")
        evidence = read(marker)
        if len(evidence["chunks"]) != 3:
            raise ValueError("Expected one complete guided episode from each family")
        snapshot = args.out / "snapshot"
        replay.check_snapshot(snapshot, evidence["snapshot_sha"], state["identity"])
        verify(snapshot / "policy.zip", MODEL)
        names = ["GUIDED.json", "snapshot/SNAPSHOT.json", "snapshot/policy.zip"]
        ledger = {}
        for i, digest in enumerate(evidence["chunks"]):
            path = args.out / f"chunk-{i:02}"
            data, key = replay.validate(path, digest, state["identity"], ledger, 0, (evidence["snapshot_sha"], 0), guided=True)
            if data is None:
                raise ValueError("Duplicate guided chunk")
            ledger[key] = digest
            names.extend(f"chunk-{i:02}/{f}" for f in ("CHUNK.json", "rows.npz"))
        archive = args.out / "guided.zip"
        contents = {name: (args.out / name).read_bytes() for name in names}
        if not archive.exists():
            c.archive(archive, contents)
        with zipfile.ZipFile(archive) as zipped:
            if set(zipped.namelist()) != set(contents) or any(zipped.read(name) != data for name, data in contents.items()):
                raise ValueError("Existing replay package differs from verified chunks")
        alias, owner = c.account_for({"account": expected})
        ref = c.upload(registry, alias, owner["owner"], "guided", "guided.zip", archive, sha(archive))
        write(args.out / "receipt.json", {"ref": ref, "version": 1, "sha": sha(archive), "policy_sha": MODEL})
    print(sha(marker))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("export").add_argument("--out", type=Path, required=True)
    for name in ("plan", "launch"):
        command = commands.add_parser(name)
        command.add_argument("--package", required=True, type=Path)
        command.add_argument("--sha", required=True)
        command.add_argument("--workload", required=True, choices=WORKLOADS)
        if name == "launch":
            command.add_argument("--server-admission", action="store_true")
    for name in ("collect", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--package", required=True, type=Path)
        command.add_argument("--sha", required=True)
        command.add_argument("--out", required=True, type=Path)
        if name == "evaluate":
            command.add_argument("--model", required=True, type=Path)
            command.add_argument("--model-sha", required=True)
            command.add_argument("--tier", required=True, choices=("Easy", "Medium", "Hard", "Extreme"))
        else:
            command.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    with c.locked():
        registry = c.registry()
        if args.command == "export":
            export(args.out, registry)
        elif args.command in {"collect", "evaluate"}:
            run_companion(args, registry)
        else:
            job = plan(args.package, args.sha, args.workload, registry)
            if args.command == "launch":
                args.job, args.campaign, args.round = "j10", job["campaign"], job["round"]
                args.fingerprint = read(WORK / f"plan-stage5-{job['round']}.json")["fingerprint"]
                c.launch(args, registry)
            print(job["job"])


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        raise SystemExit(str(error))
