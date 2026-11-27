"""Prepare and submit one bounded, private first-window consolidation job."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import mkdtemp
import zipfile

from cloud.kaggle import control
from cloud.kaggle.checkpoint import STATE_FILES
from cloud.kaggle.common import ROOT, SOURCE_SHA, WORK, fingerprint, read, sha, verify, write
from cloud.kaggle.credentials import current_private_kernel_version, operator_slot, slot, verify_owner


S01 = "s5-9d68b8f2b7bb-j10-a1"
SUFFIX = '''
dataset = workspace / "consolidation-data"
unpack(input_file(JOB["dataset_ref"], "dataset.bin", JOB["dataset_sha"]), dataset, 64_000_000)
parent = workspace / "parent"
unpack(input_file(JOB["parent_ref"], "parent.bin", JOB["parent_zip_sha"]), parent, 512_000_000)
from tempfile import TemporaryDirectory
result = working / JOB["job"]
with TemporaryDirectory(prefix=".consolidation-state-", dir=working) as transient:
    state = Path(transient) / "state"
    run([*python, "-m", "cloud.kaggle.consolidated_replay", "update",
         "--parent", str(parent), "--parent-sha", JOB["parent_sha"],
         "--dataset", str(dataset), "--dataset-sha", JOB["dataset_manifest_sha"],
         "--device", JOB["device"], "--updates", str(JOB["updates"]), "--out", str(state)], stop)
    index = run([*python, "-m", "cloud.kaggle.checkpoint", "export",
                 "--input", str(state), "--sha", digest(state / "COMPLETE.json"),
                 "--out", str(result / "final")], shutdown - 30, capture=True).strip()
    for name, target in (("runtime_versions.txt", "runtime_versions.txt"), ("observed_parity.json", "parity.json")):
        shutil.copyfile(workspace / name, result / target)
    (result / "execution.json").write_text(json.dumps({"job": JOB, "index_sha256": index}, sort_keys=True))
    evaluated = []
    for model_label, model in (("before", parent / "model.zip"), ("after", state / "model.zip")):
        for tier in ("Easy", "Medium", "Hard", "Extreme"):
            name = f"{model_label}-{tier.lower()}.json"
            try:
                run([*python, "-m", "cloud.kaggle.runtime", "evaluate",
                     "--model", str(model), "--model-sha", digest(model), "--tier", tier,
                     "--out", str(result / name)], shutdown - 45)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, TimeoutError):
                continue
            evaluated.append(name)
    (result / "EVAL.json").write_text(json.dumps({"completed": evaluated}, sort_keys=True))
'''


def prepare(dataset_zip, dataset_manifest_sha, out, account):
    state = control.registry()
    parent, parent_sha = control.parent_checkpoint(state, S01)
    marker = read(parent / "COMPLETE.json")
    input_parent = parent / "parent-deflate.zip"
    with zipfile.ZipFile(input_parent) as archive:
        if set(archive.namelist()) != set(STATE_FILES) or len(archive.namelist()) != len(STATE_FILES):
            raise ValueError("Wrong parent transport archive")
        for name in STATE_FILES:
            digest = hashlib.sha256()
            with archive.open(name) as source:
                while part := source.read(1024 * 1024):
                    digest.update(part)
            if digest.hexdigest() != (parent_sha if name == "COMPLETE.json" else marker["files"][name]):
                raise ValueError("Parent transport differs from the frozen checkpoint")
    with zipfile.ZipFile(dataset_zip) as archive:
        manifest_data = archive.read("CONSOLIDATED.json")
        manifest = json.loads(manifest_data)
        if hashlib.sha256(manifest_data).hexdigest() != dataset_manifest_sha or manifest["parent_sha"] != parent_sha:
            raise ValueError("Replay transport manifest differs from the pinned dataset or parent")
    control.audit_zip(dataset_zip, "consolidation-data")
    plan = state["jobs"][S01]["planned"]
    original = ROOT / plan["runtime_path"]
    verify(original / "runtime.zip", plan["runtime_sha"])
    verify(original / "model.zip", SOURCE_SHA)
    with zipfile.ZipFile(original / "runtime.zip") as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    inventory = json.loads(files.pop("source_inventory.json"))
    if any(hashlib.sha256(files[name]).hexdigest() != value for name, value in inventory["files"].items()):
        raise ValueError("Pinned runtime inventory differs from archive")
    delimiter = '\nlearner_module = "cloud.kaggle.runtime"'
    template = files["cloud/kaggle/kernel_template.py.txt"].decode()
    if template.count(delimiter) != 1:
        raise ValueError("Unexpected pinned runtime setup boundary")
    script = template.split(delimiter)[0] + "\n" + SUFFIX
    compile(script.replace("__JOB__", "{}"), "kernel.py", "exec")
    for name in ("cloud/kaggle/replay.py", "cloud/kaggle/consolidated_replay.py"):
        files[name] = (ROOT / name).read_bytes()
    inventory["files"] = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    inventory["runtime_id"] = fingerprint(inventory["files"])
    files["source_inventory.json"] = json.dumps(inventory, sort_keys=True).encode()
    out.mkdir(parents=True)
    runtime = out / "runtime.zip"
    if any(control.SECRET.search(data) for data in files.values()):
        raise ValueError("Consolidation runtime failed secret audit")
    control.archive(runtime, files)
    control.audit_zip(runtime, "consolidation-runtime")
    job = {"job": f"s5-{fingerprint([parent_sha, dataset_manifest_sha])[:12]}-consol-a1",
           "parent_sha": parent_sha, "source_sha": SOURCE_SHA, "dataset_manifest_sha": dataset_manifest_sha,
           "dataset_sha": sha(dataset_zip), "parent_zip_sha": sha(input_parent), "runtime_sha": sha(runtime),
           "device": "cpu", "updates": 128, "max_seconds": 3600, "export_reserve": 480}
    inputs = {
        "runtime": ("consolidation-runtime", "runtime.zip", str(runtime.resolve()), job["runtime_sha"]),
        "source": ("source", "model.zip", str((original / "model.zip").resolve()), SOURCE_SHA),
        "parent": ("long-parent", "parent.zip", str(input_parent.resolve()), job["parent_zip_sha"]),
        "dataset": ("consolidation-data", "dataset.zip", str(dataset_zip.resolve()), job["dataset_sha"]),
    }
    (out / "kernel.template").write_text(script, encoding="utf-8")
    fields = {"account": account, "job": job, "inputs": inputs, "script_sha256": sha(out / "kernel.template")}
    write(out / "PLAN.json", {**fields, "fingerprint": fingerprint(fields)})
    print(job["job"], sha(out / "PLAN.json"), flush=True)


def submit(folder, expected):
    verify(folder / "PLAN.json", expected)
    plan = read(folder / "PLAN.json")
    verify(folder / "kernel.template", plan["script_sha256"])
    alias = operator_slot()
    if alias != plan["account"]:
        raise ValueError("Set the configured operator slot for this private job")
    _, authorization = slot(alias)
    if not authorization or not authorization["authorized"]:
        raise RuntimeError("This operator has no authorized credential")
    owner = authorization["owner"]
    with control.locked():
        state = control.registry()
        if any(entry["status"] in control.ACTIVE and
               (entry.get("owner") == owner or entry["planned"]["account"] == alias)
               for entry in state["jobs"].values()):
            raise RuntimeError("No unused execution slot. Existing learners retain their sessions")
        receipt = folder / "RECEIPT.json"
        if receipt.exists():
            raise RuntimeError("Consolidation submission already has a receipt and requires reconciliation")
        if not verify_owner(alias):
            raise RuntimeError("The selected credential does not match the operator")
        job = dict(plan["job"])
        ref = f"{owner}/{job['job']}"
        # Submit only after CLI verifies no recent queued or running notebook for this owner.
        control.probe_inventory(alias, owner, ref)
        refs = []
        for key, (label, name, path, digest) in plan["inputs"].items():
            path = Path(path).resolve()
            if not path.is_relative_to(WORK.resolve()):
                raise ValueError("Private consolidation input escaped campaign workspace")
            job[f"{key}_ref"] = control.upload(state, alias, owner, label, name, path, digest)
            refs.append(job[f"{key}_ref"])
        control.probe_inventory(alias, owner, ref)
        staging = folder / "notebook"
        staging.mkdir(exist_ok=True)
        code = (folder / "kernel.template").read_text(encoding="utf-8").replace("__JOB__", repr(job))
        compile(code, "kernel.py", "exec")
        if control.SECRET.search(code.encode()):
            raise ValueError("Generated notebook failed secret audit")
        (staging / "kernel.py").write_text(code, encoding="utf-8")
        write(staging / "kernel-metadata.json", {"id": ref, "title": job["job"], "code_file": "kernel.py",
             "language": "python", "kernel_type": "script", "is_private": True,
             "enable_internet": True, "enable_gpu": False, "dataset_sources": refs})
        record = {"status": "SUBMITTING", "job": job, "account": alias, "ref": ref, "at": control.now()}
        if job["job"] in state["jobs"]:
            raise RuntimeError("Existing consolidation registry entry requires reconciliation")
        entry = {"status": "SUBMITTING", "at": record["at"], "submitted": record["at"],
                 "ref": ref, "owner": owner, "version": None, "execution": job,
                 "planned": {**job, "campaign": "stage5", "role": "consolidation", "account": alias}}
        state["jobs"][job["job"]] = entry
        control.persist(state)
        write(receipt, record)
        try:
            code, output = control.cli(alias, ["kernels", "push", "-p", str(staging), "-t", str(job["max_seconds"])],
                                       timeout=90, include_stderr=True)
        except (OSError, RuntimeError):
            code, output = -1, ""
        status, version = control.push_result(code, output, ref)
        entry.update(status=status, version=version, at=control.now())
        if version is not None:
            entry["accepted_at"] = entry["at"]
        control.persist(state)
        write(receipt, {**record, "status": status, "version": version, "at": entry["at"]})
        print(job["job"], status, version, flush=True)


def harvest(folder, expected):
    from cloud.kaggle.pilot import panel, summarize

    verify(folder / "PLAN.json", expected)
    plan = read(folder / "PLAN.json")
    receipt_path = folder / "RECEIPT.json"
    receipt = read(receipt_path)
    alias = operator_slot()
    if (alias != plan["account"] or alias != receipt["account"] or not receipt.get("version")
            or not verify_owner(alias) or current_private_kernel_version(alias, receipt["ref"]) != receipt["version"]):
        raise RuntimeError("Harvest requires the original authenticated operator and verified private notebook version")
    job = plan["job"]
    with control.locked():
        registry = control.registry()
        entry = registry["jobs"][job["job"]]
        if entry["status"] == "HARVESTED_VERIFIED":
            return
        if any(entry[key] != receipt[key] for key in ("ref", "version")) or entry["execution"] != receipt["job"]:
            raise ValueError("Private notebook receipt differs from registry")
        control.status(registry, job["job"])
        if entry["status"] not in {"CLOUD_SUCCEEDED", "FAILED"}:
            raise RuntimeError("Output harvest waits for the consolidation job to finish")
        staging = Path(mkdtemp(prefix="version-", dir=folder))
        control.fetch_output(entry, job["job"], staging, "execution.json")
        execution = read(staging / "execution.json")
        if execution["job"] != entry["execution"]:
            raise ValueError("Consolidation result has a different execution identity")
        restored = staging / "state"
        state_sha = control.fetch_state_shards(entry, f"{job['job']}/final", staging / "shards",
                                               restored, execution["index_sha256"])
        state = read(restored / "COMPLETE.json")
        metrics = read(restored / "metrics.json")
        evidence = metrics["consolidation"]
        if (state["update_mode"] != "offline_consolidated_v1" or state["parent_sha"] != job["parent_sha"]
                or evidence["dataset_sha256"] != job["dataset_manifest_sha"] or
                evidence["gradient_updates"] != job["updates"] or metrics["transitions"] != 0
                or not all(n > 0 for key, n in evidence["sampling_draws"].items() if key != "parent")
                or not any(key != "parent" for key in evidence["sampling_draws"])):
            raise ValueError("Consolidation result does not prove the planned external SAC updates")
        for name in ("runtime_versions.txt", "parity.json", "EVAL.json"):
            control.fetch_output(entry, job["job"], restored, name)
        parent_path, _ = control.parent_checkpoint(registry, S01)
        parent_model_sha = read(parent_path / "COMPLETE.json")["files"]["model.zip"]
        available = set(read(restored / "EVAL.json")["completed"])
        summaries = {}
        for tier in ("Easy", "Medium", "Hard", "Extreme"):
            for label, model_sha in (("before", parent_model_sha), ("after", state["files"]["model.zip"])):
                name = f"{label}-{tier.lower()}.json"
                if name not in available:
                    continue
                control.fetch_output(entry, job["job"], restored, name)
                doc = read(restored / name)
                if doc["model_sha"] != model_sha:
                    raise ValueError("Paired panel belongs to another model")
                summaries.setdefault(tier, {})[label] = summarize(panel(doc, tier=tier))
        write(restored / "comparison.json", summaries)
        final = WORK / "harvest" / job["job"]
        restored.rename(final)
        entry.update(status="HARVESTED_VERIFIED", at=control.now(), state_sha=state_sha,
                     output_sha=execution["index_sha256"], output_version=entry["version"], output_format="shards")
        control.persist(registry)
        print(final, flush=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="task", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--dataset-zip", type=Path, required=True)
    preparation.add_argument("--dataset-manifest-sha", required=True)
    preparation.add_argument("--out", type=Path, required=True)
    preparation.add_argument("--account", default="account_01")
    for command in (commands.add_parser("submit"), commands.add_parser("harvest")):
        command.add_argument("--folder", type=Path, required=True)
        command.add_argument("--plan-sha", required=True)
    args = parser.parse_args()
    if args.task == "prepare":
        prepare(args.dataset_zip, args.dataset_manifest_sha, args.out, args.account)
    elif args.task == "submit":
        submit(args.folder, args.plan_sha)
    else:
        harvest(args.folder, args.plan_sha)


if __name__ == "__main__":
    main()
