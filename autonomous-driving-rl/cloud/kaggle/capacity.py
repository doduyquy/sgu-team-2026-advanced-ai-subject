"""Compare verified Stage 5 capacity samples against the same preflight budget."""
import argparse
import json

from cloud.kaggle.common import WORK, read, verify, write
from cloud.kaggle.control import now, registry
from cloud.kaggle.credentials import cli, current_private_kernel_version, verify_owner

BUDGET_FIELDS = ("workers", "device", "runtime_sha", "source_sha", "template_sha",
                 "transitions", "learning_starts", "capacity", "max_seconds", "export_reserve")


def log_times(text):
    events = []
    for line in text.splitlines():
        try:
            record = json.loads(line.lstrip("[,"))
        except ValueError:
            continue
        if isinstance(record, dict) and isinstance(record.get("time"), (int, float)) and "data" in record:
            events.append(record)
    checkpoint = next((event["time"] for event in events if "COMPLETE transitions=" in event["data"]), None)
    return checkpoint, max((event["time"] for event in events), default=None)


def sample(expected, receipt, metrics, timing):
    planned = receipt["planned"]
    if any(planned.get(name) != expected[name] for name in BUDGET_FIELDS) or planned.get("parent_job"):
        raise ValueError("Capacity artifact does not match the cold warm-start benchmark budget")
    after_warmup = metrics["transitions"] - planned["learning_starts"]
    warm_seconds = metrics["post_warmup_seconds"]
    checkpoint_seconds, wall = timing
    return {
        "job": planned["job"], "version": receipt["output_version"], "device": metrics["device"],
        "gpu": metrics["gpu"], "workers": planned["workers"], "transitions": metrics["transitions"],
        "updates": metrics["updates"], "utd": metrics["utd"],
        "learner_transitions_per_second": metrics["decisions_per_second"],
        "post_warmup_transitions_per_second": after_warmup / warm_seconds,
        "learner_updates_per_second": metrics["updates_per_second"],
        "post_warmup_updates_per_second": metrics["updates"] / warm_seconds,
        "startup_to_checkpoint_seconds": checkpoint_seconds,
        "startup_to_last_cloud_log_seconds": wall,
        "end_to_end_transitions_per_second": metrics["transitions"] / wall if wall else None,
        "rss_peak_sampled_bytes": metrics["rss_peak_sampled_bytes"], "cpu_cores": metrics["affinity_cores"],
        "reset_p50_seconds": metrics["reset_p50_seconds"], "reset_p95_seconds": metrics["reset_p95_seconds"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", required=True)
    parser.add_argument("--cpu1-reference", required=True)
    args = parser.parse_args()
    from cloud.kaggle.control import identity_part
    identity_part(args.round)
    plan = read(WORK / f"plan-stage5-{args.round}.json")
    state = registry()
    samples = {}
    for label, job_id in (("j02", args.cpu1_reference), ("j03", plan["jobs"]["j03"]["job"]),
                          ("j04", plan["jobs"]["j04"]["job"])):
        row = state["jobs"][job_id]
        if row["status"] != "HARVESTED_VERIFIED":
            continue
        if row.get("gates", {}).get("G2") != "PASS" and row.get("gates", {}).get("capacity_sample") != "PASS":
            continue
        path = WORK / "harvest" / job_id
        verify(path / "COMPLETE.json", row["state_sha"])
        verify(path / "metrics.json", read(path / "COMPLETE.json")["files"]["metrics.json"])
        alias = row["planned"]["account"]
        timing = (None, None)
        if verify_owner(alias) and current_private_kernel_version(alias, row["ref"]) == row["output_version"]:
            code, output = cli(alias, ["kernels", "logs", f"{row['ref']}/{row['output_version']}"], timeout=30)
            if code == 0:
                timing = log_times(output)
        samples[label] = sample(plan["jobs"][label], row, read(path / "metrics.json"), timing)
    result = {"round": args.round, "G4": "PASS" if len(samples) == 3 and all(
        value["end_to_end_transitions_per_second"] for value in samples.values()
    ) else "PENDING_CLOUD", "samples": samples, "at": now(),
              "timing": "Kaggle kernel seconds from runtime start to last cloud log. Excludes queue time.",
              "optimizer_rate": "Updates per elapsed learner time; not an isolated optimizer kernel timer."}
    write(WORK / f"{args.round}-capacity-report.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
