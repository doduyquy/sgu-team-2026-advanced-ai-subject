"""Build portable BC-A/BC-B pilot datasets from frozen step-5 annotations."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
from itertools import zip_longest
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.stage4 import fingerprints, seal_attempt, write_record
from src.datasets.stage4_bc import (SCHEMA, DatasetError, compare_variants, content_digest, frame_id,
    history_ids, iter_examples, read_json, read_rows, verify_dataset, write_rows)
from src.imitation.audit import EvidenceError, project_path, sha256, source_hashes
from src.imitation.pilot import check_pair, verify_seal
from src.launcher.models import validate_custom_run_id

ACCEPT = {"PROGRESS", "JUSTIFIED_WAIT"}
BUILDER_DIRS = ["src/datasets"]
BUILDER_FILES = ["scripts/build_stage4_dataset.py", "scripts/stage4.py", "src/imitation/pilot.py",
                 "src/imitation/audit.py", "scripts/replay_rule_based_trace.py", "src/platform/agent_input.py",
                 "src/platform/action_adapter.py"]


def now():
    return datetime.now(timezone.utc).isoformat()


def selected_intervals(episode, variant, rules):
    """Validate the entire ledger before selecting anything; never re-label it."""
    if variant not in ("BC-A", "BC-B") or episode["partition"] not in ("FIT","DEV"):
        raise DatasetError("Unknown variant or partition")
    n = episode["steps"]
    for field in ("windows", "intervals"):
        end = 0
        for item in episode[field]:
            if type(item["start"]) is not int or type(item["end"]) is not int or item["start"] != end or not end < item["end"] <= n:
                raise DatasetError("Annotation gaps/overlaps or invalid bounds")
            end = item["end"]
        if end != n:
            raise DatasetError("Annotation does not cover full source episode")
    kept = []
    for interval in episode["intervals"]:
        if interval["decision"] not in ("KEEP","EXCLUDE"):
            raise DatasetError("Unknown quality decision")
        if interval["decision"] != "KEEP":
            continue
        start,end = interval["start"],interval["end"]
        if episode["outcome"] == "SUCCESS":
            if (not episode["whole_success_approved"] or interval["destination"] != "BC_A_AND_B" or
                    any(w["label"] not in ACCEPT for w in episode["windows"])):
                raise DatasetError("Unapproved SUCCESS cannot enter either dataset")
        elif episode["outcome"] == "TIMEOUT":
            if interval["destination"] != "BC_B_ONLY" or episode["whole_success_approved"]:
                raise DatasetError("Wrong TIMEOUT destination")
            if end > n-rules["timeout_tail_guard_steps"] or end-start < rules["min_timeout_segment_steps"]:
                raise DatasetError("TIMEOUT tail/length rule violated")
            for window in episode["windows"]:
                if window["label"] not in ACCEPT:
                    if end > max(0,window["start"]-rules["boundary_guard_steps"]) and start < min(n,window["end"]+rules["boundary_guard_steps"]):
                        raise DatasetError("TIMEOUT interval intersects quality boundary guard")
        else:
            raise DatasetError("Failed safety episode marked KEEP")
        if interval["context_before_start"] != max(0,start-rules["history_steps"]+1):
            raise DatasetError("Unexpected history boundary")
        kept.append(dict(interval))
    if sum(s["end"]-s["start"] for s in kept) != episode["eligible_steps"]:
        raise DatasetError("Quality eligibility count mismatch")
    if episode["whole_success_approved"] and (episode["outcome"] != "SUCCESS" or sum(s["end"]-s["start"] for s in kept) != n):
        raise DatasetError("Approved SUCCESS must be complete")
    return [s for s in kept if variant == "BC-B" or s["destination"] == "BC_A_AND_B"]


def pinned_inputs(root, definition):
    registry_path = project_path(root,definition["quality_registry"])
    if sha256(registry_path) != definition["quality_registry_sha256"]:
        raise DatasetError("Quality registry differs from fixed build definition")
    registry = read_json(registry_path)
    if registry["status"] != "PILOT_RULES_FROZEN_AND_TRAJECTORIES_REVIEWED":
        raise DatasetError("Quality review is not finalized")
    pins = {definition["quality_registry"]:sha256(registry_path), **registry["artifacts"]}
    for name, expected in pins.items():
        if sha256(project_path(root,name)) != expected:
            raise DatasetError("Quality artifact hash mismatch: " + name)
    freeze_path = root/"results/audits/stage4/g1/quality_rules_v1/freeze.json"
    freeze = read_json(freeze_path)
    for name, expected in freeze["files"].items():
        path = freeze_path.parent/name
        if sha256(path) != expected: raise DatasetError("Frozen rule payload changed")
        pins[path.relative_to(root).as_posix()] = expected
    for name, expected in freeze["dependency_sha256"].items():
        if sha256(project_path(root,name)) != expected:
            raise DatasetError("Original quality reviewer dependency changed")
    rules = read_json(freeze_path.parent/"rules.json")
    partition_path = project_path(root,registry["partition_manifest"])
    if sha256(partition_path) != registry["partition_manifest_sha256"]:
        raise DatasetError("Frozen FIT/DEV manifest changed")
    partition = read_json(partition_path)
    pins[registry["partition_manifest"]] = sha256(partition_path)
    if partition["status"] != "APPROVED_FROZEN": raise DatasetError("Split not approved")
    for name, expected in partition["files"].items():
        path=partition_path.parent/name
        if sha256(path) != expected: raise DatasetError("Split list changed")
        pins[path.relative_to(root).as_posix()] = expected
    with (partition_path.parent/"geometry_assignments.csv").open(encoding="utf8",newline="") as stream:
        assignments={r["geometry_sha256"]:r for r in csv.DictReader(stream)}
    if len(assignments) != 180 or Counter(r["partition"] for r in assignments.values()) != {"FIT":144,"DEV":36}:
        raise DatasetError("Unexpected geometry partition")
    report = read_json(root/"results/audits/stage4/g1/quality_v1/review.json")
    if report["status"] != "COMPLETE" or report["freeze_sha256"] != sha256(freeze_path):
        raise DatasetError("Quality application did not use frozen rules")
    if report["partition_sha256"] != sha256(partition_path) or rules["history_steps"] != 10:
        raise DatasetError("Split/history mismatch")
    for e in report["episodes"]:
        a=assignments.get(e["case"]["geometry_sha256"])
        if (not a or e["case"]["split"] != "TRAIN" or e["partition"] != a["partition"] or
                e["geometry_id"] != a["geometry_id"]):
            raise DatasetError("Episode differs from frozen TRAIN partition")
        selected_intervals(e,"BC-B",rules)
    return registry, report, rules, pins


def extract_source(root, episode):
    """Recheck temporal pairing for every source row, including excluded rows."""
    directory=root/"runs"/episode["source_run_id"]/"episodes"/episode["episode_id"]
    trace=directory/"launcher/stage1/agent_decisions.jsonl.gz"
    transitions=directory/"transitions.jsonl.gz"
    for path, key in ((trace,"trace_sha256"),(transitions,"transitions_sha256"),(directory/"map.json.gz","map_sha256")):
        if sha256(path) != episode[key]: raise DatasetError("Source episode bytes differ from quality review")
    needed=set()
    for s in episode["intervals"]:
        if s["decision"] == "KEEP":
            needed.update(range(s["context_before_start"],s["end"]))
    previous=read_json(directory/"reset.json")["observation"]
    selected, count = {}, 0
    for i,(decision,transition) in enumerate(zip_longest(read_rows(trace),read_rows(transitions))):
        if decision is None or transition is None: raise DatasetError("Source trace lengths differ")
        check_pair(decision,transition,episode["case"],i,previous)
        if i in needed:
            selected[i]={"agent_input":decision["agent_input"],"canonical_action":decision["canonical_action"]}
        previous=transition["post_observation"]
        count+=1
    if count != episode["steps"] or set(selected) != needed:
        raise DatasetError("Source sample count/coverage mismatch")
    return selected


def materialize_variant(output, variant, report, rules, cache, source_provenance, lineage, build_id):
    output.mkdir(parents=True,exist_ok=False)
    stats, index, selected_sources = {}, [], []
    quality_summaries=[]
    for e in report["episodes"]:
        intervals=selected_intervals(e,variant,rules)
        quality_summaries.append({k:e[k] for k in ("source_run_id","episode_id","partition","outcome","steps","window_label_steps","intervals")})
        if intervals:
            selected_sources.append({**{k:e[k] for k in ("source_run_id","episode_id","partition","geometry_id","case","outcome","trace_sha256","transitions_sha256","map_sha256")},
                                     "selected_intervals":intervals})
    for side in ("FIT","DEV"):
        frames, samples, geos, episodes, outcomes, tiers = {}, [], set(), set(), Counter(), Counter()
        for e in selected_sources:
            if e["partition"] != side: continue
            run,eid=e["source_run_id"],e["episode_id"]
            source=cache[(run,eid)]
            for interval in e["selected_intervals"]:
                for step in range(interval["start"],interval["end"]):
                    sid=frame_id(run,eid,step)
                    refs=history_ids(run,eid,step)
                    for past in range(max(0,step-9),step+1):
                        frames[frame_id(run,eid,past)]=source[past]["agent_input"]
                    payload=source[step]
                    samples.append({"sample_id":sid,"history_frame_ids":refs,"canonical_action":payload["canonical_action"]})
                    index.append({"sample_id":sid,"source_run_id":run,"episode_id":eid,"step_index":step,"partition":side,
                                  "geometry_id":e["geometry_id"],"geometry_sha256":e["case"]["geometry_sha256"],"case_id":e["case"]["case_id"],
                                  "outcome":e["outcome"],"trace_sha256":e["trace_sha256"],"transitions_sha256":e["transitions_sha256"],
                                  "agent_input_sha256":content_digest(payload["agent_input"]),
                                  "canonical_action_sha256":content_digest(payload["canonical_action"])})
                    outcomes[e["outcome"]]+=1
                    tiers[e["case"]["tier"]]+=1
            geos.add(e["case"]["geometry_sha256"])
            episodes.add((run,eid))
        write_rows(output/f"frames_{side}.jsonl.gz",({"frame_id":key,"agent_input":value} for key,value in frames.items()))
        write_rows(output/f"samples_{side}.jsonl.gz",samples)
        stats[side]={"samples":len(samples),"observation_frames":len(frames),"context_only_frames":len(frames)-len(samples),
                     "geometries":len(geos),"episodes":len(episodes)}
        quality_summaries.append({"partition":side,"accepted_sample_outcomes":dict(outcomes),"accepted_sample_tiers":dict(tiers)})
    write_rows(output/"sample_index.jsonl.gz",index)
    write_record(output/"provenance.json",{"episodes":selected_sources,"collection_sources":source_provenance,"lineage_sha256":lineage})
    write_record(output/"quality_report.json",{"variant":variant,"partitions":stats,"quality_rules_version":report["rules_version"],
                 "source_decisions":quality_summaries,"normalization_fitted":False,"sampling_ratios":None,"sample_weights":None,
                 "limitations":["Very small pilot; BC-A FIT has only one Medium geometry.",
                     "No independently approved stationary-wait examples; conservative screens can exclude valid lane changes.",
                     "Keep DEV for evaluation; source identifiers, outcomes and diagnostics must never become model features."]})
    write_record(output/"input_contract.json",{"schema":"AgentInputV1/STATE_DECISION_V1","core_dimensions":259,
                 "label":"canonical float32 [steering, throttle_brake] before env.step", "steering_forward":"positive_left_negative_right",
                 "history_frames_supported":[1,10],"history_padding":None,"history_mask_required":True,
                 "context_only_frames_are_labels":False,"normalization":"NOT_FIT; valid FIT elements only in trainer",
                 "forbidden_features":["source IDs","geometry/lane/branch IDs","outcomes","rewards","teacher diagnostics","quality labels","future frames"]})
    manifest={"schema_version":SCHEMA,"status":"BUILT_AND_VERIFIED_PILOT","dataset_id":build_id,"variant":variant,
              "created_at_utc":now(),"partitions":stats,"lineage_sha256":lineage,"files":fingerprints(output),
              "training_updates":0,"normalization_fitted":False}
    write_record(output/"manifest.json",manifest)
    checks={**manifest["files"],"manifest.json":sha256(output/"manifest.json")}
    (output/"CHECKSUMS.sha256").write_bytes("".join(f"{h}  {name}\n" for name,h in sorted(checks.items())).encode())
    try:
        verify_dataset(output)
    except BaseException:
        manifest["status"]="FAILED_VERIFICATION"
        write_record(output/"manifest.json",manifest,replace=True)
        raise
    return manifest


def build(root, run_id, definition_name, output_name):
    validate_custom_run_id(run_id,root/"runs")
    run=root/"runs"/run_id
    run.mkdir(parents=True,exist_ok=False)
    state={"run_id":run_id,"status":"RUNNING","started_at_utc":now(),"training_updates":0}
    write_record(run/"run_state.json",state)
    try:
        definition_path=project_path(root,definition_name)
        definition=read_json(definition_path)
        if (definition["schema_version"] != "stage4-bc-build-definition/1.0.0" or definition["variants"] != ["BC-A","BC-B"] or
                definition["history_storage_frames"] != 10 or definition["training_updates"] != 0):
            raise DatasetError("Unsupported build definition")
        sources=source_hashes(root,BUILDER_DIRS,BUILDER_FILES+[definition_name])
        for name in sources:
            target=run/"source_snapshot"/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((root/name).read_bytes())
        registry,report,rules,pins=pinned_inputs(root,definition)
        write_record(run/"request.json",{"definition":definition,"definition_path":definition_name,"definition_sha256":sha256(definition_path),
                     "command":sys.argv,"python":sys.version,"executable":sys.executable,"source_sha256":sources,"lineage_sha256":pins})
        source_provenance=[]
        for raw in report["sources"]:
            directory=root/"runs"/raw["run_id"]
            verify_seal(directory)
            if sha256(directory/"run_integrity.json") != raw["integrity_sha256"]:
                raise DatasetError("Raw source seal differs from quality review")
            expert=read_json(directory/"expert/agent_provenance.json")
            if expert["descriptor"]["agent_version"] != "0.6.1": raise DatasetError("Unexpected teacher version")
            source_provenance.append({**raw,"expert_provenance":expert,"runtime":read_json(directory/"runtime.json"),
                                      "collection_definition":read_json(directory/"request.json")["manifest"]})
        cache={}
        for e in report["episodes"]:
            if selected_intervals(e,"BC-B",rules):
                cache[(e["source_run_id"],e["episode_id"])]=extract_source(root,e)
                print(json.dumps({"phase":"source_paired","run":e["source_run_id"],"episode":e["episode_id"],"steps":e["steps"]}),flush=True)
        output=project_path(root,output_name)
        if not output.is_relative_to((root/"artifacts/stage4").resolve()):
            raise DatasetError("Build output must be under artifacts/stage4")
        output.mkdir(parents=True,exist_ok=False)
        manifests={v:materialize_variant(output/v,v,report,rules,cache,source_provenance,pins,definition["dataset_id"]) for v in definition["variants"]}
        relation=compare_variants(output/"BC-A",output/"BC-B")
        for variant,manifest in manifests.items():
            for side in ("FIT","DEV"):
                expected=registry["eligible_steps_by_partition"][side]["bc_a_eligible_steps" if variant=="BC-A" else "bc_b_eligible_steps"]
                if manifest["partitions"][side]["samples"] != expected: raise DatasetError("Build count differs from frozen annotation registry")
                if sum(1 for _ in iter_examples(output/variant,side,10,verify=False)) != expected:
                    raise DatasetError("Portable reader count mismatch")
        if sources != source_hashes(root,BUILDER_DIRS,BUILDER_FILES+[definition_name]) or any(sha256(root/p)!=h for p,h in pins.items()):
            raise DatasetError("Builder source or fixed evidence changed during build")
        for raw in report["sources"]: verify_seal(root/"runs"/raw["run_id"])
        write_record(output/"build_provenance.json",{"run_id":run_id,"definition":definition,"source_sha256":sources,"lineage_sha256":pins})
        bundle={"schema_version":"stage4-bc-pilot-bundle/1.0.0","status":"BUILT_AND_VERIFIED_PILOT","dataset_id":definition["dataset_id"],
                "source_run_id":run_id,"variants":{v:m["partitions"] for v,m in manifests.items()},"relation":relation,
                "files":fingerprints(output),"training_updates":0,"g1_complete":False}
        write_record(output/"bundle_manifest.json",bundle)
        write_record(run/"build_result.json",{"artifact_directory":output_name,"bundle_manifest_sha256":sha256(output/"bundle_manifest.json"),**bundle})
        state.update(status="COMPLETE",finished_at_utc=now())
        return bundle
    except BaseException as error:
        state.update(status="FAILED",error_type=type(error).__name__,error=str(error),finished_at_utc=now())
        raise
    finally:
        write_record(run/"run_state.json",state,replace=True)
        seal_attempt(run,state)


def publish(root, run_id, destination_name):
    source=root/"runs"/run_id
    verify_seal(source)
    result=read_json(source/"build_result.json")
    artifact=project_path(root,result["artifact_directory"])
    if sha256(artifact/"bundle_manifest.json") != result["bundle_manifest_sha256"]:
        raise DatasetError("Bundle manifest changed")
    if any(sha256(artifact/p)!=h for p,h in result["files"].items()): raise DatasetError("Bundle bytes changed")
    relation=compare_variants(artifact/"BC-A",artifact/"BC-B")
    dest=project_path(root,destination_name)
    if not dest.is_relative_to((root/"datasets/stage4_bc").resolve()): raise DatasetError("Curated output must be under datasets/stage4_bc")
    dest.mkdir(parents=True,exist_ok=False)
    for v in ("BC-A","BC-B"):
        (dest/v).mkdir()
        for name in ("manifest.json","sample_index.jsonl.gz","quality_report.json","provenance.json","input_contract.json","CHECKSUMS.sha256"):
            shutil.copyfile(artifact/v/name,dest/v/name)
    write_record(dest/"manifest.json",{"schema_version":"stage4-bc-pilot-registry/1.0.0","status":"BUILT_AND_VERIFIED_PILOT",
                 "dataset_id":result["dataset_id"],"artifact_directory":result["artifact_directory"],
                 "source_run_id":run_id,"source_integrity_sha256":sha256(source/"run_integrity.json"),
                 "bundle_manifest_sha256":result["bundle_manifest_sha256"],"variants":result["variants"],"relation":relation,
                 "files":fingerprints(dest),"raw_input_payloads_in_git":False,"training_updates":0,"g1_complete":False})
    return {"published":destination_name,"variants":result["variants"],"relation":relation}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("build")
    p.add_argument("--run-id",required=True)
    p.add_argument("--definition",default="experiments/stage4/g1_dataset_pilot_v1.json")
    p.add_argument("--output",required=True)
    p=sub.add_parser("publish")
    p.add_argument("--run-id",required=True)
    p.add_argument("--output",required=True)
    p=sub.add_parser("verify")
    p.add_argument("--dataset",required=True)
    args=parser.parse_args()
    if args.command=="build": result=build(ROOT,args.run_id,args.definition,args.output)
    elif args.command=="publish": result=publish(ROOT,args.run_id,args.output)
    else: result=verify_dataset(project_path(ROOT,args.dataset))
    print(json.dumps(result if args.command!="build" else {k:result[k] for k in ("status","variants","relation")}))


if __name__=="__main__":
    main()
