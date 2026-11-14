"""Review immutable TRAIN traces; freeze FIT rules before applying them to DEV.

This produces quality annotations, never BC datasets or policy features.
Intervals use [start, end); private telemetry is used only for quality evidence.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import gzip
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.stage4 import fingerprints, seal_attempt, write_record
from src.imitation.audit import EvidenceError, load_json, project_path, sha256, source_hashes
from src.imitation.pilot import verify_episode, verify_seal

ACCEPT = {"PROGRESS", "JUSTIFIED_WAIT"}
SAFETY = ("crash_vehicle", "crash_object", "crash_building", "crash_human", "crash_sidewalk", "out_of_road")
REVIEW_DIRS = ["src/imitation", "src/agents", "src/platform"]
REVIEW_FILES = ["scripts/stage4.py", "scripts/replay_rule_based_trace.py", "scripts/review_stage4_quality.py"]


def now():
    return datetime.now(timezone.utc).isoformat()


def partition_table(root):
    path = root / "datasets/stage4_bc/fit_dev_v1/manifest.json"
    manifest = load_json(path)
    if manifest["status"] != "APPROVED_FROZEN":
        raise EvidenceError("FIT/DEV must be frozen")
    for name, expected in manifest["files"].items():
        if sha256(path.parent / name) != expected:
            raise EvidenceError("Partition artifact hash mismatch: " + name)
    with (path.parent / "geometry_assignments.csv").open(encoding="utf8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 180 or len({r["geometry_sha256"] for r in rows}) != 180:
        raise EvidenceError("Partition must contain 180 unique geometries")
    return {r["geometry_sha256"]: r for r in rows}, sha256(path)


def read_jsonl(path):
    with gzip.open(path, "rt", encoding="utf8") as stream:
        return [json.loads(line) for line in stream]


def longest_run(values):
    maximum = current = 0
    for value in values:
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum


def public_evidence(decision, rules):
    """Independent local geometry/constant-velocity screens, not teacher labels.

    These are conservative dataset screens, not a collision-free proof. Actor
    slots are never tracked across frames; paths are not assumed to be intent.
    """
    a = decision["agent_input"]
    motion, task, traffic = a["ego_motion"], a["task_context"], a["traffic_context"]
    heading = abs(motion["lane_heading_error_rad"])
    obs = a["core_observation"]["features"]
    lateral = abs((obs[8] - .5) * rules["observation_lateral_scale_m"])
    half = (rules["ego_width_m"] * abs(math.cos(heading)) +
            rules["ego_length_m"] * abs(math.sin(heading))) / 2
    clearance = task["current_lane_width"] / 2 - lateral - half
    compatible = any(l["side"] == "CURRENT" and l["route_compatible"]
                     for l in (task.get("lane_context") or {}).get("lanes", []))
    overflow = traffic["overflow_count"] + traffic["motion_paths_overflow_count"]
    speed = 3.6 * math.hypot(motion["velocity_x_mps"], motion["velocity_y_mps"])
    front_gap, closing_ttc = None, None
    support = []
    for actor, valid in zip(traffic["actors"], traffic["validity_mask"]):
        if not valid:
            continue
        x, y = actor["relative_position_x"], actor["relative_position_y"]
        vx, vy = actor["relative_velocity_x"], actor["relative_velocity_y"]
        ah = actor["relative_heading"]
        hx = (actor["length"] * abs(math.cos(ah)) + actor["width"] * abs(math.sin(ah))) / 2
        hy = (actor["width"] * abs(math.cos(ah)) + actor["length"] * abs(math.sin(ah))) / 2
        rx, ry = rules["ego_length_m"] / 2 + hx, rules["ego_width_m"] / 2 + hy
        if x > 0 and abs(y) < ry + rules["lateral_actor_buffer_m"]:
            gap = x - rx
            front_gap = gap if front_gap is None else min(front_gap, gap)
            if vx < -.1:
                ttc = max(0, gap) / -vx
                closing_ttc = ttc if closing_ttc is None else min(closing_ttc, ttc)
            if gap <= rules["wait_front_gap_m"]:
                support.append("PUBLIC_FRONT_ACTOR")
        vv = vx * vx + vy * vy
        closest = max(0, min(rules["wait_prediction_s"], -(x * vx + y * vy) / vv)) if vv > .01 else 0
        if (0 <= x + closest * vx <= rx + rules["longitudinal_actor_buffer_m"] and
                abs(y + closest * vy) <= ry + rules["lateral_actor_buffer_m"]):
            support.append("PUBLIC_PREDICTED_OCCUPANCY")
    return {"heading_rad": heading, "lane_clearance_m": clearance, "route_compatible": compatible,
            "overflow": overflow, "speed_kmh": speed, "front_gap_m": front_gap,
            "closing_ttc_s": closing_ttc, "wait_support": sorted(set(support)),
            "lidar_unavailable": sum(v <= 1e-4 for v in obs[19:]) > 120,
            "clear_local_context": min(obs[19:]) >= rules["clear_lidar_min_normalized"] and
                                   task.get("route_passage") is None and compatible and overflow == 0}


def classify_window(evidence, rules):
    """Pure decision function; absent/ambiguous evidence stays outside BC."""
    bad = []
    if evidence["safety_events"]:
        bad.append("SAFETY_EVENT")
    if evidence["max_heading_rad"] > rules["max_heading_rad"]:
        bad.append("HEADING_DEVIATION")
    if evidence["min_lane_clearance_m"] < rules["min_lane_clearance_m"]:
        bad.append("LOW_LANE_CLEARANCE")
    if evidence["max_steering_delta"] > rules["max_steering_delta"]:
        bad.append("ABRUPT_STEERING")
    if evidence["steering_sign_changes"] > rules["max_steering_sign_changes"]:
        bad.append("STEERING_OSCILLATION")
    if evidence["max_speed_delta_kmh"] > rules["max_speed_delta_kmh"]:
        bad.append("SPEED_DISCONTINUITY")
    if evidence["speed_max_kmh"] > rules["max_speed_kmh"]:
        bad.append("EXCESSIVE_SPEED_FOR_PILOT")
    if evidence["unsafe_approach_steps"]:
        bad.append("CLOSING_RISK_WITHOUT_BRAKING")
    if bad:
        return "UNSAFE_OR_UNSTABLE", bad
    if evidence["overflow_steps"] or evidence["route_incompatible_steps"] or evidence["lidar_unavailable_steps"]:
        return "UNCERTAIN", ["INCOMPLETE_TRAFFIC_OR_ROUTE_CONTEXT"]
    if evidence["max_route_drop"] > rules["max_route_drop"]:
        return "UNCERTAIN", ["ROUTE_REGRESSION_OR_PROJECTION_JUMP"]
    if evidence["longest_unsupported_stop"] > rules["max_brief_stop_steps"]:
        if evidence["route_gain"] <= rules["min_route_gain"] and evidence["clear_unsupported_stop_steps"] >= rules["deadlock_stop_steps"]:
            return "DEADLOCK_OR_LOOP", ["SUSPECTED_STALL_WITH_CLEAR_LOCAL_CONTEXT"]
        return "UNCERTAIN", ["STOP_WITHOUT_SUFFICIENT_PUBLIC_SUPPORT"]
    if evidence["supported_stop_steps"] > rules["max_supported_wait_steps"]:
        return "UNCERTAIN", ["WAIT_DURATION_REQUIRES_SEPARATE_REVIEW"]
    if evidence["route_gain"] >= rules["min_route_gain"]:
        return "PROGRESS", ["ROUTE_PROGRESS_AND_LOCAL_SCREENS_PASSED"]
    if evidence["supported_stop_steps"] and evidence["unsupported_stop_steps"] <= rules["max_brief_stop_steps"]:
        return "JUSTIFIED_WAIT", ["BRAKING_OR_HOLD_WITH_PUBLIC_ACTOR_SUPPORT"]
    if evidence["travel_m"] >= rules["loop_travel_m"]:
        return "DEADLOCK_OR_LOOP", ["MOTION_WITHOUT_ROUTE_PROGRESS"]
    return "UNCERTAIN", ["INSUFFICIENT_PROGRESS_OR_WAIT_EVIDENCE"]


def annotate(decisions, transitions, outcome, rules):
    n = len(decisions)
    if not n or n != len(transitions):
        raise EvidenceError("Missing or misaligned trajectory")
    public = [public_evidence(d, rules) for d in decisions]
    speeds = [t["post_vehicle"]["speed_km_h"] for t in transitions]
    routes = [t["step_info"]["route_completion"] for t in transitions]
    stops, supported = [], []
    for d, e in zip(decisions, public):
        stop = e["speed_kmh"] < rules["stop_speed_kmh"]
        stops.append(stop)
        supported.append(stop and bool(e["wait_support"]) and d["canonical_action"][1] <= rules["wait_max_throttle"])
    unsupported = [s and not ok and i >= rules["startup_steps"] for i, (s, ok) in enumerate(zip(stops, supported))]
    # Count continuous duration across window boundaries; a long stationary run
    # cannot become acceptable by being cut into small windows.
    unsupported_age, supported_age = [], []
    for flags, ages in ((unsupported, unsupported_age), (supported, supported_age)):
        for flag in flags:
            ages.append((ages[-1] if ages else 0) + 1 if flag else 0)
    windows = []
    for lo in range(0, n, rules["window_steps"]):
        hi = min(n, lo + rules["window_steps"])
        ee, tt, dd = public[lo:hi], transitions[lo:hi], decisions[lo:hi]
        steer = [d["canonical_action"][0] for d in decisions[max(0, lo-1):hi]]
        nonzero = [s for s in steer if abs(s) >= rules["steering_sign_min_abs"]]
        gaps = [e["front_gap_m"] for e in ee if e["front_gap_m"] is not None]
        ttcs = [e["closing_ttc_s"] for e in ee if e["closing_ttc_s"] is not None]
        travel = sum(math.dist(t["pre_vehicle"]["position_m"], t["post_vehicle"]["position_m"]) for t in tt)
        ev = {"route_start": routes[lo-1] if lo else routes[0], "route_end": routes[hi-1],
              "route_gain": routes[hi-1] - (routes[lo-1] if lo else routes[0]),
              "max_route_drop": max([0] + [routes[i-1] - routes[i] for i in range(max(1, lo), hi)]),
              "travel_m": travel, "speed_min_kmh": min(speeds[lo:hi]), "speed_max_kmh": max(speeds[lo:hi]),
              "speed_median_kmh": statistics.median(speeds[lo:hi]),
              "max_heading_rad": max(e["heading_rad"] for e in ee),
              "min_lane_clearance_m": min(e["lane_clearance_m"] for e in ee),
              "overflow_steps": sum(e["overflow"] > 0 for e in ee),
              "route_incompatible_steps": sum(not e["route_compatible"] for e in ee),
              "lidar_unavailable_steps": sum(e["lidar_unavailable"] for e in ee),
              "max_steering_delta": max([0] + [abs(x-y) for x,y in zip(steer, steer[1:])]),
              "steering_sign_changes": sum(x*y < 0 for x,y in zip(nonzero, nonzero[1:])),
              "max_speed_delta_kmh": max([0] + [abs(speeds[i] - speeds[i-1]) for i in range(max(1,lo),hi)]),
              "safety_events": dict(Counter(k for t in tt for k in SAFETY if t["step_info"].get(k))),
              "unsupported_stop_steps": sum(unsupported[lo:hi]), "supported_stop_steps": max(supported_age[lo:hi]),
              "clear_unsupported_stop_steps": sum(unsupported[i] and public[i]["clear_local_context"] for i in range(lo,hi)),
              "longest_unsupported_stop": max(unsupported_age[lo:hi]),
              "wait_support_steps": dict(Counter(k for e in ee for k in e["wait_support"])),
              "min_front_gap_m": min(gaps) if gaps else None, "min_closing_ttc_s": min(ttcs) if ttcs else None,
              "unsafe_approach_steps": sum(e["speed_kmh"] > rules["risk_speed_kmh"] and
                  e["closing_ttc_s"] is not None and e["closing_ttc_s"] < rules["min_closing_ttc_s"] and
                  d["canonical_action"][1] > rules["risk_max_throttle"] for d,e in zip(dd,ee)),
              "teacher_behavior_counts_for_audit_only": dict(Counter((d["decision"].get("diagnostics") or {}).get("behavior_state","MISSING") for d in dd))}
        label, reasons = classify_window(ev, rules)
        windows.append({"start": lo, "end": hi, "label": label, "reasons": reasons, "evidence": ev})
    eligible = [False] * n
    whole_success = outcome == "SUCCESS" and all(w["label"] in ACCEPT for w in windows)
    if whole_success:
        eligible = [True] * n
    elif outcome == "TIMEOUT":
        for w in windows:
            if w["label"] in ACCEPT:
                eligible[w["start"]:w["end"]] = [True] * (w["end"]-w["start"])
        for w in windows:
            if w["label"] not in ACCEPT:
                a,b = max(0,w["start"]-rules["boundary_guard_steps"]), min(n,w["end"]+rules["boundary_guard_steps"])
                eligible[a:b] = [False]*(b-a)
        tail = min(n, rules["timeout_tail_guard_steps"])
        eligible[n-tail:] = [False]*tail
    else:
        # Failed safety outcomes are retained as evidence, never demonstrations.
        whole_success = False
    intervals = []
    start = 0
    while start < n:
        end = start + 1
        while end < n and eligible[end] == eligible[start]:
            end += 1
        keep = eligible[start] and (whole_success or end-start >= rules["min_timeout_segment_steps"])
        intervals.append({"start": start, "end": end, "decision": "KEEP" if keep else "EXCLUDE",
                          "destination": "BC_A_AND_B" if keep and whole_success else "BC_B_ONLY" if keep else None,
                          "reason": "REVIEWED_SUCCESS" if keep and whole_success else "REVIEWED_TIMEOUT_PROGRESS_OR_WAIT" if keep else
                                    "SUCCESS_NOT_APPROVED_AS_A_WHOLE" if outcome == "SUCCESS" else
                                    "QUALITY_OR_BOUNDARY_OR_TAIL_OR_SHORT_SEGMENT_EXCLUSION",
                          "context_before_start": max(0,start-rules["history_steps"]+1),
                          "history_policy": "CURRENT_AND_PAST_ONLY; context rows are not accepted labels"})
        start = end
    return {"whole_success_approved": whole_success, "windows": windows, "intervals": intervals,
            "eligible_steps": sum(s["end"]-s["start"] for s in intervals if s["decision"]=="KEEP"),
            "window_label_steps": dict(Counter({label:sum(w["end"]-w["start"] for w in windows if w["label"]==label)
                                                for label in sorted({w["label"] for w in windows})}))}


def review(root, run_ids, rules_path, output, fit_only=False, frozen=None):
    output.mkdir(parents=True, exist_ok=False)
    state = {"run_id": output.name, "status": "RUNNING", "started_at_utc": now()}
    write_record(output / "run_state.json", state)
    try:
        rules_hash, reviewer_hash = sha256(rules_path), sha256(Path(__file__))
        dependencies = source_hashes(root, REVIEW_DIRS, REVIEW_FILES)
        rules = load_json(rules_path)
        if rules["schema_version"] != "stage4-quality-rules/1.0.0" or rules["calibration_partition"] != "FIT":
            raise EvidenceError("Unsupported quality schema or calibration partition")
        for name in ("window_steps","min_timeout_segment_steps","history_steps"):
            if type(rules[name]) is not int or rules[name] <= 0:
                raise EvidenceError("Invalid positive integer: " + name)
        for name, value in rules.items():
            if isinstance(value,(int,float)) and (not math.isfinite(value) or (value < 0 and name != "risk_max_throttle")):
                raise EvidenceError("Invalid numeric rule: " + name)
        sides, partition_hash = partition_table(root)
        frozen_hash = None
        if not fit_only:
            if frozen is None:
                raise EvidenceError("A frozen rule package is mandatory before DEV review")
            package = load_json(frozen / "freeze.json")
            for name, expected in package["files"].items():
                if sha256(frozen / name) != expected:
                    raise EvidenceError("Frozen package modified")
            if (package["status"] != "FROZEN_FOR_PILOT" or sha256(rules_path) != package["files"]["rules.json"] or
                    sha256(Path(__file__)) != package["reviewer_source_sha256"] or partition_hash != package["partition_sha256"] or
                    dependencies != package["dependency_sha256"]):
                raise EvidenceError("Rules, reviewer source or partition changed after freeze")
            frozen_hash = sha256(frozen / "freeze.json")
        episodes, sources = [], []
        for run_id in run_ids:
            directory = project_path(root, "runs/"+run_id)
            seal = verify_seal(directory)
            summary, request = load_json(directory / "pilot_summary.json"), load_json(directory / "request.json")
            if summary["failed_episodes"] or summary["completed_episodes"] != summary["expected_episodes"]:
                raise EvidenceError("Incomplete collection")
            plans = load_json(directory / "resolved_plans.json")
            expert = load_json(directory / "expert/agent_provenance.json")
            sources.append({"run_id":run_id,"integrity_sha256":sha256(directory/"run_integrity.json"),"verified_files":len(seal["files"]),
                            "request_sha256":sha256(directory/"request.json"),"expert_provenance_sha256":sha256(directory/"expert/agent_provenance.json")})
            for i, (plan, collected) in enumerate(zip(plans,summary["episodes"])):
                case = plan["resolved_cases"][0]
                assignment = sides.get(case["geometry_sha256"])
                if not assignment or case["split"] != "TRAIN":
                    raise EvidenceError("Geometry outside frozen TRAIN split")
                if fit_only and assignment["partition"] != "FIT":
                    continue  # Do not open DEV trajectories during calibration.
                eid = f"episode_{i:03d}"
                epdir = directory/"episodes"/eid
                verified = verify_episode(root,epdir,plan,expert,replay=True)
                if any(verified[k] != collected[k] for k in ("steps","trace_sha256","transitions_sha256","map_sha256","terminal_reason")):
                    raise EvidenceError("Collection report differs from reverified trajectory")
                decisions = read_jsonl(epdir/"launcher/stage1/agent_decisions.jsonl.gz")
                transitions = read_jsonl(epdir/"transitions.jsonl.gz")
                result = annotate(decisions,transitions,verified["terminal_reason"],rules)
                episodes.append({"source_run_id":run_id,"episode_id":eid,"case":case,"geometry_id":assignment["geometry_id"],
                                 "partition":assignment["partition"],"steps":verified["steps"],"outcome":verified["terminal_reason"],
                                 "trace_sha256":verified["trace_sha256"],"transitions_sha256":verified["transitions_sha256"],
                                 "map_sha256":verified["map_sha256"],"pairing_and_replay_passed":True,**result})
            if sha256(directory/"run_integrity.json") != sources[-1]["integrity_sha256"]:
                raise EvidenceError("Source seal changed while reviewing")
        if not episodes or len({(e["source_run_id"],e["episode_id"]) for e in episodes}) != len(episodes):
            raise EvidenceError("Empty or duplicate review")
        if (sha256(rules_path) != rules_hash or sha256(Path(__file__)) != reviewer_hash or
                dependencies != source_hashes(root, REVIEW_DIRS, REVIEW_FILES)):
            raise EvidenceError("Rules or reviewer changed during review")
        report = {"schema_version":"stage4-quality-review/1.0.0","status":"COMPLETE","created_at_utc":now(),
                  "scope":"FIT_CALIBRATION_ONLY" if fit_only else "FROZEN_RULE_APPLICATION",
                  "reviewer":"Codex evidence review; no human sign-off claimed", "rules_version":rules["version"],
                  "command":sys.argv,"python":sys.version,"executable":sys.executable,"dependency_sha256":dependencies,
                  "rules_sha256":sha256(rules_path),"reviewer_source_sha256":sha256(Path(__file__)),
                  "partition_sha256":partition_hash,"freeze_sha256":frozen_hash,"sources":sources,"episodes":episodes,
                  "raw_steps":sum(e["steps"] for e in episodes),"eligible_steps":sum(e["eligible_steps"] for e in episodes),
                  "bc_datasets_built":False,"training_updates":0,"g1_complete":False}
        write_record(output/"review.json", report)
        (output/"rules.json").write_bytes(rules_path.read_bytes())
        (output/"reviewer_source.py").write_bytes(Path(__file__).read_bytes())
        state.update(status="COMPLETE",finished_at_utc=now())
        return report
    except BaseException as error:
        state.update(status="FAILED",error_type=type(error).__name__,error=str(error),finished_at_utc=now())
        raise
    finally:
        write_record(output/"run_state.json",state,replace=True)
        seal_attempt(output,state)


def freeze(profile, output):
    verify_seal(profile)
    report = load_json(profile/"review.json")
    if (report["scope"] != "FIT_CALIBRATION_ONLY" or any(e["partition"] != "FIT" for e in report["episodes"]) or
            not any(e["outcome"]=="TIMEOUT" for e in report["episodes"]) or
            not any(e["outcome"]=="SUCCESS" for e in report["episodes"]) or
            sha256(Path(__file__)) != report["reviewer_source_sha256"]):
        raise EvidenceError("Freeze requires current reviewer and FIT SUCCESS/TIMEOUT evidence")
    if (sha256(profile/"rules.json") != report["rules_sha256"] or
            source_hashes(ROOT, REVIEW_DIRS, REVIEW_FILES) != report["dependency_sha256"] or
            partition_table(ROOT)[1] != report["partition_sha256"]):
        raise EvidenceError("Calibration rules, dependencies or partition changed before freeze")
    output.mkdir(parents=True,exist_ok=False)
    for source, target in (("rules.json","rules.json"),("review.json","fit_calibration.json"),("reviewer_source.py","reviewer_source.py")):
        (output/target).write_bytes((profile/source).read_bytes())
    package = {"schema_version":"stage4-quality-freeze/1.0.0","status":"FROZEN_FOR_PILOT","frozen_at_utc":now(),
               "reviewer":"Codex evidence review; no human sign-off claimed","profile_integrity_sha256":sha256(profile/"run_integrity.json"),
               "partition_sha256":report["partition_sha256"],"reviewer_source_sha256":report["reviewer_source_sha256"],
               "dependency_sha256":report["dependency_sha256"],
               "calibration_partitions":["FIT"],"dev_rule_tuning":False,"files":fingerprints(output),
               "limitations":["Small pilot, conservative local geometry/constant velocity screens; no safety guarantee.",
                              "Uncertain windows excluded; broader coverage requires a new FIT-calibrated rule version.",
                              "No balancing/weight choice, BC dataset construction or training in step 5."]}
    write_record(output/"freeze.json",package)
    return package


def publish(source, frozen, output):
    verify_seal(source)
    report, package = load_json(source/"review.json"), load_json(frozen/"freeze.json")
    for name, expected in package["files"].items():
        if sha256(frozen/name) != expected:
            raise EvidenceError("Frozen quality evidence changed")
    if report["scope"] != "FROZEN_RULE_APPLICATION" or report["freeze_sha256"] != sha256(frozen/"freeze.json"):
        raise EvidenceError("Only a completed frozen application can be published")
    output.mkdir(parents=True,exist_ok=False)
    (output/"review.json").write_bytes((source/"review.json").read_bytes())
    totals = {}
    for side in ("FIT","DEV"):
        episodes = [e for e in report["episodes"] if e["partition"]==side]
        a = sum(e["steps"] for e in episodes if e["whole_success_approved"])
        b_extra = sum(e["eligible_steps"] for e in episodes if e["outcome"]=="TIMEOUT")
        totals[side] = {"episodes":len(episodes),"raw_steps":sum(e["steps"] for e in episodes),
                        "outcomes":dict(Counter(e["outcome"] for e in episodes)),
                        "approved_success_episodes":sum(e["whole_success_approved"] for e in episodes),
                        "bc_a_eligible_steps":a,"bc_b_additional_timeout_steps":b_extra,
                        "bc_b_eligible_steps":a+b_extra}
    write_record(output/"summary.json",{"schema_version":"stage4-quality-summary/1.0.0","partitions":totals,
                 "quality_rules_version":report["rules_version"],"training_samples_from_dev":0,
                 "bc_datasets_built":False,"bc_accepted_samples":0,"training_updates":0,
                 "note":"Eligibility annotations only. DEV remains DEV; excluded raw steps are retained."})
    write_record(output/"evidence_index.json",{"schema_version":"stage4-quality-evidence/1.0.0","published_at_utc":now(),
                 "source_review":source.relative_to(ROOT).as_posix(),"source_integrity_sha256":sha256(source/"run_integrity.json"),
                 "frozen_rules":frozen.relative_to(ROOT).as_posix(),"freeze_sha256":sha256(frozen/"freeze.json"),
                 "rules_sha256":report["rules_sha256"],"files":fingerprints(output),"g1_complete":False})
    return totals


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ("profile","apply"):
        p=sub.add_parser(name)
        p.add_argument("--source-run",action="append",required=True)
        p.add_argument("--rules",required=True)
        p.add_argument("--output",required=True)
        if name=="apply": p.add_argument("--frozen",required=True)
    p=sub.add_parser("freeze")
    p.add_argument("--profile",required=True)
    p.add_argument("--output",required=True)
    p=sub.add_parser("publish")
    p.add_argument("--review",required=True)
    p.add_argument("--frozen",required=True)
    p.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.command=="publish":
        print(json.dumps(publish(project_path(ROOT,args.review),project_path(ROOT,args.frozen),project_path(ROOT,args.output))))
    elif args.command=="freeze":
        result=freeze(project_path(ROOT,args.profile),project_path(ROOT,args.output))
        print(json.dumps({k:result[k] for k in ("status","frozen_at_utc")}))
    else:
        result=review(ROOT,args.source_run,project_path(ROOT,args.rules),project_path(ROOT,args.output),
                      fit_only=args.command=="profile",frozen=project_path(ROOT,args.frozen) if args.command=="apply" else None)
        print(json.dumps({"status":result["status"],"raw_steps":result["raw_steps"],"eligible_steps":result["eligible_steps"],
                          "episodes":[{k:e[k] for k in ("source_run_id","episode_id","partition","outcome","steps","eligible_steps","whole_success_approved")} for e in result["episodes"]]}))


if __name__=="__main__":
    main()
