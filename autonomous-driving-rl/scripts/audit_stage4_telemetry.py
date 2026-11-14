"""Kiểm chứng vận tốc và dấu lái G1 bước 1; chỉ TRAIN, không tạo dữ liệu BC."""

import argparse
import csv
from datetime import datetime, timezone
import gzip
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.imitation.audit import load_json, sha256, source_hashes
from src.launcher.cases import load_locked_geometry_block_sequence, make_jsonable
from src.launcher.executor import ExperimentExecutor, build_metadrive_case_config
from src.launcher.models import LaunchRequestV1, LauncherMode, validate_custom_run_id
from src.launcher.resolver import resolve_experiment_plan
from src.logging.agent_experiment import write_json
from src.platform import WandbMode
import metadrive
from metadrive import MetaDriveEnv


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprints(directory):
    return {p.relative_to(directory).as_posix(): sha256(p)
            for p in sorted(directory.rglob("*")) if p.is_file() and p != directory / "run_integrity.json"}


def git(root, *args):
    return subprocess.check_output(
        ["git", "-c", "safe.directory=" + root.as_posix(), *args], cwd=root,
        text=True, encoding="utf8").strip()


def measurement(vehicle):
    velocity = vehicle.body.get_linear_velocity()
    return {
        "position_m": [float(x) for x in vehicle.position[:2]],
        "heading_rad": float(vehicle.heading_theta),
        "body_velocity_mps": [float(velocity[0]), float(velocity[1])],
        "reference_speed_kmh": 3.6 * math.hypot(velocity[0], velocity[1]),
        "speed_km_h": float(vehicle.speed_km_h),
        "legacy_speed_kmh_fallback": float(getattr(vehicle, "speed_kmh", 0.0)),
        "has_speed_kmh": hasattr(vehicle, "speed_kmh"),
        "applied_steering": float(vehicle.steering),
    }


def map_metadata(env):
    data = env.current_map.get_meta_data()
    data["map_config"] = data["map_config"].get_serializable_dict()
    return make_jsonable(data)


def signed_turn(start, end):
    theta = start["heading_rad"]
    delta = end["heading_rad"] - theta
    dx, dy = (b - a for a, b in zip(start["position_m"], end["position_m"]))
    return {"heading_delta_rad": math.atan2(math.sin(delta), math.cos(delta)),
            "initial_ego_left_displacement_m": -math.sin(theta) * dx + math.cos(theta) * dy,
            "initial_ego_forward_displacement_m": math.cos(theta) * dx + math.sin(theta) * dy}


def speed_summary(rows, tolerance):
    if not rows:
        raise ValueError("Không có số đo vận tốc")
    references = [r["post"]["reference_speed_kmh"] for r in rows]
    errors = [abs(r["post"]["speed_km_h"] - v) for r, v in zip(rows, references)]
    return {"steps": len(rows), "reference_mean_speed_kmh": sum(references) / len(rows),
            "reference_max_speed_kmh": max(references), "max_speed_error_kmh": max(errors),
            "speed_matches_body_velocity": max(errors) <= tolerance,
            "moving_steps": sum(v > 1.0 for v in references),
            "legacy_false_zero_steps": sum(v > 1.0 and r["post"]["legacy_speed_kmh_fallback"] == 0
                                            for r, v in zip(rows, references))}


def run_audit(run_id, expected_logging):
    validate_custom_run_id(run_id, ROOT / "runs")
    output = ROOT / "runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    state = {"run_id": run_id, "status": "RUNNING", "started_at_utc": now(),
             "scope": "G1_STEP1_ONLY", "canonical_run": False, "dataset_eligible": False}
    write_json(output / "run_state.json", state)
    try:
        manifest = load_json(ROOT / "experiments/stage4/g1_telemetry_v1.json")
        tolerance = manifest["tolerances"]
        sources = source_hashes(ROOT, ["src", "configs/platform", "datasets/mapsuite_v1", "experiments/stage4"],
                                ["scripts/audit_stage4_telemetry.py"])
        for name in sources:
            target = output / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        protected = source_hashes(ROOT, ["src/agents", "experiments/rule_based/versions", "datasets/mapsuite_v1",
                                          "configs/platform", "results/audits/stage4/g0"])
        write_json(output / "protected_before.json", protected, exclusive=True)
        simulator = Path(metadrive.__file__).resolve().parent.parent
        runtime = {"python": sys.version, "executable": sys.executable,
                   "packages": {p: importlib.metadata.version(p) for p in
                                ("numpy", "panda3d", "metadrive-simulator")},
                   "metadrive_commit": git(simulator, "rev-parse", "HEAD"),
                   "metadrive_dirty": git(simulator, "status", "--porcelain"),
                   "project_commit": git(ROOT, "rev-parse", "HEAD"),
                   "project_status": git(ROOT, "status", "--porcelain"),
                   "command": sys.argv, "expected_logging": expected_logging}
        write_json(output / "runtime.json", runtime, exclusive=True)
        if (runtime["packages"]["metadrive-simulator"] != "0.4.3" or
                runtime["metadrive_commit"] != "85e5dadc6c7436d324348f6e3d8f8e680c06b4db" or
                runtime["metadrive_dirty"]):
            raise ValueError("MetaDrive chưa khớp phiên bản/commit/cây nguồn sạch")
        simulator_sources = {}
        for name in ("metadrive/base_class/base_object.py", "metadrive/component/vehicle/base_vehicle.py",
                     "metadrive/obs/state_obs.py", "metadrive/policy/env_input_policy.py"):
            target = output / "simulator_source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((simulator / name).read_bytes())
            simulator_sources[name] = sha256(target)
        from metadrive.component.vehicle.base_vehicle import BaseVehicle
        write_json(output / "request.json", {"manifest": manifest, "source_sha256": sources,
                   "simulator_source_sha256": simulator_sources,
                   "speed_property_source": inspect.getsource(BaseVehicle.speed_km_h.fget)}, exclusive=True)
        plans = []
        for spec in manifest["cases"]:
            plan = resolve_experiment_plan(LaunchRequestV1(
                mode=LauncherMode.SANDBOX, agent_id="stage1_rule_based", tier=spec["tier"],
                sequence=spec["sequence"], geometry_generation_seed=spec["geometry_seed"],
                environment_seed=spec["environment_seed"], agent_seed=manifest["agent_seed"],
                render_mode="OFF", wandb_mode=WandbMode.DISABLED))
            if any(c.split != "TRAIN" for c in plan.resolved_cases):
                raise ValueError("Chỉ được dùng TRAIN")
            plans.append(plan)
        write_json(output / "resolved_plans.json", [p.to_dict() for p in plans], exclusive=True)
        probes, natural = [], []
        for case_index, plan in enumerate(plans):
            case = plan.resolved_cases[0]
            blocks, _, _ = load_locked_geometry_block_sequence(case.to_dict())
            cfg = build_metadrive_case_config(case, blocks)
            probe = manifest["controlled_probe"]
            for steering in probe["steering_values"]:
                label = f"probe_{case_index:02d}_{steering:+.2f}"
                directory = output / label
                directory.mkdir()
                config = dict(cfg, traffic_density=probe["traffic_density_override"], log_level=50)
                write_json(directory / "environment.json", {"case": case.to_dict(), "config": config,
                           "intervention": "No traffic; fixed forward actions; excluded from BC"}, exclusive=True)
                env = MetaDriveEnv(config)
                rows = []
                try:
                    env.reset(seed=case.environment_seed)
                    write_json(directory / "map.json", map_metadata(env), exclusive=True)
                    with (directory / "measurements.jsonl").open("x", encoding="utf8") as stream:
                        start = None
                        for step in range(probe["warmup_steps"] + probe["turn_steps"]):
                            turning = step >= probe["warmup_steps"]
                            pre = measurement(env.agent)
                            if step == probe["warmup_steps"]:
                                start = pre
                            action = [steering if turning else 0.0,
                                      probe["turn_throttle"] if turning else probe["warmup_throttle"]]
                            _, _, term, trunc, info = env.step(action)
                            row = {"step_index": step, "action": action, "pre": pre,
                                   "post": measurement(env.agent), "terminated": term, "truncated": trunc,
                                   "flags": {k: bool(info.get(k, False)) for k in
                                             ("arrive_dest", "out_of_road", "crash_vehicle", "crash_object", "crash_sidewalk")}}
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                            rows.append(row)
                            if term or trunc:
                                raise ValueError("Phép thử dấu lái kết thúc sớm")
                    turn = signed_turn(start, rows[-1]["post"])
                    if steering:
                        sign = math.copysign(1, steering)
                        passed = (turn["heading_delta_rad"] * sign > tolerance["minimum_heading_change_rad"] and
                                  turn["initial_ego_left_displacement_m"] * sign > tolerance["minimum_lateral_displacement_m"])
                    else:
                        passed = abs(turn["heading_delta_rad"]) <= tolerance["straight_heading_change_rad"]
                    passed = passed and turn["initial_ego_forward_displacement_m"] > 0
                    passed = passed and all(abs(r["post"]["applied_steering"] - r["action"][0]) <= 1e-6 for r in rows)
                    result = {"case_id": case.case_id, "probe": label, "steering": steering, **turn,
                              **speed_summary(rows, tolerance["speed_kmh_abs"]), "steering_sign_passed": passed}
                    probes.append(result)
                    write_json(directory / "result.json", result, exclusive=True)
                finally:
                    env.close()
                print(json.dumps({"probe": label, "passed": passed}), flush=True)

            directory = output / f"natural_{case_index:02d}"
            directory.mkdir()
            rows = []
            # Passive recorder around the actual launcher, never modifies observations/actions.
            with (directory / "measurements.jsonl").open("x", encoding="utf8") as stream:
                class ObservedEnv(MetaDriveEnv):
                    def reset(self, *args, **kwargs):
                        result = super().reset(*args, **kwargs)
                        write_json(directory / "environment.json", {"case": case.to_dict(), "config": cfg}, exclusive=True)
                        write_json(directory / "map.json", map_metadata(self), exclusive=True)
                        return result

                    def step(self, action):
                        pre = measurement(self.agent)
                        result = super().step(action)
                        row = {"step_index": len(rows), "action": [float(x) for x in action],
                               "pre": pre, "post": measurement(self.agent),
                               "terminated": bool(result[2]), "truncated": bool(result[3])}
                        rows.append(row)
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        return result

                events = []
                with patch("src.launcher.executor.MetaDriveEnv", ObservedEnv):
                    report = ExperimentExecutor(plan, runs_root=directory / "launcher", custom_run_id="stage1",
                                                event_callback=lambda event: events.append(event.to_dict())).execute()
            write_json(directory / "events.json", events, exclusive=True)
            write_json(directory / "execution_report.json", report.to_dict(), exclusive=True)
            if report.status != "COMPLETE" or report.completed_episodes != 1:
                raise ValueError("Lượt chạy Stage 1 chưa hoàn tất")
            run_dir = Path(report.run_dir)
            with (run_dir / "episodes.csv").open(encoding="utf8", newline="") as stream:
                episode = next(csv.DictReader(stream))
            summary = speed_summary(rows, tolerance["speed_kmh_abs"])
            aggregate_ok = all(abs(float(episode[field]) - summary["reference_" + field]) <= tolerance["speed_kmh_abs"]
                               for field in ("mean_speed_kmh", "max_speed_kmh"))
            # Locate the launcher's single decision stream; verify timing and public speed too.
            traces = list(run_dir.rglob("*.jsonl.gz"))
            if len(traces) != 1:
                raise ValueError("Thiếu hoặc thừa vết quyết định")
            with gzip.open(traces[0], "rt", encoding="utf8") as stream:
                decisions = [json.loads(line) for line in stream]
            alignment_ok = len(decisions) == len(rows)
            public_speed_error = 0.0
            for i, (decision, row) in enumerate(zip(decisions, rows)):
                alignment_ok &= (decision["step_index"] == i and decision["case_id"] == case.case_id and
                                 decision["canonical_action"] == row["action"])
                motion = decision["agent_input"]["ego_motion"]
                speed = 3.6 * math.hypot(motion["velocity_x_mps"], motion["velocity_y_mps"])
                public_speed_error = max(public_speed_error, abs(speed - row["pre"]["reference_speed_kmh"]))
            progress = [e for e in events if e["event_type"] == "EPISODE_PROGRESS"]
            progress_ok = bool(progress) and all(
                abs(e["speed_kmh"] - rows[e["step_index"] - 1]["post"]["reference_speed_kmh"]) <= tolerance["speed_kmh_abs"]
                for e in progress)
            result = {"case_id": case.case_id, **summary, "terminal_reason": episode["primary_terminal_reason"],
                      "logged_mean_speed_kmh": float(episode["mean_speed_kmh"]),
                      "logged_max_speed_kmh": float(episode["max_speed_kmh"]),
                      "aggregate_matches": aggregate_ok, "progress_matches": progress_ok,
                      "progress_samples": len(progress), "alignment_passed": bool(alignment_ok),
                      "public_ego_motion_speed_max_error_kmh": public_speed_error}
            natural.append(result)
            write_json(directory / "result.json", result, exclusive=True)
            print(json.dumps({"natural": case.case_id, "aggregate_matches": aggregate_ok}), flush=True)
        after = source_hashes(ROOT, ["src/agents", "experiments/rule_based/versions", "datasets/mapsuite_v1",
                                    "configs/platform", "results/audits/stage4/g0"])
        write_json(output / "protected_after.json", after, exclusive=True)
        checks = {"protected_unchanged": protected == after,
                  "speed_matches_physics": all(r["speed_matches_body_velocity"] and r["moving_steps"] > 0 for r in probes + natural),
                  "positive_left_negative_right": all(r["steering_sign_passed"] for r in probes),
                  "pre_action_alignment": all(r["alignment_passed"] and r["public_ego_motion_speed_max_error_kmh"] <= tolerance["speed_kmh_abs"] for r in natural),
                  "expected_logging_observed": all((r["aggregate_matches"] and r["progress_matches"]) if expected_logging == "corrected"
                                                    else (not r["aggregate_matches"] and not r["progress_matches"] and r["legacy_false_zero_steps"] > 0)
                                                    for r in natural)}
        result = {"schema_version": "stage4-g1-telemetry-result/1.0.0", "run_id": run_id,
                  "expected_logging": expected_logging, "checks": checks, "probes": probes, "natural": natural,
                  "passed": all(checks.values()), "dataset_eligible": False, "g1_complete": False}
        write_json(output / "audit_summary.json", result, exclusive=True)
        if not result["passed"]:
            raise ValueError("Kiểm chứng không đạt; xem audit_summary.json")
        state.update(status="COMPLETE", finished_at_utc=now())
    except BaseException as error:
        state.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                     finished_at_utc=now(), error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(output / "run_state.json", state)
        write_json(output / "run_integrity.json", {"run_id": run_id, "status": state["status"],
                   "files": fingerprints(output)}, exclusive=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-logging", choices=("legacy_zero", "corrected"), required=True)
    args = parser.parse_args()
    result = run_audit(args.run_id, args.expected_logging)
    print(json.dumps({"run_id": args.run_id, "passed": result["passed"]}))
