"""Train a *pilot*, offline PDD steering/longitudinal-control baseline.

With --c0-speed, fit six future human speeds on the audited G2 TRAIN split
and compare against current-speed persistence on DEV.
Add --speed-mlp to compare an MLP neural network with the tree baseline.
With --c0-plan, learn six forward/left/speed points and compare a
single trajectory regressor with a learned mixture of trajectory experts.
Add --plan-nav --plan-factorized to cross-fit geometry and traffic residual heads.

Reads the exact CSVs produced by prepare_pdd_pilot.py:
  pdd_pilot_output/{train,val,test}.csv and metadata.json

Outputs to --out/<label>/:
  model_bundle.joblib     (feature scaler, trained regression model, target scaler)
  metrics.json            (validation and held-out test metrics)

LIMITATIONS:
  - Alignment and CARLA->MetaDrive transfer are NOT certified.
  - This is NOT a MetaDrive-ready agent. Do not feed raw 259D/19D obs into it.
  - No test data participates in model selection.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import gzip
import hashlib
import inspect
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import struct

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = {
    "at_t": ["steer_at_t_carla", "a2_at_t_metadrive_candidate"],
    "mean5": ["steer_mean_next5_carla", "a2_mean_next5_metadrive_candidate"],
}

C0_INPUT_HASHES = {
    "audit.json": "45d18ff9162d32cd34ee127ca1c2b5a28fc69c6318d1baf13b909340b5defa0d",
    "c0_screened_candidates.jsonl": "2e5c4f14b9386c91a220e5fabd0f1866c089bde7cfcc2a26b7312914f8ab6880",
}

PLAN_NAV_FEATURES = [
    f"nav_hint_{index}_{field}" for index in (1, 2) for field in ("unit_forward", "unit_left", "valid")
]
PLAN_LANE_FEATURES = ["lane_lateral_left_m", "lane_heading_error_rad", "lane_pose_valid"]
PLAN_CURVE_FEATURES = ["lane_curvature_2p5m_left_inv_m", "lane_curvature_2p5m_valid"]
CURVE_COORDINATE_FRAME = "CURRENT_LANE_QUADRATIC_V1"
CURVE_HELPER_HASHES = {
    "observed_lane_curvature": "48438295e9fcb54e8b84420ac9084204c0d5c0ccccfdf728dbcdfcd167cba219",
    "public_lane_curvature": "bb9410e2b340e39add9b3e887f5483ef7ea06679f09dbe9a81da223dc78f2b60",
}


def function_source_sha256(function):
    source = inspect.getsource(function).replace("\r\n", "\n").strip() + "\n"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def encode_plan_curvature(samples, lane_pose_valid):
    if len(samples) != 2 or [sample["distance_m"] for sample in samples] != [2.5, 5.]:
        raise ValueError("Expected audited 2.5/5m curvature samples")
    sample = samples[0]
    if type(sample["valid"]) is not bool:
        raise ValueError("Invalid curvature mask")
    if not sample["valid"]:
        if sample["curvature_left_inv_m"] is not None:
            raise ValueError("Masked curvature must be unavailable")
        return [0., 0.]
    value = float(sample["curvature_left_inv_m"])
    if not lane_pose_valid or not math.isfinite(value) or sample["flags"]:
        raise ValueError("Curvature cannot be valid without a valid source pose")
    return [value, 1.]


def public_plan_curvature(snapshot, lane_pose_valid):
    from agent4_integration_audit.audit_compatibility import public_lane_curvature

    return encode_plan_curvature(public_lane_curvature(snapshot, pose_valid=bool(lane_pose_valid)),
                                 lane_pose_valid)


def add_c0_lane_curvature(splits, names, review, source_hashes, lane_contract):
    """Join the frozen current-map audit, retaining every human target and split."""
    from agent4_integration_audit import audit_compatibility as helper

    payload = (review / "report.json").read_bytes()
    report_hash = hashlib.sha256(payload).hexdigest()
    if report_hash != "c80d3fc783d92b07432b55763dd3bbd1e949878dbcedd1f0c740b3b5c40417e5":
        raise ValueError("Expected reviewed curvature audit 20261120T142403_036487Z")
    report = json.loads(payload)
    helper_hashes = {name: function_source_sha256(getattr(helper, name)) for name in CURVE_HELPER_HASHES}
    source = report["source_sha256"]
    if (report["schema"] != "PDD_CURRENT_LANE_CURVATURE_AUDIT_V1"
            or report["implementation_sha256"] != "ab4a428bad91beb5de64c5c5ad1585a73f495cca77406a11f5a8e7312d2ede3e"
            or report["contract"]["lookahead_m"] != [2.5, 5.]
            or report["contract"]["positive_direction"] != "left"
            or report["human_labels_changed"] or report["simulator_launched"]
            or helper_hashes != CURVE_HELPER_HASHES
            or source["runs/agent4_pdd_lane_map_review_01/audit.json"] != lane_contract["audit_sha256"]
            or any(source[f"runs/agent4_pdd_development_c0_02/{name}"] != value
                   for name, value in source_hashes.items())):
        raise ValueError("Curvature measurement contract or data lineage differs")
    artifact = review / "lane_curvature_review.jsonl"
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != report["artifact_sha256"]:
        raise ValueError("Curvature artifact checksum mismatch")
    measurements = {}
    with artifact.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["driver"], row["scenario"], row["telemetry_step"])
            if key in measurements or row["partition"] not in ("TRAIN", "DEV"):
                raise ValueError(f"Duplicate or invalid curvature identity: {key}")
            measurements[key] = row
    coverage = {}
    pose_index = names.index("lane_pose_valid")
    for split, (features, targets, rows) in splits.items():
        values = []
        for index, row in enumerate(rows):
            key = (row["driver"], row["scenario"], row["telemetry_step"])
            measured = measurements.pop(key)
            if measured["partition"] != split or measured["source_pose_valid"] != bool(features[index, pose_index]):
                raise ValueError(f"Curvature partition/pose mask mismatch: {key}")
            values.append(encode_plan_curvature(measured["samples"], measured["source_pose_valid"]))
        values = np.asarray(values, dtype=np.float64)
        valid = int(values[:, 1].sum())
        if len(rows) != report["pdd"][split]["rows"] or valid != report["pdd"][split]["valid_by_distance"]["2.5"]:
            raise ValueError(f"Curvature audit coverage mismatch: {split}")
        coverage[split] = {"observed": valid, "masked": len(rows)-valid,
                           "min": float(values[values[:, 1] == 1, 0].min()),
                           "max": float(values[values[:, 1] == 1, 0].max())}
        splits[split] = (np.column_stack((features, values)), targets, rows)
    if measurements:
        raise ValueError("Curvature audit contains examples outside frozen TRAIN/DEV")
    return names + PLAN_CURVE_FEATURES, {
        "audit_sha256": report_hash, "artifact_sha256": report["artifact_sha256"],
        "map_sha256": lane_contract["map_sha256"], "public_helper_sha256": helper_hashes,
        "sample_distance_m": 2.5, "positive_direction": "left", "missing_values": [0., 0.],
        "validity": coverage, "sharp_turn_training_examples_abs_k_ge_0p04":
            report["pdd"]["TRAIN"]["curvature_distributions"]["2.5"]["abs_ge_0.04"],
        "transfer_limit": "PDD C0 lacks sharp turns; geometric normalization is a hypothesis, not learned sharp-turn evidence",
    }


def encode_plan_lane_pose(pose, flags, lane_width=None):
    """Represent unavailable/review-flagged geometry explicitly, never as measured zero."""
    if pose is None or flags:
        return [0., 0., 0.]
    lateral, error = float(pose["lateral_left_m"]), float(pose["lane_heading_error_rad"])
    if not math.isfinite(lateral) or not math.isfinite(error) or abs(error) > math.pi:
        raise ValueError("Invalid lane pose measurement")
    if lane_width is not None:
        width = float(lane_width)
        if not math.isfinite(width) or not 0 < width <= 6:
            raise ValueError("Invalid lane width for lane-pose validity")
        if abs(lateral) > width / 2 + .05:
            return [0., 0., 0.]
    return [lateral, error, 1.]


def public_plan_lane_pose(snapshot):
    from build_agent4_pdd_pilot import public_lane_pose

    pose, flags = public_lane_pose(snapshot)
    if pose is None or flags:
        return encode_plan_lane_pose(pose, flags)
    current = next(lane for lane in snapshot["task_context"]["lane_context"]["lanes"] if lane["side"] == "CURRENT")
    return encode_plan_lane_pose(pose, flags, current["width_m"])


def add_c0_lane_pose(splits, names, review, source_hashes):
    """Use the frozen, reviewed current-frame map measurements, preserving human labels."""
    report_bytes = (review / "audit.json").read_bytes()
    report_hash = hashlib.sha256(report_bytes).hexdigest()
    if report_hash != "8cfdd7c518daf6b530dbc7b3138feb32cd04583fbeebc58177d2ef1e505004ac":
        raise ValueError("Expected the reviewed lane-map audit 01")
    report = json.loads(report_bytes)
    helper_hash = hashlib.sha256(Path(__file__).with_name("build_agent4_pdd_pilot.py").read_bytes()).hexdigest()
    if (report["schema"] != "PDD_TOWN12_LANE_POSE_AUDIT_V1"
            or report["input_hashes"] != source_hashes or report["implementation_sha256"] != helper_hash
            or report["map_sha256"] != "26fa53d4902ac3397d35f948969f0c77c1bf023755ede2c67edf00bef817151f"
            or report["carla_api_version"] != "0.9.15" or not report["uses_current_frame_only"]
            or report["human_labels_changed"]):
        raise ValueError("Incompatible lane-map source, helper or measurement contract")
    review_path = review / "lane_pose_review.jsonl"
    if hashlib.sha256(review_path.read_bytes()).hexdigest() != report["review_file_sha256"]:
        raise ValueError("Lane pose review file checksum mismatch")
    measurements = {}
    with review_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            identity = (row["driver"], row["scenario"], row["telemetry_step"])
            if identity in measurements or type(row["valid"]) is not bool:
                raise ValueError(f"Duplicate or invalid lane measurement: {identity}")
            if row["valid"] != (row["pose_candidate"] is not None and not row["flags"]):
                raise ValueError(f"Inconsistent lane validity: {identity}")
            if row["flags"] not in ([], ["EGO_OUTSIDE_PROJECTED_LANE"]):
                raise ValueError(f"Unreviewed lane-map error: {identity}")
            measurements[identity] = row
    validity = {}
    for split, (features, targets, rows) in splits.items():
        poses = []
        for row in rows:
            identity = (row["driver"], row["scenario"], row["telemetry_step"])
            measured = measurements.pop(identity)
            if measured["partition"] != split:
                raise ValueError(f"Lane measurement crosses partition: {identity}")
            poses.append(encode_plan_lane_pose(measured["pose_candidate"], measured["flags"],
                                              None if measured["map_match"] is None else measured["map_match"]["width_m"]))
        poses = np.asarray(poses, dtype=np.float64)
        valid = int(poses[:, 2].sum())
        if valid != report["summary"][split]["matched_rows"] or len(rows) != report["summary"][split]["rows"]:
            raise ValueError(f"Lane observation coverage differs from audit: {split}")
        validity[split] = {"observed": valid, "masked": len(rows) - valid}
        splits[split] = (np.column_stack((features, poses)), targets, rows)
    if measurements:
        raise ValueError("Lane review contains examples outside the C0 dataset")
    return names + PLAN_LANE_FEATURES, {
        "audit_sha256": report_hash, "review_file_sha256": report["review_file_sha256"],
        "map_sha256": report["map_sha256"], "helper_sha256": helper_hash, "validity": validity,
        "missing_values": [0., 0., 0.],
        "mask_policy": "outside lane half-width plus 0.05m, missing projection, or parity flags: lane unavailable; human labels retained",
    }


def encode_plan_navigation(hints):
    """Two observed navigation bearings, independent of the distance unit."""
    result = []
    for hint in hints:
        if hint is None:
            result.extend((0., 0., 0.))
            continue
        x, y = map(float, hint)
        distance = math.hypot(x, y)
        if not math.isfinite(distance):
            raise ValueError("Nonfinite navigation hint")
        result.extend((x / distance, y / distance, 1.) if distance > 1e-6 else (0., 0., 0.))
    if len(result) != 6:
        raise ValueError("Expected exactly two navigation hints")
    return result


def public_plan_navigation(snapshot):
    """Use actual valid public waypoints, including the visible route tail."""
    task = snapshot["task_context"]
    observed = [point for point, valid in zip(task["waypoints"], task["validity_mask"]) if valid]
    points = [min(observed, key=lambda p: abs(p["lookahead_distance_m"] - distance)) if observed else None
              for distance in (25., 45.)]
    hints = [None if point is None else (point["relative_x"], point["relative_y"]) for point in points]
    distances = [None if point is None else point["lookahead_distance_m"] for point in points]
    return encode_plan_navigation(hints), distances


def load_c0_speed(folder):
    from agent4_pdd_common_v2 import FEATURE_NAMES, SCHEMA
    from build_agent4_pdd_pilot import REVISION, episode_fingerprint, partition

    hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in C0_INPUT_HASHES}
    audit = json.loads((folder / "audit.json").read_bytes())
    if hashes != C0_INPUT_HASHES:
        if (audit.get("dataset_scope") != "PDD_TRAIN_DEV_EXPANDED_C0"
                or audit.get("pilot_drivers") != list(range(1, 26))
                or audit.get("candidate_file_sha256") != hashes["c0_screened_candidates.jsonl"]):
            raise ValueError("Expected the reviewed G2 pilot or a hashed expanded TRAIN/DEV export")
    screen = audit["c0_geometry_screen"]
    if (audit["revision_from_readme_cache"] != REVISION
            or audit["unexpected_alignment_or_source_errors"] or audit["source_error_counts"]
            or not screen["exported"] or screen["ruleset"] != "C0_VEHICLE_BOX_OVERLAP_V1"
            or audit["horizons_s_provisional"] != [.25, .5, .75, 1., 1.25, 1.5]):
        raise ValueError("Unresolved data-source errors or incompatible C0 screening")
    encoder_hash = hashlib.sha256(Path(__file__).with_name("agent4_pdd_common_v2.py").read_bytes()).hexdigest()
    if (audit["schema"] != SCHEMA or audit["feature_names"] != list(FEATURE_NAMES)
            or encoder_hash != audit["implementation_sha256"]["agent4_pdd_common_v2.py"]):
        raise ValueError("Encoder differs from the reviewed G2 schema")
    names = [name for name in FEATURE_NAMES if not name.startswith("route_hint_")]
    if len(names) != 90 or names != list(FEATURE_NAMES[:90]):
        raise ValueError("Unexpected feature order for PDD_C0_SPEED_V1")
    rows = {"TRAIN": [], "DEV": []}
    episodes = defaultdict(list)
    seen = set()
    drivers = {f"driver_{driver:02d}": partition(driver) for driver in audit["pilot_drivers"]}
    with (folder / "c0_screened_candidates.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            split = row["partition"]
            identity = (row["driver"], row["scenario"], row["telemetry_step"])
            if drivers.get(row["driver"]) != split or split not in rows or identity in seen:
                raise ValueError(f"Invalid split or duplicate example: {identity}")
            if len(row["features"]) != 96:
                raise ValueError(f"Invalid source feature dimension: {identity}")
            seen.add(identity)
            rows[split].append(row)
            episodes[f'{row["driver"]}/{row["scenario"]}'].append(row)
    if {split: len(group) for split, group in rows.items()} != screen["retained_by_partition"]:
        raise ValueError("Sample counts differ from the C0 audit")
    for receipt in audit["episodes"]:
        group = episodes.pop(receipt["episode"], [])
        if (len(group) != receipt["c0_geometry"]["retained_count"]
                or episode_fingerprint(group) != receipt["c0_geometry"]["retained_sha256"]):
            raise ValueError(f"Episode fingerprint differs: {receipt['episode']}")
    if episodes:
        raise ValueError("Export contains episodes absent from its audit")
    splits = {}
    for split, group in rows.items():
        features = np.array([r["features"][:90] for r in group], dtype=np.float64)
        speeds = np.array([[point[2] for point in r["human_trajectory"]] for r in group])
        if (speeds.shape != (len(group), 6) or not np.isfinite(speeds).all()
                or not np.isfinite(features).all() or (speeds < 0).any()
                or not all(r["target_mask"] == [True] * 6 for r in group)):
            raise ValueError(f"Invalid speed examples in {split}")
        splits[split] = (features, speeds, group)
    return splits, names, audit


def c0_speed_metrics(actual, predicted, masks):
    errors = np.abs(predicted - actual)
    return {name: {"n": int(mask.sum()),
                   "mae_mps": float(errors[mask].mean()) if mask.any() else None,
                   "mae_by_horizon_mps": errors[mask].mean(axis=0).tolist() if mask.any() else None}
            for name, mask in masks.items()}


def add_c0_speed_history(splits, names, audit, pdd):
    """Append the speed change since t-0.5s and its validity in the same episode."""
    from build_agent4_pdd_pilot import read_source, xyz

    receipts = {entry["episode"]: entry for entry in audit["episodes"]}
    for split, (features, targets, rows) in splits.items():
        history = np.zeros((len(rows), 2))
        episodes = defaultdict(list)
        for index, row in enumerate(rows):
            episodes[(row["driver"], row["scenario"])].append(index)
        for (driver, scenario), indices in episodes.items():
            source = f"{driver}/data/{scenario}/metric/metric_info.json"
            inventory = {}
            telemetry = read_source(pdd / source, inventory, pdd)
            receipt = receipts[f"{driver}/{scenario}"]
            if receipt["partition"] != split or inventory[source] != receipt["source_sha256"][source]:
                raise ValueError(f"History source differs from the reviewed {split} episode: {source}")
            for i in indices:
                step = rows[i]["telemetry_step"]
                current = features[i, names.index("speed_mps")]
                actual = math.hypot(*xyz(telemetry[str(step)]["velocity"])[:2])
                if not math.isclose(actual, current, abs_tol=1e-5, rel_tol=1e-6):
                    raise ValueError(f"Current speed mismatch in {source}, step {step}")
                previous = telemetry.get(str(step - 10))
                if previous is not None:
                    previous_speed = math.hypot(*xyz(previous["velocity"])[:2])
                    history[i] = (current - previous_speed, 1.0)
        splits[split] = (np.column_stack((features, history)), targets, rows)
    return names + ["speed_change_past_0p5s_mps", "speed_change_past_0p5s_valid"]


def train_c0_speed(folder, out, seed, history_root=None, *, speed_mlp=False, max_iter=250):
    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    splits, names, audit = load_c0_speed(folder)
    temporal = history_root is not None
    if temporal:
        names = add_c0_speed_history(splits, names, audit, history_root)
    train_x, train_y, _ = splits["TRAIN"]
    dev_x, dev_y, dev_rows = splits["DEV"]
    speed = names.index("speed_mps")
    current = dev_x[:, speed:speed + 1]
    change = dev_y[:, -1] - current[:, 0]
    # Actor centres in the forward ego-lane-width corridor at t. This is
    # measurable from public features and does not assert same-lane identity.
    actor_x = dev_x[:, [names.index(f"actor_{slot}_x_m") for slot in range(8)]]
    actor_y = dev_x[:, [names.index(f"actor_{slot}_y_left_m") for slot in range(8)]]
    valid = dev_x[:, [names.index(f"actor_{slot}_valid") for slot in range(8)]] == 1
    width = dev_x[:, names.index("lane_width_m")]
    ahead = (valid & (actor_x > 0) & (np.abs(actor_y) <= width[:, None] / 2)).any(axis=1)
    ahead &= dev_x[:, names.index("lane_width_valid")] == 1
    masks = {"all": np.ones(len(dev_x), dtype=bool), "slowdown": change <= -1.0,
             "speedup": change >= 1.0, "forward_corridor_slowdown": ahead & (change <= -1.0)}
    for field in ("driver", "scenario"):
        values = np.array([row[field] for row in dev_rows])
        masks.update({f"{field}/{value}": values == value for value in np.unique(values)})

    scores = {"persistence": c0_speed_metrics(dev_y, current, masks)}
    history_contract = {}
    if temporal:
        horizons = np.array(audit["horizons_s_provisional"])
        delta = dev_x[:, names.index("speed_change_past_0p5s_mps")]
        projected = np.maximum(0.0, current + delta[:, None] * (horizons / 0.5))
        scores["constant_acceleration"] = c0_speed_metrics(dev_y, projected, masks)
        history_contract = {
            "history_seconds": 0.5, "history_missing_values": [0.0, 0.0],
            "history_source": "Same episode planar ego speed at t minus speed at t-0.5s; public ticks t and t-5 at 10 Hz.",
            "history_valid_rows": {k: int(v[0][:, -1].sum()) for k, v in splits.items()},
        }
    models = {
        "extra_trees": ExtraTreesRegressor(n_estimators=96, min_samples_leaf=8, random_state=seed, n_jobs=-1),
    }
    if not temporal:
        models = {"ridge": Pipeline([("scale", StandardScaler()), ("regressor", Ridge(alpha=10.0))]), **models}
    if speed_mlp:
        models["mlp"] = TransformedTargetRegressor(
            regressor=Pipeline([
                ("scale", StandardScaler()),
                ("regressor", MLPRegressor(
                    hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
                    max_iter=max_iter, early_stopping=False, random_state=seed,
                    alpha=0.01, batch_size=128, learning_rate_init=0.001,
                    n_iter_no_change=30,
                )),
            ]),
            transformer=StandardScaler(),
        )
    fitting = {}
    for name, model in models.items():
        print(f"Fitting {name} on {len(train_x)} TRAIN rows...", flush=True)
        model.fit(train_x, train_y - train_x[:, speed:speed + 1])
        prediction = np.maximum(0.0, current + model.predict(dev_x))
        if prediction.shape != dev_y.shape or not np.isfinite(prediction).all():
            raise ValueError(f"{name} produced invalid DEV predictions")
        scores[name] = c0_speed_metrics(dev_y, prediction, masks)
        if name == "mlp":
            network = model.regressor_.named_steps["regressor"]
            fitting[name] = {
                "family": "NEURAL_NETWORK", "hidden_layers": [128, 64], "activation": "relu",
                "input_scaler_fit": "TRAIN", "target_scaler_fit": "TRAIN",
                "solver": "adam", "batch_size": 128, "alpha": 0.01,
                "learning_rate_init": 0.001, "max_iter": max_iter,
                "n_iter": int(network.n_iter_), "loss": float(network.loss_),
                "reached_max_iter": network.n_iter_ >= max_iter,
            }
        else:
            estimator = model.named_steps["regressor"] if name == "ridge" else model
            fitting[name] = {"family": "TREE_ENSEMBLE" if name == "extra_trees" else "LINEAR",
                             "parameters": estimator.get_params(deep=False)}
        print(f"{name} DEV MAE={scores[name]['all']['mae_mps']:.6f} m/s", flush=True)
    required = ("all", "slowdown", "forward_corridor_slowdown")
    def beats_persistence(name):
        return all(masks[group].any() and scores[name][group]["mae_mps"] < scores["persistence"][group]["mae_mps"]
                   for group in required)

    eligible = [name for name in models if beats_persistence(name)]
    selected = min(eligible or models, key=lambda name: scores[name]["all"]["mae_mps"])
    status = "DEV_IMPROVED" if eligible else "NEEDS_FIX"
    contract = {
        "schema": "PDD_C0_SPEED_HISTORY_V1" if temporal else "PDD_C0_SPEED_V1",
        "source_schema": audit["schema"], "features": names, **history_contract,
        "encoder_sha256": audit["implementation_sha256"]["agent4_pdd_common_v2.py"],
        "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hashes": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                          for name in C0_INPUT_HASHES},
        "horizons_s": audit["horizons_s_provisional"],
        "prediction": "max(0, speed_mps_at_t + model.predict(features))",
        "output_unit": "m/s", "selected_model": selected, "offline_status": status,
        "model_family": fitting[selected]["family"],
        "scope": "OFFLINE_DIAGNOSTIC_UNREVIEWED_HUMAN_BEHAVIOR",
    }
    report = {**contract, "rows": {k: len(v[0]) for k, v in splits.items()},
              "drivers": {k: sorted({r["driver"] for r in v[2]}) for k, v in splits.items()},
              "seed": seed, "dev": scores, "fitting": fitting, "required_dev_groups": required,
              "rule": "Strictly improve all required DEV group MAEs over persistence, then minimize all-row MAE.",
              "closed_loop_tested": False}
    # Publish checkpoint and metrics together only after verifying a reload.
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as temp:
        stage = Path(temp) / "result"
        stage.mkdir()
        checkpoint = stage / "model_bundle.joblib"
        joblib.dump({**contract, "model": models[selected]}, checkpoint, compress=3)
        restored = joblib.load(checkpoint)["model"]
        np.testing.assert_allclose(restored.predict(dev_x[:8]), models[selected].predict(dev_x[:8]))
        (stage / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        stage.rename(out)
    for group in required:
        baseline = scores["persistence"][group]
        learned = scores[selected][group]
        print(f"{group} (n={learned['n']}): persistence={baseline['mae_mps']} {selected}={learned['mae_mps']} m/s")
    print(f"{status} | {out / 'metrics.json'}")
    print(f"Saved {selected} ({contract['model_family']}): {out / 'model_bundle.joblib'}")


def plan_reference(features, names, horizons):
    reference = np.zeros((len(features), len(horizons), 3))
    reference[:, :, 0] = features[:, names.index("velocity_forward_mps"), None] * horizons
    reference[:, :, 1] = features[:, names.index("velocity_left_mps"), None] * horizons
    reference[:, :, 2] = features[:, names.index("speed_mps"), None]
    return reference


def transform_plan_lane_frame(plan, features, lane_indices, *, inverse=False):
    """Rigid transform at the current lane projection; no future map geometry."""
    result = np.array(plan, dtype=np.float64, copy=True)
    lateral_index, heading_index, valid_index = lane_indices
    valid = features[:, valid_index] == 1.
    if not valid.any():
        return result
    lateral = features[valid, lateral_index, None]
    heading = features[valid, heading_index, None]
    cosine, sine = np.cos(heading), np.sin(heading)
    points = np.asarray(plan)[valid]
    x, y = points[:, :, 0], points[:, :, 1]
    if inverse:
        result[valid, :, 0] = cosine * x - sine * (y - lateral)
        result[valid, :, 1] = sine * x + cosine * (y - lateral)
    else:
        result[valid, :, 0] = cosine * x + sine * y
        result[valid, :, 1] = -sine * x + cosine * y + lateral
    return result


def plan_model_reference(reference, features, candidate):
    if candidate.get("coordinate_frame") != "CURRENT_LANE_TANGENT_V1":
        return reference
    indices = candidate["lane_indices"]
    result = transform_plan_lane_frame(reference, features, indices)
    # The lateral head learns the recorded future lane offset directly.
    # Missing lane geometry retains the original body-frame residual target.
    result[features[:, indices[2]] == 1., :, 1] = 0.
    return result


def predict_c0_plan(candidate, features, reference):
    """Predict a complete plan from current observations and a physical reference."""
    if candidate["kind"] == "single":
        residual = candidate["estimator"].predict(features)
    elif candidate["kind"] == "factorized":
        residual = candidate["base_estimator"].predict(features[:, candidate["geometry_indices"]])
        traffic = features[:, candidate["traffic_count_index"]] > 0
        if traffic.any():
            residual[traffic] += candidate["traffic_estimator"].predict(features[traffic])
    elif candidate["kind"] == "mixture":
        selected = candidate["classifier"].predict(features)
        residual = np.empty((len(features), reference.shape[1] * 3))
        for mode, expert in enumerate(candidate["experts"]):
            selected_rows = selected == mode
            if selected_rows.any():
                residual[selected_rows] = expert.predict(features[selected_rows])
    else:
        raise ValueError(f"Unknown plan model kind: {candidate['kind']}")
    model_reference = plan_model_reference(reference, features, candidate)
    plan = model_reference + residual.reshape(reference.shape)
    if candidate.get("coordinate_frame") == "CURRENT_LANE_TANGENT_V1":
        plan = transform_plan_lane_frame(plan, features, candidate["lane_indices"], inverse=True)
    plan[:, :, 2] = np.maximum(0.0, plan[:, :, 2])
    if not np.isfinite(plan).all():
        raise ValueError("Nonfinite trajectory predictions")
    return plan


def transform_plan_lane_curve(plan, features, lane_indices, curvature_indices, *, inverse=False):
    """Invertible local bend normalization; this is not a future lane-centre label.

    In current lane tangent coordinates, z_y = q_y - 0.5*k*q_x**2.
    The quadratic is a local approximation of observed road geometry. Unlike
    circular/Frenet inversion it has no branch ambiguity or clipped domain.
    Missing curvature retains tangent coordinates; missing pose retains ego axes.
    """
    features = np.asarray(features, dtype=np.float64)
    valid_lane = features[:, lane_indices[2]] == 1.
    curvature_index, valid_index = curvature_indices
    valid_curve = features[:, valid_index] == 1.
    if (not np.isin(features[:, valid_index], (0., 1.)).all()
            or (valid_curve & ~valid_lane).any() or not np.isfinite(features[:, curvature_index]).all()):
        raise ValueError("Invalid curvature/pose contract")
    tangent = np.array(plan, dtype=np.float64, copy=True) if inverse else transform_plan_lane_frame(
        plan, features, lane_indices)
    bend = .5 * features[valid_curve, curvature_index, None] * tangent[valid_curve, :, 0] ** 2
    tangent[valid_curve, :, 1] += bend if inverse else -bend
    return transform_plan_lane_frame(tangent, features, lane_indices, inverse=True) if inverse else tangent


def curve_plan_reference(reference, features, candidate):
    encoded = transform_plan_lane_curve(reference, features, candidate["lane_indices"], candidate["curvature_indices"])
    encoded[features[:, candidate["lane_indices"][2]] == 1., :, 1] = 0.
    return encoded


def predict_plan_candidate(candidate, features, reference):
    """Keep legacy decoders unchanged; explicitly dispatch the new representation."""
    if candidate.get("coordinate_frame") != CURVE_COORDINATE_FRAME:
        return predict_c0_plan(candidate, features, reference)
    if candidate["kind"] != "single":
        raise ValueError("The curvature pilot requires a single learned trajectory model")
    residual = candidate["estimator"].predict(features)
    encoded = curve_plan_reference(reference, features, candidate) + residual.reshape(reference.shape)
    plan = transform_plan_lane_curve(encoded, features, candidate["lane_indices"],
                                     candidate["curvature_indices"], inverse=True)
    plan[:, :, 2] = np.maximum(0., plan[:, :, 2])
    if not np.isfinite(plan).all():
        raise ValueError("Nonfinite curvature-normalized trajectory prediction")
    return plan


def plan_metrics(actual, predicted, masks):
    displacement = np.linalg.norm(predicted[:, :, :2] - actual[:, :, :2], axis=2)
    speed_error = np.abs(predicted[:, :, 2] - actual[:, :, 2])
    lateral_error = np.abs(predicted[:, :, 1] - actual[:, :, 1])
    result = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        result[name] = {
            "n": count,
            "ade_m": float(displacement[mask].mean()) if count else None,
            "fde_m": float(displacement[mask, -1].mean()) if count else None,
            "lateral_mae_m": float(lateral_error[mask].mean()) if count else None,
            "speed_mae_mps": float(speed_error[mask].mean()) if count else None,
            "displacement_by_horizon_m": displacement[mask].mean(axis=0).tolist() if count else None,
            "speed_mae_by_horizon_mps": speed_error[mask].mean(axis=0).tolist() if count else None,
        }
    return result


def probe_lane04_runtime_observations(candidate, names, horizons):
    """Unlabelled diagnostic predictions on old public states; never fit/select on them."""
    from agent4_pdd_common_v2 import encode_public

    root = Path(__file__).resolve().parent
    run = root / "agent4_integration_audit/run_20261120T131342_567885Z"
    payload = (run / "report.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != "1a5eaadc718dea4ce88f250181cb0a4a474e11cef8d610e26210518fa3c7a20b":
        raise ValueError("Expected frozen lane_04 TRAIN observation probe")
    previous = json.loads(payload)
    cases = []
    for pair in previous["pairs"]:
        case = pair["case_id"]
        folder = run / "_".join(case.split("/")[1:])
        record = json.loads((folder / "report.json").read_bytes())["agents"][previous["policy"]]
        trace = folder / "launcher" / previous["policy"] / "agent_decisions.jsonl.gz"
        trace_hash = hashlib.sha256(trace.read_bytes()).hexdigest()
        if trace_hash != record["trace_sha256"] or record["episode"]["split"] != "TRAIN":
            raise ValueError(f"Runtime probe trace integrity mismatch: {case}")
        features, logged, history = [], [], deque(maxlen=5)
        with gzip.open(trace, "rt", encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                if row["case_id"] != case or row["step_index"] != index:
                    raise ValueError("Unaligned runtime probe observations")
                snapshot, diagnostics = row["agent_input"], row["decision"]["diagnostics"]
                values = encode_public(snapshot)[:90]
                speed = values[0] if values[3] else None
                past = history[0] if len(history) == 5 else None
                history.append(speed)
                values += list((speed-past, 1.) if speed is not None and past is not None else (0., 0.))
                values += public_plan_navigation(snapshot)[0] + public_plan_lane_pose(snapshot)
                if hashlib.sha256(struct.pack(f"<{len(values)}f", *values)).hexdigest() != diagnostics["pdd_feature_sha256"]:
                    raise ValueError(f"Runtime probe feature parity mismatch: {case}/{index}")
                if candidate.get("coordinate_frame") == CURVE_COORDINATE_FRAME:
                    values += public_plan_curvature(snapshot, values[-1])
                features.append(values)
                logged.append(diagnostics["pdd_plan"])
        if len(features) != int(record["episode"]["episode_steps"]):
            raise ValueError(f"Incomplete runtime probe: {case}")
        features = np.asarray(features, dtype=np.float64)
        predicted = predict_plan_candidate(candidate, features, plan_reference(features, names, horizons))
        sample_steps = sorted({100, 250, 300, len(features)-1} & set(range(len(features))))
        samples = [{"step": step, "lane_valid": bool(features[step, names.index("lane_pose_valid")]),
                    "lateral_left_m": float(features[step, names.index("lane_lateral_left_m")]),
                    "heading_error_rad": float(features[step, names.index("lane_heading_error_rad")]),
                    "old_y_1s_m": logged[step][3][1] if logged[step] is not None else None,
                    "predicted_y_1s_m": float(predicted[step, 3, 1]),
                    "predicted_y_1p5s_m": float(predicted[step, 5, 1]),
                    "predicted_speed_1s_mps": float(predicted[step, 3, 2])}
                   for step in sample_steps if features[step, names.index("ego_valid")] == 1.]
        case_report = {"case_id": case, "rows": len(features), "trace_sha256": trace_hash, "samples": samples}
        if candidate.get("coordinate_frame") == CURVE_COORDINATE_FRAME:
            curvature_index, valid_index = candidate["curvature_indices"]
            lower, upper = candidate["curvature_train_range"]
            observed = features[:, valid_index] == 1.
            outside = observed & ((features[:, curvature_index] < lower) | (features[:, curvature_index] > upper))
            case_report["curvature_support"] = {"observed": int(observed.sum()),
                "outside_train_range": int(outside.sum()), "train_range_left_inv_m": [lower, upper],
                "clipped_to_training_range": False, "sharp_turn_generalization_certified": False}
        cases.append(case_report)
        for sample in samples:
            print(f"Unlabelled probe {case} step={sample['step']}: old y1s={sample['old_y_1s_m']} m, "
                  f"new y1s={sample['predicted_y_1s_m']:.4f} m, lane_valid={sample['lane_valid']}", flush=True)
    return {"scope": "FROZEN_TRAIN_PUBLIC_OBSERVATIONS_ONLY", "used_for_fit": False,
            "used_for_automatic_selection": False,
            "has_future_ground_truth": False, "closed_loop_test": False, "cases": cases}


def fit_factorized_plan(features, targets, rows, names, seed):
    """Fit geometry and traffic effects with driver-disjoint out-of-fold residuals."""
    geometry_indices = [i for i, name in enumerate(names)
                        if not name.startswith("actor_") and name not in ("active_count_over_8", "log1p_overflow_count")]
    geometry = features[:, geometry_indices]
    groups = np.array([row["driver"] for row in rows])
    out_of_fold = np.empty_like(targets)
    folds = []
    for index, (fit, held) in enumerate(GroupKFold(n_splits=3).split(geometry, groups=groups)):
        fit_drivers, held_drivers = sorted(set(groups[fit])), sorted(set(groups[held]))
        if set(fit_drivers) & set(held_drivers):
            raise ValueError("Driver overlap in geometry cross-fitting")
        print(f"Cross-fitting geometry fold {index+1}/3...", flush=True)
        estimator = TransformedTargetRegressor(
            regressor=ExtraTreesRegressor(n_estimators=96, min_samples_leaf=8, random_state=seed, n_jobs=-1),
            transformer=StandardScaler(),
        )
        estimator.fit(geometry[fit], targets[fit])
        out_of_fold[held] = estimator.predict(geometry[held])
        folds.append({"fit_drivers": fit_drivers, "held_drivers": held_drivers})
    base = TransformedTargetRegressor(
        regressor=ExtraTreesRegressor(n_estimators=96, min_samples_leaf=8, random_state=seed, n_jobs=-1),
        transformer=StandardScaler(),
    )
    print("Fitting geometry head on all TRAIN drivers...", flush=True)
    base.fit(geometry, targets)
    count_index = names.index("active_count_over_8")
    traffic = features[:, count_index] > 0
    if not traffic.any():
        raise ValueError("No observed traffic examples for the traffic head")
    correction = TransformedTargetRegressor(
        regressor=ExtraTreesRegressor(n_estimators=96, min_samples_leaf=8, random_state=seed, n_jobs=-1),
        transformer=StandardScaler(),
    )
    print(f"Fitting learned traffic residual on {int(traffic.sum())} TRAIN rows...", flush=True)
    correction.fit(features[traffic], (targets - out_of_fold)[traffic])
    return {"kind": "factorized", "base_estimator": base, "traffic_estimator": correction,
            "geometry_indices": geometry_indices, "traffic_count_index": count_index}, folds


def train_c0_plan(folder, out, seed, history_root, *, navigation=False, factorized=False,
                  lane_review=None, lane_frame=False, curvature_review=None):
    if factorized and not navigation:
        raise ValueError("Factorized plans require observed navigation bearings")
    if lane_review is not None and (not navigation or factorized):
        raise ValueError("Lane plans require navigation and a single trajectory model")
    if lane_frame and lane_review is None:
        raise ValueError("Lane-frame targets require audited current lane observations")
    if curvature_review is not None and not lane_frame:
        raise ValueError("Curvature normalization requires audited lane-frame targets")
    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    splits, names, audit = load_c0_speed(folder)
    names = add_c0_speed_history(splits, names, audit, history_root)
    if navigation:
        for split, (features, targets, rows) in splits.items():
            bearings = []
            for row in rows:
                raw = row["features"]
                hints = [(raw[i], raw[i + 1]) if raw[i + 2] == 1 else None for i in (90, 93)]
                bearings.append(encode_plan_navigation(hints))
            splits[split] = (np.column_stack((features, bearings)), targets, rows)
        names = names + PLAN_NAV_FEATURES
    lane_contract = None
    if lane_review is not None:
        hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in C0_INPUT_HASHES}
        names, lane_contract = add_c0_lane_pose(splits, names, lane_review, hashes)
    curve_contract = None
    if curvature_review is not None:
        names, curve_contract = add_c0_lane_curvature(splits, names, curvature_review, hashes, lane_contract)
    train_x, _, train_rows = splits["TRAIN"]
    dev_x, _, dev_rows = splits["DEV"]
    horizons = np.array(audit["horizons_s_provisional"])
    train_y = np.array([row["human_trajectory"] for row in train_rows], dtype=np.float64)
    dev_y = np.array([row["human_trajectory"] for row in dev_rows], dtype=np.float64)
    for target, features in ((train_y, train_x), (dev_y, dev_x)):
        if target.shape != (len(features), 6, 3) or not np.isfinite(target).all():
            raise ValueError("Expected six finite human forward/left/speed targets")
    train_reference = plan_reference(train_x, names, horizons)
    dev_reference = plan_reference(dev_x, names, horizons)
    residuals = (train_y - train_reference).reshape(len(train_y), 18)
    representation = {}
    if lane_frame:
        indices = [names.index(name) for name in PLAN_LANE_FEATURES]
        representation = {"coordinate_frame": "CURRENT_LANE_TANGENT_V1", "lane_indices": indices}
        if curve_contract is not None:
            representation.update(coordinate_frame=CURVE_COORDINATE_FRAME,
                                  curvature_indices=[names.index(name) for name in PLAN_CURVE_FEATURES],
                                  curvature_train_range=[curve_contract["validity"]["TRAIN"]["min"],
                                                         curve_contract["validity"]["TRAIN"]["max"]])
        def encode_target(target, features, inverse=False):
            if curve_contract is not None:
                return transform_plan_lane_curve(target, features, indices, representation["curvature_indices"], inverse=inverse)
            return transform_plan_lane_frame(target, features, indices, inverse=inverse)
        for target, features in ((train_y, train_x), (dev_y, dev_x)):
            encoded = encode_target(target, features)
            np.testing.assert_allclose(encode_target(encoded, features, inverse=True), target, rtol=0., atol=1e-10)
        encoded_train = encode_target(train_y, train_x)
        model_reference = (curve_plan_reference(train_reference, train_x, representation) if curve_contract is not None else
                           plan_model_reference(train_reference, train_x, representation))
        residuals = (encoded_train - model_reference).reshape(len(train_y), 18)

    if lane_review is not None:
        label = "bend-normalized" if curve_contract is not None else "lane-observed"
        print(f"Fitting {label} single trajectory model on {len(train_x)} TRAIN rows ({len(names)}D)...", flush=True)
        estimator = TransformedTargetRegressor(
            regressor=ExtraTreesRegressor(n_estimators=96, min_samples_leaf=8, random_state=seed, n_jobs=-1),
            transformer=StandardScaler(),
        )
        estimator.fit(train_x, residuals)
        candidates = {"single_plan": {"kind": "single", "estimator": estimator, **representation}}
        mode_report = {"lane_pose_contract": lane_contract}
        if lane_frame:
            mode_report["target_encoding"] = {
                "frame": "CURRENT_LANE_TANGENT_V1", "missing_lane_frame": "EGO_FORWARD_LEFT",
                "lateral_target": "recorded future offset in current lane tangent frame",
                "inverse_roundtrip_verified": True, "inverse_roundtrip_atol_m": 1e-10,
                "external_output_frame": "EGO_FORWARD_LEFT", "human_labels_changed": False,
            }
            if curve_contract is not None:
                mode_report["curvature_contract"] = curve_contract
                mode_report["target_encoding"].update(
                    frame=CURVE_COORDINATE_FRAME, missing_curvature_frame="CURRENT_LANE_TANGENT_V1",
                    lateral_target="human q_y minus 0.5 * observed current curvature * human q_x squared",
                    approximation="local quadratic bend, not exact circular/Frenet coordinates",
                    future_map_used=False,
                )
    elif factorized:
        candidate, folds = fit_factorized_plan(train_x, residuals, train_rows, names, seed)
        candidates = {"factorized_plan": candidate}
        mode_report = {"crossfit_folds": folds,
                       "geometry_features": [names[i] for i in candidate["geometry_indices"]]}
    else:
        candidates, mode_report = fit_trajectory_modes(train_x, train_y, residuals, dev_x, dev_y, names, seed)

    return evaluate_c0_plans(folder, out, seed, splits, names, audit, dev_y, dev_reference,
                             candidates, mode_report, navigation)


def fit_trajectory_modes(train_x, train_y, residuals, dev_x, dev_y, names, seed):
    # Cluster TRAIN trajectories only. Modes describe recorded geometry and
    # speed change, with no claim about the legality of a lane change.
    bearing = np.arctan2(train_y[:, :, 1], np.maximum(1.0, train_y[:, :, 0]))
    changes = train_y[:, :, 2] - train_x[:, names.index("speed_mps"), None]
    mode_scaler = StandardScaler()
    mode_x = mode_scaler.fit_transform(np.column_stack((bearing, changes)))
    clusterer = KMeans(n_clusters=6, n_init=10, random_state=seed)
    modes = clusterer.fit_predict(mode_x)
    counts = np.bincount(modes, minlength=6)
    print(f"TRAIN geometry/speed modes: {counts.tolist()}", flush=True)

    classifier = ExtraTreesClassifier(n_estimators=96, min_samples_leaf=8,
                                      class_weight="balanced", random_state=seed, n_jobs=-1)
    classifier.fit(train_x, modes)
    experts = []
    for mode in range(6):
        mask = modes == mode
        print(f"Fitting trajectory expert {mode} on {int(mask.sum())} TRAIN rows...", flush=True)
        expert = TransformedTargetRegressor(
            regressor=ExtraTreesRegressor(n_estimators=96, min_samples_leaf=8, random_state=seed, n_jobs=-1),
            transformer=StandardScaler(),
        )
        expert.fit(train_x[mask], residuals[mask])
        experts.append(expert)
    print("Fitting single trajectory reference model...", flush=True)
    single = TransformedTargetRegressor(
        regressor=ExtraTreesRegressor(n_estimators=96, min_samples_leaf=8, random_state=seed, n_jobs=-1),
        transformer=StandardScaler(),
    )
    single.fit(train_x, residuals)
    candidates = {
        "single_plan": {"kind": "single", "estimator": single},
        "mixture_plan": {"kind": "mixture", "classifier": classifier, "experts": experts},
    }
    dev_bearings = np.arctan2(dev_y[:, :, 1], np.maximum(1.0, dev_y[:, :, 0]))
    dev_changes = dev_y[:, :, 2] - dev_x[:, names.index("speed_mps"), None]
    dev_modes = clusterer.predict(mode_scaler.transform(np.column_stack((dev_bearings, dev_changes))))
    mode_accuracy = float((classifier.predict(dev_x) == dev_modes).mean())
    return candidates, {
        "train_mode_counts": counts.tolist(), "dev_mode_accuracy": mode_accuracy,
        "train_mode_mean_endpoints": [train_y[modes == i, -1].mean(axis=0).tolist() for i in range(6)],
    }


def evaluate_c0_plans(folder, out, seed, splits, names, audit, dev_y, dev_reference,
                      candidates, mode_report, navigation):
    dev_x, _, dev_rows = splits["DEV"]
    lane_conditioned = "lane_pose_valid" in names
    coordinate_frame = candidates.get("single_plan", {}).get("coordinate_frame")
    curve_mode = coordinate_frame == CURVE_COORDINATE_FRAME
    lane_frame = curve_mode or coordinate_frame == "CURRENT_LANE_TANGENT_V1"
    horizons = np.array(audit["horizons_s_provisional"])
    change = dev_y[:, -1, 2] - dev_x[:, names.index("speed_mps")]
    final_bearing = np.arctan2(dev_y[:, -1, 1], np.maximum(1.0, dev_y[:, -1, 0]))
    width = dev_x[:, names.index("lane_width_m")]
    ahead = np.zeros(len(dev_x), dtype=bool)
    for slot in range(8):
        ahead |= ((dev_x[:, names.index(f"actor_{slot}_valid")] == 1)
                  & (dev_x[:, names.index(f"actor_{slot}_x_m")] > 0)
                  & (np.abs(dev_x[:, names.index(f"actor_{slot}_y_left_m")]) <= width / 2))
    ahead &= dev_x[:, names.index("lane_width_valid")] == 1
    masks = {
        "all": np.ones(len(dev_x), dtype=bool), "slowdown": change <= -1.,
        "speedup": change >= 1., "forward_corridor_slowdown": ahead & (change <= -1.),
        "left_geometry": final_bearing >= .03, "right_geometry": final_bearing <= -.03,
        "near_stop": (dev_y[:, -1, 2] <= .5) & (dev_x[:, names.index("speed_mps")] >= 2.),
        "no_observed_actors": dev_x[:, names.index("active_count_over_8")] == 0,
        "observed_actors": dev_x[:, names.index("active_count_over_8")] > 0,
    }
    if lane_conditioned:
        valid_lane = dev_x[:, names.index("lane_pose_valid")] == 1.
        masks.update(lane_pose_observed=valid_lane, lane_pose_missing=~valid_lane,
                     lane_deviation=valid_lane & ((np.abs(dev_x[:, names.index("lane_lateral_left_m")]) >= .5)
                                                 | (np.abs(dev_x[:, names.index("lane_heading_error_rad")]) >= .05)))
    if curve_mode:
        valid_curve = dev_x[:, names.index(PLAN_CURVE_FEATURES[1])] == 1.
        masks.update(curvature_observed=valid_curve, curvature_missing=~valid_curve)
    for field in ("driver", "scenario"):
        values = np.array([row[field] for row in dev_rows])
        masks.update({f"{field}/{value}": values == value for value in np.unique(values)})
    scores = {"constant_velocity": plan_metrics(dev_y, dev_reference, masks)}
    for name, candidate in candidates.items():
        scores[name] = plan_metrics(dev_y, predict_plan_candidate(candidate, dev_x, dev_reference), masks)
    if lane_frame:
        candidate = candidates["single_plan"]
        tangent_candidate = {**candidate, "coordinate_frame": "CURRENT_LANE_TANGENT_V1"}
        geometric_reference = transform_plan_lane_frame(
            plan_model_reference(dev_reference, dev_x, tangent_candidate), dev_x, candidate["lane_indices"], inverse=True)
        scores["lane_tangent_reference"] = plan_metrics(dev_y, geometric_reference, masks)
        if curve_mode:
            curve_reference = transform_plan_lane_curve(
                curve_plan_reference(dev_reference, dev_x, candidate), dev_x,
                candidate["lane_indices"], candidate["curvature_indices"], inverse=True)
            scores["lane_quadratic_reference"] = plan_metrics(dev_y, curve_reference, masks)
    reference_name = "lane04_reference" if lane_frame else "nav02_reference"
    if "factorized_plan" in candidates or lane_conditioned:
        # Frozen previous checkpoint is diagnostic only, never re-fitted/selected.
        reference_folder = "agent4_pdd_plan_lane_04" if lane_frame else "agent4_pdd_plan_nav_02"
        reference_path = Path(__file__).resolve().parent / "runs" / reference_folder / "model_bundle.joblib"
        reference_hash = ("8e278ae355b98d7df69011a4126139b24a1eeb86c1de07e3f2a96bb8b86f60cb" if lane_frame else
                          "90ccfb2eba0724cdf44a4009312c6525b51e1b84a6ddc29f704e67f49c771c27")
        if hashlib.sha256(reference_path.read_bytes()).hexdigest() != reference_hash:
            raise ValueError("Previous plan comparison checkpoint checksum mismatch")
        previous = joblib.load(reference_path)
        source_hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                         for name in C0_INPUT_HASHES}
        reference_names = names[:-2] if curve_mode else names[:-3] if lane_conditioned and not lane_frame else names
        if (previous["features"] != reference_names or previous["horizons_s"] != horizons.tolist()
                or previous["source_hashes"] != source_hashes):
            raise ValueError("Previous plan comparison must use identical data and feature ordering")
        scores[reference_name] = plan_metrics(
            dev_y, predict_c0_plan(previous["plan_model"], dev_x[:, :len(reference_names)], dev_reference), masks)
        mode_report["reference_checkpoint"] = {"path": str(reference_path), "sha256": reference_hash, "label": reference_name}
    frame05_requirements = []
    if curve_mode:
        frame05_path = Path(__file__).resolve().parent / "runs/agent4_pdd_plan_lane_frame_05/model_bundle.joblib"
        frame05_hash = "190e2574cd68f68d352b1887cb0740240964bcd77db19fcfc2c614677f615512"
        if hashlib.sha256(frame05_path.read_bytes()).hexdigest() != frame05_hash:
            raise ValueError("Frame05 comparison checkpoint checksum mismatch")
        frame05 = joblib.load(frame05_path)
        if (frame05["features"] != names[:-2] or frame05["source_hashes"] != source_hashes
                or frame05["horizons_s"] != horizons.tolist()
                or frame05["plan_model"].get("coordinate_frame") != "CURRENT_LANE_TANGENT_V1"):
            raise ValueError("Frame05 comparison must use identical source data and the 101D feature prefix")
        from agent4_integration_audit.audit_compatibility import lane_frame_decoder_binding
        mode_report["frame05_decoder_source_binding"] = lane_frame_decoder_binding(frame05)
        scores["frame05_reference"] = plan_metrics(
            dev_y, predict_c0_plan(frame05["plan_model"], dev_x[:, :-2], dev_reference), masks)
        frame05_requirements = [("all", "ade_m"), ("all", "lateral_mae_m"), ("all", "speed_mae_mps"),
                                ("no_observed_actors", "speed_mae_mps"), ("slowdown", "speed_mae_mps"),
                                ("forward_corridor_slowdown", "speed_mae_mps")]
        mode_report["frame05_checkpoint"] = {"path": str(frame05_path), "sha256": frame05_hash}
    baseline = scores["constant_velocity"]
    requirements = (("all", "ade_m"), ("left_geometry", "ade_m"), ("right_geometry", "ade_m"),
                    ("left_geometry", "lateral_mae_m"), ("right_geometry", "lateral_mae_m"),
                    ("all", "speed_mae_mps"), ("speedup", "speed_mae_mps"),
                    ("slowdown", "speed_mae_mps"), ("near_stop", "speed_mae_mps"),
                    ("forward_corridor_slowdown", "speed_mae_mps"))
    unmet = {name: [f"{group}/{metric}" for group, metric in requirements
                    if not masks[group].any() or scores[name][group][metric] >= baseline[group][metric]]
             for name in candidates}
    reference_requirements = []
    if "factorized_plan" in candidates:
        reference_requirements = [("no_observed_actors", "ade_m"), ("no_observed_actors", "speed_mae_mps")]
        unmet["factorized_plan"].extend(
            f"nav02_reference/{group}/{metric}" for group, metric in reference_requirements
            if not masks[group].any() or scores["factorized_plan"][group][metric] >= scores["nav02_reference"][group][metric])
    if lane_conditioned:
        reference_requirements = [("all", "lateral_mae_m"), ("lane_deviation", "lateral_mae_m"),
                                  ("lane_deviation", "ade_m")]
        unmet["single_plan"].extend(
            f"{reference_name}/{group}/{metric}" for group, metric in reference_requirements
            if not masks[group].any() or scores["single_plan"][group][metric] >= scores[reference_name][group][metric])
    if curve_mode:
        unmet["single_plan"].extend(
            f"frame05_reference/{group}/{metric}" for group, metric in frame05_requirements
            if not masks[group].any() or scores["single_plan"][group][metric] > scores["frame05_reference"][group][metric])
    eligible = [name for name in candidates if not unmet[name]]
    def selection_score(name):
        return (scores[name]["all"]["ade_m"] / max(baseline["all"]["ade_m"], .1)
                + scores[name]["all"]["lateral_mae_m"] / max(baseline["all"]["lateral_mae_m"], .1)
                + scores[name]["all"]["speed_mae_mps"] / max(baseline["all"]["speed_mae_mps"], .1))

    selected = min(eligible or candidates, key=selection_score)
    status = "DEV_IMPROVED" if eligible else "NEEDS_FIX"
    contract = {
        "schema": ("PDD_C0_PLAN_LANE_CURVE_V1" if curve_mode else "PDD_C0_PLAN_LANE_FRAME_V1" if lane_frame else "PDD_C0_PLAN_LANE_V1" if lane_conditioned
                   else "PDD_C0_PLAN_NAV_V1" if navigation else "PDD_C0_PLAN_HISTORY_V1"),
        "source_schema": audit["schema"], "features": names,
        "horizons_s": horizons.tolist(), "output_fields": ["x_forward_m", "y_left_m", "speed_mps"],
        "reference_fields": ["velocity_forward_mps", "velocity_left_mps", "speed_mps"],
        "prediction": ("learned human bend-normalized plans; inverse quadratic and rigid transforms to ego frame"
                       if curve_mode else "learned lane-tangent offsets decoded to ego frame; body-frame residuals when lane unavailable"
                       if lane_frame else "constant-velocity reference plus predicted residual; speed clipped at zero"),
        "expert_selection": "classifier.predict(current_features)" if candidates[selected]["kind"] == "mixture" else None,
        "plan_kind": candidates[selected]["kind"],
        "traffic_residual": "learned from TRAIN driver-disjoint out-of-fold residuals; zero with no observed actors"
                            if candidates[selected]["kind"] == "factorized" else None,
        "history_seconds": .5, "history_missing_values": [0., 0.],
        "selected_model": selected, "model_family": "EXTRA_TREES_TRAJECTORY",
        "offline_status": status, "scope": "OFFLINE_HUMAN_TRAJECTORY_DIAGNOSTIC",
        "maneuver_labels_used": False, "route_conditioned": navigation,
        "lane_conditioned": lane_conditioned, "lane_pose_contract": mode_report.get("lane_pose_contract"),
        "lane_frame_conditioned": lane_frame,
        "curvature_conditioned": curve_mode, "curvature_contract": mode_report.get("curvature_contract"),
        "output_frame": "EGO_FORWARD_LEFT", "target_encoding": mode_report.get("target_encoding"),
        "navigation_contract": "Two observed forward/left unit bearings plus validity; source sampling distances differ"
                               if navigation else None,
        "encoder_sha256": audit["implementation_sha256"]["agent4_pdd_common_v2.py"],
        "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hashes": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                          for name in C0_INPUT_HASHES},
    }
    bundle = {**contract, "plan_model": candidates[selected]}
    report = {
        **contract, "seed": seed, "rows": {k: len(v[0]) for k, v in splits.items()},
        "drivers": {k: sorted({r["driver"] for r in v[2]}) for k, v in splits.items()},
        "dev": scores, "required_improvements": requirements, "unmet_requirements": unmet,
        "required_improvements_over_nav02": [] if lane_frame else reference_requirements,
        "required_improvements_over_lane04": reference_requirements if lane_frame else [],
        "required_non_regressions_over_frame05": frame05_requirements,
        "selection": "Minimum sum of ADE, lateral MAE and speed MAE divided by their baseline values floored at 0.1",
        **mode_report,
        "fitting": {"trees_per_estimator": 96, "min_samples_leaf": 8, "n_jobs": -1,
                    "target_scaler_fit": "TRAIN within each estimator"},
        "no_observed_actor_coverage": {
            split: {"rows": int((x[:, names.index("active_count_over_8")] == 0).sum()),
                    "rows_speed_ge_2mps": int(((x[:, names.index("active_count_over_8")] == 0)
                                              & (x[:, names.index("speed_mps")] >= 2.)).sum()),
                    "rows_speed_ge_5mps": int(((x[:, names.index("active_count_over_8")] == 0)
                                              & (x[:, names.index("speed_mps")] >= 5.)).sum())}
            for split, (x, _, _) in splits.items()},
        "closed_loop_tested": False,
    }
    if lane_frame:
        report["unlabelled_runtime_probe"] = probe_lane04_runtime_observations(candidates[selected], names, horizons)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as temp:
        stage = Path(temp) / "result"
        stage.mkdir()
        checkpoint = stage / "model_bundle.joblib"
        joblib.dump(bundle, checkpoint, compress=3)
        restored = joblib.load(checkpoint)
        np.testing.assert_allclose(
            predict_plan_candidate(restored["plan_model"], dev_x[:16], dev_reference[:16]),
            predict_plan_candidate(bundle["plan_model"], dev_x[:16], dev_reference[:16]))
        (stage / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        stage.rename(out)
    for name in ("constant_velocity", *candidates):
        values = scores[name]["all"]
        print(f"{name}: ADE={values['ade_m']:.4f} m, speed MAE={values['speed_mae_mps']:.4f} m/s", flush=True)
        if name in unmet:
            print(f"{name} unmet DEV criteria: {unmet[name]}", flush=True)
    if "factorized_plan" in candidates:
        for group in ("no_observed_actors", "observed_actors"):
            print(f"DEV {group} (n={scores['constant_velocity'][group]['n']}):", flush=True)
            for name in ("constant_velocity", "nav02_reference", "factorized_plan"):
                values = scores[name][group]
                print(f"  {name}: ADE={values['ade_m']} m, speed MAE={values['speed_mae_mps']} m/s", flush=True)
    if lane_conditioned:
        for group in ("all", "lane_deviation", "lane_pose_missing"):
            print(f"DEV {group} (n={scores['constant_velocity'][group]['n']}):", flush=True)
            compared = (reference_name, "lane_tangent_reference", "single_plan") if lane_frame else (reference_name, "single_plan")
            if curve_mode:
                compared = (reference_name, "frame05_reference", "lane_quadratic_reference", "single_plan")
            for name in compared:
                values = scores[name][group]
                print(f"  {name}: ADE={values['ade_m']} m, lateral MAE={values['lateral_mae_m']} m, "
                      f"speed MAE={values['speed_mae_mps']} m/s", flush=True)
    print(f"{status}, selected={selected}. Read {out / 'metrics.json'}", flush=True)


def load_split(folder: Path, name: str, features: list[str], targets: list[str]):
    path = folder / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run prepare_pdd_pilot.py first.")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    needed = features + targets + ["scenario", "driver"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    numeric = df[features + targets].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric.to_numpy(dtype=np.float64)).all(axis=1)
    if not valid.all():
        raise ValueError(f"{path.name}: {int((~valid).sum())} rows have nonnumeric/NaN/Inf features/labels")
    return df, numeric[features].to_numpy(dtype=np.float64), numeric[targets].to_numpy(dtype=np.float64)


def assess(y_true: np.ndarray, y_pred: np.ndarray, y_scale: np.ndarray):
    y_pred = np.clip(y_pred, -1.0, 1.0)
    mae = np.mean(np.abs(y_pred - y_true), axis=0)
    rmse = np.sqrt(np.mean(np.square(y_pred - y_true), axis=0))
    strong_turn = np.abs(y_true[:, 0]) >= 0.05
    braking = y_true[:, 1] < -0.05
    result = {
        "n": int(len(y_true)),
        "mae_steer": round(float(mae[0]), 6),
        "mae_longitudinal": round(float(mae[1]), 6),
        "rmse_steer": round(float(rmse[0]), 6),
        "rmse_longitudinal": round(float(rmse[1]), 6),
        "mean_normalized_mae": round(float(np.mean(mae / np.maximum(y_scale, 1e-6))), 6),
        "strong_turn_samples": int(strong_turn.sum()),
        "braking_samples": int(braking.sum()),
        "strong_turn_mae_steer": (round(float(np.mean(np.abs(y_pred[strong_turn, 0] - y_true[strong_turn, 0]))), 6)
                                  if strong_turn.any() else None),
        "braking_mae_longitudinal": (round(float(np.mean(np.abs(y_pred[braking, 1] - y_true[braking, 1]))), 6)
                                      if braking.any() else None),
    }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_dir", type=Path, help="Legacy CSV folder or audited C0 export for speed/plan training")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--c0-speed", action="store_true", help="Train the route-independent G2 speed diagnostic")
    mode.add_argument("--c0-plan", action="store_true", help="Train human forward/left/speed trajectories")
    p.add_argument("--plan-nav", action="store_true", help="Add observed navigation bearings to plan training")
    p.add_argument("--plan-factorized", action="store_true", help="Cross-fit geometry and traffic heads; compare pinned NAV_02")
    p.add_argument("--lane-pose-audit", type=Path, help="Add reviewed lane pose to --c0-plan --plan-nav, single model only")
    p.add_argument("--plan-lane-frame", action="store_true", help="Learn human plans in the audited current lane tangent frame")
    p.add_argument("--lane-curvature-audit", type=Path, help="Normalize human lane-frame plans with the reviewed current curvature audit")
    p.add_argument("--history-pdd", type=Path, help="Add causal 0.5s speed history from this raw PDD root")
    p.add_argument("--speed-mlp", action="store_true", help="Compare a 128/64 MLP with the C0 speed baseline")
    p.add_argument("--out", type=Path, default=Path("pdd_baseline_output"))
    p.add_argument("--label", choices=TARGETS.keys(), default="at_t",
                   help="at_t: action at telemetry step; mean5: 0.25-second control mean")
    p.add_argument("--max-iter", type=int, default=250, help="MLP optimizer max iterations")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    folder = args.data_dir.expanduser().resolve()
    if args.history_pdd is not None and not (args.c0_speed or args.c0_plan):
        p.error("--history-pdd requires --c0-speed or --c0-plan")
    if args.c0_plan and args.history_pdd is None:
        p.error("--c0-plan requires --history-pdd")
    if args.plan_nav and not args.c0_plan:
        p.error("--plan-nav requires --c0-plan")
    if args.plan_factorized and not (args.c0_plan and args.plan_nav):
        p.error("--plan-factorized requires --c0-plan and --plan-nav")
    if args.lane_pose_audit is not None and (not (args.c0_plan and args.plan_nav) or args.plan_factorized):
        p.error("--lane-pose-audit requires --c0-plan --plan-nav without --plan-factorized")
    if args.plan_lane_frame and args.lane_pose_audit is None:
        p.error("--plan-lane-frame requires --lane-pose-audit")
    if args.lane_curvature_audit is not None and not args.plan_lane_frame:
        p.error("--lane-curvature-audit requires --plan-lane-frame")
    if args.speed_mlp and (not args.c0_speed or args.history_pdd is None):
        p.error("--speed-mlp requires --c0-speed and --history-pdd")
    if args.max_iter <= 0:
        p.error("--max-iter must be positive")
    if args.c0_plan:
        train_c0_plan(folder, args.out.expanduser().resolve(), args.seed, args.history_pdd.expanduser().resolve(),
                      navigation=args.plan_nav, factorized=args.plan_factorized,
                      lane_review=None if args.lane_pose_audit is None else args.lane_pose_audit.expanduser().resolve(),
                      lane_frame=args.plan_lane_frame,
                      curvature_review=None if args.lane_curvature_audit is None else args.lane_curvature_audit.expanduser().resolve())
        return
    if args.c0_speed:
        history_root = args.history_pdd.expanduser().resolve() if args.history_pdd is not None else None
        train_c0_speed(folder, args.out.expanduser().resolve(), args.seed, history_root,
                       speed_mlp=args.speed_mlp, max_iter=args.max_iter)
        return
    meta_path = folder / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    if meta.get("status") != "PILOT_ONLY_LABEL_ALIGNMENT_AND_TRANSFER_NOT_PROVEN":
        raise ValueError("Unexpected dataset metadata; inspect it before training")
    features = meta["input_features"]
    targets = TARGETS[args.label]
    if not set(targets).issubset(meta.get("candidate_training_labels", [])):
        raise ValueError("Chosen labels not declared in metadata")

    df_tr, x_tr, y_tr = load_split(folder, "train", features, targets)
    df_va, x_va, y_va = load_split(folder, "val", features, targets)
    df_te, x_te, y_te = load_split(folder, "test", features, targets)
    groups = [set(df.scenario.unique()) for df in (df_tr, df_va, df_te)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("Scenario leakage detected across train/val/test")
    if len(df_tr) == 0 or len(df_va) == 0 or len(df_te) == 0:
        raise ValueError("Empty train/val/test split")

    y_scaler = StandardScaler().fit(y_tr)
    y_tr_scaled = y_scaler.transform(y_tr)
    y_std = np.maximum(y_scaler.scale_, 1e-6)

    # Both candidate models train only on train.csv.
    models = {
        "ridge": Pipeline([
            ("x_scaler", StandardScaler()),
            ("regressor", Ridge(alpha=10.0)),
        ]),
        "mlp": Pipeline([
            ("x_scaler", StandardScaler()),
            ("regressor", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=args.max_iter,
                early_stopping=False,  # Avoid random-frame internal validation leakage
                random_state=args.seed,
                alpha=0.001,
                learning_rate_init=0.001,
                n_iter_no_change=30,
            )),
        ]),
    }
    val_scores = {}
    for name, model in models.items():
        model.fit(x_tr, y_tr_scaled)
        pred = y_scaler.inverse_transform(model.predict(x_va))
        val_scores[name] = assess(y_va, pred, y_std)

    # Train-only mean baseline; not a trained driving policy.
    const_val_pred = np.repeat(y_tr.mean(axis=0, keepdims=True), len(y_va), axis=0)
    val_baseline = assess(y_va, const_val_pred, y_std)
    winner = min(models, key=lambda n: val_scores[n]["mean_normalized_mae"])
    best = models[winner]

    # Crucially, test.csv is evaluated only after selection using val.csv.
    test_pred = y_scaler.inverse_transform(best.predict(x_te))
    test_scores = assess(y_te, test_pred, y_std)
    const_test_pred = np.repeat(y_tr.mean(axis=0, keepdims=True), len(y_te), axis=0)
    test_baseline = assess(y_te, const_test_pred, y_std)

    out = args.out.expanduser().resolve() / args.label
    out.mkdir(parents=True, exist_ok=True)
    bundle_path = out / "model_bundle.joblib"
    joblib.dump({
        "model": best,
        "y_scaler": y_scaler,
        "features": features,
        "targets": targets,
        "label_mode": args.label,
        "source_metadata_status": meta["status"],
        "warning": "OFFLINE PILOT ONLY; not calibrated to MetaDrive; no Agent 4 adapter included",
    }, bundle_path)

    report = {
        "status": "OFFLINE_PDD_BASELINE_NOT_METADRIVE_VALIDATED",
        "label_mode": args.label,
        "input_features_in_order": features,
        "output_labels_in_order": targets,
        "rows": {"train": len(df_tr), "val": len(df_va), "test": len(df_te)},
        "split_scenarios": {"train": sorted(groups[0]), "val": sorted(groups[1]), "test": sorted(groups[2])},
        "train_only_target_mean": y_tr.mean(axis=0).tolist(),
        "train_only_target_std": y_std.tolist(),
        "validation": {"constant_mean": val_baseline, **val_scores},
        "selected_using_validation": winner,
        "test": {"constant_mean": test_baseline, "selected_model": test_scores},
        "model_path": str(bundle_path),
        "limitations": [
            "Only PDD offline imitation quality was tested; closed-loop route completion is NOT measured.",
            "PDD image/control timing, CARLA vs MetaDrive action calibration, waypoint coordinate equivalence remain unverified.",
            "MetaDrive physical observation adapter (including lead vehicle) has NOT been implemented.",
            "Model can lack obstacle/side-vehicle/traffic-light signals; good MAE alone is insufficient.",
            "The at_t and mean5 targets have different meanings; DO NOT directly rank their MAEs against each other.",
        ],
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Rows: train={len(df_tr)}, val={len(df_va)}, test={len(df_te)}")
    print(f"Validation normalized MAE: mean={val_baseline['mean_normalized_mae']}, "
          f"ridge={val_scores['ridge']['mean_normalized_mae']}, mlp={val_scores['mlp']['mean_normalized_mae']}")
    print(f"Chosen using validation: {winner}")
    print(f"Test normalized MAE: mean={test_baseline['mean_normalized_mae']}, "
          f"chosen={test_scores['mean_normalized_mae']}")
    print(f"Test strong turns: {test_scores['strong_turn_samples']}; braking samples: {test_scores['braking_samples']}")
    print(f"Saved: {bundle_path}")
    print(f"Saved: {out / 'metrics.json'}")
    print("WARNING: This model is NOT ready to drive in MetaDrive without an observation adapter and calibration.")


if __name__ == "__main__":
    main()
