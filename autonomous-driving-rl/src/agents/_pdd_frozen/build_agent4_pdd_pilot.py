"""Build reviewable G2 candidates from downloaded PDD telemetry and boxes."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
from statistics import median
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
import zipfile

from agent4_pdd_common_v2 import FEATURE_NAMES, SCHEMA, encode, encode_public, float32, number, project_carla


REVISION = "8f5ff297d1dff674290938c8aa7781c4fe2caf0c"
PILOT_DRIVERS = (1, 18, 21, 22, 23)
HORIZONS = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50)
PAST_EXCEPTIONS = {
    (18, "InvadingTurn"): (84, 0, 0),
    (22, "EnterActorFlow"): (0, 28, 0),
    (22, "HighwayExit"): (0, 131, None),
    (22, "ControlLoss"): (0, 0, 5),
}
UNSUPPORTED_CLASSES = {"walker", "static", "static_car", "static_trafficwarning"}
NON_DRIVING_BOX_LANES = {"Shoulder", "Biking", "Parking"}
C0_GEOMETRY_RULESET = "C0_VEHICLE_BOX_OVERLAP_V1"
C0_SCREENED_FILENAME = "c0_screened_candidates.jsonl"
BOUNDARY_MAP = {("Broken", "White"): "BROKEN_WHITE", ("Solid", "White"): "SOLID_WHITE",
                ("Solid", "Yellow"): "SOLID_YELLOW"}


class RemoteRangeReader(io.RawIOBase):
    """Seek within an HTTP ZIP without falling back to a full archive download."""

    def __init__(self, url, *, opener=None):
        super().__init__()
        self.url = url
        self.opener = urlopen if opener is None else opener
        self.position = 0
        data, total = self._range(0, 0)
        if len(data) != 1 or total < 1:
            raise ValueError("Invalid HTTP byte-range size probe")
        self.size = total

    def _range(self, start, end):
        request = Request(self.url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
        with self.opener(request, timeout=30) as response:
            if response.status != 206:
                raise ValueError("HTTP byte-range support required; refusing a full archive download")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if (match is None or (int(match[1]), int(match[2])) != (start, end)
                    or int(match[3]) <= end or (hasattr(self, "size") and int(match[3]) != self.size)):
                raise ValueError("Unexpected HTTP Content-Range; archive may have changed")
            data = response.read(end - start + 2)
            if len(data) != end - start + 1:
                raise ValueError("Truncated or oversized HTTP byte-range response")
            return data, int(match[3])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Invalid seek origin")
        target = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if target < 0:
            raise ValueError("Negative HTTP archive seek")
        self.position = target
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size is None or size < 0 else size
        size = max(0, min(size, self.size - self.position))
        if size > 128 * 1024 * 1024:
            raise ValueError("HTTP archive read exceeds the 128 MiB per-request limit")
        if not size:
            return b""
        data, _ = self._range(self.position, self.position + size - 1)
        self.position += len(data)
        return data


def fetch_town12_map(out):
    """Fetch one 0.9.15 AdditionalMaps member from CARLA's historical host."""
    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    # CARLA issues #9857 and #7054 document the historical release host.
    # Its AdditionalMaps ZIP index was checked for this exact member.
    urls = ("https://carla-releases.s3.us-east-005.backblazeb2.com/Windows/AdditionalMaps_0.9.15.zip",
            "https://downloads.carlasim.com/Windows/AdditionalMaps_0.9.15.zip")
    attempts = []
    for url in urls:
        print(f"Reading CARLA 0.9.15 AdditionalMaps index using HTTP ranges: {url}", flush=True)
        try:
            with RemoteRangeReader(url) as remote, zipfile.ZipFile(remote) as archive:
                matches = [entry for entry in archive.infolist()
                           if entry.filename.replace("\\", "/").rsplit("/", 1)[-1].lower() == "town12.xodr"]
                if not matches:
                    attempts.append({"url": url, "result": "Town12.xodr absent"})
                    continue
                if len(matches) != 1:
                    raise ValueError("Multiple Town12.xodr entries; map source is ambiguous")
                entry = matches[0]
                if not 0 < entry.file_size <= 128 * 1024 * 1024:
                    raise ValueError("Unexpected Town12.xodr uncompressed size")
                print(f"Fetching only {entry.filename} ({entry.compress_size} compressed bytes)...", flush=True)
                payload = archive.read(entry)  # zipfile verifies this member's CRC at EOF.
                road_count, root_tag = 0, None
                for event, element in ET.iterparse(io.BytesIO(payload), events=("start", "end")):
                    if root_tag is None:
                        root_tag = element.tag
                    if event == "end":
                        road_count += int(element.tag == "road")
                        element.clear()
                if root_tag != "OpenDRIVE" or not road_count:
                    raise ValueError("Downloaded map is not a populated OpenDRIVE document")
                receipt = {
                    "schema": "CARLA_TOWN12_MAP_ASSET_V1", "carla_release": "0.9.15",
                    "status": "DOWNLOADED_NOT_VALIDATED_AGAINST_PDD", "training_ready": False,
                    "source_release": "https://github.com/carla-simulator/carla/releases/tag/0.9.15",
                    "source_host_reference": "https://github.com/carla-simulator/carla/issues/9857",
                    "source_archive": url, "archive_size_bytes": remote.size,
                    "archive_member": entry.filename, "member_crc32": f"{entry.CRC:08x}",
                    "compressed_bytes": entry.compress_size, "map_bytes": len(payload),
                    "map_sha256": sha256(payload), "road_count": road_count,
                    "previous_attempts": attempts,
                }
        except (OSError, ValueError, zipfile.BadZipFile, ET.ParseError) as error:
            attempts.append({"url": url, "error": f"{type(error).__name__}: {error}"})
            print(f"Archive attempt failed: {error}", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=False)
        (out / "Town12.xodr").write_bytes(payload)
        pending = out / "audit.pending"
        pending.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        pending.rename(out / "audit.json")
        print(f"Map asset downloaded. PDD pose/road/lane matching still required. Read {out / 'audit.json'}", flush=True)
        return 0
    raise RuntimeError(f"Could not retrieve Town12 with bounded HTTP ranges: {attempts}. No full archive was downloaded.")


class EgoLaneTypeConflict(Exception):
    """The paired sources report a driving lane and a different non-driving lane."""


def partition(driver):
    if not 1 <= driver <= 30:
        raise ValueError("PDD driver must be in 01..30")
    return "TRAIN" if driver <= 21 else "DEV" if driver <= 25 else "TEST_PDD"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def bad_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def decode_boxes(source):
    marker = object()
    saw_nan = False

    def mark_constant(token):
        nonlocal saw_nan
        if token != "NaN":
            bad_constant(token)
        saw_nan = True
        return marker

    boxes = json.loads(source, object_pairs_hook=unique_object, parse_constant=mark_constant)
    if not isinstance(boxes, list) or any(not isinstance(box, dict) for box in boxes):
        raise ValueError("Expected a list of box annotations")
    if not saw_nan:
        return boxes, 0

    def reject_nan(value, path):
        if value is marker:
            raise ValueError(f"Unexpected NaN in boxes annotation at {path}")
        if isinstance(value, dict):
            for key, item in value.items():
                reject_nan(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                reject_nan(item, f"{path}[{index}]")

    replaced = 0
    for index, box in enumerate(boxes):
        if box.get("class") == "car" and box.get("brake") is marker:
            # This is optional annotation metadata for another vehicle, not an ego pedal label.
            box["brake"] = None
            replaced += 1
        reject_nan(box, f"boxes[{index}]")
    return boxes, replaced


def read_source(path, inventory, root, *, notes=None):
    raw = path.read_bytes()
    inventory[str(path.relative_to(root).as_posix())] = sha256(raw)
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    source = raw.decode("utf-8-sig")
    if path.parent.name == "boxes" and path.name.endswith(".json.gz"):
        boxes, replaced = decode_boxes(source)
        if notes is not None and replaced:
            notes["frames_with_missing_car_brake"] += 1
            notes["nan_car_brake_values"] += replaced
        return boxes
    return json.loads(source, object_pairs_hook=unique_object, parse_constant=bad_constant)


def xyz(values):
    x, y, z = values
    return number(x), number(y), number(z)


def xyz_fields(values):
    return xyz(values[key] for key in ("x", "y", "z"))


def heading(row):
    return math.radians(xyz(row["rotation"])[2])


def wrapped_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def pdd_map_lane_pose(source, lane_info, waypoint):
    """Measure current ego-left offset and desired left heading from a CARLA lane."""
    if waypoint is None:
        return None, ["NO_DRIVING_WAYPOINT"], None
    origin = xyz(source["location"])
    center = waypoint.transform.location
    lane_yaw = math.radians(number(waypoint.transform.rotation.yaw))
    lateral = project_carla(origin[0] - center.x, origin[1] - center.y, lane_yaw)[1]
    error = wrapped_angle(heading(source) - lane_yaw)
    width = number(waypoint.lane_width)
    flags = []
    if waypoint.road_id != lane_info["road_id"]:
        flags.append("MAP_ROAD_ID_MISMATCH")
    if waypoint.lane_id != lane_info["lane_id"] or waypoint.lane_id != source["lane_info"]["lane_id"]:
        flags.append("MAP_LANE_ID_MISMATCH")
    if not 0 < width <= 6 or abs(width - number(lane_info["ego_lane"]["width"])) > .05:
        flags.append("MAP_WIDTH_MISMATCH")
    if abs(origin[2] - center.z) > 1.:
        flags.append("MAP_HEIGHT_DIFFERENCE_OVER_1M")
    if abs(lateral) > width / 2 + .05:
        flags.append("EGO_OUTSIDE_PROJECTED_LANE")
    pose = {"lateral_left_m": lateral, "lane_heading_error_rad": error}
    if not all(math.isfinite(value) for value in pose.values()):
        raise ValueError("Nonfinite map lane pose")
    match = {"road_id": waypoint.road_id, "lane_id": waypoint.lane_id,
             "section_id": waypoint.section_id, "s_m": waypoint.s, "width_m": width,
             "ego_height_above_center_m": origin[2] - center.z}
    return pose, flags, match


def public_lane_pose(snapshot):
    """Measure the same quantities from CURRENT's public projection at s=0."""
    context = snapshot["task_context"]["lane_context"]
    motion = snapshot["ego_motion"]
    if context is None or motion is None:
        return None, ["MISSING_PUBLIC_LANE_OR_MOTION"]
    lanes = [lane for lane in context["lanes"] if lane["side"] == "CURRENT"]
    if len(lanes) != 1:
        raise ValueError("Expected exactly one CURRENT public lane")
    points = [point for point in lanes[0]["points"] if number(point[0]) == 0.]
    if len(points) != 1:
        return None, ["NO_PUBLIC_PROJECTION_AT_ZERO"]
    _, x, y, error = map(number, points[0])
    # The point is lane-centre minus ego in ego axes. Rotate its negative
    # into the lane's left normal, rather than using body-frame y directly.
    lateral = x * math.sin(error) - y * math.cos(error)
    discrepancy = abs(wrapped_angle(error - number(motion["lane_heading_error_rad"])))
    flags = ["PUBLIC_HEADING_PARITY_MISMATCH"] if discrepancy > 2e-5 else []
    return {"lateral_left_m": lateral, "lane_heading_error_rad": wrapped_angle(error)}, flags


def audit_public_lane_run(run):
    payload = (run / "report.json").read_bytes()
    if sha256(payload) != "3ea00cf0ecfb21be671e1cbe821fb5c92ca40a49a70edfbe38e1cc03448a800c":
        raise ValueError("Expected the reviewed factorized_03 TRAIN runtime report")
    previous = json.loads(payload)
    policy = previous["policy"]
    results = []
    for pair in previous["pairs"]:
        case = pair["case_id"]
        case_folder = "_".join(case.split("/")[1:])
        report_path = run / case_folder / "report.json"
        report = json.loads(report_path.read_bytes())
        agent = report["agents"][policy]
        trace = run / case_folder / "launcher" / policy / "agent_decisions.jsonl.gz"
        trace_hash = sha256(trace.read_bytes())
        if trace_hash != agent["trace_sha256"] or agent["episode"]["split"] != "TRAIN":
            raise ValueError(f"Public TRAIN trace integrity mismatch: {case}")
        flags, observations, valid, steps = Counter(), [], 0, 0
        with gzip.open(trace, "rt", encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                if row["case_id"] != case or row["step_index"] != index:
                    raise ValueError(f"Public case/step sequence mismatch: {case}")
                pose, reasons = public_lane_pose(row["agent_input"])
                flags.update(reasons)
                if pose is not None and not reasons:
                    observations.append(pose)
                    valid += 1
                steps = index + 1
        if steps != int(agent["episode"]["episode_steps"]):
            raise ValueError(f"Incomplete public trace: {case}")
        results.append({"case_id": case, "rows": steps, "matched_rows": valid,
                        "flags": dict(flags), "trace_sha256": trace_hash,
                        "report_sha256": sha256(report_path.read_bytes()),
                        "pose_quantiles": {field: quantiles([p[field] for p in observations])
                                           for field in ("lateral_left_m", "lane_heading_error_rad")}})
    return {"source_report_sha256": sha256(payload), "cases": results}


def audit_pdd_lane_map(root, dataset, map_path, out, public_run=None):
    """Review map-derived current measurements; never emit training-ready labels."""
    from importlib.metadata import PackageNotFoundError, version

    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    expected = {"audit.json": "3bd2be48eb74b6b869665eef4278625514d16030c43a11c86c7cd9ff4b22b5b9",
                C0_SCREENED_FILENAME: "42eaf94fdb980c64e9d6047267b5383348778c820946cc4e26b606dbd676781e"}
    hashes = {name: sha256((dataset / name).read_bytes()) for name in expected}
    if hashes != expected:
        raise ValueError("Lane map review requires the frozen expanded C0_02 export")
    audit = json.loads((dataset / "audit.json").read_bytes())
    map_bytes = map_path.read_bytes()
    map_hash = sha256(map_bytes)
    if map_hash != "26fa53d4902ac3397d35f948969f0c77c1bf023755ede2c67edf00bef817151f":
        raise ValueError("Expected the downloaded CARLA 0.9.15 Town12 map SHA256")
    try:
        api_version = version("carla")
        import carla
    except (PackageNotFoundError, ImportError) as error:
        raise RuntimeError('Install the Python API first: python -m pip install "carla==0.9.15"') from error
    if api_version != "0.9.15":
        raise ValueError(f"Expected CARLA Python API 0.9.15, found {api_version}")
    public = None if public_run is None else audit_public_lane_run(public_run)
    print("Parsing frozen Town12 OpenDRIVE locally; no CARLA server or simulation...", flush=True)
    road_map = carla.Map("Town12", map_bytes.decode("utf-8-sig"))
    del map_bytes
    episodes = defaultdict(list)
    seen = set()
    with (dataset / C0_SCREENED_FILENAME).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            identity = (row["driver"], row["scenario"], row["telemetry_step"])
            driver = int(row["driver"].split("_")[1])
            if driver > 25 or partition(driver) != row["partition"] or identity in seen:
                raise ValueError(f"Invalid TRAIN/DEV identity: {identity}")
            seen.add(identity)
            episodes[f'{row["driver"]}/{row["scenario"]}'].append(row)
    receipts = {receipt["episode"]: receipt for receipt in audit["episodes"]}
    if Counter(row["partition"] for rows in episodes.values() for row in rows) != audit["c0_geometry_screen"]["retained_by_partition"]:
        raise ValueError("C0 row counts differ from the frozen audit")
    for episode, rows in episodes.items():
        if episode_fingerprint(rows) != receipts[episode]["c0_geometry"]["retained_sha256"]:
            raise ValueError(f"C0 episode fingerprint mismatch: {episode}")
    reviewed, source_checks = [], []
    stats = {split: {"rows": 0, "matched_rows": 0, "flags": Counter(), "poses": []} for split in ("TRAIN", "DEV")}
    previous_driver = None
    for episode, rows in sorted(episodes.items()):
        driver, scenario = episode.split("/")
        if previous_driver is not None and driver != previous_driver:
            print(f"Reviewed {previous_driver}", flush=True)
        previous_driver = driver
        receipt = receipts[episode]
        inventory = {}
        base = root / driver / "data" / scenario
        telemetry = read_source(base / "metric/metric_info.json", inventory, root)
        telemetry_name = (base / "metric/metric_info.json").relative_to(root).as_posix()
        if inventory[telemetry_name] != receipt["source_sha256"][telemetry_name]:
            raise ValueError(f"Telemetry differs from the C0 audit: {episode}")
        for row in rows:
            step = row["telemetry_step"]
            box_path = base / "boxes" / f"{step // 5}.json.gz"
            boxes = read_source(box_path, inventory, root)
            box_name = box_path.relative_to(root).as_posix()
            if inventory[box_name] != receipt["source_sha256"][box_name]:
                raise ValueError(f"Boxes differ from the C0 audit: {episode}/{step}")
            info = [box for box in boxes if box["class"] == "ego_info"]
            if len(info) != 1:
                raise ValueError(f"Expected one ego_info: {episode}/{step}")
            source = telemetry[str(step)]
            # Keep the recorded ego pose. Querying the nearest lane does not
            # move the ego or repair labels; lane-ID/width/height checks can fail.
            waypoint = road_map.get_waypoint(carla.Location(*xyz(source["location"])),
                                              project_to_road=True, lane_type=carla.LaneType.Driving)
            pose, flags, match = pdd_map_lane_pose(source, info[0], waypoint)
            valid = pose is not None and not flags
            group = stats[row["partition"]]
            group["rows"] += 1
            group["matched_rows"] += int(valid)
            group["flags"].update(flags)
            if valid:
                group["poses"].append(pose)
            reviewed.append({"driver": driver, "scenario": scenario, "telemetry_step": step,
                             "partition": row["partition"], "valid": valid, "flags": flags,
                             "pose_candidate": pose, "map_match": match,
                             "recorded_lane": {"road_id": info[0]["road_id"], "lane_id": info[0]["lane_id"],
                                               "width_m": info[0]["ego_lane"]["width"]},
                             "ego_xyz_m": xyz(source["location"])})
        source_checks.append({"episode": episode, "files_verified": len(inventory),
                              "inventory_sha256": sha256(json.dumps(inventory, sort_keys=True).encode("utf-8"))})
    if previous_driver is not None:
        print(f"Reviewed {previous_driver}", flush=True)
    summary = {}
    for split, group in stats.items():
        summary[split] = {"rows": group["rows"], "matched_rows": group["matched_rows"],
                          "flags": dict(group["flags"]),
                          "pose_quantiles": {field: quantiles([p[field] for p in group["poses"]])
                                             for field in ("lateral_left_m", "lane_heading_error_rad")}}
    report = {"schema": "PDD_TOWN12_LANE_POSE_AUDIT_V1", "status": "REVIEW_REQUIRED",
              "training_ready": False, "features_added_to_model": False,
              "uses_current_frame_only": True, "human_labels_changed": False,
              "map_sha256": map_hash, "carla_api_version": api_version,
              "carla_module": str(carla.__file__), "input_hashes": hashes,
              "implementation_sha256": sha256(Path(__file__).read_bytes()),
              "query": "nearest Driving lane; recorded ego pose is never changed",
              "rules": {"width_tolerance_m": .05, "height_tolerance_m": 1.,
                        "outside_lane_margin_m": .05, "public_heading_tolerance_rad": 2e-5},
              "definitions": {"lateral_left_m": "ego minus lane centre projected onto lane-left normal",
                              "lane_heading_error_rad": "desired lane direction minus ego direction; positive left"},
              "summary": summary, "public_development": public, "source_checks": source_checks}
    out.mkdir(parents=True, exist_ok=False)
    review_path = out / "lane_pose_review.jsonl"
    with review_path.open("x", encoding="utf-8") as stream:
        for row in reviewed:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    report["review_file_sha256"] = sha256(review_path.read_bytes())
    pending = out / "audit.pending"
    pending.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.rename(out / "audit.json")
    for split, group in summary.items():
        print(f"{split}: map matched {group['matched_rows']}/{group['rows']}; flags={group['flags']}", flush=True)
    print(f"REVIEW_REQUIRED; no model training. Read {out / 'audit.json'}", flush=True)
    return 0


def actor_index(objects):
    indexed = {}
    for obj in objects:
        identity = obj["id"]
        if type(identity) is not int or identity in indexed:
            raise ValueError("Invalid or duplicate actor ID")
        indexed[identity] = obj
    return indexed


def single_box(boxes, class_name):
    matches = [row for row in boxes if row["class"] == class_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one {class_name} box")
    return matches[0]


def vehicle_boxes_overlap(ego_extent, actor_extent, actor_center, relative_yaw):
    """Test the four horizontal box axes and the vertical interval."""
    ego_length, ego_width, ego_height = ego_extent
    actor_length, actor_width, actor_height = actor_extent
    x, y, z = actor_center
    cosine, sine = math.cos(relative_yaw), math.sin(relative_yaw)
    abs_cosine, abs_sine = abs(cosine), abs(sine)
    return (
        abs(z) <= ego_height + actor_height
        and abs(x) <= ego_length + actor_length * abs_cosine + actor_width * abs_sine
        and abs(y) <= ego_width + actor_length * abs_sine + actor_width * abs_cosine
        and abs(x * cosine + y * sine) <= actor_length + ego_length * abs_cosine + ego_width * abs_sine
        and abs(-x * sine + y * cosine) <= actor_width + ego_length * abs_sine + ego_width * abs_cosine
    )


def paired_state(telemetry, boxes, *, geometry_flags=None):
    info = single_box(boxes, "ego_info")
    ego_box = single_box(boxes, "ego_car")
    origin = xyz(telemetry["location"])
    matrix = ego_box["matrix"]
    box_xy = (number(matrix[0][3]), number(matrix[1][3]))
    if math.dist(origin[:2], box_xy) > .01:
        raise ValueError("Ego box and telemetry position differ by more than 1 cm")

    telemetry_lane = telemetry["lane_info"]
    box_lane_id, telemetry_lane_id = info["lane_id"], telemetry_lane["lane_id"]
    if type(box_lane_id) is not int or type(telemetry_lane_id) is not int:
        raise ValueError("Expected integer ego lane IDs")
    width = number(info["ego_lane"]["width"])
    telemetry_width = number(telemetry_lane["lane_width"])
    if min(width, telemetry_width) <= 0:
        raise ValueError("Expected positive ego lane widths")
    box_lane_type, telemetry_lane_type = info["lane_type_str"], telemetry_lane["lane_type"]
    if type(box_lane_type) is not str or type(telemetry_lane_type) is not str:
        raise ValueError("Expected string ego lane types")
    if telemetry_lane_type == "Driving" and box_lane_type in NON_DRIVING_BOX_LANES:
        if math.dist(origin, (*box_xy, number(matrix[2][3]))) > .01:
            raise ValueError("Ego box and telemetry position differ by more than 1 cm")
        raise EgoLaneTypeConflict
    if box_lane_type != "Driving" or telemetry_lane_type != "Driving":
        raise ValueError(f"Unexpected ego lane types: boxes={box_lane_type!r}, telemetry={telemetry_lane_type!r}")
    if box_lane_id != telemetry_lane_id:
        raise ValueError("Ego lane ID differs between box and telemetry")
    if abs(width - telemetry_width) > .01:
        raise ValueError("Ego lane width differs between box and telemetry")

    ego_velocity = xyz(telemetry["velocity"])
    yaw = heading(telemetry)
    forward, right = xyz(telemetry["forward_vector"]), xyz(telemetry["right_vector"])
    ego_extent = None
    if geometry_flags is not None:
        try:
            ego_extent = xyz(ego_box["extent"])
        except (ValueError, KeyError, TypeError, OverflowError):
            geometry_flags.add("VEHICLE_GEOMETRY_UNAVAILABLE")
        else:
            if min(ego_extent) <= 0:
                geometry_flags.add("VEHICLE_GEOMETRY_UNAVAILABLE")
                ego_extent = None
    car_boxes = actor_index(box for box in boxes if box["class"] == "car")
    actors = []
    matched = 0
    for identity, actor in actor_index(telemetry["other_vehicles"]).items():
        world = xyz_fields(actor["transform"]["location"])
        delta = tuple(value - reference for value, reference in zip(world, origin))
        box = car_boxes.get(identity)
        extent = xyz_fields(actor["bounding_box_extent"])
        if ego_extent is not None and identity != ego_box.get("id"):
            x, y = project_carla(*delta[:2], yaw)
            center = (x, y, delta[2])
            try:
                actor_yaw = math.radians(number(actor["transform"]["rotation"]["yaw"]))
            except (ValueError, KeyError, TypeError, OverflowError):
                geometry_flags.add("VEHICLE_GEOMETRY_UNAVAILABLE")
            else:
                if min(extent) <= 0 or not all(map(math.isfinite, center)):
                    geometry_flags.add("VEHICLE_GEOMETRY_UNAVAILABLE")
                elif vehicle_boxes_overlap(ego_extent, extent, center, yaw - actor_yaw):
                    geometry_flags.add("VEHICLE_BOX_OVERLAP")
        if box is not None:
            x, y, _ = xyz(box["position"])
            expected = tuple(sum(a * b for a, b in zip(delta, axis)) for axis in (forward, right))
            if math.dist((x, y), expected) > .01:
                raise ValueError(f"Box and telemetry position differ for actor {identity}")
            box_extent = xyz(box["extent"])
            if math.dist(extent[:2], box_extent[:2]) > .01:
                raise ValueError(f"Box and telemetry extent differ for actor {identity}")
            extent = box_extent
            matched += 1
        distance = math.hypot(*delta[:2])
        if distance > 50:
            continue
        x, y = project_carla(*delta[:2], yaw)
        velocity = xyz_fields(actor["velocity"])
        vx, vy = project_carla(velocity[0] - ego_velocity[0], velocity[1] - ego_velocity[1], yaw)
        relative_yaw = yaw - math.radians(number(actor["transform"]["rotation"]["yaw"]))
        relative_yaw = math.atan2(math.sin(relative_yaw), math.cos(relative_yaw))
        # Match the publication precision of extract_traffic_context.
        encoded = (*map(lambda value: round(value, 4), (x, y, vx, vy, relative_yaw)),
                   round(2 * extent[0], 2), round(2 * extent[1], 2))
        actors.append((distance, encoded))
    actors.sort(key=lambda item: (item[0], item[1][0], item[1][1]))
    boundaries = tuple(BOUNDARY_MAP.get((info[f"{side}_lane_marking_type_str"],
                                         info[f"{side}_lane_marking_color_str"]), "UNKNOWN")
                       for side in ("left", "right"))
    hints = []
    for key in ("target_point", "target_point_next"):
        x, right_y = telemetry[key]
        hints.append((number(x), -number(right_y)))
    features = encode(project_carla(*ego_velocity[:2], yaw), round(width, 4), boundaries,
                      [a for _, a in actors[:8]], max(0, len(actors) - 8), hints)
    scope = set()
    for obj in boxes:
        if obj["class"] in UNSUPPORTED_CLASSES and math.hypot(*xyz(obj["position"])[:2]) <= 50:
            scope.add("UNSUPPORTED_OBJECT_WITHIN_50M")
    # C0 cannot resolve priority or signal-dependent demonstrations at intersections.
    if info["is_in_junction"] or info["traffic_light_state"] != "None":
        scope.add("JUNCTION_OR_SIGNAL_REVIEW_REQUIRED")
    return features, origin, yaw, math.hypot(*ego_velocity[:2]), scope, matched


def pedals(steer, throttle, brake):
    steer, throttle, brake = map(number, (steer, throttle, brake))
    if not (-1 <= steer <= 1 and 0 <= throttle <= 1 and 0 <= brake <= 1):
        raise ValueError("Control is outside its CARLA range")
    return {"steer": steer, "throttle": throttle, "brake": brake}


def control_index(records):
    result = {}
    for record in records:
        control = record["control"]
        step = control["step"]
        if type(step) is not int or step < 0 or step in result:
            raise ValueError(f"Invalid or duplicate control step: {step!r}")
        result[step] = control
    return result


def scan_episode(root, scene, totals, timing_samples, route_samples, *, screen_geometry=False):
    files = {}
    telemetry_raw = read_source(scene / "metric" / "metric_info.json", files, root)
    telemetry = {}
    for key, row in telemetry_raw.items():
        if re.fullmatch(r"0|[1-9][0-9]*", key) is None or int(key) % 5:
            raise ValueError(f"Unexpected telemetry step {key!r}")
        telemetry[int(key)] = row
    controls = control_index(read_source(scene / "metric" / "metrics.json", files, root)["records"])
    box_paths = {}
    for path in sorted((scene / "boxes").glob("*.json.gz")):
        key = path.name[:-8]
        if re.fullmatch(r"0|[1-9][0-9]*", key) is None:
            raise ValueError(f"Unexpected box frame ID: {key!r}")
        box_paths[int(key) * 5] = path
    steps = sorted(telemetry.keys() & controls.keys() & box_paths.keys())
    excluded = Counter()
    annotation_notes = Counter()
    error_counts = Counter()
    issues = []
    lane_type_conflict_steps = []
    geometry_frame_reviews = {}
    geometry_window_reviews = {}
    states = {}
    candidates = []
    alignment = (len(telemetry.keys() - box_paths.keys()), len(box_paths.keys() - telemetry.keys()),
                 min(telemetry, default=None))
    totals["joined"] += len(steps)
    totals["telemetry"] += len(telemetry)
    totals["controls"] += len(controls)
    totals["boxes"] += len(box_paths)
    excluded["TELEMETRY_WITHOUT_CONTROL"] = len(telemetry.keys() - controls.keys())
    for step in steps:
        try:
            control = controls[step]
            human = pedals(control["steer"], control["throttle"], control["brake"])
            modes = tuple(control[key] for key in ("reverse", "hand_brake", "manual_gear_shift"))
            if any(type(mode) is not bool for mode in modes):
                raise ValueError("Expected boolean control mode flags")
            if any(modes):
                excluded["UNSUPPORTED_CONTROL_MODE"] += 1
                continue
            source = telemetry[step]
            boxes = read_source(box_paths[step], files, root, notes=annotation_notes)
            geometry_flags = set() if screen_geometry else None
            features, origin, yaw, speed, scope, matched = paired_state(source, boxes, geometry_flags=geometry_flags)
        except EgoLaneTypeConflict:
            excluded["EGO_LANE_TYPE_CONFLICT"] += 1
            lane_type_conflict_steps.append(step)
            continue
        except (ValueError, KeyError, TypeError, OSError, EOFError, IndexError, OverflowError) as error:
            excluded["INVALID_SOURCE_ROW"] += 1
            error_counts[str(error)] += 1
            if len(issues) < 10:
                issues.append({"step": step, "error": str(error)})
            continue
        if geometry_flags:
            geometry_frame_reviews[step] = sorted(geometry_flags)
        try:
            expert = pedals(source["expert_control_steer"], source["expert_control_throttle"],
                            source["expert_control_brake"])
            expert["target_speed_raw"] = number(source["expert_target_speed"])
        except (ValueError, KeyError, TypeError, OverflowError):
            # Expert labels are optional and have a separate lineage from human labels.
            expert = None
            totals["invalid_optional_expert_labels"] += 1
        states[step] = (features, origin, yaw, speed, scope, human, expert)
        totals["validated"] += 1
        totals["matched_actor_boxes"] += matched
        if timing_samples is not None and step - 5 in states:
            _, prior, _, prior_speed, *_ = states[step - 5]
            average_speed = (prior_speed + speed) / 2
            distance = math.dist(prior[:2], origin[:2])
            if average_speed >= 2 and distance > 0:
                timing_samples.append(distance / average_speed)
        if route_samples is not None:
            for index, key in enumerate(("target_point", "target_point_next")):
                x, y = source[key]
                route_samples[index].append((x, -y))

    identity = scene.relative_to(root).parts
    for step, (features, origin, yaw, _, scope, human, expert) in states.items():
        future_steps = [step + 5 * offset for offset in range(1, len(HORIZONS) + 1)]
        future = [states.get(future_step) for future_step in future_steps]
        if any(state is None for state in future):
            excluded["INCOMPLETE_TRAJECTORY_HORIZON"] += 1
            continue
        scope = scope.union(*(state[4] for state in future))
        if scope:
            excluded.update(scope)
            continue
        if screen_geometry:
            reasons = {reason for frame_step in (step, *future_steps)
                       for reason in geometry_frame_reviews.get(frame_step, ())}
            if reasons:
                geometry_window_reviews[step] = sorted(reasons)
        trajectory = []
        for _, position, _, speed, *_ in future:
            x, y = project_carla(position[0] - origin[0], position[1] - origin[1], yaw)
            trajectory.append(float32([x, y, speed]))
        candidates.append({"driver": identity[0], "scenario": scene.name, "frame_id": step // 5,
                           "telemetry_step": step, "control_step": step, "features": features,
                           "human_trajectory": trajectory, "target_mask": [True] * 6,
                           "human_control": human, "expert_control": expert,
                           "behavior_quality": "UNREVIEWED"})
    totals.update(annotation_notes)
    receipt = {"episode": f"{identity[0]}/{scene.name}", "boxes": len(box_paths),
               "telemetry": len(telemetry), "controls": len(controls), "joined": len(steps),
               "alignment": list(alignment), "validated": len(states), "candidates": len(candidates),
               "telemetry_steps_without_boxes": sorted(telemetry.keys() - box_paths.keys()),
               "box_steps_without_telemetry": sorted(box_paths.keys() - telemetry.keys()),
               "exclusions": dict(excluded), "annotation_notes": dict(annotation_notes),
               "lane_type_conflict_steps": lane_type_conflict_steps,
               "source_error_counts": dict(error_counts), "issues": issues, "source_sha256": files}
    if screen_geometry:
        receipt["c0_geometry"] = {"frame_review_reasons": geometry_frame_reviews,
                                  "window_review_reasons": geometry_window_reviews}
    return receipt, candidates


def quantiles(values):
    if not values:
        return None
    values = sorted(values)
    last = len(values) - 1
    return {"n": len(values), "p10": values[int(last * .1)], "median": median(values),
            "p90": values[int(last * .9)]}


def route_summary(samples):
    return [{"distance_m": quantiles([math.hypot(x, y) for x, y in group]),
             "bearing_rad": quantiles([math.atan2(y, x) for x, y in group]),
             "behind_ego": sum(x < 0 for x, _ in group)} for group in samples]


def public_diagnostics(paths):
    if not paths:
        return None
    hints = ([], [])
    missing = [0, 0]
    count = 0
    for path in paths:
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8-sig") as stream:
                snapshots = [json.loads(line) for line in stream if line.strip()]
        else:
            snapshot = json.loads(path.read_text(encoding="utf-8-sig"))
            snapshots = [snapshot.get("public", snapshot)]
        for snapshot in snapshots:
            features = encode_public(snapshot)
            count += 1
            for index, start in enumerate((90, 93)):
                if features[start + 2]:
                    hints[index].append(tuple(features[start:start + 2]))
                else:
                    missing[index] += 1
    return {"frames": count, "sources": [str(path) for path in paths],
            "missing_by_hint": missing, "capped_hints": route_summary(hints)}


def episode_fingerprint(rows):
    if not rows:
        return None
    payload = [[row["features"], row["human_trajectory"]] for row in rows]
    return sha256(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode())


def lane_event_review(steps, event, telemetry, boxes, *, review_span=False):
    """Screen a sampled crossing for review; never promote it to a training label."""
    before = single_box(boxes[event - 5], "ego_info")
    after = single_box(boxes[event], "ego_info")
    source_lane, target_lane = before["lane_id"], after["lane_id"]
    road = before["road_id"]
    shift = after["ego_lane_number"] - before["ego_lane_number"]
    record = {"step": event, "source_lane": source_lane, "target_lane": target_lane,
              "road_id": road, "maneuver_mask": 0, "flags": [], "gap_samples": []}
    if (shift not in (-1, 1) or abs(target_lane) - abs(source_lane) != shift
            or source_lane * target_lane <= 0 or after["road_id"] != road):
        record["flags"].append("UNCONFIRMED_ADJACENT_DIRECTION")
        return record
    side = "left" if shift == -1 else "right"
    record["direction"] = side
    flags = set()
    start = telemetry[str(steps[0])]
    start_x, start_y, _ = xyz(start["location"])
    yaw_at_start = heading(start)
    for step in steps:
        frame = boxes[step]
        info = single_box(frame, "ego_info")
        row = telemetry[str(step)]
        lane = source_lane if step < event else target_lane
        if (info["road_id"] != road or info["lane_id"] != lane or row["lane_info"]["lane_id"] != lane
                or info["is_in_junction"] or info["lane_type_str"] != "Driving"):
            flags.add("ROAD_OR_LANE_SEQUENCE")
        if step < event:
            neighbors = info[f"{side}_lanes"]
            if (not neighbors or neighbors[0]["type:"] != "Driving"
                    or info["lane_change_str"] not in (side.title(), "Both")
                    or info[f"{side}_lane_marking_type_str"] != "Broken"):
                flags.add("PRE_CROSSING_PERMISSION_OR_BOUNDARY")
        ego = single_box(frame, "ego_car")
        ego_length = xyz(ego["extent"])[0]
        cars = actor_index(car for car in frame if car["class"] == "car")
        actors = actor_index(row["other_vehicles"])
        closest = {"front": None, "rear": None}
        min_ttc = {"front": None, "rear": None}
        origin = xyz(row["location"])
        for actor_id, actor in actors.items():
            location = xyz_fields(actor["transform"]["location"])
            if actor_id == ego.get("id") or math.dist(location[:2], origin[:2]) > 50:
                continue
            box = cars.get(actor_id)
            if box is None:
                flags.add("UNMATCHED_NEARBY_ACTOR")
                continue
            if (box["road_id"], box["lane_id"]) != (road, target_lane):
                continue
            x = xyz(box["position"])[0]
            length, width, _ = xyz(box["extent"])
            yaw = heading(row) - math.radians(number(actor["transform"]["rotation"]["yaw"]))
            gap = abs(x) - ego_length - length * abs(math.cos(yaw)) - width * abs(math.sin(yaw))
            key = "front" if x >= 0 else "rear"
            if closest[key] is None or gap < closest[key]["gap_m"]:
                closest[key] = {"actor_id": actor_id, "gap_m": gap}
            if gap <= 0:
                flags.add("TARGET_LANE_LONGITUDINAL_OVERLAP")
            if review_span:
                # Mirror Stage 1's 4 m gap and 4 s TTC as review triggers.
                if gap < 4.0:
                    flags.add("TARGET_LANE_GAP_BELOW_4M")
                av = xyz_fields(actor["velocity"])
                ev = xyz(row["velocity"])
                relative_vx, _ = project_carla(av[0] - ev[0], av[1] - ev[1], heading(row))
                closing = -relative_vx if key == "front" else relative_vx
                if gap > 0 and closing > 0:
                    ttc = gap / closing
                    if ttc < 4.0:
                        flags.add("TARGET_LANE_PROJECTED_TTC_BELOW_4S")
                    if min_ttc[key] is None or ttc < min_ttc[key]["seconds"]:
                        min_ttc[key] = {"actor_id": actor_id, "gap_m": gap, "closing_mps": closing, "seconds": ttc}
        if any(car["id"] not in actors and math.hypot(*xyz(car["position"])[:2]) <= 50 for car in cars.values()):
            flags.add("UNMATCHED_NEARBY_ACTOR")
        _, lateral = project_carla(origin[0] - start_x, origin[1] - start_y, yaw_at_start)
        sample = {"step": step, "ego_left_m_from_start": lateral, **closest}
        if review_span:
            sample["lowest_projected_ttc"] = min_ttc
        record["gap_samples"].append(sample)
    if lateral * (-shift) <= 0:
        flags.add("NET_LATERAL_DIRECTION_DISAGREES")
    record["flags"] = sorted(flags)
    return record


def span_geometry(steps, telemetry, boxes, direction):
    """Describe approach/departure geometry; heading and lane IDs alone cannot delimit a maneuver."""
    xy = [xyz(telemetry[str(s)]["location"])[:2] for s in steps]
    start, end = xy[0], xy[-1]
    dx, dy = end[0] - start[0], end[1] - start[1]
    result = {}
    for name, points in (("approach", xy[:5]), ("departure", xy[-5:])):
        x, y = points[0]
        run_x, run_y = points[-1][0] - x, points[-1][1] - y
        length = math.hypot(run_x, run_y)
        if length < 1:
            result[name] = {"travel_m": length, "signed_lane_shift_m": None}
            continue
        tangent = math.atan2(run_y, run_x)
        _, shift = project_carla(dx, dy, tangent)
        result[name] = {"travel_m": length, "heading_rad": tangent,
                        "max_line_deviation_m": max(abs(project_carla(px-x, py-y, tangent)[1]) for px, py in points),
                        "signed_lane_shift_m": shift if direction == "left" else -shift}
    result["expected_center_spacing_m"] = sum(number(single_box(boxes[s], "ego_info")["ego_lane"]["width"])
                                              for s in (steps[0], steps[-1])) / 2
    return result


def audit_lane_events(root, dataset, out, span_from=None, *, six_second_events=False):
    audit_path, rows_path = dataset / "audit.json", dataset / C0_SCREENED_FILENAME
    hashes = {path.name: sha256(path.read_bytes()) for path in (audit_path, rows_path)}
    known_datasets = (
        {"audit.json": "45d18ff9162d32cd34ee127ca1c2b5a28fc69c6318d1baf13b909340b5defa0d",
         C0_SCREENED_FILENAME: "2e5c4f14b9386c91a220e5fabd0f1866c089bde7cfcc2a26b7312914f8ab6880"},
        {"audit.json": "3bd2be48eb74b6b869665eef4278625514d16030c43a11c86c7cd9ff4b22b5b9",
         C0_SCREENED_FILENAME: "42eaf94fdb980c64e9d6047267b5383348778c820946cc4e26b606dbd676781e"},
    )
    if hashes not in known_datasets:
        raise ValueError("Expected a reviewed TRAIN/DEV C0 export")
    audit = json.loads(audit_path.read_bytes())
    receipts = {r["episode"]: r for r in audit["episodes"]}
    long_span = span_from is not None or six_second_events
    prior_events = {}
    if span_from is not None:
        prior_data = (span_from / "audit.json").read_bytes()
        prior_hash = sha256(prior_data)
        if prior_hash != "f594d377d46e3bcf469ad11ac9b00a639f0358d83a69e9a935827ee0588dc86e":
            raise ValueError("Expected the reviewed seven-frame G5 audit")
        for row in json.loads(prior_data)["events"]:
            if row["flags"] in ([], ["NET_LATERAL_DIRECTION_DISAGREES"]):
                prior_events.setdefault(row["episode"], {})[row["step"]] = row
    starts = defaultdict(set)
    with rows_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            starts[(row["driver"], row["scenario"])].add(row["telemetry_step"])
    events = []
    counts = {"TRAIN": Counter(), "DEV": Counter()}
    for (driver, scene), windows in sorted(starts.items()):
        episode = f"{driver}/{scene}"
        if span_from is not None and episode not in prior_events:
            continue
        receipt = receipts[episode]
        partition_name = receipt["partition"]
        base = root / driver / "data" / scene
        inventory = {}
        telemetry = read_source(base / "metric/metric_info.json", inventory, root)
        covered = {step for start in windows for step in range(start + 5, start + 31, 5)}
        crossings = (sorted(prior_events[episode]) if span_from is not None else
                     [step for step in sorted(covered) if telemetry[str(step)]["lane_change_count"]
                      > telemetry[str(step - 5)]["lane_change_count"]])
        frames = {}
        for event in crossings:
            record = {"episode": episode, "partition": partition_name, "step": event, "maneuver_mask": 0}
            radius = 60 if long_span else 15
            steps = range(event - radius, event + radius + 1, 5)
            if span_from is not None:
                prior = prior_events[episode][event]
                record["prior_flags"] = prior["flags"]
                record["direction"] = prior["direction"]
            if long_span and not all(step in covered for step in steps[1:]):
                record["flags"] = ["INCOMPLETE_SIX_SECOND_C0_SPAN"]
            elif not long_span and event - 15 not in windows:
                record["flags"] = ["NO_CENTERED_C0_WINDOW"]
            else:
                for step in steps:
                    if step not in frames:
                        frames[step] = read_source(base / "boxes" / f"{step // 5}.json.gz", inventory, root)
                record.update(lane_event_review(steps, event, telemetry, frames, review_span=long_span))
                if long_span and record.get("gap_samples"):
                    geometry = span_geometry(steps, telemetry, frames, record["direction"])
                    record["endpoint_geometry"] = geometry
                    offsets = [geometry[end]["signed_lane_shift_m"] for end in ("approach", "departure")]
                    if any(value is None for value in offsets):
                        record["flags"].append("ENDPOINT_REFERENCE_UNAVAILABLE")
                    elif min(offsets) <= 0:
                        record["flags"].append("ENDPOINT_DIRECTION_NOT_CORROBORATED")
            events.append(record)
            counts[partition_name]["counter_events"] += 1
            if record.get("gap_samples"):
                counts[partition_name]["six_second_gap_reviews" if long_span else "seven_frame_gap_reviews"] += 1
            for flag in record["flags"]:
                counts[partition_name][flag] += 1
            if not record["flags"]:
                counts[partition_name]["unflagged_for_review"] += 1
        if any(receipt["source_sha256"].get(name) != digest for name, digest in inventory.items()):
            raise ValueError(f"Raw PDD differs from G2 audit: {episode}")
    report = {"scope": "G5_C0_CENTERED_GAP_REVIEW_V1", "source_hashes": hashes, "training_ready": False,
              "frame_offsets_s": [-.75, -.5, -.25, 0., .25, .5, .75], "radius_m": 50,
              "counts": counts, "events": events}
    if long_span:
        report.update(scope="G5_C0_SIX_SECOND_SPAN_V1",
                      frame_offsets_s=[i / 4 for i in range(-12, 13)],
                      diagnostic_limits={"gap_m": 4.0, "projected_ttc_s": 4.0, "min_reference_travel_m": 1.0})
    if span_from is not None:
        if len(events) != sum(map(len, prior_events.values())):
            raise ValueError("Missing events from the previous G5 audit")
        report["prior_audit_sha256"] = prior_hash
    report["drivers"] = audit["pilot_drivers"]
    report["implementation_sha256"] = {name: sha256(Path(__file__).with_name(name).read_bytes())
                                        for name in ("build_agent4_pdd_pilot.py", "agent4_pdd_common_v2.py")}
    out.mkdir(parents=True, exist_ok=False)
    temporary = out / "audit.pending"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.rename(out / "audit.json")
    print(f"Lane-event review: {counts}. Read {out / 'audit.json'}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdd", type=Path, default=Path("PDD"))
    parser.add_argument("--out", required=True, type=Path,
                        help="New directory for the audit report and optional review candidates")
    parser.add_argument("--development-drivers", action="store_true",
                        help="Export all 21 TRAIN and four DEV drivers under the existing C0 rules")
    parser.add_argument("--public-snapshot", type=Path, action="append",
                        help="Repeatable JSON snapshot or JSONL of AgentInputV1.to_dict()")
    export = parser.add_mutually_exclusive_group()
    export.add_argument("--emit-candidates", action="store_true", help="Write UNREVIEWED pilot JSONL")
    export.add_argument("--emit-c0-screened", action="store_true", help="Write geometry-screened pilot JSONL")
    export.add_argument("--lane-event-review", type=Path, metavar="C0_DIR",
                        help="Audit target-lane gaps around crossings in the reviewed C0 pilot")
    export.add_argument("--fetch-town12-map", action="store_true",
                        help="Download only Town12.xodr using HTTP ranges from official CARLA 0.9.15 ZIPs")
    export.add_argument("--lane-map-review", type=Path, metavar="C0_DIR",
                        help="Audit current lane pose from frozen Town12 against C0 TRAIN/DEV")
    parser.add_argument("--lane-map", type=Path, help="Downloaded Town12.xodr, for --lane-map-review")
    parser.add_argument("--lane-public-run", type=Path,
                        help="Reviewed factorized_03 TRAIN run, for public lane-pose parity")
    parser.add_argument("--span-from", type=Path, metavar="G5_DIR",
                        help="Extend unflagged and direction-only events from the seven-frame review")
    parser.add_argument("--six-second-events", action="store_true",
                        help="Review all counter events over six seconds directly from a reviewed C0 export")
    args = parser.parse_args()
    root, out = args.pdd.resolve(), args.out.resolve()
    if out == root or root in out.parents:
        parser.error("Use an output directory outside the raw PDD tree")
    if out.exists():
        parser.error(f"Output directory already exists: {out}")
    if args.span_from is not None and args.lane_event_review is None:
        parser.error("--span-from requires --lane-event-review")
    if args.six_second_events and (args.lane_event_review is None or args.span_from is not None):
        parser.error("--six-second-events requires --lane-event-review and a direct C0 export")
    if args.development_drivers and not args.emit_c0_screened:
        parser.error("--development-drivers requires --emit-c0-screened")
    if (args.lane_map is not None or args.lane_public_run is not None) and args.lane_map_review is None:
        parser.error("--lane-map/--lane-public-run require --lane-map-review")
    if args.lane_map_review is not None:
        if args.lane_map is None or args.public_snapshot:
            parser.error("--lane-map-review requires --lane-map and does not use --public-snapshot")
        return audit_pdd_lane_map(root, args.lane_map_review.resolve(), args.lane_map.resolve(), out,
                                  None if args.lane_public_run is None else args.lane_public_run.resolve())
    if args.fetch_town12_map:
        if args.public_snapshot:
            parser.error("--fetch-town12-map does not use --public-snapshot")
        return fetch_town12_map(out)
    if args.lane_event_review is not None:
        return audit_lane_events(root, args.lane_event_review.resolve(), out, args.span_from,
                                 six_second_events=args.six_second_events)
    revision = (root / ".cache/huggingface/download/README.md.metadata").read_text().splitlines()[0]
    if revision != REVISION:
        parser.error(f"Expected surveyed revision {REVISION}, found {revision}")
    folders = {path.name for path in root.glob("driver_*") if path.is_dir()}
    if folders != {f"driver_{driver:02d}" for driver in range(1, 31)}:
        parser.error("Expected all 30 surveyed driver directories")
    public = public_diagnostics(args.public_snapshot)
    drivers = tuple(range(1, 26)) if args.development_drivers else PILOT_DRIVERS
    totals = Counter()
    timing = []
    hints = ([], [])
    offenses, receipts, episodes = [], [], []
    for driver in drivers:
        split = partition(driver)
        scenes = sorted(path for path in (root / f"driver_{driver:02d}" / "data").iterdir()
                        if path.is_dir())
        if len(scenes) != 21:
            raise ValueError(f"Expected 21 scenes for driver {driver:02d}")
        for scene in scenes:
            try:
                receipt, rows = scan_episode(root, scene, totals, timing if split == "TRAIN" else None,
                                             hints if split == "TRAIN" else None,
                                             screen_geometry=args.emit_c0_screened)
            except (ValueError, KeyError, TypeError, OSError, EOFError, IndexError, OverflowError) as error:
                name = f"driver_{driver:02d}/{scene.name}"
                offenses.append(name)
                receipts.append({"episode": name, "partition": split, "error": str(error)})
                continue
            expected = PAST_EXCEPTIONS.get((driver, scene.name), (0, 0, 0))
            if tuple(receipt["alignment"]) != expected or receipt["exclusions"]["TELEMETRY_WITHOUT_CONTROL"]:
                offenses.append(receipt["episode"])
            if receipt["issues"]:
                offenses.append(receipt["episode"])
            receipt["partition"] = split
            receipt["candidate_sha256"] = episode_fingerprint(rows)
            receipts.append(receipt)
            episodes.append((receipt, rows))
        print(f"Processed driver_{driver:02d} ({split})", flush=True)

    counts = Counter()
    lane_type_conflicts = Counter()
    source_error_counts = Counter()
    for receipt in receipts:
        source_error_counts.update(receipt.get("source_error_counts", {}))
        lane_type_conflicts[receipt["partition"]] += len(receipt.get("lane_type_conflict_steps", ()))
    coverage = defaultdict(Counter)
    screened_counts = Counter()
    screened_coverage = defaultdict(Counter)
    screened_held = Counter()
    seen = set()
    output_rows = []
    for receipt, rows in sorted(episodes, key=lambda item: (item[0]["partition"] != "DEV",
                                                           item[0]["episode"])):
        fingerprint = receipt["candidate_sha256"]
        if fingerprint is not None and fingerprint in seen:
            receipt["duplicate_candidates_excluded"] = len(rows)
            continue
        if fingerprint is not None:
            seen.add(fingerprint)
        split = receipt["partition"]
        counts[split] += len(rows)
        for row in rows:
            coverage[split][row["scenario"]] += 1
        if args.emit_c0_screened:
            geometry = receipt["c0_geometry"]
            held = geometry["window_review_reasons"]
            rows = [row for row in rows if row["telemetry_step"] not in held]
            geometry["retained_count"] = len(rows)
            geometry["retained_sha256"] = episode_fingerprint(rows)
            screened_counts[split] += len(rows)
            screened_held[split] += len(held)
            if rows:
                screened_coverage[split][rows[0]["scenario"]] += len(rows)
        for row in rows:
            output_rows.append({**row, "partition": split})

    report = {
        "schema": SCHEMA, "feature_names": FEATURE_NAMES, "horizons_s_provisional": HORIZONS,
        "revision_from_readme_cache": revision, "pilot_drivers": drivers,
        "dataset_scope": "PDD_TRAIN_DEV_EXPANDED_C0" if args.development_drivers else "PDD_FIVE_DRIVER_PILOT",
        "implementation_sha256": {name: sha256(Path(__file__).with_name(name).read_bytes())
                                  for name in ("build_agent4_pdd_pilot.py", "agent4_pdd_common_v2.py")},
        "status": "REVIEW_REQUIRED", "training_ready": False,
        "candidate_exported": args.emit_candidates and not offenses,
        "unexpected_alignment_or_source_errors": sorted(set(offenses)),
        "source_error_counts": source_error_counts,
        "lane_type_conflicts_by_partition": lane_type_conflicts,
        "counts": totals, "review_candidates_by_partition": counts,
        "review_candidates_by_scenario": coverage, "episodes": receipts,
        "timing_train_estimated_seconds": quantiles(timing),
        "route_pdd_train_uncapped": route_summary(hints),
        "route_pdd_train_capped": route_summary([
            [(x * 50 / max(50, math.hypot(x, y)), y * 50 / max(50, math.hypot(x, y)))
             for x, y in group] for group in hints]),
        "route_metadrive_development": public,
        "gates": {"DATA": "REVIEW_REQUIRED", "TIME": "PROVISIONAL_UNTIL_PILOT",
                  "CONTROL_ALIGNMENT": "UNVERIFIED", "SPLIT": "PILOT_ONLY",
                  "OBS": "REVIEW_REQUIRED", "GOAL.A": "REVIEW_REQUIRED",
                  "GOAL.B": "PENDING_DRIVING_PILOT", "RUNTIME": "PENDING_LAUNCHER"},
    }
    filename = "review_candidates.jsonl" if args.emit_candidates else None
    if args.emit_c0_screened:
        filename = C0_SCREENED_FILENAME
        report["c0_geometry_screen"] = {
            "ruleset": C0_GEOMETRY_RULESET,
            "filename": filename,
            "exported": not offenses,
            "samples_per_window": len(HORIZONS) + 1,
            "retained_by_partition": screened_counts,
            "held_by_partition": screened_held,
            "retained_by_scenario": screened_coverage,
        }
    # The report is the completion marker. An existing destination is never reused.
    out.mkdir(parents=True, exist_ok=False)
    if filename is not None and not offenses:
        with (out / filename).open("x", encoding="utf-8") as stream:
            for row in output_rows:
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        report["candidate_file_sha256"] = sha256((out / filename).read_bytes())
    temporary = out / "audit.pending"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.rename(out / "audit.json")
    print(f"Review candidates: {dict(counts)}. Read {out / 'audit.json'}", flush=True)
    if args.emit_c0_screened:
        print(f"Geometry retained: {dict(screened_counts)}. Held for review: {dict(screened_held)}", flush=True)
    return 2 if offenses else 0


if __name__ == "__main__":
    raise SystemExit(main())
