"""Standard-library BC dataset reader. Storage IDs are never model features."""

import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct

SCHEMA = "stage4-bc-dataset/1.0.0"
PUBLIC_KEYS = {"profile_id", "step_index", "core_observation", "traffic_context", "task_context", "ego_motion"}
FORBIDDEN = {"case_id", "geometry_id", "geometry_sha256", "tier", "split", "partition", "outcome",
             "reward", "route_completion", "terminal_reason", "quality_label", "diagnostics",
             "expert_hidden_state", "decision", "canonical_action", "action_payload", "source_run_id",
             "environment_seed", "geometry_generation_seed", "episode_id", "arrive_dest", "out_of_road",
             "crash_vehicle", "crash_object", "crash_building", "crash_human", "crash_sidewalk"}


class DatasetError(ValueError):
    """Missing, inconsistent or modified dataset evidence."""


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")


def content_digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _bad_constant(value):
    raise DatasetError("Nonfinite JSON constant: " + value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"), parse_constant=_bad_constant)


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf8") as stream:
        for line in stream:
            yield json.loads(line, parse_constant=_bad_constant)


def write_rows(path, rows):
    """Deterministic gzip bytes, independent of wall time and filename."""
    with Path(path).open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=3) as stream:
            for row in rows:
                stream.write(canonical_bytes(row) + b"\n")


def safe_path(directory, relative):
    directory = Path(directory).resolve()
    relative = str(relative).replace("\\", "/")
    if not relative or ":" in relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise DatasetError("Expected a relative artifact path")
    result = (directory / relative).resolve()
    if result == directory or not result.is_relative_to(directory):
        raise DatasetError("Artifact path escapes dataset")
    return result


def frame_id(run_id, episode_id, step):
    if type(step) is not int or step < 0 or any("/" in x or "\\" in x or not x for x in (run_id, episode_id)):
        raise DatasetError("Invalid source identity")
    return f"{run_id}/{episode_id}/{step}"


def history_ids(run_id, episode_id, step, length=10):
    return [frame_id(run_id, episode_id, i) if i >= 0 else None for i in range(step-length+1, step+1)]


def validate_public(value):
    if set(value) != PUBLIC_KEYS or value["profile_id"] != "STATE_DECISION_V1":
        raise DatasetError("Unexpected public input schema")
    core = value["core_observation"]
    if core["shape"] != [259] or core["dtype"] != "float32" or len(core["features"]) != 259:
        raise DatasetError("Invalid core observation shape/type")
    def walk(obj):
        if isinstance(obj, dict):
            if FORBIDDEN.intersection(obj):
                raise DatasetError("Private metadata or labels inside policy input")
            for child in obj.values(): walk(child)
        elif isinstance(obj, list):
            for child in obj: walk(child)
        elif isinstance(obj, float) and not math.isfinite(obj):
            raise DatasetError("Nonfinite public input")
    walk(value)


def validate_action(action):
    if not isinstance(action, list) or len(action) != 2:
        raise DatasetError("Expected [steering, throttle_brake]")
    for x in action:
        if type(x) not in (int,float) or not math.isfinite(x) or not -1 <= x <= 1:
            raise DatasetError("Invalid canonical action")
        if struct.unpack("<f", struct.pack("<f", x))[0] != x:
            raise DatasetError("Label is not the exact submitted float32 value")


def load_partition(directory, partition):
    if partition not in ("FIT", "DEV"):
        raise DatasetError("Expected FIT or DEV")
    directory = Path(directory)
    frames = {}
    for row in read_rows(directory / f"frames_{partition}.jsonl.gz"):
        if set(row) != {"frame_id", "agent_input"} or row["frame_id"] in frames:
            raise DatasetError("Duplicate frame or extra fields")
        validate_public(row["agent_input"])
        frames[row["frame_id"]] = row["agent_input"]
    return frames, list(read_rows(directory / f"samples_{partition}.jsonl.gz"))


def verify_dataset(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest["schema_version"] != SCHEMA or manifest["status"] != "BUILT_AND_VERIFIED_PILOT":
        raise DatasetError("Dataset is incomplete or schema unsupported")
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"manifest.json", "CHECKSUMS.sha256"}:
        raise DatasetError("Unexpected or missing dataset files")
    for name, expected in manifest["files"].items():
        if digest(safe_path(directory, name)) != expected:
            raise DatasetError("Dataset file hash mismatch: " + name)
    checks = {**manifest["files"], "manifest.json": digest(directory / "manifest.json")}
    expected_text = "".join(f"{h}  {name}\n" for name,h in sorted(checks.items()))
    if (directory / "CHECKSUMS.sha256").read_bytes() != expected_text.encode():
        raise DatasetError("CHECKSUMS file mismatch")
    index = list(read_rows(directory / "sample_index.jsonl.gz"))
    by_id = {r["sample_id"]: r for r in index}
    if len(by_id) != len(index):
        raise DatasetError("Duplicate sample in index")
    stats, geometry_sides, seen_samples = {}, {}, set()
    provenance = read_json(directory / "provenance.json")
    episode_by_id = {(e["source_run_id"],e["episode_id"]): e for e in provenance["episodes"]}
    for side in ("FIT", "DEV"):
        frames, samples = load_partition(directory, side)
        used_frames, geometries, episodes = set(), set(), set()
        expected_ids={frame_id(e["source_run_id"],e["episode_id"],step) for e in provenance["episodes"] if e["partition"]==side
                      for interval in e["selected_intervals"] for step in range(interval["start"],interval["end"])}
        if {r["sample_id"] for r in samples} != expected_ids:
            raise DatasetError("Missing/extra sample relative to approved intervals")
        for row in samples:
            if set(row) != {"sample_id", "history_frame_ids", "canonical_action"}:
                raise DatasetError("Extra fields in training sample")
            sid = row["sample_id"]
            if sid in seen_samples or sid not in by_id:
                raise DatasetError("Duplicate sample or missing source index")
            seen_samples.add(sid)
            meta = by_id[sid]
            run, episode, step = meta["source_run_id"],meta["episode_id"],meta["step_index"]
            source = episode_by_id[(run,episode)]
            if (meta["partition"] != side or sid != frame_id(run,episode,step) or source["partition"] != side or
                    source["case"]["split"] != "TRAIN" or source["case"]["geometry_sha256"] != meta["geometry_sha256"] or
                    any(meta[k] != source[k] for k in ("trace_sha256","transitions_sha256","outcome"))):
                raise DatasetError("Sample partition/source mismatch")
            if source["outcome"] not in ("SUCCESS","TIMEOUT") or (manifest["variant"]=="BC-A" and source["outcome"]!="SUCCESS"):
                raise DatasetError("Outcome not permitted in dataset variant")
            if not any(s["start"] <= step < s["end"] for s in source["selected_intervals"]):
                raise DatasetError("Sample outside selected quality intervals")
            if row["history_frame_ids"] != history_ids(run,episode,step):
                raise DatasetError("History crosses episode, uses future or is not contiguous")
            for key in row["history_frame_ids"]:
                if key is not None:
                    if key not in frames or frames[key]["step_index"] != int(key.rsplit("/",1)[1]):
                        raise DatasetError("Missing/wrong history frame")
                    used_frames.add(key)
            validate_action(row["canonical_action"])
            if (content_digest(frames[sid]) != meta["agent_input_sha256"] or
                    content_digest(row["canonical_action"]) != meta["canonical_action_sha256"]):
                raise DatasetError("Input/label differs from source index")
            geo=meta["geometry_sha256"]
            if geo in geometry_sides and geometry_sides[geo] != side:
                raise DatasetError("Geometry leaks across FIT/DEV")
            geometry_sides[geo]=side
            geometries.add(geo)
            episodes.add((run,episode))
        if used_frames != set(frames):
            raise DatasetError("Extra/unreferenced observation frames")
        stats[side]={"samples":len(samples),"observation_frames":len(frames),
                     "context_only_frames":len(frames)-len(samples),"geometries":len(geometries),"episodes":len(episodes)}
    if seen_samples != set(by_id) or stats != manifest["partitions"]:
        raise DatasetError("Dataset statistics/index mismatch")
    return stats


def iter_examples(directory, partition, history_frames=1, verify=True):
    """Yield public inputs/mask/label only. The future encoder must omit IDs.

    Padded observations are None and must be masked. Raw public lane/branch IDs
    remain for lossless source equivalence, never to be embedded as features.
    """
    if history_frames not in (1,10):
        raise DatasetError("Supported history lengths: 1 or 10")
    if verify: verify_dataset(directory)
    frames, samples = load_partition(directory, partition)
    for sample in samples:
        keys = sample["history_frame_ids"][-history_frames:]
        yield {"observations":[copy.deepcopy(frames[k]) if k is not None else None for k in keys],
               "history_mask":[k is not None for k in keys],"canonical_action":list(sample["canonical_action"])}


def compare_variants(a_dir, b_dir):
    verify_dataset(a_dir)
    verify_dataset(b_dir)
    ai={r["sample_id"]:r for r in read_rows(Path(a_dir)/"sample_index.jsonl.gz")}
    bi={r["sample_id"]:r for r in read_rows(Path(b_dir)/"sample_index.jsonl.gz")}
    if not set(ai) <= set(bi) or any(ai[k] != bi[k] for k in ai):
        raise DatasetError("BC-A examples changed or missing in BC-B")
    if any(bi[k]["outcome"] != "TIMEOUT" for k in set(bi)-set(ai)):
        raise DatasetError("BC-B additions must be approved TIMEOUT data")
    for side in ("FIT","DEV"):
        af,aa=load_partition(a_dir,side)
        bf,bb=load_partition(b_dir,side)
        lookup={r["sample_id"]:r for r in bb}
        if any(row != lookup.get(row["sample_id"]) for row in aa) or any(af[k] != bf.get(k) for k in af):
            raise DatasetError("BC-A payload/history differs from BC-B")
    return {"common_samples":len(ai),"timeout_added_samples":len(set(bi)-set(ai)),"bc_a_subset_of_bc_b":True}
