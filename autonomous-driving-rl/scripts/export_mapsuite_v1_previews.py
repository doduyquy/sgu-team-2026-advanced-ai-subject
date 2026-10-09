#!/usr/bin/env python3
"""Derived Level 1 previews of the frozen MapSuite V1 package (Python 3.10).

The CLI always verifies the actual simulator pin and never accepts a fixture
renderer. Pure packaging functions can use an explicitly labelled test renderer.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import sys
import time
from typing import Any
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_mapsuite_v1_dataset import (
    DATASET_ID, DATASET_VERSION, PINNED_METADRIVE_COMMIT,
    PINNED_METADRIVE_VERSION, compute_file_sha256,
)
from scripts.verify_mapsuite_v1_dataset import verify_dataset
from src.platform.protocol import canonical_json_sha256

EXPORTER_VERSION = "1.0.1"
EXPORTER_PATH = "autonomous-driving-rl/scripts/export_mapsuite_v1_previews.py"
REPOSITORY = "https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject"
AUTHORITY = ("GitHub MapSuite V1 is authoritative. "
             "This package contains derived visualization artifacts only.")
SOURCE_PACKAGE_FILES = (
    "geometries.csv", "generation_config.json", "dataset_manifest.json", "CHECKSUMS.sha256",
)
COLUMNS = [
    "preview_filename", "geometry_id", "sequence", "geometry_generation_seed",
    "difficulty_tier", "split", "geometry_sha256", "image_sha256",
    "image_width_px", "image_height_px", "dataset_id", "dataset_version",
    "metadrive_version", "metadrive_commit",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def safe_component(value: str) -> str:
    """Reversible escaping, including escape marker; never discard characters."""
    if not value or value in {".", ".."}:
        raise ValueError("Empty or traversal filename component")
    return "".join(c if re.fullmatch(r"[A-Za-z0-9_-]", c)
                   else f"~{ord(c):06x}" for c in value)


def preview_filename(row: dict[str, Any]) -> str:
    seed = int(row["geometry_generation_seed"])
    if not 0 <= seed <= 19:
        raise ValueError("Generation seed outside frozen range")
    name = (f"{safe_component(row['geometry_id'])}__{safe_component(row['tier'])}__"
            f"{safe_component(row['sequence'])}__gseed-{seed:02d}.png")
    if len(name) > 240:
        raise ValueError("Preview filename exceeds portable component length")
    return name


def load_geometries(root: Path) -> tuple[list[dict[str, str]], dict, dict]:
    dataset = root / "datasets/mapsuite_v1"
    if not verify_dataset(dataset, root):
        raise ValueError("Frozen MapSuite package/source verification failed")
    manifest = read_json(dataset / "dataset_manifest.json")
    config = read_json(dataset / "generation_config.json")
    if (manifest["dataset_id"], manifest["dataset_version"]) != (DATASET_ID, DATASET_VERSION):
        raise ValueError("Unsupported dataset identity")
    if manifest["simulator"] != {
        "name": "MetaDrive", "version": PINNED_METADRIVE_VERSION,
        "commit": PINNED_METADRIVE_COMMIT,
    }:
        raise ValueError("Dataset simulator pin mismatch")
    with (dataset / "geometries.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = [preview_filename(row).casefold() for row in rows]
    if len(rows) != 240 or len({r['geometry_id'] for r in rows}) != 240:
        raise ValueError("Expected exactly 240 unique geometry IDs")
    if len(set(names)) != 240:
        raise ValueError("Duplicate preview filename (including Windows case collisions)")
    if Counter(r["split"] for r in rows) != {"TRAIN": 180, "VALIDATION": 48, "TEST": 12}:
        raise ValueError("Frozen split mismatch")
    return rows, manifest, config


def git_blob(root: Path, commit: str, repository_path: str) -> bytes:
    """Read exact object bytes, unaffected by checkout EOL conversion."""
    result = subprocess.run(
        ["git", "-C", str(root), "cat-file", "blob", f"{commit}:{repository_path}"],
        capture_output=True,
    )
    if result.returncode:
        raise ValueError(f"Recorded-source Git provenance: missing blob {repository_path} at {commit}")
    return result.stdout


def blob_exporter_version(blob: bytes) -> str:
    # Inspect a literal assignment, never execute code from a recorded commit.
    for node in ast.parse(blob.decode("utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "EXPORTER_VERSION" for target in node.targets
        ):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
    raise ValueError("Recorded-source Git provenance: exporter version literal is missing")


def git_source_identity(root: Path, commit: str) -> dict[str, Any]:
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Recorded-source Git provenance: invalid Git source SHA")
    # Requiring the exact object to be a commit excludes tree/blob/tag IDs.
    result = subprocess.run(["git", "-C", str(root), "cat-file", "-t", commit], capture_output=True)
    if result.returncode or result.stdout.strip() != b"commit":
        raise ValueError("Recorded-source Git provenance: source_git_sha is not an existing Git commit")
    exporter = git_blob(root, commit, EXPORTER_PATH)
    return {
        "source_git_sha": commit,
        "exporter_sha256": hashlib.sha256(exporter).hexdigest(),
        "exporter_version": blob_exporter_version(exporter),
        "source_package_sha256": {
            name: hashlib.sha256(git_blob(root, commit, "autonomous-driving-rl/datasets/mapsuite_v1/" + name)).hexdigest()
            for name in SOURCE_PACKAGE_FILES
        },
    }


def verify_git_provenance(root: Path, source: dict[str, Any]) -> dict[str, Any]:
    """Verify the recorded commit and its blobs, including legitimate ancestors."""
    if source.get("exporter_path") != EXPORTER_PATH:
        raise ValueError("Recorded-source Git provenance: exporter_path mismatch")
    identity = git_source_identity(root, source.get("source_git_sha"))
    for key in ("exporter_sha256", "exporter_version", "source_package_sha256"):
        if source.get(key) != identity[key]:
            raise ValueError(f"Recorded-source Git provenance: {key} mismatch against recorded commit blobs")
    return identity


def source_metadata(root: Path, *, require_clean: bool = True) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    sha = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    if require_clean:
        if dirty:
            raise ValueError("Commit tracked source changes before exporting")
        git("ls-files", "--error-unmatch", "scripts/export_mapsuite_v1_previews.py")
    identity = git_source_identity(root, sha)
    return {
        "repository": REPOSITORY, "source_git_sha": sha,
        "source_tracked_tree_clean": not dirty,
        "dataset_id": DATASET_ID, "dataset_version": DATASET_VERSION,
        "metadrive_version": PINNED_METADRIVE_VERSION,
        "metadrive_commit": PINNED_METADRIVE_COMMIT,
        "exporter_path": EXPORTER_PATH, "exporter_version": identity["exporter_version"],
        "exporter_sha256": identity["exporter_sha256"],
        "source_package_sha256": identity["source_package_sha256"],
        "authority": AUTHORITY,
    }


def png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) < 45 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError(f"Invalid/non-empty PNG required: {path.name}")
    width, height = struct.unpack(">II", data[16:24])
    if width <= 0 or height <= 0 or b"IEND" not in data[-12:]:
        raise ValueError(f"Incomplete PNG: {path.name}")
    return width, height


class MetaDriveRenderer:
    """Geometry-only native surface render after exact block fingerprint check."""
    def __init__(self, config: dict, size: int):
        # Import lazily so pure tests and package verification do not start graphics.
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        from scripts.audit_mapsuite_candidates import verify_metadrive_source
        verify_metadrive_source()  # Requires clean imported source at exact pin.
        import cv2
        import numpy as np
        from scripts.audit_evaluation_protocol import make_jsonable
        from metadrive import MetaDriveEnv
        from metadrive.engine.top_down_renderer import draw_top_down_map_native
        self.cv2, self.np = cv2, np
        self.make_jsonable, self.env_type = make_jsonable, MetaDriveEnv
        self.draw = draw_top_down_map_native
        self.env = None
        self.family = None
        self.config, self.size = config, size
        import importlib.metadata
        self.metadata = {
            "backend": "metadrive_native_top_down", "renderer_version": EXPORTER_VERSION,
            "synthetic_test_only": False, "image_size_px": size,
            "semantic_map": True, "margin_fraction": 0.05,
            "traffic_density": 0.0, "environment_steps": 0,
            "opencv_version": cv2.__version__, "numpy_version": np.__version__,
            "pygame_version": importlib.metadata.version("pygame"),
            "panda3d_version": importlib.metadata.version("panda3d"),
        }

    def __call__(self, row: dict[str, str], destination: Path) -> None:
        family = (row["tier"], row["sequence"])
        if family != self.family:
            self.close()
            lanes = self.config["lane_configuration"]
            self.env = self.env_type(dict(
                use_render=False, num_scenarios=20, start_seed=0,
                map=row["sequence"], traffic_density=0.0,
                random_lane_width=lanes["random_lane_width"],
                random_lane_num=lanes["random_lane_num"],
                log_level=50,
            ))
            self.family = family
            # Follow Gate 5's exact shorthand/default path. Package lane_num
            # metadata is historical (2), while this pin's default is 3. The
            # frozen block hash below decides identity; never alter geometry
            # to force agreement with descriptive package metadata.
            actual = self.env.config["map_config"]
            self.metadata["effective_lane_width_m"] = actual["lane_width"]
            self.metadata["effective_base_lane_num"] = actual["lane_num"]
            self.metadata["dataset_declared_base_lane_num"] = lanes["base_lane_num"]
            if actual["lane_num"] != lanes["base_lane_num"]:
                self.metadata["warnings"] = [
                    "Dataset declares base_lane_num=2; pinned Gate 5 reconstruction uses 3. "
                    "Frozen block-sequence hashes remain the geometry authority."
                ]
        self.env.reset(seed=int(row["geometry_generation_seed"]))
        road_map = self.env.current_map
        blocks = self.make_jsonable(road_map.get_meta_data()["block_sequence"])
        if canonical_json_sha256(blocks) != row["geometry_sha256"]:
            raise ValueError(f"Reconstructed geometry hash mismatch: {row['geometry_id']}")
        if "".join(block.ID for block in road_map.blocks) != row["block_ids"]:
            raise ValueError(f"Reconstructed block IDs mismatch: {row['geometry_id']}")
        bbox = road_map.road_network.get_bounding_box()
        span = max(bbox[1] - bbox[0], bbox[3] - bbox[2])
        if not self.np.isfinite(span) or span <= 0:
            raise ValueError("Invalid complete-geometry bounding box")
        image = self.draw(road_map, film_size=(self.size, self.size), semantic_map=True,
                          scaling=self.size * 0.90 / span, return_surface=False)
        if image.shape[:2] != (self.size, self.size) or self.np.ptp(image) == 0:
            raise ValueError(f"Empty/invalid road render: {row['geometry_id']}")
        if not self.cv2.imwrite(str(destination), image, [self.cv2.IMWRITE_PNG_COMPRESSION, 9]):
            raise OSError(f"Could not write {destination.name}")

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            finally:
                self.env, self.family = None, None


def checksum_entries(directory: Path) -> dict[str, str]:
    entries = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlinks are not permitted in a preview package")
        if path.is_file() and path != directory / "CHECKSUMS.sha256":
            entries[path.relative_to(directory).as_posix()] = compute_file_sha256(path)
    return entries


def write_checksums(directory: Path) -> None:
    entries = checksum_entries(directory)
    (directory / "CHECKSUMS.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(entries.items())),
        encoding="utf-8", newline="\n")


def package_zip(directory: Path, destination: Path) -> None:
    # Exclusive creation avoids replacing another export. Fixed archive metadata.
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            if path.is_symlink():
                raise ValueError("Symlink in package")
            entry = zipfile.ZipInfo(path.relative_to(directory).as_posix(), (1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    verify_zip(directory, destination)


def verify_zip(directory: Path, archive_path: Path) -> None:
    expected = {p.relative_to(directory).as_posix(): p for p in directory.rglob("*") if p.is_file()}
    with zipfile.ZipFile(archive_path) as archive:
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != set(expected):
            raise ValueError("ZIP content coverage mismatch")
        for name, path in expected.items():
            if archive.read(name) != path.read_bytes():
                raise ValueError(f"ZIP content mismatch: {name}")


def verify_package(root: Path, directory: Path, *, allow_synthetic: bool = False) -> dict[str, Any]:
    directory = directory.resolve()
    geometries, _, _ = load_geometries(root)
    manifest = read_json(directory / "preview_manifest.json")
    source = manifest["source"]
    if manifest["rendering"]["synthetic_test_only"] and not allow_synthetic:
        raise ValueError("Synthetic test package is not a real geometry preview export")
    git_identity = verify_git_provenance(root, source)
    expected_source = source_metadata(root, require_clean=False)
    for key in ("repository", "dataset_id", "dataset_version", "metadrive_version",
                "metadrive_commit", "exporter_path", "authority"):
        if source[key] != expected_source[key]:
            raise ValueError(f"Source metadata mismatch: {key}")
    checkout_hashes = {name: compute_file_sha256(root / "datasets/mapsuite_v1" / name)
                       for name in SOURCE_PACKAGE_FILES}
    if source["source_package_sha256"] != checkout_hashes:
        raise ValueError("Current checkout compatibility: canonical package differs from recorded Git source")
    if not source["source_tracked_tree_clean"] and not allow_synthetic:
        raise ValueError("Export source was dirty")
    rows = manifest["previews"]
    if len(rows) != 240 or len({r['geometry_id'] for r in rows}) != 240:
        raise ValueError("Manifest requires 240 unique geometries")
    expected_names = set()
    for row, geom in zip(rows, geometries):
        expected = {
            "preview_filename": preview_filename(geom), "geometry_id": geom["geometry_id"],
            "sequence": geom["sequence"], "geometry_generation_seed": int(geom["geometry_generation_seed"]),
            "difficulty_tier": geom["tier"], "split": geom["split"], "geometry_sha256": geom["geometry_sha256"],
            "dataset_id": DATASET_ID, "dataset_version": DATASET_VERSION,
            "metadrive_version": PINNED_METADRIVE_VERSION, "metadrive_commit": PINNED_METADRIVE_COMMIT,
        }
        for key, value in expected.items():
            if row[key] != value:
                raise ValueError(f"Manifest geometry/source mismatch: {key}")
        filename = row["preview_filename"]
        expected_names.add(filename)
        path = directory / "previews" / filename
        if compute_file_sha256(path) != row["image_sha256"]:
            raise ValueError(f"Image checksum mismatch: {filename}")
        if png_dimensions(path) != (row["image_width_px"], row["image_height_px"]):
            raise ValueError(f"Image dimension mismatch: {filename}")
        if row["image_width_px"] != manifest["rendering"]["image_size_px"] or row["image_height_px"] != manifest["rendering"]["image_size_px"]:
            raise ValueError("Image dimensions disagree with renderer config")
    if set(p.name for p in (directory / "previews").iterdir()) != expected_names:
        raise ValueError("Expected exactly 240 preview PNG files")
    with (directory / "preview_manifest.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        csv_rows = list(reader)
        if reader.fieldnames != COLUMNS or csv_rows != [{k: str(r[k]) for k in COLUMNS} for r in rows]:
            raise ValueError("CSV/JSON manifest mismatch")
    if (directory / "SOURCE_REF.txt").read_text(encoding="utf-8") != source_ref(source, manifest["rendering"]):
        raise ValueError("SOURCE_REF mismatch")
    recorded = {}
    for line in (directory / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name or name in recorded:
            raise ValueError("Unsafe/duplicate checksum path")
        recorded[name] = digest
    expected_files = {"README.md", "preview_manifest.csv", "preview_manifest.json", "SOURCE_REF.txt"}
    expected_files.update("previews/" + name for name in expected_names)
    if set(recorded) != expected_files or recorded != checksum_entries(directory):
        raise ValueError("Checksum digest/coverage mismatch")
    return {"png_count": len(rows), "failures": 0, "checksum_status": "VERIFIED",
            "source_git_sha": source["source_git_sha"],
            "internal_integrity_status": "VERIFIED",
            "recorded_source_git_provenance_status": "VERIFIED",
            "current_checkout_compatibility_status": "VERIFIED",
            "exporter_sha256": git_identity["exporter_sha256"],
            "bundle_bytes": sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())}


def source_ref(source: dict, rendering: dict) -> str:
    return "\n".join([AUTHORITY, *[f"{key}: {value}" for key, value in sorted(source.items())
                                if key not in {"authority", "source_package_sha256"}],
                      "source_package_sha256: " + json.dumps(source["source_package_sha256"], sort_keys=True),
                      "rendering: " + json.dumps(rendering, sort_keys=True)]) + "\n"


def validate_output(root: Path, output: Path) -> Path:
    output = output.resolve()
    # Never write inside/above authoritative source, even through a symlink.
    for protected in (root.parent / ".git", root / "datasets", root / "results", root / "configs", root / "src",
                      root / "scripts", root / "tests", root / "docs"):
        protected = protected.resolve()
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("Output overlaps canonical/source files")
    if output.exists():
        raise FileExistsError("Output already exists; use a fresh directory")
    return output


def export_bundle(root: Path, output: Path, renderer: Any, source: dict, *, make_zip: bool = False) -> dict:
    """Transactional publication; failed work remains only in <output>.incomplete."""
    started = time.perf_counter()
    output = validate_output(root, output)
    staging = output.with_name(output.name + ".incomplete")
    validate_output(root, staging)
    zip_path = output.parent / f"mapsuite_v1_previews_{DATASET_VERSION}_{source['source_git_sha'][:12]}.zip"
    if make_zip and zip_path.exists():
        raise FileExistsError("ZIP already exists; choose another output parent")
    rows, _, _ = load_geometries(root)
    before = source_metadata(root, require_clean=False)["source_package_sha256"]
    staging.mkdir(parents=True)
    (staging / "previews").mkdir()
    preview_rows = []
    try:
        for index, row in enumerate(rows, 1):
            name = preview_filename(row)
            path = staging / "previews" / name
            renderer(row, path)
            width, height = png_dimensions(path)
            preview_rows.append({
                "preview_filename": name, "geometry_id": row["geometry_id"],
                "sequence": row["sequence"], "geometry_generation_seed": int(row["geometry_generation_seed"]),
                "difficulty_tier": row["tier"], "split": row["split"],
                "geometry_sha256": row["geometry_sha256"], "image_sha256": compute_file_sha256(path),
                "image_width_px": width, "image_height_px": height,
                "dataset_id": DATASET_ID, "dataset_version": DATASET_VERSION,
                "metadrive_version": PINNED_METADRIVE_VERSION, "metadrive_commit": PINNED_METADRIVE_COMMIT,
            })
            if index % 20 == 0:
                print(f"[PREVIEW] {index}/240", flush=True)
    finally:
        renderer.close()
    if before != source_metadata(root, require_clean=False)["source_package_sha256"]:
        raise ValueError("Canonical package changed during export")
    write_json(staging / "preview_manifest.json", {
        "source": source, "rendering": renderer.metadata, "previews": preview_rows,
    })
    with (staging / "preview_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(preview_rows)
    (staging / "SOURCE_REF.txt").write_text(source_ref(source, renderer.metadata), encoding="utf-8", newline="\n")
    label = "SYNTHETIC TEST ONLY" if renderer.metadata["synthetic_test_only"] else "240 frozen road geometries"
    (staging / "README.md").write_text(
        f"# MapSuite V1 Level 1 previews — {label}\n\n{AUTHORITY}\n\n"
        f"Dataset: {DATASET_ID} {DATASET_VERSION}. Source: {source['source_git_sha']}.\n"
        f"MetaDrive: {PINNED_METADRIVE_VERSION}, commit {PINNED_METADRIVE_COMMIT}.\n\n"
        "Exactly 240 top-down PNGs; no trajectories, agent results, meshes or RL transitions.\n"
        "The frozen GitHub CSV/config/manifests remain the scientific benchmark.\n"
        "Do not use previews to change splits/seeds/difficulty or select agents using TEST.\n\n"
        "Extract the ZIP, then run `sha256sum -c CHECKSUMS.sha256` here.\n"
        "With a reviewed repository checkout, from autonomous-driving-rl run:\n"
        "`python scripts/export_mapsuite_v1_previews.py --verify <extracted-directory>`\n"
        "Inspect SOURCE_REF.txt and both manifests for identity/provenance; compare to GitHub.\n",
        encoding="utf-8", newline="\n")
    write_checksums(staging)
    result = verify_package(root, staging, allow_synthetic=renderer.metadata["synthetic_test_only"])
    if output.exists():
        raise FileExistsError("Output appeared during export")
    staging.rename(output)
    if make_zip:
        package_zip(output, zip_path)
    result.update({"output_directory": str(output), "zip_path": str(zip_path) if make_zip else None,
                   "zip_bytes": zip_path.stat().st_size if make_zip else 0,
                   "duration_seconds": round(time.perf_counter() - started, 3)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/mapsuite_v1_previews"))
    parser.add_argument("--zip", action="store_true", help="Create and verify a deterministic ZIP beside output")
    parser.add_argument("--size", type=int, default=1024, help="Square PNG dimensions (256..4096)")
    parser.add_argument("--verify", type=Path, help="Verify a complete package without rendering")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_package(PROJECT_ROOT, args.verify), indent=2))
        return
    if not 256 <= args.size <= 4096:
        parser.error("--size must be between 256 and 4096")
    validate_output(PROJECT_ROOT, args.output)
    source = source_metadata(PROJECT_ROOT)
    _, _, config = load_geometries(PROJECT_ROOT)
    renderer = MetaDriveRenderer(config, args.size)
    try:
        result = export_bundle(PROJECT_ROOT, args.output, renderer, source, make_zip=args.zip)
    finally:
        renderer.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
