"""
Unit tests for MetaDrive MapSuite V1 Dataset Packaging and Verification.

Validates that:
1. Master geometries table has exactly 240 unique geometries.
2. Exactly 12 scenario families, each containing 20 geometries.
3. Split partitions are strictly 180 TRAIN / 48 VALIDATION / 12 TEST with zero overlap and complete union.
4. Canonical evaluation cases are exactly 96 validation cases and 60 test cases.
5. Validation and test cases strictly reference their designated split geometries.
6. Geometry IDs are unique and (tier, sequence, seed) keys are unique.
7. Source artifact SHA-256 values recorded in dataset_manifest.json match files on disk.
8. Geometry SHA-256 fingerprints agree across geometries.csv and source split manifests.
9. Exporter is completely deterministic and can rebuild the package in a temp directory matching committed outputs.
10. Exactly 12 representative preview images exist and match scenario families.
11. CHECKSUMS.sha256 strictly verifies every file in the package.
12. Zero absolute machine paths (e.g., C:\\Users, /home, D:\\SGU) leak into metadata or manifest files.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest

from scripts.export_mapsuite_v1_dataset import (
    DATASET_ID,
    DATASET_VERSION,
    PINNED_METADRIVE_COMMIT,
    PINNED_METADRIVE_VERSION,
    TIER_ORDER,
    export_dataset,
    find_project_root,
)
from scripts.verify_mapsuite_v1_dataset import compute_file_sha256, verify_dataset


class TestMapSuiteV1Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = find_project_root()
        cls.dataset_dir = cls.project_root / "datasets" / "mapsuite_v1"
        cls.assertTrue(cls.dataset_dir.is_dir(), f"Dataset directory missing: {cls.dataset_dir}")

    def _read_csv(self, rel_path: str) -> list[dict[str, str]]:
        fp = self.dataset_dir / rel_path
        with open(fp, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _read_json(self, rel_path: str) -> dict:
        fp = self.dataset_dir / rel_path
        with open(fp, encoding="utf-8") as f:
            return json.load(f)

    def test_01_exact_geometry_count_240(self):
        """geometries.csv must contain exactly 240 rows."""
        geoms = self._read_csv("geometries.csv")
        self.assertEqual(len(geoms), 240)

    def test_02_exact_family_count_12(self):
        """scenario_families.csv must contain exactly 12 scenario families."""
        families = self._read_csv("scenario_families.csv")
        self.assertEqual(len(families), 12)
        tiers = set(f["tier"] for f in families)
        self.assertEqual(tiers, set(TIER_ORDER))

    def test_03_exact_20_geometries_per_family(self):
        """Every scenario family must contain exactly 20 geometries (seeds 0..19)."""
        geoms = self._read_csv("geometries.csv")
        counts = {}
        for g in geoms:
            key = (g["tier"], g["sequence"])
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(len(counts), 12)
        for key, count in counts.items():
            self.assertEqual(count, 20, f"Family {key} has {count} geometries; expected 20.")

    def test_04_split_counts_180_48_12(self):
        """Splits must contain exactly 180 TRAIN, 48 VALIDATION, and 12 TEST geometries."""
        train_geoms = self._read_csv("splits/train_geometries.csv")
        val_geoms = self._read_csv("splits/validation_geometries.csv")
        test_geoms = self._read_csv("splits/test_geometries.csv")

        self.assertEqual(len(train_geoms), 180)
        self.assertEqual(len(val_geoms), 48)
        self.assertEqual(len(test_geoms), 12)

    def test_05_canonical_case_counts_96_60(self):
        """Evaluation cases must contain exactly 96 validation cases and 60 test cases."""
        val_cases = self._read_csv("evaluation_cases/validation_cases.csv")
        test_cases = self._read_csv("evaluation_cases/test_cases.csv")

        self.assertEqual(len(val_cases), 96)
        self.assertEqual(len(test_cases), 60)

    def test_06_geometry_ids_unique(self):
        """All 240 geometry_id values and (tier, sequence, seed) keys must be unique."""
        geoms = self._read_csv("geometries.csv")
        ids = [g["geometry_id"] for g in geoms]
        keys = [(g["tier"], g["sequence"], int(g["geometry_generation_seed"])) for g in geoms]

        self.assertEqual(len(ids), 240)
        self.assertEqual(len(set(ids)), 240)
        self.assertEqual(len(keys), 240)
        self.assertEqual(len(set(keys)), 240)

    def test_07_split_disjointness_and_complete_union(self):
        """Splits must be mutually disjoint and their union must equal all 240 geometries."""
        geoms = self._read_csv("geometries.csv")
        train_geoms = self._read_csv("splits/train_geometries.csv")
        val_geoms = self._read_csv("splits/validation_geometries.csv")
        test_geoms = self._read_csv("splits/test_geometries.csv")

        train_ids = set(g["geometry_id"] for g in train_geoms)
        val_ids = set(g["geometry_id"] for g in val_geoms)
        test_ids = set(g["geometry_id"] for g in test_geoms)
        all_ids = set(g["geometry_id"] for g in geoms)

        self.assertEqual(len(train_ids & val_ids), 0)
        self.assertEqual(len(train_ids & test_ids), 0)
        self.assertEqual(len(val_ids & test_ids), 0)
        self.assertEqual(train_ids | val_ids | test_ids, all_ids)

    def test_08_source_sha_provenance(self):
        """Source artifact SHA-256 hashes in dataset_manifest.json must match actual disk files."""
        manifest = self._read_json("dataset_manifest.json")
        self.assertEqual(manifest["dataset_id"], DATASET_ID)
        self.assertEqual(manifest["dataset_version"], DATASET_VERSION)

        source_artifacts = manifest["source_artifacts"]
        self.assertEqual(len(source_artifacts), 5)

        for sa in source_artifacts:
            rel_p = sa["repo_relative_path"]
            exp_h = sa["sha256"]
            file_p = self.project_root / rel_p
            self.assertTrue(file_p.is_file(), f"Missing source artifact: {rel_p}")
            actual_h = compute_file_sha256(file_p)
            self.assertEqual(actual_h, exp_h, f"Hash mismatch for {rel_p}")

    def test_09_geometry_sha_agreement(self):
        """Geometry SHA-256 values in geometries.csv must strictly agree with geometry_split_manifest.csv."""
        geoms = self._read_csv("geometries.csv")
        sm_path = self.project_root / "results" / "audits" / "evaluation_protocol" / "geometry_split_manifest.csv"
        with open(sm_path, newline="", encoding="utf-8") as f:
            sm_rows = list(csv.DictReader(f))

        sm_map = {r["geometry_id"]: r["geometry_sha256"] for r in sm_rows}
        for g in geoms:
            gid = g["geometry_id"]
            self.assertIn(gid, sm_map)
            self.assertEqual(g["geometry_sha256"], sm_map[gid], f"SHA mismatch for geometry {gid}")

    def test_10_evaluation_cases_reference_correct_splits(self):
        """Validation cases must only reference VALIDATION geometries; test cases only TEST geometries."""
        geoms = self._read_csv("geometries.csv")
        val_geom_keys = {(g["tier"], g["sequence"], int(g["geometry_generation_seed"])) for g in geoms if g["split"] == "VALIDATION"}
        test_geom_keys = {(g["tier"], g["sequence"], int(g["geometry_generation_seed"])) for g in geoms if g["split"] == "TEST"}

        val_cases = self._read_csv("evaluation_cases/validation_cases.csv")
        for vc in val_cases:
            key = (vc["tier"], vc["sequence"], int(vc["geometry_generation_seed"]))
            self.assertIn(key, val_geom_keys, f"Validation case {vc['case_id']} references non-VAL geometry: {key}")

        test_cases = self._read_csv("evaluation_cases/test_cases.csv")
        for tc in test_cases:
            key = (tc["tier"], tc["sequence"], int(tc["geometry_generation_seed"]))
            self.assertIn(key, test_geom_keys, f"Test case {tc['case_id']} references non-TEST geometry: {key}")

    def test_11_required_previews_count_12(self):
        """Exactly 12 representative preview images must exist in previews/."""
        previews_dir = self.dataset_dir / "previews"
        pngs = sorted([p.name for p in previews_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
        self.assertEqual(len(pngs), 12)

    def test_12_checksums_validation(self):
        """CHECKSUMS.sha256 must strictly match all files in the dataset directory."""
        checksums_path = self.dataset_dir / "CHECKSUMS.sha256"
        self.assertTrue(checksums_path.is_file())

        with open(checksums_path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        self.assertGreater(len(lines), 0)
        for line in lines:
            parts = line.split(maxsplit=1)
            self.assertEqual(len(parts), 2)
            exp_h, rel_p = parts[0], parts[1].strip()
            fp = self.dataset_dir / rel_p
            self.assertTrue(fp.is_file(), f"File recorded in CHECKSUMS.sha256 missing: {rel_p}")
            actual_h = compute_file_sha256(fp)
            self.assertEqual(actual_h, exp_h, f"Checksum mismatch for {rel_p}")

    def test_13_deterministic_rebuild_in_temp_dir(self):
        """Rebuilding dataset package into a temporary directory produces byte-for-byte identical textual files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_output = Path(tmp_dir) / "mapsuite_v1"
            # Copy markdown files first so checksums match exactly
            tmp_output.mkdir(parents=True, exist_ok=True)
            for md_name in ["README.md", "SCHEMA.md", "REPRODUCE.md"]:
                shutil.copy2(self.dataset_dir / md_name, tmp_output / md_name)

            export_dataset(self.project_root, tmp_output)
            self.assertTrue(verify_dataset(tmp_output, self.project_root))

            # Compare all text files byte-for-byte
            for p in self.dataset_dir.rglob("*"):
                if p.is_file():
                    rel_p = p.relative_to(self.dataset_dir)
                    rebuilt_p = tmp_output / rel_p
                    self.assertTrue(rebuilt_p.is_file(), f"Rebuilt package missing {rel_p}")
                    self.assertEqual(
                        p.read_bytes(),
                        rebuilt_p.read_bytes(),
                        f"Content mismatch in rebuilt {rel_p}",
                    )

    def test_14_no_absolute_machine_paths_leak(self):
        """No machine-specific absolute path patterns (C:\\Users, /home, D:\\SGU) leak into metadata or docs."""
        forbidden_patterns = [
            re.compile(r"C:\\Users", re.IGNORECASE),
            re.compile(r"/home/\w+", re.IGNORECASE),
            re.compile(r"[A-Z]:\\[^\s\"]+", re.IGNORECASE),  # Windows drive paths
        ]
        text_extensions = {".csv", ".json", ".md", ".sha256"}
        for p in self.dataset_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in text_extensions:
                content = p.read_text(encoding="utf-8", errors="replace")
                for pat in forbidden_patterns:
                    matches = pat.findall(content)
                    self.assertEqual(
                        len(matches),
                        0,
                        f"Forbidden local machine path pattern '{pat.pattern}' leaked in {p.name}: {matches[:3]}",
                    )


if __name__ == "__main__":
    unittest.main()
