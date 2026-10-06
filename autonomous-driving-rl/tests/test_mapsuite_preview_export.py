"""Fast export/integrity tests; no MetaDrive engine or benchmark agents."""
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import shutil
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
import zlib

from scripts import export_mapsuite_v1_previews as previews


def tiny_png() -> bytes:
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\x00" + b"\x80\x90\xa0" * 8) * 8)) + chunk(b"IEND", b""))


class SyntheticRenderer:
    metadata = {"backend": "synthetic_unit_test", "synthetic_test_only": True, "image_size_px": 8}

    def __init__(self):
        self.calls = []
        self.closed = False

    def __call__(self, row, path):
        self.calls.append(dict(row))
        path.write_bytes(tiny_png())

    def close(self):
        self.closed = True


class TestMapSuitePreviewExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = previews.PROJECT_ROOT
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.output = cls.base / "complete"
        cls.renderer = SyntheticRenderer()
        with redirect_stdout(io.StringIO()):
            cls.rows, cls.dataset, cls.config = previews.load_geometries(cls.root)
            cls.source = previews.source_metadata(cls.root, require_clean=False)
            cls.result = previews.export_bundle(cls.root, cls.output, cls.renderer, cls.source, make_zip=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def verify(self, path):
        with redirect_stdout(io.StringIO()):
            return previews.verify_package(self.root, path, allow_synthetic=True)

    def copied_package(self, parent):
        target = Path(parent) / "copy"
        shutil.copytree(self.output, target)
        return target

    def test_01_frozen_240_records_and_unique_ids_names(self):
        self.assertEqual(len(self.rows), 240)
        self.assertEqual(len({r['geometry_id'] for r in self.rows}), 240)
        self.assertEqual(len({previews.preview_filename(r).casefold() for r in self.rows}), 240)
        with (self.root / "datasets/mapsuite_v1/geometries.csv").open(newline="", encoding="utf-8") as handle:
            self.assertEqual(self.rows, list(csv.DictReader(handle)))

    def test_02_deterministic_names_preserve_identity(self):
        row = self.rows[0]
        self.assertEqual(previews.preview_filename(row), "geom_easy_SCS_seed0__Easy__SCS__gseed-00.png")
        self.assertEqual(previews.preview_filename(row), previews.preview_filename(dict(row)))
        self.assertNotEqual(previews.safe_component("a/b"), previews.safe_component("a_b"))
        self.assertNotEqual(previews.safe_component("a~00002fb"), previews.safe_component("a/b"))

    def test_03_unsafe_filename_characters_are_escaped(self):
        row = {**self.rows[0], "geometry_id": "../../CON:<x>\\a", "sequence": "S/C"}
        name = previews.preview_filename(row)
        self.assertFalse(any(c in name for c in '/\\:<>"|?*'))
        self.assertFalse(name.startswith("."))
        for value in ["", ".", ".."]:
            with self.assertRaises(ValueError):
                previews.safe_component(value)

    def test_04_full_synthetic_manifest_count_and_source_fields(self):
        manifest = previews.read_json(self.output / "preview_manifest.json")
        self.assertEqual(len(manifest['previews']), 240)
        self.assertEqual(self.result['png_count'], 240)
        self.assertTrue(manifest['rendering']['synthetic_test_only'])
        self.assertEqual(manifest['source'], self.source)
        self.assertEqual(manifest['previews'][0]['metadrive_commit'], previews.PINNED_METADRIVE_COMMIT)
        self.assertEqual(manifest['previews'][0]['geometry_sha256'], self.rows[0]['geometry_sha256'])
        self.assertIn(previews.AUTHORITY, (self.output / 'SOURCE_REF.txt').read_text())
        self.assertTrue(self.renderer.closed)

    def test_05_checksums_exact_coverage_and_repeatability(self):
        path = self.output / "CHECKSUMS.sha256"
        before = path.read_bytes()
        previews.write_checksums(self.output)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(len(before.splitlines()), 244)
        self.assertEqual(self.verify(self.output)['checksum_status'], 'VERIFIED')

    def test_06_zip_self_describing_deterministic_and_complete(self):
        original = Path(self.result['zip_path'])
        with zipfile.ZipFile(original) as archive:
            self.assertEqual(len(archive.namelist()), 245)
            self.assertEqual(sum(name.endswith('.png') for name in archive.namelist()), 240)
            self.assertIn('SOURCE_REF.txt', archive.namelist())
            self.assertIn('CHECKSUMS.sha256', archive.namelist())
            self.assertTrue(all(i.date_time == (1980, 1, 1, 0, 0, 0) for i in archive.infolist()))
        with tempfile.TemporaryDirectory() as parent:
            second = Path(parent) / 'second.zip'
            previews.package_zip(self.output, second)
            self.assertEqual(original.read_bytes(), second.read_bytes())

    def test_07_source_package_remains_unchanged_and_all_rows_rendered(self):
        after = previews.source_metadata(self.root, require_clean=False)
        self.assertEqual(self.source['source_package_sha256'], after['source_package_sha256'])
        self.assertEqual(self.renderer.calls, self.rows)

    def test_08_protect_canonical_paths_and_overwrites(self):
        for protected in ['datasets/mapsuite_v1/previews/new', 'results/audits/new', 'src/new', '../.git/new', '.', '..']:
            with self.subTest(protected=protected), self.assertRaises(ValueError):
                previews.validate_output(self.root, self.root / protected)
        with self.assertRaises(FileExistsError):
            previews.validate_output(self.root, self.output)

    def test_09_fail_loudly_and_close_resources_without_publishing(self):
        class BrokenRenderer(SyntheticRenderer):
            def __call__(self, row, path):
                raise ValueError('Reconstructed geometry hash mismatch')
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / 'failed'
            renderer = BrokenRenderer()
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'hash mismatch'):
                previews.export_bundle(self.root, output, renderer, self.source)
            self.assertTrue(renderer.closed)
            self.assertFalse(output.exists())
            self.assertTrue(output.with_name('failed.incomplete').exists())

    def test_10_manifest_identity_mutation_rejected_even_with_new_checksums(self):
        with tempfile.TemporaryDirectory() as parent:
            package = self.copied_package(parent)
            manifest = previews.read_json(package / 'preview_manifest.json')
            manifest['previews'][0]['split'] = 'TEST'
            previews.write_json(package / 'preview_manifest.json', manifest)
            previews.write_checksums(package)
            with self.assertRaisesRegex(ValueError, 'split'):
                self.verify(package)

    def test_11_missing_or_corrupt_image_rejected(self):
        with tempfile.TemporaryDirectory() as parent:
            package = self.copied_package(parent)
            image = next((package / 'previews').iterdir())
            image.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                self.verify(package)
            image.unlink()
            with self.assertRaises(FileNotFoundError):
                self.verify(package)

    def test_12_checksum_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as parent:
            package = self.copied_package(parent)
            (package / 'CHECKSUMS.sha256').write_text('0' * 64 + '  ../escape\n')
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                self.verify(package)

    def test_13_source_provenance_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as parent:
            package = self.copied_package(parent)
            manifest = previews.read_json(package / 'preview_manifest.json')
            manifest['source']['metadrive_commit'] = '0' * 40
            previews.write_json(package / 'preview_manifest.json', manifest)
            previews.write_checksums(package)
            with self.assertRaisesRegex(ValueError, 'metadata mismatch'):
                self.verify(package)

    def test_14_fixture_export_cannot_pass_real_verifier(self):
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Synthetic'):
            previews.verify_package(self.root, self.output)

    def test_15_canonical_geometry_mutation_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as parent:
            root = Path(parent) / 'project'
            shutil.copytree(self.root / 'datasets', root / 'datasets')
            shutil.copytree(self.root / 'results/audits', root / 'results/audits')
            path = root / 'datasets/mapsuite_v1/geometries.csv'
            path.write_text(path.read_text().replace('geom_easy_SCS_seed0', 'new_identity', 1))
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'verification failed'):
                previews.load_geometries(root)

    def test_16_duplicate_names_rejected_before_rendering(self):
        with patch.object(previews, 'preview_filename', return_value='collision.png'):
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Duplicate'):
                previews.load_geometries(self.root)

    def test_17_extra_package_file_rejected(self):
        with tempfile.TemporaryDirectory() as parent:
            package = self.copied_package(parent)
            (package / 'unexpected.txt').write_text('not a Level 1 package member')
            previews.write_checksums(package)
            with self.assertRaisesRegex(ValueError, 'coverage'):
                self.verify(package)

    def test_18_full_repeat_export_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / 'repeat'
            with redirect_stdout(io.StringIO()):
                previews.export_bundle(self.root, output, SyntheticRenderer(), self.source)
            for path in self.output.rglob('*'):
                if path.is_file():
                    self.assertEqual(path.read_bytes(), (output / path.relative_to(self.output)).read_bytes())

    def fake_native_renderer(self):
        # Check real renderer orchestration with a small in-memory road map.
        # The geometry hash check, reset seed, generation config and rendering
        # path remain production code; only the native engine is substituted.
        import numpy as np
        renderer = previews.MetaDriveRenderer.__new__(previews.MetaDriveRenderer)
        renderer.config, renderer.size = self.config, 8
        renderer.family = renderer.env = None
        renderer.metadata = {}
        renderer.np = np
        renderer.make_jsonable = lambda value: value
        blocks = [{"id": c} for c in self.rows[0]['block_ids']]
        road_map = SimpleNamespace(
            get_meta_data=lambda: {"block_sequence": blocks},
            blocks=[SimpleNamespace(ID=c) for c in self.rows[0]['block_ids']],
            road_network=SimpleNamespace(get_bounding_box=lambda: (0, 200, -50, 100)),
        )
        settings, seeds = {}, []
        fake_env = SimpleNamespace(config={"map_config": {"lane_width": 3.5, "lane_num": 3}},
                                   current_map=road_map, reset=lambda seed: seeds.append(seed),
                                   close=lambda: None)
        def factory(config):
            settings.update(config)
            return fake_env
        renderer.env_type = factory
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[0, 0] = 255
        renderer.draw = lambda *args, **kwargs: image
        def imwrite(path, image, config):
            Path(path).write_bytes(tiny_png())
            return True
        renderer.cv2 = SimpleNamespace(imwrite=imwrite, IMWRITE_PNG_COMPRESSION=16)
        row = {**self.rows[0], "geometry_sha256": previews.canonical_json_sha256(blocks)}
        return renderer, row, settings, seeds

    def test_19_native_reconstruction_uses_gate5_path_and_frozen_hash(self):
        renderer, row, settings, seeds = self.fake_native_renderer()
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / 'road.png'
            try:
                renderer(row, output)
                self.assertEqual(previews.png_dimensions(output), (8, 8))
                self.assertEqual(settings['map'], row['sequence'])
                self.assertNotIn('map_config', settings)
                self.assertEqual(settings['traffic_density'], 0.0)
                self.assertEqual(seeds, [int(row['geometry_generation_seed'])])
                self.assertEqual(renderer.metadata['effective_base_lane_num'], 3)
                self.assertIn('Frozen block-sequence hashes', renderer.metadata['warnings'][0])
            finally:
                renderer.close()

    def test_20_native_hash_mismatch_fails_before_image_write(self):
        renderer, row, _, _ = self.fake_native_renderer()
        row['geometry_sha256'] = '0' * 64
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / 'wrong.png'
            try:
                with self.assertRaisesRegex(ValueError, 'geometry hash mismatch'):
                    renderer(row, output)
                self.assertFalse(output.exists())
            finally:
                renderer.close()


if __name__ == '__main__':
    unittest.main()
