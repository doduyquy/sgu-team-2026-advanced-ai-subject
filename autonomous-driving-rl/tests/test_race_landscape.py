"""The low-resolution visual raster should retain reference-width road marks."""
from types import SimpleNamespace
from unittest.mock import MagicMock
import unittest
from PySide6.QtCore import QProcess  # noqa: F401
from src.visualization.race_landscape import marking_pixel_widths, _reference_markings
from src.visualization.race_landscape import visual_map_layout, _landscape_semantics
from src.visualization.race_road_markings import marking_paths, ribbon_vertices, rebuild_markings
from unittest.mock import patch
import numpy as np


class RoadMarkingTests(unittest.TestCase):
    def test_pixel_width_tracks_resolution_instead_of_widening_markings(self):
        self.assertEqual(marking_pixel_widths(22),(2,3))
        self.assertEqual(marking_pixel_widths(8),(1,1))
        self.assertEqual(marking_pixel_widths(6),(1,1))

    def test_visual_sampler_uses_current_map_without_changing_height_or_geometry(self):
        current_map=MagicMock()
        terrain=SimpleNamespace(engine=SimpleNamespace(current_map=current_map,
            global_config={'show_crosswalk':True}),_semantic_map_size=2048,
            _semantic_map_pixel_per_meter=8)
        result=_reference_markings(terrain,(5,10))
        self.assertIs(result,current_map.get_semantic_map.return_value)
        current_map.get_semantic_map.assert_called_once_with((5,10),size=2048,
            pixels_per_meter=8,white_line_thickness=1,yellow_line_thickness=1,
            layer=['lane','lane_line','crosswalk'])
        current_map.get_height_map.assert_not_called()

    def test_detail_map_covers_offset_roads_with_padding_within_gpu_budget(self):
        for bounds in ((-200, 600, -500, 10), (10, 40, 90, 120), (-1100, 900, -900, 800)):
            center, size, ppm = visual_map_layout(bounds, max_dimension=8192)
            self.assertLessEqual(int(size*ppm), 8192)
            for value, origin in zip(bounds, (center[0], center[0], center[1], center[1])):
                self.assertLessEqual(abs(value-origin), size/2-16)

    def test_reset_rebuilds_rgb_detail_and_preserves_numeric_semantics(self):
        original = np.full((4, 4, 1), 20, dtype=np.uint8)
        terrain = SimpleNamespace()
        with patch('src.visualization.race_landscape._reference_markings', return_value=original), \
             patch('src.visualization.race_landscape._update_visual_semantics') as update:
            self.assertIs(_landscape_semantics(terrain, (10, 20)), original)
            update.assert_called_once_with(terrain, (10, 20))

    def test_curved_dash_preserves_interior_points_and_physical_two_metre_gaps(self):
        from metadrive.constants import MetaDriveType
        s = np.arange(0, 12.01, .25)
        points = np.column_stack((20*np.sin(s/20), 20*(1-np.cos(s/20))))
        features = {'curve': dict(type=MetaDriveType.LINE_BROKEN_SINGLE_WHITE, polyline=points),
                    'lane': dict(type=MetaDriveType.LANE_SURFACE_STREET)}
        paths = list(marking_paths(features))
        self.assertEqual(len(paths), 3)
        np.testing.assert_equal(paths[0][0], points[:9])
        np.testing.assert_equal(paths[1][0], points[16:25])
        ribbon = ribbon_vertices(paths[0][0], .1)
        np.testing.assert_allclose(ribbon.mean(axis=1), points[:9], atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(ribbon[:, 0]-ribbon[:, 1], axis=1), .1, atol=1e-5)

    def test_marking_rebuild_removes_old_map_without_adding_physics(self):
        from panda3d.core import NodePath
        from metadrive.constants import CamMask, MetaDriveType
        current_map = MagicMock()
        current_map.get_map_features.return_value = {
            'line': dict(type=MetaDriveType.LINE_SOLID_SINGLE_YELLOW, polyline=[[0, 0], [5, 0]])}
        terrain = SimpleNamespace(engine=SimpleNamespace(current_map=current_map, render=NodePath('world')))
        rebuild_markings(terrain)
        previous = terrain._race_markings
        self.assertFalse(previous.isHidden(CamMask.MainCam))
        self.assertTrue(previous.isHidden(CamMask.SemanticCam))
        self.assertEqual(len(previous.findAllMatches('**/+BulletBodyNode')), 0)
        rebuild_markings(terrain)
        self.assertTrue(previous.isEmpty())
        self.assertEqual(len(terrain.engine.render.getChildren()), 1)
        terrain.engine.current_map = None
        rebuild_markings(terrain)
        self.assertEqual(len(terrain.engine.render.getChildren()), 0)


if __name__=='__main__': unittest.main()
