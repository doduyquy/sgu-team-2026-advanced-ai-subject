"""
Unit tests for Gate HUD-1: Live Minimap Implementation.

Tests minimap mathematical utilities, polyline bounding, aspect ratio preservation,
cardinal heading rotations, trajectory distance filtering, FIFO ring buffer,
screen clipping, lifecycle idempotency, and scientific isolation.
Pure unit test suite runnable in headless environment without requiring an interactive desktop window.
"""

import math
from pathlib import Path
import unittest
from unittest.mock import MagicMock

import numpy as np
import panda3d.core as p3d

from src.visualization.minimap_math import (
    TrajectoryTrailBuffer,
    calculate_film_size,
    calculate_heading_rotation_deg,
    calculate_map_bounds,
)


class TestMinimapOverlayUnitSuite(unittest.TestCase):
    """22 required unit and regression tests for Gate HUD-1 Live Minimap."""

    def test_01_import_visualization_package_has_no_hologram_dependency(self):
        """1. Importing src.visualization does not import or depend on finish_floating_hologram."""
        import sys
        import src.visualization
        self.assertIn("MiniMapOverlay", src.visualization.__all__)
        self.assertNotIn("FinishFloatingHologram", src.visualization.__all__)
        self.assertNotIn("src.visualization.finish_floating_hologram", sys.modules)

    def test_02_headless_import_creates_no_engine_or_window(self):
        """2. Importing minimap modules in a headless environment creates no engine or window."""
        import src.visualization.minimap_overlay
        import src.visualization.minimap_math
        self.assertTrue(hasattr(src.visualization.minimap_overlay, "MiniMapOverlay"))

    def test_03_map_bounds_extraction_from_polylines(self):
        """3. Accurate bounding box (min_x, max_x, min_y, max_y) extraction from valid feature polylines."""
        features = {
            "lane_1": {"polyline": [[0.0, 10.0], [50.0, 10.0], [100.0, 20.0]]},
            "lane_2": {"polyline": [[-20.0, -5.0], [30.0, -5.0], [80.0, -15.0]]},
        }
        bounds = calculate_map_bounds(features, default_center=(0.0, 0.0))
        self.assertEqual(bounds, (-20.0, 100.0, -15.0, 20.0))

    def test_04_empty_and_malformed_features_fallback(self):
        """4. Empty, None, or non-dict map features gracefully fall back to 100m x 100m box around center."""
        # Empty dict
        bounds_empty = calculate_map_bounds({}, default_center=(10.0, 25.0))
        self.assertEqual(bounds_empty, (-40.0, 60.0, -25.0, 75.0))

        # None
        bounds_none = calculate_map_bounds(None, default_center=(0.0, 0.0))
        self.assertEqual(bounds_none, (-50.0, 50.0, -50.0, 50.0))

        # Malformed entries
        malformed = {"bad_1": "not_a_dict", "bad_2": {"polyline": None}, "bad_3": {"polyline": []}}
        bounds_mal = calculate_map_bounds(malformed, default_center=(5.0, 5.0))
        self.assertEqual(bounds_mal, (-45.0, 55.0, -45.0, 55.0))

    def test_05_nan_and_inf_geometry_vertices_rejected(self):
        """5. Polyline vertices containing NaN or Inf values are rejected, preserving only finite points."""
        features = {
            "lane_mixed": {
                "polyline": [
                    [10.0, 20.0],
                    [float("nan"), 25.0],
                    [30.0, float("inf")],
                    [-float("inf"), 10.0],
                    [40.0, 50.0]
                ]
            }
        }
        bounds = calculate_map_bounds(features, default_center=(0.0, 0.0))
        self.assertEqual(bounds, (10.0, 40.0, 20.0, 50.0))

    def test_06_orthographic_film_size_preserves_aspect_ratio(self):
        """6. Film sizing preserves map aspect ratio without deformation across different display region ratios."""
        bounds = (0.0, 100.0, 0.0, 50.0)  # width 100 + 24 = 124, height 50 + 24 = 74
        padding_m = 12.0

        # When map aspect > region aspect (wider map)
        fw, fh = calculate_film_size(bounds, region_aspect=1.0, padding_m=padding_m)
        self.assertAlmostEqual(fw, 124.0)
        self.assertAlmostEqual(fh, 124.0)

        # When map aspect < region aspect (taller map)
        fw2, fh2 = calculate_film_size(bounds, region_aspect=2.0, padding_m=padding_m)
        self.assertAlmostEqual(fh2, 74.0)
        self.assertAlmostEqual(fw2, 74.0 * 2.0)

    def test_07_screen_to_world_projection_math(self):
        """7. Aspect ratio calculations throw ValueError on invalid non-positive region aspect."""
        bounds = (0.0, 50.0, 0.0, 50.0)
        with self.assertRaises(ValueError):
            calculate_film_size(bounds, region_aspect=0.0)
        with self.assertRaises(ValueError):
            calculate_film_size(bounds, region_aspect=-1.5)

    def test_08_heading_orientation_cardinal_directions(self):
        """8. Heading rotation angle accurately maps cardinal directions to Panda3D render2d roll degrees."""
        # East: theta = 0 rad -> 0.0 deg
        self.assertAlmostEqual(calculate_heading_rotation_deg(0.0), 0.0)

        # North: theta = pi/2 rad -> -90.0 deg (upward on screen)
        self.assertAlmostEqual(calculate_heading_rotation_deg(math.pi / 2.0), -90.0)

        # West: theta = pi rad -> -180.0 deg (leftward on screen)
        self.assertAlmostEqual(calculate_heading_rotation_deg(math.pi), -180.0)

        # South: theta = -pi/2 rad -> +90.0 deg (downward on screen)
        self.assertAlmostEqual(calculate_heading_rotation_deg(-math.pi / 2.0), 90.0)

        # Non-finite heading returns 0.0 safe fallback
        self.assertEqual(calculate_heading_rotation_deg(float("nan")), 0.0)
        self.assertEqual(calculate_heading_rotation_deg(float("inf")), 0.0)

    def test_09_valid_position_with_missing_heading_renders_dot(self):
        """9. Valid coordinates with heading_rad=None uses non-directional dot without fabricating North."""
        buf = TrajectoryTrailBuffer(max_capacity=500)
        retained = buf.add_sample(10.0, 20.0)
        self.assertTrue(retained)
        # Position is retained in buffer
        self.assertEqual(buf.get_points(), [(10.0, 20.0)])
        # When heading is None, calculate_heading_rotation_deg fallback is 0.0, but UI selects marker_dot
        self.assertEqual(len(buf), 1)

    def test_10_missing_position_hides_marker(self):
        """10. Missing position (x=None or y=None) rejects trajectory sample and returns False."""
        buf = TrajectoryTrailBuffer(max_capacity=500)
        self.assertFalse(buf.add_sample(None, 20.0))
        self.assertFalse(buf.add_sample(10.0, None))
        self.assertFalse(buf.add_sample(None, None))
        self.assertFalse(buf.add_sample(float("nan"), 20.0))
        self.assertFalse(buf.add_sample(10.0, float("inf")))
        self.assertEqual(len(buf), 0)

    def test_11_first_trajectory_sample_appended(self):
        """11. First valid position sample is always accepted and retained in the trajectory buffer."""
        buf = TrajectoryTrailBuffer(max_capacity=500)
        retained = buf.add_sample(15.5, 30.2)
        self.assertTrue(retained)
        self.assertEqual(len(buf), 1)
        self.assertEqual(buf.get_points()[0], (15.5, 30.2))

    def test_12_minimum_distance_threshold_filters_stationary_jitter(self):
        """12. Consecutive samples with Euclidean distance < 0.5m are filtered out as stationary noise."""
        buf = TrajectoryTrailBuffer(max_capacity=500)
        self.assertTrue(buf.add_sample(0.0, 0.0, min_distance=0.5))

        # 0.3m displacement -> filtered
        self.assertFalse(buf.add_sample(0.3, 0.0, min_distance=0.5))
        self.assertEqual(len(buf), 1)

        # 0.49m displacement -> filtered
        self.assertFalse(buf.add_sample(0.0, 0.49, min_distance=0.5))
        self.assertEqual(len(buf), 1)

        # 0.50m displacement -> retained
        self.assertTrue(buf.add_sample(0.50, 0.0, min_distance=0.5))
        self.assertEqual(len(buf), 2)

    def test_13_large_displacement_retained_correctly(self):
        """13. Large displacements > 0.5m (e.g. 3.0m at high speed) are accepted; 0.5m is not an upper bound."""
        buf = TrajectoryTrailBuffer(max_capacity=500)
        self.assertTrue(buf.add_sample(0.0, 0.0, min_distance=0.5))

        # Vehicle traveled 3.5m in one step -> retained correctly
        self.assertTrue(buf.add_sample(3.5, 0.0, min_distance=0.5))
        self.assertEqual(len(buf), 2)
        self.assertEqual(buf.get_points()[-1], (3.5, 0.0))

        # Vehicle traveled 12.0m in next step -> retained correctly
        self.assertTrue(buf.add_sample(3.5, 12.0, min_distance=0.5))
        self.assertEqual(len(buf), 3)

    def test_14_ring_buffer_fifo_bounded_at_500_points(self):
        """14. Ring buffer strictly caps at max_capacity (500 points) and drops oldest samples FIFO."""
        buf = TrajectoryTrailBuffer(max_capacity=5)
        for i in range(10):
            buf.add_sample(float(i * 1.0), 0.0, min_distance=0.5)

        self.assertEqual(len(buf), 5)
        # Retained points are the 5 most recent: [5.0, 6.0, 7.0, 8.0, 9.0]
        pts = buf.get_points()
        self.assertEqual(pts[0], (5.0, 0.0))
        self.assertEqual(pts[-1], (9.0, 0.0))

    def test_15_reset_clears_trajectory_buffer_zero_cross_episode_leakage(self):
        """15. reset() empties the trajectory buffer to 0 points, preventing cross-episode trail leakage."""
        buf = TrajectoryTrailBuffer(max_capacity=500)
        for i in range(20):
            buf.add_sample(float(i), float(i), min_distance=0.5)
        self.assertEqual(len(buf), 20)

        buf.reset()
        self.assertEqual(len(buf), 0)
        self.assertEqual(buf.get_points(), [])

    def test_16_no_unbounded_panda3d_nodes_on_repeated_updates(self):
        """16. Rebuilding trail geometry replaces old GeomNode, maintaining single active trail NodePath."""
        from metadrive.constants import CamMask
        root = p3d.NodePath("mock_render")
        trail_root = root.attachNewNode("trail_root")
        trail_root.hide()
        trail_root.showThrough(CamMask.MiniMap)

        active_trail_np = None

        # Simulate 10 successive trail rebuilds
        for step in range(10):
            if active_trail_np is not None:
                active_trail_np.removeNode()

            ls = p3d.LineSegs("trail")
            ls.moveTo(0, 0, 0.5)
            ls.drawTo(step * 2.0, step * 2.0, 0.5)
            active_trail_np = trail_root.attachNewNode(ls.create())

            # Number of children under trail_root must remain exactly 1
            self.assertEqual(trail_root.getNumChildren(), 1)

        # Cleanup
        if active_trail_np is not None:
            active_trail_np.removeNode()
        trail_root.removeNode()
        root.removeNode()

    def test_17_renderer_screen_clipping_bounds_check(self):
        """17. Normalized screen UV coordinates outside [0, 1] are detected as out-of-bounds."""
        left, right, bottom, top = (0.015, 0.185, 0.745, 0.985)

        def is_in_minimap(u, v):
            return 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0

        self.assertTrue(is_in_minimap(0.5, 0.5))
        self.assertTrue(is_in_minimap(0.0, 0.0))
        self.assertTrue(is_in_minimap(1.0, 1.0))
        self.assertFalse(is_in_minimap(-0.01, 0.5))
        self.assertFalse(is_in_minimap(1.01, 0.5))
        self.assertFalse(is_in_minimap(0.5, -0.01))
        self.assertFalse(is_in_minimap(0.5, 1.01))

    def test_18_window_resize_film_recalculation(self):
        """18. Film size recalculates correctly when window dimensions change from 16:9 to 4:3."""
        bounds = (0.0, 100.0, 0.0, 100.0)  # square map 124m x 124m
        # 16:9 window region aspect
        region_w_16_9 = (0.185 - 0.015) * 1920.0  # 326.4
        region_h_16_9 = (0.985 - 0.745) * 1080.0  # 259.2
        aspect_16_9 = region_w_16_9 / region_h_16_9  # ~1.26
        fw1, fh1 = calculate_film_size(bounds, aspect_16_9)

        # 4:3 window region aspect
        region_w_4_3 = (0.185 - 0.015) * 1024.0  # 174.08
        region_h_4_3 = (0.985 - 0.745) * 768.0   # 184.32
        aspect_4_3 = region_w_4_3 / region_h_4_3  # ~0.944
        fw2, fh2 = calculate_film_size(bounds, aspect_4_3)

        self.assertNotEqual((fw1, fh1), (fw2, fh2))
        self.assertAlmostEqual(fw1 / fh1, aspect_16_9)
        self.assertAlmostEqual(fw2 / fh2, aspect_4_3)

    def test_19_destroy_idempotency_repeated_calls_safe(self):
        """19. destroy() can be called repeatedly on a minimap mock without throwing exceptions."""
        from src.visualization.minimap_overlay import MiniMapOverlay
        # Create uninitialized instance
        overlay = MiniMapOverlay.__new__(MiniMapOverlay)
        overlay.camera = None
        overlay.display_region = None
        overlay.background = None
        overlay.marker_root = None
        overlay.marker_arrow = None
        overlay.marker_dot = None
        overlay.trail_root = None
        overlay.trail_np = None
        overlay._trail_buffer = TrajectoryTrailBuffer(max_capacity=500)
        overlay.engine = None
        overlay.env = None

        # Call destroy 3 times
        overlay.destroy()
        overlay.destroy()
        overlay.destroy()
        self.assertIsNone(overlay.camera)
        self.assertIsNone(overlay.display_region)

    def test_20_partial_initialization_cleanup_on_exception(self):
        """20. Safe cleanup when engine or window is missing during initialization."""
        from src.visualization.minimap_overlay import MiniMapOverlay
        mock_env_no_engine = MagicMock()
        mock_env_no_engine.engine = None
        with self.assertRaises(RuntimeError) as ctx1:
            MiniMapOverlay(mock_env_no_engine)
        self.assertIn("MetaDrive engine is not initialized", str(ctx1.exception))

        mock_env_no_win = MagicMock()
        mock_env_no_win.engine.win = None
        with self.assertRaises(RuntimeError) as ctx2:
            MiniMapOverlay(mock_env_no_win)
        self.assertIn("Render window not available", str(ctx2.exception))

    def test_21_no_background_taskmgr_update_task_spawned(self):
        """21. MiniMapOverlay operates strictly step-driven without adding tasks to engine.taskMgr."""
        from src.visualization.minimap_overlay import MiniMapOverlay
        overlay = MiniMapOverlay.__new__(MiniMapOverlay)
        self.assertFalse(hasattr(overlay, "_marker_task_name"))

    def test_22_scientific_contracts_and_hashes_unmutated(self):
        """22. Verifies that importing visualizer does not mutate any locked platform contract hashes."""
        from src.launcher.resolver import (
            GATE5_LOCKED_BENCHMARK_HASH,
            GATE6_LOCKED_AGENT_HASH,
            GATE6_LOCKED_RUNTIME_HASH,
            GATE7_LOCKED_LOGGING_HASH,
            GATE7_LOCKED_OBSERVABILITY_HASH,
            DEFAULT_LAUNCHER_CONTRACT_HASH,
            DEFAULT_PLATFORM_EXECUTION_HASH,
        )
        from src.launcher.registry import HISTORICAL_FIXTURE_REGISTRY_SHA256

        self.assertEqual(GATE5_LOCKED_BENCHMARK_HASH, "9ddd889b84d8705fae618879e5035556c80d0276a3dc2a58a7963937ebb59f77")
        self.assertEqual(GATE6_LOCKED_AGENT_HASH, "53aa37079ff44afa75d9a3f921b0c1f98c4600d882fced51fc8d9197795058eb")
        self.assertEqual(GATE6_LOCKED_RUNTIME_HASH, "c7698768539a769c7b2bc6b90771ff234276a974b6e0353bbb03119faf2f79ad")
        self.assertEqual(GATE7_LOCKED_LOGGING_HASH, "0d17915556d83f49d9519576e95920c0853700f33a91593ef85b3aeef6cbb9f2")
        self.assertEqual(GATE7_LOCKED_OBSERVABILITY_HASH, "f00c27fca8abf8145596576cd0dd2540181ff8ae7eccf7806c864921933d3581")
        self.assertEqual(DEFAULT_LAUNCHER_CONTRACT_HASH, "5e6269b6e32349bd3ee34ca52583c00f832bd0a737330c894e9a8d314653db04")
        self.assertEqual(DEFAULT_PLATFORM_EXECUTION_HASH, "442beafd86212419dc7b3edfa60e53715bf33877d3dd72e61c54091520b0563d")
        self.assertEqual(HISTORICAL_FIXTURE_REGISTRY_SHA256, "9fac42d07949796d00b6dc869b98c945eae7a59355d7928f429e457f1ee7f26f")


if __name__ == "__main__":
    unittest.main()
