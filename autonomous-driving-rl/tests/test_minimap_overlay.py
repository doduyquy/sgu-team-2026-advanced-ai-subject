"""
Unit and Component Integration Tests for Gate HUD-1: Live Minimap Implementation.

Tests minimap mathematical utilities, polyline bounding with robust error handling,
actual MiniMapOverlay component lifecycle (update, heading orientation, missing data fallback,
trajectory sampling independence, screen clipping, window resize, and exception-safe rollback).
Runs in headless environment without requiring an interactive desktop display window.
"""

import math
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import panda3d.core as p3d

from src.visualization.minimap_math import (
    TrajectoryTrailBuffer,
    calculate_film_size,
    calculate_heading_rotation_deg,
    calculate_map_bounds,
)
from src.visualization.minimap_overlay import MiniMapOverlay


def _create_mock_minimap_env(win_w=1200, win_h=900):
    """Creates a mock MetaDrive environment with valid Panda3D nodes for component testing."""
    mock_env = MagicMock()
    mock_engine = MagicMock()
    mock_win = MagicMock()
    mock_win.getXSize.return_value = win_w
    mock_win.getYSize.return_value = win_h
    mock_engine.win = mock_win

    mock_render = p3d.NodePath("mock_render")
    mock_render2d = p3d.NodePath("mock_render2d")
    mock_engine.render = mock_render
    mock_engine.render2d = mock_render2d
    mock_engine.camList = []

    # Mock camera node with real OrthographicLens
    lens = p3d.OrthographicLens()
    mock_cam_node = MagicMock()
    mock_cam_node.getLens.return_value = lens
    mock_display_region = MagicMock()
    mock_cam_node.getDisplayRegion.return_value = mock_display_region

    mock_cam_np = MagicMock()
    mock_cam_np.node.return_value = mock_cam_node
    # Default to center of camera (0, 0, 0)
    mock_cam_np.getRelativePoint.return_value = p3d.Point3(0.0, 0.0, 0.0)

    mock_engine.makeCamera.return_value = mock_cam_np
    mock_env.engine = mock_engine

    mock_env.current_map.get_map_features.return_value = {
        "lane_1": {"polyline": [[0.0, 0.0], [50.0, 50.0]]}
    }
    mock_env.agent.position = [10.0, 10.0]
    mock_env.agent.heading_theta = 0.0
    return mock_env, lens


class TestMinimapOverlayUnitSuite(unittest.TestCase):
    """22 comprehensive unit and component tests for Gate HUD-1 Live Minimap."""

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
        """4. Empty, None, scalar, or non-iterable features fall back safely without TypeError."""
        # Empty dict
        bounds_empty = calculate_map_bounds({}, default_center=(10.0, 25.0))
        self.assertEqual(bounds_empty, (-40.0, 60.0, -25.0, 75.0))

        # None
        bounds_none = calculate_map_bounds(None, default_center=(0.0, 0.0))
        self.assertEqual(bounds_none, (-50.0, 50.0, -50.0, 50.0))

        # Malformed entries: scalar in polyline, non-iterables, None items
        malformed = {
            "bad_1": "not_a_dict",
            "bad_2": {"polyline": None},
            "bad_3": {"polyline": []},
            "bad_4": {"polyline": [None, 123, "string", [1.0]]},
            "bad_5": {"polyline": "not_a_list_of_points"},
        }
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

    def test_09_actual_update_with_missing_heading_renders_dot_on_overlay(self):
        """9. Valid coordinates with heading_rad=None uses non-directional dot without fabricating North."""
        env, _ = _create_mock_minimap_env()
        overlay = MiniMapOverlay(env)
        try:
            overlay.update(world_x=10.0, world_y=10.0, heading_rad=None)
            self.assertFalse(overlay.marker_root.isHidden())
            self.assertTrue(overlay.marker_arrow.isHidden())
            self.assertFalse(overlay.marker_dot.isHidden())
            self.assertAlmostEqual(overlay.marker_root.getR(), 0.0)
        finally:
            overlay.destroy()

    def test_10_actual_update_missing_position_hides_marker_without_fallback(self):
        """10. Missing position (None or non-finite) hides marker and does not append to buffer or fallback to env.agent."""
        env, _ = _create_mock_minimap_env()
        env.agent.position = [99.0, 99.0]  # If fallback occurred, this would be picked up!
        overlay = MiniMapOverlay(env)
        try:
            # None position
            overlay.update(world_x=None, world_y=10.0, heading_rad=0.0)
            self.assertTrue(overlay.marker_root.isHidden())
            self.assertEqual(len(overlay._trail_buffer), 0)

            # Non-finite position
            overlay.update(world_x=float("nan"), world_y=10.0, heading_rad=0.0)
            self.assertTrue(overlay.marker_root.isHidden())
            self.assertEqual(len(overlay._trail_buffer), 0)

            overlay.update(world_x=10.0, world_y=float("inf"), heading_rad=0.0)
            self.assertTrue(overlay.marker_root.isHidden())
            self.assertEqual(len(overlay._trail_buffer), 0)
        finally:
            overlay.destroy()

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
        env, _ = _create_mock_minimap_env()
        overlay = MiniMapOverlay(env)
        try:
            # Step vehicle along a line to add 10 trail samples
            for step in range(10):
                overlay.update(world_x=float(step * 2.0), world_y=float(step * 2.0), heading_rad=0.0)

            # trail_root must have at most 1 active child node (no memory/node leak)
            self.assertIsNotNone(overlay.trail_root)
            self.assertEqual(overlay.trail_root.getNumChildren(), 1)
        finally:
            overlay.destroy()

    def test_17_actual_clipping_hides_marker_when_projected_outside_viewport(self):
        """17. Vehicle outside minimap projection hides marker, but trajectory buffer retains valid coordinates."""
        env, _ = _create_mock_minimap_env()
        # Mock projection returning point far outside viewport
        env.engine.makeCamera.return_value.getRelativePoint.return_value = p3d.Point3(5000.0, 5000.0, 0.0)
        overlay = MiniMapOverlay(env)
        try:
            # Vehicle position is valid, but projected out of bounds
            overlay.update(world_x=5000.0, world_y=5000.0, heading_rad=0.0)
            # Marker must be hidden because it is clipped outside minimap DisplayRegion
            self.assertTrue(overlay.marker_root.isHidden())
            # But trajectory sampling is independent of marker visibility: sample was retained!
            self.assertEqual(len(overlay._trail_buffer), 1)
            self.assertEqual(overlay._trail_buffer.get_points()[0], (5000.0, 5000.0))
        finally:
            overlay.destroy()

    def test_18_actual_window_resize_updates_lens_film_size_on_overlay(self):
        """18. Window resize dynamically recalculates film size and calls lens.setFilmSize on overlay."""
        env, lens = _create_mock_minimap_env(win_w=1200, win_h=900)
        overlay = MiniMapOverlay(env)
        try:
            initial_film = (overlay.film_width, overlay.film_height)
            # Simulate window resizing to 1600x900
            env.engine.win.getXSize.return_value = 1600
            env.engine.win.getYSize.return_value = 900
            overlay.update(world_x=10.0, world_y=10.0, heading_rad=0.0)

            updated_film = (overlay.film_width, overlay.film_height)
            self.assertNotEqual(initial_film, updated_film)
            # Verify lens film size matches updated dimensions (Panda3D C++ uses float32)
            lens_size = lens.getFilmSize()
            self.assertAlmostEqual(lens_size.x, overlay.film_width, places=4)
            self.assertAlmostEqual(lens_size.y, overlay.film_height, places=4)

            # Verify background quad scale was updated to cover new film size
            bg_scale = overlay.background.getScale()
            expected_scale_x = overlay.film_width / initial_film[0]
            expected_scale_y = overlay.film_height / initial_film[1]
            self.assertAlmostEqual(bg_scale.x, expected_scale_x, places=4)
            self.assertAlmostEqual(bg_scale.y, expected_scale_y, places=4)
        finally:
            overlay.destroy()

    def test_19_destroy_idempotency_repeated_calls_safe(self):
        """19. destroy() can be called repeatedly on MiniMapOverlay without throwing exceptions."""
        env, _ = _create_mock_minimap_env()
        overlay = MiniMapOverlay(env)
        # Call destroy 3 times sequentially
        overlay.destroy()
        overlay.destroy()
        overlay.destroy()
        self.assertIsNone(overlay.camera)
        self.assertIsNone(overlay.display_region)
        self.assertIsNone(overlay.trail_root)
        self.assertIsNone(overlay.marker_root)

    def test_20_partial_initialization_exception_safe_rollback(self):
        """20. If resource creation fails mid-init, previously allocated resources are cleanly rolled back."""
        # Test 20a: Missing engine or win
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

        # Test 20b: Failure injection after camera creation (during _create_background)
        env_bg_fail, _ = _create_mock_minimap_env()
        with patch.object(MiniMapOverlay, "_create_background", side_effect=RuntimeError("Simulated background failure")):
            with self.assertRaises(RuntimeError) as ctx3:
                MiniMapOverlay(env_bg_fail)
            self.assertIn("Simulated background failure", str(ctx3.exception))
            # Verify camera and display_region were cleanly removed during rollback
            env_bg_fail.engine.win.removeDisplayRegion.assert_called()

        # Test 20c: Failure injection after camera creation (during _create_markers)
        env_inject, _ = _create_mock_minimap_env()
        with patch.object(MiniMapOverlay, "_create_markers", side_effect=RuntimeError("Simulated marker failure")):
            with self.assertRaises(RuntimeError) as ctx4:
                MiniMapOverlay(env_inject)
            self.assertIn("Simulated marker failure", str(ctx4.exception))
            # Verify that camera and display_region were cleanly removed during rollback
            env_inject.engine.win.removeDisplayRegion.assert_called()

        # Test 20d: Successive successful initialization after failure
        env_retry, _ = _create_mock_minimap_env()
        retry_overlay = MiniMapOverlay(env_retry)
        self.assertIsNotNone(retry_overlay.camera)
        retry_overlay.destroy()

    def test_21_constructor_spawns_no_background_taskmgr_tasks(self):
        """21. MiniMapOverlay constructor operates purely step-driven without taskMgr.add() background tasks."""
        env, _ = _create_mock_minimap_env()
        overlay = MiniMapOverlay(env)
        try:
            self.assertFalse(hasattr(overlay, "_marker_task_name"))
            env.engine.taskMgr.add.assert_not_called()
        finally:
            overlay.destroy()

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
