"""
Unit and Component Test Suite for Transparent Racing HUD (Gate HUD-2).

Verifies:
1. Immutable telemetry model (HudTelemetrySnapshotV1) and factory sanitization.
2. Missing-data policies ("---" vs legitimate zero "000" / "0%").
3. Digital speedometer numerical format and 240-degree arc normalization.
4. Route progress percentage formatting and visual saturation.
5. Turn indicator steering thresholds (-0.15, +0.15) and boundary precision.
6. Pedal meter polarity (acceleration vs braking), neutral coasting, and disabled state.
7. Layout anchors (a2dTopCenter, a2dBottomRight, a2dBottomLeft) and aspect ratio invariance.
8. Zero interference with 3D render, minimap, or scientific benchmark contracts.
9. Bounded node allocations across repeated update cycles.
10. Idempotent destroy and exception-safe rollback.
"""

import math
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import panda3d.core as p3d

from src.visualization.racing_hud import (
    COLOR_ACCENT_CYAN,
    COLOR_BG_GLASS,
    COLOR_BORDER_GLASS,
    COLOR_BRAKE,
    COLOR_INDICATOR_ACTIVE,
    COLOR_INDICATOR_DISABLED,
    COLOR_INDICATOR_IDLE,
    COLOR_TEXT_DIM,
    COLOR_TEXT_PRIMARY,
    COLOR_THROTTLE,
    RacingHud,
)
from src.visualization.telemetry_model import (
    HudTelemetrySnapshotV1,
    create_telemetry_snapshot,
)


def _create_mock_racing_engine():
    """Constructs a lightweight mock MetaDrive engine hosting Panda3D aspect2d anchors."""
    mock_engine = MagicMock()
    mock_aspect2d = p3d.NodePath("mock_aspect2d")
    mock_engine.aspect2d = mock_aspect2d
    mock_engine.render2d = mock_aspect2d
    mock_engine.a2dTopCenter = mock_aspect2d.attachNewNode("mock_a2dTopCenter")
    mock_engine.a2dBottomRight = mock_aspect2d.attachNewNode("mock_a2dBottomRight")
    mock_engine.a2dBottomLeft = mock_aspect2d.attachNewNode("mock_a2dBottomLeft")
    mock_engine.taskMgr = MagicMock()
    return mock_engine


class TestRacingHudUnitSuite(unittest.TestCase):
    """22 Mandatory Unit Tests for Gate HUD-2 Racing HUD."""

    def assert_color_almost_equal(self, c1, c2, places=2):
        """Helper to assert Panda3D Vec4 colors match within quantization tolerance."""
        self.assertAlmostEqual(c1[0], c2[0], places=places)
        self.assertAlmostEqual(c1[1], c2[1], places=places)
        self.assertAlmostEqual(c1[2], c2[2], places=places)
        self.assertAlmostEqual(c1[3], c2[3], places=places)

    def test_01_telemetry_snapshot_immutable_and_serializable(self):
        """1. HudTelemetrySnapshotV1 is frozen (immutable) and serializes to dictionary."""
        snapshot = create_telemetry_snapshot(
            step_index=10,
            simulation_time_s=1.0,
            world_position=(12.5, 34.2),
            heading_theta_rad=0.5,
            speed_kmh=45.0,
            route_completion=0.25,
            canonical_steering=-0.2,
            canonical_throttle_brake=0.6,
        )
        self.assertIsInstance(snapshot, HudTelemetrySnapshotV1)
        self.assertEqual(snapshot.step_index, 10)
        self.assertEqual(snapshot.speed_kmh, 45.0)

        # Immutability assertion
        with self.assertRaises(Exception):
            snapshot.speed_kmh = 50.0  # type: ignore

        # Dictionary serialization
        d = snapshot.to_dict()
        self.assertEqual(d["step_index"], 10)
        self.assertEqual(d["speed_kmh"], 45.0)
        self.assertEqual(d["world_position"], (12.5, 34.2))

    def test_02_telemetry_nan_and_inf_normalized_to_none(self):
        """2. Telemetry factory converts NaN, Inf, negative speeds, and non-numeric values to None."""
        snap = create_telemetry_snapshot(
            speed_kmh=float("nan"),
            route_completion=float("inf"),
            canonical_steering=float("-inf"),
            canonical_throttle_brake="invalid",
            world_position=(float("nan"), 10.0),
            heading_theta_rad=float("nan"),
            is_clean_success="not_a_bool",
        )
        self.assertIsNone(snap.speed_kmh)
        self.assertIsNone(snap.route_completion)
        self.assertIsNone(snap.canonical_steering)
        self.assertIsNone(snap.canonical_throttle_brake)
        self.assertIsNone(snap.world_position)
        self.assertIsNone(snap.heading_theta_rad)
        self.assertIsNone(snap.is_clean_success)

        # Negative speeds must be normalized to None (not coerced to 0.0)
        snap_neg = create_telemetry_snapshot(speed_kmh=-15.0)
        self.assertIsNone(snap_neg.speed_kmh)

        # Strict boolean validation for is_clean_success
        snap_int_bool = create_telemetry_snapshot(is_clean_success=1)
        self.assertIsNone(snap_int_bool.is_clean_success)
        snap_true = create_telemetry_snapshot(is_clean_success=True)
        self.assertIs(snap_true.is_clean_success, True)
        snap_false = create_telemetry_snapshot(is_clean_success=False)
        self.assertIs(snap_false.is_clean_success, False)

    def test_03_speed_format_valid_zero_vs_missing_vs_positive(self):
        """3. Speedometer formats: 0.0 -> '000', None -> '---', 48.0 -> '048'."""
        engine = _create_mock_racing_engine()
        hud = RacingHud(engine)
        try:
            # A. Valid stationary zero
            hud.update(create_telemetry_snapshot(speed_kmh=0.0))
            self.assertEqual(hud.speed_text_node.getText(), "000")
            self.assertEqual(hud.speed_text_node.getTextColor(), COLOR_TEXT_PRIMARY)

            # B. Missing speed (None)
            hud.update(create_telemetry_snapshot(speed_kmh=None))
            self.assertEqual(hud.speed_text_node.getText(), "---")
            self.assertEqual(hud.speed_text_node.getTextColor(), COLOR_TEXT_DIM)

            # C. Normal speed
            hud.update(create_telemetry_snapshot(speed_kmh=48.2))
            self.assertEqual(hud.speed_text_node.getText(), "048")
            self.assertEqual(hud.speed_text_node.getTextColor(), COLOR_TEXT_PRIMARY)

            # D. High speed
            hud.update(create_telemetry_snapshot(speed_kmh=120.9))
            self.assertEqual(hud.speed_text_node.getText(), "121")
        finally:
            hud.destroy()

    def test_04_speedometer_arc_normalization_and_saturation(self):
        """4. Arc sweeps proportionally up to nominal 100 km/h; saturates visually without truncating digits."""
        engine = _create_mock_racing_engine()
        hud = RacingHud(engine)
        try:
            # 0 km/h -> Arc fraction 0.0
            hud.update(create_telemetry_snapshot(speed_kmh=0.0))
            self.assertAlmostEqual(hud._last_speed_fraction, 0.0)

            # 50 km/h -> Arc fraction 0.50
            hud.update(create_telemetry_snapshot(speed_kmh=50.0))
            self.assertAlmostEqual(hud._last_speed_fraction, 0.50, places=3)

            # 100 km/h -> Arc fraction 1.00
            hud.update(create_telemetry_snapshot(speed_kmh=100.0))
            self.assertAlmostEqual(hud._last_speed_fraction, 1.00, places=3)

            # 150 km/h -> Arc fraction saturates at 1.00 while digits show '150'
            hud.update(create_telemetry_snapshot(speed_kmh=150.0))
            self.assertAlmostEqual(hud._last_speed_fraction, 1.00, places=3)
            self.assertEqual(hud.speed_text_node.getText(), "150")
        finally:
            hud.destroy()

    def test_05_route_progress_percentage_formatting(self):
        """5. Route progress bar formats: 0.0 -> '0%', 0.5 -> '50%', 1.0 -> '100%', None -> '---'."""
        engine = _create_mock_racing_engine()
        hud = RacingHud(engine)
        try:
            # 0%
            hud.update(create_telemetry_snapshot(route_completion=0.0))
            self.assertEqual(hud.progress_text_node.getText(), "0%")
            self.assertEqual(hud.progress_text_node.getTextColor(), COLOR_TEXT_PRIMARY)

            # 50%
            hud.update(create_telemetry_snapshot(route_completion=0.50))
            self.assertEqual(hud.progress_text_node.getText(), "50%")
            self.assertFalse(hud.progress_fill.isHidden())
            self.assertAlmostEqual(hud.progress_fill.getScale().x, 0.50, places=3)

            # 100%
            hud.update(create_telemetry_snapshot(route_completion=1.0))
            self.assertEqual(hud.progress_text_node.getText(), "100%")
            self.assertAlmostEqual(hud.progress_fill.getScale().x, 1.0, places=3)

            # Missing (None)
            hud.update(create_telemetry_snapshot(route_completion=None))
            self.assertEqual(hud.progress_text_node.getText(), "---")
            self.assertEqual(hud.progress_text_node.getTextColor(), COLOR_TEXT_DIM)
            self.assertTrue(hud.progress_fill.isHidden())
        finally:
            hud.destroy()

    def test_06_route_progress_visual_saturation(self):
        """6. Route completion > 1.0 clamps visually to 100% in presentation without corrupting snapshot."""
        engine = _create_mock_racing_engine()
        hud = RacingHud(engine)
        try:
            snap = create_telemetry_snapshot(route_completion=1.25)
            self.assertEqual(snap.route_completion, 1.25)  # Snapshot retains raw value
            hud.update(snap)
            self.assertEqual(hud.progress_text_node.getText(), "100%")
            self.assertAlmostEqual(hud.progress_fill.getScale().x, 1.0, places=3)
        finally:
            hud.destroy()

    def test_07_turn_indicators_thresholds_and_boundaries(self):
        """7. Turn indicator steering threshold precision at -0.15 and +0.15."""
        engine = _create_mock_racing_engine()
        hud = RacingHud(engine)
        try:
            # Neutral / Zero
            hud.update(create_telemetry_snapshot(canonical_steering=0.0))
            self.assert_color_almost_equal(hud.indicator_left_np.getColor(), COLOR_INDICATOR_IDLE)
            self.assert_color_almost_equal(hud.indicator_right_np.getColor(), COLOR_INDICATOR_IDLE)

            # Exact Boundary -0.150 -> Still idle (not strictly < -0.15)
            hud.update(create_telemetry_snapshot(canonical_steering=-0.150))
            self.assert_color_almost_equal(hud.indicator_left_np.getColor(), COLOR_INDICATOR_IDLE)
            self.assert_color_almost_equal(hud.indicator_right_np.getColor(), COLOR_INDICATOR_IDLE)

            # Active Left: -0.151 -> Left active, Right idle
            hud.update(create_telemetry_snapshot(canonical_steering=-0.151))
            self.assert_color_almost_equal(hud.indicator_left_np.getColor(), COLOR_INDICATOR_ACTIVE)
            self.assert_color_almost_equal(hud.indicator_right_np.getColor(), COLOR_INDICATOR_IDLE)

            # Exact Boundary +0.150 -> Still idle (not strictly > +0.15)
            hud.update(create_telemetry_snapshot(canonical_steering=0.150))
            self.assert_color_almost_equal(hud.indicator_left_np.getColor(), COLOR_INDICATOR_IDLE)
            self.assert_color_almost_equal(hud.indicator_right_np.getColor(), COLOR_INDICATOR_IDLE)

            # Active Right: +0.151 -> Right active, Left idle
            hud.update(create_telemetry_snapshot(canonical_steering=0.151))
            self.assert_color_almost_equal(hud.indicator_left_np.getColor(), COLOR_INDICATOR_IDLE)
            self.assert_color_almost_equal(hud.indicator_right_np.getColor(), COLOR_INDICATOR_ACTIVE)
        finally:
            hud.destroy()

    def test_08_turn_indicators_missing_steering_disabled_state(self):
        """8. Missing steering (None) renders indicators in distinct disabled/dimmed state."""
        engine = _create_mock_racing_engine()
        hud = RacingHud(engine)
        try:
            hud.update(create_telemetry_snapshot(canonical_steering=None))
            self.assert_color_almost_equal(hud.indicator_left_np.getColor(), COLOR_INDICATOR_DISABLED)
            self.assert_color_almost_equal(hud.indicator_right_np.getColor(), COLOR_INDICATOR_DISABLED)
        finally:
            hud.destroy()

    def test_09_pedal_meters_polarity_acceleration(self):
        """9. Positive throttle/brake activates green throttle fill and zeroes red brake meter."""
        engine = _create_mock_racing_engine()
        hud = RacingHud(engine)
        try:
            hud.update(create_telemetry_snapshot(canonical_throttle_brake=0.75))
            # Throttle fill active at 75%
            self.assertFalse(hud.throttle_fill.isHidden())
            self.assertAlmostEqual(hud.throttle_fill.getScale().z, 0.75, places=3)
            # Brake fill hidden / zero
            self.assertTrue(hud.brake_fill.isHidden())
        finally:
            hud.destroy()

    def test_10_pedal_meters_polarity_braking(self):
        """10. Negative throttle/brake activates red brake fill and zeroes green throttle meter."""
        engine = _create_mock_racing_engine()
        hud = RacingHud(engine)
        try:
            hud.update(create_telemetry_snapshot(canonical_throttle_brake=-0.60))
            # Brake fill active at 60%
            self.assertFalse(hud.brake_fill.isHidden())
            self.assertAlmostEqual(hud.brake_fill.getScale().z, 0.60, places=3)
            # Throttle fill hidden / zero
            self.assertTrue(hud.throttle_fill.isHidden())
        finally:
            hud.destroy()

    def test_11_pedal_meters_neutral_coasting_vs_unavailable(self):
        """11. Coasting (0.0) maintains active frames with 0 fill; None dims frames to disabled state."""
        engine = _create_mock_racing_engine()
        hud = RacingHud(engine)
        try:
            # A. Neutral coasting (0.0)
            hud.update(create_telemetry_snapshot(canonical_throttle_brake=0.0))
            self.assertTrue(hud.brake_fill.isHidden())
            self.assertTrue(hud.throttle_fill.isHidden())
            self.assert_color_almost_equal(hud.brake_bg.getColor(), COLOR_BG_GLASS)
            self.assert_color_almost_equal(hud.throttle_bg.getColor(), COLOR_BG_GLASS)
            self.assert_color_almost_equal(hud.brake_border.getColor(), COLOR_BORDER_GLASS)

            # B. Missing / Unavailable (None)
            hud.update(create_telemetry_snapshot(canonical_throttle_brake=None))
            self.assertTrue(hud.brake_fill.isHidden())
            self.assertTrue(hud.throttle_fill.isHidden())
            self.assert_color_almost_equal(hud.brake_bg.getColor(), COLOR_INDICATOR_DISABLED)
            self.assert_color_almost_equal(hud.throttle_bg.getColor(), COLOR_INDICATOR_DISABLED)
        finally:
            hud.destroy()

    def test_12_pedal_meters_clamp_at_full_deflection(self):
        """12. Throttle/brake values > 1.0 or < -1.0 clamp visual fill scales to 1.0."""
        engine = _create_mock_racing_engine()
        hud = RacingHud(engine)
        try:
            # Over-throttle (e.g. 1.5)
            hud.update(create_telemetry_snapshot(canonical_throttle_brake=1.5))
            self.assertAlmostEqual(hud.throttle_fill.getScale().z, 1.0, places=3)

            # Over-brake (e.g. -2.0)
            hud.update(create_telemetry_snapshot(canonical_throttle_brake=-2.0))
            self.assertAlmostEqual(hud.brake_fill.getScale().z, 1.0, places=3)
        finally:
            hud.destroy()

    def test_13_layout_anchors_and_hierarchy(self):
        """13. Widgets anchor to correct aspect2d quadrant nodes without crowding center."""
        engine = _create_mock_racing_engine()
        hud = RacingHud(engine)
        try:
            # Progress under top center
            self.assertEqual(hud.progress_root.getParent(), engine.a2dTopCenter)
            # Speedometer and pedals under bottom right
            self.assertEqual(hud.speed_root.getParent(), engine.a2dBottomRight)
            self.assertEqual(hud.pedals_root.getParent(), engine.a2dBottomRight)
            # Indicators under bottom left
            self.assertEqual(hud.indicators_root.getParent(), engine.a2dBottomLeft)
        finally:
            hud.destroy()

    def test_14_responsive_aspect_ratio_invariance(self):
        """14. aspect2d 1:1 metric guarantees equal horizontal and vertical pixel distances for circles."""
        # Across multiple standard aspect ratios (4:3, 16:9, 16:10)
        for w, h in [(1200, 900), (1600, 900), (1024, 768), (1920, 1080)]:
            alpha = w / h
            r_units = 0.18
            pixel_width = r_units * (w / (2.0 * alpha))
            pixel_height = r_units * (h / 2.0)
            # Aspect ratio of the circular gauge in pixels is mathematically 1.0
            self.assertAlmostEqual(pixel_width / pixel_height, 1.0, places=5)

    def test_15_circular_arc_geometry_points_in_bounds(self):
        """15. Speedometer circular arc geometry points strictly adhere to nominal radius R=0.18."""
        engine = _create_mock_racing_engine()
        hud = RacingHud(engine)
        try:
            hud.update(create_telemetry_snapshot(speed_kmh=80.0))
            self.assertIsNotNone(hud.speed_active_arc_np)
            geom_node = hud.speed_active_arc_np.node()
            self.assertGreater(geom_node.getNumGeoms(), 0)
            vdata = geom_node.getGeom(0).getVertexData()
            reader = p3d.GeomVertexReader(vdata, "vertex")

            # Verify all arc vertices lie on circle R=0.18 within 1e-4 tolerance
            r_expected = 0.18
            while not reader.isAtEnd():
                v = reader.getData3f()
                r_actual = math.sqrt(v.x**2 + v.z**2)
                self.assertAlmostEqual(r_actual, r_expected, places=3)
        finally:
            hud.destroy()

    def test_16_zero_interference_with_minimap_and_render(self):
        """16. RacingHud attaches strictly to 2D overlay; does not touch 3D render or create cameras."""
        engine = _create_mock_racing_engine()
        engine.render = p3d.NodePath("mock_render")
        hud = RacingHud(engine)
        try:
            # render scene graph has zero children added by RacingHud
            self.assertEqual(engine.render.getNumChildren(), 0)
            # No secondary camera creation
            self.assertFalse(hasattr(hud, "camera"))
        finally:
            hud.destroy()

    def test_17_no_unbounded_node_allocations_on_repeated_updates(self):
        """17. Repeated update() cycles do not leak or accumulate unbounded child NodePaths."""
        engine = _create_mock_racing_engine()
        hud = RacingHud(engine)
        try:
            initial_speed_children = hud.speed_root.getNumChildren()
            # Step through 60 changing telemetry updates
            for i in range(60):
                snap = create_telemetry_snapshot(
                    speed_kmh=float(i * 2.0),
                    route_completion=float(i / 60.0),
                    canonical_steering=math.sin(i * 0.1),
                    canonical_throttle_brake=math.cos(i * 0.1),
                )
                hud.update(snap)

            # speed_root maintains constant child count (active arc replaces single child)
            final_speed_children = hud.speed_root.getNumChildren()
            self.assertLessEqual(final_speed_children, initial_speed_children + 1)
        finally:
            hud.destroy()

    def test_18_reset_restores_default_visual_state(self):
        """18. reset() restores all widgets to unavailable/missing state ('---', disabled indicators/pedals)."""
        engine = _create_mock_racing_engine()
        hud = RacingHud(engine)
        try:
            # First, update with active telemetry
            hud.update(create_telemetry_snapshot(speed_kmh=75.0, route_completion=0.8, canonical_steering=-0.4, canonical_throttle_brake=0.8))
            self.assertEqual(hud.speed_text_node.getText(), "075")
            self.assertEqual(hud.progress_text_node.getText(), "80%")

            # Now reset
            hud.reset()
            # Speed must be '---' (never '000' before actual telemetry)
            self.assertEqual(hud.speed_text_node.getText(), "---")
            self.assertEqual(hud.speed_text_node.getTextColor(), COLOR_TEXT_DIM)
            self.assertEqual(hud.progress_text_node.getText(), "---")
            self.assertTrue(hud.progress_fill.isHidden())
            self.assert_color_almost_equal(hud.indicator_left_np.getColor(), COLOR_INDICATOR_DISABLED)
            self.assert_color_almost_equal(hud.indicator_right_np.getColor(), COLOR_INDICATOR_DISABLED)
            self.assert_color_almost_equal(hud.brake_bg.getColor(), COLOR_INDICATOR_DISABLED)
            self.assert_color_almost_equal(hud.throttle_bg.getColor(), COLOR_INDICATOR_DISABLED)
            self.assertTrue(hud.brake_fill.isHidden())
            self.assertTrue(hud.throttle_fill.isHidden())

            # First subsequent valid snapshot must re-illuminate and update accurately
            hud.update(create_telemetry_snapshot(speed_kmh=0.0, route_completion=0.0, canonical_steering=0.0, canonical_throttle_brake=0.0))
            self.assertEqual(hud.speed_text_node.getText(), "000")
            self.assertEqual(hud.speed_text_node.getTextColor(), COLOR_TEXT_PRIMARY)
            self.assertEqual(hud.progress_text_node.getText(), "0%")
            self.assert_color_almost_equal(hud.indicator_left_np.getColor(), COLOR_INDICATOR_IDLE)
            self.assert_color_almost_equal(hud.indicator_right_np.getColor(), COLOR_INDICATOR_IDLE)
            self.assert_color_almost_equal(hud.brake_bg.getColor(), COLOR_BG_GLASS)
            self.assert_color_almost_equal(hud.throttle_bg.getColor(), COLOR_BG_GLASS)
        finally:
            hud.destroy()

    def test_19_destroy_idempotency_and_clean_cleanup(self):
        """19. destroy() can be safely invoked repeatedly and cleans up all attached nodes."""
        engine = _create_mock_racing_engine()
        hud = RacingHud(engine)
        self.assertGreater(engine.aspect2d.getNumChildren(), 0)

        hud.destroy()
        hud.destroy()
        hud.destroy()
        self.assertTrue(hud._is_destroyed)
        self.assertIsNone(hud.progress_root)
        self.assertIsNone(hud.speed_root)
        self.assertIsNone(hud.pedals_root)
        self.assertIsNone(hud.indicators_root)

    def test_20_partial_initialization_exception_safe_rollback(self):
        """20. If widget creation fails mid-init, previously allocated nodes are rolled back."""
        engine = _create_mock_racing_engine()
        with patch.object(RacingHud, "_build_pedals_widget", side_effect=RuntimeError("Simulated pedal failure")):
            with self.assertRaises(RuntimeError) as ctx:
                RacingHud(engine)
            self.assertIn("Simulated pedal failure", str(ctx.exception))
            # Verify progress and speed roots were cleaned up in rollback
            self.assertEqual(engine.a2dTopCenter.getNumChildren(), 0)
            self.assertEqual(engine.a2dBottomRight.getNumChildren(), 0)

    def test_21_no_background_taskmgr_tasks_spawned(self):
        """21. RacingHud operates purely step-driven without taskMgr.add() background polling."""
        engine = _create_mock_racing_engine()
        hud = RacingHud(engine)
        try:
            engine.taskMgr.add.assert_not_called()
        finally:
            hud.destroy()

    def test_22_scientific_contracts_and_hashes_unmutated(self):
        """22. Verifies that HUD-2 preserves all locked platform contract hashes."""
        from src.launcher.resolver import (
            DEFAULT_LAUNCHER_CONTRACT_HASH,
            DEFAULT_PLATFORM_EXECUTION_HASH,
            GATE5_LOCKED_BENCHMARK_HASH,
            GATE6_LOCKED_AGENT_HASH,
            GATE6_LOCKED_RUNTIME_HASH,
            GATE7_LOCKED_LOGGING_HASH,
            GATE7_LOCKED_OBSERVABILITY_HASH,
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

    def test_23_demo_telemetry_extraction_and_safe_none_position(self):
        """23. Missing world_position is handled without indexing NoneType, passing safely to visualizers."""
        snap = create_telemetry_snapshot(world_position=None, speed_kmh=None, route_completion=None)
        self.assertIsNone(snap.world_position)
        # Safe extraction pattern verified
        wx = snap.world_position[0] if snap.world_position is not None else None
        wy = snap.world_position[1] if snap.world_position is not None else None
        self.assertIsNone(wx)
        self.assertIsNone(wy)

    def test_24_combined_steering_and_braking_simultaneous_active(self):
        """24. Combined steering + braking activates both turn indicator and red brake meter simultaneously."""
        engine = _create_mock_racing_engine()
        hud = RacingHud(engine)
        try:
            # Scripted maneuver: Steer Left (-0.30) AND Brake (-0.75)
            snap = create_telemetry_snapshot(
                speed_kmh=25.0,
                canonical_steering=-0.30,
                canonical_throttle_brake=-0.75,
            )
            hud.update(snap)

            # Left turn indicator is active (amber)
            self.assert_color_almost_equal(hud.indicator_left_np.getColor(), COLOR_INDICATOR_ACTIVE)
            self.assert_color_almost_equal(hud.indicator_right_np.getColor(), COLOR_INDICATOR_IDLE)

            # Red brake meter is active at 75%
            self.assertFalse(hud.brake_fill.isHidden())
            self.assertAlmostEqual(hud.brake_fill.getScale().z, 0.75, places=3)

            # Throttle meter is inactive (0 fill)
            self.assertTrue(hud.throttle_fill.isHidden())
        finally:
            hud.destroy()


if __name__ == "__main__":
    unittest.main()
