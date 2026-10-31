"""
Transparent Racing HUD Implementation for MetaDrive NATIVE Window (Gate HUD-2).

Renders the six 2D situational awareness widgets on aspect2d:
1. Route Progress Bar (top-center, width ~35% of screen, slender cyan fill).
2. Large Digital Speedometer (bottom-right, bold digits + 240-degree circular arc).
3. Left Turn Indicator (bottom-left, amber chevron active when steering < -0.15).
4. Right Turn Indicator (bottom-left, amber chevron active when steering > +0.15).
5. Brake Meter (bottom-right, left of speedometer, vertical red fill).
6. Throttle Meter (bottom-right, left of speedometer, vertical green fill).

Adheres strictly to the Minimal Transparent Racing HUD design specification:
- Translucent dark glass backings (#0B131B @ 0.75 alpha).
- Thin subtle frame borders (#E0F0FF @ 0.35 alpha).
- Electric cyan accents (#00E5FF).
- Distinct missing telemetry ("---") vs legitimate zero formatting.
- Complete scientific isolation: read-only consumption of HudTelemetrySnapshotV1.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import panda3d.core as p3d

from src.visualization.telemetry_model import HudTelemetrySnapshotV1

# ==============================================================================
# Locked Visual Color Tokens (HUD-0 & HUD-2 Specification)
# ==============================================================================
COLOR_BG_GLASS = p3d.Vec4(0.043, 0.075, 0.106, 0.75)          # #0B131B @ 0.75
COLOR_BORDER_GLASS = p3d.Vec4(0.878, 0.941, 1.000, 0.35)      # #E0F0FF @ 0.35
COLOR_ACCENT_CYAN = p3d.Vec4(0.000, 0.898, 1.000, 1.00)       # #00E5FF
COLOR_THROTTLE = p3d.Vec4(0.000, 0.902, 0.463, 0.85)          # #00E676 @ 0.85
COLOR_BRAKE = p3d.Vec4(1.000, 0.090, 0.267, 0.85)             # #FF1744 @ 0.85
COLOR_INDICATOR_ACTIVE = p3d.Vec4(1.000, 0.839, 0.000, 1.00)  # #FFD600
COLOR_INDICATOR_IDLE = p3d.Vec4(0.216, 0.278, 0.310, 0.40)    # #37474F @ 0.40
COLOR_INDICATOR_DISABLED = p3d.Vec4(0.118, 0.149, 0.169, 0.20) # #1E262B @ 0.20
COLOR_TEXT_PRIMARY = p3d.Vec4(1.000, 1.000, 1.000, 0.95)      # #FFFFFF @ 0.95
COLOR_TEXT_DIM = p3d.Vec4(0.600, 0.650, 0.700, 0.60)          # Dim text for "---"
COLOR_FRAME_DISABLED = p3d.Vec4(0.150, 0.180, 0.200, 0.30)    # Disabled border


class RacingHud:
    """
    Procedural 2D transparent racing HUD attached to Panda3D aspect2d.
    Operates in decoupled step-driven mode via update(snapshot).
    """

    def __init__(self, engine: Any) -> None:
        """
        Initializes the six racing HUD widgets under Panda3D aspect2d anchors.
        Requires engine to be initialized with an active render window.
        """
        if engine is None:
            raise RuntimeError("Engine is not initialized; RacingHud requires active engine.")

        self.engine = engine
        self._is_destroyed = False

        # 1. Resolve aspect2d and quadrant anchors
        if hasattr(engine, "aspect2d") and engine.aspect2d is not None:
            self.aspect2d = engine.aspect2d
        elif hasattr(engine, "render2d") and engine.render2d is not None:
            self.aspect2d = engine.render2d
        else:
            raise RuntimeError("Panda3D aspect2d or render2d not available on engine.")

        self.a2d_top_center = (
            getattr(engine, "a2dTopCenter", None)
            or getattr(self.aspect2d, "find", lambda n: None)("**/a2dTopCenter")
            or self.aspect2d
        )
        self.a2d_bottom_right = (
            getattr(engine, "a2dBottomRight", None)
            or getattr(self.aspect2d, "find", lambda n: None)("**/a2dBottomRight")
            or self.aspect2d
        )
        self.a2d_bottom_left = (
            getattr(engine, "a2dBottomLeft", None)
            or getattr(self.aspect2d, "find", lambda n: None)("**/a2dBottomLeft")
            or self.aspect2d
        )

        # 2. Node references for clean teardown and exception safety
        self.progress_root: Optional[p3d.NodePath] = None
        self.speed_root: Optional[p3d.NodePath] = None
        self.pedals_root: Optional[p3d.NodePath] = None
        self.indicators_root: Optional[p3d.NodePath] = None

        # Widget element references
        self.progress_fill: Optional[p3d.NodePath] = None
        self.progress_text_node: Optional[p3d.TextNode] = None

        self.speed_text_node: Optional[p3d.TextNode] = None
        self.speed_unit_node: Optional[p3d.TextNode] = None
        self.speed_active_arc_np: Optional[p3d.NodePath] = None
        self._last_speed_fraction: Optional[float] = None

        self.indicator_left_np: Optional[p3d.NodePath] = None
        self.indicator_right_np: Optional[p3d.NodePath] = None

        self.brake_bg: Optional[p3d.NodePath] = None
        self.brake_border: Optional[p3d.NodePath] = None
        self.brake_fill: Optional[p3d.NodePath] = None
        self.throttle_bg: Optional[p3d.NodePath] = None
        self.throttle_border: Optional[p3d.NodePath] = None
        self.throttle_fill: Optional[p3d.NodePath] = None

        # 3. Construct widgets in an exception-safe rollback block
        try:
            self._build_progress_widget()
            self._build_speedometer_widget()
            self._build_pedals_widget()
            self._build_indicators_widget()
        except Exception:
            self.destroy()
            raise

    # ==========================================================================
    # Widget Construction
    # ==========================================================================

    def _configure_node_rendering(self, np: p3d.NodePath) -> None:
        """Applies fixed 2D overlay render states to prevent 3D lighting or depth artifacts."""
        np.setLightOff()
        np.setShaderOff(1)
        np.setFogOff(1)
        np.setTextureOff(1)
        np.setDepthTest(False)
        np.setDepthWrite(False)
        np.setTwoSided(True)
        np.setBin("fixed", 230)

    def _build_progress_widget(self) -> None:
        """Constructs the slender top-center route progress bar (width ~35% of 16:9)."""
        self.progress_root = self.a2d_top_center.attachNewNode("hud_progress_root")
        self.progress_root.setPos(0.0, 0.0, -0.07)
        self._configure_node_rendering(self.progress_root)

        w = 1.244  # 35% of 16:9 width
        h = 0.024
        half_w = w / 2.0
        half_h = h / 2.0

        # Background card
        cm = p3d.CardMaker("progress_bg")
        cm.setFrame(-half_w, half_w, -half_h, half_h)
        bg = self.progress_root.attachNewNode(cm.generate())
        bg.setColor(COLOR_BG_GLASS)
        bg.setTransparency(p3d.TransparencyAttrib.MAlpha)

        # Subtle border frame
        ls = p3d.LineSegs("progress_border")
        ls.setThickness(1.5)
        ls.setColor(COLOR_BORDER_GLASS)
        ls.moveTo(-half_w, 0.0, -half_h)
        ls.drawTo(half_w, 0.0, -half_h)
        ls.drawTo(half_w, 0.0, half_h)
        ls.drawTo(-half_w, 0.0, half_h)
        ls.drawTo(-half_w, 0.0, -half_h)
        self.progress_root.attachNewNode(ls.create())

        # Active cyan fill bar (spans [0, w] along +X, placed at -half_w)
        cm_fill = p3d.CardMaker("progress_fill")
        cm_fill.setFrame(0.0, w, -half_h + 0.003, half_h - 0.003)
        self.progress_fill = self.progress_root.attachNewNode(cm_fill.generate())
        self.progress_fill.setPos(-half_w, 0.0, 0.0)
        self.progress_fill.setColor(COLOR_ACCENT_CYAN)
        self.progress_fill.setScale(0.001, 1.0, 1.0)
        self.progress_fill.hide()

        # Percentage text readout
        self.progress_text_node = p3d.TextNode("progress_text")
        self.progress_text_node.setText("---")
        self.progress_text_node.setTextColor(COLOR_TEXT_DIM)
        self.progress_text_node.setAlign(p3d.TextNode.ACenter)
        txt_np = self.progress_root.attachNewNode(self.progress_text_node)
        txt_np.setPos(0.0, 0.0, -0.040)
        txt_np.setScale(0.032)

    def _build_speedometer_widget(self) -> None:
        """Constructs the bottom-right digital speedometer and 240-degree circular arc."""
        self.speed_root = self.a2d_bottom_right.attachNewNode("hud_speed_root")
        self.speed_root.setPos(-0.35, 0.0, 0.30)
        self._configure_node_rendering(self.speed_root)

        r = 0.18

        # 1. Background Arc (240 degrees from 210 deg down to -30 deg)
        ls_bg = p3d.LineSegs("speed_bg_arc")
        ls_bg.setThickness(2.0)
        ls_bg.setColor(COLOR_BORDER_GLASS)
        segments = 36
        for idx in range(segments + 1):
            ang = math.radians(210.0 - idx * (240.0 / segments))
            x = r * math.cos(ang)
            z = r * math.sin(ang)
            if idx == 0:
                ls_bg.moveTo(x, 0.0, z)
            else:
                ls_bg.drawTo(x, 0.0, z)
        self.speed_root.attachNewNode(ls_bg.create())

        # 2. Dynamic active arc NodePath placeholder
        self.speed_active_arc_np = None
        self._last_speed_fraction = None

        # 3. Large digital speed readout
        self.speed_text_node = p3d.TextNode("speed_digits")
        self.speed_text_node.setText("---")
        self.speed_text_node.setTextColor(COLOR_TEXT_DIM)
        self.speed_text_node.setAlign(p3d.TextNode.ACenter)
        txt_np = self.speed_root.attachNewNode(self.speed_text_node)
        txt_np.setPos(0.0, 0.0, -0.025)
        txt_np.setScale(0.075)

        # 4. Unit label "km/h"
        self.speed_unit_node = p3d.TextNode("speed_unit")
        self.speed_unit_node.setText("km/h")
        self.speed_unit_node.setTextColor(COLOR_TEXT_DIM)
        self.speed_unit_node.setAlign(p3d.TextNode.ACenter)
        unit_np = self.speed_root.attachNewNode(self.speed_unit_node)
        unit_np.setPos(0.0, 0.0, -0.088)
        unit_np.setScale(0.026)

    def _update_speedometer_arc(self, fraction: float) -> None:
        """Updates the active cyan arc geometry corresponding to normalized speed [0.0, 1.0]."""
        if self._last_speed_fraction is not None and abs(fraction - self._last_speed_fraction) < 1e-4:
            return  # No change

        if self.speed_active_arc_np is not None:
            self.speed_active_arc_np.removeNode()
            self.speed_active_arc_np = None

        if fraction <= 0.001:
            self._last_speed_fraction = fraction
            return

        r = 0.18
        ls = p3d.LineSegs("speed_active_arc")
        ls.setThickness(3.5)
        ls.setColor(COLOR_ACCENT_CYAN)

        total_sweep = 240.0 * fraction
        num_pts = max(2, int(total_sweep / 6.0))
        for idx in range(num_pts + 1):
            ang = math.radians(210.0 - idx * (total_sweep / num_pts))
            x = r * math.cos(ang)
            z = r * math.sin(ang)
            if idx == 0:
                ls.moveTo(x, 0.0, z)
            else:
                ls.drawTo(x, 0.0, z)

        self.speed_active_arc_np = self.speed_root.attachNewNode(ls.create())
        self._last_speed_fraction = fraction

    def _build_pedals_widget(self) -> None:
        """Constructs the vertical Brake and Throttle meters left of the speedometer."""
        self.pedals_root = self.a2d_bottom_right.attachNewNode("hud_pedals_root")
        self.pedals_root.setPos(-0.62, 0.0, 0.30)
        self._configure_node_rendering(self.pedals_root)

        w = 0.032
        h = 0.22
        half_w = w / 2.0
        half_h = h / 2.0

        # --- A. Brake Meter (Left, X = -0.025) ---
        bx = -0.025
        cm_b_bg = p3d.CardMaker("brake_bg")
        cm_b_bg.setFrame(-half_w, half_w, -half_h, half_h)
        self.brake_bg = self.pedals_root.attachNewNode(cm_b_bg.generate())
        self.brake_bg.setPos(bx, 0.0, 0.0)
        self.brake_bg.setColor(COLOR_BG_GLASS)
        self.brake_bg.setTransparency(p3d.TransparencyAttrib.MAlpha)

        ls_b = p3d.LineSegs("brake_border")
        ls_b.setThickness(1.5)
        ls_b.setColor(COLOR_BORDER_GLASS)
        ls_b.moveTo(bx - half_w, 0.0, -half_h)
        ls_b.drawTo(bx + half_w, 0.0, -half_h)
        ls_b.drawTo(bx + half_w, 0.0, half_h)
        ls_b.drawTo(bx - half_w, 0.0, half_h)
        ls_b.drawTo(bx - half_w, 0.0, -half_h)
        self.brake_border = self.pedals_root.attachNewNode(ls_b.create())

        cm_b_fill = p3d.CardMaker("brake_fill")
        cm_b_fill.setFrame(-half_w + 0.003, half_w - 0.003, 0.0, h - 0.006)
        self.brake_fill = self.pedals_root.attachNewNode(cm_b_fill.generate())
        self.brake_fill.setPos(bx, 0.0, -half_h + 0.003)
        self.brake_fill.setColor(COLOR_BRAKE)
        self.brake_fill.setScale(1.0, 1.0, 0.001)
        self.brake_fill.hide()

        b_lbl = p3d.TextNode("brake_label")
        b_lbl.setText("B")
        b_lbl.setTextColor(COLOR_TEXT_DIM)
        b_lbl.setAlign(p3d.TextNode.ACenter)
        b_lbl_np = self.pedals_root.attachNewNode(b_lbl)
        b_lbl_np.setPos(bx, 0.0, -half_h - 0.035)
        b_lbl_np.setScale(0.024)

        # --- B. Throttle Meter (Right, X = +0.025) ---
        tx = 0.025
        cm_t_bg = p3d.CardMaker("throttle_bg")
        cm_t_bg.setFrame(-half_w, half_w, -half_h, half_h)
        self.throttle_bg = self.pedals_root.attachNewNode(cm_t_bg.generate())
        self.throttle_bg.setPos(tx, 0.0, 0.0)
        self.throttle_bg.setColor(COLOR_BG_GLASS)
        self.throttle_bg.setTransparency(p3d.TransparencyAttrib.MAlpha)

        ls_t = p3d.LineSegs("throttle_border")
        ls_t.setThickness(1.5)
        ls_t.setColor(COLOR_BORDER_GLASS)
        ls_t.moveTo(tx - half_w, 0.0, -half_h)
        ls_t.drawTo(tx + half_w, 0.0, -half_h)
        ls_t.drawTo(tx + half_w, 0.0, half_h)
        ls_t.drawTo(tx - half_w, 0.0, half_h)
        ls_t.drawTo(tx - half_w, 0.0, -half_h)
        self.throttle_border = self.pedals_root.attachNewNode(ls_t.create())

        cm_t_fill = p3d.CardMaker("throttle_fill")
        cm_t_fill.setFrame(-half_w + 0.003, half_w - 0.003, 0.0, h - 0.006)
        self.throttle_fill = self.pedals_root.attachNewNode(cm_t_fill.generate())
        self.throttle_fill.setPos(tx, 0.0, -half_h + 0.003)
        self.throttle_fill.setColor(COLOR_THROTTLE)
        self.throttle_fill.setScale(1.0, 1.0, 0.001)
        self.throttle_fill.hide()

        t_lbl = p3d.TextNode("throttle_label")
        t_lbl.setText("T")
        t_lbl.setTextColor(COLOR_TEXT_DIM)
        t_lbl.setAlign(p3d.TextNode.ACenter)
        t_lbl_np = self.pedals_root.attachNewNode(t_lbl)
        t_lbl_np.setPos(tx, 0.0, -half_h - 0.035)
        t_lbl_np.setScale(0.024)

    def _build_indicators_widget(self) -> None:
        """Constructs the bottom-left Left and Right turn indicator chevrons."""
        self.indicators_root = self.a2d_bottom_left.attachNewNode("hud_indicators_root")
        self.indicators_root.setPos(0.22, 0.0, 0.20)
        self._configure_node_rendering(self.indicators_root)

        # Left Chevron (<) at X = -0.04
        ls_left = p3d.LineSegs("indicator_left")
        ls_left.setThickness(3.0)
        ls_left.setColor(COLOR_INDICATOR_IDLE)
        ls_left.moveTo(+0.018, 0.0, +0.025)
        ls_left.drawTo(-0.018, 0.0, 0.0)
        ls_left.drawTo(+0.018, 0.0, -0.025)
        self.indicator_left_np = self.indicators_root.attachNewNode(ls_left.create())
        self.indicator_left_np.setPos(-0.04, 0.0, 0.0)

        # Right Chevron (>) at X = +0.04
        ls_right = p3d.LineSegs("indicator_right")
        ls_right.setThickness(3.0)
        ls_right.setColor(COLOR_INDICATOR_IDLE)
        ls_right.moveTo(-0.018, 0.0, +0.025)
        ls_right.drawTo(+0.018, 0.0, 0.0)
        ls_right.drawTo(-0.018, 0.0, -0.025)
        self.indicator_right_np = self.indicators_root.attachNewNode(ls_right.create())
        self.indicator_right_np.setPos(+0.04, 0.0, 0.0)

    # ==========================================================================
    # Step-Driven Update Interface
    # ==========================================================================

    def update(self, snapshot: HudTelemetrySnapshotV1) -> None:
        """
        Updates all six racing HUD widgets strictly from an immutable telemetry snapshot.
        Enforces read-only data consumption without simulator side-effects.
        """
        if self._is_destroyed:
            return

        # ----------------------------------------------------------------------
        # 1. Route Progress Bar
        # ----------------------------------------------------------------------
        if snapshot.route_completion is not None:
            pct = max(0.0, min(1.0, float(snapshot.route_completion)))
            if self.progress_fill is not None:
                if pct > 0.001:
                    self.progress_fill.show()
                    self.progress_fill.setScale(pct, 1.0, 1.0)
                else:
                    self.progress_fill.hide()
            if self.progress_text_node is not None:
                self.progress_text_node.setText(f"{int(round(pct * 100))}%")
                self.progress_text_node.setTextColor(COLOR_TEXT_PRIMARY)
        else:
            if self.progress_fill is not None:
                self.progress_fill.hide()
            if self.progress_text_node is not None:
                self.progress_text_node.setText("---")
                self.progress_text_node.setTextColor(COLOR_TEXT_DIM)

        # ----------------------------------------------------------------------
        # 2. Digital Speedometer & Arc
        # ----------------------------------------------------------------------
        if snapshot.speed_kmh is not None:
            spd = max(0.0, float(snapshot.speed_kmh))
            if self.speed_text_node is not None:
                self.speed_text_node.setText(f"{int(round(spd)):03d}")
                self.speed_text_node.setTextColor(COLOR_TEXT_PRIMARY)
            # Arc fills proportionally up to nominal 100 km/h (saturates visually at 100)
            arc_frac = max(0.0, min(1.0, spd / 100.0))
            self._update_speedometer_arc(arc_frac)
        else:
            if self.speed_text_node is not None:
                self.speed_text_node.setText("---")
                self.speed_text_node.setTextColor(COLOR_TEXT_DIM)
            self._update_speedometer_arc(0.0)

        # ----------------------------------------------------------------------
        # 3. Turn Indicators (Steering Thresholds -0.15 / +0.15)
        # ----------------------------------------------------------------------
        st = snapshot.canonical_steering
        if st is not None:
            left_active = bool(st < -0.15)
            right_active = bool(st > 0.15)
            if self.indicator_left_np is not None:
                self.indicator_left_np.setColor(COLOR_INDICATOR_ACTIVE if left_active else COLOR_INDICATOR_IDLE)
            if self.indicator_right_np is not None:
                self.indicator_right_np.setColor(COLOR_INDICATOR_ACTIVE if right_active else COLOR_INDICATOR_IDLE)
        else:
            if self.indicator_left_np is not None:
                self.indicator_left_np.setColor(COLOR_INDICATOR_DISABLED)
            if self.indicator_right_np is not None:
                self.indicator_right_np.setColor(COLOR_INDICATOR_DISABLED)

        # ----------------------------------------------------------------------
        # 4. Pedal Meters (Brake & Throttle Polarity)
        # ----------------------------------------------------------------------
        tb = snapshot.canonical_throttle_brake
        if tb is not None:
            val = float(tb)
            thr_frac = max(0.0, min(1.0, val)) if val > 0.0 else 0.0
            brk_frac = max(0.0, min(1.0, -val)) if val < 0.0 else 0.0

            # Frame illuminated in active mode
            if self.brake_bg is not None:
                self.brake_bg.setColor(COLOR_BG_GLASS)
            if self.throttle_bg is not None:
                self.throttle_bg.setColor(COLOR_BG_GLASS)
            if self.brake_border is not None:
                self.brake_border.setColor(COLOR_BORDER_GLASS)
            if self.throttle_border is not None:
                self.throttle_border.setColor(COLOR_BORDER_GLASS)

            if self.brake_fill is not None:
                if brk_frac > 0.001:
                    self.brake_fill.show()
                    self.brake_fill.setScale(1.0, 1.0, brk_frac)
                else:
                    self.brake_fill.hide()

            if self.throttle_fill is not None:
                if thr_frac > 0.001:
                    self.throttle_fill.show()
                    self.throttle_fill.setScale(1.0, 1.0, thr_frac)
                else:
                    self.throttle_fill.hide()
        else:
            # Disabled / unavailable telemetry
            if self.brake_bg is not None:
                self.brake_bg.setColor(COLOR_INDICATOR_DISABLED)
            if self.throttle_bg is not None:
                self.throttle_bg.setColor(COLOR_INDICATOR_DISABLED)
            if self.brake_border is not None:
                self.brake_border.setColor(COLOR_FRAME_DISABLED)
            if self.throttle_border is not None:
                self.throttle_border.setColor(COLOR_FRAME_DISABLED)
            if self.brake_fill is not None:
                self.brake_fill.hide()
            if self.throttle_fill is not None:
                self.throttle_fill.hide()

    def reset(self) -> None:
        """
        Resets all widgets to their default unavailable/missing-data resting state.
        Does not fabricate zero values ('000') prior to receiving actual telemetry.
        """
        if self._is_destroyed:
            return

        # 1. Route progress: reset to missing
        if self.progress_fill is not None:
            self.progress_fill.hide()
        if self.progress_text_node is not None:
            self.progress_text_node.setText("---")
            self.progress_text_node.setTextColor(COLOR_TEXT_DIM)

        # 2. Speedometer: reset to missing ('---'), empty arc
        if self.speed_text_node is not None:
            self.speed_text_node.setText("---")
            self.speed_text_node.setTextColor(COLOR_TEXT_DIM)
        if self.speed_unit_node is not None:
            self.speed_unit_node.setTextColor(COLOR_TEXT_DIM)
        self._update_speedometer_arc(0.0)

        # 3. Turn indicators: reset to disabled
        if self.indicator_left_np is not None:
            self.indicator_left_np.setColor(COLOR_INDICATOR_DISABLED)
        if self.indicator_right_np is not None:
            self.indicator_right_np.setColor(COLOR_INDICATOR_DISABLED)

        # 4. Pedal meters: reset to disabled frames with hidden fills
        if self.brake_bg is not None:
            self.brake_bg.setColor(COLOR_INDICATOR_DISABLED)
        if self.throttle_bg is not None:
            self.throttle_bg.setColor(COLOR_INDICATOR_DISABLED)
        if self.brake_border is not None:
            self.brake_border.setColor(COLOR_FRAME_DISABLED)
        if self.throttle_border is not None:
            self.throttle_border.setColor(COLOR_FRAME_DISABLED)
        if self.brake_fill is not None:
            self.brake_fill.hide()
        if self.throttle_fill is not None:
            self.throttle_fill.hide()

    def destroy(self) -> None:
        """
        Idempotently and safely destroys all racing HUD nodes on aspect2d.
        Can be invoked repeatedly without throwing exceptions.
        """
        self._is_destroyed = True

        for root_name in ("progress_root", "speed_root", "pedals_root", "indicators_root"):
            np = getattr(self, root_name, None)
            if np is not None:
                try:
                    np.removeNode()
                except Exception:
                    pass
                setattr(self, root_name, None)

        if self.speed_active_arc_np is not None:
            try:
                self.speed_active_arc_np.removeNode()
            except Exception:
                pass
            self.speed_active_arc_np = None

        self.progress_fill = None
        self.progress_text_node = None
        self.speed_text_node = None
        self.speed_unit_node = None
        self.indicator_left_np = None
        self.indicator_right_np = None
        self.brake_bg = None
        self.brake_border = None
        self.brake_fill = None
        self.throttle_bg = None
        self.throttle_border = None
        self.throttle_fill = None
