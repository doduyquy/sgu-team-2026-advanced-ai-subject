"""
MetaDrive Live Minimap Overlay with Vehicle Heading & Trajectory Trail (Gate HUD-1).

This module implements hardware-accelerated top-down orthographic minimap rendering
directly in the MetaDrive Panda3D window (render_mode="NATIVE"):
- Uses secondary Panda3D camera with OrthographicLens and CamMask.MiniMap.
- Dedicated DisplayRegion anchored to top-left corner per HUD-0 specification.
- Directional heading marker indicating exact yaw travel direction in real-time.
- Bounded in-memory trajectory ring buffer (K <= 500) rendering traveled trail.
- Step-driven decoupled pose update API: update(world_x, world_y, heading_rad).
- Complete scientific isolation: zero invocation and zero overhead when render_mode="OFF".

Attribution:
Adopts and evolves the foundational Panda3D camera and DisplayRegion architecture
originally developed by Đặng Khoa on branch origin/khoa (commit 271d704).
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import numpy as np
from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LineSegs,
    OrthographicLens,
    Point2,
    Point3,
    Vec4,
)

from metadrive.constants import CamMask
from src.visualization.minimap_math import (
    TrajectoryTrailBuffer,
    calculate_film_size,
    calculate_heading_rotation_deg,
    calculate_map_bounds,
)


class MiniMapOverlay:
    """
    Hardware-accelerated top-down orthographic minimap overlay rendered in MetaDrive NATIVE mode.

    Visual Elements:
    1. Static road geometry rendered by GPU via CamMask.MiniMap.
    2. Translucent dark glass background quad (#0B131B @ 0.75 alpha).
    3. Directional vehicle chevron marker oriented by heading angle.
    4. Non-directional circular dot fallback when heading is unavailable.
    5. Bounded cyan trajectory trail showing recent traveled route (K <= 500).
    """

    # Top-Left Normalized Screen Coordinates (left, right, bottom, top) in [0.0, 1.0] per HUD-0 spec
    REGION = (0.015, 0.185, 0.745, 0.985)

    def __init__(self, env: Any, padding_m: float = 12.0) -> None:
        self.env = env
        self.engine = getattr(env, "engine", None)
        self.camera = None
        self.display_region = None
        self.background = None
        self.marker_root = None
        self.marker_arrow = None
        self.marker_dot = None
        self.trail_root = None
        self.trail_np = None
        self._trail_buffer = TrajectoryTrailBuffer(max_capacity=500)

        if self.engine is None:
            raise RuntimeError(
                "MetaDrive engine is not initialized. "
                "MiniMapOverlay must be instantiated after env.reset()."
            )

        if getattr(self.engine, "win", None) is None:
            raise RuntimeError(
                "Render window not available. "
                "MiniMapOverlay requires render_mode='NATIVE' (use_render=True)."
            )

        # Extract map bounds and configure camera
        map_features = getattr(env.current_map, "get_map_features", lambda interval: {})(interval=2.0)
        default_pos = (0.0, 0.0)
        if hasattr(env, "agent") and hasattr(env.agent, "position") and env.agent.position is not None:
            try:
                default_pos = (float(env.agent.position[0]), float(env.agent.position[1]))
            except (IndexError, TypeError, ValueError):
                pass

        self.bounds = calculate_map_bounds(map_features, default_center=default_pos, padding_m=padding_m)
        self._create_camera(self.bounds, padding_m)
        self._create_background()
        self._create_markers()
        self._create_trail_root()

    def _create_camera(
        self,
        bounds: Tuple[float, float, float, float],
        padding_m: float
    ) -> None:
        min_x, max_x, min_y, max_y = bounds
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0

        left, right, bottom, top = self.REGION
        win_w = max(float(self.engine.win.getXSize()), 1.0)
        win_h = max(float(self.engine.win.getYSize()), 1.0)
        region_w = (right - left) * win_w
        region_h = (top - bottom) * win_h
        region_aspect = region_w / max(region_h, 1.0)

        self.film_width, self.film_height = calculate_film_size(bounds, region_aspect, padding_m=padding_m)

        lens = OrthographicLens()
        lens.setFilmSize(self.film_width, self.film_height)
        camera_height = max(self.film_width, self.film_height) * 2.0 + 50.0
        lens.setNearFar(1.0, camera_height + 1000.0)

        self.camera = self.engine.makeCamera(
            self.engine.win,
            sort=5,
            displayRegion=self.REGION,
            camName="project_minimap_camera",
        )
        self.camera.node().setLens(lens)
        self.camera.node().setCameraMask(CamMask.MiniMap)
        self.camera.reparentTo(self.engine.render)
        self.camera.setPos(center_x, center_y, camera_height)
        self.camera.setHpr(0.0, -90.0, 0.0)

        self.display_region = self.camera.node().getDisplayRegion(0)
        self.display_region.setClearColorActive(True)
        # Transparent dark glass background tint (#0B131B with 0.75 alpha)
        self.display_region.setClearColor(Vec4(0.043, 0.075, 0.106, 0.75))
        self.display_region.setClearDepthActive(True)
        self.display_region.setClearDepth(1.0)

    def _create_background(self) -> None:
        """Creates transparent dark backing quad in render with CamMask.MiniMap."""
        min_x, max_x, min_y, max_y = self.bounds
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        half_w = self.film_width / 2.0
        half_h = self.film_height / 2.0

        vertex_data = GeomVertexData("minimap_bg_data", GeomVertexFormat.getV3c4(), Geom.UHStatic)
        vertices = GeomVertexWriter(vertex_data, "vertex")
        colors = GeomVertexWriter(vertex_data, "color")
        bg_col = (0.043, 0.075, 0.106, 0.75)

        for x, y in ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)):
            vertices.addData3f(x, y, 0.0)
            colors.addData4f(*bg_col)

        triangles = GeomTriangles(Geom.UHStatic)
        triangles.addVertices(0, 1, 2)
        triangles.addVertices(0, 2, 3)
        triangles.closePrimitive()

        geom = Geom(vertex_data)
        geom.addPrimitive(triangles)
        bg_node = GeomNode("minimap_bg_node")
        bg_node.addGeom(geom)

        self.background = self.engine.render.attachNewNode(bg_node)
        self.background.setPos(center_x, center_y, -1.0)
        self.background.setLightOff()
        self.background.setTwoSided(True)
        self.background.hide()
        self.background.showThrough(CamMask.MiniMap)

    def _create_markers(self) -> None:
        """
        Creates 2D screen-space marker root in render2d with fixed pixel dimensions:
        - marker_arrow: Directional chevron pointing East (right) at rest (heading=0).
        - marker_dot: Non-directional circular dot used when heading is None.
        """
        win_w = max(float(self.engine.win.getXSize()), 1.0)
        win_h = max(float(self.engine.win.getYSize()), 1.0)
        radius_px = 4.0
        rx = 2.0 * radius_px / win_w
        rz = 2.0 * radius_px / win_h

        self.marker_root = self.engine.render2d.attachNewNode("minimap_marker_root")
        self.marker_root.setLightOff()
        self.marker_root.setShaderOff(1)
        self.marker_root.setFogOff(1)
        self.marker_root.setTextureOff(1)
        self.marker_root.setDepthTest(False)
        self.marker_root.setDepthWrite(False)
        self.marker_root.setTwoSided(True)
        self.marker_root.setBin("fixed", 225)

        # 1. Directional Chevron (Tip points along +X at heading=0)
        # Electric cyan accent color (#00E5FF)
        arrow_color = (0.0, 0.898, 1.0, 1.0)
        arrow_vdata = GeomVertexData("marker_arrow_vdata", GeomVertexFormat.getV3c4(), Geom.UHStatic)
        av = GeomVertexWriter(arrow_vdata, "vertex")
        ac = GeomVertexWriter(arrow_vdata, "color")

        # Vertices: tip (+X), left wing, right wing, center notch
        tip = (1.8 * rx, 0.0, 0.0)
        left_wing = (-1.2 * rx, 0.0, 1.2 * rz)
        right_wing = (-1.2 * rx, 0.0, -1.2 * rz)
        notch = (-0.4 * rx, 0.0, 0.0)

        for p in (tip, left_wing, notch, right_wing):
            av.addData3f(*p)
            ac.addData4f(*arrow_color)

        tri_arrow = GeomTriangles(Geom.UHStatic)
        tri_arrow.addVertices(0, 1, 2)
        tri_arrow.addVertices(0, 2, 3)
        tri_arrow.closePrimitive()

        g_arrow = Geom(arrow_vdata)
        g_arrow.addPrimitive(tri_arrow)
        node_arrow = GeomNode("marker_arrow_node")
        node_arrow.addGeom(g_arrow)
        self.marker_arrow = self.marker_root.attachNewNode(node_arrow)

        # 2. Non-directional Circular Dot (fallback when heading is None)
        dot_vdata = GeomVertexData("marker_dot_vdata", GeomVertexFormat.getV3c4(), Geom.UHStatic)
        dv = GeomVertexWriter(dot_vdata, "vertex")
        dc = GeomVertexWriter(dot_vdata, "color")
        dv.addData3f(0.0, 0.0, 0.0)
        dc.addData4f(*arrow_color)

        segments = 16
        for idx in range(segments + 1):
            angle = 2.0 * math.pi * idx / segments
            dv.addData3f(rx * math.cos(angle), 0.0, rz * math.sin(angle))
            dc.addData4f(*arrow_color)

        tri_dot = GeomTriangles(Geom.UHStatic)
        for idx in range(1, segments + 1):
            tri_dot.addVertices(0, idx, idx + 1)
        tri_dot.closePrimitive()

        g_dot = Geom(dot_vdata)
        g_dot.addPrimitive(tri_dot)
        node_dot = GeomNode("marker_dot_node")
        node_dot.addGeom(g_dot)
        self.marker_dot = self.marker_root.attachNewNode(node_dot)
        self.marker_dot.hide()

        self.marker_root.hide()

    def _create_trail_root(self) -> None:
        """Initializes trajectory trail root attached to engine.render with CamMask.MiniMap."""
        self.trail_root = self.engine.render.attachNewNode("minimap_trail_root")
        self.trail_root.setLightOff()
        self.trail_root.setShaderOff(1)
        self.trail_root.hide()
        self.trail_root.showThrough(CamMask.MiniMap)

    def _rebuild_trail_geometry(self) -> None:
        """Reconstructs LineSegs polyline connecting retained trajectory buffer points."""
        if self.trail_np is not None:
            self.trail_np.removeNode()
            self.trail_np = None

        points = self._trail_buffer.get_points()
        if len(points) < 2:
            return

        ls = LineSegs("minimap_trail_lines")
        ls.setThickness(2.5)
        # Fading cyan trail color (#00E5FF @ 0.85 alpha)
        ls.setColor(0.0, 0.898, 1.0, 0.85)

        first = True
        for px, py in points:
            if first:
                ls.moveTo(px, py, 0.8)
                first = False
            else:
                ls.drawTo(px, py, 0.8)

        geom_node = ls.create()
        self.trail_np = self.trail_root.attachNewNode(geom_node)

    def update(
        self,
        world_x: Optional[float] = None,
        world_y: Optional[float] = None,
        heading_rad: Optional[float] = None
    ) -> None:
        """
        Step-driven decoupled pose update:
        - If world_x or world_y is omitted or None: falls back to env.agent position if available.
        - If coordinates are missing/invalid: hides marker without appending to trajectory.
        - If coordinates are valid but heading_rad is None: displays non-directional position dot.
        - If coordinates and heading are valid: displays directional chevron rotated to heading.
        - Appends valid coordinates to trajectory buffer under minimum distance filter (0.5m).
        """
        if self.camera is None or self.marker_root is None:
            return

        # Fallback to env.agent if position arguments not provided directly
        if world_x is None or world_y is None:
            if self.env is not None and hasattr(self.env, "agent") and self.env.agent is not None:
                pos = getattr(self.env.agent, "position", None)
                if pos is not None:
                    try:
                        world_x, world_y = float(pos[0]), float(pos[1])
                    except (IndexError, TypeError, ValueError):
                        world_x, world_y = None, None

                if heading_rad is None:
                    h_val = getattr(self.env.agent, "heading_theta", None)
                    if h_val is not None:
                        try:
                            heading_rad = float(h_val)
                        except (TypeError, ValueError):
                            heading_rad = None

        # Handle missing or non-finite position
        if world_x is None or world_y is None or not (math.isfinite(world_x) and math.isfinite(world_y)):
            self.marker_root.hide()
            return

        # 1. Project world coordinate through minimap camera lens
        world_pos = Point3(float(world_x), float(world_y), 0.0)
        cam_pos = self.camera.getRelativePoint(self.engine.render, world_pos)
        projected = Point2()

        if not self.camera.node().getLens().project(cam_pos, projected):
            self.marker_root.hide()
            return

        left, right, bottom, top = self.REGION
        region_u = (float(projected.x) + 1.0) / 2.0
        region_v = (float(projected.y) + 1.0) / 2.0

        # Screen-space clipping: hide marker if vehicle is outside minimap display region
        if not (0.0 <= region_u <= 1.0 and 0.0 <= region_v <= 1.0):
            self.marker_root.hide()
            return

        screen_u = left + region_u * (right - left)
        screen_v = bottom + region_v * (top - bottom)
        screen_x = screen_u * 2.0 - 1.0
        screen_z = screen_v * 2.0 - 1.0

        # Position marker in render2d
        self.marker_root.show()
        self.marker_root.setPos(screen_x, 0.0, screen_z)

        # 2. Update heading orientation
        if heading_rad is not None and math.isfinite(heading_rad):
            self.marker_arrow.show()
            self.marker_dot.hide()
            self.marker_root.setR(calculate_heading_rotation_deg(heading_rad))
        else:
            self.marker_dot.show()
            self.marker_arrow.hide()
            self.marker_root.setR(0.0)

        # 3. Update trajectory trail ring buffer
        retained = self._trail_buffer.add_sample(world_x, world_y, min_distance=0.5)
        if retained and self.trail_root is not None:
            self._rebuild_trail_geometry()

    def reset(self) -> None:
        """
        Resets episodic visualization state:
        - Clears trajectory trail buffer to 0 points.
        - Removes active trail line geometry.
        - Hides marker until next step.
        Does NOT destroy camera or display region.
        """
        self._trail_buffer.reset()
        if self.trail_np is not None:
            self.trail_np.removeNode()
            self.trail_np = None
        if self.marker_root is not None:
            self.marker_root.hide()

    def destroy(self) -> None:
        """
        Safely and idempotently destroys minimap visualizer resources:
        - Removes trail geometry and marker nodes.
        - Removes DisplayRegion from window.
        - Removes camera from scene graph and camList.
        Can be safely invoked multiple times without exception.
        """
        self.reset()

        if self.trail_root is not None:
            self.trail_root.removeNode()
            self.trail_root = None

        if self.marker_root is not None:
            self.marker_root.removeNode()
            self.marker_root = None
            self.marker_arrow = None
            self.marker_dot = None

        if self.background is not None:
            self.background.removeNode()
            self.background = None

        if self.display_region is not None:
            if self.engine is not None and getattr(self.engine, "win", None) is not None:
                try:
                    self.engine.win.removeDisplayRegion(self.display_region)
                except Exception:
                    pass
            self.display_region = None

        if self.camera is not None:
            if self.engine is not None and hasattr(self.engine, "camList"):
                try:
                    if self.camera in self.engine.camList:
                        self.engine.camList.remove(self.camera)
                except Exception:
                    pass
            try:
                self.camera.removeNode()
            except Exception:
                pass
            self.camera = None
