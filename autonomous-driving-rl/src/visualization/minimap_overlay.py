from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    OrthographicLens,
    Point2,
    Point3,
    Vec4,
)

from metadrive.constants import CamMask


class MiniMapOverlay:
    """
    Camera top-down cố định hiển thị toàn bộ map ở góc trái.

    Dấu đỏ là một node riêng và được đồng bộ với tọa độ ego vehicle
    ở từng frame để không chịu ảnh hưởng từ camera mask của model xe.
    """

    # DisplayRegion dùng hệ tọa độ chuẩn hóa:
    # (left, right, bottom, top), mỗi giá trị trong khoảng [0, 1].
    # Khoảng 200 x 200 px với cửa sổ mặc định 1200 x 900.
    REGION = (0.015, 0.18, 0.755, 0.975)

    def __init__(self, env: Any, padding_m: float = 12.0) -> None:
        self.env = env
        self.engine = env.engine
        self.camera = None
        self.display_region = None
        self.background = None
        self.marker = None
        self._marker_task_name = None

        if self.engine is None:
            raise RuntimeError(
                "MetaDrive engine chưa được khởi tạo. "
                "Hãy tạo minimap sau env.reset()."
            )

        if self.engine.win is None:
            raise RuntimeError(
                "Không có cửa sổ render. Minimap chỉ hoạt động "
                "khi use_render=True."
            )

        bounds = self._calculate_map_bounds(env)
        self._create_camera(bounds, padding_m)
        self._create_vehicle_marker(env)

    @staticmethod
    def _calculate_map_bounds(
        env: Any,
    ) -> Tuple[float, float, float, float]:
        """
        Trả về (min_x, max_x, min_y, max_y) của toàn bộ bản đồ.

        get_map_features() cung cấp các polyline của làn đường,
        vạch đường và các thành phần bản đồ khác.
        """
        map_features = env.current_map.get_map_features(interval=2.0)

        point_groups = []

        for feature in map_features.values():
            polyline = feature.get("polyline")

            if polyline is None:
                continue

            points = np.asarray(polyline, dtype=np.float64)

            if (
                points.ndim == 2
                and points.shape[0] > 0
                and points.shape[1] >= 2
            ):
                point_groups.append(points[:, :2])

        if not point_groups:
            # Trường hợp dự phòng nếu map không trả về polyline.
            vehicle_position = np.asarray(env.agent.position, dtype=np.float64)
            x = float(vehicle_position[0])
            y = float(vehicle_position[1])
            return x - 50.0, x + 50.0, y - 50.0, y + 50.0

        all_points = np.concatenate(point_groups, axis=0)

        min_x = float(np.min(all_points[:, 0]))
        max_x = float(np.max(all_points[:, 0]))
        min_y = float(np.min(all_points[:, 1]))
        max_y = float(np.max(all_points[:, 1]))

        return min_x, max_x, min_y, max_y

    def _create_camera(
        self,
        bounds: Tuple[float, float, float, float],
        padding_m: float,
    ) -> None:
        min_x, max_x, min_y, max_y = bounds

        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0

        map_width = max(max_x - min_x, 20.0) + 2.0 * padding_m
        map_height = max(max_y - min_y, 20.0) + 2.0 * padding_m

        left, right, bottom, top = self.REGION

        window_width = max(float(self.engine.win.getXSize()), 1.0)
        window_height = max(float(self.engine.win.getYSize()), 1.0)

        region_pixel_width = (right - left) * window_width
        region_pixel_height = (top - bottom) * window_height
        region_aspect = region_pixel_width / max(region_pixel_height, 1.0)

        map_aspect = map_width / map_height

        # Điều chỉnh kích thước film để map không bị kéo giãn hoặc cắt mất.
        if map_aspect > region_aspect:
            film_width = map_width
            film_height = map_width / region_aspect
        else:
            film_height = map_height
            film_width = map_height * region_aspect

        lens = OrthographicLens()
        lens.setFilmSize(film_width, film_height)

        camera_height = max(film_width, film_height) * 2.0 + 50.0
        lens.setNearFar(1.0, camera_height + 1000.0)

        # Vẽ sau main 3D camera nhưng trước camera render2d của giao diện.
        # Marker ego dạng 2D nhờ đó luôn nằm trên minimap.
        self.camera = self.engine.makeCamera(
            self.engine.win,
            sort=5,
            displayRegion=self.REGION,
            camName="project_minimap_camera",
        )

        self.camera.node().setLens(lens)

        # MetaDrive dành riêng mask này cho góc nhìn minimap. Dùng MainCam
        # ở đây làm một số lớp terrain bị render thành nền đen.
        self.camera.node().setCameraMask(CamMask.MiniMap)

        # Camera nhìn thẳng từ trên xuống.
        self.camera.reparentTo(self.engine.render)
        self.camera.setPos(center_x, center_y, camera_height)
        self.camera.setHpr(0.0, -90.0, 0.0)

        self.display_region = self.camera.node().getDisplayRegion(0)
        self.display_region.setClearColorActive(True)
        # Màu nền xanh-xám giúp phân biệt phần ngoài đường thay vì một ô đen.
        self.display_region.setClearColor(Vec4(0.16, 0.20, 0.16, 1.0))
        self.display_region.setClearDepthActive(True)
        self.display_region.setClearDepth(1.0)

        self._map_visual_size = max(film_width, film_height)
        self._create_background(
            center_x=center_x,
            center_y=center_y,
            width=film_width,
            height=film_height,
        )

    def _create_background(
        self,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
    ) -> None:
        """Tạo nền xanh-xám chỉ xuất hiện trong camera minimap."""
        vertex_data = GeomVertexData(
            "project_minimap_background_data",
            GeomVertexFormat.getV3c4(),
            Geom.UHStatic,
        )
        vertices = GeomVertexWriter(vertex_data, "vertex")
        colors = GeomVertexWriter(vertex_data, "color")

        half_width = width / 2.0
        half_height = height / 2.0
        background_color = (0.16, 0.20, 0.16, 1.0)

        for x, y in (
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        ):
            vertices.addData3f(x, y, 0.0)
            colors.addData4f(*background_color)

        triangles = GeomTriangles(Geom.UHStatic)
        triangles.addVertices(0, 1, 2)
        triangles.addVertices(0, 2, 3)
        triangles.closePrimitive()

        geometry = Geom(vertex_data)
        geometry.addPrimitive(triangles)

        background_node = GeomNode("project_minimap_background")
        background_node.addGeom(geometry)
        self.background = self.engine.render.attachNewNode(background_node)
        self.background.setPos(center_x, center_y, -1.0)
        self.background.setLightOff()
        self.background.setTwoSided(True)

        # showThrough vượt qua mask của các node cha trong scene graph.
        self.background.hide()
        self.background.showThrough(CamMask.MiniMap)

    def _create_vehicle_marker(self, env: Any) -> None:
        """
        Tạo chấm đỏ sáng dành riêng cho ego vehicle.

        Marker được vẽ trong render2d với kích thước pixel cố định. Vì vậy
        nó không bị camera mask, shader của scene hay tỉ lệ map làm mờ/thu nhỏ.
        """
        self.marker = self._create_screen_circle(
            name="project_minimap_vehicle_marker",
            radius_px=3.0,
            # MetaDrive 0.4.3's render2d/filter path swaps R/B for this
            # vertex format; blue input is displayed as bright red.
            color=(0.0, 0.0, 1.0, 1.0),
            draw_order=221,
        )

        self.update()
        self._marker_task_name = f"project_minimap_marker_update_{id(self)}"
        self.engine.taskMgr.add(
            self._update_marker_task,
            self._marker_task_name,
            sort=50,
        )

    def _create_screen_circle(
        self,
        name: str,
        radius_px: float,
        color: Tuple[float, float, float, float],
        draw_order: int,
    ) -> Any:
        """Tạo một hình tròn render2d có bán kính cố định theo pixel."""
        segments = 24

        window_width = max(float(self.engine.win.getXSize()), 1.0)
        window_height = max(float(self.engine.win.getYSize()), 1.0)
        radius_x = 2.0 * radius_px / window_width
        radius_z = 2.0 * radius_px / window_height

        vertex_data = GeomVertexData(
            f"{name}_data",
            GeomVertexFormat.getV3c4(),
            Geom.UHStatic,
        )
        vertices = GeomVertexWriter(vertex_data, "vertex")
        colors = GeomVertexWriter(vertex_data, "color")

        vertices.addData3f(0.0, 0.0, 0.0)
        colors.addData4f(*color)

        for index in range(segments + 1):
            angle = 2.0 * np.pi * index / segments
            vertices.addData3f(
                radius_x * float(np.cos(angle)),
                0.0,
                radius_z * float(np.sin(angle)),
            )
            colors.addData4f(*color)

        triangles = GeomTriangles(Geom.UHStatic)
        for index in range(1, segments + 1):
            triangles.addVertices(0, index, index + 1)
        triangles.closePrimitive()

        geometry = Geom(vertex_data)
        geometry.addPrimitive(triangles)

        marker_node = GeomNode(name)
        marker_node.addGeom(geometry)
        circle = self.engine.render2d.attachNewNode(marker_node)
        circle.setLightOff()
        circle.setShaderOff(1)
        circle.setFogOff(1)
        circle.setTextureOff(1)
        circle.setDepthTest(False)
        circle.setDepthWrite(False)
        circle.setTwoSided(True)
        circle.setBin("fixed", draw_order)
        return circle

    def update(self) -> None:
        """Đồng bộ chấm đỏ với tọa độ thế giới hiện tại của ego vehicle."""
        if self.marker is None or self.env is None or self.env.agent is None:
            return

        vehicle_position = self.env.agent.position
        world_position = Point3(
            float(vehicle_position[0]),
            float(vehicle_position[1]),
            0.0,
        )
        camera_position = self.camera.getRelativePoint(
            self.engine.render,
            world_position,
        )
        projected = Point2()

        if not self.camera.node().getLens().project(camera_position, projected):
            self.marker.hide()
            return

        left, right, bottom, top = self.REGION
        region_u = (float(projected.x) + 1.0) / 2.0
        region_v = (float(projected.y) + 1.0) / 2.0

        if not (0.0 <= region_u <= 1.0 and 0.0 <= region_v <= 1.0):
            self.marker.hide()
            return

        screen_u = left + region_u * (right - left)
        screen_v = bottom + region_v * (top - bottom)
        screen_x = screen_u * 2.0 - 1.0
        screen_z = screen_v * 2.0 - 1.0

        self.marker.show()
        self.marker.setPos(screen_x, 0.0, screen_z)

    def _update_marker_task(self, task: Any) -> Any:
        self.update()
        return task.cont

    def destroy(self) -> None:
        """
        Xóa camera, marker và display region trước khi env.close().
        """
        if self._marker_task_name is not None:
            self.engine.taskMgr.remove(self._marker_task_name)
            self._marker_task_name = None

        if self.marker is not None:
            self.marker.removeNode()
            self.marker = None

        if self.background is not None:
            self.background.removeNode()
            self.background = None

        if self.display_region is not None and self.engine.win is not None:
            self.engine.win.removeDisplayRegion(self.display_region)
            self.display_region = None

        if self.camera is not None:
            if self.camera in self.engine.camList:
                self.engine.camList.remove(self.camera)

            self.camera.removeNode()
            self.camera = None
