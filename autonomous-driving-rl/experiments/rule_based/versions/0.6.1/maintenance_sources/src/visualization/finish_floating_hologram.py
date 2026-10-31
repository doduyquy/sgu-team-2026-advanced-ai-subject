from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from metadrive.constants import CamMask
from metadrive.component.road_network.road import Road
from panda3d.core import (
    CardMaker,
    CullFaceAttrib,
    Filename,
    NodePath,
    TexturePool,
    TransparencyAttrib,
)


_TEXTURE_CACHE_DIR = Path(__file__).resolve().parents[2] / "runs" / ".cache" / "finish_visual"


class FinishFloatingHologram:
    """
    Vạch đích 3D Hologram lơ lửng thuần túy (Pure Floating 3D Hologram).

    Đặc điểm:
    - 100% lơ lửng trên không trung, TUYỆT ĐỐI KHÔNG CÓ CỘT TRỤ, CỔNG CHỮ U HAY DẦM ĐEN.
    - Kéo dài bao trùm toàn bộ mặt đường ở cả hai chiều tại đích.
    - Bảng Hologram phát sáng rực rỡ với dòng chữ 'FINISH' neon cyan/magenta/white cực rõ.
    - Dải vạch Laser neon công nghệ cao phủ trên mặt đường dọc tất cả các làn xe.
    - Hoàn toàn độc lập với Bullet Physics: không va chạm, không ảnh hưởng LiDAR.
    """

    def __init__(
        self,
        env: Any,
        banner_height_m: float = 4.0,
        text: str = "FINISH",
    ) -> None:
        self.env = env
        self.engine = getattr(env, "engine", None)
        self.root: NodePath | None = None
        self.hologram_card: NodePath | None = None
        self.ground_laser_card: NodePath | None = None
        self._elapsed_time = 0.0

        if self.engine is None or not hasattr(self.engine, "render") or self.engine.render is None:
            raise RuntimeError(
                "MetaDrive engine chưa được khởi tạo. Hãy tạo FinishFloatingHologram sau env.reset()."
            )

        pos_mid, heading_deg, total_road_width = self.calculate_finish_pose(env)
        self.pos_mid = pos_mid
        self.heading_deg = heading_deg
        self.total_road_width = total_road_width
        self.banner_width = self.total_road_width + 1.2
        self.banner_height_m = banner_height_m
        self.text = text

        self._texture_cache_dir = _TEXTURE_CACHE_DIR
        self._texture_cache_dir.mkdir(parents=True, exist_ok=True)
        # Version the layout and text so old cached subtitles cannot reappear.
        text_key = hashlib.sha256(self.text.strip().encode("utf-8")).hexdigest()[:12]
        self._holo_path = self._texture_cache_dir / f"finish_holo_finish_only_v2_{text_key}.png"

        try:
            self._ensure_textures()
            self._build_pure_floating_hologram()
        except Exception:
            self.destroy()
            raise

    @staticmethod
    def calculate_finish_pose(env: Any) -> Tuple[Tuple[float, float, float], float, float]:
        """
        Tính tâm và bề rộng toàn bộ mặt đường, gồm cả chiều ngược lại.
        Trả về: ((mid_x, mid_y, 0.0), heading_deg, total_road_width)
        """
        nav = env.agent.navigation
        lane = nav.final_lane
        s = max(float(lane.length) - 5.0, 0.0)

        heading_theta = float(lane.heading_theta_at(s))
        anchor = np.asarray(lane.position(s, 0.0), dtype=float)[:2]
        lateral_axis = np.array([math.sin(heading_theta), -math.cos(heading_theta)])
        u, v, _ = lane.index
        graph = env.current_map.road_network.graph
        lanes = list(graph.get(u, {}).get(v, ())) or [lane]
        reverse_road = -Road(u, v)
        lanes.extend(graph.get(reverse_road.start_node, {}).get(reverse_road.end_node, ()))

        # Opposite-direction lanes start at the other end. Project the same
        # world cross-section into each lane instead of reusing longitudinal s.
        lateral_edges = []
        for road_lane in lanes:
            lane_s, _ = road_lane.local_coordinates(anchor)
            lane_s = float(np.clip(lane_s, 0.0, road_lane.length))
            half_width = float(road_lane.width_at(lane_s)) / 2.0
            for side in (-half_width, half_width):
                edge = np.asarray(road_lane.position(lane_s, side), dtype=float)[:2]
                lateral_edges.append(float(np.dot(edge - anchor, lateral_axis)))
        left, right = min(lateral_edges), max(lateral_edges)
        pos_mid = anchor + (left + right) / 2.0 * lateral_axis
        total_road_width = right - left
        heading_deg = float(np.rad2deg(heading_theta) - 90.0)

        return (float(pos_mid[0]), float(pos_mid[1]), 0.0), heading_deg, total_road_width

    def _ensure_textures(self) -> None:
        """Sinh và lưu trữ texture độ phân giải cao cho Hologram và Laser mặt đường."""
        holo_path = self._holo_path
        laser_path = self._texture_cache_dir / "finish_ground_laser.png"

        # 1. Texture bảng Hologram
        if not holo_path.exists():
            w, h = 2048, 512
            img = Image.new("RGBA", (w, h), (4, 10, 22, 225))
            draw = ImageDraw.Draw(img)

            # Viền Cyberpunk kép
            draw.rounded_rectangle([(10, 10), (w - 11, h - 11)], radius=32, outline=(0, 240, 255, 255), width=8)
            draw.rounded_rectangle([(26, 26), (w - 27, h - 27)], radius=24, outline=(255, 0, 128, 220), width=4)

            # Khung góc công nghệ
            cw = 80
            for x0, y0, x1, y1 in [
                (10, 10, 10 + cw, 10 + cw),
                (w - 11 - cw, 10, w - 11, 10 + cw),
                (10, h - 11 - cw, 10 + cw, h - 11),
                (w - 11 - cw, h - 11 - cw, w - 11, h - 11),
            ]:
                draw.rectangle([(x0, y0), (x1, y1)], fill=(0, 240, 255, 75), outline=(0, 240, 255, 255), width=5)

            try:
                main_font = ImageFont.truetype("arialbd.ttf", 220)
            except Exception:
                main_font = ImageFont.load_default()

            title = self.text.strip()
            bbox_m = draw.textbbox((0, 0), title, font=main_font)
            tw, th = bbox_m[2] - bbox_m[0], bbox_m[3] - bbox_m[1]
            tx, ty = (w - tw) // 2 - bbox_m[0], (h - th) // 2 - bbox_m[1]

            # Multi-layer bloom glow
            for r in range(18, 0, -3):
                alpha = int(50 * (20 - r) / 20)
                for dx, dy in [(-r, 0), (r, 0), (0, -r), (0, r), (-r, -r), (r, r), (-r, r), (r, -r)]:
                    draw.text((tx + dx, ty + dy), title, font=main_font, fill=(0, 240, 255, alpha))
                for dx, dy in [(-r // 2, 0), (r // 2, 0)]:
                    draw.text((tx + dx, ty + dy), title, font=main_font, fill=(255, 0, 128, alpha))

            draw.text((tx, ty), title, font=main_font, fill=(255, 255, 255, 255))
            img.save(str(holo_path))

        # 2. Texture Laser mặt đường
        if not laser_path.exists():
            gw, gh = 1024, 256
            g_img = Image.new("RGBA", (gw, gh), (0, 0, 0, 0))
            g_draw = ImageDraw.Draw(g_img)

            g_draw.line([(0, 25), (gw, 25)], fill=(0, 240, 255, 255), width=10)
            g_draw.line([(0, 128), (gw, 128)], fill=(255, 255, 255, 255), width=16)
            g_draw.line([(0, 230), (gw, 230)], fill=(255, 0, 128, 255), width=10)

            step = gw // 14
            for i in range(14):
                cx = i * step + step // 2
                g_draw.polygon(
                    [
                        (cx - 26, 45),
                        (cx + 26, 128),
                        (cx - 26, 211),
                        (cx - 12, 211),
                        (cx + 40, 128),
                        (cx - 12, 45),
                    ],
                    fill=(0, 240, 255, 220),
                )

            g_img.save(str(laser_path))

    def _build_pure_floating_hologram(self) -> None:
        """Dựng bảng hologram lơ lửng thuần túy (KHÔNG CÓ CỘT, KHÔNG CÓ KHUNG U ĐEN)."""
        self.root = self.engine.render.attachNewNode("pure_floating_hologram")
        # MetaDrive assigns its PBR shader to render. CardMaker's unlit RGBA
        # cards must opt out: setLightOff alone does not override that shader.
        self.root.setShaderOff(1)
        self.root.setLightOff(1)
        self.root.setDepthWrite(False, 1)
        # Decoration belongs to the viewer/minimap, not RGB/depth/semantic
        # observation cameras or the shadow pass. No Bullet nodes are created.
        self.root.hide(CamMask.AllOn)
        self.root.show(CamMask.MainCam | CamMask.MiniMap)
        self.root.setPos(*self.pos_mid)
        self.root.setH(self.heading_deg)

        fn_holo = Filename.fromOsSpecific(str(self._holo_path.resolve()))
        fn_laser = Filename.fromOsSpecific(str((self._texture_cache_dir / "finish_ground_laser.png").resolve()))

        # 1. Bảng Hologram 3D phát sáng lơ lửng giữa trời bao trùm toàn bộ các làn đường
        tex_holo = TexturePool.loadTexture(fn_holo)
        if tex_holo is None:
            raise RuntimeError(f"Không tải được texture hologram: {fn_holo}")
        cm_holo = CardMaker("hologram_card")
        hw = self.banner_width / 2.0
        hh = 1.15  # Chiều cao bảng 2.3m
        cm_holo.setFrame(-hw, hw, -hh, hh)

        self.hologram_card = NodePath(cm_holo.generate())
        self.hologram_card.reparentTo(self.root)
        self.hologram_card.setPos(0.0, 0.0, self.banner_height_m)
        self.hologram_card.setTexture(tex_holo, 1)
        self.hologram_card.setTransparency(TransparencyAttrib.MAlpha, 1)
        self.hologram_card.setLightOff(1)
        self.hologram_card.setTwoSided(True, 1)
        self.hologram_card.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone), 1)

        # 2. Vạch Laser công nghệ cao phủ trên mặt đường dọc tất cả các làn xe
        tex_laser = TexturePool.loadTexture(fn_laser)
        if tex_laser is None:
            raise RuntimeError(f"Không tải được texture laser: {fn_laser}")
        cm_laser = CardMaker("ground_laser_card")
        hl = 1.2
        laser_half_width = self.total_road_width / 2.0
        cm_laser.setFrame(-laser_half_width, laser_half_width, -hl, hl)

        self.ground_laser_card = NodePath(cm_laser.generate())
        self.ground_laser_card.reparentTo(self.root)
        self.ground_laser_card.setP(-90)  # Nằm phẳng trên mặt đường
        self.ground_laser_card.setPos(0.0, 0.0, 0.03)  # Cao hơn mặt đường 3cm tránh Z-fighting
        self.ground_laser_card.setTexture(tex_laser, 1)
        self.ground_laser_card.setTransparency(TransparencyAttrib.MAlpha, 1)
        self.ground_laser_card.setLightOff(1)
        self.ground_laser_card.setTwoSided(True, 1)
        self.ground_laser_card.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone), 1)

    def update(self) -> None:
        """Cập nhật chuyển động nhấp nhô lơ lửng êm ái cho Hologram."""
        if self.hologram_card is None:
            return

        self._elapsed_time += 0.03
        hover_offset = 0.05 * math.sin(self._elapsed_time * 1.5)
        self.hologram_card.setZ(self.banner_height_m + hover_offset)

    def destroy(self) -> None:
        """Dọn dẹp sạch toàn bộ node trong scene graph khi kết thúc episode."""
        if self.root is not None:
            self.root.removeNode()
            self.root = None
            self.hologram_card = None
            self.ground_laser_card = None
