"""Vietnamese DirectGUI screens inside MetaDrive's existing Panda3D window."""

import os
from pathlib import Path

from direct.gui.DirectGui import DirectButton, DirectFrame, DirectLabel
from direct.gui import DirectGuiGlobals as DGG
from panda3d.core import Filename, TextNode, TransparencyAttrib

from src.visualization.race_session import OUTCOME_LABELS, RACE_AGENTS, ROAD_NAMES
from src.visualization.race_style import (
    RACE_COLORS, draw_vector_check, draw_vector_plus, rounded_card, tint,
)


AGENT_SPECS = {
    0: {
        "title": "A* Search",
        "category": "Thuật toán tìm kiếm Heuristic tối ưu đường đi",
        "mechanism": "Heuristic f = g + h",
        "memory": "Trung bình (O(b^d))",
        "strength": "Đường đi ngắn nhất tối ưu",
    },
    1: {
        "title": "BFS (Breadth-First)",
        "category": "Duyệt không gian trạng thái theo chiều rộng",
        "mechanism": "Hàng đợi FIFO toàn diện",
        "memory": "Rất cao (Tốn RAM)",
        "strength": "Số bước di chuyển ít nhất",
    },
    2: {
        "title": "Dijkstra Search",
        "category": "Tìm đường tối ưu chi phí thực tế lũy kế",
        "mechanism": "Priority Queue (Hàng đợi ưu tiên)",
        "memory": "Cao (O(V + E log V))",
        "strength": "Tối ưu tuyệt đối mọi trọng số",
    },
    3: {
        "title": "Hybrid A* Kinematic",
        "category": "Thuật toán tìm đường có xét mô hình động học xe",
        "mechanism": "Bicubic & Reeds-Shepp",
        "memory": "Trung bình - Cao",
        "strength": "Quỹ đạo bám sát góc bẻ lái",
    },
    4: {
        "title": "MPPI Stochastic Control",
        "category": "Model Predictive Path Integral · Tối ưu dự đoán",
        "mechanism": "Monte Carlo Sampling (GPU/CPU)",
        "memory": "Tính toán song song cao",
        "strength": "Thích ứng động & tránh vật cản",
    },
}

ROAD_DESCRIPTIONS = (
    ("CẤP 1", "Đường cong nhẹ · 4 đoạn"),
    ("CẤP 2", "Khúc cua liên tục chữ S"),
    ("CẤP 3", "Góc cua hẹp & rẽ gấp"),
    ("VIỆT NAM", "Võ Văn Kiệt · 10 km"),
)


class RaceMenu:
    BG = (.040, .044, .052, .98)          # Deep graphite carbon (#0A0B0D)
    PANEL = (.072, .080, .095, 1.)        # Titanium slate panel (#121418)
    PANEL_ALT = (.095, .106, .125, 1.)    # Elevated panel (#181B20)
    CARD_SURFACE = (.060, .068, .082, 1.) # Clean sub-card (#0F1115)
    INK = (.96, .97, .98, 1.)             # Crisp pure white (#F5F7FA)
    MUTED = (.54, .59, .68, 1.)           # Sleek silver slate (#8A96AE)
    DIMMED = (.35, .39, .48, 1.)          # Subtle slate (#59647A)
    LAUNCH_RED = (.86, .14, .18, 1.)      # Racing Launch Red (#DC242E)
    ACCENT = (.86, .14, .18, 1.)          # Launch Accent
    ACCENT_DIM = (.18, .08, .10, 1.)
    BORDER = (.14, .16, .20, 1.)          # Structural graphite line (#242933)
    BORDER_BRIGHT = (.28, .32, .40, 1.)   # Active border (#404A5C)
    BUTTON = (.090, .100, .120, 1.)       # Matte tactile button (#171A1F)
    SUCCESS = (.12, .80, .50, 1.)         # Performance green
    DANGER = (.90, .20, .24, 1.)          # Racing Red
    GOLD = (.96, .70, .18, 1.)            # Racing Amber

    def __init__(self, engine, enqueue):
        self.engine, self.enqueue = engine, enqueue
        self.preview = None
        self._preview_index = 0
        self._previous_indices = tuple(range(5))
        self._geometry_cache = {}
        self.root = engine.pixel2d.attachNewNode("race_mode_ui")
        self.root.setBin("fixed", 500)
        self.root.setDepthTest(False)
        self.root.setDepthWrite(False)
        self.root.setTransparency(TransparencyAttrib.MAlpha)
        self.font = None
        font_paths = [Path("C:/Windows/Fonts/segoeui.ttf"),
                      Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/segoeui.ttf",
                      Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
        for path in font_paths:
            if path.exists():
                self.font = engine.loader.loadFont(Filename.fromOsSpecific(str(path)).getFullpath())
                break
        bold_path = Path("C:/Windows/Fonts/segoeuib.ttf")
        self.bold_font = engine.loader.loadFont(Filename.fromOsSpecific(str(bold_path)).getFullpath()) if bold_path.exists() else self.font
        self.groups = {name: self.root.attachNewNode(name) for name in ("garage", "running", "results", "paused", "countdown", "comparison")}
        self._size = None
        self._build_garage()
        self._build_running()
        self._build_results()
        self._build_comparison()
        self._build_paused()
        self._build_countdown()
        self.layout()
        self._anim_task_name = f"race_menu_anim_{id(self)}"
        engine.taskMgr.add(self._animation_tick, self._anim_task_name)

    def _animation_tick(self, task):
        import math
        t = task.time
        if hasattr(self, "live_badge_dot") and not self.live_badge_dot.isEmpty():
            pulse = 0.5 + 0.5 * math.sin(t * 4.0)
            self.live_badge_dot.setColorScale(1.0, 1.0, 1.0, 0.35 + 0.65 * pulse)
        if hasattr(self, "start_button") and not self.start_button.isEmpty():
            if self.start_button["state"] == DGG.NORMAL:
                pulse = 0.92 + 0.08 * math.sin(t * 3.5)
                self.start_button.setColorScale(pulse, pulse, pulse, 1.0)
            else:
                self.start_button.clearColorScale()
        if self.preview and hasattr(self.preview, "glow_ring") and not self.preview.glow_ring.isEmpty():
            ring_pulse = 0.78 + 0.22 * math.sin(t * 2.8)
            self.preview.glow_ring.setColorScale(ring_pulse, ring_pulse, ring_pulse, 1.0)
        if self.preview and hasattr(self.preview, "stage_ticks") and not self.preview.stage_ticks.isEmpty():
            tick_pulse = 0.70 + 0.30 * math.sin(t * 2.8 + 1.0)
            self.preview.stage_ticks.setColorScale(tick_pulse, tick_pulse, tick_pulse, 1.0)
        return task.cont

    def frame(self, parent, x, y, w, h, color=None, border=None, radius=6, border_width=1.2, left_stripe=None):
        return DirectFrame(parent=parent, frameColor=(0, 0, 0, 0), relief=None,
                           geom=rounded_card(w, h, color or self.PANEL, radius, border, border_width, left_stripe),
                           frameSize=(0, w, -h, 0), pos=(x, 0, -y), state=DGG.DISABLED)

    def label(self, parent, text, x, y, size=18, color=None, wrap=None, align=TextNode.ALeft, bold=False):
        options = dict(parent=parent, text=text, text_scale=size, text_fg=color or self.INK,
                       text_align=align, text_font=self.bold_font if bold else self.font,
                       frameColor=(0, 0, 0, 0), state=DGG.DISABLED, relief=None, pos=(x, 0, -y))
        if wrap:
            options["text_wordwrap"] = wrap / size
        return DirectLabel(**options)

    def button(self, parent, text, x, y, w, command, arg=None, h=46, primary=False, size=16,
               align=TextNode.ALeft, border=None, left_stripe=None):
        text_pos = (w / 2, -h * .64) if align == TextNode.ACenter else (16, -h * .64)
        button = DirectButton(
            parent=parent, text=text, text_font=self.bold_font if primary else self.font, text_scale=size,
            text_align=align, text_pos=text_pos,
            text_fg=(1, 1, 1, 1) if primary else self.INK,
            frameColor=(0, 0, 0, 0),
            frameSize=(0, w, -h, 0), relief=None, pos=(x, 0, -y),
            rolloverSound=None, clickSound=None, pressEffect=False,
            command=self.enqueue, extraArgs=[command, arg],
        )
        button.setPythonTag("race_size", (w, h))
        self._paint(button, self.LAUNCH_RED if primary else self.BUTTON, border=border, left_stripe=left_stripe)
        return button

    def _paint(self, button, color, border=None, left_stripe=None, border_width=1.2, stripe_width=4.0):
        w, h = button.getPythonTag("race_size")
        key = (w, h, tuple(color), tuple(border or self.BORDER),
               tuple(left_stripe) if left_stripe else None, border_width, stripe_width)
        if button.getPythonTag("race_paint") == key:
            return
        if key not in self._geometry_cache:
            self._geometry_cache[key] = tuple(
                rounded_card(w, h, c, 6, border or self.BORDER, border_width, left_stripe, stripe_width)
                for c in (color, tint(color, -.02), tint(color, .04), (.055, .070, .09, 1))
            )
        button["geom"] = self._geometry_cache[key]
        button.setPythonTag("race_paint", key)

    def _build_garage(self):
        g = self.groups["garage"]
        # Background full-screen backdrop
        self.frame(g, -2000, -2000, 5280, 4800, (*self.BG[:3],1.), radius=0)
        # Top illuminated racing red strip
        self.frame(g, 0, 0, 1280, 2, self.LAUNCH_RED, radius=0)

        # Header branding - sleek racing header
        self.label(g, "XETUHANH", 40, 42, 17, self.INK, bold=True)
        self.label(g, "/   AUTONOMOUS RACING BENCHMARK", 158, 42, 11, self.MUTED, bold=True)

        # Simulation status telemetry top-right
        self.live_badge_dot = self.frame(g, 978, 38, 8, 8, self.SUCCESS, radius=4)
        self.label(g, "MÔ PHỎNG VẬT LÝ THỜI GIAN THỰC", 994, 45, 11, self.MUTED, bold=True)

        # Main Title & Subtitle
        self.label(g, "ĐƯỜNG ĐUA THUẬT TOÁN", 40, 84, 28, bold=True)
        self.mode_hint = self.label(g, "Chọn đội hình AI · Chọn cấp độ đường · Trải nghiệm mô phỏng thời gian thực.",
                                    42, 114, 13, self.MUTED, wrap=1190)

        # Separator line
        self.frame(g, 40, 134, 1200, 1, (.14, .16, .20, 0.8), radius=0)

        # Mode Selector: Unified Segmented Container
        self.mode_container = self.frame(g, 40, 142, 916, 44, color=(.055, .062, .075, .95), border=self.BORDER, radius=6)
        self.mode_single = self.button(g, "Chạy đơn (1 AI)", 44, 145, 224, "mode", "single", h=38, size=13, align=TextNode.ACenter)
        self.mode_compare = self.button(g, "Đua đồng thời (2–5 AI)", 272, 145, 224, "mode", "compare", h=38, size=13, align=TextNode.ACenter)

        # ----------------------------------------------------
        # Panel: ĐỘI HÌNH XUẤT PHÁT (Left column)
        # ----------------------------------------------------
        self.frame(g, 40, 202, 424, 508, color=self.PANEL, border=self.BORDER, radius=6)
        self.label(g, "ĐỘI HÌNH XUẤT PHÁT", 58, 226, 12, self.MUTED, bold=True)
        self.selected = self.label(g, "", 444, 226, 12, self.INK, align=TextNode.ARight, bold=True)

        self.agent_buttons = []
        self.agent_markers = []
        self.agent_titles = []

        for i, (_, name, description) in enumerate(RACE_AGENTS):
            color = RACE_COLORS[i]
            y_pos = 246 + i * 72
            b = self.button(g, "", 52, y_pos, 400, "agent", i, h=66, left_stripe=color)

            # Left number badge
            self.frame(b, 10, 12, 38, 42, tuple(c * .16 for c in color[:3]) + (1,),
                       border=tuple(c * .40 for c in color[:3]) + (1,), radius=4)
            self.label(b, f"#{i+1}", 29, 37, 14, color, align=TextNode.ACenter, bold=True)

            # Algorithm Title
            title = self.label(b, name.replace(" · thử nghiệm", ""), 60, 23, 16, color, bold=True)
            self.agent_titles.append(title)

            # Experimental Pill for MPPI
            if i == 4:
                self.frame(b, 134, 12, 78, 18, (.22, .06, .10, .9), border=(.8, .25, .35, .8), radius=4)
                self.label(b, "THỬ NGHIỆM", 173, 25, 9, (1., .45, .55, 1.), align=TextNode.ACenter, bold=True)

            # Subtitle description
            self.label(b, description, 60, 46, 11, self.MUTED)

            # Selection indicator: clean square checkbox
            marker = b.attachNewNode("selection_check")
            self.frame(marker, 360, 21, 24, 24, color, radius=4)
            draw_vector_check(marker, 372, 33, 14, (.02, .04, .06, 1.), thickness=2.2)

            plus = b.attachNewNode("selection_plus")
            self.frame(plus, 360, 21, 24, 24, (.06, .07, .09, 1), border=self.BORDER, radius=4)
            draw_vector_plus(plus, 372, 33, 12, self.MUTED, thickness=1.6)

            self.agent_markers.append((marker, plus))
            self.agent_buttons.append(b)

        # Batch Selection Buttons
        self.select_all = self.button(g, "Chọn cả 5 xe", 52, 616, 194, "all_agents", h=38, size=13, align=TextNode.ACenter)
        self.clear_all = self.button(g, "Bỏ chọn tất cả", 258, 616, 194, "clear_agents", h=38, size=13, align=TextNode.ACenter)

        # Single Mode Agent Info Card
        self.single_agent_info = self.frame(g, 52, 614, 400, 84, color=(.055, .065, .080, .95),
                                            border=self.BORDER, radius=6)
        self.single_agent_spec_title = self.label(self.single_agent_info, "", 14, 22, 11, self.INK, bold=True)
        self.single_agent_spec_detail = self.label(self.single_agent_info, "", 14, 46, 10.5, self.MUTED)

        # ----------------------------------------------------
        # Panel: PHÒNG TRƯNG BÀY XE 3D (Showroom)
        # ----------------------------------------------------
        self.frame(g, 484, 202, 756, 300, color=self.PANEL, border=self.BORDER, radius=6)
        self.preview_frame = DirectFrame(parent=g, frameColor=(0, 0, 0, 0), frameSize=(0, 726, -240, 0),
                                         pos=(500, 0, -220), state=DGG.DISABLED, relief=None)
        self.label(g, "PHÒNG TRƯNG BÀY XE 3D", 504, 226, 11, self.MUTED, bold=True)
        self.showroom_telemetry_label = self.label(g, "AUTONOMOUS TELEMETRY", 1220, 226, 10, self.MUTED, align=TextNode.ARight, bold=True)
        self.preview_name = self.label(g, "01  /  A*", 504, 254, 22, self.INK, bold=True)
        self.preview_category = self.label(g, "Thuật toán tìm kiếm Heuristic tối ưu đường đi", 504, 278, 12, self.MUTED)

        # Technical Specification Bar at bottom of showroom
        self.spec_bar = self.frame(g, 504, 452, 716, 38, color=(.055, .065, .080, .95), border=self.BORDER, radius=6)
        self.preview_spec_mech = self.label(g, "Cơ chế: Heuristic f = g + h", 520, 475, 11, self.INK)
        self.preview_spec_mem = self.label(g, "Bộ nhớ: Trung bình (O(b^d))", 760, 475, 11, self.MUTED)
        self.preview_spec_app = self.label(g, "Đặc trưng: Tối ưu khoảng cách", 990, 475, 11, self.INK)

        # ----------------------------------------------------
        # Panel: THIẾT LẬP ĐƯỜNG ĐUA (Track Setup)
        # ----------------------------------------------------
        self.frame(g, 484, 514, 756, 196, color=self.PANEL, border=self.BORDER, radius=6)
        self.label(g, "THIẾT LẬP ĐƯỜNG ĐUA", 504, 536, 11, self.MUTED, bold=True)
        self.track_setup_button = self.button(g, "Kho 12 Bản đồ Test & Kịch bản ↓", 948, 524, 272,
                                              'track_open', h=28, size=12, align=TextNode.ACenter)
        self._paint(self.track_setup_button, self.BUTTON, border=self.BORDER_BRIGHT)
        self.track_setup_button['text_fg'] = self.INK

        self.road_buttons, self.road_titles, self.road_details, self.road_tags, self.road_badges = [], [], [], [], []
        ROAD_DIFFICULTY_PILLS = (
            ("CẤP 1", (.12, .88, .60, 1.), (.03, .16, .10, 1.)),
            ("CẤP 2", (1., .70, .20, 1.), (.18, .12, .03, 1.)),
            ("CẤP 3", (1., .32, .45, 1.), (.20, .05, .08, 1.)),
            ("VN", (1., .82, .28, 1.), (.25, .045, .065, 1.)),
        )
        for i in range(4):
            level_name, desc = ROAD_DESCRIPTIONS[i]
            pill_text, pill_color, pill_bg = ROAD_DIFFICULTY_PILLS[i]
            b = self.button(g, "", 504 + i * 180, 558, 172, "road", i, h=76)
            self.road_titles.append(self.label(b, ROAD_NAMES[i], 12, 26, 14, bold=True))
            badge_frame = self.frame(b, 116, 10, 44, 20, pill_bg, border=tuple(c * .4 for c in pill_color[:3]) + (1,), radius=4)
            self.label(badge_frame, pill_text, 22, 14, 9, pill_color, align=TextNode.ACenter, bold=True)
            self.road_badges.append(badge_frame)
            self.road_details.append(self.label(b, desc, 12, 46, 9.5, self.MUTED))
            tag = self.label(b, "", 12, 64, 9, self.DIMMED)
            self.road_tags.append(tag)
            self.road_buttons.append(b)

        # Bottom Options inside Road Card: Traffic & Speed
        self.traffic_button = self.button(g, "", 504, 644, 350, "traffic", h=44, size=13)
        self.speed_button = self.button(g, "", 866, 644, 350, "speed", h=44, size=13)

        # ----------------------------------------------------
        # Bottom Global Action Bar
        # ----------------------------------------------------
        self.label(g, "[Enter] Xuất phát     [1]–[5] Chọn xe     [P] Tạm dừng", 42, 736, 13, self.INK, bold=True)
        self.label(g, "Mô phỏng MetaDrive 0.4.3  ·  Telemetry thời gian thực  ·  Vật lý va chạm", 42, 760, 11, self.MUTED)
        self.status = self.label(g, "", 484, 742, 13, self.MUTED, wrap=410, bold=True)
        self.start_button = self.button(g, "XUẤT PHÁT  →", 906, 722, 236, "start", h=54, primary=True, size=17, align=TextNode.ACenter)
        self.button(g, "Thoát", 1152, 722, 88, "quit", h=54, size=14, align=TextNode.ACenter)
        self.button(g, "Thoát", 1152, 722, 88, "quit", h=54, size=14, align=TextNode.ACenter)

    def attach_preview(self, vehicle):
        from src.visualization.race_showroom import GaragePreview
        self.preview = GaragePreview(self.engine, vehicle)
        self.preview_frame["image"] = self.preview.texture
        self.preview_frame["image_scale"] = (363, 1, 125)
        self.preview_frame["image_pos"] = (363, 0, -125)
        self.preview_frame.setTransparency(TransparencyAttrib.MAlpha)

    def _build_running(self):
        g = self.groups["running"]
        # Keep the original minimap at the top left fully visible.
        self.frame(g, 280, 16, 964, 58, self.BG, self.BORDER, radius=12)
        self.frame(g, 292, 24, 134, 32, (.035, .14, .12, .95), border=(.20, .75, .60, .8), radius=16)
        self.label(g, "● LIVE ĐUA XE", 359, 44, 11, self.ACCENT, align=TextNode.ACenter, bold=True)
        self.run_title = self.label(g, "", 442, 46, 15, bold=True)
        self.focus_button = self.button(g, "Đổi xe theo dõi · [V]", 996, 22, 234, "focus", h=44, size=13, align=TextNode.ACenter)

        # Live Standings Panel on right
        self.standings = g.attachNewNode("live_standings")
        self.frame(self.standings, 976, 86, 268, 446, self.BG, self.BORDER, radius=14)
        self.label(self.standings, "BẢNG XẾP HẠNG TRỰC TIẾP", 992, 112, 12, self.MUTED, bold=True)
        self.label(self.standings, "Bấm xe / phím 1–5 để theo dõi", 992, 132, 10, self.DIMMED)
        self.standing_rows = []
        for i in range(5):
            button = self.button(self.standings, "", 988, 144 + i * 48, 244,
                                 "number", i, h=44, size=12, left_stripe=RACE_COLORS[i])
            self.standing_rows.append(button)

        # Bottom Control Toolbar
        self.frame(g, 36, 706, 1208, 80, self.BG, self.BORDER, radius=14)
        self.pause_button = self.button(g, "Tạm dừng [P]", 398, 714, 138, "pause", h=40, size=13, align=TextNode.ACenter)
        self.restart_button = self.button(g, "Chạy lại [R]", 544, 714, 138, "restart", h=40, size=13, align=TextNode.ACenter)
        self.garage_button = self.button(g, "Về Garage [G]", 690, 714, 144, "garage", h=40, size=13, align=TextNode.ACenter)
        self.details_button = self.button(g, "Dữ liệu AI [I]", 842, 714, 140, "details", h=40, size=13, align=TextNode.ACenter)
        self.camera_button = self.button(g, "Đổi Camera [C]", 990, 714, 144, "camera", h=40, size=13, align=TextNode.ACenter)
        self.run_hint = self.label(g, "Chuột trái: Xoay camera  •  [C] Góc nhìn  •  [V]/[1–5] Theo dõi  •  [I] Dữ liệu AI  •  [P] Dừng  •  [R] Chạy lại", 640, 766, 11, self.MUTED, align=TextNode.ACenter)

    def update_standings(self, racers, env):
        self.standings.show() if len(racers) > 1 else self.standings.hide()
        for index, button in enumerate(self.standing_rows):
            key = f"racer_{index}"
            if key not in racers:
                button.hide()
                continue
            button.show()
            racer = racers[key]
            progress = f"{racer.route * 100:.1f}%" if racer.route is not None else "—"
            status = (OUTCOME_LABELS.get(racer.summary["reason"], racer.summary["reason"])
                      if racer.summary else f"{env.agents[key].speed_km_h:.0f} km/h")
            prefix = "► " if env.focus_id == key else "   "
            color = RACE_COLORS[index]
            name = RACE_AGENTS[index][1].replace(" · thử nghiệm", "")
            button["text"] = f"{prefix}{index + 1:02d}  {name}\n      Tiến độ: {progress}  ·  {status}"
            button["text_scale"] = 12
            button["text_fg"] = color
            self._paint(button, (.055, .15, .16, 1) if env.focus_id == key else self.BUTTON,
                        color if env.focus_id == key else self.BORDER,
                        left_stripe=color)

    def _build_results(self):
        g = self.groups["results"]
        self.frame(g, 260, 80, 760, 600, self.BG, self.BORDER, radius=16)
        self.label(g, "KẾT QUẢ LƯỢT CHẠY ĐƠN", 300, 126, 13, self.ACCENT, bold=True)
        self.result_title = self.label(g, "", 300, 168, 26, bold=True, wrap=680)
        # Detailed telemetry panel
        self.frame(g, 290, 208, 700, 240, (.032, .044, .065, .9), border=self.BORDER, radius=12)
        self.result_body = self.label(g, "", 315, 246, 17, wrap=650)
        self.result_error = self.label(g, "", 300, 480, 13, (.99, .73, .40, 1), wrap=680)
        self.result_action = self.button(g, "Chạy lại [R]", 300, 540, 200, "result_action", primary=True, h=48, size=15, align=TextNode.ACenter)
        self.button(g, "Đổi thuật toán [G]", 520, 540, 220, "garage", h=48, size=15, align=TextNode.ACenter)
        self.button(g, "Thoát [Esc]", 760, 540, 140, "quit", h=48, size=15, align=TextNode.ACenter)
        self.result_hint = self.label(g, "Phím tắt: [Enter] hoặc [R] chạy lại  •  [G] về garage  •  [Esc] thoát", 640, 626, 12, self.MUTED, align=TextNode.ACenter)

    def _build_comparison(self):
        g = self.groups["comparison"]
        self.frame(g, 25, 75, 1230, 650, self.BG, self.BORDER, radius=16)
        self.comparison_title = self.label(g, "BẢNG SO SÁNH KẾT QUẢ ĐUA ĐỒNG THỜI", 55, 120, 26, self.ACCENT, bold=True)
        self.comparison_context = self.label(g, "", 55, 158, 14, self.MUTED, wrap=1160)
        columns = ((60, "Thuật toán"), (290, "Trạng thái"), (560, "Tiến độ đường"),
                   (710, "TG mô phỏng"), (880, "Va chạm / Lệch làn"), (1080, "Độ trễ TB"))
        for x, text in columns:
            self.label(g, text, x, 206, 14, self.MUTED, bold=True)
        self.comparison_rows = []
        for i in range(5):
            y = 250 + i * 58
            self.frame(g, 45, y - 26, 1180, 52, (.045, .065, .095, 1) if i % 2 == 0 else self.PANEL,
                       border=self.BORDER, radius=10, left_stripe=RACE_COLORS[i])
            self.comparison_rows.append([self.label(g, "", x, y, 16,
                                                   wrap=(255 if x == 290 else None), bold=(x == 60)) for x, _ in columns])
        self.label(g, "Mô phỏng đồng thời trên cùng đường đua · Các xe xuất phát cách nhau an toàn và tương tác vật lý trực tiếp", 55, 595, 14, self.MUTED)
        self.button(g, "Chạy lại cả nhóm [R]", 55, 636, 250, "restart", primary=True, h=50, size=15, align=TextNode.ACenter)
        self.button(g, "Về Garage [G]", 325, 636, 200, "garage", h=50, size=15, align=TextNode.ACenter)
        self.button(g, "Thoát [Esc]", 545, 636, 150, "quit", h=50, size=15, align=TextNode.ACenter)

    def update_comparison(self, comparison):
        self.comparison_title["text"] = f"BẢNG SO SÁNH KẾT QUẢ ĐUA ĐỒNG THỜI · {len(comparison.results)}/{len(comparison.indices)} XE HOÀN THÀNH"
        case = comparison.case
        self.comparison_context["text"] = f"Đường {case['sequence']} · Seed môi trường: {case['environment_seed']} · Giao thông NPC: {case['traffic_density']:g} · Phân vùng TRAIN"
        names = {aid: name for aid, name, _ in RACE_AGENTS}
        results = {row["agent_id"]: row for row in comparison.results}
        for i, labels in enumerate(self.comparison_rows):
            result = results.get(RACE_AGENTS[comparison.indices[i]][0]) if i < len(comparison.indices) else None
            if result is None:
                values = ("",) * 6
            else:
                route, latency, flags = result["route_completion"], result["mean_act_latency_ms"], result["flags"]
                crashes = any(value for key, value in flags.items() if key.startswith("crash_"))
                raw_reason = result["reason"]
                reason_text = OUTCOME_LABELS.get(raw_reason, raw_reason)
                values = (
                    names[result["agent_id"]].replace(" · thử nghiệm", ""),
                    reason_text,
                    f"{route * 100:.1f}%" if route is not None else "—",
                    f"{result['simulation_time_s']:.1f} s" if result['simulation_time_s'] is not None else "—",
                    (f"{'Có' if crashes else 'Không'} / {'Có' if flags.get('out_of_road') else 'Không'}"
                     if result['simulation_time_s'] is not None else "— / —"),
                    f"{latency:.1f} ms" if latency is not None else "—",
                )
            for col_idx, (label, text) in enumerate(zip(labels, values)):
                label["text"] = text
                if col_idx == 0 and result is not None:
                    label["text_fg"] = RACE_COLORS[comparison.indices[i]]
                elif col_idx == 1 and result is not None:
                    if result["reason"] == "SUCCESS":
                        label["text_fg"] = self.SUCCESS
                    elif "CRASH" in result["reason"]:
                        label["text_fg"] = self.DANGER
                    elif result["reason"] == "OUT_OF_ROAD":
                        label["text_fg"] = self.GOLD
                    else:
                        label["text_fg"] = self.INK
                else:
                    label["text_fg"] = self.INK

    def _build_paused(self):
        g = self.groups["paused"]
        self.frame(g, -2000, -2000, 5280, 4800, (.010, .015, .025, .75), radius=0)
        self.frame(g, 360, 210, 560, 330, self.BG, self.BORDER, radius=16)
        self.label(g, "● ĐÃ TẠM DỪNG MÔ PHỎNG", 400, 258, 22, self.ACCENT, bold=True)
        self.label(g, "Mô phỏng đang tạm dừng. Tất cả các xe giữ nguyên vị trí vật lý.", 400, 296, 14, self.MUTED)
        self.button(g, "Tiếp tục đua [P]", 395, 344, 160, "pause", primary=True, h=48, size=15, align=TextNode.ACenter)
        self.button(g, "Chạy lại [R]", 568, 344, 150, "restart", h=48, size=15, align=TextNode.ACenter)
        self.button(g, "Về Garage [G]", 732, 344, 150, "garage", h=48, size=15, align=TextNode.ACenter)
        self.label(g, "Phím tắt: [P] hoặc [Esc] tiếp tục  •  [R] chạy lại  •  [G] về garage", 640, 428, 12, self.MUTED, align=TextNode.ACenter)

    def _build_countdown(self):
        g = self.groups["countdown"]
        self.frame(g, -2000, -2000, 5280, 4800, (.005, .010, .018, .45), radius=0)
        self.frame(g, 530, 240, 220, 220, (.02, .035, .055, .90), border=self.ACCENT, radius=110, border_width=2.5)
        self.label(g, "CHUẨN BỊ XUẤT PHÁT", 640, 276, 11, self.ACCENT, align=TextNode.ACenter, bold=True)
        self.countdown = self.label(g, "3", 640, 366, 72, self.INK, align=TextNode.ACenter, bold=True)

    def layout(self):
        size = (self.engine.win.getXSize(), self.engine.win.getYSize())
        if size != self._size:
            self._size = size
            scale = min(size[0] / 1280., size[1] / 800.)
            self.root.setScale(scale)
            self.root.setPos((size[0] - 1280 * scale) / 2, 0, -(size[1] - 800 * scale) / 2)

    def show(self, state):
        for name, group in self.groups.items():
            visible = name == state or (name == "running" and state in ("paused", "countdown"))
            group.show() if visible else group.hide()
        if self.preview is not None:
            self.preview.set_active(state == "garage")

    def update_selection(self, selection, roads):
        compare = selection.mode == "compare"
        for button, active in ((self.mode_single, not compare), (self.mode_compare, compare)):
            self._paint(button, (.18, .21, .27, 1.) if active else self.BUTTON,
                        self.BORDER_BRIGHT if active else self.BORDER,
                        border_width=1.5 if active else 1.0)
            button["text_fg"] = self.INK if active else self.MUTED

        self.mode_hint["text"] = (
            "Chế độ Đua đồng thời: 2–5 thuật toán AI cùng xuất phát trên cùng đường đua, tương tác và va chạm trực tiếp."
            if compare else
            "Chế độ Chạy đơn: Xem 1 thuật toán AI lái xe. Phân tích chi tiết quỹ đạo, góc lái và độ trễ."
        )

        changed = sorted(set(selection.chosen_indices) ^ set(self._previous_indices))
        if not compare:
            self._preview_index = selection.agent_index
        elif changed:
            self._preview_index = changed[0]
        self._previous_indices = selection.chosen_indices

        for i, button in enumerate(self.agent_buttons):
            selected = i in selection.chosen_indices
            is_preview = (i == self._preview_index)
            color = RACE_COLORS[i]
            if is_preview:
                bg = tuple(c * .12 + .06 for c in color[:3]) + (1,)
                border = tuple(c * .85 for c in color[:3]) + (1,)
                bw = 1.8
                sw = 5.0
            elif selected:
                bg = tuple(c * .06 + .04 for c in color[:3]) + (1,)
                border = tuple(c * .45 for c in color[:3]) + (1,)
                bw = 1.2
                sw = 4.0
            else:
                bg = self.BUTTON
                border = self.BORDER
                bw = 1.0
                sw = 3.0
            self._paint(button, bg, border, left_stripe=color, border_width=bw, stripe_width=sw)
            marker, plus = self.agent_markers[i]
            if selected:
                marker.show()
                plus.hide()
            else:
                marker.hide()
                plus.show()
            raw_title = RACE_AGENTS[i][1].replace(" · thử nghiệm", "")
            self.agent_titles[i]["text"] = ("→ " if is_preview else "") + raw_title
            self.agent_titles[i]["text_fg"] = self.INK if (selected or is_preview) else self.MUTED

        for i, button in enumerate(self.road_buttons):
            case = roads[i][0]
            selected = selection.test_case is None and i == selection.road_index
            self.road_titles[i]["text_fg"] = self.INK if selected else self.MUTED
            reference = case.case_id.startswith('original/')
            self.road_tags[i]["text"] = case.sequence + ('   /   MAP GỐC' if reference else '   /   '+case.split)
            if case.sequence == 'VN':
                self.road_tags[i]['text'] = 'OSM · ĐUA DÀI · 80 km/h'
            self.road_titles[i]['text'] = ROAD_NAMES[i]
            self._paint(button, (.18, .21, .27, 1.) if selected else self.BUTTON,
                        self.BORDER_BRIGHT if selected else self.BORDER, border_width=1.5 if selected else 1.0)
            if i == 3:
                self.road_titles[i]['text_fg'] = self.GOLD if selected else (.78,.65,.40,1)
                self._paint(button,(.22,.065,.08,1) if selected else (.12,.075,.085,1),
                            self.GOLD if selected else (.40,.19,.15,1),border_width=1.5)

        self.selected["text"] = f"ĐÃ CHỌN: {len(selection.chosen_indices):02d} / 05 XE" if compare else "CHẾ ĐỘ: 01 XE"
        for button in (self.select_all, self.clear_all):
            button.show() if compare else button.hide()
        self.single_agent_info.hide() if compare else self.single_agent_info.show()

        ready = not compare or len(selection.chosen_indices) >= 2
        self.start_button["state"] = DGG.NORMAL if ready else DGG.DISABLED
        if ready:
            self._paint(self.start_button, self.LAUNCH_RED, border=(1., .4, .4, 1.), border_width=1.5)
            self.start_button["text"] = "XUẤT PHÁT  →"
            self.start_button["text_fg"] = (1, 1, 1, 1)
            self.status["text"] = f"[SẴN SÀNG]  {len(selection.chosen_indices):02d} xe xuất phát  ·  Đường {ROAD_NAMES[selection.road_index]} ({roads[selection.road_index][0].sequence})"
            self.status["text_fg"] = self.SUCCESS
        else:
            self._paint(self.start_button, self.BUTTON, self.BORDER, border_width=1.2)
            self.start_button["text"] = "CHỌN ÍT NHẤT 2 XE"
            self.start_button["text_fg"] = self.DIMMED
            self.status["text"] = "Chọn thêm ít nhất 2 thuật toán để bắt đầu đua đồng thời."
            self.status["text_fg"] = self.GOLD

        if selection.traffic:
            self.traffic_button["text"] = "Giao thông NPC:   BẬT   (Có xe cùng lưu thông)"
            self.traffic_button["text_fg"] = self.INK
            self._paint(self.traffic_button, (.14, .17, .22, 1.), self.BORDER_BRIGHT)
        else:
            self.traffic_button["text"] = "Giao thông NPC:   TẮT   (Đường đua trống)"
            self.traffic_button["text_fg"] = self.MUTED
            self._paint(self.traffic_button, self.BUTTON, self.BORDER)

        self.speed_button["text"] = f"Tốc độ mô phỏng:   {selection.speed:g}×   [Bấm đổi: 1× · 2× · 5×]"
        self.speed_button["text_fg"] = self.INK

        spec = AGENT_SPECS[self._preview_index]
        self.preview_name["text"] = f"{self._preview_index+1:02d}  /  {spec['title']}"
        self.preview_name["text_fg"] = RACE_COLORS[self._preview_index]
        self.preview_category["text"] = spec["category"]
        self.preview_spec_mech["text"] = f"Cơ chế: {spec['mechanism']}"
        self.preview_spec_mem["text"] = f"Bộ nhớ: {spec['memory']}"
        self.preview_spec_app["text"] = f"Đặc trưng: {spec['strength']}"

        if not compare:
            self.single_agent_spec_title["text"] = f"THÔNG SỐ: {spec['title']}  •  {spec['strength']}"
            self.single_agent_spec_title["text_fg"] = RACE_COLORS[self._preview_index]
            self.single_agent_spec_detail["text"] = f"Cơ chế: {spec['mechanism']}   |   Bộ nhớ: {spec['memory']}"

        if self.preview is not None:
            self.preview.select(self._preview_index)

    def destroy(self):
        if hasattr(self, "_anim_task_name"):
            self.engine.taskMgr.remove(self._anim_task_name)
        if self.preview is not None:
            self.preview.destroy()
            self.preview = None
        self.root.removeNode()
        self._geometry_cache.clear()
