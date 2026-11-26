"""Short factual explanations of search diagnostics, plus a native rolling log.

This reports observable planner results and controller outputs. It never invents
intermediate search events or an explanation for why a search failed.
"""

from collections import deque


SEARCH_LABELS = {
    "stage2_bfs": ("BFS", "Duyệt theo tầng; chọn đường ít cạnh nhất"),
    "stage2_dijkstra": ("Dijkstra", "Ưu tiên tổng chi phí đã đi nhỏ nhất"),
    "stage2_astar": ("A*", "Ưu tiên chi phí đã đi + ước lượng đến đích"),
    "stage2_hybrid_astar": ("Hybrid A*", "Tìm trên các cung chuyển động và hướng xe"),
}


def explain_search_decision(agent_id, decision, action):
    """Return a compact Vietnamese explanation using only actual diagnostics."""
    label, strategy = SEARCH_LABELS[agent_id]
    diag = decision.diagnostics or {}
    status = diag.get("planner_status", "UNKNOWN")
    expanded = diag.get("expansions")
    count = f"; duyệt {expanded} nút" if expanded is not None else ""
    if status == "PATH_FOUND":
        result = "Tìm được đường" + count
        if "local_path_cost" in diag:
            result += f"; chi phí {diag['local_path_cost']:.2f}"
    elif status == "NO_PATH":
        result = "Chưa tìm được đường trong lần tìm này" + count
    elif status == "NO_PUBLIC_ROUTE":
        result = "Không có điểm dẫn đường trong input"
    elif status == "AT_LOCAL_GOAL":
        result = "Đã tới cuối đường cục bộ"
    else:
        result = status + count
    target = ""
    if "target_x" in diag and "target_y" in diag:
        target = f"Bám điểm x={diag['target_x']:.1f}, y={diag['target_y']:+.1f} m → "
    pedal = "tăng ga" if action.throttle_brake > 0 else "phanh" if action.throttle_brake < 0 else "thả ga"
    control = f"{target}lái {action.steering:+.2f}; {pedal} ({action.throttle_brake:+.2f})"
    return {"agent": label, "strategy": strategy, "result": result, "control": control,
            "text": f"{label}: {result} | {control}"}


class SearchTraceOverlay:
    """Vietnamese current-decision panel and five sampled log rows below it."""

    def __init__(self, engine, heading):
        from panda3d.core import CardMaker, Filename, TextNode, TransparencyAttrib
        self.engine = engine
        self.history = deque(maxlen=5)
        self.last_history_step = -10
        self.last_status = None
        self.root = engine.aspect2d.attachNewNode("search_decision_trace")
        self.root.setLightOff()
        self.root.setShaderOff(1)
        self.root.setFogOff(1)
        self.root.setDepthTest(False)
        self.root.setDepthWrite(False)
        self.root.setBin("fixed", 245)
        font = engine.loader.loadFont(Filename.fromOsSpecific("C:/Windows/Fonts/segoeui.ttf").getFullpath())

        def panel(name, frame):
            card = CardMaker(name)
            card.setFrame(*frame)
            node = self.root.attachNewNode(card.generate())
            node.setColor(.025, .045, .070, .9)
            node.setTransparency(TransparencyAttrib.MAlpha)

        def text(name, x, y, size, color):
            node = TextNode(name)
            node.setFont(font)
            node.setTextColor(*color)
            path = self.root.attachNewNode(node)
            path.setPos(x, -.02, y)
            path.setScale(size)
            return node

        panel("decision_panel", (-.08, 1.55, .39, .86))
        panel("history_panel", (-1.55, .77, -.98, -.66))
        self.heading = text("trace_heading", -.04, .81, .027, (0, .9, 1, 1))
        self.heading.setWordwrap(58)
        self.current = text("trace_current", -.04, .70, .032, (.95, .97, 1, 1))
        self.current.setWordwrap(48)
        self.history_title = text("history_title", -1.51, -.71, .027, (0, .9, 1, 1))
        self.history_title.setText("LOG QUYẾT ĐỊNH · 5 dòng gần nhất · file lưu đủ mỗi bước")
        self.history_text = text("trace_history", -1.51, -.76, .025, (.84, .89, .94, 1))
        self.heading.setText(heading)
        self.hint = text("trace_keys", -1.51, -.64, .027, (.95, .9, .6, 1))
        self.hint.setText("M: mượt/nhanh  |  1/2/5: tốc độ  |  0: tối đa  |  P: dừng  |  N: case  |  A: thuật toán  |  Q: thoát  |  I: input")
        self._layout()

    def _layout(self):
        aspect = self.engine.win.getXSize() / max(1, self.engine.win.getYSize())
        self.root.setScale(min(1., aspect / 1.60))

    def update(self, step, trace, speed_kmh, route_completion, *, playback_label="1×", observed_speed=0.,
               render_fps=0., view_mode="smooth"):
        self._layout()
        self.current.setText(
            f"{trace['strategy']}\n{trace['result']}\n{trace['control']}\n"
            f"t={step * .1:.1f}s | {speed_kmh:.1f} km/h | tuyến {route_completion:.1%}\n"
            f"{'MƯỢT' if view_mode == 'smooth' else 'NHANH'} | {render_fps:.0f} FPS\n"
            f"Phát {playback_label} | thực tế {observed_speed:.1f}×"
        )
        if step - self.last_history_step >= 10 or trace["result"].split(";")[0] != self.last_status:
            self.last_history_step = step
            self.last_status = trace["result"].split(";")[0]
            # One concise row per simulated second or planner-status transition.
            self.history.append(f"{step * .1:6.1f}s | {trace['text']}")
            self.history_text.setText("\n".join(self.history))

    def destroy(self):
        self.root.removeNode()
