"""Read-only, paged AgentInput/AgentDecision inspector in the native window.

Input time is BEFORE the action, unlike the racing HUD's post-step telemetry.
Only one copied input/decision and four compact history rows are retained.
"""

from collections import deque
import json

import numpy as np
from direct.showbase.DirectObject import DirectObject
from panda3d.core import CardMaker, TextNode, TransparencyAttrib


class AgentIoOverlay:
    PAGE_NAMES = (
        "Summary", "Ego + navigation", "LiDAR 0-79", "LiDAR 80-159",
        "LiDAR 160-239", "Traffic", "Route waypoints", "Decision",
    )

    def __init__(self, engine):
        self.engine = engine
        self.snapshot = None
        self.history = deque(maxlen=4)
        self.page = 0
        self._destroyed = False
        self.root = engine.aspect2d.attachNewNode("agent_io_overlay")
        self.root.setLightOff()
        self.root.setShaderOff(1)
        self.root.setFogOff(1)
        self.root.setDepthTest(False)
        self.root.setDepthWrite(False)
        self.root.setBin("fixed", 240)
        self._events = DirectObject()
        try:
            card = CardMaker("agent_io_background")
            card.setFrame(-1.06, 0, -1.21, 0)
            background = self.root.attachNewNode(card.generate())
            background.setColor(0.025, 0.045, 0.070, 0.91)
            background.setTransparency(TransparencyAttrib.MAlpha)
            self.title = TextNode("agent_io_title")
            self.title.setTextColor(0.0, 0.90, 1.0, 1.0)
            title_np = self.root.attachNewNode(self.title)
            title_np.setPos(-1.02, -0.01, -0.055)
            title_np.setScale(0.033)
            self.body = TextNode("agent_io_body")
            self.body.setTextColor(0.91, 0.95, 1.0, 1.0)
            self.body.setWordwrap(37)
            body_np = self.root.attachNewNode(self.body)
            body_np.setPos(-1.02, -0.01, -0.12)
            body_np.setScale(0.027)
            hint = TextNode("agent_io_hint")
            hint.setText("TAB: details    I: hide/show    P: pause")
            hint.setTextColor(0.65, 0.77, 0.87, 1.0)
            hint_np = self.root.attachNewNode(hint)
            hint_np.setPos(-1.02, -0.01, -1.175)
            hint_np.setScale(0.025)
            self._events.accept("i", self.toggle)
            self._events.accept("tab", self.next_page)
            self._layout()
            self._refresh()
        except Exception:
            self.destroy()
            raise

    def _layout(self):
        width = max(1, self.engine.win.getXSize())
        height = max(1, self.engine.win.getYSize())
        aspect = width / height
        self.root.setPos(aspect - 0.035, 0, 0.77)
        # Keep the panel within half the viewport even in narrow windows.
        self.root.setScale(min(1.0, aspect / 1.10))

    def update(self, agent_input, decision, canonical_action, *, agent_id,
               control_dt_s, act_latency_ms=None):
        if self._destroyed:
            return
        self.snapshot = {
            "agent_id": agent_id,
            "step_index": agent_input.step_index,
            "input_time_s": agent_input.step_index * control_dt_s,
            "agent_input": agent_input.to_dict(),
            "agent_decision": decision.to_dict(),
            "canonical_action": canonical_action.to_dict(),
            "act_latency_ms": act_latency_ms,
        }
        action = canonical_action
        self.history.append(
            f"#{agent_input.step_index:04d}  steer {action.steering:+.2f}"
            f"  gas/brake {action.throttle_brake:+.2f}"
        )
        self._layout()
        if not self.root.isHidden():
            self._refresh()

    @staticmethod
    def _numbers(values):
        return " ".join(f"{float(value):.2f}" for value in values)

    def _refresh(self):
        self.title.setText(f"AGENT I/O  |  {self.PAGE_NAMES[self.page]} [{self.page + 1}/8]")
        if self.snapshot is None:
            self.body.setText("Waiting for the first decision...")
            return
        snap = self.snapshot
        inp = snap["agent_input"]
        features = inp["core_observation"]["features"]
        task, traffic = inp["task_context"], inp["traffic_context"]
        dec = snap["agent_decision"]
        diag = dec["diagnostics"] or {}
        action = snap["canonical_action"]
        lines = [snap["agent_id"],
                 f"Step {snap['step_index']} | input t={snap['input_time_s']:.1f}s",
                 "Input sampled BEFORE this action.", ""]
        waypoints = [w for w, valid in zip(task["waypoints"], task["validity_mask"]) if valid]
        actors = [a for a, valid in zip(traffic["actors"], traffic["validity_mask"]) if valid]
        if self.page == 0:
            lidar = np.asarray(features[19:259])
            lines += ["AGENT INPUT  (summary)",
                      f"Core: 259 | speed(norm): {features[3]:.3f}",
                      "Ego[0:9] (normalized):",
                      self._numbers(features[:5]), self._numbers(features[5:9]),
                      f"LiDAR(norm) min/max: {lidar.min():.2f} / {lidar.max():.2f}",
                      f"Traffic: {traffic['active_count']} | waypoints: {len(waypoints)}",
                      f"Lane width: {task['current_lane_width']:.2f} m"]
            if waypoints:
                first = waypoints[0]
                lines.append(f"First WP x/y: {first['relative_x']:.2f}, {first['relative_y']:.2f} m")
            lines += ["", "AGENT DECISION", f"Payload: {dec['action_payload']}",
                      f"Steering: {action['steering']:+.2f}",
                      f"Throttle/brake: {action['throttle_brake']:+.2f}",
                      f"Planner: {diag.get('planner_status', '---')}",
                      f"Expanded nodes: {diag.get('expansions', '---')}"]
            if "local_path_cost" in diag:
                lines.append(f"Local path cost: {diag['local_path_cost']:.3f}")
            if "target_x" in diag and "target_y" in diag:
                lines.append(f"Target x/y: {diag['target_x']:.2f}, {diag['target_y']:.2f} m")
            if snap["act_latency_ms"] is not None:
                lines.append(f"agent.act: {snap['act_latency_ms']:.3f} ms")
            lines += ["", "RECENT DECISIONS", *self.history]
        elif self.page == 1:
            lines += ["EGO: core indices 0..8 (normalized)"]
            lines += [f"ego[{i}]: {v:.6f}" for i, v in enumerate(features[:9])]
            lines += ["", "NAVIGATION: core indices 9..18"]
            lines += [f"nav[{i}]: {v:.6f}" for i, v in enumerate(features[9:19])]
        elif 2 <= self.page <= 4:
            start = (self.page - 2) * 80
            lidar = features[19:259]
            lines += ["LiDAR ray values (normalized 0..1)", ""]
            for i in range(start, start + 80, 5):
                lines.append(f"{i:03d}: " + self._numbers(lidar[i:i + 5]))
        elif self.page == 5:
            lines += [f"Active: {traffic['active_count']} | overflow: {traffic['overflow_count']}",
                      "Ego frame: +x forward, +y left.", ""]
            for i, actor in enumerate(actors):
                lines += [f"Actor {i}: x/y {actor['relative_position_x']:.2f} / {actor['relative_position_y']:.2f} m",
                          f"  vx/vy {actor['relative_velocity_x']:.2f} / {actor['relative_velocity_y']:.2f} m/s",
                          f"  yaw {actor['relative_heading']:.2f} | L/W {actor['length']:.1f}/{actor['width']:.1f}"]
            if not actors:
                lines.append("No actors in the public input.")
        elif self.page == 6:
            lines += [f"Lane width: {task['current_lane_width']:.2f} m",
                      "WP: relative x / y (m), heading (rad)"]
            lines += [f"{i:02d}: {w['relative_x']:.2f} / {w['relative_y']:.2f} / {w['relative_heading']:.2f}"
                      for i, w in enumerate(waypoints)]
            lines.append(f"Route end in lookahead: {task['route_end_within_lookahead']}")
        else:
            payload = np.asarray(dec["action_payload"]).reshape(-1)
            lines += ["ACTION PAYLOAD", self._numbers(payload), "", "CANONICAL ACTION",
                      f"steering: {action['steering']:+.4f}",
                      f"throttle_brake: {action['throttle_brake']:+.4f}", "", "DIAGNOSTICS (bounded preview)"]
            # Other agents may have large diagnostics; bound displayed text.
            preview = json.dumps(diag, indent=2, ensure_ascii=True).splitlines()
            lines += [line[:65] for line in preview[:17]]
            if len(preview) > 17:
                lines.append("...")
        self.body.setText("\n".join(lines))

    def next_page(self):
        if not self._destroyed:
            self.page = (self.page + 1) % len(self.PAGE_NAMES)
            self._refresh()

    def toggle(self):
        if self._destroyed:
            return
        if self.root.isHidden():
            self.root.show()
            self._refresh()
        else:
            self.root.hide()

    def reset(self):
        self.snapshot = None
        self.history.clear()
        self.page = 0
        if not self._destroyed:
            self._refresh()

    def destroy(self):
        if self._destroyed:
            return
        self._destroyed = True
        self._events.ignoreAll()
        self.root.removeNode()
        self.snapshot = None
        self.history.clear()
