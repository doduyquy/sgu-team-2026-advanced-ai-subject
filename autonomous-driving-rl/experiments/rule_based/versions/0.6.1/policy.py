"""Stage 1 entrypoint coordinating lane, traffic, junction and lane-change rules."""

import math
from typing import Optional

import numpy as np

from src.platform.agent import (
    AgentDecision,
    AgentDescriptor,
    AgentPublicEpisodeContext
)
from src.platform.agent_input import AgentInputV1
from .rule_based_traffic import RouteCorridor, assess_traffic, assess_rear_side_escape
from .rule_based_memory import LocalFrame, ActorTracker
from .rule_based_junction import JunctionController
from .rule_based_lane_change import LaneChangeController

class Stage1RuleBasedAgent:
    def __init__(self, cruise_speed_kmh: float = 45.0):
        if not math.isfinite(cruise_speed_kmh) or not 0 < cruise_speed_kmh <= 80:
            raise ValueError("Cruise speed must be finite and in (0, 80] km/h.")
        # Khoảng nhìn trước tối thiểu (m); dùng HƯỚNG waypoint để nhìn trước cua.
        # Input extractor now preserves the current lane and connected route.
        # Steering still uses heading preview and current lateral feedback.
        self.lookahead_m = 5.0
        # Thêm quãng nhìn trước bằng tốc độ (m/s) * thời gian này (s).
        # Tăng: chuẩn bị cua sớm hơn, nhưng có thể bắt đầu rẽ quá sớm.
        self.lookahead_time_s = 0.5
        # Độ mạnh sửa sai hướng xe so với hướng làn. Tăng dễ gây dao động.
        self.k_angle = 1.2
        # Độ mạnh kéo xe về tâm làn; hiệu chỉnh được giảm khi tốc độ tăng.
        self.k_lateral = 1.5
        # Mức chuẩn bị lái theo hướng cua phía trước; quá cao dễ cắt cua.
        self.k_preview = 0.5
        # Giá trị cộng vào tốc độ (m/s) để sửa lệch tâm không quá gắt khi xe chậm.
        self.speed_softening_mps = 2.0
        # Giới hạn action steering chuẩn hóa trong ±0.5, không phải góc theo độ.
        self.max_steering = 0.5
        # Trần thử nghiệm đường thoáng, không phải giới hạn pháp lý của đường.
        # Tốc độ thực tế còn bị chặn bởi cua, quãng phanh, xe trước và giao lộ.
        self.cruise_speed_kmh = float(cruise_speed_kmh)
        self.comfortable_deceleration_mps2 = 2.5
        self.response_time_s = 0.4
        self.braking_buffer_m = 3.0
        self.junction_speed_kmh = 24.0
        self.following_time_s = 1.8
        self.minimum_following_gap_m = 3.0
        self.emergency_ttc_s = 1.0
        self.traffic_margin_m = 0.25
        self.obstacle_margin_m = 0.35
        self.prediction_dt_s = 0.2
        self.prediction_horizon_s = 4.0
        self.prediction_lateral_uncertainty_mps = 0.15
        self.junction_scan_m = 30.0
        self.junction_conflict_radius_m = 12.0
        self.yield_release_s = 0.5
        self.lidar_range_m = 50.0
        self.overflow_speed_kmh = 27.0
        self.track_match_distance_m = 2.5
        self.track_ttl_s = 0.6
        self.junction_prediction_s = 12.0
        self.junction_gap_s = 3.0
        self.junction_entry_buffer_m = 4.0
        self.junction_exit_buffer_m = 1.0
        self.junction_exit_clearance_m = 6.0
        self.junction_commit_distance_m = 8.0
        self.junction_launch_acceleration_mps2 = 1.3
        self.stationary_actor_acceleration_mps2 = 1.0
        self.junction_queue_speed_mps = 1.0
        self.junction_blocked_extra_buffer_m = 8.0
        self.queue_release_s = 2.0
        self.blocked_after_s = 10.0
        self.route_memory_ttl_s = 1.0
        self.rear_escape_acceleration_mps2 = 1.5
        self.lane_change_min_gap_m = 4.0
        self.lane_change_gap_time_s = 1.8
        self.lane_change_ttc_s = 4.0
        self.lane_change_prediction_s = 4.0
        self.lane_change_accel_mps2 = 1.2
        self.lane_change_min_length_m = 16.0
        self.lane_change_duration_s = 3.5
        self.lane_change_speed_kmh = 24.0
        self.lane_change_start_speed_kmh = 30.0
        self.lane_change_end_buffer_m = 10.0
        self.lane_change_slow_wait_s = 1.0
        self.lane_change_commit_fraction = 0.22
        self.lane_change_settle_s = 0.5
        self.lane_change_cooldown_s = 6.0
        self.lane_change_overtake_junction_buffer_m = 50.0
        self._control_dt_s = 0.1
        self._yield_timer_s = 0.0
        self._frame = LocalFrame()
        self._tracker = ActorTracker()
        self._junction = JunctionController()
        self._lane_change = LaneChangeController()
        self._step_seen = False
        # Tốc độ thấp khi cần phục hồi vị trí trong làn (km/h).
        self.recovery_speed_kmh = 8.0
        # Ngưỡng gia tốc ngang để giảm tốc theo độ cong (m/s²).
        # Giảm: vào cua chậm hơn; tăng: cua nhanh hơn, tăng nguy cơ chạm vạch.
        self.max_lateral_accel_mps2 = 1.5
        # Ga/phanh = hệ số này * sai số tốc độ (km/h), sau đó giới hạn [-1, 1].
        self.k_speed = 0.12
        # Giới hạn ga dương; phanh vẫn có thể tới -1 khi quá tốc độ.
        self.max_throttle = 0.55
        # Khoảng dự phòng từ thân xe tới mép làn/biên đường (m).
        # Đây là ước lượng hình học; không phải khoảng cách đo tới từng vạch.
        self.safety_margin_m = 0.3

        # Hệ số giải mã của cấu hình MetaDrive 0.4.3 đang dùng, không đọc live env.
        # Nếu đổi loại xe/map/sensor, phải xác minh lại các hằng số này.
        self.observation_max_speed_kmh = 80.0
        self.observation_max_lane_width_m = 4.5
        self.observation_road_scale_m = 18.0  # (MAX_LANE_NUM + 1) * MAX_LANE_WIDTH
        self.vehicle_width_m = 1.852
        self.vehicle_length_m = 4.515

        # Metadata để platform nhận diện và gọi agent; không phải tham số lái.
        self._descriptor = AgentDescriptor(
            # ID phải khớp với agent_id đăng ký trong launcher/registry.py.
            agent_id="stage1_rule_based",
            # Phiên bản giúp phân biệt các lần thay đổi thuật toán/tham số.
            agent_version="0.6.1",
            # Nhận đủ observation, thông tin xe xung quanh và tuyến đường.
            input_profile_id="STATE_DECISION_V1",
            # Action là [steering, throttle_brake], mỗi thành phần trong [-1, 1].
            action_adapter_id="continuous_box2_v1",
            # Cùng input và cấu hình sẽ cho cùng action, không bốc thăm.
            inference_stochasticity="deterministic",
            stateful_within_episode=True,
            # Nhóm phương pháp: quyết định bằng luật/công thức viết sẵn.
            method_family="RULE_BASED",
        )

    @property
    def descriptor(self) -> AgentDescriptor:
        return self._descriptor

    def reset(
            self,
            public_context: AgentPublicEpisodeContext,
            agent_seed: Optional[int] = None,
    ) -> None:
        self._control_dt_s = public_context.control_dt_s
        self._yield_timer_s = 0.0
        self._frame = LocalFrame()
        self._tracker = ActorTracker()
        self._junction = JunctionController()
        self._lane_change = LaneChangeController()
        self._step_seen = False

    def export_config(self):
        """All public numeric policy parameters, for exact experiment records."""
        return {name: value for name, value in vars(self).items()
                if not name.startswith("_") and isinstance(value, (float, int, str, bool))}

    def act(self, agent_input: AgentInputV1) -> AgentDecision:
        # agent_input là dữ liệu quan sát công khai ở bước điều khiển hiện tại.
        task = agent_input.task_context

        # Giải mã state của cấu hình không bật side/lane-line lasers:
        # [3] = (speed_kmh + 1) / (max_speed_kmh + 1).
        # [8] = 0.5 + lateral / MAX_LANE_WIDTH; lateral DƯƠNG là lệch PHẢI.
        # [2] = 0.5 + sin(hướng làn - hướng xe)/2.
        # arcsin chỉ giải mã đúng nhánh khi xe gần cùng hướng làn (<90 độ).
        obs = agent_input.core_observation.features
        speed_kmh = max(0.0, float(obs[3]) * (self.observation_max_speed_kmh + 1.0) - 1.0)
        speed_mps = speed_kmh / 3.6
        lateral_m = (float(obs[8]) - 0.5) * self.observation_max_lane_width_m
        heading_error = math.asin(float(np.clip(2.0 * float(obs[2]) - 1.0, -1.0, 1.0)))
        if agent_input.ego_motion is not None:
            heading_error = agent_input.ego_motion.lane_heading_error_rad
        measured_lane_heading, measured_lateral = heading_error, lateral_m
        self._frame.update(agent_input.ego_motion, speed_mps, self._control_dt_s, first=not self._step_seen)
        self._step_seen = True
        tracks = self._tracker.update(agent_input, self._frame, speed_mps, self._control_dt_s, self)

        # Chỉ xét waypoint có dữ liệu hợp lệ và nằm phía trước xe (+x).
        route_candidates = [
            wp
            for wp, valid in zip(task.waypoints, task.validity_mask)
            if valid
        ]
        route_candidates = self._junction.select_waypoints(route_candidates, self._frame, self._control_dt_s, self)
        lane_plan = self._lane_change.update(agent_input, route_candidates, heading_error, lateral_m,
                                             tracks, self._frame, speed_mps, self, self._junction.locked)
        if lane_plan.waypoints:
            route_candidates = lane_plan.waypoints
            heading_error, lateral_m = lane_plan.heading, lane_plan.lateral
        candidates = [wp for wp in route_candidates if wp.relative_x > 0.0]

        if not candidates:
            # Không có điểm mục tiêu phía trước: giữ lái thẳng và phanh 0.5.
            # Đây là luật dự phòng; không có nghĩa là đã đến đích.
            return AgentDecision(
                action_payload=[0.0, -0.5],
                diagnostics={"rule": "no_forward_waypoints", "behavior_state": "DEGRADED_STOP",
                             "tracked_actor_count": len(tracks), **self._junction.diagnostics()},
            )

        lookahead = min(12.0, self.lookahead_m + self.lookahead_time_s * speed_mps)
        target = min(
            candidates,
            key=lambda wp: abs(
                wp.lookahead_distance_m - lookahead
            ),
        )

        # Dùng hướng làn ở waypoint thay cho góc tới vị trí waypoint.
        # Đổi hướng giữa các điểm phía trước cho biết cua sắp tới.
        preview_delta = self._wrap_angle(target.relative_heading - heading_error)
        # Đã đo trong simulator: steering DƯƠNG tăng heading, quay sang TRÁI
        # trong hệ ego +y trái. Công thức âm ở bản trước làm sai lệch lớn thêm.
        # Lệch phải (lateral dương) -> sửa sang trái; lệch trái -> sửa sang phải.
        lateral_correction = math.atan2(
            self.k_lateral * lateral_m, speed_mps + self.speed_softening_mps
        )
        steering = float(np.clip(
            self.k_angle * heading_error + lateral_correction + self.k_preview * preview_delta,
            -self.max_steering,
            self.max_steering,
        ))

        # Mỗi luật tạo một trần tốc độ. Chọn trần nhỏ nhất, có tính quãng
        # phanh tới cua/giao lộ thay vì ép xe chạy chậm trên cả đoạn thẳng.
        speed_limits = {"cruise": self.cruise_speed_kmh}
        if lane_plan.speed_cap is not None:
            speed_limits["lane_change"] = lane_plan.speed_cap
        route_samples = [(0.0, heading_error)] + [
            (wp.lookahead_distance_m, wp.relative_heading) for wp in candidates
        ]
        curve_limit = self.cruise_speed_kmh
        for (distance, heading), (next_distance, next_heading) in zip(
            route_samples, route_samples[1:]
        ):
            spacing = next_distance - distance
            if spacing <= 0:
                continue
            local_curvature = abs(self._wrap_angle(next_heading - heading)) / spacing
            if local_curvature <= 1e-4:
                continue
            corner_speed_mps = math.sqrt(self.max_lateral_accel_mps2 / local_curvature)
            curve_limit = min(curve_limit, self._approach_speed_kmh(
                corner_speed_mps, distance, speed_mps
            ))
        speed_limits["curve"] = curve_limit

        # Không chạy nhanh hơn khả năng dừng trong phần tuyến còn quan sát được.
        visible_distance = max(wp.lookahead_distance_m for wp in candidates)
        speed_limits["visibility"] = self._approach_speed_kmh(0.0, visible_distance, speed_mps)
        if task.decision_point is not None:
            speed_limits["junction"] = self._approach_speed_kmh(
                self.junction_speed_kmh / 3.6, task.decision_point.distance_m, speed_mps
            )

        corridor = RouteCorridor(route_candidates, heading_error, lateral_m)
        junction_mode = task.route_passage is not None or self._junction.locked
        traffic = assess_traffic(agent_input, corridor, speed_mps, self, tracks, junction_mode)
        junction_limit, junction_diagnostics = self._junction.update(
            task, corridor, tracks, self._frame, speed_mps, self._control_dt_s, self,
            agent_input.traffic_context.motion_paths_overflow_count)
        if junction_limit is not None:
            speed_limits["junction_phase"] = junction_limit
        if traffic.speed_limit_kmh is not None:
            speed_limits["front_vehicle"] = traffic.speed_limit_kmh
        if traffic.obstacle_gap_m is not None:
            speed_limits["lidar_obstacle"] = self._approach_speed_kmh(
                0.0, traffic.obstacle_gap_m, speed_mps)
        if traffic.conflict_distance_m is not None:
            self._yield_timer_s = self.yield_release_s
            conflict_distance = traffic.conflict_distance_m
            if self._junction.passage_id is not None and not self._junction.locked:
                conflict_distance = min(conflict_distance, max(0.0,
                    self._junction.entry - self.vehicle_length_m / 2 - self.junction_entry_buffer_m))
            speed_limits["crossing_conflict"] = self._approach_speed_kmh(
                0.0, conflict_distance, speed_mps)
            if traffic.conflict_time_s is not None and not junction_mode:
                # A braking-distance cap alone can still authorize accelerating
                # towards a future crossing/cut-in. Reserve time for the other
                # footprint to pass, instead of arriving at the conflict sooner.
                timed_limit = 3.6 * max(0.0, conflict_distance - self.braking_buffer_m) / (
                    traffic.conflict_time_s + self.following_time_s)
                speed_limits["crossing_conflict"] = min(speed_limits["crossing_conflict"], timed_limit)
        elif self._yield_timer_s > 0:
            self._yield_timer_s = max(0.0, self._yield_timer_s - self._control_dt_s)
            speed_limits["yield_clearance"] = 0.0
        if self._yield_timer_s > 0 and any(track.missed_s > 0 for track in tracks):
            speed_limits["unobserved_conflict"] = 0.0
        if agent_input.traffic_context.overflow_count:
            speed_limits["traffic_overflow"] = self.overflow_speed_kmh
        if agent_input.traffic_context.motion_paths_overflow_count:
            speed_limits["actor_path_overflow"] = self.overflow_speed_kmh
        lidar_unavailable = bool(np.mean(agent_input.core_observation.lidar_points <= 1e-4) > 0.5)
        if lidar_unavailable:
            speed_limits["lidar_unavailable"] = 0.0
        # The full relative route heading can expose the reversed/ambiguous
        # branch that the normalized asin observation cannot represent.
        heading_ambiguous = abs(candidates[0].relative_heading) > math.pi / 2
        if heading_ambiguous:
            speed_limits["heading_ambiguous"] = 0.0

        # Khi xe quay lệch, đầu/đuôi chiếm nhiều bề ngang hơn chiều rộng xe.
        # Ước lượng dưới đây dùng tiếp tuyến làn, chưa mô phỏng footprint trên cua.
        half_envelope = (
            self.vehicle_width_m * abs(math.cos(measured_lane_heading))
            + self.vehicle_length_m * abs(math.sin(measured_lane_heading))
        ) / 2.0
        lane_clearance = task.current_lane_width / 2.0 - abs(measured_lateral) - half_envelope
        road_clearance = min(float(obs[0]), float(obs[1])) * self.observation_road_scale_m - half_envelope
        # A permitted broken divider lies inside the maneuver corridor. Outer
        # road-edge clearance still applies while the body straddles two lanes.
        recovering = (road_clearance if lane_plan.active else min(lane_clearance, road_clearance)) < self.safety_margin_m
        if recovering:
            speed_limits["lane_recovery"] = self.recovery_speed_kmh

        escape_cap = min(value for reason, value in speed_limits.items() if reason != "front_vehicle")
        if traffic.front_hard_speed_limit_kmh is not None:
            escape_cap = min(escape_cap, traffic.front_hard_speed_limit_kmh)
        rear_side = assess_rear_side_escape(corridor, tracks, speed_mps, escape_cap, self) if not junction_mode and not lane_plan.active else {
            "rear_side_risk_ttc_s": None, "rear_escape_safe": False, "rear_escape_cap_kmh": float(escape_cap)}
        rear_side["prediction_scope"] = "JUNCTION_ROAD_PATHS" if junction_mode else "ROAD_CUT_IN_4S"
        if rear_side["rear_escape_safe"] and not traffic.emergency and "front_vehicle" in speed_limits and not junction_mode:
            speed_limits["front_vehicle"] = max(speed_limits["front_vehicle"], escape_cap)
            rear_side["rear_escape_applied"] = True
        else:
            rear_side["rear_escape_applied"] = False

        speed_limit_reason = min(speed_limits, key=speed_limits.get)
        target_speed = speed_limits[speed_limit_reason]

        throttle = float(np.clip(
            self.k_speed * (target_speed - speed_kmh), -1.0, self.max_throttle
        ))
        emergency_brake = traffic.emergency
        if emergency_brake:
            throttle = -1.0

        behavior = "CRUISE"
        if task.decision_point is not None:
            behavior = "APPROACH_JUNCTION"
        if speed_limit_reason == "front_vehicle":
            behavior = "FOLLOW"
        if speed_limit_reason == "lidar_obstacle":
            behavior = "STOP_FOR_OBSTACLE"
        if traffic.conflict_distance_m is not None or self._yield_timer_s > 0:
            behavior = "YIELD"
        if self._junction.phase != "CRUISE":
            behavior = self._junction.phase
        if rear_side["rear_escape_applied"]:
            behavior = "REAR_SIDE_ESCAPE"
        if lane_plan.active:
            behavior = self._lane_change.phase
        if recovering:
            behavior = "RECOVER_LANE"
        if heading_ambiguous or lidar_unavailable:
            behavior = "DEGRADED_STOP"
        if emergency_brake:
            behavior = "EMERGENCY_BRAKE"

        return AgentDecision(
            action_payload=[steering, throttle],
            # Diagnostics giải thích quyết định để xem log, không điều khiển xe.
            diagnostics={
                "rule": "emergency_brake" if emergency_brake else (
                    "recover_lane" if recovering else "follow_lane"
                ),
                "lateral_offset_right_m": float(lateral_m),
                "heading_error_rad": float(heading_error),
                "lookahead_m": float(lookahead),
                "speed_kmh": float(speed_kmh),
                "target_speed_kmh": float(target_speed),
                "speed_limit_reason": speed_limit_reason,
                "speed_limits_kmh": speed_limits,
                "behavior_state": behavior,
                "front_gap_m": traffic.front_gap_m,
                "min_ttc_s": traffic.min_ttc_s,
                "conflict_distance_m": traffic.conflict_distance_m,
                "conflict_time_s": traffic.conflict_time_s,
                "junction_entry_conflict": traffic.junction_conflict or bool(junction_diagnostics.get("junction_conflict_track_ids")),
                "lidar_obstacle_gap_m": traffic.obstacle_gap_m,
                "yield_clear_timer_s": self._yield_timer_s,
                "traffic_overflow_count": int(agent_input.traffic_context.overflow_count),
                "tracked_actor_count": len(tracks),
                "ghost_track_count": sum(track.missed_s > 0 for track in tracks),
                "track_ambiguous_matches": self._tracker.ambiguous_matches,
                "actor_motion_path_count": len(agent_input.traffic_context.motion_paths),
                "actor_motion_path_overflow_count": agent_input.traffic_context.motion_paths_overflow_count,
                "tracks": [{"id": track.track_id, "missed_s": track.missed_s,
                            "yaw_rate_rad_s": track.yaw_rate, "hits": track.hits, "road_paths": len(track.paths),
                            "x": track.actor.relative_position_x, "y": track.actor.relative_position_y}
                           for track in tracks],
                **junction_diagnostics,
                "rear_side": rear_side,
                "lane_change": lane_plan.diagnostics,
                "emergency_brake": emergency_brake,
                "lane_clearance_estimate_m": float(lane_clearance),
                "road_clearance_estimate_m": float(road_clearance),
                "steering": steering,
                "throttle_brake": throttle,
            },
        )

    def _approach_speed_kmh(self, arrival_speed_mps, distance_m, speed_mps):
        """Speed bound from v² = v_arrival² + 2*a*d after response and buffer."""
        usable = max(0.0, distance_m - self.braking_buffer_m - self.response_time_s * speed_mps)
        return 3.6 * math.sqrt(
            arrival_speed_mps ** 2 + 2.0 * self.comfortable_deceleration_mps2 * usable
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def close(self) -> None:
        pass
