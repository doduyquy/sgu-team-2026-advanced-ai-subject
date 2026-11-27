"""Read-only behavior metrics for deterministic, guidance-free evaluation."""
from collections import Counter, defaultdict
import math

import numpy as np

from cloud.kaggle.risk_guidance import risk_action


class RiskMetrics:
    def __init__(self):
        self.counts = Counter()
        self.values = defaultdict(list)
        self.action = None
        self.last_pedal = 0.0
        self.last_steer = None
        self.last_sign = 0
        self.last_lane = None
        self.lane_dwell = 0
        self.last_lateral_accel = None
        self.last_heading = None
        self.last_velocity = None
        self.was_turning = False

    def before_step(self, observation, action):
        self.action = action
        _, self.risk = risk_action(observation, action)

    def after_step(self, ego):
        self.counts["decisions"] += 1
        steer, pedal = map(float, self.action)
        speed = float(ego.speed_km_h) / 3.6
        risk = self.risk
        curvature = risk["curvature_1pm"]
        gap = risk["gap_m"]
        clear = (curvature is not None and curvature < .025 and risk["forward_lidar_m"] > 13
                 and (gap is None or gap > 18) and risk["hazard"] == "none")
        if speed < 1 / 3.6:
            self.counts["stationary"] += 1
            if clear:
                self.counts["stationary_clear_proxy"] += 1
        if pedal < -.15:
            self.counts["brake"] += 1
            self.counts["brake_clear_proxy"] += int(clear and speed > 2.0)
            if self.last_pedal >= -.15:
                self.counts["brake_onsets"] += 1
                for name, field in (("brake_gap_m", "gap_m"), ("brake_ttc_s", "ttc_s"),
                                    ("brake_closing_mps", "closing_mps")):
                    if risk[field] is not None:
                        self.values[name].append(risk[field])
        self.last_pedal = pedal
        if gap is not None:
            self.values["following_gap_m"].append(gap)
        if self.last_steer is not None:
            self.values["steer_delta"].append(abs(steer - self.last_steer))
        self.last_steer = steer
        sign = int(np.sign(steer)) if abs(steer) > .05 else 0
        if sign:
            if self.last_sign:
                self.counts["steer_sign_pairs"] += 1
                self.counts["steer_reversals"] += int(sign != self.last_sign)
            self.last_sign = sign
        if curvature is not None:
            turning = curvature >= .035
            if turning and not self.was_turning:
                self.values["curve_entry_kmh"].append(speed * 3.6)
            if not turning and self.was_turning:
                self.values["curve_exit_kmh"].append(speed * 3.6)
            self.was_turning = turning
        lane = ego.lane
        index = ego.lane_index
        if lane is not None and index is not None:
            self.lane_dwell = self.lane_dwell + 1 if index == self.last_lane else 1
            station, offset = lane.local_coordinates(ego.position)
            width = float(lane.width_at(station))
            self.values["lane_abs_offset_m"].append(abs(float(offset)))
            if width > 0:
                self.counts["lane_sample"] += 1
                self.counts["lane_center_quarter"] += int(abs(offset) <= width * .25)
                self.counts["stable_lane"] += int(abs(offset) <= width * .25 and self.lane_dwell >= 10)
                if abs(offset) + float(ego.WIDTH) * .5 >= width * .5:
                    road = ego.navigation.current_ref_lanes
                    if road and lane in road:
                        lane_id = road.index(lane)
                        for j in (lane_id - 1, lane_id + 1):
                            if 0 <= j < len(road):
                                other = road[j]
                                u, _ = other.local_coordinates(ego.position)
                                _, neighbor_offset = lane.local_coordinates(other.position(u, 0))
                                if offset * neighbor_offset > 0:
                                    self.counts["same_direction_line_overlap_proxy"] += 1
                                    break
            if self.last_lane and index[:2] == self.last_lane[:2] and index[-1] != self.last_lane[-1]:
                self.counts["same_road_lane_index_changes"] += 1
            self.last_lane = index
        heading = float(ego.heading_theta)
        velocity = np.asarray(ego.velocity, dtype=float).copy()
        if self.last_heading is not None:
            yaw = math.remainder(heading - self.last_heading, 2.0 * math.pi) / .1
            self.values["yaw_abs_rad_s"].append(abs(yaw))
            normal = np.array([-math.sin(heading), math.cos(heading)])
            acc = float((velocity - self.last_velocity) @ normal / .1)
            self.values["lateral_acc_abs_mps2"].append(abs(acc))
            if self.last_lateral_accel is not None:
                self.values["lateral_jerk_abs_mps3"].append(abs(acc - self.last_lateral_accel) / .1)
            self.last_lateral_accel = acc
        self.last_heading = heading
        self.last_velocity = velocity

    def summary(self):
        means = {name: float(np.mean(values)) for name, values in self.values.items() if values}
        means["steer_delta_rms"] = (float(np.sqrt(np.mean(np.square(self.values["steer_delta"]))))
                                    if self.values["steer_delta"] else None)
        return {"counts": dict(self.counts), "means": means,
                "samples": {name: len(values) for name, values in self.values.items()}}
