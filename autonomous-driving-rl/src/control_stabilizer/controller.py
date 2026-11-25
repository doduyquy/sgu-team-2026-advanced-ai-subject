"""Public-geometry safety controller for an explicitly labeled BC/control hybrid.

This is a new policy, not a modification of the model's inference output.
Coordinates: x forward, y left; positive actuator steering turns left.
"""

from dataclasses import asdict, dataclass
import math


def bound(value, low, high):
    return max(low, min(high, value))


def wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


@dataclass(frozen=True)
class StabilizerConfig:
    version: str = '0.2.2'
    wheelbase_m: float = 2.46894
    max_wheel_angle_deg: float = 40.
    lookahead_min_m: float = 5.
    lookahead_max_m: float = 12.
    lookahead_time_s: float = .65
    steering_tolerance: float = .01
    max_steering: float = .7
    steering_rate_per_s: float = 1.5
    cruise_kmh: float = 32.
    lateral_acceleration_mps2: float = 2.
    comfortable_deceleration_mps2: float = 3.
    speed_gain: float = .65
    ego_length_m: float = 4.515
    ego_width_m: float = 1.852
    traffic_margin_m: float = .35
    traffic_time_gap_s: float = 1.2
    traffic_standstill_gap_m: float = 3.
    clear_creep_speed_mps: float = 2.5
    clear_creep_wait_s: float = 2.
    clear_creep_throttle: float = .3
    clear_creep_gain: float = .45
    clear_creep_center_m: float = .35
    clear_creep_heading_rad: float = .12
    clear_creep_max_model_brake: float = .25

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name != 'version' and (not math.isfinite(value) or value <= 0):
                raise ValueError('Invalid stabilizer parameter: '+name)
        if not 0 < self.max_steering <= 1 or self.lookahead_max_m < self.lookahead_min_m:
            raise ValueError('Invalid stabilizer bounds')


def interpolate(points, distance):
    if distance <= points[0][0]:
        return points[0][1:]
    for left, right in zip(points, points[1:]):
        if distance <= right[0]:
            weight = (distance-left[0])/(right[0]-left[0])
            return (left[1]+weight*(right[1]-left[1]), left[2]+weight*(right[2]-left[2]),
                    wrap(left[3]+weight*wrap(right[3]-left[3])))
    return points[-1][1:]


class LaneStabilizer:
    def __init__(self, config=None):
        self.config = config or StabilizerConfig()
        self.reset(.1)

    def reset(self, dt):
        if abs(dt-.1) > 1e-9:
            raise ValueError('Stabilizer requires its 10 Hz control contract')
        self.dt = dt
        self.previous_steering = 0.
        self.idle_seconds = 0.
        self.creep_active = False

    def apply(self, observation, model_action):
        if hasattr(observation, 'to_dict'):
            observation = observation.to_dict()
        if len(model_action) != 2 or any(not math.isfinite(x) or abs(x) > 1 for x in model_action):
            raise ValueError('Invalid raw model action; stabilization cannot conceal model failure')
        cfg, task = self.config, observation['task_context']
        motion = observation['ego_motion']
        if motion is not None:
            speed = math.hypot(motion['velocity_x_mps'], motion['velocity_y_mps'])
        else:
            speed = max(0., observation['core_observation']['features'][3]*81.-1.)/3.6
        route = [(w['lookahead_distance_m'], w['relative_x'], w['relative_y'], w['relative_heading'])
                 for w, valid in zip(task['waypoints'], task['validity_mask']) if valid]
        reasons = []
        if not route:
            applied = [0., min(float(model_action[1]), -.4)]
            self.previous_steering = 0.
            self.idle_seconds = 0.
            self.creep_active = False
            return applied, {'raw_model_action': list(model_action), 'applied_action': applied,
                'steering_intervened': abs(model_action[0]) > 1e-8,
                'longitudinal_intervened': abs(applied[1]-model_action[1]) > 1e-8,
                'reasons': ['NO_PUBLIC_ROUTE_BRAKE'], 'controller_version': cfg.version}
        current = next((l for l in (task['lane_context'] or {}).get('lanes', []) if l['side'] == 'CURRENT'), None)
        current_points = current['points'] if current else []
        lookahead = bound(cfg.lookahead_min_m+cfg.lookahead_time_s*speed,
                          cfg.lookahead_min_m, cfg.lookahead_max_m)
        if current_points:
            center = interpolate(current_points, 0.)
            first = route[0]
            route_origin = (first[1]-first[0]*math.cos(first[3]), first[2]-first[0]*math.sin(first[3]))
            # Keep the current lane's lateral offset when route samples name a parallel lane.
            lane_offset = (center[0]-route_origin[0])*(-math.sin(center[2])) + (center[1]-route_origin[1])*math.cos(center[2])
            path = [p for p in current_points if p[0] >= 0]
            if not path or path[0][0] > 1e-4:
                path.insert(0, (0., *center))
            end = path[-1][0]
            path += [(d, x-math.sin(h)*lane_offset, y+math.cos(h)*lane_offset, h)
                     for d, x, y, h in route if d > end+1e-4]
        else:
            path = route
            center = (0., 0., motion['lane_heading_error_rad'] if motion else route[0][3])
        target = interpolate(path, lookahead)
        radius_squared = target[0]**2+target[1]**2
        curvature = 2.*target[1]/max(radius_squared, 1.)
        reference = bound(math.atan(cfg.wheelbase_m*curvature)/math.radians(cfg.max_wheel_angle_deg),
                          -cfg.max_steering, cfg.max_steering)
        # Project inconsistent suggestions to the geometric reference; preserve consistent model outputs.
        steering = float(model_action[0])
        if abs(steering-reference) > cfg.steering_tolerance or abs(steering) > cfg.max_steering:
            steering = reference
            reasons.append('STEERING_GEOMETRY_DISAGREEMENT')
        limit = cfg.steering_rate_per_s*self.dt
        rate_limited = bound(steering, self.previous_steering-limit, self.previous_steering+limit)
        if abs(rate_limited-steering) > 1e-8:
            reasons.append('STEERING_RATE_LIMIT')
        steering = rate_limited
        self.previous_steering = steering
        cap = cfg.cruise_kmh/3.6
        samples = [(0., center[2])] + [(d, h) for d, _, _, h in route]
        for (distance, heading), (next_distance, next_heading) in zip(samples, samples[1:]):
            ds = next_distance-distance
            if ds <= .1:
                continue
            bend = abs(wrap(next_heading-heading))/ds
            if bend > 1e-4:
                corner_speed = math.sqrt(cfg.lateral_acceleration_mps2/bend)
                cap = min(cap, math.sqrt(corner_speed**2+2*cfg.comfortable_deceleration_mps2*max(0., distance-2.)))
        throttle = float(model_action[1])
        if speed > cap:
            throttle = min(throttle, -min(1., cfg.speed_gain*(speed-cap)))
            reasons.append('GEOMETRIC_SPEED_CAP')
        nearest_gap = None
        corridor = [(0., 0.)] + [(p[1], p[2]) for p in path if p[0] > 0]
        traffic = observation['traffic_context']
        for actor, present in zip(traffic['actors'], traffic['validity_mask']):
            if not present:
                continue
            best, arc = None, 0.
            for a, b in zip(corridor, corridor[1:]):
                dx, dy = b[0]-a[0], b[1]-a[1]
                length = math.hypot(dx, dy)
                if length < .01:
                    continue
                weight = bound(((actor['relative_position_x']-a[0])*dx+(actor['relative_position_y']-a[1])*dy)/length**2, 0., 1.)
                x, y = a[0]+weight*dx, a[1]+weight*dy
                separation = math.hypot(actor['relative_position_x']-x, actor['relative_position_y']-y)
                item = (separation, arc+weight*length, dx/length, dy/length)
                if best is None or item[0] < best[0]:
                    best = item
                arc += length
            if best is None or best[1] < .1 or actor['relative_position_x'] < 0:
                continue
            width = (cfg.ego_width_m+actor['width'])/2.+cfg.traffic_margin_m
            if best[0] > width:
                continue
            gap = best[1]-(cfg.ego_length_m+actor['length'])/2.
            nearest_gap = gap if nearest_gap is None else min(nearest_gap, gap)
            closing = max(0., -(actor['relative_velocity_x']*best[2]+actor['relative_velocity_y']*best[3]))
            wanted = cfg.traffic_standstill_gap_m+cfg.traffic_time_gap_s*speed+closing**2/(2*cfg.comfortable_deceleration_mps2)
            if gap < wanted:
                brake = bound((wanted-gap)/max(wanted, 1.), .1, 1.)
                throttle = min(throttle, -brake)
                if 'TRAFFIC_CORRIDOR_BRAKE' not in reasons:
                    reasons.append('TRAFFIC_CORRIDOR_BRAKE')
        # A narrowly scoped recovery for a weak learned brake deadlock on a clear road.
        # Never release model brakes near a junction/passage or any in-corridor traffic.
        clear_path = (task['decision_point'] is None
            and task['route_passage'] is None and nearest_gap is None and target[0] > .1
            and abs(center[1]) < cfg.clear_creep_center_m and abs(center[2]) < cfg.clear_creep_heading_rad
            and speed <= cap+1e-6)
        if clear_path and -cfg.clear_creep_max_model_brake < model_action[1] < .1:
            if speed < .8:
                self.idle_seconds += self.dt
            elif not self.creep_active:
                self.idle_seconds = 0.
            if self.idle_seconds >= cfg.clear_creep_wait_s:
                self.creep_active = True
            if self.creep_active:
                recovery = bound(cfg.clear_creep_gain*(min(cfg.clear_creep_speed_mps, cap)-speed), -1., cfg.clear_creep_throttle)
                throttle = max(throttle, recovery)
                reasons.append('CLEAR_PATH_WEAK_BRAKE_RECOVERY')
        else:
            self.idle_seconds = 0.
            self.creep_active = False
        applied = [float(steering), float(throttle)]
        return applied, {'controller_version': cfg.version, 'raw_model_action': list(model_action),
            'applied_action': applied, 'geometric_steering': reference,
            'steering_intervened': abs(steering-model_action[0]) > 1e-8,
            'longitudinal_intervened': abs(throttle-model_action[1]) > 1e-8,
            'lookahead_m': lookahead, 'center_left_m': center[1], 'heading_error_rad': center[2],
            'speed_kmh': speed*3.6, 'speed_cap_kmh': cap*3.6, 'nearest_traffic_gap_m': nearest_gap,
            'clear_creep_active': self.creep_active, 'idle_seconds': self.idle_seconds,
            'reasons': reasons}
