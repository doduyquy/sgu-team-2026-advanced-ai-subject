"""Write the P0 409D index map for the pinned Stage 5 configuration."""
import csv
import sys


def rows():
    def row(name, units, normalization, expression, mask=None):
        return (name, units, normalization, expression, mask)

    for side in ("left", "right"):
        yield row(f"road_border_{side}", "m", "clip [0,1]",
                  f"dist_to_{side}_side / ((MAX_LANE_NUM+1)*MAX_LANE_WIDTH)")
    yield row("heading_relative_to_lane", "cosine", "clip [-1,1], /2+.5",
              "vehicle.heading_diff(current_ref_lanes[-1]); dot(vehicle forward,lane lateral)")
    yield row("speed", "km/h", "clip [0,1]", "(vehicle.speed_km_h+1)/(vehicle.max_speed_km_h+1)")
    yield row("current_steering", "native vehicle steering", "clip [0,1]", "(vehicle.steering/vehicle.MAX_STEERING+1)/2")
    for i, part in enumerate(("steering", "throttle_brake")):
        yield row(f"latest_{part}", "Box2", "clip [0,1]", f"(vehicle.last_current_action[1][{i}]+1)/2")
    yield row("yaw_rate_magnitude", "rad/s", "clip [0,1]", "acos(clip(dot(heading,last_heading)/(norms),0,1))/.1")
    yield row("lane_lateral_offset", "m", "clip [0,1]", "vehicle.lane.local_coordinates(position)[1]/MAX_LANE_WIDTH+.5")
    for checkpoint in ("current", "next"):
        for name, units, expression in (
            ("forward", "m", "(ckpt_in_heading/NAVI_POINT_DIST+1)/2"),
            ("lateral", "m", "(ckpt_in_rhs/NAVI_POINT_DIST+1)/2; native frame"),
            ("bend_radius", "m", "radius/(max_curve_radius+lane_count*lane_width); straight zero"),
            ("bend_direction", "direction", "(-CircularLane.direction+1)/2; straight .5"),
            ("bend_angle", "degree", "(rad2deg(angle)/max_curve_angle+1)/2; straight .5"),
        ):
            yield row(f"nav.{checkpoint}.{name}", units, "clip [0,1]", expression)
    for i in range(240):
        yield row(f"lidar.{i}", "range fraction", "native ray fraction [0,1]", f"Lidar.perceive.cloud_points[{i}]; 1=no return")
    for i in range(8):
        for name, units in (("relative_position_x", "m"), ("relative_position_y", "m"),
                            ("relative_velocity_x", "m/s"), ("relative_velocity_y", "m/s"),
                            ("relative_heading", "rad"), ("length", "m"), ("width", "m")):
            yield row(f"traffic.{i}.{name}", units, "raw", f"TrafficActorV1[{i}].{name}", 315+i)
    for i in range(8):
        yield row(f"traffic_mask.{i}", "boolean", "float32", f"TrafficContextV1.validity_mask[{i}]")
    yield row("traffic_active_ratio", "count", "/capacity", "active_count/max(1,capacity)")
    yield row("traffic_overflow_ratio", "count", "cap 10 / 10", "min(overflow_count,10)/10")
    for i in range(20):
        for name, units in (("relative_x", "m"), ("relative_y", "m"), ("relative_heading", "rad")):
            yield row(f"task.{i}.{name}", units, "raw ego frame", f"RouteWaypointV1[{i}].{name}", 385+i)
    for i in range(20):
        yield row(f"task_mask.{i}", "boolean", "float32", f"TaskContextV1.validity_mask[{i}]")
    yield row("task_lane_width_ratio", "m", "/5m", "TaskContextV1.current_lane_width/5")
    yield row("task_goal_x", "unit vector", "raw", "navigation_goal_direction[0]")
    yield row("task_goal_y", "unit vector", "raw", "navigation_goal_direction[1]")
    yield row("task_route_end_near", "boolean", "float32", "TaskContextV1.route_end_within_lookahead")


if __name__ == "__main__":
    features = list(rows())
    assert len(features) == 409
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("index", "name", "physical_units", "encoder_normalization", "source_expression", "mask_index"))
    writer.writerows((index, *feature) for index, feature in enumerate(features))
