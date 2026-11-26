"""Bundled, attributed OSM corridor; independent of the locked MapSuite splits."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

ROUTE_PATH = Path(__file__).resolve().parents[2]/'assets/maps/vietnam/vo_van_kiet_route.json'
VN_INDEX = 3
VN_SPEED_LIMIT_KMH = 80.
VN_CRUISE_KMH = 60.


@lru_cache(maxsize=1)
def route_data():
    return json.loads(ROUTE_PATH.read_text(encoding='utf-8'))


def definition():
    return [{'id': 'VN', 'route_sha256': hashlib.sha256(ROUTE_PATH.read_bytes()).hexdigest(),
             'lane_count': 3, 'lane_width': 3.5, 'version': 1}]


def race_road():
    from src.platform import canonical_json_sha256
    from src.launcher.models import ResolvedCaseV1
    blocks = definition()
    case = ResolvedCaseV1(case_id='custom/VN/vo-van-kiet/v1', case_index=1,
        protocol_order_index=1, split='CUSTOM', tier='Vietnam', sequence='VN',
        geometry_generation_seed=2026, geometry_sha256=canonical_json_sha256(blocks),
        environment_seed=2026, traffic_density=0., horizon_steps=18000)
    return case, blocks


def is_vietnam(case):
    return bool(case and getattr(case, 'case_id', '').startswith('custom/VN/'))


def centerline():
    return np.asarray(route_data()['points_xy_m'], dtype=float)


def road_frame(points):
    delta = np.gradient(points, axis=0)
    tangent = delta/np.maximum(np.linalg.norm(delta, axis=1)[:, None], 1e-9)
    right = np.column_stack((tangent[:, 1], -tangent[:, 0]))
    return tangent, right


def point_at(distance):
    points = centerline()
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    result = np.array([np.interp(distance, lengths, points[:, i]) for i in range(2)])
    index = min(len(points)-2, max(0, int(np.searchsorted(lengths, distance)-1)))
    heading = float(np.arctan2(*(points[index+1]-points[index])[::-1]))
    return result, heading
