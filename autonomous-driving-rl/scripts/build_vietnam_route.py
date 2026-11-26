"""Build the bundled VN corridor from an attributed Overpass OSM extract.

Offline by default. No invented GPS coordinates or gap-joining is allowed.
Input ways must share real OSM node IDs. Scenery/elevation are not surveyed.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import heapq
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def build(source, target):
    raw = json.loads(source.read_text(encoding='utf-8'))
    ways = [w for w in raw['elements'] if w.get('tags', {}).get('highway') in
            ('trunk', 'trunk_link', 'primary', 'primary_link')]
    positions, graph = {}, defaultdict(list)
    for way in ways:
        for node, point in zip(way['nodes'], way['geometry']):
            positions[node] = (point['lon'], point['lat'])
    for way in ways:
        tags = way['tags']
        named = 'Võ Văn Kiệt' in tags.get('name', '') and 'Hẻm' not in tags.get('name', '')
        if not named and not tags.get('bridge'):
            continue
        pairs = list(zip(way['nodes'], way['nodes'][1:]))
        if tags.get('oneway') == '-1':
            pairs = [(b, a) for a, b in pairs]
        for a, b in pairs:
            dx = (positions[b][0]-positions[a][0])*111195*math.cos(math.radians(10.75))
            dy = (positions[b][1]-positions[a][1])*111195
            graph[a].append((b, math.hypot(dx, dy), way['id']))
            if tags.get('oneway') not in ('yes', '1', '-1'):
                graph[b].append((a, math.hypot(dx, dy), way['id']))
    # East-to-west main carriageway. Prefer the continuous real 10 km corridor.
    start = 1769173959
    distances, parent, queue = {start: 0.}, {}, [(0., start)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for nxt, length, way_id in graph[node]:
            if distance+length < distances.get(nxt, math.inf):
                distances[nxt] = distance+length
                parent[nxt] = (node, way_id)
                heapq.heappush(queue, (distance+length, nxt))
    # Follow the westbound boulevard, not a longer graph path that returns
    # through a U-turn onto the opposite carriageway.
    end = min(distances, key=lambda node: positions[node][0])
    nodes, way_ids = [end], []
    while nodes[-1] != start:
        previous, way_id = parent[nodes[-1]]
        nodes.append(previous)
        way_ids.append(way_id)
    nodes.reverse()
    way_ids = list(dict.fromkeys(reversed(way_ids)))
    coordinates = np.array([positions[n] for n in nodes])
    origin = coordinates[0]
    xy = (coordinates-origin)*np.array([111195*math.cos(math.radians(origin[1])), 111195])
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    length = min(10000., distance[-1])
    samples = np.r_[np.arange(0, length, 2.), length]
    points = np.column_stack([np.interp(samples, distance, xy[:, axis]) for axis in range(2)])
    kernel = np.exp(-.5*(np.arange(-8, 9)/2.)**2)
    kernel /= kernel.sum()
    smooth = np.column_stack([np.convolve(np.pad(points[:, i], (8, 8), mode='edge'),
                                         kernel, mode='valid') for i in range(2)])
    smooth[0], smooth[-1] = points[0], points[-1]
    deviation = float(np.linalg.norm(points-smooth, axis=1).max())
    if deviation > 3.:
        raise ValueError(f'Smoothing displacement {deviation:.3f} m exceeds the 3 m budget')
    smooth_length = float(np.linalg.norm(np.diff(smooth, axis=0), axis=1).sum())
    used = [w for w in ways if w['id'] in way_ids]
    payload = {
        'id': 'vn-vo-van-kiet-v1', 'name': 'Võ Văn Kiệt · Sài Gòn',
        'source': {'provider': 'OpenStreetMap contributors', 'license': 'ODbL-1.0',
                   'attribution': '© OpenStreetMap contributors',
                   'copyright_url': 'https://www.openstreetmap.org/copyright',
                   'downloaded_utc': datetime.now(timezone.utc).isoformat(),
                   'osm_base_timestamp': raw.get('osm3s', {}).get('timestamp_osm_base'),
                   'way_ids': way_ids, 'start_node': start, 'source_path_end_node': end,
                   'bridge_geometry_source': 'https://api.openstreetmap.org/api/0.6/',
                   'input_file': source.name},
        'reconstruction': {'lane_count': 3, 'lane_width_m': 3.5,
            'note': 'Single closed race corridor following OSM geometry; uniform 3 lanes, flat elevation, intersections omitted; scenery is procedural, not surveyed.',
            'smoothing_max_displacement_m': round(deviation, 4), 'resample_spacing_m': 2.},
        'length_m': round(smooth_length, 3), 'origin_lonlat': origin.tolist(),
        'coordinates_lonlat': coordinates.tolist(), 'points_xy_m': smooth.round(4).tolist(),
        'source_ways': used,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(json.dumps({'output': str(target), 'length_m': payload['length_m'],
                      'osm_ways': len(way_ids), 'smoothing_max_displacement_m': deviation}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'assets/maps/vietnam/vo_van_kiet_osm.json')
    parser.add_argument('--output', type=Path, default=ROOT/'assets/maps/vietnam/vo_van_kiet_route.json')
    args = parser.parse_args()
    build(args.source, args.output)
