"""The custom corridor stays continuous, attributed and isolated from benchmarks."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from src.platform import CanonicalActionV1, canonical_json_sha256
from src.visualization.race_session import load_race_roads, load_original_roads, RaceSelection, RaceComparison
from src.visualization.vietnam_track import ROUTE_PATH, centerline, definition, race_road, route_data, point_at, road_frame
from src.visualization.vietnam_speed import curve_speed_limit, decision_action, vehicle_speed_limit


class VietnamTrackTests(unittest.TestCase):
    def test_fourth_choice_keeps_original_geometries_and_split(self):
        root=Path(__file__).resolve().parents[1]
        original=load_original_roads(root)
        roads=load_race_roads(root,'original')
        self.assertEqual(roads[:3],original)
        case,blocks=roads[3]
        self.assertEqual(case.split,'CUSTOM')
        self.assertEqual(case.geometry_sha256,canonical_json_sha256(blocks))
        self.assertEqual(blocks[0]['route_sha256'],hashlib.sha256(ROUTE_PATH.read_bytes()).hexdigest())

    def test_route_is_ten_kilometres_without_breaks_or_return_loop(self):
        points=centerline()
        segments=np.diff(points,axis=0)
        distances=np.linalg.norm(segments,axis=1)
        self.assertTrue(np.all(distances>.1))
        self.assertLess(float(distances.max()),3.)
        self.assertTrue(9990<float(distances.sum())<10010)
        # The previous service-road route accidentally doubled back at 8.95 km.
        self.assertTrue(np.all(segments[-600:,0]<0))
        angles=np.unwrap(np.arctan2(segments[:,1],segments[:,0]))
        self.assertLess(float(np.max(np.abs(np.diff(angles)))),.13)
        data=route_data()
        self.assertEqual(data['source']['license'],'ODbL-1.0')
        self.assertGreater(len(data['source']['way_ids']),50)
        self.assertLess(data['reconstruction']['smoothing_max_displacement_m'],3.)

    def test_right_lane_orientation_follows_heading_at_distant_points(self):
        points=centerline()
        tangent,right=road_frame(points)
        self.assertTrue(np.allclose(np.sum(tangent*right,axis=1),0))
        for distance in (30,2500,5000,9500):
            p,heading=point_at(distance)
            self.assertTrue(np.isfinite(p).all())
            self.assertAlmostEqual(np.linalg.norm((np.cos(heading),np.sin(heading))),1.)

    def test_common_speed_limit_slows_before_tight_curves(self):
        limits=[curve_speed_limit(SimpleNamespace(position=point_at(d)[0])) for d in range(50,9900,20)]
        self.assertGreater(max(limits),75.)
        self.assertLess(min(limits),35.)
        self.assertTrue(all(8.<=v<=80. for v in limits))
        car=SimpleNamespace(position=point_at(100)[0])
        self.assertEqual(vehicle_speed_limit(SimpleNamespace(),car),20.)
        self.assertGreater(vehicle_speed_limit(SimpleNamespace(race_speed_limit_kmh=80.),car),20.)

    def test_pursuit_preserves_braking_and_player_mppi_actions(self):
        requested=CanonicalActionV1(.4,-.8)
        decision=SimpleNamespace(diagnostics={'target_x':8.,'target_y':1.})
        action=decision_action(decision,requested,None,True)
        self.assertGreater(action.steering,0.)
        self.assertLess(action.steering,.4)
        self.assertEqual(action.throttle_brake,-.8)
        self.assertIs(decision_action(decision,requested,None,False),requested)
        self.assertIs(decision_action(SimpleNamespace(diagnostics={}),requested,None,True),requested)

    def test_selection_and_result_report_custom_speed_without_changing_defaults(self):
        root=Path(__file__).resolve().parents[1]
        roads=load_race_roads(root,'original')
        selection=RaceSelection(road_index=3,traffic=True)
        case=selection.case(roads)
        self.assertEqual(case.traffic_density,.015)
        report=RaceComparison((0,),case.to_dict(),1).to_dict()
        self.assertEqual(report['race_rules']['speed_limit_kmh'],80.)
        selection.road_index=1
        original=selection.case(roads)
        self.assertEqual(original.traffic_density,.1)
        self.assertEqual(RaceComparison((0,),original.to_dict(),2).to_dict()['race_rules']['speed_limit_kmh'],20.)

    def test_real_leaf_asset_has_alpha_to_avoid_black_rectangles(self):
        # Poly Haven's JPG glTF omits the separate leaf alpha; packed PNG is required.
        from PIL import Image
        root=Path(__file__).resolve().parents[1]/'assets/scenery/island_tree_02'
        with Image.open(root/'textures/island_tree_02_leaves_rgba_1k.png') as image:
            self.assertEqual(image.mode,'RGBA')
            alpha=np.asarray(image)[:,:,3]
            self.assertGreater(float((alpha<10).mean()),.3)
            self.assertGreater(float((alpha>245).mean()),.1)


if __name__=='__main__':
    unittest.main()
