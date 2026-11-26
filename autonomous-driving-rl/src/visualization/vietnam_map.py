"""A real OSM corridor exposed through the existing NodeRoadNetwork contract.

Only this map uses polyline lanes and full-corridor collision shapes. The
locked PG maps and their managers keep their original generation path.
"""
from copy import deepcopy
import math

import numpy as np
from panda3d.bullet import BulletBoxShape, BulletTriangleMesh, BulletTriangleMeshShape
from panda3d.core import NodePath, Quat, TransformState, Vec3
from metadrive.component.block.base_block import BaseBlock
from metadrive.component.lane.point_lane import PointLane
from metadrive.component.map.pg_map import PGMap
from metadrive.component.road_network import Road
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.component.pgblock.pg_block import PGBlockSocket
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.constants import CollisionGroup, MetaDriveType, PGLineColor, PGLineType
from metadrive.engine.physics_node import BaseGhostBodyNode, BaseRigidBodyNode
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.manager.traffic_manager import PGTrafficManager, TrafficMode
from metadrive.utils.vertex import make_polygon_model

from src.visualization.vietnam_track import centerline, definition, road_frame, route_data


class VietnamBlock(BaseBlock):
    ID = 'V'

    def __init__(self, network):
        super().__init__(1, network, random_seed=2026, ignore_intersection_checking=True)
        self.pre_block_socket = PGBlockSocket(Road(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3))

    @property
    def block_network_type(self):
        return NodeRoadNetwork

    def _sample_topology(self):
        points = centerline()
        _, right = road_frame(points)
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        cuts = [0, int(np.searchsorted(distance, 10)), int(np.searchsorted(distance, 110))]
        cuts += list(range(cuts[-1]+25, len(points)-1, 25)) + [len(points)-1]
        nodes = [FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, FirstPGBlock.NODE_3]
        nodes += [f'VN{i:04}' for i in range(len(cuts)-3+1)]
        for segment, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
            for index in range(3):
                lane = PointLane(points[a:b+1]+right[a:b+1]*(index-1)*3.5, 3.5,
                                 speed_limit=80.)
                lane.line_colors = (PGLineColor.GREY, PGLineColor.GREY)
                lane.line_types = (PGLineType.CONTINUOUS if index == 0 else PGLineType.BROKEN,
                                   PGLineType.CONTINUOUS if index == 2 else PGLineType.BROKEN)
                lane.radius = 0.
                self.block_network.add_lane(nodes[segment], nodes[segment+1], lane)
        roads = [Road(u, v) for u, outs in self.block_network.graph.items() for v in outs]
        self._respawn_roads = roads[14::16]
        self._socket = PGBlockSocket(roads[-1], Road(roads[-1].end_node, roads[-1].start_node))
        return True

    def get_socket_list(self):
        return [self._socket]

    def get_intermediate_spawn_lanes(self):
        return [road.get_lanes(self.block_network) for road in self._respawn_roads]

    def create_in_world(self):
        for u, outgoing in self.block_network.graph.items():
            for v, lanes in outgoing.items():
                for index, lane in enumerate(lanes):
                    lane.index = (u, v, index)
                    # A triangle mesh retains the true curved lane footprint;
                    # a convex hull would cut across concave OSM bends.
                    model = make_polygon_model(lane.polygon, .1)
                    mesh = BulletTriangleMesh()
                    mesh.addGeom(model.node().getGeom(0))
                    body = BaseRigidBodyNode(lane.index, lane.metadrive_type)
                    body.setStatic(True)
                    body.setIntoCollideMask(CollisionGroup.LaneSurface)
                    body.addShape(BulletTriangleMeshShape(mesh, dynamic=False))
                    node = self.lane_node_path.attachNewNode(body)
                    self.static_nodes.append(body)
                    self._node_path_list.extend((node, body))
                    model.removeNode()
                for lateral, kind in ((-1.75, MetaDriveType.LINE_SOLID_SINGLE_WHITE),
                                      (1.75, MetaDriveType.LINE_BROKEN_SINGLE_WHITE)):
                    for index, lane in enumerate(lanes):
                        if lateral > 0 and index == 2:
                            continue
                        if lateral < 0 and index > 0:
                            continue
                        self._line_body(lane, lateral, kind)
                self._line_body(lanes[-1], 1.75, MetaDriveType.LINE_SOLID_SINGLE_WHITE)

    def _line_body(self, lane, lateral, kind):
        broken = MetaDriveType.is_broken_line(kind)
        body = BaseGhostBodyNode(None, kind)
        body.setStatic(True)
        body.setActive(False)
        body.setIntoCollideMask(CollisionGroup.BrokenLaneLine if broken else CollisionGroup.ContinuousLaneLine)
        origin = lane.position(0, lateral)
        step = 4. if broken else 4.
        for start in np.arange(0, lane.length, step):
            end = min(lane.length, start+(2. if broken else step))
            a, b = lane.position(start, lateral), lane.position(end, lateral)
            length = float(np.linalg.norm(b-a))
            if length < .01:
                continue
            middle = (a+b)/2-origin
            rotation = Quat()
            rotation.setFromAxisAngle(math.degrees(math.atan2(*(b-a)[::-1])), Vec3(0, 0, 1))
            body.addShape(BulletBoxShape(Vec3(length/2, .0375, 1.)),
                          TransformState.makePosQuatScale(Vec3(float(middle[0]), float(middle[1]), .5),
                                                          rotation, Vec3(1)))
        node = self.lane_line_node_path.attachNewNode(body)
        node.setPos(float(origin[0]), float(origin[1]), 0)
        self.static_nodes.append(body)
        self._node_path_list.extend((node, body))


class VietnamMap(PGMap):
    is_vietnam = True

    def _generate(self):
        self.route = route_data()
        self.route_points = centerline()
        block = VietnamBlock(self.road_network)
        block.construct_block(self.engine.worldNP, self.engine.physics_world)
        self.blocks = [block]
        self.road_network.after_init()

    def get_meta_data(self):
        return {'block_sequence': definition(), 'map_type': 'VIETNAM_OSM_CORRIDOR',
                'source': self.route['source'], 'reconstruction': self.route['reconstruction']}


class RaceMapManager(PGMapManager):
    def spawn_object(self, object_class, *args, **kwargs):
        config = kwargs.get('map_config', {})
        blocks = config.get('config')
        if object_class is PGMap and isinstance(blocks, list) and blocks and blocks[0].get('id') == 'VN':
            if blocks != definition():
                raise ValueError('Vietnam route source hash differs from the selected course')
            return super().spawn_object(VietnamMap, *args, **kwargs)
        return super().spawn_object(object_class, *args, **kwargs)


class RaceTrafficManager(PGTrafficManager):
    def reset(self):
        self.mode = TrafficMode.Basic if getattr(self.current_map,'is_vietnam',False) else self.engine.global_config['traffic_mode']
        return super().reset()
