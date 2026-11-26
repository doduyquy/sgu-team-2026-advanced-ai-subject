"""Continuous RGB lane ribbons, independent of terrain raster resolution."""

import numpy as np


def marking_paths(features, sample_interval=.25, stripe_length=2.):
    """Match pinned PGBlock's 2 m paint / 2 m gap without straightening curves."""
    from metadrive.constants import MapTerrainSemanticColor, MetaDriveType
    skip = round(stripe_length / sample_interval)
    for feature in features.values():
        kind = feature['type']
        if not (MetaDriveType.is_road_line(kind) or MetaDriveType.is_road_boundary_line(kind)):
            continue
        points = np.asarray(feature['polyline'], dtype=float)[:, :2]
        yellow = MapTerrainSemanticColor.get_color(kind) == MapTerrainSemanticColor.YELLOW
        if MetaDriveType.is_broken_line(kind):
            for start in range(0, len(points)-1, skip*2):
                if start+skip < len(points):
                    yield points[start:start+skip+1], yellow
        elif len(points) > 1:
            yield points, yellow


def ribbon_vertices(points, width):
    """Shared mitered edges prevent cracks at joins and follow the real polyline."""
    points = np.asarray(points, dtype=float)
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-7]
    points = points[keep]
    if len(points) < 2:
        return np.empty((0, 2, 2))
    delta = np.diff(points, axis=0)
    tangent = delta / np.linalg.norm(delta, axis=1)[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    miter = np.vstack((normal[0], normal[:-1]+normal[1:], normal[-1]))
    miter /= np.maximum(np.linalg.norm(miter, axis=1)[:, None], 1e-7)
    denominator = np.einsum('ij,ij->i', miter, np.vstack((normal, normal[-1])))
    offset = miter * (width/2 / np.maximum(denominator, .25))[:, None]
    return np.stack((points+offset, points-offset), axis=1)


def rebuild_markings(terrain):
    from metadrive.constants import CamMask
    from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexData,
                             GeomVertexFormat, GeomVertexWriter)

    previous = getattr(terrain, '_race_markings', None)
    if previous is not None:
        previous.removeNode()
    current_map = terrain.engine.current_map
    if current_map is None:
        terrain._race_markings = None
        return
    data = GeomVertexData('race_lane_paint', GeomVertexFormat.getV3c4(), Geom.UHStatic)
    vertex, color = GeomVertexWriter(data, 'vertex'), GeomVertexWriter(data, 'color')
    triangles = GeomTriangles(Geom.UHStatic)
    count = 0
    features = current_map.get_map_features(interval=.25)
    for points, yellow in marking_paths(features):
        # Same physical widths as the reference 2/3 pixel strokes at 22 px/m.
        ribbon = ribbon_vertices(points, (3 if yellow else 2)/22)
        paint = (1., .78, 0., 1.) if yellow else (1., 1., 1., 1.)
        for left, right in ribbon:
            for point in (left, right):
                vertex.addData3(float(point[0]), float(point[1]), .015)
                color.addData4(*paint)
        for i in range(len(ribbon)-1):
            a = count+2*i
            triangles.addVertices(a, a+1, a+2)
            triangles.addVertices(a+1, a+3, a+2)
        count += 2*len(ribbon)
    geometry = Geom(data)
    geometry.addPrimitive(triangles)
    node = GeomNode('race_road_markings')
    node.addGeom(geometry)
    root = terrain.engine.render.attachNewNode(node)
    root.setShaderOff()
    root.setLightOff()
    root.hide(CamMask.AllOn)
    root.show(CamMask.MainCam | CamMask.RgbCam)
    terrain._race_markings = root
