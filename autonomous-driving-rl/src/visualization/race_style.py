"""Code-drawn UI surfaces and render-only race liveries; no physics objects."""

import math

from panda3d.core import (
    Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat,
    GeomVertexWriter, LineSegs, Material, NodePath, TextNode, TransparencyAttrib,
)


RACE_COLORS = (
    (.12, .76, 1., 1.),   # 0: A* - Electric Cyan (#1EBDFE)
    (1., .64, .18, 1.),   # 1: BFS - Racing Amber (#FFA42E)
    (.70, .48, 1., 1.),   # 2: Dijkstra - Neon Violet (#B27BFF)
    (.14, .88, .60, 1.),  # 3: Hybrid A* - Emerald Mint (#24E099)
    (1., .34, .48, 1.),   # 4: MPPI - Crimson Coral (#FF577A)
    (.98, .88, .30, 1.),  # 5: Player / Human - Titanium Gold (#FAE04D)
)


def tint(color, amount=.15):
    return tuple(min(1., c + amount) for c in color[:3]) + (color[3],)


def polygon(points, color, name="surface", normal=(0, -1, 0)):
    data = GeomVertexData(name, GeomVertexFormat.getV3n3c4(), Geom.UHStatic)
    vertex, normals, colors = (GeomVertexWriter(data, k) for k in ("vertex", "normal", "color"))
    for point in points:
        vertex.addData3(*point)
        normals.addData3(*normal)
        colors.addData4(*color)
    triangles = GeomTriangles(Geom.UHStatic)
    for i in range(1, len(points) - 1):
        triangles.addVertices(0, i, i + 1)
    geom = Geom(data)
    geom.addPrimitive(triangles)
    node = GeomNode(name)
    node.addGeom(geom)
    result = NodePath(node)
    result.setTwoSided(True)
    if color[3] < 1:
        result.setTransparency(TransparencyAttrib.MAlpha)
    return result


def rounded_card(width, height, color, radius=6, border=None, border_width=1.2, left_stripe=None, stripe_width=4.0):
    root = NodePath("rounded_card")

    def face(w, h, r, shade, inset=0):
        r = min(r, w / 2, h / 2)
        points = []
        for cx, cy, angle in ((w-r, r, -90), (w-r, h-r, 0), (r, h-r, 90), (r, r, 180)):
            for step in range(7):
                a = math.radians(angle + step * 15)
                points.append((inset + cx + r*math.cos(a), -inset*.01, -inset-cy-r*math.sin(a)))
        polygon(points, shade).reparentTo(root)

    face(width, height, radius, border or color)
    bw = int(round(border_width)) if border else 0
    if border and bw > 0:
        face(width - 2 * bw, height - 2 * bw, max(0, radius - bw), color, bw)
    if color[3] > 0.6 and height > 28 and width > 60:
        hl_color = (min(1.0, color[0] + 0.05), min(1.0, color[1] + 0.06), min(1.0, color[2] + 0.08), min(1.0, color[3] * 0.25))
        r_hl = max(0, radius - bw)
        hl_points = [
            (bw + r_hl, -0.015, -bw - 1),
            (width - bw - r_hl, -0.015, -bw - 1),
            (width - bw - r_hl, -0.015, -bw - 2),
            (bw + r_hl, -0.015, -bw - 2),
        ]
        polygon(hl_points, hl_color, "top_highlight").reparentTo(root)
    if left_stripe:
        sw = min(stripe_width, width / 4)
        inset = bw
        stripe_points = [
            (inset + 2, -0.02, -inset - 2),
            (inset + 2 + sw, -0.02, -inset - 2),
            (inset + 2 + sw, -0.02, -(height - inset - 2)),
            (inset + 2, -0.02, -(height - inset - 2)),
        ]
        polygon(stripe_points, left_stripe, "left_stripe").reparentTo(root)
    root.setLightOff()
    root.setShaderOff()
    return root


def draw_vector_check(parent, cx, cy, size=16, color=(.02, .04, .06, 1.), thickness=2.2):
    """Draw a clean, resolution-independent vector checkmark centered around cx, cy."""
    segs = LineSegs("vector_check")
    segs.setColor(*color)
    segs.setThickness(thickness)
    x = cx - size / 2
    y = cy - size / 2
    segs.moveTo(x + size * 0.22, -0.05, -(y + size * 0.52))
    segs.drawTo(x + size * 0.44, -0.05, -(y + size * 0.76))
    segs.drawTo(x + size * 0.82, -0.05, -(y + size * 0.28))
    node = parent.attachNewNode(segs.create())
    node.setLightOff()
    node.setShaderOff()
    return node


def draw_vector_plus(parent, cx, cy, size=14, color=(.6, .7, .8, 1.), thickness=1.8):
    """Draw a clean vector plus sign centered around cx, cy."""
    segs = LineSegs("vector_plus")
    segs.setColor(*color)
    segs.setThickness(thickness)
    x = cx - size / 2
    y = cy - size / 2
    segs.moveTo(x + size * 0.25, -0.05, -(y + size * 0.50))
    segs.drawTo(x + size * 0.75, -0.05, -(y + size * 0.50))
    segs.moveTo(x + size * 0.50, -0.05, -(y + size * 0.25))
    segs.drawTo(x + size * 0.50, -0.05, -(y + size * 0.75))
    node = parent.attachNewNode(segs.create())
    node.setLightOff()
    node.setShaderOff()
    return node


def box(parent, name, position, size, color):
    x, y, z = (s / 2 for s in size)
    root = parent.attachNewNode(name)
    root.setPos(*position)
    for normal, points in (
        ((0, 0, 1), ((-x,-y,z), (x,-y,z), (x,y,z), (-x,y,z))),
        ((0, 0,-1), ((-x,-y,-z), (-x,y,-z), (x,y,-z), (x,-y,-z))),
        ((0, 1, 0), ((-x,y,-z), (-x,y,z), (x,y,z), (x,y,-z))),
        ((0,-1, 0), ((-x,-y,-z), (x,-y,-z), (x,-y,z), (-x,-y,z))),
        ((1, 0, 0), ((x,-y,-z), (x,y,-z), (x,y,z), (x,-y,z))),
        ((-1,0, 0), ((-x,-y,-z), (-x,-y,z), (-x,y,z), (-x,y,-z))),
    ):
        polygon(points, color, name, normal).reparentTo(root)
    return root


def paint_material(color, metallic=.45, roughness=.24, emission=None):
    material = Material()
    material.setBaseColor(color)
    material.setDiffuse(color)
    material.setAmbient(tuple(c*.38 for c in color[:3]) + (1.,))
    material.setSpecular((.8, .87, 1., 1.))
    material.setShininess(96)
    material.setMetallic(metallic)
    material.setRoughness(roughness)
    if emission:
        material.setEmission(emission)
    return material


def paint_car(root, index):
    """Paint body panels only, preserving glass, tires and interior materials."""
    color = RACE_COLORS[index]
    for node in root.findAllMatches("**/body_carpaint*"):
        node.setMaterial(paint_material(color), 20)
        node.setColor(1, 1, 1, 1)
    for pattern, shade in (("**/body_glass*", (.045,.085,.13,1)),
                           ("**/body_light_red*", (1.,.06,.10,1)),
                           ("**/body_lights_yellow_front*", (.76,.93,1.,1))):
        for node in root.findAllMatches(pattern):
            node.setMaterial(paint_material(shade, .2, .16,
                             shade if "light" in pattern else None), 20)


def style_vehicle(vehicle, index, name, font):
    """Copy instanced render meshes before applying a per-car material."""
    from panda3d.bullet import BulletBodyNode
    root = vehicle.origin
    old = root.find("race_trim")
    if not old.isEmpty():
        old.removeNode()
    old_label = root.find("race_identity")
    if not old_label.isEmpty():
        old_label.removeNode()
    # MetaDrive shares loaded model nodes. A copy avoids painting all five cars
    # when the material on one car's body mesh changes.
    if not root.hasPythonTag("race_mesh_copy"):
        for child in list(root.getChildren()):
            if not isinstance(child.node(), BulletBodyNode) and child.getName() != "wheel":
                child.copyTo(root)
                child.removeNode()
        root.setPythonTag("race_mesh_copy", True)
    root.clearMaterial()
    paint_car(root, index)
    trim = root.attachNewNode("race_trim")
    carbon = (.025, .035, .052, 1.)
    box(trim, "rear_wing", (0,-1.72,.58), (1.64,.25,.065), carbon)
    for x in (-.52,.52):
        box(trim, "wing_support", (x,-1.72,.35), (.06,.10,.40), carbon)
    box(trim, "front_splitter", (0,1.98,-.22), (1.66,.22,.055), carbon)
    for x in (-.82,.82):
        box(trim, "side_skirt", (x,0,-.22), (.045,2.8,.065), carbon)
    # A floating, compact identity tag remains readable from either camera.
    text = TextNode("race_identity")
    text.setText(f"{index+1:02d}  {name.replace(' · thử nghiệm', '')}")
    if font is not None:
        text.setFont(font)
    text.setAlign(TextNode.ACenter)
    text.setTextColor(*RACE_COLORS[index])
    text.setCardColor(.018,.027,.045,.90)
    text.setCardAsMargin(.24,.24,.12,.10)
    text.setCardDecal(True)
    label = root.attachNewNode(text)
    label.setPos(0,0,1.65)
    label.setScale(.20)
    label.setBillboardPointEye()
    label.setLightOff()
    label.setShaderOff()
    vehicle._panda_color = RACE_COLORS[index]
    return label
