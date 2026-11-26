"""CC BY Lamborghini/Tesla artwork and an original VF3 illustration.

Only visuals change; MetaDrive chassis and wheel physics stay shared.
Attribution and modifications are recorded in assets/vehicles/ATTRIBUTION.md.
"""

from panda3d.core import LODNode, NodePath, Vec3
from src.visualization.race_style import box, polygon, paint_material, RACE_COLORS
from src.visualization.race_car_mesh import build_detailed, build_wheel
from src.visualization.race_imported_assets import (
    IMPORTED_MODELS, build_imported_body, build_imported_wheel,
)

CAR_MODELS = ('lamborghini', 'tesla', 'vf3')
CAR_NAMES = {'lamborghini': 'Lamborghini Aventador', 'tesla': 'Tesla Cybertruck', 'vf3': 'VinFast VF3'}


def hull(parent, name, sections, color):
    """Cross sections (y, half width, bottom z, top z), along vehicle length."""
    root = parent.attachNewNode(name)
    rings = [[(-w,y,lo),(w,y,lo),(w,y,hi),(-w,y,hi)] for y,w,lo,hi in sections]
    def face(points):
        normal=(Vec3(*points[1])-Vec3(*points[0])).cross(Vec3(*points[2])-Vec3(*points[0]))
        normal.normalize()
        polygon(points,color,name,normal=normal).reparentTo(root)
    face(rings[0][::-1])
    face(rings[-1])
    for a,b in zip(rings,rings[1:]):
        for k in range(4):
            face([a[k],b[k],b[(k+1)%4],a[(k+1)%4]])
    root.setMaterial(paint_material(color, .55, .26), 10)
    return root


_BODY_CACHE = {}


def _batch_static_body(root):
    """Vertex colours carry the livery; three shared materials batch surfaces."""
    if root.hasPythonTag('studio_mesh'):
        root.flattenStrong()
        return
    if root.hasPythonTag('imported_asset'):
        root.flattenStrong()
        return
    white = (1., 1., 1., 1.)
    body = paint_material(white, .4, .30)
    glass = paint_material(white, .12, .16)
    trim = paint_material(white, .15, .48)
    for node in root.findAllMatches('**'):
        node.clearMaterial()
    root.setMaterial(body, 10)
    for node in root.getChildren():
        if 'glass' in node.getName() or 'roof' in node.getName():
            node.setMaterial(glass, 20)
        elif any(k in node.getName() for k in ('wing', 'strut', 'sill', 'intake', 'grille', 'splitter', 'bumper', 'arch')):
            node.setMaterial(trim, 20)
    root.flattenStrong()


def _build_far(parent, model, color):
    root = parent.attachNewNode('far_body')
    if model == 'vf3':
        hull(root, 'body', [(-2.04,.91,.015,.96),(1.69,.87,.04,.90)], color)
        hull(root, 'glass', [(-1.93,.81,.94,1.57),(.32,.84,.94,1.61),(.86,.84,.94,1.12)], (.012,.024,.032,1))
        box(root,'roof',(0,-.76,1.65),(1.74,2.33,.08),(.012,.017,.022,1))
    elif model=='tesla':
        hull(root,'body',[(-2.20,.89,-.17,.61),(-.53,.96,-.17,.85),(1.65,.91,-.14,.58)],(.48,.52,.57,1))
        hull(root,'glass',[(-1.02,.78,.62,.77),(.08,.78,.62,1.01),(1.26,.83,.59,.61)],(.012,.026,.052,1))
    else:
        low=-.08 if model=='lamborghini' else -.28
        hull(root, 'body', [(-2.10,.78,low,.22),(.98,.86,low,.29),(2.11,.73,low+.04,.15)], color)
        top = .76 if model == 'lamborghini' else .92
        hull(root, 'glass', [(-1.25,.63,.25,top),(.24,.62,.18,top),(.98,.66,.10,.22)], (.08,.17,.23,1))
    _batch_static_body(root)
    return root


def build_car(parent, model, color):
    """Copy cached immutable visual geometry; keep wheels/Bullet outside LOD."""
    if model not in CAR_MODELS:
        raise ValueError('Unknown car model')
    key = (model, tuple(color))
    if key not in _BODY_CACHE:
        prototype = NodePath('race_custom_body')
        lod = prototype.attachNewNode(LODNode('body_lod'))
        close = build_imported_body(lod,model) if model in IMPORTED_MODELS else build_detailed(lod, model, color)
        close.setName('near_body')
        _batch_static_body(close)
        far=_build_far(lod, model, color)
        if model in IMPORTED_MODELS and IMPORTED_MODELS[model]['full_wheels']:
            for x in (-.815,.815):
                for y in (-1.4166,1.05234):
                    wheel=build_wheel(far,model=model)
                    wheel.setPos(x,y,.113)
            far.flattenStrong()
        # The default overhead camera is ~40 m above the focused vehicle.
        # Keep its detail; simplify only cars farther away from that camera.
        lod.node().addSwitch(55., 0.)
        lod.node().addSwitch(100000., 55.)
        prototype.setPythonTag('asset_origin', 'cc_by_downloaded' if model in IMPORTED_MODELS else 'procedural_original')
        _BODY_CACHE[key] = prototype
    return _BODY_CACHE[key].copyTo(parent)


def install_car(vehicle, model, index):
    from panda3d.bullet import BulletBodyNode
    from src.visualization.race_graphics import limit_visual_textures
    wheel_index=0
    for child in vehicle.origin.getChildren():
        name = child.getName()
        if name == 'race_custom_body':
            child.removeNode()
        elif not isinstance(child.node(), BulletBodyNode) and name not in ('wheel','race_identity'):
            child.hide()
        elif name == 'wheel':
            limit_visual_textures(child)
            custom = child.find('race_wheel_visual')
            if custom.isEmpty() or custom.getPythonTag('model') != model:
                for old in list(child.getChildren()):
                    if old.getName() == 'race_wheel_visual':
                        old.removeNode()
                    else:
                        old.hide()
                if model in IMPORTED_MODELS:
                    if not IMPORTED_MODELS[model]['full_wheels']:
                        build_imported_wheel(child,model,wheel_index<2,wheel_index%2==0,
                                             radius=getattr(vehicle,'TIRE_RADIUS',.313),
                                             axle_half_width=getattr(vehicle,'LATERAL_TIRE_TO_CENTER',.815))
                else:
                    visual=build_wheel(child, getattr(vehicle,'TIRE_RADIUS',.313),
                                       getattr(vehicle,'TIRE_WIDTH',.25), model)
                    if model=='vf3':
                        visual.setX(.11 if wheel_index%2==0 else -.11)
            wheel_index+=1
    body = build_car(vehicle.origin, model, RACE_COLORS[index])
    if model in IMPORTED_MODELS and not IMPORTED_MODELS[model]['full_wheels']:
        # Imported artwork is normalized to a static showroom ground plane.
        # Bullet's loaded suspension puts its wheel hubs ~0.33 m lower than
        # those preview pivots. Lower the render body only, retaining physics.
        body.setZ(-getattr(vehicle,'TIRE_RADIUS',.313)-.02)
    vehicle._race_model = model
    return body
