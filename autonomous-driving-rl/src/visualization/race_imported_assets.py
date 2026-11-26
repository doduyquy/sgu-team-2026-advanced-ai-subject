"""Local CC BY glTF artwork, detached from the simulator's Bullet chassis."""

from pathlib import Path
from panda3d.core import NodePath, Vec3
from src.visualization.race_graphics import limit_visual_textures


ASSET_ROOT=Path(__file__).resolve().parents[2]/'assets'/'vehicles'
IMPORTED_MODELS={
    'lamborghini': {'title':'Lamborghini Aventador','heading':180.,'full_wheels':False},
    'tesla': {'title':'Tesla Cybertruck','heading':0.,'full_wheels':False},
}
_CACHE={}


def _center(path, relative):
    lo,hi=path.getTightBounds(relative)
    return (lo+hi)*.5


def _load(model):
    if model in _CACHE:
        return _CACHE[model]
    from gltf import load_model
    config=IMPORTED_MODELS[model]
    root=NodePath('normalized_vehicle')
    artwork=NodePath(load_model(str(ASSET_ROOT/model/'model.glb')))
    artwork.reparentTo(root)
    artwork.setH(config['heading'])
    parts=[]
    if not config['full_wheels']:
        parts=[artwork.find('**/'+name) for name in ('Wheel_FL','Wheel_FR','Wheel_RL','Wheel_RR')]
        centers=[_center(p,root) for p in parts]
        front=(centers[0]+centers[1])*.5
        rear=(centers[2]+centers[3])*.5
        scale=(1.05234+1.4166)/abs(front.y-rear.y)
    else:
        lo,hi=artwork.getTightBounds(root)
        scale=4.4/(hi.y-lo.y)
    artwork.setScale(scale)
    lo,hi=artwork.getTightBounds(root)
    if parts:
        center=sum((_center(p,root) for p in parts),Vec3(0,0,0))*.25
        offset=Vec3(-center.x,-.18213-center.y,-.20-lo.z)
    else:
        offset=Vec3(-(lo.x+hi.x)*.5,-(lo.y+hi.y)*.5,-.20-lo.z)
    artwork.setPos(offset)
    wheels={}
    for part in parts:
        center=_center(part,root)
        key=(center.y>-.18213,center.x>0)
        wheel=NodePath('race_wheel_visual')
        copy=part.copyTo(wheel)
        copy.setMat(part.getMat(root))
        copy.setPos(copy.getPos()-center)
        limit_visual_textures(wheel)
        wheel.flattenStrong()
        wheel.setPythonTag('model',model)
        wheel.setPythonTag('source_center',tuple(center))
        lo,hi=wheel.getTightBounds()
        wheel.setPythonTag('source_radius',float((hi.z-lo.z)*.5))
        wheels[key]=wheel
        part.removeNode()
    if model=='tesla':
        from src.visualization.race_style import box,paint_material
        from src.visualization.race_car_mesh import panel
        glass_color=(.014,.025,.044,1.)
        for side in (-1,1):
            glass=panel(root,'cyber_side_glass',[
                (side*.91,-.82,.645),(side*.80,-.82,.91),
                (side*.79,.03,1.00),(side*.91,1.14,.645),
            ],glass_color)
            glass.setMaterial(paint_material(glass_color,.25,.16),20)
            pillar=panel(root,'cyber_window_pillar',[
                (side*.915,-.285,.645),(side*.815,-.285,.966),
                (side*.815,-.25,.969),(side*.915,-.25,.645),
            ],(.012,.017,.023,1.))
            pillar.setMaterial(paint_material((.012,.017,.023,1.),.1,.35),20)
        for name,position,size,color in (
            ('cyber_front_led',(0,1.655,.495),(1.44,.025,.021),(.78,.9,1.,1.)),
            ('cyber_rear_led',(0,-2.20,.685),(1.74,.025,.022),(1.,.012,.006,1.)),
        ):
            lamp=box(root,name,position,size,color)
            material=paint_material(color,.05,.18);material.setEmission((*color[:3],1.))
            lamp.setMaterial(material,20)
    limit_visual_textures(root)
    root.flattenStrong()
    root.setPythonTag('asset_origin','cc_by_downloaded')
    _CACHE[model]=(root,wheels)
    return root,wheels


def build_imported_body(parent,model):
    body,_=_load(model)
    result=body.copyTo(parent)
    result.setName('near_body')
    result.setPythonTag('imported_asset',True)
    return result


def build_imported_wheel(parent,model,front,positive_x,radius=None,axle_half_width=None):
    _,wheels=_load(model)
    result=wheels[(front,positive_x)].copyTo(parent)
    if radius is not None:
        result.setScale(radius/result.getPythonTag('source_radius'))
    if axle_half_width is not None:
        # Move only along the wheel's axle. A vertical offset under Bullet's
        # rolling node would orbit the hub and make the wheel visibly wobble.
        source_x=result.getPythonTag('source_center')[0]
        result.setX(source_x-(axle_half_width if positive_x else -axle_half_width))
    return result
