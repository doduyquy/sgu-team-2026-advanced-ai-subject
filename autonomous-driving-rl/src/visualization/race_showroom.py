"""An isolated render-to-texture garage preview, with no simulator calls."""

import math
from array import array
from types import SimpleNamespace

from panda3d.core import (AmbientLight, DirectionalLight, LineSegs, NodePath,
                         Texture, TransparencyAttrib, Vec4)

from src.visualization.race_style import RACE_COLORS, paint_car, polygon


_STUDIO_ENVIRONMENT = None


def studio_environment():
    """Tiny analytic softboxes, baked once; no live reflection camera."""
    global _STUDIO_ENVIRONMENT
    if _STUDIO_ENVIRONMENT is None:
        from simplepbr.envmap import EnvMap
        size = 32
        pixels = array('f')
        for face in range(6):
            for y in range(size):
                v = 2*(y+.5)/size-1
                for x in range(size):
                    u = 2*(x+.5)/size-1
                    direction = ((1,-v,-u),(-1,-v,u),(u,1,v),
                                 (u,-1,-v),(u,-v,1),(u,v,-1))[face]
                    length = math.sqrt(sum(c*c for c in direction))
                    dx,dy,dz = (c/length for c in direction)
                    base = .075 + .16*max(0,dz)
                    # Broad ceiling strip, vertical white key and blue rim.
                    ceiling = 3.5 if dz>.65 and abs(dx)<.16 else 0
                    key = 2.8 if dx>.55 and -.55<dy<.3 and -.1<dz<.85 else 0
                    rim = 1.1 if dy<-.65 and abs(dx)<.4 and dz>-.2 else 0
                    r,g,b = base+ceiling+key, base*1.12+ceiling+key+rim*.7, base*1.35+ceiling+key+rim
                    pixels.extend((b,g,r))
        cubemap = Texture('garage_analytic_softboxes')
        cubemap.setupCubeMap(size,Texture.TFloat,Texture.FRgb32)
        cubemap.setRamImage(pixels.tobytes())
        _STUDIO_ENVIRONMENT = EnvMap(cubemap,prefiltered_size=32,
                                     prefiltered_samples=8,blocking_prepare=True)
    return _STUDIO_ENVIRONMENT


class GaragePreview:
    def __init__(self, engine, vehicle):
        self.engine = engine
        from src.visualization.race_graphics import get_graphics_profile
        profile = get_graphics_profile()
        width, height = profile.preview_size
        self.scene = NodePath("garage_preview_scene")
        self.buffer = engine.win.makeTextureBuffer("garage_car_preview", width, height)
        if self.buffer is None:
            raise RuntimeError("Cannot create garage preview buffer")
        self.buffer.setClearColor(Vec4(0.01217, 0.01489, 0.01990, 1.0))
        self.camera = engine.makeCamera(self.buffer, scene=self.scene)
        self.camera.reparentTo(self.scene)
        self.camera.node().getLens().setFov(35)
        self.camera.node().getLens().setAspectRatio(width/height)
        self.camera.node().getLens().setFilmOffset(-.16,0)
        self.camera.setPos(8.6, 12.2, 5.8)
        self.camera.lookAt(0, 0, .28)
        ambient = AmbientLight("showroom_ambient")
        ambient.setColor((.14, .17, .22, 1))
        self.scene.setLight(self.scene.attachNewNode(ambient))
        for name, color, hpr in (("key", (2.2, 2.1, 1.95, 1), (-35, -55, 0)),
                                 ("fill", (.28, .42, .68, 1), (15, -30, 0)),
                                 ("rim", (.55, .85, 1.3, 1), (140, -20, 0))):
            light = DirectionalLight(name)
            light.setColor(color)
            node = self.scene.attachNewNode(light)
            node.setHpr(*hpr)
            self.scene.setLight(node)
        self.car = self.scene.attachNewNode("showroom_car")
        from src.visualization.garage_orbit import GarageOrbit
        self.orbit = GarageOrbit()
        self.wheel_radius, self.wheel_width = vehicle.TIRE_RADIUS, vehicle.TIRE_WIDTH
        self.wheel_positions=[]
        # Copy just visual children. No Bullet body or simulator state enters
        # the preview; moving this scene cannot move a race vehicle.
        from panda3d.bullet import BulletBodyNode
        wheel_index = 0
        for child in vehicle.origin.getChildren():
            if not isinstance(child.node(), BulletBodyNode) and child.getName() != "race_identity":
                copy = child.copyTo(self.car)
                if child.getName() == "wheel":
                    copy.setPos(vehicle.LATERAL_TIRE_TO_CENTER * (1 if wheel_index%2 == 0 else -1),
                                vehicle.FRONT_WHEELBASE if wheel_index < 2 else -vehicle.REAR_WHEELBASE,
                                vehicle.TIRE_RADIUS - vehicle.CHASSIS_TO_WHEEL_AXIS)
                    copy.setHpr(0,0,0)
                    self.wheel_positions.append(tuple(copy.getPos()))
                    wheel_index += 1
        self.car.clearMaterial()
        self.car.setH(-16)
        self.ring = self.scene.attachNewNode("stage_rings")
        for layer, (radius, color) in enumerate(((3.03, (.034, .052, .078, 1)), (3., (.011, .019, .031, 1)),
                                                (2.85, (.025, .038, .057, 1)), (2.82, (.014, .024, .039, 1)))):
            points = [(radius*math.cos(i*math.tau/96), radius*math.sin(i*math.tau/96), -.25+layer*.002)
                      for i in range(96)]
            disk = polygon(points, color, normal=(0, 0, 1))
            disk.setLightOff()
            disk.setShaderOff()
            disk.reparentTo(self.ring)
        ring_segs = LineSegs("stage_accent_ring")
        ring_segs.setThickness(1.5)
        ring_segs.setColor(.4, .5, .6, .4)
        for i in range(65):
            a = i * math.tau / 64
            (ring_segs.moveTo if i == 0 else ring_segs.drawTo)(2.94 * math.cos(a), 2.94 * math.sin(a), -.241)
        self.glow_ring = self.scene.attachNewNode(ring_segs.create())
        self.glow_ring.setLightOff()
        self.glow_ring.setShaderOff()

        # Layered translucent ellipses approximate contact occlusion cheaply.
        shadow = self.scene.attachNewNode('contact_shadow')
        self.shadow = shadow
        for layer in range(10):
            scale = 1.25-layer*.055
            points=[(1.02*scale*math.cos(i*math.tau/48),2.02*scale*math.sin(i*math.tau/48),
                     -.237+layer*.0002) for i in range(48)]
            blob=polygon(points,(0,0,0,.065))
            blob.reparentTo(shadow)
            blob.setLightOff(); blob.setShaderOff(); blob.setDepthWrite(False)
            blob.setTransparency(TransparencyAttrib.MAlpha)
        shadow.setH(-16)
        self.index = self.model = None
        self.select(0)
        from simplepbr import Pipeline
        self._task_name = f"garage_pbr_{id(self)}"
        tasks = SimpleNamespace(add=lambda callback, name, **options:
                                engine.taskMgr.add(callback, self._task_name, **options))
        self.pipeline = Pipeline(render_node=self.scene, window=self.buffer,
                                 camera_node=self.camera, taskmgr=tasks, msaa_samples=max(2,profile.msaa_samples),
                                 enable_shadows=False, exposure=-.45, env_map=studio_environment())

    @property
    def texture(self):
        return self.buffer.getTexture()

    def select(self, index, model=None):
        model=model or self.model or 'lamborghini'
        if index == self.index and model == self.model:
            return
        from src.visualization.race_vehicle_models import build_car
        from src.visualization.race_car_mesh import build_wheel
        from src.visualization.race_imported_assets import IMPORTED_MODELS, build_imported_wheel
        self.index, self.model = index, model
        wheel_index=0
        for child in self.car.getChildren():
            if child.getName() == 'race_custom_body':
                child.removeNode()
            elif child.getName() != 'wheel':
                child.hide()
            else:
                child.setPos(*self.wheel_positions[wheel_index])
                child.setHpr(0,0,0)
                for visual in list(child.getChildren()):
                    if visual.getName() == 'race_wheel_visual':
                        visual.removeNode()
                    else:
                        visual.hide()
                if model in IMPORTED_MODELS:
                    if not IMPORTED_MODELS[model]['full_wheels']:
                        wheel=build_imported_wheel(child,model,wheel_index<2,wheel_index%2==0)
                        child.setPos(*wheel.getPythonTag('source_center'))
                else:
                    visual=build_wheel(child,self.wheel_radius,self.wheel_width,model)
                    if model=='vf3':
                        visual.setX(.11 if wheel_index%2==0 else -.11)
                wheel_index+=1
        build_car(self.car, model, RACE_COLORS[index])
        paint_car(self.car, index)
        if hasattr(self, "glow_ring") and not self.glow_ring.isEmpty():
            self.glow_ring.setColor(*RACE_COLORS[index])
        self.reset_view()

    def reset_view(self):
        from src.visualization.garage_orbit import GarageOrbit
        self.orbit=GarageOrbit()
        if self.model=='vf3':
            self.orbit.elevation=15.
            self.orbit.distance=17.5
        elif self.model=='tesla':
            self.orbit.elevation=16.
            self.orbit.distance=13.6
        self.apply_view()

    def apply_view(self):
        self.car.setH(self.orbit.yaw)
        if hasattr(self,'shadow'):
            self.shadow.setH(self.orbit.yaw)
        elevation=math.radians(self.orbit.elevation)
        azimuth=math.radians(35)
        radius=self.orbit.distance
        target=.58 if self.model=='vf3' else .45 if self.model=='tesla' else .28
        self.camera.setPos(radius*math.cos(elevation)*math.sin(azimuth),
                           radius*math.cos(elevation)*math.cos(azimuth),
                           target+radius*math.sin(elevation))
        self.camera.lookAt(0,0,target)

    def set_active(self, active):
        self.buffer.setActive(active)
        for buffer in self.pipeline._filtermgr.buffers:
            buffer.setActive(active)

    def destroy(self):
        self.engine.taskMgr.remove(self._task_name)
        self.pipeline._filtermgr.cleanup()
        self.camera.removeNode()
        self.engine.graphicsEngine.removeWindow(self.buffer)
        self.scene.removeNode()
