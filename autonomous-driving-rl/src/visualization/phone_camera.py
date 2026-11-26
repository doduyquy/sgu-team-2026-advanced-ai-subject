"""A second world camera, independent of the laptop's spectator camera."""

import time
from panda3d.core import GraphicsOutput, Texture, Vec4


class PhoneCamera:
    VIEWS = ('chase', 'hood', 'driver')

    def __init__(self, engine, frames, width=960, height=540):
        self.engine, self.frames = engine, frames
        self.buffer = engine.win.makeTextureBuffer('phone_view', width, height)
        if self.buffer is None:
            raise RuntimeError('Không tạo được camera điện thoại')
        self.texture = Texture('phone_rgb')
        self.buffer.addRenderTexture(self.texture, GraphicsOutput.RTMTriggeredCopyRam, GraphicsOutput.RTPColor)
        self.buffer.setClearColor(Vec4(.38,.62,.83,1))
        self.camera = engine.makeCamera(self.buffer, scene=engine.render)
        # MetaDrive's terrain uses a camera tag state, and its PBR scene is
        # linear HDR. A raw Panda camera would show a black road. Reuse the
        # pinned RGB sensor's terrain state and tone-map this buffer only.
        from metadrive.constants import CamMask, CameraTagStateKey, Semantics
        from metadrive.engine.core.terrain import Terrain
        from metadrive.third_party.simplepbr import _load_shader_str
        from direct.filter.FilterManager import FilterManager
        from panda3d.core import FrameBufferProperties, Shader
        self.camera.node().setCameraMask(CamMask.RgbCam)
        self.camera.node().setTagStateKey(CameraTagStateKey.RGB)
        terrain_state = getattr(engine, '_race_terrain_render_state', None)
        if terrain_state is None:
            terrain_state = Terrain.make_render_state(engine,'terrain.vert.glsl','terrain.frag.glsl')
        self.camera.node().setTagState(Semantics.TERRAIN.label, terrain_state)
        self.manager = FilterManager(self.buffer,self.camera)
        props = FrameBufferProperties()
        props.setFloatColor(True)
        props.setRgbaBits(16,16,16,16)
        props.setDepthBits(24)
        props.setMultisamples(2)
        scene_texture=Texture('phone_hdr')
        quad=self.manager.renderSceneInto(colortex=scene_texture,fbprops=props)
        quad.setShader(Shader.make(Shader.SL_GLSL,
            vertex=_load_shader_str('post.vert',{}),fragment=_load_shader_str('tonemap.frag',{})))
        quad.setShaderInput('tex',scene_texture)
        quad.setShaderInput('exposure',1.)
        lens = self.camera.node().getLens()
        lens.setAspectRatio(width/height)
        lens.setFov(engine.global_config['camera_fov'])
        # The real sky dome is 20 km wide. The old 1200 m far plane clipped
        # it away, replacing the original clouds with the flat clear color.
        lens.setNearFar(.08, max(50000,engine.cam.node().getLens().getFar()))
        self.view, self.vehicle = 'chase', None
        self.last_capture = 0.
        self.next_capture = 0.
        self.enabled = self.pending_capture = False
        self.render_gate = True
        self.width,self.height = width,height
        self.task_name = f'phone_capture_{id(self)}'
        engine.taskMgr.add(self.capture, self.task_name, sort=55)
        engine.taskMgr.add(self.prepare, self.task_name+'_prepare', sort=45)
        self._set_buffers(False)

    def track(self, vehicle, view=None):
        if view is not None:
            if view not in self.VIEWS:
                raise ValueError('Unknown phone camera')
            self.view = view
        self.vehicle = vehicle
        self.camera.reparentTo(vehicle.origin)
        if self.view == 'chase':
            self.camera.setPos(0,-self.engine.global_config['camera_dist'],
                               self.engine.global_config['camera_height']+vehicle.HEIGHT/2)
            # Aim at the car rather than above its roof: a landscape phone
            # must show the rear wheels, with room below for touch controls.
            self.camera.lookAt(0,0,.9)
        elif self.view == 'hood':
            self.camera.setPos(0,1.35,.85)
            self.camera.lookAt(0,25,.70)
        else:
            self.camera.setPos(-.34,.36,.89)
            self.camera.lookAt(-.34,25,.78)
        # Hide only our car's body and identity in first person using a camera
        # mask. Other vehicles remain visible on both displays.
        from metadrive.constants import CamMask
        bit = CamMask.RgbCam
        self.camera.node().setCameraMask(bit)
        for node in vehicle.origin.getChildren():
            if node.getName() == 'race_identity':
                node.hide(bit)
            elif node.getName() == 'race_custom_body':
                node.show(bit) if self.view == 'chase' else node.hide(bit)

    def set_active(self, active):
        self.enabled=bool(active and self.vehicle is not None)
        if not self.enabled:
            self._set_buffers(False)

    def _set_buffers(self, active):
        self.buffer.setActive(active)
        for buffer in self.manager.buffers:
            buffer.setActive(active)

    def prepare(self, task):
        now=time.perf_counter()
        due=self.enabled and self.render_gate and now>=self.next_capture
        self._set_buffers(due)
        self.pending_capture=due
        if due:
            self.buffer.triggerCopy()
            self.next_capture=max(self.next_capture+1/30,now-1/30)
        return task.cont

    def capture(self, task):
        if self.pending_capture:
            self.pending_capture=False
            data = self.texture.getRamImageAs('RGB')
            if data:
                import numpy as np
                rgb = np.frombuffer(data, dtype=np.uint8).reshape(
                    self.texture.getYSize(), self.texture.getXSize(), 3)[::-1].copy()
                self.frames.put(rgb)
                self.last_capture = time.perf_counter()
        return task.cont

    def detach(self):
        # Reset destroys all vehicle origins: preserve this camera beforehand.
        self.camera.reparentTo(self.engine.render)
        self.vehicle = None
        self.set_active(False)

    def destroy(self):
        self.engine.taskMgr.remove(self.task_name)
        self.engine.taskMgr.remove(self.task_name+'_prepare')
        self.manager.cleanup()
        if self.camera in self.engine.camList:
            self.engine.camList.remove(self.camera)
        self.camera.removeNode()
        self.engine.graphicsEngine.removeWindow(self.buffer)
