"""Leave the desktop pointer free while MetaDrive follows a race vehicle.

The pinned MainCamera recenters on track(), pointer re-entry and idle input.
Its instance methods are adapted here instead of editing the pinned package
or replacing its chase/physics tasks. Explicit hold-drag adds a view offset
after the normal camera update; moving the pointer alone never changes it.
"""

from types import MethodType
import math


def _ignore_center(camera):
    """Camera changes must never warp the desktop pointer."""


def _ignore_mouse_look(camera):
    camera.mouse_rotate = 0.
    camera.last_mouse_pos = 0.
    camera.static_timer = 0
    camera.move_into_window_timer = 0
    camera._in_recover = False
    camera._last_frame_has_mouse = False


class RaceCameraInput:
    def __init__(self, engine):
        from panda3d.core import WindowProperties
        self.engine = engine
        self.camera = engine.main_camera
        self._original = {}
        self.yaw = self.pitch = 0.
        self.pan = (0., 0.)
        self._view_key = None
        self._task_name = f'race_drag_camera_{id(self)}'
        if self.camera is not None:
            # Track which methods were instance attributes so close() restores
            # the class descriptor when appropriate, rather than leaking a shim.
            for name, replacement in (('set_mouse_to_center', _ignore_center),
                                      ('update_mouse_info', _ignore_mouse_look)):
                self._original[name] = self.camera.__dict__.get(name)
                setattr(self.camera, name, MethodType(replacement, self.camera))
            self.camera.update_mouse_info()
        props = WindowProperties()
        props.setCursorHidden(False)
        props.setMouseMode(WindowProperties.MAbsolute)
        engine.win.requestProperties(props)
        # Run after MetaDrive's chase/bird task, before rendering. Its normal
        # following still owns the base pose; this adds a user-controlled view.
        engine.taskMgr.add(self._apply_task, self._task_name, sort=40)

    def _sync_view(self):
        if self.camera is None:
            return
        key = (self.camera.current_track_agent, self.camera.is_bird_view_camera())
        if key != self._view_key:
            self.yaw = self.pitch = 0.
            self.pan = (0., 0.)
            self._view_key = key

    def drag(self, dx, dy):
        """Only the pointer controller calls this while a scene drag is held."""
        self._sync_view()
        if self.camera is None:
            return
        if self.camera.is_bird_view_camera():
            scale = max(1., self.camera.top_down_camera_height)/800.
            right = self.camera.camera.getQuat(self.engine.render).getRight()
            up = self.camera.camera.getQuat(self.engine.render).getUp()
            self.pan = (self.pan[0] + (-dx*right.x+dy*up.x)*scale,
                        self.pan[1] + (-dx*right.y+dy*up.y)*scale)
        else:
            self.yaw = (self.yaw + dx*.25 + 180.) % 360. - 180.
            self.pitch = max(-30., min(60., self.pitch+dy*.18))

    def _apply_task(self, task):
        from panda3d.core import Vec3
        if self.camera is None:
            return task.cont
        self._sync_view()
        node = self.camera.camera
        if self.camera.is_bird_view_camera():
            if self.pan != (0., 0.):
                node.setPos(self.camera.camera_x+self.pan[0],
                            self.camera.camera_y+self.pan[1],self.camera.top_down_camera_height)
                node.lookAt(self.camera.camera_x+self.pan[0],self.camera.camera_y+self.pan[1],0)
        elif (self.yaw or self.pitch) and self.camera.current_track_agent is not None:
            pivot = self.camera.current_track_agent.chassis.getPos(self.engine.render)+Vec3(0,0,2)
            offset = node.getPos(self.engine.render)-pivot
            radius = offset.length()
            azimuth = math.atan2(offset.y,offset.x)+math.radians(self.yaw)
            elevation = math.atan2(offset.z,math.hypot(offset.x,offset.y))+math.radians(self.pitch)
            elevation = max(math.radians(3.),min(math.radians(75.),elevation))
            node.setPos(self.engine.render,pivot+Vec3(radius*math.cos(elevation)*math.cos(azimuth),
                                                    radius*math.cos(elevation)*math.sin(azimuth),
                                                    radius*math.sin(elevation)))
            node.lookAt(self.engine.render,pivot)
        return task.cont

    def close(self):
        self.engine.taskMgr.remove(self._task_name)
        for name, original in self._original.items():
            if original is None:
                self.camera.__dict__.pop(name, None)
            else:
                setattr(self.camera, name, original)
        self._original.clear()
