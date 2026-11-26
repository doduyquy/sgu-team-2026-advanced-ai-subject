"""Regress actual pinned track()/mouse update behavior without cursor injection."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from PySide6.QtCore import QProcess  # noqa: F401 - Windows DLL order
from panda3d.core import NodePath, WindowProperties
from metadrive.engine.core.main_camera import MainCamera
from src.visualization.race_camera_input import RaceCameraInput


class RaceCameraInputTests(unittest.TestCase):
    def camera(self):
        engine = MagicMock()
        camera = MainCamera.__new__(MainCamera)
        camera.enable_cuda = False
        camera.engine = engine
        camera.camera = NodePath('camera')
        engine.render=NodePath('render')
        camera.camera.reparentTo(engine.render)
        camera.current_track_agent=None
        engine.task_manager.hasTaskNamed.return_value=False
        camera.FOLLOW_LANE = False
        camera.queue_length = 3
        camera.mouse_rotate = .7
        engine.main_camera = camera
        engine.mouseWatcherNode.hasMouse.return_value = True
        engine.mouseWatcherNode.getMouseX.return_value = .8
        return engine, camera

    def test_repeated_track_and_pointer_reentry_never_warp_or_rotate(self):
        engine, camera = self.camera()
        control = RaceCameraInput(engine)
        try:
            for _ in range(4):
                # Real pinned track() is called by the C toggle and V focus.
                camera.track(SimpleNamespace(position=(10., 20.)))
                camera.update_mouse_info()
                self.assertEqual(camera.mouse_rotate, 0)
                # Re-entry and the old idle trigger used to call movePointer.
                camera._last_frame_has_mouse = False
                camera.static_timer = camera.STATIC_MOUSE_HOLD_TIME + 1
                camera.update_mouse_info()
            engine.win.movePointer.assert_not_called()
            props = engine.win.requestProperties.call_args.args[0]
            self.assertFalse(props.getCursorHidden())
            self.assertEqual(props.getMouseMode(), WindowProperties.MAbsolute)
            self.assertEqual(camera.camera_queue.qsize(), 2)
            self.assertEqual(engine.task_manager.add.call_count, 4)
        finally:
            control.close()

    def test_close_restores_vendor_methods_and_is_idempotent(self):
        engine, camera = self.camera()
        control = RaceCameraInput(engine)
        control.close()
        control.close()
        self.assertIs(camera.set_mouse_to_center.__func__, MainCamera.set_mouse_to_center)
        self.assertIs(camera.update_mouse_info.__func__, MainCamera.update_mouse_info)
        # This spy confirms why the adapter is necessary, without moving any
        # real pointer: the unadapted method calls the mocked window directly.
        camera.set_mouse_to_center()
        engine.win.movePointer.assert_called_once()

    def test_drag_changes_chase_view_without_moving_vehicle_or_pointer(self):
        engine,camera=self.camera()
        chassis=engine.render.attachNewNode('chassis')
        chassis.setPos(1,2,.5)
        camera.current_track_agent=SimpleNamespace(chassis=chassis)
        camera.camera.setPos(1,-5.5,3.5)
        base=camera.camera.getPos()
        vehicle_pose=chassis.getTransform()
        control=RaceCameraInput(engine)
        try:
            control.drag(120,30)
            task=SimpleNamespace(cont='continue')
            control._apply_task(task)
            changed=camera.camera.getPos()
            self.assertGreater((changed-base).length(),1.)
            self.assertEqual(chassis.getTransform(),vehicle_pose)
            # MetaDrive supplies the base chase pose every frame. No new
            # drag means the offset remains fixed, rather than accumulating.
            camera.camera.setPos(base)
            control._apply_task(task)
            self.assertLess((camera.camera.getPos()-changed).length(),1e-5)
            engine.win.movePointer.assert_not_called()
        finally:
            control.close()

    def test_bird_view_drag_pans_and_switching_view_resets_offset(self):
        engine,camera=self.camera()
        camera.is_bird_view_camera=lambda:True
        camera.camera_x,camera.camera_y,camera.top_down_camera_height=1.,2.,40.
        camera.camera.setPos(1,2,40)
        camera.camera.lookAt(1,2,0)
        control=RaceCameraInput(engine)
        try:
            control.drag(80,20)
            control._apply_task(SimpleNamespace(cont='continue'))
            self.assertNotEqual(tuple(camera.camera.getPos())[:2],(1,2))
            camera.is_bird_view_camera=lambda:False
            control._sync_view()
            self.assertEqual(control.pan,(0.,0.))
            self.assertEqual((control.yaw,control.pitch),(0.,0.))
            engine.win.movePointer.assert_not_called()
        finally:
            control.close()


if __name__ == '__main__':
    unittest.main()
