"""Orbit is bounded and input remains scoped to the garage preview card."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import unittest
from src.visualization.garage_orbit import GarageOrbit, GaragePointer, design_coordinates


class GarageOrbitTests(unittest.TestCase):
    def test_orbit_limits_and_letterboxed_pointer(self):
        orbit=GarageOrbit()
        orbit.drag(900,900)
        self.assertLess(orbit.yaw,360)
        self.assertEqual(orbit.elevation,42)
        orbit.drag(-999,-999)
        self.assertEqual(orbit.elevation,8)
        orbit.zoom(999)
        self.assertEqual(orbit.distance,10)
        orbit.zoom(-999)
        self.assertEqual(orbit.distance,23)
        self.assertEqual(design_coordinates((0,0),(1920,1080)),(640,400))

    def pointer(self):
        pointer=GaragePointer.__new__(GaragePointer)
        preview=MagicMock()
        preview.orbit=GarageOrbit()
        pointer.app=SimpleNamespace(state='garage',ui=SimpleNamespace(preview=preview))
        pointer.camera=MagicMock()
        pointer.last=pointer.click=None
        pointer.last_click=-1
        pointer.position=MagicMock(return_value=(950,350))
        return pointer

    def test_drag_only_starts_in_preview_and_releases(self):
        pointer=self.pointer()
        pointer.position.return_value=(100,350)
        pointer.press()
        self.assertIsNone(pointer.last)
        pointer.position.return_value=(950,350)
        pointer.press()
        pointer.position.return_value=(990,370)
        pointer.tick(SimpleNamespace(cont='continue'))
        self.assertEqual(pointer.app.ui.preview.orbit.yaw,30)
        pointer.release()
        before=pointer.app.ui.preview.orbit.yaw
        pointer.position.return_value=(1050,400)
        pointer.tick(SimpleNamespace(cont='continue'))
        self.assertEqual(pointer.app.ui.preview.orbit.yaw,before)
        pointer.camera._move_to_pointer.assert_not_called()

    def test_zoom_scoped_to_preview_and_ui_clicks_do_not_pan_race_camera(self):
        pointer=self.pointer()
        pointer.wheel(1)
        self.assertLess(pointer.app.ui.preview.orbit.distance,15.8)
        pointer.position.return_value=(1200,600)
        before=pointer.app.ui.preview.orbit.distance
        pointer.wheel(1)
        self.assertEqual(pointer.app.ui.preview.orbit.distance,before)
        pointer.app.state='running'
        pointer.wheel(1)
        pointer.camera._wheel_up_height.assert_called_once()
        pointer.press()
        pointer.camera._move_to_pointer.assert_not_called()

    def test_double_click_resets_and_leaving_garage_cancels_drag(self):
        pointer=self.pointer()
        with patch('src.visualization.garage_orbit.time.perf_counter',side_effect=[10.,10.15]):
            pointer.press(); pointer.press()
        pointer.app.ui.preview.reset_view.assert_called_once()
        pointer.last=(900,300)
        pointer.app.state='running'
        pointer.tick(SimpleNamespace(cont='continue'))
        self.assertIsNone(pointer.last)

    def test_race_view_only_rotates_during_left_drag_and_stops_on_release(self):
        pointer=self.pointer()
        pointer.app.state='running'
        pointer.app.camera_input=MagicMock()
        pointer.position.return_value=(640,350)
        task=SimpleNamespace(cont='continue')
        pointer.tick(task)
        pointer.app.camera_input.drag.assert_not_called()
        pointer.press()
        pointer.position.return_value=(700,380)
        pointer.tick(task)
        pointer.app.camera_input.drag.assert_called_once_with(60,30)
        pointer.release()
        pointer.position.return_value=(750,400)
        pointer.tick(task)
        pointer.app.camera_input.drag.assert_called_once()

    def test_race_buttons_and_lost_button_up_do_not_start_or_continue_drag(self):
        pointer=self.pointer()
        pointer.app.state='running'
        pointer.app.camera_input=MagicMock()
        task=SimpleNamespace(cont='continue')
        for position in ((900,742),(1050,40),(100,100)):
            pointer.position.return_value=position
            pointer.press()
            pointer.position.return_value=(640,350)
            pointer.tick(task)
        pointer.app.camera_input.drag.assert_not_called()
        pointer.press()
        pointer.button_held=lambda:False
        pointer.position.return_value=(730,390)
        pointer.tick(task)
        self.assertIsNone(pointer.race_last)
        pointer.app.camera_input.drag.assert_not_called()

    def test_ui_region_and_leaving_scene_cancel_drag(self):
        pointer=self.pointer()
        pointer.app.state='running'
        pointer.app.camera_input=MagicMock()
        pointer.position.return_value=(640,350)
        pointer.over_ui=lambda:True
        pointer.press()
        self.assertIsNone(pointer.race_last)
        pointer.over_ui=lambda:False
        pointer.press()
        pointer.position.return_value=None
        pointer.tick(SimpleNamespace(cont='continue'))
        self.assertIsNone(pointer.race_last)


if __name__=='__main__': unittest.main()
