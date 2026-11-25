"""
Tests for 3D Cyberpunk Hologram Finish Line Visuals (Gate 7.5 Visual Extensions).

Verifies:
1. FinishFloatingHologram construction, scene graph presence, and destroy cleanup.
2. Full road width coverage spanning all lanes.
3. create_finish_visual() factory routing.
4. Strict Information Parity: 3D visual nodes do NOT modify simulator observations or LiDAR.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from panda3d.core import DepthWriteAttrib, Shader, ShaderAttrib

from metadrive.constants import CamMask
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.envs.metadrive_env import MetaDriveEnv
from src.visualization import (
    FinishFloatingHologram,
    FinishLineVisual,
    create_finish_visual,
)
from src.launcher.executor import create_optional_finish_visual


class TestFinishVisual(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = MetaDriveEnv(dict(
            use_render=False,
            num_scenarios=1,
            start_seed=11,
            map="SCS",
            traffic_density=0.0
        ))
        cls.obs, _ = cls.env.reset()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_cyber_hologram_creation_and_destroy(self) -> None:
        """Kiểm tra dựng Cổng Hologram 3D Cyberpunk và dọn dẹp sạch node."""
        holo = FinishFloatingHologram(self.env, text="TEST_FINISH")
        self.assertIsNotNone(holo.root)
        self.assertFalse(holo.root.isEmpty())
        self.assertEqual(holo.root.getParent(), self.env.engine.render)
        self.assertGreaterEqual(holo.total_road_width, 7.0)

        # Kiểm tra các thành phần: chỉ có bảng hologram lơ lửng và laser mặt đường (không có trụ hay dầm đen)
        children_names = [child.getName() for child in holo.root.getChildren()]
        self.assertIn("hologram_card", children_names)
        self.assertIn("ground_laser_card", children_names)
        self.assertNotIn("left_pylon", children_names)
        self.assertNotIn("right_pylon", children_names)
        self.assertNotIn("top_beam", children_names)

        # Kiểm tra update hoạt động (chuyển động nhịp thở mượt mà)
        initial_z = holo.hologram_card.getZ()
        for _ in range(5):
            holo.update()
        self.assertIsNotNone(holo.hologram_card)
        self.assertGreater(holo.hologram_card.getZ(), initial_z)

        # Dọn dẹp
        holo.destroy()
        self.assertIsNone(holo.root)
        self.assertIsNone(holo.hologram_card)

    def test_cards_override_inherited_world_shader(self) -> None:
        """A node can exist yet remain invisible under the inherited PBR shader."""
        render = self.env.engine.render
        previous_state = render.getState()
        shader = Shader.make(Shader.SL_GLSL,
                             vertex="#version 130\nin vec4 p3d_Vertex; void main() { gl_Position = p3d_Vertex; }",
                             fragment="#version 130\nout vec4 color; void main() { color = vec4(0.0); }")
        holo = None
        try:
            render.setShader(shader)
            holo = FinishFloatingHologram(self.env)
            for card in (holo.hologram_card, holo.ground_laser_card):
                shader_state = card.getNetState().getAttrib(ShaderAttrib)
                self.assertIsNotNone(shader_state)
                self.assertIsNone(shader_state.getShader())
                depth_write = card.getNetState().getAttrib(DepthWriteAttrib)
                self.assertEqual(depth_write.getMode(), DepthWriteAttrib.MOff)
        finally:
            if holo is not None:
                holo.destroy()
            render.setState(previous_state)

    def test_visual_is_visible_to_viewers_but_not_observation_cameras(self) -> None:
        holo = FinishFloatingHologram(self.env)
        try:
            self.assertFalse(holo.root.isHidden(CamMask.MainCam))
            self.assertFalse(holo.root.isHidden(CamMask.MiniMap))
            for camera in (CamMask.RgbCam, CamMask.DepthCam, CamMask.SemanticCam, CamMask.Shadow):
                self.assertTrue(holo.root.isHidden(camera))
        finally:
            holo.destroy()

    def test_missing_texture_is_reported_and_partial_node_is_removed(self) -> None:
        render = self.env.engine.render
        previous_children = render.getNumChildren()
        with patch("src.visualization.finish_floating_hologram.TexturePool") as pool:
            pool.loadTexture.return_value = None
            with self.assertRaisesRegex(RuntimeError, "texture hologram"):
                FinishFloatingHologram(self.env)
        self.assertEqual(render.getNumChildren(), previous_children)

    def test_launcher_logs_visual_creation_failure(self) -> None:
        with patch("src.launcher.executor.create_finish_visual", side_effect=OSError("texture failure")):
            with self.assertLogs("src.launcher.executor", level="ERROR") as logs:
                self.assertIsNone(create_optional_finish_visual(self.env))
        self.assertIn("texture failure", "\n".join(logs.output))

    def test_launcher_passes_selected_style_to_factory(self) -> None:
        with patch.dict("os.environ", {"METADRIVE_FINISH_VISUAL_STYLE": "none"}):
            with patch("src.launcher.executor.create_finish_visual", return_value=None) as factory:
                self.assertIsNone(create_optional_finish_visual(self.env))
                factory.assert_called_once_with(self.env, style="none")

    def test_factory_create_finish_visual(self) -> None:
        """Kiểm tra factory function create_finish_visual."""
        # 1. Default (hologram)
        v1 = create_finish_visual(self.env)
        self.assertIsInstance(v1, FinishFloatingHologram)
        v1.destroy()

        # 2. Explicit 'hologram'
        v2 = create_finish_visual(self.env, style="hologram")
        self.assertIsInstance(v2, FinishFloatingHologram)
        v2.destroy()

        # 3. None / Off
        v3 = create_finish_visual(self.env, style="none")
        self.assertIsNone(v3)

    def test_missing_generated_textures_regenerate_in_runtime_cache(self) -> None:
        """A fresh checkout needs no generated PNG files under src/."""
        with TemporaryDirectory() as temporary:
            cache = Path(temporary) / "finish_cache"
            with patch("src.visualization.finish_floating_hologram._TEXTURE_CACHE_DIR", cache):
                visual = create_finish_visual(self.env)
                try:
                    self.assertEqual(visual._holo_path.parent, cache)
                    self.assertTrue(visual._holo_path.is_file())
                    self.assertTrue((cache / "finish_ground_laser.png").is_file())
                    self.assertFalse(visual.hologram_card.isEmpty())
                    self.assertFalse(visual.ground_laser_card.isEmpty())
                finally:
                    visual.destroy()

    def test_finish_pose_spans_both_directions_at_same_cross_section(self) -> None:
        forward = [StraightLane((0, y), (100, y), width=3.5) for y in (-1.75, -5.25)]
        reverse = [StraightLane((100, y), (0, y), width=3.5) for y in (1.75, 5.25)]
        forward[-1].index = ("start", "end", 1)
        env = SimpleNamespace(
            agent=SimpleNamespace(navigation=SimpleNamespace(final_lane=forward[-1])),
            current_map=SimpleNamespace(road_network=SimpleNamespace(graph={
                "start": {"end": forward}, "-end": {"-start": reverse},
            })),
        )
        center, heading, width = FinishFloatingHologram.calculate_finish_pose(env)
        np.testing.assert_allclose(center, (95.0, 0.0, 0.0), atol=1e-6)
        self.assertAlmostEqual(width, 14.0)
        self.assertAlmostEqual(heading, -90.0)

    def test_information_parity_lidar_invariance(self) -> None:
        """Kiểm tra Information Parity: thêm visual KHÔNG làm thay đổi observation hoặc LiDAR."""
        base_obs = self.env.get_single_observation().observe(self.env.agent)

        holo = FinishFloatingHologram(self.env)
        obs_with_holo = self.env.get_single_observation().observe(self.env.agent)

        self.assertTrue(
            np.array_equal(base_obs, obs_with_holo),
            f"Observation bị thay đổi khi gắn Hologram! Max diff: {np.max(np.abs(base_obs - obs_with_holo))}"
        )
        holo.destroy()


if __name__ == "__main__":
    unittest.main()
