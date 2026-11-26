"""Rendering optimizations must preserve physics, shared textures and livery."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PySide6.QtCore import QProcess  # noqa: F401 - Windows DLL order
from panda3d.bullet import BulletRigidBodyNode
from panda3d.core import (CardMaker, GeomVertexReader, NodePath, PNMImage, Texture)

from src.visualization.race_graphics import (
    RaceGraphicsRuntime, configure_graphics, limit_visual_textures,scene_buffer_ratio,PROFILES,
)
from src.visualization.race_style import RACE_COLORS
from src.visualization.race_vehicle_models import CAR_MODELS, build_car, install_car
from src.visualization.race_imported_assets import IMPORTED_MODELS


def geoms(root):
    return sum(n.node().getNumGeoms() for n in root.findAllMatches('**/+GeomNode'))


class VehicleGraphicsTests(unittest.TestCase):
    def test_cybertruck_replaces_tesla_and_keeps_four_independent_physical_wheels(self):
        from panda3d.core import Vec3
        from src.visualization.race_vehicle_models import CAR_NAMES
        self.assertEqual(CAR_NAMES['tesla'],'Tesla Cybertruck')
        self.assertFalse(IMPORTED_MODELS['tesla']['full_wheels'])
        origin=NodePath('truck')
        chassis=origin.attachNewNode(BulletRigidBodyNode('chassis'));chassis.node().setMass(1300)
        wheels=[]
        for x,y in ((.815,1.05234),(-.815,1.05234),(.815,-1.4166),(-.815,-1.4166)):
            wheel=origin.attachNewNode('wheel');wheel.setPos(x,y,-.22);wheels.append(wheel)
        before=[wheel.getMat() for wheel in wheels]
        vehicle=SimpleNamespace(origin=origin,TIRE_RADIUS=.313,LATERAL_TIRE_TO_CENTER=.815)
        body=install_car(vehicle,'tesla',0)
        self.assertEqual(before,[wheel.getMat() for wheel in wheels])
        self.assertEqual(chassis.node().getMass(),1300)
        self.assertAlmostEqual(body.getZ(),-.333,places=5)
        for wheel in wheels:
            visual=wheel.find('race_wheel_visual')
            self.assertFalse(visual.isEmpty())
            lo,hi=visual.getTightBounds(wheel)
            self.assertAlmostEqual((hi.z-lo.z)/2,.313,places=4)
            hub=visual.getPos(origin)
            point=origin.getRelativePoint(visual,Vec3(0,.1,.1))
            wheel.setP(70)
            self.assertLess((visual.getPos(origin)-hub).length(),1e-5)
            self.assertGreater((origin.getRelativePoint(visual,Vec3(0,.1,.1))-point).length(),.1)

    def test_lamborghini_hubs_fit_arches_and_roll_about_their_centres(self):
        from panda3d.core import Vec3
        origin=NodePath('vehicle')
        chassis=origin.attachNewNode(BulletRigidBodyNode('chassis'))
        chassis.node().setMass(1100)
        wheels=[]
        for x,y in ((.815,1.05234),(-.815,1.05234),(.815,-1.4166),(-.815,-1.4166)):
            wheel=origin.attachNewNode('wheel');wheel.setPos(x,y,-.22);wheels.append(wheel)
        vehicle=SimpleNamespace(origin=origin,TIRE_RADIUS=.313,LATERAL_TIRE_TO_CENTER=.815)
        before=[w.getMat() for w in wheels]
        body=install_car(vehicle,'lamborghini',0)
        self.assertEqual(before,[w.getMat() for w in wheels])
        self.assertEqual(chassis.node().getMass(),1100)
        for wheel in wheels:
            visual=wheel.find('race_wheel_visual')
            source=visual.getPythonTag('source_center')
            lo,hi=visual.getTightBounds(wheel)
            self.assertAlmostEqual((hi.z-lo.z)/2,.313,places=4)
            self.assertAlmostEqual(visual.getY(),0,places=6)
            self.assertAlmostEqual(visual.getZ(),0,places=6)
            self.assertLess(abs(source[2]+body.getZ()-wheel.getZ()),.04)
            hub_before=visual.getPos(origin)
            point_before=origin.getRelativePoint(visual,Vec3(0,.1,.1))
            wheel.setP(80)  # Bullet rolls its wheel node about the X axle.
            self.assertLess((visual.getPos(origin)-hub_before).length(),1e-5)
            self.assertGreater((origin.getRelativePoint(visual,Vec3(0,.1,.1))-point_before).length(),.1)
        # Returning to a different model must not retain the lowered Lambo body.
        self.assertAlmostEqual(install_car(vehicle,'vf3',0).getZ(),0)

    def test_install_and_replace_preserves_bullet_and_animated_wheel_nodes(self):
        origin = NodePath('vehicle')
        chassis = origin.attachNewNode(BulletRigidBodyNode('chassis'))
        chassis.node().setMass(1300)
        wheels = [origin.attachNewNode('wheel') for _ in range(4)]
        for i, wheel in enumerate(wheels):
            wheel.setPos(i, 2, 3)
        old_positions = [w.getPos() for w in wheels]
        vehicle = SimpleNamespace(origin=origin)
        for model in CAR_MODELS:
            install_car(vehicle, model, 0)
            self.assertEqual(chassis.node().getMass(), 1300)
            self.assertFalse(chassis.isHidden())
            self.assertEqual([w.getPos() for w in wheels], old_positions)
            self.assertTrue(all(not w.isHidden() for w in wheels))
            self.assertEqual(len(origin.findAllMatches('race_custom_body')), 1)

    def test_lod_reduces_batches_and_far_detail_without_gaps(self):
        for model in CAR_MODELS:
            root = build_car(NodePath('world'), model, RACE_COLORS[0])
            lod = root.find('**/+LODNode')
            near, far = lod.getChildren()
            self.assertLessEqual(geoms(near), 24 if model in IMPORTED_MODELS else 4)
            self.assertLessEqual(geoms(far), 4)
            self.assertEqual(lod.node().getOut(0), 0)
            self.assertEqual(lod.node().getIn(0), lod.node().getOut(1))
            self.assertGreater(lod.node().getIn(1), 10000)
            self.assertEqual(root.getPythonTag('asset_origin'),
                             'cc_by_downloaded' if model in IMPORTED_MODELS else 'procedural_original')

    def test_cached_liveries_are_independent(self):
        world = NodePath('world')
        blue = build_car(world, 'vf3', RACE_COLORS[0])
        orange = build_car(world, 'vf3', RACE_COLORS[1])
        same = build_car(world, 'vf3', RACE_COLORS[0])
        def colors(body):
            found = set()
            for path in body.findAllMatches('**/+GeomNode'):
                for i in range(path.node().getNumGeoms()):
                    reader = GeomVertexReader(path.node().getGeom(i).getVertexData(), 'color')
                    while not reader.isAtEnd():
                        found.add(tuple(round(v, 2) for v in reader.getData4()))
            return found
        self.assertEqual(colors(blue), colors(same))
        self.assertNotEqual(colors(blue), colors(orange))
        blue.removeNode()
        self.assertFalse(same.isEmpty())
        self.assertFalse(orange.isEmpty())

    def test_vehicle_texture_copy_preserves_numeric_source_and_srgb(self):
        configure_graphics('lite')
        image = PNMImage(1024, 512, 4)
        image.fill(.25, .5, .75)
        image.alphaFill(1)
        source = Texture('shared-image')
        source.load(image)
        source.setFormat(Texture.FSrgbAlpha)
        artwork = NodePath(CardMaker('artwork').generate())
        numeric_map = NodePath(CardMaker('numeric-map').generate())
        artwork.setTexture(source)
        numeric_map.setTexture(source)
        limit_visual_textures(artwork)
        optimized = artwork.findTexture('*')
        self.assertEqual((optimized.getXSize(), optimized.getYSize()), (512, 256))
        self.assertEqual(optimized.getFormat(), source.getFormat())
        self.assertEqual(optimized.getCompression(), Texture.CMOn)
        self.assertEqual(numeric_map.findTexture('*'), source)
        self.assertEqual((source.getXSize(), source.getYSize()), (1024, 512))
        limit_visual_textures(artwork)
        self.assertEqual(artwork.findTexture('*'), optimized)


class GraphicsRuntimeTests(unittest.TestCase):
    def test_maximizing_caps_3d_pixels_and_balanced_keeps_native_size(self):
        from direct.filter.FilterManager import FilterManager
        lite=PROFILES['lite']
        self.assertEqual(scene_buffer_ratio(lite,1280,800),(3,4))
        for width,height in ((1280,800),(1920,1080),(2560,1440),(3840,2160)):
            mul,div=scene_buffer_ratio(lite,width,height)
            self.assertLess(width*height*(mul/div)**2,lite.scene_pixel_budget*1.005)
            manager=FilterManager.__new__(FilterManager)
            manager.forcex=manager.forcey=0
            manager.win=SimpleNamespace(getXSize=lambda:width,getYSize=lambda:height)
            actual_width,actual_height=manager.getScaledSize(mul,div,1)
            self.assertGreater(actual_height,400)
            self.assertGreater(actual_width,700)
            self.assertLessEqual(actual_width*actual_height,lite.scene_pixel_budget)
            self.assertEqual(scene_buffer_ratio(PROFILES['balanced'],width,height),(1,1))

    def test_scaling_trees_does_not_scale_the_detail_distance(self):
        from src.visualization import vietnam_assets as assets
        from panda3d.core import Shader
        shader=MagicMock(spec=Shader)
        with patch.object(assets,'shade'),patch.object(assets,'_model',return_value=NodePath('mesh')),patch.object(assets,'_leaf_cloud',return_value=NodePath('leaves')):
            trees=[assets.tree(NodePath('scene'),shader,(0,0,0),scale) for scale in (1.,2.3)]
        a,b=[tree.find('**/+LODNode') for tree in trees]
        self.assertEqual(a.getNetTransform().getScale(),b.getNetTransform().getScale())
        self.assertAlmostEqual(b.getChild(0).getScale().x,2.3,places=5)
        self.assertEqual(a.node().getIn(0),b.node().getIn(0))
        self.assertEqual(a.node().getIn(0),a.node().getOut(1))

    def test_draw_gate_does_not_skip_simulation_tasks_and_defers_to_phone(self):
        engine = MagicMock()
        engine.pbrpipe.manager.buffers = [MagicMock()]
        engine.pbrpipe.manager.sizes = [(1, 1, 1)]
        engine.win.getXSize.return_value=1280
        engine.win.getYSize.return_value=800
        app = SimpleNamespace(env=SimpleNamespace(engine=engine), state='garage',
                              ui=MagicMock(), realtime=None)
        runtime = RaceGraphicsRuntime(app)
        task = SimpleNamespace(cont='continue')
        with patch('src.visualization.race_graphics.time.perf_counter', side_effect=[1., 1.001]):
            self.assertEqual(runtime._tick(task), 'continue')
            self.assertEqual(runtime._tick(task), 'continue')
        engine.win.setActive.assert_called_with(False)
        # Existing physics/controller entry points are not touched by the gate.
        engine.taskMgr.step.assert_not_called()
        engine.step.assert_not_called()
        app.state, app.realtime = 'running', object()
        engine.win.setActive.reset_mock()
        runtime._tick(task)
        engine.win.setActive.assert_not_called()
        runtime.close()
        engine.win.setActive.assert_called_with(True)


if __name__ == '__main__':
    unittest.main()
