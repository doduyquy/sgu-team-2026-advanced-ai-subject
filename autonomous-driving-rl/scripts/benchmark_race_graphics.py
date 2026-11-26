"""Finite render-only comparison: same five parked cars and road.

This isolates graphics; it does not measure driving quality or AI throughput.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def geometry_stats(root):
    nodes = root.findAllMatches('**/+GeomNode')
    return {
        'geoms': sum(n.node().getNumGeoms() for n in nodes),
        'triangles': sum(n.node().getGeom(i).getPrimitive(j).decompose().getNumPrimitives()
                         for n in nodes for i in range(n.node().getNumGeoms())
                         for j in range(n.node().getGeom(i).getNumPrimitives())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--graphics', choices=('lite', 'balanced'), default='lite')
    parser.add_argument('--seconds', type=float, default=6)
    args = parser.parse_args()
    if args.seconds < 2:
        parser.error('--seconds must be >= 2')
    from PySide6.QtCore import QProcess  # noqa: F401 - Qt before Panda DLLs
    from src.training.runtime import local_training_runtime
    from src.visualization.race_mode import RaceApp, configure_race_graphics
    from src.visualization.race_vehicle_models import build_car, CAR_MODELS
    from panda3d.core import NodePath
    configure_race_graphics(args.graphics)
    report = {'kind': 'RENDER_ONLY_SAME_SCENE', 'physics_advanced': False,
              'requested_profile': args.graphics, 'seconds': args.seconds}
    with local_training_runtime(ROOT):
        app = RaceApp(ROOT, args.output)
        app.graphics_benchmark_uncapped = True
        try:
            app.open()
            from panda3d.core import WindowProperties
            properties=WindowProperties()
            properties.setMinimized(False)
            properties.setForeground(True)
            app.env.engine.win.requestProperties(properties)
            report['renderer'] = app.env.engine.win.getGsg().getDriverRenderer()
            report['body_geometry'] = {}
            for model in CAR_MODELS:
                root = build_car(NodePath('probe'), model, (.2, .79, 1., 1.))
                report['body_geometry'][model] = geometry_stats(root)
                lod = root.find('**/+LODNode')
                if not lod.isEmpty():
                    report['body_geometry'][model]['levels'] = {
                        n.getName(): geometry_stats(n) for n in lod.getChildren()}
                root.removeNode()
                app.selection.vehicle_models[0] = model
                app.ui.update_selection(app.selection, app.roads)
                for _ in range(8):
                    app.env.engine.taskMgr.step()
                app.screenshot('garage-' + model + '.png')
                app.ui.preview.orbit.drag(180,30)
                app.ui.preview.apply_view()
                app.screenshot('garage-' + model + '-rotated.png')
                app.ui.preview.reset_view()
            app.selection.vehicle_models[0] = 'lamborghini'
            app.selection.set_mode('compare')
            app.selection.comparison_indices = tuple(range(5))
            app.selection.traffic = True
            app._start(countdown=False)
            # Benchmark every frame, independently of the interactive FPS cap.
            runtime = getattr(app, 'graphics_runtime', None)
            if runtime:
                runtime.uncapped = True
            report['cars'] = len(app.env.agents)
            report['traffic'] = len(app.env.engine.traffic_manager.traffic_vehicles)
            report['graphics'] = app.manifest.get('graphics')
            for _ in range(30):
                app.env.engine.taskMgr.step()
            samples = []
            started = time.perf_counter()
            while time.perf_counter() - started < args.seconds:
                before = time.perf_counter()
                app.env.engine.taskMgr.step()
                samples.append((time.perf_counter() - before) * 1000)
            elapsed = time.perf_counter() - started
            samples.sort()
            report['frames'] = len(samples)
            report['render_loop_fps'] = round(len(samples) / elapsed, 2)
            report['frame_ms'] = {
                'mean': round(statistics.mean(samples), 3),
                'p95': round(samples[min(len(samples)-1, int(len(samples)*.95))], 3),
            }
            report['physics_steps'] = app.steps
            report['window_size'] = [app.env.engine.win.getXSize(), app.env.engine.win.getYSize()]
            pipe = app.env.engine.pbrpipe
            report['scene_buffers'] = [[b.getXSize(), b.getYSize()] for b in pipe.manager.buffers]
            app.screenshot('five-cars-traffic.png')
            report['passed'] = app.steps == 0 and len(app.env.agents) == 5
        finally:
            if app.episode_open:
                app._finish('ABORTED', show_results=False)
            app._destroy_hud()
            app._close_agents()
            app._close_phone()
            if app.events:
                app.events.ignoreAll()
            if app.ui:
                app.ui.destroy()
            if getattr(app, 'graphics_runtime', None):
                app.graphics_runtime.close()
            if app.env:
                app.env.close()
            app.manifest['status'] = 'CLOSED'
            app.save_manifest()
            app.decisions.close()
            app.results.close()
            (app.output / 'graphics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
