"""Measure VN frame pacing with real AI, independently of benchmark scoring."""
import argparse
import cProfile
import json
from pathlib import Path
import pstats
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=8)
    parser.add_argument('--graphics', choices=('lite', 'balanced'), default='balanced')
    parser.add_argument('--size', default='1280x800')
    parser.add_argument('--lineup', choices=('all', 'astar', 'practice'), default='all')
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--cpu-profile', action='store_true')
    parser.add_argument('--ablate', choices=('none', 'trees', 'shadows', 'facades'), default='none')
    args = parser.parse_args()
    if args.seconds < 2:
        parser.error('--seconds must be at least 2')
    try:
        width, height = map(int, args.size.split('x'))
        if min(width, height) < 200:
            raise ValueError()
    except ValueError:
        parser.error('--size must be WIDTHxHEIGHT with both dimensions >= 200')

    # Qt must load before Panda3D DLLs in this Windows installation.
    from PySide6.QtCore import QProcess  # noqa: F401
    from panda3d.core import WindowProperties
    from src.training.runtime import local_training_runtime
    from src.visualization.race_mode import RaceApp, configure_race_graphics
    from src.visualization.vietnam_track import point_at

    configure_race_graphics(args.graphics)
    with local_training_runtime(ROOT):
        app = RaceApp(ROOT, args.output)
        app.selection.road_index = 3
        app.selection.set_mode({'all': 'compare', 'astar': 'single', 'practice': 'practice'}[args.lineup])
        app.realtime_enabled = not args.render_only
        draws = []
        report = dict(canonical_run=False, graphics=args.graphics, lineup=args.lineup,
                      render_only=args.render_only, ablate=args.ablate, cpu_profile=args.cpu_profile)

        def observe(task):
            if app.state == 'running' and app.env.engine.win.isActive():
                draws.append((time.perf_counter(), tuple(map(float, app.env.agent.position))))
            return task.cont

        def tick():
            app.env.update_race_camera()
            app.ui.layout()
            if args.render_only:
                app.env.engine.taskMgr.step()
            elif time.perf_counter() >= app._next_decision:
                app._step()
            else:
                time.sleep(.001)

        try:
            app.open()
            properties = WindowProperties()
            properties.setSize(width, height)
            properties.setMinimized(False)
            app.env.engine.win.requestProperties(properties)
            for _ in range(4):
                app.env.engine.taskMgr.step()
            app.ui.layout()
            app._start(countdown=False)
            if args.render_only:
                app.graphics_runtime.uncapped = True
                position, heading = point_at(1200)
                for i, vehicle in enumerate(app.env.agents.values()):
                    vehicle.set_position(position + (i*8, 0), height=.6)
                    vehicle.set_heading_theta(heading)
                app.env.focus('racer_0', False)
            target = {'trees': 'vn_photographic_tree', 'shadows': 'vn_tree_shadow',
                      'facades': 'vn_detailed_facade'}.get(args.ablate, '__nothing__')
            for node in app.env.engine.render.findAllMatches('**/'+target):
                node.hide()
            app.env.engine.taskMgr.add(observe, 'vn_motion_probe', sort=65)
            deadline = time.perf_counter()+20
            while app.realtime and not app.realtime.workers_ready():
                app.realtime._poll()
                app.env.engine.taskMgr.step()
                time.sleep(.01)
                if time.perf_counter() > deadline:
                    raise RuntimeError('AI worker startup timeout')
            if app.realtime:
                app.realtime.reset_clock()
            warmup_end = time.perf_counter()+2
            while time.perf_counter() < warmup_end:
                tick()
            draws.clear()
            started, sim_start = time.perf_counter(), app.steps
            profiler = cProfile.Profile()
            if args.cpu_profile:
                profiler.enable()
            while time.perf_counter()-started < args.seconds:
                tick()
                if app.state != 'running':
                    raise RuntimeError('Race ended during sample')
            profiler.disable()
            elapsed = time.perf_counter()-started
            if args.cpu_profile:
                with (app.output/'cpu-profile.txt').open('w') as stream:
                    pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(45)
            gaps = sorted((b[0]-a[0])*1000 for a, b in zip(draws, draws[1:]))
            if not gaps:
                raise RuntimeError('Renderer produced no measurable frames')
            moving = sum(sum((x-y)**2 for x, y in zip(a[1], b[1])) > 1e-8
                         for a, b in zip(draws, draws[1:]))
            report.update(
                fps=round(len(draws)/elapsed, 2), moving_fps=round(moving/elapsed, 2),
                frame_ms_p95=round(gaps[min(len(gaps)-1, int(.95*len(gaps)))], 2),
                frame_ms_max=round(max(gaps), 2),
                simulation_wall_ratio=round((app.steps-sim_start)*app.control_dt_s/elapsed, 3),
                window=[app.env.engine.win.getXSize(), app.env.engine.win.getYSize()],
                scene_buffers=[[b.getXSize(), b.getYSize()] for b in app.env.engine.pbrpipe.manager.buffers],
                renderer=app.env.engine.win.getGsg().getDriverRenderer(),
                realtime=app.realtime.stats() if app.realtime else None,
                errors=[r.summary for r in app.racers.values()
                        if r.summary and r.summary['reason'] == 'TECHNICAL_ERROR'])
            app.screenshot('scene.png')
            # A HUD can continue drawing while an incorrectly resized HDR
            # target is black. Count FPS only after verifying the world image.
            from PIL import Image
            import numpy as np
            with Image.open(app.output/'scene.png') as screenshot:
                w,h=screenshot.size
                crop=np.asarray(screenshot.convert('RGB').crop((int(w*.32),int(h*.32),int(w*.65),int(h*.60))))
            report['scene_visible']=bool(np.mean(np.max(crop,axis=2)>8)>.8)
            if not report['scene_visible']:
                raise RuntimeError('3D scene is black; this FPS sample is invalid')
            report['passed'] = not report['errors']
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
            if app.graphics_runtime:
                app.graphics_runtime.close()
            if app.env:
                app.env.close()
            app.decisions.close()
            app.results.close()
            (app.output/'performance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
