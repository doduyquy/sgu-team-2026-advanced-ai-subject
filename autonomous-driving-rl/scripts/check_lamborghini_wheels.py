"""Finite rendered wheel-fit / Bullet rotation diagnostic."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--video',action='store_true')
    args=parser.parse_args()
    from PySide6.QtCore import QProcess  # noqa:F401
    from src.training.runtime import local_training_runtime
    from src.visualization.race_mode import RaceApp,configure_race_graphics
    configure_race_graphics('lite')
    with local_training_runtime(ROOT):
        app=RaceApp(ROOT,args.output)
        report={}
        try:
            app.open()
            vehicle=app.env.agent
            report['vehicle']={name:getattr(vehicle,name) for name in
                               ('TIRE_RADIUS','TIRE_WIDTH','CHASSIS_TO_WHEEL_AXIS','LATERAL_TIRE_TO_CENTER')}
            app.ui.show('running')
            app.ui.groups['running'].hide()
            camera=app.env.engine.main_camera
            camera.stop_track()
            camera.run_task=False
            def record(label):
                wheels=[]
                for i,node in enumerate(c for c in vehicle.origin.getChildren() if c.getName()=='wheel'):
                    visual=node.find('race_wheel_visual')
                    lo,hi=visual.getTightBounds(vehicle.origin)
                    wheels.append({'index':i,'node_pos':list(node.getPos()),'node_hpr':list(node.getHpr()),
                                   'visual_quat':list(visual.getQuat(vehicle.origin)),
                                   'visual_center':list((lo+hi)*.5),'size':list(hi-lo),
                                   'rotation':vehicle.system.getWheel(i).getRotation()})
                report[label]={'origin_z':vehicle.origin.getZ(),'wheels':wheels}
                for angle,pos in [('side',(8,0,1.0)),('front',(5,8,2.2)),('rear',(-5,-8,2.2))]:
                    app.env.engine.camera.reparentTo(vehicle.origin)
                    app.env.engine.camera.setPos(*pos)
                    app.env.engine.camera.lookAt(0,0,.15)
                    app.screenshot(f'{label}-{angle}.png')
            for _ in range(15):app.env.step({'racer_0':[0.,0.]})
            record('stationary')
            for _ in range(20):app.env.step({'racer_0':[.12,.7]})
            record('moving')
            report['moved_speed_kmh']=float(vehicle.speed_km_h)
            report['all_wheels_rolled']=all(abs(b['rotation']-a['rotation'])>1 for a,b in
                zip(report['stationary']['wheels'],report['moving']['wheels']))
            assert report['all_wheels_rolled']
            assert all(abs(w['node_hpr'][0])>1 for w in report['moving']['wheels'][:2]),'Front wheels did not steer'
            if args.video:
                import subprocess
                import imageio_ffmpeg
                frames=app.output/'frames';frames.mkdir()
                app.env.engine.camera.setPos(6.8,4.2,1.5)
                app.env.engine.camera.lookAt(0,0,-.05)
                app.env.config['decision_repeat']=2
                for index in range(80):
                    app.env.step({'racer_0':[0.,.18]})
                    app.screenshot(f'frames/frame-{index:03d}.png')
                command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y',
                         '-framerate','25','-i',str(frames/'frame-%03d.png'),'-vf','scale=960:600',
                         '-c:v','h264_nvenc','-preset','p1','-pix_fmt','yuv420p','-movflags','+faststart',
                         str(app.output/'lamborghini-wheels.mp4')]
                subprocess.run(command,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            camera.camera.reparentTo(app.env.engine.render)
            app.ui.show('garage')
            app.ui.update_selection(app.selection,app.roads)
            app.screenshot('garage.png')
        finally:
            app._destroy_hud();app._close_agents();app._close_phone()
            if app.events:app.events.ignoreAll()
            if app.garage_pointer:app.garage_pointer.close()
            if app.camera_input:app.camera_input.close()
            if app.ui:app.ui.destroy()
            if app.graphics_runtime:app.graphics_runtime.close()
            if app.env:app.env.close()
            app.save_manifest();app.decisions.close();app.results.close()
            (app.output/'wheel-check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
