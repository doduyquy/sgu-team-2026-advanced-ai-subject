"""Finite native check: human + five AI, all models, three phone cameras.

This tests the actual simulator, using synthetic phone input. Separately run
test_phone_racing.py and browser QA for WebRTC and the touch interface.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from PySide6.QtWidgets import QApplication  # noqa: F401
    from src.training.runtime import local_training_runtime
    from src.visualization.race_mode import RaceApp,configure_race_graphics
    from src.visualization.phone_camera import PhoneCamera
    from src.visualization.race_vehicle_models import CAR_MODELS
    import numpy as np
    configure_race_graphics()
    with local_training_runtime(ROOT):
        app=RaceApp(ROOT,args.output)
        app.realtime_enabled=False  # This diagnostic retains its historical 0.1 s tick assertions.
        report={'kind':'SYNTHETIC_PHONE_NATIVE_SMOKE','runs':[]}
        try:
            app.open()
            app.phone_camera=PhoneCamera(app.env.engine,app.phone_frames)
            app.phone_control.claim('native-smoke')
            seq=0
            for model in CAR_MODELS:
                app.selection.set_mode('human' if model=='lamborghini' else 'practice')
                app.selection.player_model=model
                app.ui.update_selection(app.selection,app.roads)
                app.screenshot(f'garage-{model}.png')
                app._start(countdown=False)
                original=np.array(app.env.agents['racer_5'].position)
                for i in range(16):
                    seq+=1
                    app.phone_control.receive('native-smoke',{'type':'input','seq':seq,'steering':0.,'throttle':1.})
                    app._step()
                    app.phone_camera.set_active(True)
                    app.env.engine.taskMgr.step()
                    if i in (3,7,11):
                        view=app.phone_camera.VIEWS[(i-3)//4]
                        app.phone_camera.track(app.env.agents['racer_5'],view)
                        app.env.engine.taskMgr.step()
                        app.env.engine.graphicsEngine.renderFrame()
                        time.sleep(.04)
                        app.env.engine.taskMgr.step()
                        rgb=app.phone_frames.get()
                        assert rgb is not None and rgb.std()>5,'Empty phone image'
                        from PIL import Image
                        Image.fromarray(rgb).save(app.output/f'phone-{model}-{view}.png')
                moved=float(np.linalg.norm(np.array(app.env.agents['racer_5'].position)-original))
                assert moved>.5,f'Player failed to move: {moved}'
                assert app.racers['racer_5'].steps==16
                app.handle('pause')
                positions={k:tuple(v.position) for k,v in app.env.agents.items()}
                for _ in range(3): app.env.engine.taskMgr.step()
                assert positions=={k:tuple(v.position) for k,v in app.env.agents.items()},'Moved while paused'
                if app.selection.mode=='practice':
                    app.handle('recover')
                    assert app.recoveries==1,'Practice recovery failed'
                else:
                    assert not app.env.recover('racer_5'),'Competitive recovery must be blocked'
                    app.handle('recover')
                    assert app.recoveries==0,'Competitive command bypassed recovery guard'
                report['runs'].append({'model':model,'mode':app.selection.mode,'cars':len(app.racers),
                                       'moved_m':moved,'frames':app.phone_frames.produced})
                app.screenshot(f'race-{model}.png')
                app._garage()
            report['passed']=True
            app.manifest['mobile_native_smoke_passed']=True
        finally:
            if app.episode_open: app._finish('ABORTED',show_results=False)
            app._destroy_hud();app._close_agents();app._close_phone()
            if app.events: app.events.ignoreAll()
            if app.garage_pointer: app.garage_pointer.close()
            if app.camera_input: app.camera_input.close()
            if app.ui: app.ui.destroy()
            if app.graphics_runtime: app.graphics_runtime.close()
            if app.env: app.env.close()
            app.save_manifest();app.decisions.close();app.results.close()
            (app.output/'mobile-native-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
