"""Finite same-machine phone runtime comparison; not an AI quality benchmark."""
import argparse
import json
from pathlib import Path
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def summarize(values):
    values=sorted(v for v in values if v is not None)
    return {'mean':round(sum(values)/len(values),2),'p95':round(values[min(len(values)-1,int(len(values)*.95))],2),
            'max':round(max(values),2),'n':len(values)} if values else {}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=12)
    parser.add_argument('--only',choices=('both','optimized','legacy'),default='both')
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--encoder',choices=('nvenc','qsv','cpu','auto'),default='nvenc')
    args=parser.parse_args()
    from PySide6.QtWidgets import QApplication  # noqa:F401
    from src.training.runtime import local_training_runtime
    from src.visualization.race_mode import RaceApp,configure_race_graphics
    from src.visualization.phone_camera import PhoneCamera
    from src.visualization.phone_encoder import PhoneEncoder
    configure_race_graphics()
    with local_training_runtime(ROOT):
        app=RaceApp(ROOT,args.output)
        app.map_set='train'
        app.selection.road_index=0
        report={'kind':'INTERACTIVE_RUNTIME_COMPARISON','seconds_per_phase':args.seconds,'phases':[]}
        stop=threading.Event()
        encoder=PhoneEncoder(args.encoder)
        encoding_errors=[]
        def input_loop():
            sequence=0
            while not stop.wait(.02):
                sequence+=1
                app.phone_control.receive('benchmark',{'type':'input','seq':sequence,'steering':0.,'throttle':.55})
        def encode_loop():
            previous=0
            origin=time.perf_counter()
            while not stop.wait(.002):
                sequence,rgb,_=app.phone_frames.latest()
                if rgb is not None and sequence!=previous:
                    previous=sequence
                    try:
                        encoder.encode(rgb,int((time.perf_counter()-origin)*90000))
                    except Exception as error:
                        encoding_errors.append(str(error))
                        return
        try:
            app.open()
            report['renderer']=app.env.engine.win.getGsg().getDriverRenderer()
            app.phone_camera=PhoneCamera(app.env.engine,app.phone_frames)
            app.phone_control.claim('benchmark')
            threads=[threading.Thread(target=f,daemon=True) for f in (input_loop,encode_loop)]
            for thread in threads: thread.start()
            phases=(False,True) if args.only=='both' else (args.only=='optimized',)
            for realtime in phases:
                app.realtime_enabled=realtime
                app.selection.set_mode('human')
                app._start(countdown=False)
                app.phone_camera.set_active(True)
                started=time.monotonic()
                costs,delays,active=[],[],[]
                sim_start=None
                import cProfile
                profile=cProfile.Profile() if args.profile else None
                while time.monotonic()-started<args.seconds+2:
                    app.env.update_race_camera()
                    if time.perf_counter()>=app._next_decision:
                        tick=time.perf_counter()
                        if profile: profile.enable()
                        app._step()
                        if profile: profile.disable()
                        if time.monotonic()-started>=2:
                            if sim_start is None: sim_start=(time.monotonic(),app.steps)
                            costs.append((time.perf_counter()-tick)*1000)
                            delays.append(app.phone_control.snapshot()['input_apply_ms'])
                            active.append(sum(r.index!=5 and not r.summary for r in app.racers.values()))
                    else:
                        if not realtime: app.env.engine.taskMgr.step()
                        time.sleep(.002 if realtime else .012)
                elapsed=time.monotonic()-sim_start[0]
                simulated=(app.steps-sim_start[1])*app.control_dt_s
                phase={'mode':'optimized' if realtime else 'legacy_lockstep',
                    'physics_hz':round((app.steps-sim_start[1])/elapsed,1),
                    'simulation_wall_ratio':round(simulated/elapsed,3),
                    'step_ms':summarize(costs),'input_apply_ms':summarize(delays),
                    'active_ai':{'min':min(active),'max':max(active)},
                    'camera':app.phone_frames.stats(),'encoder':encoder.stats(),
                    'workers':app.realtime.stats() if app.realtime else None,
                    'errors':encoding_errors+[r.summary['error'] for r in app.racers.values() if r.summary and r.summary['reason']=='TECHNICAL_ERROR']}
                assert not phase['errors'],phase['errors']
                report['phases'].append(phase)
                if profile: profile.dump_stats(str(app.output/'runtime.pstats'))
                app.screenshot(f"{phase['mode']}.png")
                app._garage()
            report['passed']=True
        finally:
            stop.set()
            if 'threads' in locals():
                for thread in threads: thread.join(timeout=2)
            encoder.close()
            if app.episode_open: app._finish('ABORTED',show_results=False)
            app._destroy_hud(); app._close_agents(); app._close_phone()
            if app.events: app.events.ignoreAll()
            if app.ui: app.ui.destroy()
            if app.env: app.env.close()
            app.save_manifest(); app.decisions.close(); app.results.close()
            (app.output/'performance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))


if __name__=='__main__': main()
