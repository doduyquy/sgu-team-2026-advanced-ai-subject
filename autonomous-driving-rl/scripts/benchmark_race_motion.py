"""Measure visible motion cadence with identical MPPI/car/map, then close."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def percentiles(values):
    values=sorted(values)
    return {'mean':round(sum(values)/len(values),2),
            'p95':round(values[min(len(values)-1,int(len(values)*.95))],2),
            'max':round(values[-1],2)} if values else {}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=6)
    parser.add_argument('--graphics',choices=('lite','balanced'),default='balanced')
    parser.add_argument('--lineup',choices=('mppi','all'),default='mppi')
    parser.add_argument('--only',choices=('both','smooth','legacy'),default='both')
    args=parser.parse_args()
    from PySide6.QtCore import QProcess  # noqa:F401
    from src.training.runtime import local_training_runtime
    from src.visualization.race_mode import RaceApp,configure_race_graphics
    configure_race_graphics(args.graphics)
    with local_training_runtime(ROOT):
        app=RaceApp(ROOT,args.output)
        report={'kind':'VISIBLE_MOTION_COMPARISON','canonical_run':False,'phases':[]}
        samples=[]
        task_samples=[]
        def observe(task):
            if app.state=='running':
                sample=(time.perf_counter(),tuple(map(float,app.env.agent.position)))
                task_samples.append((sample[0],app.env.engine.win.isActive()))
                if app.env.engine.win.isActive():samples.append(sample)
            return task.cont
        try:
            app.open()
            report['renderer']=app.env.engine.win.getGsg().getDriverRenderer()
            app.selection.set_mode('single' if args.lineup=='mppi' else 'compare')
            app.selection.agent_index=4
            app.selection.vehicle_models[4]='tesla'
            app.env.engine.taskMgr.add(observe,'motion-measurement',sort=60)
            phases=(False,True) if args.only=='both' else (args.only=='smooth',)
            for smooth in phases:
                app.realtime_enabled=smooth
                app._start(countdown=False)
                report['case']=app.case.to_dict()
                samples.clear()
                task_samples.clear()
                started=time.perf_counter()
                ready_at=started if not smooth else None
                measured_from=started+2 if not smooth else float('inf')
                sim_start=None
                while time.perf_counter()<measured_from+args.seconds and app.state=='running':
                    app.env.update_race_camera()
                    if time.perf_counter()>=app._next_decision:
                        app._step()
                    elif app.realtime:
                        time.sleep(.001)
                    else:
                        app.env.engine.taskMgr.step();time.sleep(.012)
                    if smooth and ready_at is None:
                        if app.realtime and app.realtime.workers_ready():
                            ready_at=time.perf_counter();measured_from=ready_at+2
                        elif time.perf_counter()-started>15:
                            raise RuntimeError('AI worker did not become ready')
                    if sim_start is None and time.perf_counter()>=measured_from:
                        sim_start=(time.perf_counter(),app.steps)
                ended=time.perf_counter()
                draws=[s for s in samples if s[0]>=measured_from]
                moving=[b[0] for a,b in zip(draws,draws[1:])
                        if sum((x-y)**2 for x,y in zip(a[1],b[1]))>1e-8]
                draw_gaps=[(b[0]-a[0])*1000 for a,b in zip(draws,draws[1:])]
                move_gaps=[(b-a)*1000 for a,b in zip(moving,moving[1:])]
                elapsed=ended-measured_from
                phase={'mode':'smooth' if smooth else 'legacy_lockstep',
                       'seconds':round(elapsed,2),'draw_fps':round(len(draws)/elapsed,1),
                       'worker_startup_s':round(ready_at-started,2),
                       'task_samples':len(task_samples),'active_samples':len(samples),
                       'moving_frames_per_s':round(len(moving)/elapsed,1),
                       'same_pose_fraction':round(1-len(moving)/max(1,len(draws)-1),3),
                       'draw_gap_ms':percentiles(draw_gaps),'moving_gap_ms':percentiles(move_gaps),
                       'simulation_wall_ratio':round((app.steps-sim_start[1])*app.control_dt_s/(ended-sim_start[0]),3),
                       'ai_ms':percentiles(app.racers['racer_4'].latencies),
                       'active_ai':sum(not r.summary for r in app.racers.values()),
                       'realtime':app.realtime.stats() if app.realtime else None,
                       'errors':[r.summary['error'] for r in app.racers.values()
                                 if r.summary and r.summary['reason']=='TECHNICAL_ERROR']}
                report['phases'].append(phase)
                app.screenshot(phase['mode']+'.png')
                print(json.dumps(phase),flush=True)
                assert not phase['errors'],phase['errors']
                assert len(moving)>10,'Car did not move during measurement'
                app._garage()
            report['passed']=True
        finally:
            if app.episode_open: app._finish('ABORTED',show_results=False)
            app._destroy_hud();app._close_agents();app._close_phone()
            if app.events:app.events.ignoreAll()
            if app.garage_pointer:app.garage_pointer.close()
            if app.ui:app.ui.destroy()
            if app.graphics_runtime:app.graphics_runtime.close()
            if app.env:app.env.close()
            app.save_manifest();app.decisions.close();app.results.close()
            (app.output/'motion-performance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
