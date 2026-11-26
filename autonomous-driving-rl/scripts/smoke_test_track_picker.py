"""Finite native QA: browse TEST dataset, apply four tiers, verify exact resets."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    from PySide6.QtCore import QProcess  # noqa:F401
    from src.training.runtime import local_training_runtime
    from src.visualization.race_mode import RaceApp,configure_race_graphics
    from src.visualization.test_track_catalog import TIERS
    configure_race_graphics('lite')
    with local_training_runtime(ROOT):
        app=RaceApp(ROOT,args.output)
        report={'kind':'TEST_DATASET_PICKER_NATIVE_SMOKE','canonical_run':False,'runs':[]}
        try:
            app.open()
            app.selection.set_mode('single')
            app.realtime_enabled=False  # Small deterministic diagnostic, no worker startup.
            for tier in TIERS:
                before=app.env.episode_step
                app.handle('track_open')
                picker=app.ui.track_picker
                app.handle('track_tier',tier)
                for track in app.test_track_catalog.in_tier(tier):
                    app.handle('track_map',track.geometry_id)
                app.handle('track_scenario',4)
                assert app.env.episode_step==before,'Browsing advanced the simulation'
                app.screenshot(f'picker-{tier}.png')
                expected=picker.draft.case
                app.handle('track_apply')
                assert app.selection.test_case==expected and not picker.is_open
                assert app.state=='garage'
                app._start(countdown=False)
                assert app.case.to_dict()==expected.to_dict()
                assert app.env.config['traffic_density']==expected.traffic_density
                assert app.env.config['horizon']==expected.horizon_steps
                assert app.env.current_seed==expected.environment_seed
                app._step()
                assert not any(r.summary and r.summary['reason']=='TECHNICAL_ERROR' for r in app.racers.values())
                app.screenshot(f'race-{tier}.png')
                report['runs'].append(expected.to_dict())
                app._garage()
            app.handle('track_open');app.handle('track_original')
            assert app.selection.test_case is None
            app.screenshot('garage-original.png')
            report['passed']=True
        finally:
            if app.episode_open:app._finish('ABORTED',show_results=False)
            app._destroy_hud();app._close_agents();app._close_phone()
            if app.events:app.events.ignoreAll()
            if app.garage_pointer:app.garage_pointer.close()
            if app.camera_input:app.camera_input.close()
            if app.ui:app.ui.destroy()
            if app.graphics_runtime:app.graphics_runtime.close()
            if app.env:app.env.close()
            app.save_manifest();app.decisions.close();app.results.close()
            (app.output/'picker-smoke.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
