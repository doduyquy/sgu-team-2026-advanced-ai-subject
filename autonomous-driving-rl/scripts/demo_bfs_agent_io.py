"""Run the real BFS agent with native input/decision inspection on a TRAIN case.

Only the terrain texture resolution is reduced, process-locally, to avoid the
default ~2 GB semantic image. The planner, geometry, physics and actions use
the production executor. This is an exploratory SANDBOX demo, not a benchmark.
"""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pause-at-step", type=int, default=0,
                        help="Pause after this completed step; P resumes")
    parser.add_argument("--capture-dir", type=Path,
                        help="Save screenshots of the 8 inspector pages at step 10")
    parser.add_argument("--smoke-steps", type=int, default=0,
                        help="Interrupt gracefully after N steps (verification only)")
    args = parser.parse_args()

    # Qt must precede MetaDrive on this Windows environment (documented setup issue).
    from PySide6.QtCore import QProcess  # noqa: F401
    from PySide6.QtWidgets import QApplication  # noqa: F401
    from metadrive.constants import TerrainProperty
    from panda3d.core import Filename
    from src.launcher.executor import ExperimentExecutor
    from src.launcher.models import LaunchRequestV1, LauncherMode
    from src.launcher.resolver import resolve_experiment_plan
    from src.platform.experiment_logging import WandbMode
    from src.training.runtime import local_training_runtime
    from src.visualization.hud_manager import RacingHudManager

    TerrainProperty.get_semantic_map_pixel_per_meter = classmethod(lambda cls: 4)
    print("BFS demo: terrain texture 4 pixels/m (default 22); TAB details, I panel, P pause.", flush=True)
    original_update = RacingHudManager.update

    def update_with_demo_controls(manager, *positional, **kwargs):
        snapshot = original_update(manager, *positional, **kwargs)
        step = snapshot.step_index
        if step == 10 and args.capture_dir is not None:
            directory = args.capture_dir.resolve()
            directory.mkdir(parents=True, exist_ok=True)
            overlay = manager.agent_io
            for page in range(len(overlay.PAGE_NAMES)):
                overlay.page = page
                overlay._refresh()
                manager.env.engine.graphicsEngine.renderFrame()
                manager.env.engine.graphicsEngine.renderFrame()
                target = directory / f"agent_io_page_{page}.png"
                if not manager.env.engine.win.saveScreenshot(Filename.fromOsSpecific(str(target))):
                    raise RuntimeError(f"Could not save screenshot: {target}")
            (directory / "inspected_decision.json").write_text(
                json.dumps(overlay.snapshot, indent=2), encoding="utf-8")
            overlay.page = 0
            overlay._refresh()
            print(f"Saved 8 actual BFS inspector screenshots: {directory}", flush=True)
        if args.pause_at_step and step == args.pause_at_step:
            manager.env.stop()
            print(f"Paused at completed step {step}; press P in MetaDrive to resume.", flush=True)
        if args.smoke_steps and step >= args.smoke_steps:
            raise KeyboardInterrupt
        return snapshot

    RacingHudManager.update = update_with_demo_controls
    try:
        with local_training_runtime(PROJECT_ROOT):
            request = LaunchRequestV1(
                mode=LauncherMode.SANDBOX, agent_id="stage2_bfs", tier="Easy",
                sequence="SCS", geometry_generation_seed=0, environment_seed=0,
                render_mode="NATIVE", wandb_mode=WandbMode.DISABLED,
            )
            plan = resolve_experiment_plan(request, project_root=PROJECT_ROOT)
            report = ExperimentExecutor(plan, project_root=PROJECT_ROOT).execute()
            print(json.dumps(report.to_dict(), indent=2), flush=True)
            return 0 if report.status in ("COMPLETE", "INTERRUPTED") else 1
    finally:
        RacingHudManager.update = original_update


if __name__ == "__main__":
    raise SystemExit(main())
