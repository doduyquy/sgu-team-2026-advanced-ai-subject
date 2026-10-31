"""
Standalone Sandbox Demo for Live Minimap in MetaDrive NATIVE Mode (Gate HUD-1).

Demonstrates hardware-accelerated top-down orthographic minimap rendering:
- Confined strictly to exploratory TRAIN split geometries (SANDBOX mode).
- Does NOT execute on canonical TEST or VALIDATION holdout suites.
- Does NOT modify or patch production src/launcher/executor.py.
- Creates MetaDriveEnv with exact locked physics (10 Hz control frequency).
- Updates minimap step-by-step: minimap.update(world_x, world_y, heading_rad).
- Demonstrates vehicle heading orientation, static road geometry, and trajectory trail.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Optional

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metadrive.envs.metadrive_env import MetaDriveEnv

from src.launcher.cases import load_locked_geometry_block_sequence, resolve_sandbox_case
from src.launcher.executor import build_metadrive_case_config
from src.launcher.models import LaunchRequestV1, LauncherMode
from src.launcher.registry import build_canonical_agent_registry
from src.platform import get_action_adapter
from src.visualization.minimap_overlay import MiniMapOverlay


def run_minimap_demo(
    steps: int = 80,
    render_mode: str = "NATIVE",
    tier: str = "Easy",
    capture_screenshot_path: Optional[str] = None,
) -> bool:
    """
    Executes a standalone sandbox demo of the HUD-1 live minimap using scripted constant drive.
    Returns True if completed successfully, False otherwise.
    """
    print("=" * 65)
    print("XETUHANH — HUD-1 LIVE MINIMAP STANDALONE DEMO (SANDBOX)")
    print("=" * 65)
    print(f"  Launcher Mode:       SANDBOX (Exploratory TRAIN only)")
    print(f"  Render Mode:         {render_mode}")
    print(f"  Difficulty Tier:     {tier}")
    print(f"  Driving Policy:      Scripted Constant Drive (steering=0.0, throttle=0.4)")
    print(f"  Action Adapter:      continuous_box2_v1")
    print(f"  Max Steps:           {steps}")

    # 1. Resolve sandbox target case (strictly TRAIN split)
    # Uses fixture_constant_continuous solely for deterministic plan resolution
    req = LaunchRequestV1(
        mode=LauncherMode.SANDBOX,
        agent_id="fixture_constant_continuous",
        tier=tier,
        render_mode=render_mode,
    )
    case = resolve_sandbox_case(req, project_root=PROJECT_ROOT)
    split_val = case.split.value if hasattr(case.split, "value") else str(case.split)
    print(f"  Resolved Case ID:    {case.case_id} (Split={split_val})")

    # 2. Reconstruct locked road blocks
    blocks, _, _ = load_locked_geometry_block_sequence(case.to_dict(), PROJECT_ROOT)

    # 3. Build locked MetaDrive environment config
    env_config = build_metadrive_case_config(case, blocks, render_mode=render_mode)

    # 4. Action adapter for canonical physical actuation
    adapter = get_action_adapter("continuous_box2_v1")
    canonical_act = adapter.to_canonical([0.0, 0.4])

    env = None
    minimap = None

    try:
        print("\n--- Initializing MetaDrive Environment ---")
        env = MetaDriveEnv(env_config)
        raw_obs, info = env.reset(seed=case.environment_seed)

        # 5. Initialize MiniMapOverlay (NATIVE mode only)
        if render_mode.upper() == "NATIVE":
            print("--- Initializing MiniMapOverlay in Panda3D Window ---")
            minimap = MiniMapOverlay(env)
            print(f"  Minimap Viewport:    {minimap.REGION}")
            print(f"  Calculated Bounds:   {minimap.bounds}")
            print(f"  Film Dimensions:     {minimap.film_width:.1f}m x {minimap.film_height:.1f}m")

        print(f"\n--- Stepping Simulation ({steps} steps @ 10 Hz) ---")
        step_idx = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and step_idx < steps:
            # Simulation physics step with constant forward actuation
            raw_obs, r_env, terminated, truncated, step_info = env.step(canonical_act.to_numpy())
            step_idx += 1

            # Extract telemetry pose
            pos = getattr(env.agent, "position", (0.0, 0.0))
            heading = getattr(env.agent, "heading_theta", 0.0)
            speed = getattr(env.agent, "speed_km_h", 0.0)
            route_comp = step_info.get("route_completion", 0.0)

            # Update minimap overlay step-by-step
            if minimap is not None:
                minimap.update(world_x=float(pos[0]), world_y=float(pos[1]), heading_rad=float(heading))

            if step_idx % 20 == 0 or step_idx == 1:
                trail_pts = len(minimap._trail_buffer) if minimap is not None else 0
                print(
                    f"  Step {step_idx:3d}/{steps} | "
                    f"Pos=({pos[0]:6.1f}, {pos[1]:6.1f}) | "
                    f"Heading={math.degrees(heading):+6.1f}° | "
                    f"Speed={speed:5.1f} km/h | "
                    f"Route={route_comp:5.1%} | "
                    f"TrailPts={trail_pts:3d}"
                )

            # Brief real-time pacing for visual inspection if desktop display is active
            if render_mode.upper() == "NATIVE":
                time.sleep(0.02)

        # Optional screenshot capture
        if capture_screenshot_path and minimap is not None:
            try:
                import panda3d.core as p3d
                screen_path = Path(capture_screenshot_path).resolve()
                screen_path.parent.mkdir(parents=True, exist_ok=True)
                fn = p3d.Filename.fromOsSpecific(str(screen_path))
                env.engine.win.saveScreenshot(fn)
                # Flush rendering engine frames so Panda3D commits image to disk before window close
                env.engine.graphicsEngine.renderFrame()
                env.engine.graphicsEngine.renderFrame()
                if screen_path.exists():
                    print(f"\n  [SCREENSHOT] Saved native render capture ({screen_path.stat().st_size} bytes) to: {screen_path}")
                else:
                    print(f"\n  [SCREENSHOT QUEUED] saveScreenshot called for: {screen_path}")
            except Exception as e:
                print(f"  [SCREENSHOT WARNING] Could not save screenshot: {e}")

        print(f"\n[DEMO COMPLETE] Stepped {step_idx} simulation steps successfully.")
        return True

    except Exception as e:
        print(f"\n[DEMO EXCEPTION] Error during demo execution: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 6. Safe cleanup in reverse order
        if minimap is not None:
            try:
                minimap.destroy()
                print("  [CLEANUP] MiniMapOverlay destroyed safely.")
            except Exception as e:
                print(f"  [CLEANUP WARNING] Minimap destroy exception: {e}")

        if env is not None:
            try:
                env.close()
                print("  [CLEANUP] MetaDriveEnv closed safely.")
            except Exception as e:
                print(f"  [CLEANUP WARNING] Env close exception: {e}")


def main():
    parser = argparse.ArgumentParser(description="Standalone Sandbox Demo for Live Minimap (Gate HUD-1)")
    parser.add_argument("--steps", type=int, default=80, help="Number of simulation steps (default: 80)")
    parser.add_argument("--render", type=str, default="NATIVE", choices=["NATIVE", "OFF", "native", "off"], help="Render mode")
    parser.add_argument("--tier", type=str, default="Easy", choices=["Easy", "Medium", "Hard", "Extreme"], help="Difficulty tier")
    parser.add_argument("--screenshot", type=str, default=None, help="Optional output path for screenshot capture")
    args = parser.parse_args()

    success = run_minimap_demo(
        steps=args.steps,
        render_mode=args.render.upper(),
        tier=args.tier,
        capture_screenshot_path=args.screenshot,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
