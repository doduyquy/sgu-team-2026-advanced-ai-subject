"""
Standalone Sandbox Demo for Transparent Racing HUD in MetaDrive NATIVE Mode (Gate HUD-2).

Demonstrates the complete seven-component visual experience:
- Minimap (HUD-1): Top-left orthographic road geometry, ego marker, and trajectory trail.
- Route Progress Bar (HUD-2): Top-center percentage bar.
- Digital Speedometer & Arc (HUD-2): Bottom-right large digits and 240-degree circular arc.
- Left & Right Turn Indicators (HUD-2): Bottom-left amber chevrons reacting to steering commands.
- Brake & Throttle Meters (HUD-2): Bottom-right vertical meters reflecting command intensity.

Key Verification Constraints:
- Confined strictly to exploratory TRAIN split geometries (SANDBOX mode).
- Does NOT execute on canonical TEST or VALIDATION holdout suites.
- Does NOT modify production src/launcher/executor.py.
- Demonstrates throttle, brake, left steer, and right steer via a clearly labeled scripted sequence.
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
from src.platform import get_action_adapter
from src.visualization.minimap_overlay import MiniMapOverlay
from src.visualization.racing_hud import RacingHud
from src.visualization.telemetry_model import create_telemetry_snapshot


def run_racing_hud_demo(
    steps: int = 80,
    render_mode: str = "NATIVE",
    tier: str = "Easy",
    capture_screenshot_path: Optional[str] = None,
) -> bool:
    """
    Executes a standalone sandbox demo of the HUD-2 Transparent Racing HUD.
    Returns True if completed successfully, False otherwise.
    """
    print("=" * 68)
    print("XETUHANH — HUD-2 TRANSPARENT RACING HUD STANDALONE DEMO (SANDBOX)")
    print("=" * 68)
    print(f"  Launcher Mode:       SANDBOX (Exploratory TRAIN only)")
    print(f"  Render Mode:         {render_mode}")
    print(f"  Difficulty Tier:     {tier}")
    print(f"  Driving Policy:      Scripted Multi-Phase Maneuvers (Throttle / Turn / Brake)")
    print(f"  Action Adapter:      continuous_box2_v1")
    print(f"  Max Steps:           {steps}")

    # 1. Resolve sandbox target case (strictly TRAIN split)
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

    # 4. Action adapter
    adapter = get_action_adapter("continuous_box2_v1")

    env = None
    minimap = None
    hud = None

    try:
        print("\n--- Initializing MetaDrive Environment ---")
        env = MetaDriveEnv(env_config)
        raw_obs, info = env.reset(seed=case.environment_seed)

        # 5. Initialize visualizers (NATIVE mode only)
        if render_mode.upper() == "NATIVE":
            print("--- Initializing MiniMapOverlay and RacingHud in Window ---")
            minimap = MiniMapOverlay(env)
            hud = RacingHud(env.engine)
            print(f"  Minimap Viewport:    {minimap.REGION}")
            print(f"  Racing HUD:          All 6 widgets attached to aspect2d successfully")

        print(f"\n--- Stepping Simulation ({steps} steps @ 10 Hz) ---")
        step_idx = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and step_idx < steps:
            # Multi-phase scripted action for comprehensive UI state verification:
            # - Phase 1 (0..24): Straight acceleration (throttle=0.6, steering=0.0)
            # - Phase 2 (25..44): Left turn (steering=+0.25, throttle=0.2)
            # - Phase 3 (45..64): Right turn (steering=-0.25, throttle=0.2)
            # - Phase 4 (65..85): Combined Left Steering + Braking (steering=+0.30, throttle=-0.75)
            if step_idx < 25:
                steer_cmd = 0.0
                tb_cmd = 0.6
                phase_name = "ACCEL"
            elif step_idx < 45:
                steer_cmd = 0.25
                tb_cmd = 0.2
                phase_name = "LEFT"
            elif step_idx < 65:
                steer_cmd = -0.25
                tb_cmd = 0.2
                phase_name = "RIGHT"
            else:
                steer_cmd = 0.30
                tb_cmd = -0.75
                phase_name = "BRK+LEFT"

            canonical_act = adapter.to_canonical([steer_cmd, tb_cmd])
            raw_obs, r_env, terminated, truncated, step_info = env.step(canonical_act.to_numpy())
            step_idx += 1

            # Extract authoritative telemetry pose & metrics without fabricating fallbacks
            pos = getattr(env.agent, "position", None) if hasattr(env, "agent") else None
            heading = getattr(env.agent, "heading_theta", None) if hasattr(env, "agent") else None
            speed = getattr(env.agent, "speed_km_h", None) if hasattr(env, "agent") else None
            route_comp = step_info.get("route_completion") if isinstance(step_info, dict) else None
            sim_time = step_idx * 0.1

            # Construct immutable telemetry snapshot via sanitizing factory
            snapshot = create_telemetry_snapshot(
                step_index=step_idx,
                simulation_time_s=sim_time,
                world_position=pos,
                heading_theta_rad=heading,
                speed_kmh=speed,
                route_completion=route_comp,
                canonical_steering=steer_cmd,
                canonical_throttle_brake=tb_cmd,
            )

            # Update visualizers defensively
            if minimap is not None:
                wx = snapshot.world_position[0] if snapshot.world_position is not None else None
                wy = snapshot.world_position[1] if snapshot.world_position is not None else None
                minimap.update(world_x=wx, world_y=wy, heading_rad=snapshot.heading_theta_rad)
            if hud is not None:
                hud.update(snapshot)

            if step_idx % 15 == 0 or step_idx == 1:
                disp_spd = f"{snapshot.speed_kmh:5.1f} km/h" if snapshot.speed_kmh is not None else "  --- km/h"
                disp_route = f"{snapshot.route_completion:5.1%}" if snapshot.route_completion is not None else "  ---"
                print(
                    f"  Step {step_idx:2d}/{steps} [{phase_name:8s}] | "
                    f"Spd={disp_spd} | "
                    f"Route={disp_route} | "
                    f"Steer={steer_cmd:+5.2f} | "
                    f"ThrBrk={tb_cmd:+5.2f}"
                )

            if render_mode.upper() == "NATIVE":
                time.sleep(0.02)

        # Optional screenshot capture
        if capture_screenshot_path and hud is not None:
            try:
                import panda3d.core as p3d
                screen_path = Path(capture_screenshot_path).resolve()
                screen_path.parent.mkdir(parents=True, exist_ok=True)
                fn = p3d.Filename.fromOsSpecific(str(screen_path))
                # Render current frame so latest step geometry is committed to framebuffer
                env.engine.graphicsEngine.renderFrame()
                env.engine.win.saveScreenshot(fn)
                # Flush frames so image is committed to disk before window close
                env.engine.graphicsEngine.renderFrame()
                env.engine.graphicsEngine.renderFrame()
                if screen_path.exists():
                    print(f"\n  [SCREENSHOT] Saved native render capture ({screen_path.stat().st_size} bytes) to: {screen_path}")
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
        # Safe cleanup in reverse order
        if hud is not None:
            try:
                hud.destroy()
                print("  [CLEANUP] RacingHud destroyed safely.")
            except Exception as e:
                print(f"  [CLEANUP WARNING] RacingHud destroy exception: {e}")

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
    parser = argparse.ArgumentParser(description="Standalone Sandbox Demo for Transparent Racing HUD (Gate HUD-2)")
    parser.add_argument("--steps", type=int, default=80, help="Number of simulation steps (default: 80)")
    parser.add_argument("--render", type=str, default="NATIVE", choices=["NATIVE", "OFF", "native", "off"], help="Render mode")
    parser.add_argument("--tier", type=str, default="Easy", choices=["Easy", "Medium", "Hard", "Extreme"], help="Difficulty tier")
    parser.add_argument("--screenshot", type=str, default=None, help="Optional output path for screenshot capture")
    args = parser.parse_args()

    success = run_racing_hud_demo(
        steps=args.steps,
        render_mode=args.render.upper(),
        tier=args.tier,
        capture_screenshot_path=args.screenshot,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
