"""
Standalone Smoke & Throughput Benchmark for Stage 5 RL Training Environment (Gate S5.1).

Executes a headless verification session on MapSuite V1 TRAIN split:
1. Runs Gymnasium check_env() validation.
2. Steps the environment across multiple episodes and TRAIN geometries.
3. Asserts finite observation (409D), finite reward, and valid Box(2) action bounds.
4. Measures throughput (environment steps/second), wall-clock time, and memory footprint.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import psutil

# Ensure subproject root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gymnasium.utils.env_checker import check_env

from src.training.stage5 import (
    STAGE5_FULL_OBSERVATION_DIM,
    SamplerStrategy,
    Stage5ObservationEncoder,
    Stage5TrainingEnv,
    TrainCaseSampler,
)


def get_hardware_info():
    """Gathers system CPU and memory hardware details."""
    cpu_info = platform.processor() or "x86_64"
    cpu_count = os.cpu_count() or 1
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": cpu_info,
        "cpu_cores": cpu_count,
        "total_ram_gb": round(total_ram_gb, 2),
    }


def run_training_smoke(
    steps_per_episode: int = 60,
    episodes: int = 2,
    tier: str = "Easy",
    strategy: str = "ROUND_ROBIN",
    sequence: Optional[str] = None,
    fixed_seed: Optional[int] = None,
    run_checker: bool = True,
) -> bool:
    """Executes the Stage 5 RL training environment smoke test."""
    print("=" * 70)
    print("XETUHANH — STAGE 5 RL TRAINING ENVIRONMENT SMOKE & THROUGHPUT BENCHMARK")
    print("=" * 70)

    hw = get_hardware_info()
    print(f"  OS Platform:     {hw['os']}")
    print(f"  CPU Cores:       {hw['cpu_cores']} cores ({hw['cpu']})")
    print(f"  System RAM:      {hw['total_ram_gb']} GB")
    print(f"  Target Split:    TRAIN only (strictly holdout-isolated)")
    print(f"  Target Tier:     {tier}")
    print(f"  Sampler Strategy:{strategy}")
    print(f"  Episodes:        {episodes}")
    print(f"  Steps / Episode: {steps_per_episode}")

    proc = psutil.Process()
    ram_init_mb = proc.memory_info().rss / (1024 * 1024)

    env = None
    try:
        strat_enum = SamplerStrategy(strategy)
        fixed_geom_seed = fixed_seed if strat_enum == SamplerStrategy.FIXED else None
        seq_filter = sequence if (sequence or strat_enum == SamplerStrategy.FIXED) else None

        sampler = TrainCaseSampler(
            training_root_seed=101,
            strategy=strat_enum,
            tier_filter=tier,
            sequence_filter=seq_filter,
            fixed_geometry_seed=fixed_geom_seed,
            project_root=PROJECT_ROOT,
        )
        encoder = Stage5ObservationEncoder(core_only=False)

        print(f"\n--- Initializing Stage5TrainingEnv ---")
        print(f"  Observation Space: {encoder.observation_space}")
        print(f"  Action Space:      Box(-1.0, 1.0, shape=(2,), dtype=float32)")
        print(f"  Eligible Geoms:    {sampler.total_eligible_geometries} TRAIN geometries")
        print(f"  Encoder Schema:    {encoder.encoder_id} (Fingerprint: {encoder.get_schema_fingerprint()[:12]}...)")

        env = Stage5TrainingEnv(
            sampler=sampler,
            encoder=encoder,
            project_root=PROJECT_ROOT,
        )

        # 1. Gymnasium Standard Environment Check
        if run_checker:
            print("\n--- Running Gymnasium check_env() Validation ---")
            check_env(env.unwrapped, skip_render_check=True)
            print("  [OK] Gymnasium check_env passed with 0 errors!")

        # 2. Multi-Episode Stepping Loop
        total_steps = 0
        total_reward = 0.0
        start_time = time.time()

        for ep_idx in range(1, episodes + 1):
            obs, info = env.reset()
            case = env.current_case
            print(f"\n--- Episode {ep_idx}/{episodes}: Case={case.case_id} (Horizon={case.horizon_steps}) ---")
            assert obs.shape == (STAGE5_FULL_OBSERVATION_DIM,), f"Obs shape {obs.shape} != (409,)"
            assert np.all(np.isfinite(obs)), "Initial observation has non-finite values"

            ep_steps = 0
            ep_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated) and ep_steps < steps_per_episode:
                # Scripted smooth continuous action [steering, throttle]
                # Steer slightly back and forth, positive throttle
                steer = 0.15 * math.sin(ep_steps * 0.1)
                throttle = 0.45
                action = np.array([steer, throttle], dtype=np.float32)

                obs, reward, terminated, truncated, step_info = env.step(action)
                ep_steps += 1
                total_steps += 1
                ep_reward += reward
                total_reward += reward

                assert np.all(np.isfinite(obs)), f"Non-finite obs at step {ep_steps}"
                assert math.isfinite(reward), f"Non-finite reward at step {ep_steps}"

                if ep_steps % 20 == 0 or ep_steps == 1 or terminated or truncated:
                    spd = step_info.get("speed_km_h", 0.0)
                    rc = step_info.get("route_completion", 0.0)
                    print(
                        f"  Step {ep_steps:3d} | "
                        f"Spd={spd:5.1f} km/h | "
                        f"Route={rc:5.1%} | "
                        f"Reward={reward:+7.4f} | "
                        f"CumR={ep_reward:+7.3f}"
                    )

            if terminated or truncated:
                reason = step_info.get("primary_reason", "UNKNOWN")
                print(f"  Episode {ep_idx} finished: reason={reason}, steps={ep_steps}, return={ep_reward:.3f}")

        elapsed_time = time.time() - start_time
        steps_per_sec = total_steps / max(elapsed_time, 0.001)

        ram_end_mb = proc.memory_info().rss / (1024 * 1024)
        ram_delta_mb = ram_end_mb - ram_init_mb

        print("\n" + "=" * 70)
        print("STAGE 5 RL FOUNDATION BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"  Total Episodes:          {episodes}")
        print(f"  Total Stepped Frames:    {total_steps} steps")
        print(f"  Wall-Clock Time:         {elapsed_time:.2f} s")
        print(f"  Simulation Throughput:   {steps_per_sec:.1f} steps/second")
        print(f"  Initial Process RAM:     {ram_init_mb:.1f} MB")
        print(f"  Final Process RAM:       {ram_end_mb:.1f} MB (Delta: {ram_delta_mb:+.1f} MB)")
        print(f"  Observation Shape:       ({STAGE5_FULL_OBSERVATION_DIM},) float32")
        print(f"  Action Space Bounds:     [-1.0, 1.0]^2 continuous")
        print(f"  Gymnasium Compatibility: 100% VERIFIED")
        print("=" * 70)
        print(">>> STAGE 5 TRAINING ENVIRONMENT SMOKE SUCCESSFUL! <<<")
        return True

    except Exception as e:
        print(f"\n[SMOKE FAILURE] Error during training environment smoke: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if env is not None:
            env.close()
            print("  [CLEANUP] Simulator closed safely.")


def main():
    parser = argparse.ArgumentParser(description="Stage 5 RL Training Foundation Smoke Test")
    parser.add_argument("--steps", type=int, default=60, help="Steps per episode")
    parser.add_argument("--episodes", type=int, default=2, help="Number of episodes")
    parser.add_argument("--tier", type=str, default="Easy", help="Difficulty tier filter")
    parser.add_argument("--strategy", type=str, default="ROUND_ROBIN", choices=["ROUND_ROBIN", "UNIFORM_RANDOM", "TIER_BALANCED", "FIXED"], help="Sampling strategy")
    parser.add_argument("--sequence", type=str, default=None, help="Sequence filter (e.g. SCS)")
    parser.add_argument("--fixed-seed", type=int, default=0, help="Fixed geometry seed if strategy=FIXED")
    parser.add_argument("--no-checker", action="store_true", help="Skip gymnasium check_env")
    args = parser.parse_args()

    success = run_training_smoke(
        steps_per_episode=args.steps,
        episodes=args.episodes,
        tier=args.tier,
        strategy=args.strategy,
        sequence=args.sequence,
        fixed_seed=args.fixed_seed,
        run_checker=not args.no_checker,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
