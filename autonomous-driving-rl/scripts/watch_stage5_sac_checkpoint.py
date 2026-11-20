"""Open a native MetaDrive window and watch a frozen SAC checkpoint on TRAIN geometry."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

try:  # Load torch before Panda3D on Windows.
    import torch  # noqa: F401
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.stage5.checkpoint import load_sac_checkpoint
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.training.stage5.pilot_protocol import diagnostic_schedule, load_pilot_split
from src.training.stage5.train_sampler import SamplerStrategy, TrainCaseSampler


def watch(checkpoint: Path, split_path: Path, geometry_id: str | None, delay_s: float) -> None:
    encoder = Stage5ObservationEncoder(core_only=False)
    model, metadata = load_sac_checkpoint(
        checkpoint, device="cpu", expected_encoder_fingerprint=encoder.get_schema_fingerprint()
    )
    split = load_pilot_split(split_path)
    schedule = diagnostic_schedule(split)
    selected = next((row for row in schedule if row["geometry_id"] == geometry_id), None) if geometry_id else schedule[0]
    if selected is None:
        raise ValueError(f"geometry_id is not in PILOT_DIAGNOSTIC: {geometry_id}")

    sampler = TrainCaseSampler(
        training_root_seed=101, strategy=SamplerStrategy.ROUND_ROBIN,
        allowed_geometry_ids=[row["geometry_id"] for row in split["pilot_diagnostic"]],
        project_root=ROOT,
    )
    case = sampler.get_case_for_geometry_id(selected["geometry_id"], selected["environment_seed"])
    env = Stage5TrainingEnv(sampler=sampler, encoder=encoder, project_root=ROOT,
                            simulator_render_mode="NATIVE")
    try:
        observation, info = env.reset(options={"case": case})
        print(f"Checkpoint: {checkpoint} (training_step={metadata['total_timesteps']})")
        print(f"Internal TRAIN diagnostic: {selected['geometry_id']} | {selected['tier']} / {selected['sequence']} "
              f"| geometry_seed={selected['geometry_generation_seed']} | environment_seed={selected['environment_seed']}")
        print("Frozen deterministic policy. Close the simulator window or press Ctrl+C to stop.")
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            if int(info.get("step_index", 0)) % 10 == 0:
                print(f"step={info['step_index']:4d} speed={info.get('speed_km_h', 0):5.1f} km/h "
                      f"route={info.get('route_completion', 0):.3f} "
                      f"steer={info.get('action_steering', 0):+.3f} "
                      f"throttle_brake={info.get('action_throttle_brake', 0):+.3f} reward={reward:+.4f}")
            if delay_s > 0:
                time.sleep(delay_s)
        print(f"Episode ended: {info.get('primary_reason')} | clean_success={info.get('clean_success')} "
              f"| route={info.get('route_completion')}")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=ROOT / "results/stage5/pilot_split_v1.json")
    parser.add_argument("--geometry-id", default=None, help="Must belong to the fixed internal diagnostic split")
    parser.add_argument("--delay", type=float, default=0.03, help="Wall-clock delay per simulated step")
    args = parser.parse_args()
    watch(args.checkpoint, args.split, args.geometry_id, args.delay)


if __name__ == "__main__":
    main()
