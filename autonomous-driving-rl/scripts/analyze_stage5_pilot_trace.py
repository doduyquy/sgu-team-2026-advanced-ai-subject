"""Analyze policy action variation and first actor-layer ReLU activity in traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.stage5.checkpoint import load_sac_checkpoint
from src.training.stage5.pilot_observation_transform import transform_fingerprint


def analyze_trace(trace_path: Path, checkpoints: List[Path]) -> Dict[str, Any]:
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Trace is empty: {trace_path}")
    observations = np.asarray([row["policy_observation"] for row in rows], dtype=np.float32)
    actions = np.asarray([row["action"] for row in rows], dtype=np.float32)
    result: Dict[str, Any] = {
        "trace": str(trace_path), "steps": len(rows),
        "geometry_id": rows[0].get("geometry_id", trace_path.stem.split("_", 2)[-1]),
        "action_mean": actions.mean(axis=0).tolist(),
        "action_std": actions.std(axis=0).tolist(),
        "action_unique_4dp": int(len({tuple(x) for x in np.round(actions, 4)})),
        "route_start": rows[0].get("route_completion"),
        "route_end": rows[-1].get("route_completion"),
        "terminal_safety_flags": rows[-1].get("raw_safety_flags", {}),
        "checkpoints": {},
    }
    for checkpoint in checkpoints:
        model, metadata = load_sac_checkpoint(checkpoint, device="cpu")
        if metadata.get("observation_transform_fingerprint") != transform_fingerprint():
            raise ValueError(f"Checkpoint {checkpoint} does not use this trace's waypoint-scale transform")
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            preactivation = model.policy.actor.latent_pi[0](obs_tensor)
            postactivation = model.policy.actor.latent_pi[1](preactivation)
        positive = (preactivation > 0).sum(dim=1).cpu().numpy()
        nonzero = (postactivation != 0).sum(dim=1).cpu().numpy()
        result["checkpoints"][str(metadata["total_timesteps"])] = {
            "model_sha256": metadata["model_sha256"],
            "first_layer_units": int(preactivation.shape[1]),
            "positive_preactivation_units_mean": float(positive.mean()),
            "positive_preactivation_units_min": int(positive.min()),
            "positive_preactivation_units_max": int(positive.max()),
            "zero_output_units_mean": float((postactivation == 0).sum(dim=1).float().mean().item()),
            "zero_output_unit_fraction": float(1.0 - nonzero.mean() / postactivation.shape[1]),
            "fully_dead_layer_fraction": float(np.mean(nonzero == 0)),
            "preactivation_positive_fraction": float(np.mean(preactivation.cpu().numpy() > 0)),
            "preactivation_mean": float(preactivation.mean().item()),
            "preactivation_std": float(preactivation.std(unbiased=False).item()),
            "first_layer_bias_mean": float(model.policy.actor.latent_pi[0].bias.mean().item()),
            "first_layer_bias_std": float(model.policy.actor.latent_pi[0].bias.std(unbiased=False).item()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = [analyze_trace(path, args.checkpoint) for path in args.trace]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
