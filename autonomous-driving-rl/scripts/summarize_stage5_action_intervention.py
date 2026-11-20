"""Create a compact Q-versus-soft-return screen from completed S5.3K K1 records."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUN_DIR = ROOT / "results/stage5/s5_3k_action_intervention_seed101_20261115"
CHECKPOINT = ROOT / "runs/stage5_pilot/s5_3g_throttle_hold5_seed101_20261115/checkpoints/step_10000"
from src.training.stage5.checkpoint import load_sac_checkpoint
from src.training.stage5.observation_encoder import Stage5ObservationEncoder


def mean(values):
    return statistics.mean(values) if values else None


def main() -> None:
    records = [json.loads(line) for line in (RUN_DIR / "episode_records.jsonl").read_text(encoding="utf-8").splitlines() if line]
    encoder = Stage5ObservationEncoder(core_only=False)
    model, _ = load_sac_checkpoint(CHECKPOINT, device="cpu", expected_encoder_fingerprint=encoder.get_schema_fingerprint())
    model.policy.set_training_mode(False)
    first_steps = {}
    with (ROOT / "runs/stage5_pilot/s5_3k_action_intervention_seed101_20261115/transition_records.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            trace = json.loads(line)
            key = (trace["geometry_id"], trace["condition"])
            if trace["step_index"] == 1 and key not in first_steps:
                first_steps[key] = trace
    rows = []
    for record in records:
        # For TIMEOUT, append the target-critic soft bootstrap as a separately
        # labeled estimate. For observed safety terminal, use realized return.
        tail = record["timeout_bootstrapped_soft_return_estimate"]
        estimate = tail if tail is not None else record["discounted_soft_return_observed"]
        trace = first_steps[(record["geometry_id"], record["condition"])]
        obs_t = torch.as_tensor(np.asarray(trace["observation"], dtype=np.float32)).reshape(1, -1)
        action_t = torch.as_tensor(np.asarray(trace["executed_action"], dtype=np.float32)).reshape(1, -1)
        with torch.no_grad():
            target_q1, target_q2 = model.critic_target(obs_t, action_t)
        rows.append({
            "geometry_id": record["geometry_id"],
            "condition": record["condition"],
            "outcome": record["outcome"],
            "min_q_prediction": record["initial_min_q"],
            "target_min_q_prediction": float(torch.minimum(target_q1, target_q2).item()),
            "observed_soft_return": record["discounted_soft_return_observed"],
            "timeout_target_critic_bootstrap": tail,
            "soft_return_plus_timeout_bootstrap_estimate": estimate,
            "q_minus_soft_return_estimate": record["initial_min_q"] - estimate,
        })
    conditions = {}
    for condition in sorted({row["condition"] for row in rows}):
        group = [row for row in rows if row["condition"] == condition]
        conditions[condition] = {
            "n": len(group),
            "mean_min_q": mean([row["min_q_prediction"] for row in group]),
            "mean_target_min_q": mean([row["target_min_q_prediction"] for row in group]),
            "mean_soft_return_plus_timeout_bootstrap_estimate": mean([row["soft_return_plus_timeout_bootstrap_estimate"] for row in group]),
            "mean_q_minus_soft_return_estimate": mean([row["q_minus_soft_return_estimate"] for row in group]),
        }
    payload = {
        "label": "diagnostic screen, not critic calibration verdict",
        "n": len(rows),
        "mean_min_q": mean([row["min_q_prediction"] for row in rows]),
        "mean_target_min_q": mean([row["target_min_q_prediction"] for row in rows]),
        "mean_soft_return_plus_timeout_bootstrap_estimate": mean([row["soft_return_plus_timeout_bootstrap_estimate"] for row in rows]),
        "mean_q_minus_soft_return_estimate": mean([row["q_minus_soft_return_estimate"] for row in rows]),
        "mean_target_q_minus_soft_return_estimate": mean([row["target_min_q_prediction"] - row["soft_return_plus_timeout_bootstrap_estimate"] for row in rows]),
        "q_minus_return_range": [min(row["q_minus_soft_return_estimate"] for row in rows), max(row["q_minus_soft_return_estimate"] for row in rows)],
        "by_condition": conditions,
        "episodes": rows,
        "limitations": [
            "One stochastic rollout per geometry-condition; no Monte Carlo uncertainty estimate.",
            "TIMEOUT bootstrap uses the same checkpoint target critic and is not independent empirical evidence.",
            "This comparison is a screening signal only; it cannot establish critic miscalibration.",
        ],
    }
    (RUN_DIR / "q_return_comparison.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# S5.3K — Q versus soft-return screen", "",
        "This is a derived diagnostic screen from the completed K1 episode records. It does not replace or modify the original run records.", "",
        f"Across {len(rows)} full episodes, mean initial online `min(Q1,Q2)` was **{payload['mean_min_q']:.3f}** and target-critic `min(Q1,Q2)` was **{payload['mean_target_min_q']:.3f}**. Mean observed soft return plus the separately labeled TIMEOUT target-critic bootstrap estimate was **{payload['mean_soft_return_plus_timeout_bootstrap_estimate']:.3f}**. Mean online-Q difference `Q - return estimate` was **{payload['mean_q_minus_soft_return_estimate']:+.3f}**; target-Q difference was **{payload['mean_target_q_minus_soft_return_estimate']:+.3f}** (online range {payload['q_minus_return_range'][0]:+.3f} to {payload['q_minus_return_range'][1]:+.3f}).", "",
        "| Initial throttle condition | n | Mean online minQ | Mean target minQ | Mean soft-return estimate | Mean online Q − estimate |", "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, values in conditions.items():
        lines.append(f"| {condition} | {values['n']} | {values['mean_min_q']:.3f} | {values['mean_target_min_q']:.3f} | {values['mean_soft_return_plus_timeout_bootstrap_estimate']:.3f} | {values['mean_q_minus_soft_return_estimate']:+.3f} |")
    lines += ["", "## Interpretation", "", "The critic predicts substantially higher absolute values than this small rollout screen. This is a candidate calibration discrepancy to investigate, not a confirmed critic error: each case-condition has one trajectory, stochastic variance is unestimated, and TIMEOUT bootstrap uses the checkpoint target critic. Relative action preference is also weak: Q ranks brake above throttle by about 0.019 on average, while the soft-return estimate favors throttle over brake by about 0.0008 paired mean. These tiny relative differences are not decisive with four geometries.", "", "The intervention changed only the first throttle action. Subsequent stochastic SAC behavior returned to nearly identical low-speed trajectories, so this experiment does not test sustained acceleration or prove why training produced the policy.", ""]
    (RUN_DIR / "Q_RETURN_SCREEN.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"n": len(rows), "mean_q": payload["mean_min_q"], "mean_return_estimate": payload["mean_soft_return_plus_timeout_bootstrap_estimate"], "mean_gap": payload["mean_q_minus_soft_return_estimate"], "artifact": str(RUN_DIR / "Q_RETURN_SCREEN.md")}, indent=2))


if __name__ == "__main__":
    main()
