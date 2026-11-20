"""Offline actor-gradient audit on the frozen P2-D anchors/replay samples."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage5_p2d_actor_activation import read_protocol
from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER
from src.training.stage5.actor_activation_ablation import (
    ACTOR_LEAKY_RELU_001_V1, ACTOR_RELU_V1,
)
from src.training.stage5.checkpoint import load_sac_checkpoint
from src.training.stage5.observation_encoder import Stage5ObservationEncoder

RUN = ROOT / "runs/stage5_p2d/s5_p2d_actor_activation_seed101_20261121"
OUT = ROOT / "results/stage5/p2d_actor_activation_seed101_20261121/posthoc_layer_gradient_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gradients(model, observations: np.ndarray) -> dict:
    if observations.ndim != 2 or observations.shape[1] != 409 or len(observations) == 0:
        return {"count": int(len(observations)), "status": "NO_OBSERVATION_SAMPLE"}
    actor, critic = model.policy.actor, model.policy.critic
    obs = torch.as_tensor(observations, dtype=torch.float32, device=model.device)
    actions, log_prob = actor.action_log_prob(obs)
    q1, q2 = critic(obs, actions)
    alpha = model.log_ent_coef.detach().exp() if getattr(model, "log_ent_coef", None) is not None else torch.as_tensor(model.ent_coef, device=model.device)
    loss_entropy = (alpha * log_prob.reshape(-1, 1)).mean()
    loss_q = -torch.minimum(q1, q2).mean()
    loss_combined = loss_entropy + loss_q
    named = list(actor.named_parameters())
    params = [p for _, p in named]
    outputs = {}
    for key, loss in (("entropy", loss_entropy), ("negative_q", loss_q), ("combined", loss_combined)):
        grads = torch.autograd.grad(loss, params, retain_graph=(key != "combined"), allow_unused=True)
        groups: dict[str, float] = {}
        for (name, _), grad in zip(named, grads):
            if name.startswith("latent_pi."):
                group = ".".join(name.split(".")[:2])
            else:
                group = name.split(".")[0]
            groups[group] = groups.get(group, 0.0) + (float(grad.detach().pow(2).sum().cpu()) if grad is not None else 0.0)
        outputs[key] = {k: float(np.sqrt(v)) for k, v in groups.items()}
    with torch.no_grad():
        sampled_q1, sampled_q2 = critic(obs, actions)
    outputs["q_snapshot"] = {
        "q1_mean": float(sampled_q1.mean().cpu()), "q2_mean": float(sampled_q2.mean().cpu()),
        "twin_gap_mean_abs": float((sampled_q1 - sampled_q2).abs().mean().cpu()),
        "warning": "Q values are model estimates and are not ground-truth returns",
    }
    return {"count": int(len(observations)), "actor_loss_gradient_norms_by_layer": outputs}


def main() -> None:
    protocol = read_protocol()
    anchors_path = ROOT / protocol["anchors"]["path"]
    anchors = np.load(anchors_path, allow_pickle=False)
    encoder = Stage5ObservationEncoder(core_only=False)
    result = {"protocol_sha256": protocol["protocol_sha256"], "anchors_sha256": sha256(anchors_path), "arms": {}}
    for arm in ("A", "B"):
        expected = ACTOR_RELU_V1 if arm == "A" else ACTOR_LEAKY_RELU_001_V1
        result["arms"][arm] = {}
        for step in (0, 2000, 5000, 10000):
            checkpoint = RUN / f"arm_{arm}/checkpoints/step_{step:06d}"
            model, meta = load_sac_checkpoint(
                checkpoint, expected_encoder_fingerprint=encoder.get_schema_fingerprint(),
                task_context_producer_id=REFERENCE_TASK_PRODUCER, expected_actor_activation_id=expected,
            )
            rows = {"checkpoint_sha256": meta["model_sha256"], "fixed_201_anchors": gradients(model, anchors)}
            replay_sample_path = RUN / f"arm_{arm}/replay_samples/observations_step_{step:05d}.npz"
            if replay_sample_path.is_file():
                with np.load(replay_sample_path, allow_pickle=False) as sample:
                    data = sample["observations"]
                    moving = sample["speed_km_h"] >= 1.0
                    positive = sample["delta_route"] > 0
                    rows["replay_sample"] = {
                        "sha256": sha256(replay_sample_path),
                        "all_sample": gradients(model, data),
                        "moving": gradients(model, data[moving]),
                        "positive_progress": gradients(model, data[positive]),
                    }
            result["arms"][arm][str(step)] = rows
            del model
    result["analysis_limitations"] = [
        "Offline gradients at saved weights and samples do not reconstruct per-update gradients.",
        "The 2k and 5k replay observation samples are deterministic subsamples, not full replay buffers.",
        "Critic Q values remain estimates; this audit does not establish action ranking correctness.",
    ]
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    digest = sha256(OUT)
    OUT.with_suffix(".sha256").write_text(f"{digest}  {OUT.name}\n", encoding="ascii")
    print(json.dumps({"path": OUT.relative_to(ROOT).as_posix(), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
