"""Load six frozen checkpoints and run a no-update SB3 replay fidelity probe."""
import gzip
import hashlib
import json
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from scripts.audit_stage5_task_alignment import GEOMETRY, HISTORY
from src.training.stage5.checkpoint import load_sac_checkpoint
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.pilot_observation_transform import GroupwisePhysicalObservation, apply_observation_transform
from src.training.stage5.train_sampler import TrainCaseSampler


class ObserveStep(BaseCallback):
    def _on_step(self):
        return True


def replay_probe(model):
    sampler = TrainCaseSampler(allowed_geometry_ids=[GEOMETRY])
    env = Stage5TrainingEnv(sampler=sampler)
    try:
        model.set_env(GroupwisePhysicalObservation(env))
        model._last_obs = model.env.reset()
        model.ep_info_buffer = deque(maxlen=100)
        model.ep_success_buffer = deque(maxlen=100)
        model._total_timesteps = model.num_timesteps + 2
        callback = ObserveStep()
        callback.init_callback(model)
        callback.on_training_start({}, {})
        received = []
        original = env._env.step

        def capture(action):
            received.append(np.asarray(action).copy())
            return original(action)

        env._env.step = capture
        initial_updates = model._n_updates
        records = []
        for label, starts in (("warmup", model.num_timesteps + 1), ("actor_sampling", 0)):
            position = model.replay_buffer.pos
            model.collect_rollouts(model.env, callback, TrainFreq(1, TrainFrequencyUnit.STEP),
                                   model.replay_buffer, learning_starts=starts)
            replay = model.replay_buffer.actions[position, 0]
            error = float(np.max(np.abs(replay - received[-1])))
            if error > 1e-7:
                raise AssertionError(f"{label} replay mismatch {error}")
            records.append({"phase": label, "metadrive_action": received[-1].tolist(),
                            "replay_action": replay.tolist(), "max_error": error})
        return {"optimizer_updates_delta": model._n_updates - initial_updates, "records": records,
                "scope": "sample/store only; fresh in-memory buffer; no learning; not historical replay"}
    finally:
        env.close()


def main():
    run = ROOT / "runs/stage5_meter_reward" / HISTORY
    anchors = run / "evaluation/policy_response_anchors.jsonl.gz"
    with gzip.open(anchors, "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    observations = np.asarray([row["raw_observation"] for row in records], dtype=np.float32)
    report = {"anchors_sha256": hashlib.sha256(anchors.read_bytes()).hexdigest(), "anchors": len(records), "checkpoints": {}}
    for arm in "ABC":
        for step in (10000, 30000):
            path = run / "arms" / arm / "checkpoints" / f"step_{step:06d}"
            model, meta = load_sac_checkpoint(path)
            transform = meta["observation_transform"]
            transformed = apply_observation_transform(observations, transform["transform_id"])
            actions, _ = model.predict(transformed, deterministic=True)
            with torch.no_grad():
                q = model.critic(torch.as_tensor(transformed), torch.as_tensor(actions))
            report["checkpoints"][f"{arm}-{step}"] = {
                "model_sha256": meta["model_sha256"], "transform": transform["transform_id"],
                "action_mean": actions.mean(axis=0).tolist(), "action_std": actions.std(axis=0).tolist(),
                "q1_q2_mean_at_deterministic_action": [float(x.mean()) for x in q],
            }
            if arm == "C" and step == 30000:
                report["replay_probe"] = replay_probe(model)
    output = ROOT / "runs/stage5_task_alignment_p0/frozen_policies.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(output)


if __name__ == "__main__":
    main()
