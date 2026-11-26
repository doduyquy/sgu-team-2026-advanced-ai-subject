"""Prepare portable SAC binding evidence for review of a locked development candidate."""
from __future__ import annotations

import argparse
from pathlib import Path

from cloud.kaggle.common import fingerprint, read, verify, write


def binding_evidence(metadata):
    """Use content identities so moving the private bundle leaves the evidence unchanged."""
    recipe = metadata["training_recipe"]
    files = recipe["files"]
    return {
        "model_sha256": metadata["model_sha256"],
        "training_state_sha256": metadata["training_state_sha256"],
        "training_job": metadata["training_job"],
        "checkpoint_format": "stable-baselines3 SAC archive",
        "framework_version": metadata["framework_version"],
        "metadrive_commit": metadata["metadrive_commit"],
        "encoder": {"id": metadata["encoder_id"], "fingerprint": metadata["encoder_fingerprint"],
                    "dimensions": metadata["observation_dim"]},
        "observation_transform": metadata["observation_transform_fingerprint"],
        "policy_input_adapter": metadata["policy_input_adapter"],
        "task_context_producer": metadata["task_context_producer"],
        "action": {"id": metadata["action_adapter_id"], "dimensions": metadata["action_dim"]},
        "reward_recipe_sha256": fingerprint(metadata["native_reward"]),
        "training_recipe_sha256": fingerprint(recipe),
        "train_split_sha256": files["results/stage5/pilot_split_v1.json"],
        "training_root_seed": metadata["training_root_seed"],
        "cumulative_decisions": metadata["total_timesteps"],
        "optimizer_updates": metadata["optimizer_updates"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lock, metadata = read(args.model_lock), read(args.bundle / "metadata.json")
    verify(args.bundle / "model.zip", lock["model_sha256"])
    evidence = binding_evidence(metadata)
    if any(evidence[key] != lock[locked] for key, locked in (
        ("model_sha256", "model_sha256"), ("training_state_sha256", "full_state_sha256"), ("training_job", "job")
    )):
        raise ValueError("The deployment bundle does not match the frozen model lock")
    write(args.out, {"scope": "development candidate binding review", "approval": "pending",
                     "evidence": evidence, "evidence_sha256": fingerprint(evidence)})


if __name__ == "__main__":
    main()
