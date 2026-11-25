"""Export a verified Guided SAC full state for the XeTuHanh sandbox launcher."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from cloud.kaggle.common import METADRIVE_PIN, ROOT, read, sha, verify, write
from src.agents.stage5_terminal_preview_v1 import Stage5TerminalPreviewAdapterV1
from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER, task_producer_identity
from src.training.stage5.checkpoint import verify_checkpoint_integrity
from src.training.stage5.observation_encoder import Stage5ObservationEncoder
from src.training.stage5.pilot_observation_transform import GROUPWISE_PHYSICAL_V1, transform_fingerprint


def export(state, state_sha, out):
    from cloud.kaggle.runtime import check_state

    marker = check_state(state, state_sha)
    identity = marker["identity"]
    if marker.get("learning_mode") != "champion_guided_v1":
        raise ValueError("Expected a verified Terminal Preview Guided SAC state")
    for name in ("src/agents/stage5_terminal_preview_v1.py", "src/platform/task_context_producer.py",
                 "src/training/stage5/observation_encoder.py", "src/training/stage5/pilot_observation_transform.py"):
        verify(ROOT / name, identity["files"][name])
    encoder = Stage5ObservationEncoder(core_only=False)
    adapter = Stage5TerminalPreviewAdapterV1()
    model = marker["files"]["model.zip"]
    metrics = read(state / "metrics.json")
    metadata = {"model_name": "stage5_sac", "algorithm": "SAC", "framework": "stable-baselines3",
                "framework_version": identity["sb3"], "metadrive_commit": METADRIVE_PIN,
                "encoder_id": encoder.encoder_id, "encoder_fingerprint": encoder.get_schema_fingerprint(),
                "task_context_producer": task_producer_identity(REFERENCE_TASK_PRODUCER),
                "observation_transform": GROUPWISE_PHYSICAL_V1,
                "observation_transform_fingerprint": transform_fingerprint(GROUPWISE_PHYSICAL_V1),
                "policy_input_adapter": {"id": adapter.adapter_id, "fingerprint": adapter.fingerprint},
                "action_adapter_id": "continuous_box2_v1", "observation_dim": 409, "action_dim": 2,
                "model_sha256": model, "training_state_sha256": state_sha, "training_job": marker["job"],
                "training_root_seed": identity["recipe"]["seed"], "total_timesteps": marker["transitions"],
                "optimizer_updates": marker["updates"], "phase": identity["phase"], "training_recipe": identity,
                "native_reward": identity["source_recipe"], "behavior": metrics["online_behavior"],
                "replay_buffer_saved": False, "protocol_scope": "SANDBOX development candidate"}
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    with TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / "delivery"
        stage.mkdir()
        shutil.copyfile(state / "model.zip", stage / "model.zip")
        write(stage / "metadata.json", metadata)
        (stage / "CHECKSUMS.sha256").write_text(f"{model}  model.zip\n{sha(stage / 'metadata.json')}  metadata.json\n", encoding="utf-8")
        verify_checkpoint_integrity(stage, model, encoder.get_schema_fingerprint(),
                                    task_context_producer_id=REFERENCE_TASK_PRODUCER)
        stage.rename(out)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(export(args.state, args.sha, args.out)["model_sha256"])


if __name__ == "__main__":
    main()
