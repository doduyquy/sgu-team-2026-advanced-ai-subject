"""
Durable SAC Checkpoint Serialization and Metadata Verification (Gate S5.2).

Manages model saving, cryptographic verification, metadata binding, and loading:
- model.zip: Stable-Baselines3 SAC policy weights and architecture.
- metadata.json: Authoritative training provenance (seed, timesteps, git commit,
  encoder ID, encoder fingerprint, dependency versions, model SHA-256).
- Optional replay_buffer.pkl: Serialization of the experience replay buffer for
  exact, bit-for-bit training resumption.
- Strict Integrity Verification: Rejects checkpoints with tampered SHA-256 digests
  or mismatched observation encoder schemas.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3 import SAC
import torch

from src.training.stage5.observation_encoder import Stage5ObservationEncoder


class CheckpointIntegrityError(ValueError):
    """Raised when checkpoint bundle verification fails cryptographically or structurally."""
    pass


def compute_file_sha256(file_path: Path) -> str:
    """Computes the cryptographic SHA-256 digest of a local file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found for SHA-256 computation: {file_path}")
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit_sha(project_root: Optional[Path] = None) -> str:
    """Retrieves the current git commit SHA if available, else 'UNKNOWN'."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        cwd = project_root if project_root else Path.cwd()
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def save_sac_checkpoint(
    model: SAC,
    checkpoint_dir: Path,
    encoder: Stage5ObservationEncoder,
    total_timesteps: int,
    training_root_seed: int,
    training_config: Optional[Dict[str, Any]] = None,
    training_summary: Optional[Dict[str, Any]] = None,
    save_replay_buffer: bool = False,
    project_root: Optional[Path] = None,
) -> Path:
    """
    Saves a complete, verifiable SAC checkpoint bundle with metadata.

    Args:
        model: Trained Stable-Baselines3 SAC model.
        checkpoint_dir: Target directory path to store checkpoint artifacts.
        encoder: The Stage5ObservationEncoder used to train this policy.
        total_timesteps: Cumulative environment steps completed during training.
        training_root_seed: Master root seed governing the training run.
        training_config: Optional dictionary of hyperparameters and environment settings.
        training_summary: Optional summary of training progress and metrics.
        save_replay_buffer: If True, saves replay_buffer.pkl alongside model.zip.
        project_root: Subproject root directory path.

    Returns:
        Path to the validated checkpoint directory.
    """
    checkpoint_dir = Path(checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save SB3 SAC model
    model_path = checkpoint_dir / "model.zip"
    model.save(str(model_path))

    # 2. Compute model SHA-256
    model_sha256 = compute_file_sha256(model_path)

    # 3. Optionally save replay buffer
    rb_path = checkpoint_dir / "replay_buffer.pkl"
    rb_sha256 = None
    if save_replay_buffer:
        # Never leave or reuse a stale replay buffer from a previous attempt
        if rb_path.exists():
            rb_path.unlink()
        try:
            model.save_replay_buffer(str(rb_path))
            if not rb_path.exists():
                raise CheckpointIntegrityError(f"Replay buffer file was not created: {rb_path}")
            rb_sha256 = compute_file_sha256(rb_path)
        except Exception as e:
            if rb_path.exists():
                rb_path.unlink()
            raise CheckpointIntegrityError(f"Failed to persist replay buffer: {e}") from e

    # 4. Construct comprehensive metadata dictionary
    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    git_sha = get_git_commit_sha(project_root)

    metadata: Dict[str, Any] = {
        "model_name": "stage5_sac",
        "algorithm": "SAC",
        "framework": "stable-baselines3",
        "framework_version": str(sb3.__version__),
        "torch_version": str(torch.__version__),
        "gymnasium_version": str(gym.__version__),
        "encoder_id": encoder.encoder_id,
        "encoder_version": encoder.encoder_version,
        "encoder_fingerprint": encoder.get_schema_fingerprint(),
        "observation_dim": int(encoder.dimension),
        "action_dim": 2,
        "total_timesteps": int(total_timesteps),
        "training_root_seed": int(training_root_seed),
        "model_sha256": model_sha256,
        "replay_buffer_saved": bool(save_replay_buffer and rb_path.exists()),
        "replay_buffer_sha256": rb_sha256,
        "git_commit": git_sha,
        "created_at_utc": now_utc,
        "training_resume_notice": (
            "model.zip and replay_buffer.pkl preserve neural network parameters and transition buffers. "
            "They do NOT guarantee bit-for-bit deterministic training resume because PRNG seeds, case sampler cursor, "
            "optimizer momentum/hyperparameter schedules, and MetaDrive simulator internal states are not fully restored."
        ),
        "training_config": training_config or {},
        "training_summary": training_summary or {},
    }

    metadata_path = checkpoint_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    # 5. Write checksums manifest
    checksums = {
        "model.zip": model_sha256,
        "metadata.json": compute_file_sha256(metadata_path),
    }
    if rb_sha256:
        checksums["replay_buffer.pkl"] = rb_sha256

    checksums_path = checkpoint_dir / "CHECKSUMS.sha256"
    with open(checksums_path, "w", encoding="utf-8") as f:
        for fname, digest in sorted(checksums.items()):
            f.write(f"{digest}  {fname}\n")

    return checkpoint_dir


def verify_checkpoint_integrity(
    checkpoint_dir: Path,
    expected_sha256: Optional[str] = None,
    expected_encoder_fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Verifies that a checkpoint directory is structurally complete and cryptographically valid.

    Enforces:
    1. CHECKSUMS.sha256 existence, syntax format, and coverage of all required artifacts.
    2. Exact bitwise match for model.zip, metadata.json, and replay_buffer.pkl against CHECKSUMS.sha256.
    3. Exact match between CHECKSUMS.sha256 and metadata.json internal fields.
    4. Optional verification against trusted external expected_sha256 anchor.
    5. Observation schema fingerprint matching to prevent evaluating on mismatched state representations.

    Args:
        checkpoint_dir: Directory containing model.zip, metadata.json, and CHECKSUMS.sha256.
        expected_sha256: Optional expected SHA-256 of model.zip.
        expected_encoder_fingerprint: Optional expected schema fingerprint.

    Returns:
        Verified metadata dictionary.
    """
    checkpoint_dir = Path(checkpoint_dir).resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checksums_path = checkpoint_dir / "CHECKSUMS.sha256"
    model_path = checkpoint_dir / "model.zip"
    metadata_path = checkpoint_dir / "metadata.json"

    # 1. Verify existence of manifest
    if not checksums_path.exists():
        raise CheckpointIntegrityError(f"Missing checksum manifest: {checksums_path}")

    # 2. Parse CHECKSUMS.sha256 manifest
    manifest_checksums: Dict[str, str] = {}
    with open(checksums_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        raise CheckpointIntegrityError(f"Checksum manifest {checksums_path} is empty")

    manifest_regex = re.compile(r"^([0-9a-f]{64})\s\s(\S+)$")
    for line_idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        m = manifest_regex.match(stripped)
        if not m:
            raise CheckpointIntegrityError(
                f"Malformed line {line_idx} in {checksums_path}: '{stripped}'. "
                "Expected '<sha256_hex>  <filename>'."
            )
        digest, fname = m.group(1), m.group(2)
        manifest_checksums[fname] = digest

    # 3. Verify required entries in CHECKSUMS.sha256
    for required_file in ("model.zip", "metadata.json"):
        if required_file not in manifest_checksums:
            raise CheckpointIntegrityError(
                f"Checksum manifest {checksums_path} missing required entry '{required_file}'"
            )

    # 4. Verify physical files on disk against manifest digests
    for fname, declared_hash in manifest_checksums.items():
        fpath = checkpoint_dir / fname
        if not fpath.exists():
            raise CheckpointIntegrityError(
                f"Artifact '{fname}' declared in {checksums_path} is missing from disk: {fpath}"
            )
        computed_hash = compute_file_sha256(fpath)
        if computed_hash != declared_hash:
            raise CheckpointIntegrityError(
                f"Checksum verification failed for '{fname}': "
                f"computed={computed_hash} != declared={declared_hash}"
            )

    # 5. Load and verify metadata.json
    with open(metadata_path, "r", encoding="utf-8") as f:
        try:
            metadata = json.load(f)
        except json.JSONDecodeError as e:
            raise CheckpointIntegrityError(f"Corrupt checkpoint metadata in {metadata_path}: {e}") from e

    # 6. Verify internal consistency between metadata and manifest
    recorded_model_sha = metadata.get("model_sha256")
    manifest_model_sha = manifest_checksums["model.zip"]
    if recorded_model_sha != manifest_model_sha:
        raise CheckpointIntegrityError(
            f"Model digest discrepancy! metadata.json={recorded_model_sha} != CHECKSUMS.sha256={manifest_model_sha}"
        )

    # 7. Verify replay buffer if declared
    if metadata.get("replay_buffer_saved"):
        if "replay_buffer.pkl" not in manifest_checksums:
            raise CheckpointIntegrityError(
                "metadata.json declares replay_buffer_saved=True, but replay_buffer.pkl is not in CHECKSUMS.sha256"
            )
        recorded_rb_sha = metadata.get("replay_buffer_sha256")
        manifest_rb_sha = manifest_checksums["replay_buffer.pkl"]
        if recorded_rb_sha != manifest_rb_sha:
            raise CheckpointIntegrityError(
                f"Replay buffer digest discrepancy! metadata.json={recorded_rb_sha} != CHECKSUMS.sha256={manifest_rb_sha}"
            )

    # 8. Verify against trusted external expected_sha256 anchor if supplied
    actual_model_sha256 = manifest_checksums["model.zip"]
    if expected_sha256 and actual_model_sha256 != expected_sha256:
        raise CheckpointIntegrityError(
            f"FATAL: Checkpoint digest does not match expected trusted anchor! "
            f"Actual={actual_model_sha256} != Expected={expected_sha256}"
        )

    # 9. Verify encoder schema fingerprint if requested
    if expected_encoder_fingerprint:
        actual_fingerprint = metadata.get("encoder_fingerprint")
        if actual_fingerprint != expected_encoder_fingerprint:
            raise CheckpointIntegrityError(
                f"FATAL: Observation encoder schema mismatch! "
                f"Model was trained with fingerprint '{actual_fingerprint}', "
                f"but runtime expected '{expected_encoder_fingerprint}'. "
                "Evaluating model on incompatible observation schema is strictly forbidden."
            )

    return metadata


def load_sac_checkpoint(
    checkpoint_dir: Path,
    env: Optional[Any] = None,
    device: str = "cpu",
    expected_sha256: Optional[str] = None,
    expected_encoder_fingerprint: Optional[str] = None,
    load_replay_buffer: bool = False,
) -> Tuple[SAC, Dict[str, Any]]:
    """
    Loads and cryptographically verifies a saved SAC checkpoint bundle.

    Args:
        checkpoint_dir: Directory containing model.zip and metadata.json.
        env: Optional Gymnasium environment instance.
        device: Target execution device ('cpu' or 'cuda').
        expected_sha256: Optional expected SHA-256 of model.zip.
        expected_encoder_fingerprint: Optional expected observation schema fingerprint.
        load_replay_buffer: If True, attempts to load replay_buffer.pkl.

    Returns:
        Tuple of (loaded_sac_model, verified_metadata).
    """
    metadata = verify_checkpoint_integrity(
        checkpoint_dir,
        expected_sha256=expected_sha256,
        expected_encoder_fingerprint=expected_encoder_fingerprint,
    )

    model_path = checkpoint_dir / "model.zip"
    model = SAC.load(str(model_path), env=env, device=device)

    # Optional replay buffer loading for resume workflows
    if load_replay_buffer:
        rb_path = checkpoint_dir / "replay_buffer.pkl"
        if rb_path.exists():
            model.load_replay_buffer(str(rb_path))

    return model, metadata
