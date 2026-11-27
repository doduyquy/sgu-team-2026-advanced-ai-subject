"""Artifact identities shared by launcher and runtime."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "runs/kaggle"
SOURCE_SHA = "96b6b78cc1e89ddf967df6bf98ead74cb7415553c7fb7c4f07a8265de6750b8f"
SOURCE = ROOT / "runs/stage5_native_reward/stage5_native_reward_easy_scs0_seed101_20261121_retry1/checkpoints/step_100000/model.zip"
METADRIVE_PIN = "85e5dadc6c7436d324348f6e3d8f8e680c06b4db"

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def write(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)

def verify(path, expected):
    if not path.is_file() or sha(path) != expected:
        raise ValueError(f"Missing or mismatched SHA-256 for {path.name}")

