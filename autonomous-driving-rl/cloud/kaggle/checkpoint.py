"""Transport complete Stage 5 checkpoints as individually verifiable files."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from cloud.kaggle.common import read, sha, verify, write

PARTS = ("model.zip", "replay.pkl", "rng.pt", "metrics.json")
STATE_FILES = (*PARTS, "COMPLETE.json")
MAX_SHARD_BYTES = 64 * 1024**2


def export_checkpoint(state, state_sha, out, shard_bytes=MAX_SHARD_BYTES):
    if not 0 < shard_bytes <= MAX_SHARD_BYTES:
        raise ValueError("Invalid checkpoint shard size")
    verify(state / "COMPLETE.json", state_sha)
    marker = read(state / "COMPLETE.json")
    if marker["schema"] != 1 or set(marker["files"]) != set(PARTS):
        raise ValueError("Invalid checkpoint file manifest")
    if out.exists():
        raise FileExistsError(out)
    stage = out.with_name(out.name + ".staging")
    stage.mkdir(parents=True)
    files = {}
    for name in STATE_FILES:
        expected = state_sha if name == "COMPLETE.json" else marker["files"][name]
        digest = hashlib.sha256()
        chunks = []
        with (state / name).open("rb") as source:
            while block := source.read(shard_bytes):
                digest.update(block)
                chunk = stage / f"{name}.{len(chunks):06d}.bin"
                with chunk.open("xb") as target:
                    target.write(block)
                chunks.append({"bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()})
        if digest.hexdigest() != expected or not chunks:
            raise ValueError(f"Missing or mismatched checkpoint data for {name}")
        files[name] = chunks
    write(stage / "INDEX.json", {"schema": 1, "state_sha256": state_sha, "files": files})
    stage.rename(out)
    return sha(out / "INDEX.json")


def restore_checkpoint(package, index_sha, out):
    marker_path = package / "INDEX.json"
    verify(marker_path, index_sha)
    if marker_path.stat().st_size > 4_000_000:
        raise ValueError("Checkpoint index exceeds size limit")
    index = read(marker_path)
    if index["schema"] != 1 or set(index["files"]) != set(STATE_FILES):
        raise ValueError("Unexpected checkpoint index")
    if out.exists():
        raise FileExistsError(out)
    stage = out.with_name(out.name + ".staging")
    stage.mkdir(parents=True)
    for name in STATE_FILES:
        chunks = index["files"][name]
        if not chunks:
            raise ValueError("Checkpoint file is missing shards")
        with (stage / name).open("xb") as output:
            for number, shard in enumerate(chunks):
                path = package / f"{name}.{number:06d}.bin"
                if path.is_symlink() or not 0 < shard["bytes"] <= MAX_SHARD_BYTES or path.stat().st_size != shard["bytes"]:
                    raise ValueError(f"Invalid checkpoint shard {path.name}")
                block = path.read_bytes()
                if len(block) != shard["bytes"] or hashlib.sha256(block).hexdigest() != shard["sha256"]:
                    raise ValueError(f"Checksum mismatch for {path.name}")
                output.write(block)
    verify(stage / "COMPLETE.json", index["state_sha256"])
    marker = read(stage / "COMPLETE.json")
    if marker["schema"] != 1 or set(marker["files"]) != set(PARTS):
        raise ValueError("Unexpected restored checkpoint file manifest")
    for name in PARTS:
        verify(stage / name, marker["files"][name])
    stage.rename(out)
    return index["state_sha256"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("export", "restore"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    action = export_checkpoint if args.command == "export" else restore_checkpoint
    print(action(args.input, args.sha, args.out))


if __name__ == "__main__":
    main()
