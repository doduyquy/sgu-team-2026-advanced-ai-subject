"""Run collector and evaluation jobs inside the Kaggle runtime."""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import zipfile

from cloud.kaggle.common import sha, write


def main():
    from cloud.kaggle import replay, runtime

    p = argparse.ArgumentParser()
    p.add_argument("--kind", required=True, choices=("collect", "evaluate"))
    p.add_argument("--parent", required=True, type=Path)
    p.add_argument("--parent-sha", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tier", choices=("Easy", "Medium"), default="Easy")
    p.add_argument("--stream", choices=(106, 116), type=int)
    args = p.parse_args()
    parent = runtime.check_state(args.parent, args.parent_sha)
    model = args.parent / "model.zip"
    source_sha = parent["files"]["model.zip"]
    args.out.mkdir(parents=True, exist_ok=False)
    if args.kind == "evaluate":
        artifact = args.out / "panel.json"
        runtime.evaluate(SimpleNamespace(model=model, model_sha=source_sha, tier=args.tier, out=artifact))
    else:
        if args.stream is None:
            raise ValueError("A collector requires a unique stream")
        root = args.parent.parent / "collection"
        snap = root / "snapshot"
        snapshot_sha = replay.publish(model, source_sha, parent["identity"], 0, snap)
        families = {}
        for row in runtime.pool("easy"):
            families.setdefault(row["sequence"], []).append(row["geometry_id"])
        chunks = []
        for seq, geometries in enumerate(families[k] for k in sorted(families)):
            geometry = sorted(geometries)[(args.stream - 106) % len(geometries)]
            path = root / f"chunk-{seq:02}"
            replay.collect(SimpleNamespace(snapshot=snap, snapshot_sha=snapshot_sha, rows=replay.MAX_ROWS,
                                           stream=args.stream, sequence=seq, geometry=geometry,
                                           seed=31100 + args.stream * 10 + seq, out=path))
            chunks.append(sha(path / "CHUNK.json"))
        write(root / "GUIDED.json", {"snapshot_sha": snapshot_sha, "chunks": chunks})
        artifact = args.out / "guided.zip"
        with zipfile.ZipFile(artifact, "x", zipfile.ZIP_DEFLATED) as z:
            for file in sorted(root.rglob("*")):
                if file.is_file():
                    z.write(file, file.relative_to(root).as_posix())
    write(args.out / "AUX.json", {"kind": args.kind, "parent_sha": args.parent_sha, "model_sha": source_sha,
                                "file": artifact.name, "sha256": sha(artifact)})


if __name__ == "__main__":
    main()
