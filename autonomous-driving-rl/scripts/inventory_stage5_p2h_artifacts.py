"""Hash compact P2-H evidence and ignored run/checkpoint/replay artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/stage5/p2h_terminal_replay_seed101_20261121"
RUNS = ROOT / "runs/stage5_p2h"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    files = sorted(p for p in RESULTS.rglob("*") if p.is_file() and p.name not in {"CHECKSUMS.sha256", "artifact_inventory.json"})
    files += sorted(p for p in RUNS.rglob("*") if p.is_file())
    unique = sorted(set(files))
    rows = [{"path": p.relative_to(ROOT).as_posix(), "size_bytes": p.stat().st_size,
             "sha256": digest(p), "storage": "gitignored_run_artifact" if RUNS in p.parents else "compact_git_evidence"}
            for p in unique]
    manifest = {"experiment": "P2-H", "training_run": "s5_p2h_terminal_replay_seed101_20261121",
                "artifact_count": len(rows), "total_size_bytes": sum(x["size_bytes"] for x in rows),
                "raw_run_artifact_size_bytes": sum(x["size_bytes"] for x in rows if x["storage"] == "gitignored_run_artifact"),
                "artifacts": rows,
                "note": "Runs, checkpoints and replay buffers remain gitignored; this manifest is the local integrity inventory, not a remote backup."}
    inventory_path = RESULTS / "artifact_inventory.json"
    inventory_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    rows.append({"path": inventory_path.relative_to(ROOT).as_posix(), "size_bytes": inventory_path.stat().st_size,
                 "sha256": digest(inventory_path), "storage": "compact_git_evidence"})
    checksum_path = RESULTS / "CHECKSUMS.sha256"
    checksum_path.write_text("".join(f"{x['sha256']}  {x['path']}\n" for x in rows), encoding="ascii")
    print(json.dumps({"artifact_count": len(rows), "total_size_bytes": sum(x["size_bytes"] for x in rows),
                      "raw_run_artifact_size_bytes": manifest["raw_run_artifact_size_bytes"],
                      "checksum_manifest": checksum_path.relative_to(ROOT).as_posix(),
                      "checksum_manifest_sha256": digest(checksum_path)}, indent=2))


if __name__ == "__main__":
    main()
