"""Reproduce the paired Stage 5 steering comparison through the SANDBOX launcher."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 7))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with args.cases.open(encoding="utf-8-sig", newline="") as stream:
        cases = list(csv.DictReader(stream))
    profiles = ("off", "fixed", "adaptive")
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def run(item):
        index, row, profile = item
        name = f"{row['tier'].lower()}-{index:02d}-{profile}"
        command = [
            sys.executable, "-B", "-m", "src.launcher.cli", "run", "--mode", "SANDBOX",
            "--agent", "stage5_sac_terminal_preview", "--checkpoint", str(args.checkpoint),
            "--task-context-producer", "reference_route_station_proposal_v1",
            "--tier", row["tier"], "--sequence", row["family"], "--geom-seed", row["geometry_seed"],
            "--env-seed", row["environment_seed"], "--steering-profile", profile, "--render", "OFF",
            "--wandb", "DISABLED", "--runs-root", str(args.out), "--run-id", name,
        ]
        with (args.out / f"{name}.log").open("x", encoding="utf-8") as log:
            result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        print(name, result.returncode, flush=True)
        return result.returncode

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        return int(any(list(pool.map(run, ((i, row, profile) for i, row in enumerate(cases, 1)
                                          for profile in profiles)))))


if __name__ == "__main__":
    raise SystemExit(main())
