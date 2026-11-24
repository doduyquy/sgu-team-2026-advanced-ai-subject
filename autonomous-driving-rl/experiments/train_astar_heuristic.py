"""Train Stage 6's heuristic from actual TRAIN scenarios using fixed presets."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "standard"), default="smoke")
    parser.add_argument("--plan", action="store_true", help="Inspect TRAIN geometries and training budget without simulation")
    args = parser.parse_args()
    from src.agents.planning_graph import load_astar_config
    from src.training.astar_heuristic import select_train_geometries, train_astar_heuristic
    config = load_astar_config()
    if args.plan:
        settings = config["presets"][args.preset]
        geometries = select_train_geometries(PROJECT_ROOT, settings["geometry_limit"])
        print(json.dumps(dict(preset=args.preset, settings=settings,
                              geometry_ids=[g["geometry_id"] for g in geometries]), indent=2))
        return 0
    report = train_astar_heuristic(PROJECT_ROOT, args.preset,
                                   progress=lambda event: print(json.dumps(event, ensure_ascii=False), flush=True))
    print(json.dumps(dict(status=report["status"], manifest=report["manifest_path"],
                         models=[dict(seed=m["training_run_seed"], best_epoch=m["best_epoch"],
                                      initial_holdout_mse=m["initial_holdout_mse"], best_holdout_mse=m["best_holdout_mse"])
                                 for m in report["models"]]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
