"""Isolated CPU training worker, launched by Workbench or directly by Python."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.astar_heuristic import train_astar_heuristic
from src.training.protocol import training_line
from src.workbench.protocol import WorkbenchMessageType, WorkbenchMessageV1
from src.platform import sanitize_error_message


def emit(kind, payload):
    print(training_line(kind, payload), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("smoke", "standard"), default="smoke")
    parser.add_argument("--no-publish", action="store_true", help="Keep the user's selected checkpoint unchanged during integration checks")
    args = parser.parse_args()
    success = False
    try:
        report = train_astar_heuristic(ROOT, args.preset, lambda event: emit("PROGRESS", event), publish_latest=not args.no_publish)
        emit("REPORT", report)
        success = report["status"] == "COMPLETE"
    except (Exception, KeyboardInterrupt) as exc:
        emit("ERROR", dict(message=sanitize_error_message(exc), type=type(exc).__name__))
    finally:
        print(WorkbenchMessageV1.create(WorkbenchMessageType.WORKER_DONE,
                                        dict(operation="TRAIN", success=success)).to_line(), flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
