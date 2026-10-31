"""Run backend and Qt tests in separate processes, with durable UTF-8 logs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("all", "backend", "gui"), default="all")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or ROOT / "runs" / ("tests_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:6])
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(QT_QPA_PLATFORM="offscreen", PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    modules = sorted((ROOT / "tests").glob("test_*.py"))
    groups = ("backend", "gui") if args.group == "all" else (args.group,)
    request = {"python": sys.version, "executable": sys.executable, "scope": "Regression tests; no new benchmark score",
               "groups": groups, "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for folder in ("src", "tests") for p in sorted((ROOT / folder).rglob("*.py"))},
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(output / "request.json", request)
    results = []
    try:
        for group in groups:
            selected = ["tests." + p.stem for p in modules if p.stem.startswith("test_workbench") == (group == "gui")]
            command = [sys.executable, "-B", "-X", "utf8", "-X", "faulthandler", "-m", "unittest", *selected, "-q"]
            log = output / (group + ".log")
            with log.open("xb") as stream:
                result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            match = re.search(r"Ran (\d+) tests in ([\d.]+)s", log.read_text(encoding="utf8", errors="replace"))
            entry = {"group": group, "command": command, "exit_code": result.returncode,
                     "status": "PASSED" if result.returncode == 0 else "FAILED", "log": str(log),
                     "tests": int(match[1]) if match else None, "seconds": float(match[2]) if match else None,
                     "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()}
            results.append(entry)
            print(json.dumps(entry), flush=True)
    finally:
        passed = len(results) == len(groups) and all(row["exit_code"] == 0 for row in results)
        write_json(output / "summary.json", {"status": "PASSED" if passed else "FAILED_OR_INTERRUPTED",
                   "finished_at_utc": datetime.now(timezone.utc).isoformat(), "groups": results,
                   "total_tests": sum(row["tests"] or 0 for row in results)})
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
