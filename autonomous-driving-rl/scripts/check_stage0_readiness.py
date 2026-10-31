"""Verify Stage 0 and optionally run a TRAIN episode using repo-local Python.

Git ownership exceptions are scoped to this process and the two named checkouts.
The normalization option only restores bytes already recorded in CHECKSUMS.sha256.
Logs go under artifacts/; episodes go under runs/. TEST is never executed here.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import unittest


PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent
sys.path.insert(0, str(PROJECT))


@contextmanager
def scoped_git_trust():
    count = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
    additions = {}
    for index, path in enumerate((REPOSITORY, REPOSITORY / ".venv" / "metadrive-src"), count):
        additions[f"GIT_CONFIG_KEY_{index}"] = "safe.directory"
        additions[f"GIT_CONFIG_VALUE_{index}"] = path.resolve().as_posix()
    additions["GIT_CONFIG_COUNT"] = str(count + 2)
    previous = {key: os.environ.get(key) for key in additions}
    os.environ.update(additions)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def normalize_dataset_eol():
    dataset = PROJECT / "datasets" / "mapsuite_v1"
    pending = []
    for line in (dataset / "CHECKSUMS.sha256").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = (dataset / relative).resolve()
        path.relative_to(dataset.resolve())
        original = path.read_bytes()
        if hashlib.sha256(original).hexdigest() == expected:
            continue
        if path.suffix not in {".csv", ".json", ".md"}:
            raise ValueError(f"Unexpected checksum mismatch: {relative}")
        normalized = original.replace(b"\r\n", b"\n")
        if hashlib.sha256(normalized).hexdigest() != expected:
            raise ValueError(f"Content mismatch beyond line endings: {relative}")
        pending.append((path, normalized))
    # The checksum list itself has no self-digest; its records are unchanged.
    checksum_file = dataset / "CHECKSUMS.sha256"
    checksum_bytes = checksum_file.read_bytes()
    if b"\r\n" in checksum_bytes:
        pending.append((checksum_file, checksum_bytes.replace(b"\r\n", b"\n")))
    for path, normalized in pending:
        path.write_bytes(normalized)
    print(f"Restored recorded LF bytes in {len(pending)} package files.")
    return len(pending)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalize-dataset-eol", action="store_true")
    parser.add_argument("--skip-tests", action="store_true", help="Reuse prior regression evidence for a new exploratory episode")
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--render", choices=["OFF", "NATIVE"], default="OFF")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = PROJECT / "artifacts" / "stage0_readiness" / stamp
    output.mkdir(parents=True)
    original_cwd = Path.cwd()
    os.chdir(PROJECT)
    summary = {"status": "RUNNING", "python": sys.executable, "output_dir": str(output),
               "test_suite": None, "smoke": None}
    try:
        if Path(sys.executable).resolve() != (REPOSITORY / ".venv" / "metadrive" / "python.exe").resolve():
            raise RuntimeError("Use the repository's .venv/metadrive/python.exe interpreter.")
        with scoped_git_trust():
            if args.normalize_dataset_eol:
                summary["normalized_file_count"] = normalize_dataset_eol()
            # Load Qt before MetaDrive to avoid the documented Windows import-order crash.
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            from PySide6.QtCore import QProcess
            from PySide6.QtWidgets import QApplication
            from scripts.verify_mapsuite_v1_dataset import verify_dataset
            with (output / "dataset_verification.log").open("w", encoding="utf-8") as log:
                with redirect_stdout(log), redirect_stderr(log):
                    valid = verify_dataset(PROJECT / "datasets/mapsuite_v1", PROJECT)
            if not valid:
                raise RuntimeError("MapSuite verification failed; see dataset_verification.log")
            print("MapSuite verification: PASS")
            from src.platform import capture_environment_provenance, PINNED_METADRIVE_COMMIT, PINNED_METADRIVE_VERSION
            environment = capture_environment_provenance(PROJECT)
            summary["environment"] = environment
            if (environment["metadrive_version"], environment["metadrive_commit"]) != (PINNED_METADRIVE_VERSION, PINNED_METADRIVE_COMMIT):
                raise RuntimeError("Imported MetaDrive does not match the pinned version/commit")
            print("MetaDrive exact pin: PASS")
            diff_check = subprocess.run(["git", "diff", "--check"], cwd=REPOSITORY,
                                        capture_output=True, text=True)
            if diff_check.returncode:
                raise RuntimeError(f"Git whitespace check failed: {diff_check.stdout}{diff_check.stderr}")
            summary["git_diff_check"] = "PASS"
            if not args.skip_tests:
                with (output / "regression.log").open("w", encoding="utf-8") as log:
                    with redirect_stdout(log), redirect_stderr(log):
                        suite = unittest.defaultTestLoader.discover(str(PROJECT / "tests"), pattern="test_*.py")
                        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
                summary["test_suite"] = {"run": result.testsRun, "failures": len(result.failures),
                                         "errors": len(result.errors), "skipped": len(result.skipped)}
                print("Regression:", summary["test_suite"])
                if not result.wasSuccessful():
                    raise RuntimeError("Regression failed; see regression.log")
            if args.run_smoke:
                from src.launcher.executor import ExperimentExecutor
                from src.launcher.models import LaunchRequestV1, LauncherMode
                from src.launcher.preflight import run_preflight
                from src.launcher.resolver import resolve_experiment_plan
                from src.platform import WandbMode
                from src.workbench.results_repository import RunArtifactRepository, IntegrityDisplayStatus
                request = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="stage0_random_naive",
                    tier="Easy", sequence="SCS", geometry_generation_seed=0, environment_seed=0,
                    agent_seed=101, render_mode=args.render, wandb_mode=WandbMode.DISABLED)
                plan = resolve_experiment_plan(request)
                assert len(plan.resolved_cases) == 1 and plan.resolved_cases[0].split == "TRAIN"
                preflight = run_preflight(plan)
                (output / "preflight.json").write_text(json.dumps(preflight.to_dict(), indent=2), encoding="utf-8")
                if not preflight.can_execute:
                    raise RuntimeError("Sandbox preflight blocked; see preflight.json")
                print("Sandbox preflight:", preflight.summary_verdict)
                with (output / "smoke.log").open("w", encoding="utf-8") as log:
                    with redirect_stdout(log), redirect_stderr(log):
                        report = ExperimentExecutor(plan, custom_run_id=f"stage0_sandbox_{stamp}").execute()
                summary["smoke"] = report.to_dict()
                snapshot = RunArtifactRepository(PROJECT / "runs").load_run_snapshot(report.run_id)
                assert report.status == "COMPLETE" and report.completed_episodes == 1
                assert snapshot is not None and not snapshot.canonical_run
                assert snapshot.integrity_status == IntegrityDisplayStatus.VERIFIED
                assert snapshot.environment_verification_status == "VERIFIED", snapshot.environment_verification_status
                # This fixed TRAIN smoke moves; zero speed would expose an API/logging error.
                assert float(snapshot.episode_rows[0]["max_speed_kmh"]) > 0
                summary["smoke"]["results_browser_integrity"] = snapshot.integrity_status.value
                summary["smoke"]["episode"] = snapshot.episode_rows[0]
                print("Sandbox episode:", report.run_id, "COMPLETE / VERIFIED")
            summary["status"] = "PASS"
            return 0
    except Exception as error:
        summary["status"] = "FAIL"
        summary["error"] = str(error)
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    finally:
        (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Evidence:", output / "summary.json")
        os.chdir(original_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
