# Reproducing and checking Stage 5 results

## Runtime used for S5.3M

The measured run used the Windows `metadrive_sb3` environment:

- Python 3.10.21
- MetaDrive 0.4.3
- Stable-Baselines3 2.9.2
- PyTorch 2.14.1 CPU
- Gymnasium 1.3.0
- CPU device, no observation normalization

Use the environment and dependency pins documented by the repository. Seeds constrain the experiment, but neither MetaDrive nor CPU training is guaranteed to reproduce model weights bit-for-bit on a different runtime or machine.

## Run a fresh registered A/B experiment

From `autonomous-driving-rl/`, with a unique run ID:

```powershell
python scripts/run_stage5_tau_ab.py --run-id s5_3m_tau_ab_seed101_YYYYMMDD
```

The runner refuses to reuse an existing run/results directory. It writes a preregistered protocol and shared fresh initialization, then runs control A and treatment B sequentially in child processes. Each arm saves 0/5k/10k checkpoints, diagnostics, and per-step training telemetry. It then evaluates the fixed 8-case schedule and writes paired summaries, critic diagnostics, and an artifact manifest.

Outputs:

- `runs/stage5_tau_ab/<run-id>/`: ignored raw run output, checkpoint bundles, per-step telemetry and evaluator logs.
- `results/stage5/<run-id>/`: compact preregistration, record summaries, hashes and analysis suitable for review.

Do not delete or reuse an existing run ID. A failed technical attempt remains recorded separately from completed scientific results.

## Recheck a completed run

Focused checks:

```powershell
python -m unittest tests.test_stage5_tau_ab -v
python scripts/analyze_stage5_tau_ab.py --run-id <existing-run-id> --overwrite
```

`--overwrite` replaces only the derived compact anchor/Q/analysis/manifest JSON files. It does not modify training records or checkpoint bundles. Without that explicit flag, analysis refuses to replace existing derived files.

To verify a particular checkpoint bundle from Python:

```powershell
python -c "from pathlib import Path; from src.training.stage5.checkpoint import verify_checkpoint_integrity; from src.training.stage5.observation_encoder import Stage5ObservationEncoder; print(verify_checkpoint_integrity(Path('runs/stage5_tau_ab/<run-id>/arm_B/checkpoints/step_10000'), expected_encoder_fingerprint=Stage5ObservationEncoder(core_only=False).get_schema_fingerprint())['model_sha256'])"
```

Keep the verified bundle together with `metadata.json` and `CHECKSUMS.sha256`. Raw checkpoints and logs are not stored in Git; transfer them through an approved artifact location and verify the SHA-256 after transfer.

## Existing S5.3M run

The completed seed-101 run is `s5_3m_tau_ab_seed101_20261119_retry1`. Its protocol, full per-case analysis and artifact manifest are under `results/stage5/`; the 53.7 MB local raw run bundle remains ignored under `runs/stage5_tau_ab/`. The first launch attempt with the base run ID failed before creating a trainer because of a runner log-directory collision; it did not train or contribute any results.

The earlier S5.3L trace set is also local/ignored. Its 96 replicate summaries match all eight declared SHA-256 values, and the 753,867,555 raw transition-trace bytes match the manifest. Do not rerun S5.3L to reconstruct it.

## Test commands and current repository regression

Stage 5 focused regression:

```powershell
python -m unittest tests.test_stage5_training_foundation tests.test_stage5_sac tests.test_stage5_pilot tests.test_stage5_pilot_monitor tests.test_stage5_tau_ab tests.test_workbench_comparison -v
```

Workbench integration and stale tab-count regression:

```powershell
python -m unittest tests.test_benchmark_campaign.TestBenchmarkCampaign.test_guided_page_is_default_and_manual_form_remains_available tests.test_workbench tests.test_workbench_results tests.test_stage5_pilot_monitor tests.test_workbench_comparison -v
```

The repository documents a Windows full-suite Qt preload for import-order stability:

```powershell
python -X faulthandler -u -c "from PySide6.QtCore import QProcess; from PySide6.QtWidgets import QApplication; import unittest; suite=unittest.defaultTestLoader.discover('tests',pattern='test_*.py'); result=unittest.TextTestRunner(verbosity=1).run(suite); raise SystemExit(not result.wasSuccessful())"
```

The Prompt 2 full-suite run used `C:\Users\ADMIN\anaconda3\envs\metadrive_sb3\python.exe` with `KMP_DUPLICATE_LIB_OK=TRUE` and Qt preloaded. Result: **602 tests, 601 passed, 1 skipped, 0 failed/errors; exit code 0; 871.327 seconds**. A first attempt through the Anaconda base `python` was invalid because that environment lacks MetaDrive; it failed 44 modules at import time and was not counted as a test result.

The earlier 2026-11-19 baseline full run found a stale assertion expecting 6 tabs; Prompt 2 updates it to assert the existing 7 tabs and SAC tab label. The Stage 5 monitor regression covers pending refresh shutdown, repeated open/close, selected runs/checkpoints, and process completion races. The dedicated Stage 5/Workbench set ran **157 tests, all passed, exit code 0**.
