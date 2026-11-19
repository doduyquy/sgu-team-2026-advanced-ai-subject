# Stage 5 live demo guide (5–10 minutes)

This walkthrough uses only existing local run records and frozen checkpoints. It does not train a model, alter a run, or produce an official benchmark result.

## Before class

1. Activate the project's Windows environment:

   ```powershell
   conda activate metadrive_sb3
   cd D:\SGU\CNTT\TTNTNC\sgu-team-2026-advanced-ai-subject\autonomous-driving-rl
   python -c "import metadrive, stable_baselines3, torch; print(metadrive.__version__, stable_baselines3.__version__, torch.__version__)"
   ```

   Expected in the reviewed environment: MetaDrive 0.4.3, Stable-Baselines3 2.9.2, Python 3.10.21. A machine without the matching environment can read the committed reports but cannot load the local model bundle.

2. Confirm the local artifacts exist:

   - Monitor run: `runs/stage5_pilot/s5_3d_tierrr_gamma099_seed101_20261115/`
   - A/B raw checkpoint bundles: `runs/stage5_tau_ab/s5_3m_tau_ab_seed101_20261119_retry1/`
   - Compact paired result: `results/stage5/s5_3m_tau_ab_seed101_20261119_retry1/`
   - Optional independent backup: `D:\SGU\Backups\XeTuHanh_Stage5_20261119\`

3. Start the Workbench:

   ```powershell
   python -m src.workbench.app
   ```

   The application was opened and closed successfully on the project Windows desktop. If the Intel OpenMP duplicate-runtime error appears on another workstation, do not treat `KMP_DUPLICATE_LIB_OK=TRUE` as a scientific or durable fix. Use the pinned project environment and resolve the duplicate runtime installation there; that environment workaround was used only for the recorded full test run.

## Live walkthrough

### Minute 0–1: frame the question

Say: “Stage 5 asks whether an agent can learn continuous steering and longitudinal control from interaction and reward. A trained network is not automatically a good driver; we evaluate behavior, not just whether its weights changed.”

Point out that this is MetaDrive and that the Workbench is a read-only presentation/monitoring layer over the experiment artifacts.

### Minute 1–3: open the training monitor

1. Click **7. SAC Training**.
2. In **Run SAC**, choose `s5_3d_tierrr_gamma099_seed101_20261115` (status `COMPLETE`). This is an existing 10,000-step development run, not the final S5.3M A/B arm.
3. Explain the progress cards: steps, completed episodes, geometries encountered, throughput, action distribution, route progress, actor/critic loss and RAM.
4. Point to the lower traces: steering/throttle from training records, actual vehicle speed, and route progress. These are telemetry, not benchmark scores.
5. Read the checkpoint table for 0/5k/10k. Set the checkpoint selector to **10000** and show that the model bundle exists.
6. Note that the selected run has zero clean successes and that the monitor's status is historical. Do not present the loss curve as proof of driving skill.

If the run list is empty, click **Làm mới**. The monitor discovers runs under `runs/stage5_pilot/`; the S5.3M arms live under `runs/stage5_tau_ab/` and are shown in the Results comparison, not in this monitor selector.

### Minute 3–5: compare the actual S5.3M A/B result

1. Click **6. Results**, then the right-hand **Comparison** subtab.
2. Refresh sources if needed. Select **SAC A · 10,000 steps** and **SAC B · 10,000 steps** from the S5.3M result `s5_3m_tau_ab_seed101_20261119_retry1`.
3. Read the status banner aloud: **TRAIN ONLY — NOT BENCHMARK** and `PAIRED COMPARISON`.
4. Show the SHA-256 checkpoint provenance and that both bundles are `VERIFIED` locally.
5. Show the eight per-case rows, per-tier summary and B−A route-progress chart. Point out `PROBE_CENSORED` in the outcome column; those capped probes are not successes or official timeouts.
6. Explain the key result: B's mean paired route-progress delta is +0.002675, but both policies were stationary for 100% of diagnostic steps and had zero clean successes. This does not make B a winner.
7. Optional: click **Export comparison JSON…**, choose a presentation folder, and retain the generated file with the source result and checkpoint SHA values. The export explicitly says `official_leaderboard=false`.

### Minute 5–8: optional simulator viewer

1. Return to **7. SAC Training** and select the COMPLETE S5.3D run at **10000** steps.
2. Click **Mở simulator** (viewer button).
3. The viewer starts a separate MetaDrive process with the frozen deterministic policy and a case from the internal `PILOT_DIAGNOSTIC` TRAIN schedule. It does not update SAC weights or replay data.
4. Narrate the behavior honestly: “This is a diagnostic visualization of the learned policy; it may move very little. We are observing the model, not demonstrating a successful driver.”
5. Close the simulator window or stop the preview. The recorded handoff smoke confirmed `Render Mode: onscreen` and cleanly stopped the viewer process. A preview stopped mid-episode has no outcome classification and is not an experiment result.

If native rendering fails, skip the viewer. Continue with the comparison tables and the static paired chart; do not change the outcome label to timeout.

### Minute 8–10: conclude

Say: “The pipeline, checkpoints, diagnostics and comparison are implemented. The scientific result is still negative on clean driving: after 10k steps the policy does not drive usefully. We have a small positive route-progress difference between tau settings, but it is not enough to select a winner or claim generalization. Stage 5 remains ineligible for canonical evaluation.”

## Recovery notes

- **Workbench does not open:** check the active Python is `metadrive_sb3`, confirm PySide6/Qt and project dependencies, then run `python -c "from src.workbench.main_window import MainWindow; print('import ok')"`. Do not launch a training command as a substitute.
- **Checkpoint missing:** the Git clone contains compact reports, not the ignored model bundles. Restore them from the verified backup or ask for the approved team artifact copy; verify `CHECKSUMS.sha256` before opening.
- **Comparison has no sources:** ensure the working directory is `autonomous-driving-rl/`, compact S5.3M files are present, and click **Refresh sources**. A/B source validation failure must be reported, not bypassed.
- **Viewer hangs or must be stopped:** close its native window. The GUI owns it as a separate process; use the application's existing close/termination confirmation. Do not relabel an interrupted preview as a completed episode.
- **No GUI/display:** run `python -m unittest tests.test_stage5_pilot_monitor tests.test_workbench_comparison -v` in the project environment and say explicitly that the demonstration was offscreen, not visually inspected.

