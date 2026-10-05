# MetaDrive Setup and Troubleshooting

Platform V1 currently requires the exact MetaDrive baseline below:

- package/version: MetaDrive `0.4.3`
- pinned source commit: `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`
- upstream: https://github.com/metadriverse/metadrive
- Python: 3.10

## Recommended workspace layout

Keep the MetaDrive source outside the course repository:

```text
<workspace>/
├── sgu-team-2026-advanced-ai-subject/
│   └── autonomous-driving-rl/
└── metadrive-src/
```

This avoids accidental Python module shadowing and keeps the upstream simulator separate from team code.

## Windows / Conda installation

```powershell
# Create environment
conda create -n metadrive python=3.10 -y
conda activate metadrive

# Clone course repository
cd <workspace>
git clone https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject.git
cd sgu-team-2026-advanced-ai-subject
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# Clone exact MetaDrive source next to the course repository
cd <workspace>
git clone https://github.com/metadriverse/metadrive.git metadrive-src
cd metadrive-src
git checkout 85e5dadc6c7436d324348f6e3d8f8e680c06b4db
python -m pip install -e .
```

Replace `<workspace>` with your actual local workspace path. Do not copy another team member's drive path.

## Verify import resolution

From `autonomous-driving-rl/`:

```powershell
python -c "import metadrive; print('MetaDrive module:', metadrive.__file__); from metadrive import MetaDriveEnv; print('MetaDriveEnv import OK')"
```

The printed module path should resolve to the external `metadrive-src` checkout (or its editable-install mapping), not to a local folder under `autonomous-driving-rl`.

## Python shadowing warning

Do not create a local file/folder named `metadrive.py` or `metadrive/` in or above the project working directory. Python may import that object instead of the installed simulator.

Avoid hard-coded `sys.path.append(...)` fixes. Correct the environment/install instead.

## Platform smoke checks

```powershell
python -m src.launcher.cli agents
python -m src.launcher.cli cases --split TRAIN --limit 3
python -m src.launcher.cli plan --mode SANDBOX --agent fixture_seeded_random --tier Easy --agent-seed 101
```

Optional GUI:

```powershell
python -m src.workbench.app
```

## Headless simulator diagnostic

The legacy `inspect_metadrive.py` script remains available as a lightweight import/environment diagnostic:

```powershell
python inspect_metadrive.py
```

It is not a benchmark entrypoint.

## Common problems

### `ImportError` or `MetaDriveEnv` comes from an unexpected path

Check:

```powershell
python -c "import metadrive; print(metadrive.__file__)"
```

Remove/rename any local object shadowing the `metadrive` package and reinstall the pinned editable source if necessary.

### Workbench cannot import PySide6

From the repository root:

```powershell
python -m pip install -r requirements.txt
```

The project pins PySide6 in the repository requirements.

### W&B online mode fails preflight

Online W&B is optional. If explicitly using ONLINE mode, provide `WANDB_API_KEY` through the process environment. Never commit the key to the repository.

### VALIDATION/TEST is blocked for a fixture

Expected behavior. Current fixture agents are for infrastructure verification and are not benchmark-eligible. Use SANDBOX while onboarding; Stage 0 will introduce the first true research baseline.
