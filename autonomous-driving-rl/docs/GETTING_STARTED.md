# Getting Started on a New Machine

This guide is intended for a teammate who has just cloned the course repository and wants to run the autonomous-driving subproject without knowing the earlier implementation history.

## 1. Prerequisites

Recommended environment:

- Windows 10/11 (the current platform was developed and audited primarily on Windows).
- Git.
- Conda/Miniconda.
- Python 3.10.
- A GPU is not required for the current platform/fixture smoke tests. Future RL training requirements depend on the selected algorithm.

The project pins MetaDrive to:

- version: `0.4.3`
- commit: `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`

## 2. Clone the course repository

Use any workspace path. Example:

```powershell
mkdir D:\work\sgu-ai
cd D:\work\sgu-ai
git clone https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject.git
```

The project does not require a hard-coded drive or username.

## 3. Create the Python environment

```powershell
conda create -n metadrive python=3.10 -y
conda activate metadrive
cd D:\work\sgu-ai\sgu-team-2026-advanced-ai-subject
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

The repository-level `requirements.txt` contains the team dependencies such as NumPy, pandas, matplotlib, W&B, and PySide6.

## 4. Install the pinned MetaDrive source

Keep MetaDrive outside the course repository so its package directory cannot shadow the installed module.

```powershell
cd D:\work\sgu-ai
git clone https://github.com/metadriverse/metadrive.git metadrive-src
cd metadrive-src
git checkout 85e5dadc6c7436d324348f6e3d8f8e680c06b4db
python -m pip install -e .
```

Recommended layout:

```text
<workspace>/
├── sgu-team-2026-advanced-ai-subject/
│   └── autonomous-driving-rl/
└── metadrive-src/
```

Do not clone MetaDrive as a folder named `metadrive` inside the project working directory. See [`environment/METADRIVE_SETUP.md`](environment/METADRIVE_SETUP.md) for troubleshooting.

## 5. Verify the simulator import

```powershell
cd D:\work\sgu-ai\sgu-team-2026-advanced-ai-subject\autonomous-driving-rl
python -c "import metadrive; print(metadrive.__version__ if hasattr(metadrive,'__version__') else 'MetaDrive import OK'); print(metadrive.__file__)"
```

Then run a platform-only query:

```powershell
python -m src.launcher.cli agents
python -m src.launcher.cli cases --split TRAIN --tier Easy --limit 3
```

These commands should not require a simulation run.

## 6. Resolve a sandbox plan

```powershell
python -m src.launcher.cli plan --mode SANDBOX --agent fixture_seeded_random --tier Easy --agent-seed 101
```

Expected behavior:

- the request resolves to one TRAIN case;
- preflight shows the checks and a verdict;
- no simulator episode is executed by the `plan` command.

## 7. Run one development smoke episode

```powershell
python -m src.launcher.cli run --mode SANDBOX --agent fixture_seeded_random --tier Easy --agent-seed 101 --render NATIVE
```

This is a platform smoke run using an audit fixture. It is **not** a Stage 0 scientific baseline and must not be reported as a benchmark result.

For headless execution, replace `--render NATIVE` with `--render OFF`.

## 8. Launch Research Workbench

```powershell
python -m src.workbench.app
```

Use `SANDBOX` with a fixture agent for learning/debugging. Do not expect a fixture to run on VALIDATION or TEST: preflight intentionally blocks it because fixtures are `benchmark_eligible=False`.

See [`WORKBENCH_GUIDE.md`](WORKBENCH_GUIDE.md).

## 9. Run the regression tests

From `autonomous-driving-rl/`:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

The final Platform V1 hardening baseline had 351 tests discovered, with one Windows symlink test conditionally skipped on non-admin machines. Exact counts may grow after Stage 0 and later agents are added; the important condition is zero unexpected failures/errors.

## 10. What to read before coding an agent

Do not start by modifying the simulator. Read:

1. [`PLATFORM_V1_OVERVIEW.md`](PLATFORM_V1_OVERVIEW.md)
2. [`EXPERIMENT_WORKFLOW.md`](EXPERIMENT_WORKFLOW.md)
3. [`AGENT_DEVELOPMENT_GUIDE.md`](AGENT_DEVELOPMENT_GUIDE.md)
4. [`STAGE_ROADMAP.md`](STAGE_ROADMAP.md)

The current immediate development milestone is the real **Stage 0 Random / Naive baseline**.
