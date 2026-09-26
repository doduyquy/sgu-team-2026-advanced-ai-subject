# MetaDrive Environment Setup & Troubleshooting Guide

This document details the configuration, editable installation, and package shadowing prevention guidelines for the MetaDrive simulator used in this project.

---

## 1. Environment Specifications

- **Simulator**: MetaDrive
- **Package Name**: `metadrive-simulator`
- **Package Version**: `0.4.3`
- **Pinned Git Commit**: `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`
- **Upstream Repository**: `https://github.com/metadriverse/metadrive.git`
- **Python Version**: `3.10` (tested with 3.10.16 on Windows 10/11)

---

## 2. Recommended Directory Layout

MetaDrive source code **MUST NOT** reside inside the course repository or inside `autonomous-driving-rl/`. It must be placed as an external sibling folder:

```text
D:\SGU\CNTT\TTNTNC\
├── sgu-team-2026-advanced-ai-subject\      # Course repository
│   └── autonomous-driving-rl\             # Subproject directory
└── metadrive-src\                          # External MetaDrive source
```

---

## 3. Python Package Shadowing Warning (Critical)

### The Problem

If MetaDrive is cloned into a folder named exactly `metadrive` inside your working directory or parent directory:
```text
D:\SGU\CNTT\TTNTNC\metadrive\
```
When running Python from `D:\SGU\CNTT\TTNTNC\`, Python's default import mechanism (`sys.path[0]`) will resolve:
```python
import metadrive
```
to the local directory `metadrive/` instead of the properly installed package or package root, resulting in:
```text
ImportError: cannot import name 'MetaDriveEnv' from 'metadrive' (unknown location)
```

### The Solution

1. **Name the external folder `metadrive-src`**, never plain `metadrive`.
2. **Keep `metadrive-src` strictly outside** the course repository.
3. **Install with pip editable install** (`pip install -e`).
4. **Never create a local directory or script named `metadrive`** in your current working directory or PYTHONPATH.
5. **No `sys.path` hacks**: Do NOT add hardcoded `sys.path.append(...)` lines to project scripts. Dependency resolution must flow naturally from the active Conda environment.
6. **Diagnostic check**: Always verify `metadrive.__file__` when diagnosing import anomalies.

---

## 4. Step-by-Step Installation (Windows PowerShell)

### Step 1: Clone MetaDrive into `metadrive-src`

```powershell
cd D:\SGU\CNTT\TTNTNC
git clone https://github.com/metadriverse/metadrive.git metadrive-src
cd metadrive-src
git checkout 85e5dadc6c7436d324348f6e3d8f8e680c06b4db
```

### Step 2: Create and Activate Conda Environment

```powershell
conda create -n metadrive python=3.10 -y
conda activate metadrive
```

### Step 3: Upgrade Build Tools and Install Editable Package

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e D:\SGU\CNTT\TTNTNC\metadrive-src
```

### Step 4: Verify Installation and Import Resolution

```powershell
cd D:\SGU\CNTT\TTNTNC\sgu-team-2026-advanced-ai-subject\autonomous-driving-rl
python -c "import metadrive; print('File:', metadrive.__file__); from metadrive import MetaDriveEnv; print('MetaDriveEnv import OK')"
```

**Expected output**:
```text
File: D:\SGU\CNTT\TTNTNC\metadrive-src\metadrive\__init__.py
MetaDriveEnv import OK
```
Notice that `File:` must resolve to `D:\SGU\CNTT\TTNTNC\metadrive-src\metadrive\__init__.py`, NOT to any directory inside the course repository.

---

## 5. Headless Verification

To verify that the simulation runs without GUI dependency issues:

```powershell
python inspect_metadrive.py
```

This instantiates a headless `MetaDriveEnv` with `"use_render": False`, prints observation and action spaces, and safely closes the environment.
