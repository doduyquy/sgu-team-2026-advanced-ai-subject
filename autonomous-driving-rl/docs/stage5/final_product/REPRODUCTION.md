# Stage 5 experiment handoff

The report-facing results and chart tables are in [BAO_CAO_STAGE5.md](BAO_CAO_STAGE5.md). This handoff references private local artifacts and includes the launcher command. Run the commands from the project root using the MetaDrive/SB3 environment.

## Frozen models

S01 is available at `runs/kaggle/harvest/s5-9d68b8f2b7bb-j10-a1`. Its model digest is `69aded291280cf3ce0cc0a69c2f343b5f5c3dcc97611d8e7e2e4aefbd934775a` and its full-state digest is `5fb35ed4a744d45000e2d3ce4eb120bc88f90032898bed5ca35f3787566e07fd`. The verified inference bundle is at `runs/kaggle/delivery/s01-window-final`. J11 remains the frozen fallback.

## Paired steering panel and demo

```powershell
python -B -m experiments.stage5_native_reward.steering_panel --checkpoint runs/kaggle/delivery/s01-window-final --cases docs/stage5/final_product/steering_cases.csv --out runs/kaggle/steering-reproduction --workers 4
python -B -m experiments.stage5_native_reward.steering_metrics --cases docs/stage5/final_product/steering_cases.csv --runs runs/kaggle/steering-reproduction --out docs/stage5/final_product
python -B -m src.launcher.cli run --mode SANDBOX --agent stage5_sac_terminal_preview --checkpoint runs/kaggle/delivery/s01-window-final --task-context-producer reference_route_station_proposal_v1 --tier Medium --sequence SCTCS --geom-seed 11 --env-seed 33701 --render NATIVE --wandb DISABLED --runs-root runs/kaggle/final-product-demo
```

For a labeled experimental demo, add `--steering-profile adaptive` or `--steering-profile fixed`. Use `--steering-profile off` to export baseline per-step telemetry. Omitting the flag runs the standard controller. The manifest records the explicit profile and `steering-*.csv` captures both the raw actor action and the simulator-facing action.

The original paired runs are in `runs/kaggle/final-product-steering/paired`. Full-run outcome metrics come from the common launcher's episode CSV. The kinematic and steering statistics use the shared prefix of the three profiles for each case.

The fixed EMA visual demo used `--sequence SCXCCS --geom-seed 17 --env-seed 33701 --steering-profile fixed`. Raw evidence is in `runs/kaggle/final-product-steering/native-demo-20261127-144937/medium-fixed-demo`.

## Consolidation pilot

The collection plan, verified replay chunks and admission accounting are in `runs/kaggle/first-window-product`. `consolidated-train.zip` is the private transfer archive. `cloud-pilot-ready` contains the pinned plan for a 128-update CPU pilot. The launcher requires an authorized operator and a free declared slot, verifies each private input, and records the submission outcome before allowing another action. At the time of this handoff there is no submission receipt or cloud optimizer update for this pilot.

```powershell
$plan = 'runs/kaggle/first-window-product/cloud-pilot-ready'
$pin = '31759502c6e4594f42b0260037e04f5d84af6b9dc61d81443ff44b7f07bc4fe6'
python -B -m cloud.kaggle.consolidation_job submit --folder $plan --plan-sha $pin
python -B -m cloud.kaggle.consolidation_job harvest --folder $plan --plan-sha $pin
```

The four planned inputs are the pinned runtime, parity model, S01 full state and verified TRAIN dataset. They use private version-one transfer payloads `runtime.bin`, `model.bin`, `parent.bin` and `dataset.bin`. The launcher resolves each input under `/kaggle/input/<dataset-slug>/` or `/kaggle/input/datasets/<owner>/<dataset-slug>/` after provisioning. The notebook installs pinned dependencies and MetaDrive assets with Internet access. Verified outputs use shards under `/kaggle/working/<job>/final/` and paired panels beside that directory. The archive and minibatch accounting remain separate from the protected S01 ring.

## Local consolidation

The local runs are under `runs/kaggle/first-window-local` on D. Use the
original `collection-0*.json` plans and the verified `CONSOLIDATED.json`
manifests to reproduce the added TRAIN replay. Each plan fixes the
geometry and environment seed, donor model digest, and collector stream.
The previously verified 5,586 rows are under
`runs/kaggle/first-window-product/recollected-train-01`.

The local experiment used CPU because the installed PyTorch build did
not support CUDA. The update process used two Torch threads. Collection
and evaluation ran headless with one Torch thread per process and
below-normal Windows priority.

Use a fresh output directory and the frozen S01 state copy for the pilot.
Pass each additional dataset with its recorded manifest digest as
`--extra-dataset PATH MANIFEST_SHA`. To continue from an offline milestone,
use its full state and pass the cumulative replay manifests required by
that milestone and any new batches for the next stage.

```powershell
$root = 'runs/kaggle/first-window-local'
$dataset = 'runs/kaggle/first-window-product/recollected-train-01'
$parentSha = '5fb35ed4a744d45000e2d3ce4eb120bc88f90032898bed5ca35f3787566e07fd'
$datasetSha = 'b2454b19afd436d6b1c1b1fabb7f84d2895d4734c0137e6d984aa83635b156cf'
python -B -m cloud.kaggle.consolidated_replay update --local --parent "$root/parent-copy" --parent-sha $parentSha --dataset $dataset --dataset-sha $datasetSha --out "$root/pilot-reproduction" --device cpu --torch-threads 2 --updates 128
$stateSha = (Get-FileHash "$root/pilot-reproduction/COMPLETE.json" -Algorithm SHA256).Hash.ToLowerInvariant()
python -B -m cloud.kaggle.local_consolidation --state "$root/pilot-reproduction" --state-sha $stateSha --out "$root/pilot-reproduction-evaluation"
```

The full-state directory preserves the optimizer and S01 replay ring.
Keep the pinned TRAIN datasets beside the state to reproduce mixed-replay
updates. `metrics.json` records the optimizer steps, replay draws and
dataset digests. The inference export is smaller and can be passed as
`--checkpoint` to the standard SANDBOX launcher. The full-state and
inference-only artifacts serve different entry points.

The four measured full states are `pilot-00128`, `stage-02000`,
`stage-05000` and `stage-10000`. Each has a matching evaluation directory
with the original 48 panel cases. `selection/SELECTION.json` binds the
inference export to its matching original full-state digest.

## Local hybrid consolidation V2

For inference from a new checkout or a Download ZIP, see the
[Stage 5 demo instructions](../../../README.md#stage-5-sac-demo).
The remaining commands document the original research workspace and its
private training artifacts.

The artifacts are under `runs/kaggle/first-window-v2`. The donor inventory,
locked case schedule and parity evidence are at the root. Each
`external-*` or `online-*` folder contains independent, checksummed
trajectory chunks and its `CONSOLIDATED.json`. The online manifests refer
to the exact frozen snapshot that generated the episode.

Run the first command with a new output directory. Pass the actual fixed
campaign deadline to the second. Each command uses four spawned simulation
workers with one Torch thread per worker.

```powershell
$root = 'runs/kaggle/first-window-v2-reproduction'
$archive = Join-Path $env:LOCALAPPDATA 'XeTuHanh/Stage5CampaignHarvest/1795719157/consolidation-ready.json'
$deadline = '2026-11-27T19:30:00+07:00' # Set the fixed deadline for the replication.
python -B -m cloud.kaggle.local_donors --out $root --archive $archive
python -B -m cloud.kaggle.local_campaign --root $root --deadline $deadline --target-rows 200000
```

For the delivered model, use `best-v2-extension-10000-bundle` with the
SANDBOX launcher. `extension-10000-result.json` selects `stage-05000`
after evaluating the later checkpoints. The V2 full state
contains its model, optimizers, RNG and original S01 replay. Its manifest
also pins the external and online dataset sidecars used at that checkpoint.
Keep those sidecars and snapshots at their recorded locations. V2 resume
reconstructs the online ring and validates the exact next minibatch hash.
The simulator resumes at an episode boundary. See `resume-proof.json` and
the `extension-*-resume-proof.json` receipts for the tested states.

```powershell
python -B -m src.launcher.cli run --mode SANDBOX --agent stage5_sac_terminal_preview --checkpoint runs/kaggle/first-window-v2/best-v2-extension-10000-bundle --task-context-producer reference_route_station_proposal_v1 --tier Medium --sequence SCTCS --geom-seed 11 --env-seed 33701 --render OFF --wandb DISABLED --runs-root runs/kaggle/demo-v2-reproduction
```

## Risk-aware SAC experiment

This local experiment uses the locked V2–5.000 parent and reads its
replay sidecars in their recorded locations. The experiment directory
is `runs/kaggle/first-window-risk`. Each collection wave stores the
policy fingerprint, TRAIN case, validated replay and an optional
step-by-step guidance trace. Online waves use the raw learned policy.
All training reports and comparisons are scoped to the development split.

Verify a saved experimental checkpoint without gradient updates. The
state pin comes from its recorded screening or resume receipt.

```powershell
$risk = 'runs/kaggle/first-window-risk'
$proof = Get-Content "$risk/resume-proof.json" -Raw | ConvertFrom-Json
python -B -m cloud.kaggle.local_risk --verify-state "$risk/$($proof.state)" --state-sha $proof.state_sha
```

Use the experimental `experiment-bundle` with the same SANDBOX launcher
and raw SAC policy as above. The retention experiment and its independently
verified state are under `retention`. `selection.json` records the overall
comparison. The locked production reference remains V2–5.000.

## Checkpoint navigation and storage

The focused Gate audit is in `runs/kaggle/postwindow-product/checkpoint-gate-audit.json`. It records the actual runtime option, F09 lookahead changes and the native navigation index transitions. The pinned native navigation and reward code are available in the corresponding MetaDrive source. [Navigation implementation](https://github.com/metadriverse/metadrive/blob/85e5dadc6c7436d324348f6e3d8f8e680c06b4db/metadrive/component/navigation_module/node_network_navigation.py) · [Reward and destination implementation](https://github.com/metadriverse/metadrive/blob/85e5dadc6c7436d324348f6e3d8f8e680c06b4db/metadrive/envs/metadrive_env.py)

The cleanup plan and NTFS allocation totals are in `runs/kaggle/postwindow-product/cleanup-plan.csv` and `cleanup-plan-summary.json`. Two complete local mirrors on C match their canonical copies on D, with 0.768 GiB of allocated space. A further 1.335 GiB of screening policy candidates is conditional on clearing the evaluation/source shortlist. Unique historical replay remains in ARCHIVE pending full-state resume verification and a separate verified archive. Apply any cleanup only after approval and a fresh reference check.
