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

## Checkpoint navigation and storage

The focused Gate audit is in `runs/kaggle/postwindow-product/checkpoint-gate-audit.json`. It records the actual runtime option, F09 lookahead changes and the native navigation index transitions. The pinned native navigation and reward code are available in the corresponding MetaDrive source. [Navigation implementation](https://github.com/metadriverse/metadrive/blob/85e5dadc6c7436d324348f6e3d8f8e680c06b4db/metadrive/component/navigation_module/node_network_navigation.py) · [Reward and destination implementation](https://github.com/metadriverse/metadrive/blob/85e5dadc6c7436d324348f6e3d8f8e680c06b4db/metadrive/envs/metadrive_env.py)

The cleanup plan and NTFS allocation totals are in `runs/kaggle/postwindow-product/cleanup-plan.csv` and `cleanup-plan-summary.json`. Two complete local mirrors on C match their canonical copies on D, with 0.768 GiB of allocated space. A further 1.335 GiB of screening policy candidates is conditional on clearing the evaluation/source shortlist. Unique historical replay remains in ARCHIVE pending full-state resume verification and a separate verified archive. Apply any cleanup only after approval and a fresh reference check.
