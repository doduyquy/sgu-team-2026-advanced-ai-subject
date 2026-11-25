# Stage 5 Parallel SAC

Each member runs one assigned slot on their own Windows session and Kaggle account. Use a private team channel for the deployment ZIP and its SHA-256. The ZIP contains the frozen J11–5.000 full state, runtime, source model for parity, and historical G2/G3 evidence. Each operator's Kaggle uploads are private and verified at version 1 by the launcher.

| Slot | Workload | Execution |
| --- | --- | --- |
| 01 | `medium` | CPU, two environments, 10k new decisions, Easy 35% and Medium 65% episode scheduling |
| 02 | `main` | CPU, two environments, Guided SAC with UTD 0.5, 10k new decisions |
| 03 | `steering` | CPU, two environments, champion steering during collection, 5k new decisions |
| 04 | `throttle` | CPU, two environments, champion throttle/brake during collection, 5k new decisions |
| 05 | `retention` | CPU, two environments, Easy-only UTD 1.0 reference, 10k new decisions |
| 06 | `collect` | Local frozen champion collector, one complete Easy TRAIN episode per family and optional private dataset publish |
| 07 | `evaluate` | Local Easy and Medium diagnostic evaluation and paired champion comparison |
| 08 | Reserve | Available for an independent replicate after the first group reports |

Retention branches record the executed anchored axis in replay. The evaluation runs the learned policy without the collection anchor. Medium starts from the verified full state and carries the old replay forward. The curriculum weights apply to episode selection, so the realized fraction of transitions is measured separately. Run Medium, Hard and Extreme in order after reviewing candidate retention on earlier tiers.

## Launch a learner

Run from `autonomous-driving-rl` on `feat/stage5-kaggle-parallel-training`. Install the official Kaggle CLI and SDK in the launcher environment. Each operator enters their own token in the hidden setup prompt once. `--concurrency` can be supplied during setup when the owner has observed it. Unspecified capacity stays unknown.

```powershell
$env:STAGE5_KAGGLE_OPERATOR_SLOT = 'account_02'
python -B -m cloud.kaggle.control setup --account account_02 --owner YOUR_KAGGLE_USERNAME
python -B -m cloud.kaggle.control doctor
$parcel = 'D:\private\stage5-parallel01.zip'
$digest = 'SHA256_RECEIVED_FROM_THE_PACKAGE_OWNER'
python -B -m cloud.kaggle.parallel launch --package $parcel --sha $digest --workload main --server-admission
```

Use your assigned slot and workload. Omit `--server-admission` when a fresh owner observation is already registered. With unknown capacity the flag checks the operator's notebook inventory, then performs a single server-admitted submit. If the launcher reports that a private input is still processing, run the same command to check its readiness. A submission receipt prevents another push. Preserve receipts for `SUBMIT_UNKNOWN` or capacity rejection for reconciliation.

```powershell
python -B -m cloud.kaggle.control status --job JOB_PRINTED_BY_LAUNCHER
python -B -m cloud.kaggle.control harvest --job JOB_PRINTED_BY_LAUNCHER
```

Return the job ID, version, actual API status and checkpoint SHA to the team. Each operator performs their own harvest. A recipient verifies hashes before using any artifact shared by another member.

## Collector and evaluation

Use the project's MetaDrive/SB3 environment for these local commands. The replay upload uses the configured operator's authenticated Kaggle CLI. The owner controls private access to the resulting dataset.

```powershell
$env:STAGE5_KAGGLE_OPERATOR_SLOT = 'account_06'
python -B -m cloud.kaggle.control setup --account account_06 --owner YOUR_KAGGLE_USERNAME
python -B -m cloud.kaggle.parallel collect --package $parcel --sha $digest --out runs/kaggle/parallel-collector01 --publish
```

```powershell
$env:STAGE5_KAGGLE_OPERATOR_SLOT = 'account_07'
python -B -m cloud.kaggle.parallel evaluate --package $parcel --sha $digest --model PATH_TO_VERIFIED_MODEL_ZIP --model-sha VERIFIED_MODEL_SHA --tier Easy --out runs/kaggle/eval-candidate-easy.json
python -B -m cloud.kaggle.parallel evaluate --package $parcel --sha $digest --model PATH_TO_VERIFIED_MODEL_ZIP --model-sha VERIFIED_MODEL_SHA --tier Medium --out runs/kaggle/eval-candidate-medium.json
python -B -m cloud.kaggle.pilot --baseline PATH_TO_FROZEN_EASY_PANEL --candidate runs/kaggle/eval-candidate-easy.json --reference-sha f82ca94d27ebd9fc3468081e20631536c959990a708398f5474bebee986e6d07 --out runs/kaggle/eval-candidate-paired.json
```

Use Easy for retention and compare Medium on the locked internal diagnostic panel. Keep confirmation seeds from the frozen champion out of the tuning loop. Collector chunks require provenance verification and a matched learner comparison before claiming a learning gain.
