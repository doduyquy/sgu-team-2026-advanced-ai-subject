# Stage 5 Parallel SAC

## Central campaign queue

`cloud.kaggle.campaign` pins 16 session IDs with a target of two per identity. The coordinator writes `campaign.json` in a private shared directory. Give each agent read access to the campaign and package, and write access only to `receipts/<account>`. Keep the account-specific DPAPI credential on the agent host. Keep the coordinator and execution logs outside the shared directory. Reported API status is the last observation from the corresponding identity, with its timestamp in `status.json`.

```powershell
# Coordinator on the existing operator's host, once per campaign
python -B -m cloud.kaggle.parallel export --out runs/kaggle/stage5-central01.zip
python -B -m cloud.kaggle.campaign init --root PATH_TO_PRIVATE_SHARED_QUEUE --package runs/kaggle/stage5-central01.zip

# Each authorized host runs its long-lived agent against the same queue and parcel
$env:STAGE5_KAGGLE_OPERATOR_SLOT = 'account_02'
python -B -m cloud.kaggle.campaign agent --root PATH_TO_PRIVATE_SHARED_QUEUE --package PATH_TO_PRIVATE_PACKAGE --watch

# Coordinator, without accessing credentials on the agent hosts
python -B -m cloud.kaggle.campaign report --root PATH_TO_PRIVATE_SHARED_QUEUE
```

Omit `--watch` for one reconciliation cycle. Watch mode exits once all actionable jobs for that identity are terminal. Agents fetch Kaggle status and harvest versioned artifacts, then refill their queue within verified capacity. Unknown capacity permits at most the existing server-admission path when the identity has no active receipt. The CLI distinguishes a local admission block, an API inventory observation, a capacity rejection, and an unresolved submit. A blocked Mixed session requires a verified multi-tier candidate before a new continuation can be selected. The output includes eligible continuation candidates and the verified parent digest. Artifact handoff to another identity uses the existing private transfer path.

The central package includes Kaggle-native collector and evaluation entry points. The older deployment bundle remains usable for the individual learner workloads below. Collector and evaluator notebooks are harvested by `campaign agent`. `status.json` contains receipt evidence and `status.csv` lists the 16 base sessions plus queued continuation rounds.

### Continuation rounds

Queue a new round from a harvested checkpoint on its owning host. Enqueue fingerprints the reviewed plan and uses the verified parity inputs from that parent. The agent reads the round from the same queue and handles submission, status and harvest within its existing capacity limits.

```powershell
python -B -m cloud.kaggle.campaign enqueue --root PATH_TO_PRIVATE_SHARED_QUEUE --round med-next --parent VERIFIED_PARENT_JOB --device cuda --transitions 25000 --max-seconds 3600
```

An Easy-to-Medium round can use the `--phase medium --traffic-guidance` recipe. It limits collection throttle from the existing public traffic fields and writes actual executed Box2 actions to replay. Report traffic-guided action counts alongside tier decisions. Standard candidate evaluation continues to use the learned SAC policy without collection guidance.

A verified traffic-guided Medium parent can use `--traffic-teacher` to mix in complete episodes from the frozen J11 policy with both traffic throttle limits. The agent assigns a subset of Medium episodes to the teacher before the midpoint, lets those episodes finish, then collects with the learner alone. The checkpoint records the teacher model SHA and separates Medium decisions and terminal outcomes by behavior. `--autonomous-traffic` explicitly retires the guidance in a subsequent continuation while preserving replay and optimizer state.

Each member runs one assigned slot on their own Windows session and Kaggle account. Use a private team channel for the deployment ZIP and its SHA-256. The ZIP contains the frozen J11–5.000 full state, runtime, source model for parity, and historical G2/G3 evidence. Each operator's Kaggle uploads are private and verified at version 1 by the launcher.

| Slot | Workload | Execution |
| --- | --- | --- |
| 01 | `medium` | CPU, two environments, 10k new decisions, Easy 35% and Medium 65% episode scheduling |
| 01 | `easy-t4` | T4, two environments, 25k new decisions, UTD 0.5 and quarter actor learning rate |
| 02 | `main` | CPU, two environments, Guided SAC with UTD 0.5, 10k new decisions |
| 03 | `steering` | CPU, two environments, champion steering during collection, 5k new decisions |
| 04 | `throttle` | CPU, two environments, champion throttle/brake during collection, 5k new decisions |
| 05 | `retention` | CPU, two environments, Easy-only UTD 1.0 reference, 10k new decisions |
| 06 | `collect` | Local frozen champion collector, one complete Easy TRAIN episode per family and optional private dataset publish |
| 07 | `evaluate` | Local Easy and Medium diagnostic evaluation and paired champion comparison |
| 08 | `easy-stable` | CPU, two environments, Guided SAC with UTD 0.5 and quarter actor learning rate, 10k new decisions |

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

Use your assigned slot and workload. Slot 01 runs one of its workloads at a time while concurrency remains unknown. Omit `--server-admission` when a fresh owner observation is already registered. With unknown capacity the flag checks the operator's notebook inventory, then performs a single server-admitted submit. The launcher automatically rechecks private dataset readiness. A submission receipt prevents another push. Preserve receipts for `SUBMIT_UNKNOWN` or capacity rejection for reconciliation.

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
$champion = 'runs/kaggle/harvest/s5-e2938ff0e622-j11-a1/model.zip'
$championSha = 'f82ca94d27ebd9fc3468081e20631536c959990a708398f5474bebee986e6d07'
python -B -m cloud.kaggle.parallel evaluate --package $parcel --sha $digest --model $champion --model-sha $championSha --tier Easy --out runs/kaggle/eval-champion-easy.json
python -B -m cloud.kaggle.parallel evaluate --package $parcel --sha $digest --model $champion --model-sha $championSha --tier Medium --out runs/kaggle/eval-champion-medium.json
python -B -m cloud.kaggle.parallel evaluate --package $parcel --sha $digest --model PATH_TO_VERIFIED_MODEL_ZIP --model-sha VERIFIED_MODEL_SHA --tier Easy --out runs/kaggle/eval-candidate-easy.json
python -B -m cloud.kaggle.parallel evaluate --package $parcel --sha $digest --model PATH_TO_VERIFIED_MODEL_ZIP --model-sha VERIFIED_MODEL_SHA --tier Medium --out runs/kaggle/eval-candidate-medium.json
python -B -m cloud.kaggle.pilot --baseline runs/kaggle/eval-champion-easy.json --candidate runs/kaggle/eval-candidate-easy.json --reference-sha $championSha --out runs/kaggle/eval-candidate-paired.json
```

Use Easy for retention and compare Medium on the locked internal diagnostic panel. Keep confirmation seeds from the frozen champion out of the tuning loop. Collector chunks require provenance verification and a matched learner comparison before claiming a learning gain.
