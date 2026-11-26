# Stage 5 post-window delivery

The 12-hour Kaggle window closed at 01:52:37 ICT on 27 November 2026. Versioned API checks found 13 completed logical workloads and three terminal F09 failures, S04, S08 and S12. Earlier failed attempts with recoverable artifacts remain in the report. The 24 submitted attempts account for 173.04 hours to the last available log event. This observation measures notebook elapsed time separately from learner time and Kaggle quota billing.

## Model selection

S01 recovery, job `s5-9d68b8f2b7bb-j10-a1` version 1, is the leading deployment candidate among the evaluated checkpoints. Its model SHA-256 is `69aded291280cf3ce0cc0a69c2f343b5f5c3dcc97611d8e7e2e4aefbd934775a`. The verified full-state SHA-256 is `5fb35ed4a744d45000e2d3ce4eb120bc88f90032898bed5ca35f3787566e07fd`.

| Candidate | Easy clean, original | Medium clean, original | Medium clean, seeds 33701/33702 | Safety failures, Medium confirmation |
| --- | ---: | ---: | ---: | ---: |
| J11?5.000 | 12/12 | 0/12 | ? | ? |
| S01 recovery | 12/12 | 11/12 | 10/12 | 2 |
| S13 recovery | 12/12 | 11/12 | 7/12 | 5 |
| S09 recovery | 12/12 | 7/12 | 9/12 | 3 |
| S15 recovery | 12/12 | 9/12 | 8/12 | 4 |
| S11 | 12/12 | 10/12 | 0/12 | 0 |

S11 timed out on all 12 confirmation cases. The confirmation panel uses the development diagnostic geometries with different environment seeds. Both selection panels measure development performance. Formal VALIDATION and TEST evaluation requires the governed Stage 5 checkpoint binding and protocol approval.

The S13 policy-only checkpoint at 2.700M cumulative transitions passed 6/6 on the original one-seed Medium screen and 8/12 on the two confirmation seeds. It remains an inference challenger, with full-state continuation tied to a separately verified state.

| S01 confirmation tier | Clean | Safety failures | Timeouts | Mean route completion | Mean speed | Mean clean success time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Easy | 12/12 | 0 | 0 | 98.9% | 45.15 km/h | 32.62 s |
| Medium | 10/12 | 2 | 0 | 88.4% | 30.66 km/h | 61.35 s |
| Hard | 2/12 | 10 | 0 | 61.4% | 23.97 km/h | 134.90 s |
| Extreme | 0/12 | 10 | 2 | 19.4% | 19.36 km/h | ? |

These 48 episodes used deterministic SAC inference. The model retains Easy performance, shows a substantial Medium gain over J11 and has limited Hard and Extreme capability.

## Replay accounting

J11 and MT30 share six guided chunks containing 5,300 rows. The report deduplicates these identities by ledger key and SHA-256. Bounded replay retains at most 122,880 live rows per learner, including the protected guided prefix. Complete-state rings retain the data needed to resume the selected learner's optimizer and policy. The archive has no newly eligible trajectory chunks with row-level TRAIN provenance, so the rings were retained for resume and excluded from cross-learner replay consolidation.

`cloud.kaggle.consolidation` reports environment decisions, valid replay admissions, gradient updates and localization discards separately. Failed-attempt counters are lower bounds when the available evidence is a log marker or periodic checkpoint. Episode counts from final metrics summarize the retained telemetry sample. The 16-workload table and detailed attempt record are generated as `workloads.csv` and `campaign.json` under the private campaign directory.

## Reproduce the S01 sandbox demo

Use the verified full state in `runs/kaggle/harvest/s5-9d68b8f2b7bb-j10-a1` or the prepared inference bundle in `runs/kaggle/delivery/s01-window-final`. In the project MetaDrive/SB3 environment, the following commands reproduce the demo.

```powershell
python -B -m cloud.kaggle.delivery --state runs/kaggle/harvest/s5-9d68b8f2b7bb-j10-a1 --sha 5fb35ed4a744d45000e2d3ce4eb120bc88f90032898bed5ca35f3787566e07fd --out runs/kaggle/delivery/s01-window-final
python -B -m src.launcher.cli run --mode SANDBOX --agent stage5_sac_terminal_preview --checkpoint runs/kaggle/delivery/s01-window-final --task-context-producer reference_route_station_proposal_v1 --tier Medium --sequence SCTCS --geom-seed 11 --env-seed 33701 --render OFF --wandb DISABLED --runs-root runs/kaggle/delivery/s01-demo
```

The launcher completed the recorded Medium demo cleanly with 99.1% route progress in 68.8 simulator seconds. The inference bundle documents the 409D observation contract, Box2 actions, adapter fingerprints and training lineage. [The original J11 fallback and its demo](STAGE5_DELIVERY.md) remain available.

## Hard continuation and Extreme admission

The one-hour Hard continuation `s5-584e7f67a0fe-j10-a1` version 1 starts from the verified S01 full state. S01 and J11 remain frozen. The confirmation compares the final state and available shortlisted milestones against the exact same 12 cases per tier. Technical localization failures are reported separately.

For this continuation, an Extreme branch is supported when a verified resumable Hard checkpoint gains at least two clean Hard completions and reduces Hard safety failures by at least two. Easy and Medium may each lose at most one clean completion and gain at most one safety failure. The three candidate panels must also complete without a technical failure. A matching policy-only checkpoint remains eligible for inference analysis, with resume assessed separately.

## Checkpoint binding review

The following command checks the delivered model against the frozen model lock and prepares a content-based review record. The record carries model and training-state hashes, encoder and adapter identities, simulator and dependency versions, TRAIN split hash and training-recipe hash.

```powershell
python -B -m cloud.kaggle.binding_review --model-lock runs/kaggle/central12-recovery/post-campaign/model-lock.json --bundle runs/kaggle/delivery/s01-window-final --out runs/kaggle/central12-recovery/post-campaign/s01-binding-review.json
```

The record remains pending governance review. The [checkpoint binding proposal](STAGE5_CHECKPOINT_BINDING_PROPOSAL.md) describes the separate versioned launcher contract and approval conditions. The launcher stays in SANDBOX for S01 while those conditions are pending.
