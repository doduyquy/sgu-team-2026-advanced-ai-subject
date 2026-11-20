# S5.5 Results — Three Fresh SAC Reward Arms

**Run:** `s5_5_meter_reward_seed101_20261120`

**Status:** all three 30k training runs and the fixed TRAIN-only evaluation completed

**Decision:** no durable driving improvement; root cause remains unresolved

**Eligibility:** `benchmark_eligible=false`

## Training execution

| Arm | Reward | Env steps | Optimizer updates | Completed episodes* | Geometries visited | Wall time | Throughput | Final sampled RSS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | V1 | 30,000 | 29,000 | 24 | 8/8 | 707.89 s | 42.38 steps/s | 489.0 MB |
| B | Scale | 30,000 | 29,000 | 23 | 8/8 | 1,973.40 s | 15.20 steps/s | 448.9 MB |
| C | Metre | 30,000 | 29,000 | 24 | 8/8 | 1,182.11 s | 25.38 steps/s | 428.0 MB |

*The training runner retained completed episode counts and geometry exposure but overwrote its detailed per-episode outcome list with a compact summary. The corrected analysis marks those outcomes `NOT_PERSISTED_BY_RUNNER`; it does not infer that no outcomes occurred. Evaluation outcome records are independently preserved. Geometry IDs, per-geometry steps, per-geometry episode completions, runtime, losses, alpha, RAM samples, and checkpoint hashes are in `analysis_summary.json`.

Environment: Python 3.10.21, MetaDrive 0.4.3, Stable-Baselines3 2.9.2, PyTorch 2.14.1+cpu, Gymnasium 1.3.0, CPU. All arms used the same fresh initialization fingerprint. All 15 checkpoints passed integrity verification.

## Fixed 300-decision probe results

Each cell reports mean speed / mean route-progress delta / stationary fraction / terminal safety failures / censored cases. Progress is relative to progress at reset. Any probe without a terminal event is `PROBE_CENSORED`; censoring is not a timeout or evidence of safety.

| Steps | Arm A — V1 | Arm B — Scale | Arm C — Metre |
|---:|---|---|---|
| 0 | 0.015 km/h / +0.000026 / 1.000 / 0 / 6 | same | same |
| 2k | 1.468 / +0.029636 / 0.337 / 3 / 3 | 1.731 / +0.035050 / 0.288 / 1 / 5 | 1.720 / +0.031434 / 0.292 / 3 / 3 |
| 5k | 0.300 / +0.006098 / 1.000 / 0 / 6 | 0.014 / +0.000031 / 1.000 / 0 / 6 | 1.327 / +0.026961 / 0.377 / 0 / 6 |
| 10k | 0.497 / +0.010100 / 0.991 / 0 / 6 | 0.016 / +0.000022 / 1.000 / 0 / 6 | 4.753 / +0.091069 / 0.109 / 3 / 3 |
| 30k | 0.024 / -0.000099 / 1.000 / 0 / 6 | 0.022 / -0.000115 / 1.000 / 0 / 6 | 0.024 / -0.000103 / 1.000 / 0 / 6 |

Probe safety counts only describe terminal events observed within 300 decisions. Censored cases are excluded from outcome classification.

## Full episodes on the same six diagnostic cases

| Checkpoint | Arm | Clean success | Safety failure | Timeout | Mean route-progress delta | Mean speed | Stationary fraction |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10k | A — V1 | 0/6 | 4/6 | 2/6 | +0.109985 | 1.587 km/h | 0.338 |
| 10k | B — Scale | 0/6 | 0/6 | 6/6 | +0.000082 | 0.014 km/h | 1.000 |
| 10k | C — Metre | 0/6 | 6/6 | 0/6 | +0.113794 | 5.243 km/h | 0.102 |
| 30k | A — V1 | 0/6 | 0/6 | 6/6 | +0.000062 | 0.016 km/h | 1.000 |
| 30k | B — Scale | 0/6 | 0/6 | 6/6 | -0.000110 | 0.015 km/h | 1.000 |
| 30k | C — Metre | 0/6 | 0/6 | 6/6 | +0.000004 | 0.016 km/h | 1.000 |

At 10k, C produced substantially more movement, but all six full episodes ended in safety failure. A also had four safety failures. B mostly remained stationary and timed out. At 30k, all arms were stationary for the full episodes and timed out on all six cases. None of the 36 full diagnostic episodes was a clean success.

## Paired comparisons

All arms shared the same six geometry/environment-seed cases, so paired differences are valid for those cases. At 10k full episodes, mean `C - A` route-progress difference was `+0.003809`, with mean speed `+3.656 km/h`, stationary fraction `-0.2355`, and safety-failure rate difference `+0.3333`. This is faster, riskier movement, not a driving win. `B - A` route-progress was `-0.109903`, speed `-1.573 km/h`, stationary fraction `+0.6622`, safety-failure rate `-0.6667`, and timeout rate `+0.6667`: the safety count was lower because B mostly did not move.

At 30k full episodes, paired route progress was near zero for all arms. Mean differences were `B - A = -0.0001725` and `C - A = -0.0000581`. Every arm had six timeouts and 100% stationary fraction. A single training root seed and six evaluation cases do not support a statistical superiority claim.

## Policy and training signals

The 10k-to-30k training window showed mean speeds of `0.328`, `0.113`, and `0.199 km/h` for A/B/C, with stationary fractions `0.930`, `0.989`, and `0.949`. Deterministic throttle means were `-0.356`, `-0.700`, and `-0.636`. Cumulative throttle sign switches at 30k were 11,481 / 7,652 / 7,926; the longest moving run was only 16 decisions (1.6 seconds) for A and C, and 15 decisions for B. Alpha at 30k was approximately `0.000218`, `0.000658`, and `0.000890`. These are diagnostics of behavior and optimization, not success metrics.

The primary interpretation is `NO_IMPROVEMENT` in durable clean driving. C has a transient 10k movement signal, but its six safety failures and the later stationary collapse prevent calling it improved driving. The experiment does not isolate a root cause and does not establish that SAC cannot learn the task.

## Artifact references

- Registered protocol and geometry identity: `protocol.json`, `protocol.sha256`.
- Hash/size inventory of compact and ignored artifacts: `artifact_manifest.json`.
- Reward trace audit: `reward_preflight.json`.
- Integrity-checked checkpoint identities and paired summaries: `analysis_summary.json`.
- Per-episode evaluation: ignored `runs/stage5_meter_reward/s5_5_meter_reward_seed101_20261120/evaluation/per_episode.jsonl.gz`, SHA-256 `de7b0d741c9fd0680bb614f03ce9cbce5870818fed0795c65d2f4da4f3030467`.
- Observation anchor samples: ignored `runs/stage5_meter_reward/s5_5_meter_reward_seed101_20261120/evaluation/policy_response_eval_states.jsonl.gz`, SHA-256 `3ac58093aebb2ea33f1f03ff02669bbe05d0a882820413d293aa9508ae739e85`.
- An initial evaluator attempt failed on transform metadata before scoring cases; it was preserved with its log and output, then fixed and rerun. Only the successful second attempt contributes to the results above.
- Superseded generated summaries/preflight copies are retained, with hashes, under ignored `runs/stage5_meter_reward/s5_5_meter_reward_seed101_20261120/analysis_archive/`; earlier Stage 1 state samples and preflight trace copies remain under the same run's ignored `preflight/` folder.
