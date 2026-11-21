# Stage 5 P2-G — Action Consequence Results

**Decision: `INCONCLUSIVE` on the unique learning mechanism; action consequences and the local actor/critic mismatch were measured.** No SAC training was run. `benchmark_eligible=false` remains unchanged.

## Evidence and integrity

- Canonical protocol SHA-256 (sorted compact JSON): `52aeb7cf732a366fa240ea2038c2ed3cc7721b11fb3f541e4f22219d7adb8f8e`.
- Machine evidence: [`evidence.json`](../../results/stage5/p2g_action_consequence_20261121/evidence.json), SHA-256 `b497bba6c4415be8ed2ec59ddefb1a90eab03893e3bd8e18fdebfbc3c047fc66`.
- Compact analysis: [`analysis_summary.json`](../../results/stage5/p2g_action_consequence_20261121/analysis_summary.json), SHA-256 `8f7526fa536c7bee888281ff813a638eedd47953d33a9d467be0375d130481f5`.
- Hash inventory: [`CHECKSUMS.sha256`](../../results/stage5/p2g_action_consequence_20261121/CHECKSUMS.sha256).
- Runner: [`run_stage5_p2g_action_consequence.py`](../../scripts/run_stage5_p2g_action_consequence.py).
- Runtime: Python 3.10.21, MetaDrive 0.4.3, Gymnasium 1.3.0, SB3 2.9.2, NumPy 2.2.6, Torch 2.14.1+cpu. End-to-end audit runtime was about 54 seconds.
- Source episode replayed all 862 recorded decisions and matched the logged speed, signed route delta, and RewardSpecV1 reward within the preregistered tolerances.

The five selected states were physically reconstructed three times each. Maximum pairwise ranges were: position 0.000257 m, heading 0.000038 rad, speed 0.000873 km/h, route completion 0.00000064, and transformed observation 0.0000413. All passed the amended limits. The sample is one TRAIN geometry, one environment seed, and one stochastic training trajectory.

## Physical intervention findings

45 fixed-action rollouts were executed: nine actions at each of five matched states, each capped at 20 decisions. There were 9 `OUT_OF_ROAD` terminal outcomes and 36 `PROBE_CENSORED` runs. There were no clean successes or timeouts. Censored runs are not evidence of safety.

| Throttle/brake | Mean route progress over 20 decisions | Mean summed V1 reward | OUT_OF_ROAD |
|---:|---:|---:|---:|
| `-0.5` brake | 0.010 m | -0.0041 | 0/15 |
| `0` neutral | 0.093 m | -0.1368 | 2/15 |
| `+0.5` throttle | 2.056 m | -0.4647 | 7/15 |

These are marginal summaries across different steering values and importantly different starting states. They do not mean throttle is generally unsafe or brake is generally safe. The state-conditioned results show the interaction:

- Reset, first-moving, and progressed states: positive throttle usually produced several metres of route progress in two seconds. Some steering/throttle combinations also reduced the estimated lane clearance or reached `OUT_OF_ROAD`.
- Near the most curved selected state (valid Task waypoint heading magnitude 0.121 rad): every `+0.5` throttle combination ended `OUT_OF_ROAD` within 20 decisions; neutral or braking combinations mostly remained censored with negligible progress.
- At the smallest-clearance selected state (estimated reference-lane body clearance 0.320 m): 4/9 action combinations ended `OUT_OF_ROAD`; the other five were censored. This estimated clearance is descriptive and is not the MetaDrive terminal classifier.

RewardSpecV1 assigned small progress/time rewards during nonterminal movement and applied the registered `-1` safety terminal component when `OUT_OF_ROAD` occurred. Thus it distinguished observed terminal failures from nonterminal progress at this short horizon. It has no separate continuous reward for declining safety margin before a terminal event. No evidence here supports changing the canonical reward: the action grid is narrow, short, and one-seed/one-case.

## Actor/critic comparison

The frozen 10k checkpoints were integrity-verified with their recorded F09 producer and observation transform:

| Checkpoint | Model SHA-256 | Deterministic action SD across five states (steer / throttle) |
|---|---|---:|
| P2-D A, ReLU actor | `46874728028afeacc823392f934ffb95e4a6defb28efa1a765090d68ba728464` | `0` / `0` |
| P2-D B, LeakyReLU actor | `7e29ec40f2269141c4fb57347bc20b4cbea7376e18f776920ce8c0d2a91581db` | `1.14e-5` / `1.57e-5` |
| P2-F A, ReLU actor | `fcddba97aefbc462054189e8140febd6ec12403bbb8d84ca9bb5c91c665750ed` | `1.14e-5` / `1.57e-5` |
| P2-F B, LeakyReLU actor | `b3614d224de18f15e713dc4bf440d85145ff4af998dc862ee66acc133edc0f9b` | `5.56e-5` / `1.45e-5` |

Across the nine-action grid, average min-Q range per checkpoint/state was about `0.0026–0.0041`, while Q baselines were around `14.6–14.8`. Q gradients with respect to Box2 action had mean norm about `0.0023–0.0047`. The Q-preferred grid action was almost constant across all five states within each checkpoint, and differed by checkpoint. It was not a robust state-conditioned control ranking. Replay local action envelopes contained all nine grid actions in the sampled neighborhoods, but this is a weak marginal support check; P2-F B's nearest-state distances rose to 0.61 on the curve/edge states, so its critic estimates there are especially uncertain.

At the selected states, the entropy-objective gradient norm (`0.11–0.26`) exceeded the negative-Q gradient norm (`0.0014–0.0051`); cosine directions varied in sign by checkpoint and state. The actor outputs remained almost constant despite these gradients. This is a local snapshot, not proof that entropy caused collapse. Q ranking and short fixed-action consequences are not directly equal: SAC Q includes future stochastic policy behavior and entropy, whereas these rollouts hold an action for 20 decisions.

## Evidence classification

### CONFIRMED

- The historical action trace can reconstruct five physically distinct states reproducibly within stated tolerances.
- Box2 actions produce distinct real consequences: positive throttle creates movement/progress in low-risk states; the same throttle near the selected curve/low-clearance states often reaches `OUT_OF_ROAD`.
- RewardSpecV1 records the small progress/time signal and the terminal safety penalty as implemented; the 9 observed safety outcomes each receive the terminal penalty once.
- The four frozen actors are effectively observation-insensitive across these five states, and each critic has a nearly state-invariant preferred action within this small grid.

### SUPPORTED_HYPOTHESIS

- Safety-outcome support is sparse in historical training replay; greater coverage of authentic terminal consequences may help the critic distinguish moving actions from stationary actions. This remains a hypothesis, not a root cause.
- Weak/near-constant actor action and weak state-conditioned Q ranking may jointly prevent exploitation of context-dependent action consequences.

### UNKNOWN

- Whether these local rankings predict longer-horizon outcomes under the learned stochastic policy.
- Whether the observed action effects generalize to other geometry/environment seeds, curves, or traffic.
- Whether the critic is miscalibrated versus merely trained on sparse or mismatched long-horizon outcomes.
- Whether more terminal-outcome sampling would improve safe driving or instead increase conservative stationary behavior.
- Whether reward changes are warranted.

No finding establishes a unique root cause or SAC driving competence.
