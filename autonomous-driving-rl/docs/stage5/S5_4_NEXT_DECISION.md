# S5.4 Next Decision

## Decision: `NO_GO_MULTI_MAP`

Do not start multi-map training from either S5.4 candidate.

The gate required repeatable clean driving on distinct unseen internal TRAIN geometries, sustained movement, observation-dependent actions and a non-failure-dominated safety profile. At 100k, both A and B had 0/6 clean successes, 6/6 full-episode timeouts and stationary fraction 1.0. The B−A route-progress difference was only `+0.0001009` on six paired cases (about 0.0101 percentage points), with no change in success, safety or stationary behavior. A 10k B probe showed temporary progress that did not persist.

This is a no-go for **these candidates and this protocol**, not a theorem that SAC cannot learn driving. One training seed and six diagnostic geometries do not support a general algorithm-level claim.

## One next task

Before spending compute on more geometries, run one pre-registered, two-seed **actor responsiveness recovery** experiment that changes the actor's hidden-layer activation to a non-dead-activation choice (for example, Tanh), while keeping the same fixed eight-map Easy training pool, diagnostic schedule, reward, SAC settings and 30k screening budget. Retain an unchanged-ReLU control. The specific hypothesis is that the observed collapse of the first ReLU layer (98.8–100% nonpositive at the measured checkpoints) prevents the actor from retaining observation-dependent actions.

The experiment should stop at 30k if both seeds show a repeated early clean success and sustained movement on at least two fixed diagnostic geometries; otherwise report a negative/inconclusive result and do not escalate to multi-map training. This is a new, separately approved experiment; it is not part of S5.4 and no training is started by this decision document.

## Readiness state

- A is the unchanged SAC reference candidate; B is an exploratory observation-transform candidate. Neither is suitable for multi-map continuation from this evidence.
- Both remain development-only; `stage5_sac` stays `CANDIDATE` and `benchmark_eligible=false`.
- No VALIDATION or TEST data were used. Official eligibility, checkpoint binding and any team sign-off remain outside this experiment.
- The raw checkpoints and episode logs are local under the ignored `runs/stage5_learnability/s5_4_learnability_seed101_20261119/`; committed summary artifacts alone are insufficient to run the agents.

See [S5.4 results](S5_4_RESULTS.md) for per-checkpoint metrics, paired cases, hashes and plots.
