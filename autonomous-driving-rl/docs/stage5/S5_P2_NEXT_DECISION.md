# S5 P2 — Next Decision

## Decision: `NO_IMPROVEMENT`

Do not extend either arm and do not open multi-map training from this result.

The registered treatment achieved its intended OU-noise temporal correlation (measured lag-1 0.844 versus target 0.85), but its action sequences remained almost as short as control. Both arms reached similar movement at 2k, lost nearly all deterministic action sensitivity by 5k, and were stationary on all six full-episode diagnostic cases at 10k. Neither arm achieved a clean success; all six full episodes per arm ended as `TIMEOUT`.

The result does not isolate the cause. P2 does show that adding temporally correlated action noise in this form was insufficient to preserve the early movement signal. Because the policy became nearly constant while the recorded replay contained broad stochastic actions, the next investigation—if separately approved—should prioritize **representation/critic learning on real moving and geometry-diverse states**, while retaining replay support as an explicit limitation. The available one-seed run is not enough to choose between critic ranking, representation learning, reward/entropy interaction, or their combination.

Keep A as the reference control and preserve B as a development treatment record. Neither is a driving candidate. Do not change RewardSpecV1, declare a root cause, promote benchmark eligibility, or infer safety from the `TIMEOUT` outcome.

## Evidence boundary

This is a single seed, 10k steps per arm, eight Easy training geometries and six fixed internal TRAIN diagnostic geometries. The short probes are censored, not completed episodes. Official VALIDATION/TEST were not accessed. Candidate selection and benchmark certification remain out of scope.
