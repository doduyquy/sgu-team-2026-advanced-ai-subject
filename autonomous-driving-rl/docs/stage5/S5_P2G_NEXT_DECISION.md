# Stage 5 P2-G — Next Decision

## Decision

`INCONCLUSIVE` on root cause. Keep RewardSpecV1 and the existing SAC/observation/action contracts unchanged. Do not call any P2-G rollout a successful driving episode.

## One justified follow-up experiment

If separately authorized, run one fresh, bounded A/B SAC test that changes only **online replay sampling toward true terminal transitions**, with standard importance-sampling correction:

- Control: current uniform replay sampling.
- Treatment: terminal-aware prioritized sampling of real online transitions, including any clean-success or safety-terminal transitions that actually occur; no synthetic or counterfactual transitions are inserted.
- Keep fresh initialization, F09 producer, 409D transform, Box2, RewardSpecV1, gamma/alpha tuning, architecture, TRAIN geometry/seed schedule and step budget paired.
- Register a small budget and stop rule before training. Evaluate full deterministic TRAIN diagnostic episodes for clean success, safety failures, route progress, sustained movement, and actor sensitivity. Movement or more failures alone is not improvement.

### Why this is the single next test

P2-G directly showed that the same throttle has different consequences by state and that actual `OUT_OF_ROAD` events carry the terminal `-1` reward. Historical P2-D replay had only two safety transitions in arm A and none in arm B; P2-F reported no clean successes in its bounded runs. This motivates testing whether better exposure to genuine terminal transitions, when they occur online, helps critic learning. It does not establish that replay imbalance is causal, so the control and driving-level outcomes are essential. This is not P2-E deterministic-episode mixing and is not started by P2-G.

## Stop boundary

No training, sweep, reward change, validation/test access, benchmark promotion, or PR merge was performed. `benchmark_eligible=false` remains in force.
