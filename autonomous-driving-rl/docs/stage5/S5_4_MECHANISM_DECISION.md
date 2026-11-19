# S5.4 mechanism decision

**Decision frozen before fresh A/B training:** test one exploratory, versioned group-wise physical-unit transform on Treatment B. Control A keeps the current 409D policy input unchanged. No reward, entropy, gamma, tau, activation, architecture, sampler strategy, or canonical encoder change is included.

## Bounded evidence reviewed

The read-only analysis used the six fresh S5.3M checkpoint bundles (A/B at 0, 5k, 10k) and the eight saved, real S5.3M reset observations. It did not start MetaDrive, train a model, read a replay buffer, or access VALIDATION/TEST. Machine-readable measurements and source hashes are in [`mechanism_analysis.json`](../../results/stage5/s5_4_learnability_seed101_20261119/mechanism_analysis.json).

| Measurement on the same eight reset observations | 0 steps, A/B | A at 5k | B at 5k | A at 10k | B at 10k |
|---|---:|---:|---:|---:|---:|
| Deterministic steering SD across observations | 0.02857 | 0.00030 | 0.00373 | 0.000205 | 0.000011 |
| Deterministic throttle SD across observations | 0.04569 | 0.00323 | 0.00117 | 0.000117 | 0.000071 |
| First actor layer preactivation ≤ 0 | 48.73% | 95.51% | 94.38% | 99.61% | 99.61% |
| Mean first-layer post-activation SD | 2.263 | 0.350 | 0.339 | 0.061 | 0.092 |
| SAC entropy coefficient | 1.000 | 0.301 | 0.301 | 0.06713 | 0.06713 |

The saved S5.3M fast probes at 10k were stationary on all eight cases for both arms. They had zero clean successes and were `PROBE_CENSORED`; they were not full episodes. At 5k, both arms moved more but each had two safety failures in the eight short probes. These are historical development diagnostics, not canonical evaluation.

## Feature scale observations

Across the eight saved reset observations, normalized Core 259D values stayed in [0, 1]. Per-feature absolute maxima averaged 0.957 for Core versus 11.085 for the 60 waypoint coordinates/headings; waypoint values ranged from about −0.084 to 50.0. The average per-feature cross-anchor standard deviation was 0.00114 for Core and 0.507 for waypoints. The 56 traffic actor fields and their masks/metrics were all zero in these reset anchors, so this sample says nothing about traffic-state scaling.

The input-scale contrast and the near-constant actor outputs make a feature-scale interaction testable. The rise in nonpositive first-layer preactivations is consistent with activation inactivity, but does not establish that scale caused the policy behavior. Eight reset observations are not a representative replay or moving-state distribution.

Prior S5.3L reported mean min(Q1,Q2)=15.232 and mean soft-return estimate=9.097, a +6.135 gap over 96 TRAIN-only stochastic rollouts. This is evidence of local absolute Q overestimation on tested states/actions. It did not establish that critic bias caused stationary behavior or that the critic ranked throttle actions incorrectly. Reward accounting in the inspected pilot remained consistent with RewardSpecV1.

## Frozen intervention

Treatment B applies `pilot_groupwise_physical_v1` after the canonical encoder and before the policy, in both training and inference:

- Core `[0:259]`: unchanged.
- Traffic actors `[259:315]`, reshaped 8×7: x/y ÷ 50 m; vx/vy ÷ 30 m/s; heading ÷ π radians; length/width ÷ 5 m.
- Traffic masks and count metrics `[315:325]`: unchanged.
- Waypoints `[325:385]`, reshaped 20×3: x/y ÷ 50 m; heading ÷ π radians.
- Waypoint masks and task scalars `[385:409]`: unchanged.
- No clipping, fitted mean/variance, reward/outcome fields, or holdout-derived statistics.

The 50 m factors use the declared context/lookahead scale. 30 m/s and 5 m are fixed engineering reference scales, not bounds. Transform definition and SHA-256 are carried in checkpoint metadata and the protocol. The canonical 409D encoder fingerprint itself is unchanged.

## What remains uncertain

This A/B can test whether fixed unit scaling helps under the chosen short, eight-map Easy learnability protocol. One training seed and an internal diagnostic set cannot establish generalization, prove the cause of historical policy collapse, or certify benchmark performance. The treatment is exploratory and was chosen from the pre-training evidence above, not from S5.4 outcomes.
