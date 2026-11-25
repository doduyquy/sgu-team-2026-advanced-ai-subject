# Rule-based paired TRAIN development comparison

Baseline: `rule_based_v050_trial01_20261030`; candidate: `rule_based_v050_trial05_geometry_20261030`.

Matched cases: 4; clean success 2 → 3; crashes 2 → 1.

Both runs completed. Source/configuration hashes and every recorded trace row count were verified.

This compares the policy and extractor together. It is not a canonical benchmark or policy-only ablation.

| Case | Baseline | Candidate | Completion before → after |
|---|---|---|---|
| sandbox/Medium/SCXCCS/0/0 | SUCCESS | SUCCESS | 0.9936 → 0.9933 |
| sandbox/Hard/SCTXrCS/2/0 | SUCCESS | SUCCESS | 0.9933 → 0.9933 |
| sandbox/Hard/SCXOCS/0/0 | CRASH_VEHICLE | SUCCESS | 0.5689 → 0.9920 |
| sandbox/Hard/XTOCS/0/0 | CRASH_VEHICLE | CRASH_VEHICLE | 0.1429 → 0.4632 |

Unmatched baseline cases (excluded from paired score):


Lower completion remains visible in paired_cases.csv; faster failure is not an improvement.
