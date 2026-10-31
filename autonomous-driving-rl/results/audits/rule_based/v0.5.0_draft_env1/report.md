# Rule-based paired TRAIN development comparison

Baseline: `rule_based_v050_env1_targeted_20261030`; candidate: `rule_based_v050_env1_targeted_20261030`.

Matched cases: 4; clean success 3 → 4; crashes 1 → 0.

Both runs completed. Source/configuration hashes and every recorded trace row count were verified.

This compares the policy and extractor together. It is not a canonical benchmark or policy-only ablation.

| Case | Baseline | Candidate | Completion before → after |
|---|---|---|---|
| sandbox/Medium/SCXCCS/0/1 | SUCCESS | SUCCESS | 0.9934 → 0.9934 |
| sandbox/Hard/SCTXrCS/2/1 | SUCCESS | SUCCESS | 0.9930 → 0.9931 |
| sandbox/Hard/SCXOCS/0/1 | CRASH_VEHICLE | SUCCESS | 0.6733 → 0.9923 |
| sandbox/Hard/XTOCS/0/1 | SUCCESS | SUCCESS | 0.9870 → 0.9870 |

Unmatched baseline cases (excluded from paired score):


Lower completion remains visible in paired_cases.csv; faster failure is not an improvement.
