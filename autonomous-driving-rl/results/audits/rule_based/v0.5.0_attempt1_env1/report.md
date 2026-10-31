# Rule-based paired TRAIN development comparison

Baseline: `rule_based_v050_env1_targeted_20261030`; candidate: `rule_based_v050_final_env1_20261030`.

Matched cases: 4; clean success 3 → 3; crashes 1 → 1.

Both runs completed. Source/configuration hashes and every recorded trace row count were verified.

This compares the policy and extractor together. It is not a canonical benchmark or policy-only ablation.

| Case | Baseline | Candidate | Completion before → after |
|---|---|---|---|
| sandbox/Medium/SCXCCS/0/1 | SUCCESS | SUCCESS | 0.9934 → 0.9934 |
| sandbox/Hard/SCTXrCS/2/1 | SUCCESS | SUCCESS | 0.9930 → 0.9929 |
| sandbox/Hard/SCXOCS/0/1 | CRASH_VEHICLE | CRASH_VEHICLE | 0.6733 → 0.6713 |
| sandbox/Hard/XTOCS/0/1 | SUCCESS | SUCCESS | 0.9870 → 0.9867 |

Unmatched baseline cases (excluded from paired score):


Lower completion remains visible in paired_cases.csv; faster failure is not an improvement.
