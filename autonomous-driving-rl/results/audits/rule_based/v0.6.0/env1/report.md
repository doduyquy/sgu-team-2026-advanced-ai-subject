# Rule-based paired TRAIN development comparison

Baseline: `rule_based_v050_acceptance_env1_20261030/current`; candidate: `rule_based_v060_accept04_env1_20261030/current`.

Matched cases: 4; clean success 4 → 4; crashes 0 → 0.

Both runs completed. Source/configuration hashes and every recorded trace row count were verified.

This compares the policy and extractor together. It is not a canonical benchmark or policy-only ablation.

| Case | Baseline | Candidate | Completion before → after |
|---|---|---|---|
| sandbox/Medium/SCXCCS/0/1 | SUCCESS | SUCCESS | 0.9934 → 0.9934 |
| sandbox/Hard/SCTXrCS/2/1 | SUCCESS | SUCCESS | 0.9929 → 0.9929 |
| sandbox/Hard/SCXOCS/0/1 | SUCCESS | SUCCESS | 0.9924 → 0.9920 |
| sandbox/Hard/XTOCS/0/1 | SUCCESS | SUCCESS | 0.9870 → 0.9870 |

Unmatched baseline cases (excluded from paired score):


Lower completion remains visible in paired_cases.csv; faster failure is not an improvement.
