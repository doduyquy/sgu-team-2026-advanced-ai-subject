# Rule-based paired TRAIN development comparison

Baseline: `rule_based_v050_acceptance_family12_20261030/current`; candidate: `rule_based_v060_accept04_extreme_cr_20261030/current`.

Matched cases: 1; clean success 0 → 0; crashes 0 → 0.

Both runs completed. Source/configuration hashes and every recorded trace row count were verified.

This compares the policy and extractor together. It is not a canonical benchmark or policy-only ablation.

| Case | Baseline | Candidate | Completion before → after |
|---|---|---|---|
| sandbox/Extreme/CrXROSTR/0/0 | TIMEOUT | TIMEOUT | 0.4031 → 0.4031 |

Unmatched baseline cases (excluded from paired score):

- sandbox/Easy/SCCS/0/0
- sandbox/Easy/SCS/0/0
- sandbox/Easy/SCSS/0/0
- sandbox/Medium/SCTCS/2/0
- sandbox/Medium/SCXCCS/0/0
- sandbox/Medium/SCXCS/14/0
- sandbox/Hard/SCTXrCS/2/0
- sandbox/Hard/SCXOCS/0/0
- sandbox/Hard/XTOCS/0/0
- sandbox/Extreme/SCTXORyCCS/0/0
- sandbox/Extreme/SCXOCrTYCS/0/0

Lower completion remains visible in paired_cases.csv; faster failure is not an improvement.
