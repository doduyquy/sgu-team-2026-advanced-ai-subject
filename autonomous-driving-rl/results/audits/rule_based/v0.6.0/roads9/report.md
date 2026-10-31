# Rule-based paired TRAIN development comparison

Baseline: `rule_based_v050_acceptance_family12_20261030/current`; candidate: `rule_based_v060_accept04_roads9_20261030/current`.

Matched cases: 9; clean success 9 → 9; crashes 0 → 0.

Both runs completed. Source/configuration hashes and every recorded trace row count were verified.

This compares the policy and extractor together. It is not a canonical benchmark or policy-only ablation.

| Case | Baseline | Candidate | Completion before → after |
|---|---|---|---|
| sandbox/Easy/SCCS/0/0 | SUCCESS | SUCCESS | 0.9918 → 0.9918 |
| sandbox/Easy/SCS/0/0 | SUCCESS | SUCCESS | 0.9877 → 0.9877 |
| sandbox/Easy/SCSS/0/0 | SUCCESS | SUCCESS | 0.9890 → 0.9890 |
| sandbox/Medium/SCTCS/2/0 | SUCCESS | SUCCESS | 0.9909 → 0.9909 |
| sandbox/Medium/SCXCCS/0/0 | SUCCESS | SUCCESS | 0.9933 → 0.9933 |
| sandbox/Medium/SCXCS/14/0 | SUCCESS | SUCCESS | 0.9865 → 0.9865 |
| sandbox/Hard/SCTXrCS/2/0 | SUCCESS | SUCCESS | 0.9932 → 0.9932 |
| sandbox/Hard/SCXOCS/0/0 | SUCCESS | SUCCESS | 0.9920 → 0.9918 |
| sandbox/Hard/XTOCS/0/0 | SUCCESS | SUCCESS | 0.9870 → 0.9875 |

Unmatched baseline cases (excluded from paired score):

- sandbox/Extreme/CrXROSTR/0/0
- sandbox/Extreme/SCTXORyCCS/0/0
- sandbox/Extreme/SCXOCrTYCS/0/0

Lower completion remains visible in paired_cases.csv; faster failure is not an improvement.
