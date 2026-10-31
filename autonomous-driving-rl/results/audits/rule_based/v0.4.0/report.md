# Rule-based paired TRAIN development comparison

Baseline: `rule_based_v030_baseline12_20261030`; candidate: `rule_based_v040_regression9_20261030`.

Matched cases: 9; clean success 5 → 7; crashes 4 → 2.

Both runs completed. Source/configuration hashes and every recorded trace row count were verified.

This compares the policy and extractor together. It is not a canonical benchmark or policy-only ablation.

| Case | Baseline | Candidate | Completion before → after |
|---|---|---|---|
| sandbox/Easy/SCCS/0/0 | SUCCESS | SUCCESS | 0.9918 → 0.9918 |
| sandbox/Easy/SCS/0/0 | SUCCESS | SUCCESS | 0.9877 → 0.9877 |
| sandbox/Easy/SCSS/0/0 | SUCCESS | SUCCESS | 0.9890 → 0.9890 |
| sandbox/Medium/SCTCS/2/0 | SUCCESS | SUCCESS | 0.9912 → 0.9912 |
| sandbox/Medium/SCXCCS/0/0 | CRASH_VEHICLE | SUCCESS | 0.4641 → 0.9936 |
| sandbox/Medium/SCXCS/14/0 | SUCCESS | SUCCESS | 0.9863 → 0.9868 |
| sandbox/Hard/SCTXrCS/2/0 | CRASH_VEHICLE | SUCCESS | 0.2677 → 0.9933 |
| sandbox/Hard/SCXOCS/0/0 | CRASH_VEHICLE | CRASH_VEHICLE | 0.7161 → 0.5689 |
| sandbox/Hard/XTOCS/0/0 | CRASH_VEHICLE | CRASH_VEHICLE | 0.1459 → 0.1429 |

Unmatched baseline cases (excluded from paired score):

- sandbox/Extreme/CrXROSTR/0/0
- sandbox/Extreme/SCTXORyCCS/0/0
- sandbox/Extreme/SCXOCrTYCS/0/0

Lower completion remains visible in paired_cases.csv; faster failure is not an improvement.
