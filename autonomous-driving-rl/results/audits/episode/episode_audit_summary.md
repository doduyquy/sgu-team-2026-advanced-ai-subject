# Gate 3 Episode Lifecycle & Termination Audit Summary

## 1. Verified MetaDrive Source
- **Pinned Commit:** `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`
- **Package Version:** `0.4.3`
- **Working Tree:** Verified clean via `git status --porcelain`.

## 2. Termination & Outcome Precedence Findings
- **`truncate_as_terminate=False`:** Verified as standard Gymnasium contract (`terminated=False, truncated=True` on timeout).
- **Control Test (`truncate_as_terminate=True`):** Confirmed simulator produces `terminated=True, truncated=True` on timeout.
- **Sidewalk Contact:** `crash_sidewalk` is NOT directly checked in `done_function()`, but triggers termination **indirectly through `out_of_road`** via `_is_out_of_road()`.
- **Safety-First Precedence:** Safety-critical events (`crash_human > crash_vehicle > crash_object > crash_building > crash_sidewalk > out_of_road`) take absolute precedence over destination arrival.
- **Clean Success:** Arrival (`arrive_dest=True`) is categorized as `clean_success=True` IF AND ONLY IF zero safety failure flags occurred on the same step.

## 3. Horizon Calibration Across 12 Canonical Scenarios (Actual MapSuite Traffic)
Evaluated with candidate benchmark traffic (`traffic_density = cand['traffic_density']`, `traffic_mode = 'trigger'`) and exact reconstructed geometry via `PG_MAP_FILE`:

| Tier | Role | Sequence | Seed | Route Length | Planned Traffic | Reference Outcome | IDM Speed | Proposed Horizon | Budget Seconds | Margin over IDM |
|---|---|---|---|---|---|---|---|---|---|---|
| Easy | primary_canonical | `SCS` | 11 | 349.6 m | 0 | SUCCESS @ 407 steps | 28.97 km/h | **1049 steps** | 104.9 s | 2.58 |
| Easy | alternate_canonical | `SCSS` | 9 | 403.15 m | 0 | SUCCESS @ 478 steps | 29.15 km/h | **1210 steps** | 121.0 s | 2.53 |
| Easy | alternate_canonical | `SCCS` | 9 | 444.15 m | 0 | SUCCESS @ 533 steps | 29.24 km/h | **1333 steps** | 133.3 s | 2.5 |
| Medium | primary_canonical | `SCXCS` | 11 | 523.75 m | 9 | OUT_OF_ROAD @ 383 steps | 28.76 km/h | **1572 steps** | 157.2 s | N/A |
| Medium | alternate_canonical | `SCTCS` | 0 | 521.06 m | 7 | SUCCESS @ 629 steps | 29.28 km/h | **1564 steps** | 156.4 s | 2.49 |
| Medium | alternate_canonical | `SCXCCS` | 13 | 688.37 m | 12 | SUCCESS @ 844 steps | 28.91 km/h | **2066 steps** | 206.6 s | 2.45 |
| Hard | primary_canonical | `SCXOCS` | 2 | 643.01 m | 27 | CRASH_VEHICLE @ 249 steps | 27.53 km/h | **1930 steps** | 193.0 s | N/A |
| Hard | alternate_canonical | `SCTXrCS` | 1 | 748.76 m | 28 | CRASH_VEHICLE @ 780 steps | 27.71 km/h | **2247 steps** | 224.7 s | N/A |
| Hard | alternate_canonical | `XTOCS` | 19 | 496.89 m | 17 | SUCCESS @ 678 steps | 25.96 km/h | **1491 steps** | 149.1 s | 2.2 |
| Extreme | primary_canonical | `CrXROSTR` | 6 | 938.58 m | 70 | CRASH_VEHICLE @ 563 steps | 12.43 km/h | **2816 steps** | 281.6 s | N/A |
| Extreme | alternate_canonical | `SCXOCrTYCS` | 16 | 1051.77 m | 79 | CRASH_VEHICLE @ 618 steps | 20.86 km/h | **3156 steps** | 315.6 s | N/A |
| Extreme | alternate_canonical | `SCTXORyCCS` | 4 | 1003.23 m | 58 | CRASH_VEHICLE @ 405 steps | 26.54 km/h | **3010 steps** | 301.0 s | N/A |

## 4. Secondary Zero-Traffic Reference Traversal Times
Evaluated with zero traffic (`traffic_density = 0.0`) to measure clean traversal capability:

| Tier | Role | Sequence | Seed | Route Length | Reference Outcome | Steps | Time | IDM Speed | Counterfactual 1000 Status |
|---|---|---|---|---|---|---|---|---|---|
| Easy | primary_canonical | `SCS` | 11 | 349.6 m | SUCCESS | 407 | 40.7 s | 28.97 km/h | COMPLETED_WITHIN_1000 |
| Easy | alternate_canonical | `SCSS` | 9 | 403.15 m | SUCCESS | 478 | 47.8 s | 29.15 km/h | COMPLETED_WITHIN_1000 |
| Easy | alternate_canonical | `SCCS` | 9 | 444.15 m | SUCCESS | 533 | 53.3 s | 29.24 km/h | COMPLETED_WITHIN_1000 |
| Medium | primary_canonical | `SCXCS` | 11 | 523.75 m | OUT_OF_ROAD | 383 | 38.3 s | 28.76 km/h | OUT_OF_ROAD_AT_STEP_383 |
| Medium | alternate_canonical | `SCTCS` | 0 | 521.06 m | SUCCESS | 629 | 62.9 s | 29.28 km/h | COMPLETED_WITHIN_1000 |
| Medium | alternate_canonical | `SCXCCS` | 13 | 688.37 m | SUCCESS | 830 | 83.0 s | 29.47 km/h | COMPLETED_WITHIN_1000 |
| Hard | primary_canonical | `SCXOCS` | 2 | 643.01 m | SUCCESS | 779 | 77.9 s | 29.36 km/h | COMPLETED_WITHIN_1000 |
| Hard | alternate_canonical | `SCTXrCS` | 1 | 748.76 m | SUCCESS | 897 | 89.7 s | 29.5 km/h | COMPLETED_WITHIN_1000 |
| Hard | alternate_canonical | `XTOCS` | 19 | 496.89 m | SUCCESS | 599 | 59.9 s | 29.21 km/h | COMPLETED_WITHIN_1000 |
| Extreme | primary_canonical | `CrXROSTR` | 6 | 938.58 m | SUCCESS | 1123 | 112.3 s | 29.45 km/h | WOULD_TRUNCATE_UNDER_1000 |
| Extreme | alternate_canonical | `SCXOCrTYCS` | 16 | 1051.77 m | SUCCESS | 1255 | 125.5 s | 29.6 km/h | WOULD_TRUNCATE_UNDER_1000 |
| Extreme | alternate_canonical | `SCTXORyCCS` | 4 | 1003.23 m | SUCCESS | 1218 | 121.8 s | 29.57 km/h | WOULD_TRUNCATE_UNDER_1000 |

*Finding:* Under zero traffic, all 3 Extreme scenarios completed successfully, requiring 1123 to 1255 steps at ~29.5 km/h. Under a fixed 1000-step budget, these clean reference rollouts would be truncated prior to arrival.

## 5. Traffic Lifecycle & Reset Reproducibility
- **`TrafficMode.Trigger`:** Verified finite preplanned population and repeatable observed activation/state trace under the tested same-seed, same-action configuration.
- **`TrafficMode.Respawn`:** Source semantics replace vehicles after removal; the tested trace maintained 9 active vehicles.
- **Reset Reproducibility:** Verified zero state leakage across resets following real terminal failures; initial positions, headings, and observations match with 0.00e+00 error across all required invariants.
