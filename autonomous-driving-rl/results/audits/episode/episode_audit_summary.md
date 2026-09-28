# Gate 3 Episode Lifecycle & Termination Audit Summary

## 1. Verified MetaDrive Source
- **Pinned Commit:** `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`
- **Package Version:** `0.4.3`
- **Working Tree:** Verified clean via `git status --porcelain`.

## 2. Termination & Outcome Precedence Findings
- **`truncate_as_terminate=False`:** Verified as standard Gymnasium contract (`terminated=False, truncated=True` on timeout).
- **Sidewalk Contact:** `crash_sidewalk` is NOT directly checked in `done_function()`, but triggers termination **indirectly through `out_of_road`** via `_is_out_of_road()`.
- **Safety-First Precedence:** Safety-critical events (`crash_human > crash_vehicle > crash_object > crash_building > crash_sidewalk > out_of_road`) take absolute precedence over destination arrival.
- **Clean Success:** Arrival (`arrive_dest=True`) is categorized as `clean_success=True` IF AND ONLY IF zero safety failure flags occurred on the same step.

## 3. Horizon Calibration Across 12 Canonical Scenarios (Actual MapSuite Traffic)
- **Default `horizon=1000` Defect:** In Extreme scenarios (938 m to 1052 m), reference IDM driving requires 1123 to 1255 steps even at ~29.5 km/h. Under a 1000-step budget, these reference rollouts would be truncated prior to arrival.
- **Route-Aware Formula:** horizon = max(1000, ceil(route_len / 5.0 * 1.5 * 10)) provides healthy emergency safety margins (2.47x to 2.58x over reference IDM time) without arbitrary speed pressure.

### Calibrated Primary Canonical Horizons
| Tier | Primary Sequence | Seed | Route Length | Planned Traffic | IDM Completion | Proposed Horizon | Budget Seconds |
|---|---|---|---|---|---|---|---|
| Easy | `SCS` | 11 | 349.6 m | 0 | 407 steps (40.7s) | **1049 steps** | 104.9 s |
| Medium | `SCXCS` | 11 | 523.75 m | 9 | 383 steps (38.3s) | **1572 steps** | 157.2 s |
| Hard | `SCXOCS` | 2 | 643.01 m | 27 | 249 steps (24.9s) | **1930 steps** | 193.0 s |
| Extreme | `CrXROSTR` | 6 | 938.58 m | 70 | 563 steps (56.3s) | **2816 steps** | 281.6 s |

## 4. Traffic Lifecycle & Reset Reproducibility
- **`TrafficMode.Trigger`:** Verified finite preplanned traffic population and repeatable initialization/activation across independent runs.
- **`TrafficMode.Respawn`:** Source semantics permit continual replacement; this can make traffic exposure episode-duration dependent.
- **Reset Reproducibility:** Verified zero state leakage across resets following real terminal failures; initial positions, headings, and observations match with 0.00e+00 error.
