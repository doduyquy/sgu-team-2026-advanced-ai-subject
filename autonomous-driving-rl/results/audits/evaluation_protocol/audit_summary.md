# Gate 5 Scenario Splits & Scientific Evaluation Protocol Summary

## 1. Verified MetaDrive Source
- **Pinned Commit:** `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`
- **Package Version:** `0.4.3`
- **Working Tree:** Verified clean via `git status --porcelain`.

## 2. Geometry Split Architecture (180 Train / 48 Validation / 12 Test)
- **Universe Verification:** Exactly 240 geometries audited from Gate-2 candidate metrics (12 sequence families x 20 procedural seeds).
- **True Geometry Fingerprinting:** All 240 geometries fingerprinted from canonical serialized block sequences. Zero duplicate geometry hashes detected across the 240 universe.
- **Full-Precision Route Lengths:** Regenerated directly from agent navigation and reconstructed PG_MAP_FILE geometries; verified against Gate-2 CSV values within rounding tolerance.
- **Horizon Invariance:** Recomputing route-aware horizons with full-precision route length changed 0/12 test horizons and 0/48 validation horizons.
- **Stratified Split Method:** Stratified per sequence family using deterministic SHA-256 assignment (`platform-v1-geometry-split-v1`).
- **Split Counts:**
  - **TRAIN:** 180 geometries (45 per tier, 15 per sequence family)
  - **VALIDATION:** 48 geometries (12 per tier, 4 per sequence family)
  - **TEST:** 12 geometries (3 per tier, 1 per sequence family - all Gate-2 human-reviewed canonicals)
- **Zero Leakage:** Complete disjointness verified; zero sequence+seed pair overlap, zero test geometry hash in train/val, zero canonical test geometries in train/val.

## 3. Seed Taxonomy & Stochasticity Channels
- **`geometry_generation_seed` (0..19):** Controls procedural map generation. Frozen in geometry manifests.
- **`environment_seed`:** Controls traffic spawn placement under fixed `PG_MAP_FILE` geometry. Tested on seeds `9101..9105`.
- **`agent_seed`:** Controls agent exploratory stochasticity (`101, 202, 303`). Strictly isolated from simulator environment seed.
- **`training_run_seed`:** Root seed for learning replicates; training env seeds are derived deterministically via `derive_seed()`.
- **`protocol_order_seed` (424242):** Deterministic shuffle seed ensuring all algorithms evaluate identical paired test case order.

## 4. Test & Validation Case Suites
- **Test Suite:** Exactly 60 cases (12 test geometries x 5 test env seeds: `9101, 9102, 9103, 9104, 9105`). Exactly 15 cases per tier.
- **Validation Suite:** Exactly 96 cases (48 validation geometries x 2 validation env seeds: `5101, 5102`). Exactly 24 cases per tier.

## 5. Benchmark Locking & Contract Hashes
- **`benchmark_contract_sha256`:** `c273e3e1376eb6b4349824f5be8bc909cf5208c40edbdbe22802c1e2e979648b`
- **`test_manifest_sha256`:** `0832c38e2e8a0a3bb0c6cafbb2ba63cfcd1dcbbf84b4c7d6b31b9eaf5dc8ec77`
- **`geometry_split_manifest_sha256`:** `3b07e94b99766f409b000455304ae2980a05467156734ee519ccf27068bdc481`

## 6. Evaluation Principles
- **Paired Evaluation:** All algorithms evaluate the identical ordered 60 test cases.
- **Reporting:** Per-tier primary scorecards on Clean Success, Safety Failure, Route Completion, and Conditional Time-to-Success. Rejection of arbitrary geometric-mean mega-scores.
- **Holdout Enforcement:** TEST manifest is public but strictly locked; tuning on test cases is strictly prohibited.
