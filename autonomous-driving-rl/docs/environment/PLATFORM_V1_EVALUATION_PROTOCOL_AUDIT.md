# Platform V1 Scenario Splits, Seed Taxonomy, and Scientific Evaluation Protocol Audit

> **Status:** AUDIT & SCIENTIFIC SPECIFICATION (GATE 5)  
> **Freeze Status:** LOCKED-FOR-PLATFORM-V1 (Test Manifest & Seed Plan Locked)  
> **Target Scope:** Standardized evaluation protocol supporting Stages 0 through 7  
> **Date:** September 2026  

---

## 1. Executive Summary & Objective

Gate 5 establishes the **scientific experimental protocol, scenario partition contracts, and seed taxonomy** for Research Platform V1 across all research stages:
- **Stage 0:** Random Baseline
- **Stage 1:** Rule / Heuristic
- **Stage 2:** Planning / Search
- **Stage 3:** Simulation Look-ahead (MCTS)
- **Stage 4:** Imitation Learning
- **Stage 5:** Model-Free RL (PPO/SAC)
- **Stage 6:** Search + Learning
- **Stage 7:** Model-Based RL (World Models)

### Core Scientific Mandates:
1. **Strict Partitioning (180 Train / 48 Validation / 12 Test):** The entire audited 240-geometry universe is deterministically partitioned without overlap or data leakage.
2. **Canonical Test-Set Holdout:** All 12 Gate-2 human-reviewed canonical geometries (1 primary + 2 alternates per tier) are strictly sequestered as **TEST-ONLY**. Zero test geometries appear in training or validation pools.
3. **Explicit Seed Taxonomy:** Disentangles `geometry_generation_seed`, `environment_seed`, `agent_seed`, `training_run_seed`, and `protocol_order_seed` to prevent seed aliasing.
4. **Paired Evaluation:** All algorithms evaluate the identical ordered list of 60 test cases (`test_case_manifest.csv`) shuffled once by `protocol_order_seed = 424242`.
5. **Tier-Level Scorecards:** Benchmark results are reported as four distinct tier scorecards (Easy, Medium, Hard, Extreme) on Gate-4 primary metrics. Arbitrary geometric-mean mega-scores are explicitly prohibited.
6. **Benchmark Contract Fingerprinting:** Cryptographically locks the platform via `benchmark_contract_sha256` and `test_manifest_sha256`.

---

## 2. Source Inputs from Gates 1–4

Gate 5 unifies all previous platform contracts into an executable evaluation protocol:
- **Gate 1:** Uncompressed observation (259D/275D) and continuous actuator (`Box(2)`) contracts.
- **Gate 2 (`MapSuiteV1`):** 12 sequence families $\times$ 20 procedural seeds = 240 candidate geometries (`candidate_metrics.csv`, `canonical_candidates.json`).
- **Gate 3 (`EpisodeSpecV1`):** Gymnasium `terminated` vs. `truncated` semantics, safety-first outcome precedence (`crash_human > crash_vehicle > crash_object > crash_building > crash_sidewalk > out_of_road > arrive_dest > timeout`), and calibrated route-aware horizons.
- **Gate 4 (`RewardSpecV1` & `EvaluationMetricsV1`):** Decoupled reward training signal from the evaluation scorecard; primary benchmark metrics (`clean_success_rate`, `safety_failure_rate`, `mean/median_final_route_completion`, `mean_time_to_clean_success_s`).

---

## 3. Explicit Seed Taxonomy: Never Overload "Seed"

To eliminate ambiguity where a single integer is conflated across multiple roles, Platform V1 formalizes five distinct seed channels:

| Seed Channel | Concept / Identifier | Role & Scope | Controlled Component |
|---|---|---|---|
| **A. Geometry Seed** | `geometry_generation_seed` | Static map generation ($0 \dots 19$). | Procedural block length, curve radius, angle, socket choice. Frozen in geometry manifest. |
| **B. Environment Seed** | `environment_seed` | Simulator stochasticity for a fixed geometry. | Dynamic traffic placement and vehicle configurations. Predeclared test pool: `9101 \dots 9105`. |
| **C. Agent Seed** | `agent_seed` | Agent-side algorithmic stochasticity. | Action sampling noise, neural network weight initialization, exploration RNG. Replicates: `101, 202, 303`. |
| **D. Training Run Seed** | `training_run_seed` | Master seed for one training replicate. | Root seed from which infinite training environment seeds are deterministically derived via `derive_seed()`. |
| **E. Protocol Order Seed** | `protocol_order_seed` | Evaluation case ordering ($424242$). | Fixed deterministic shuffle seed ensuring all algorithms evaluate identical paired test case order. |

*Critical Isolation Rule:* `agent_seed` must **never** be passed into MetaDrive as `environment_seed`.

---

## 4. MetaDrive Seed Semantics & Simulator-Backed Audit

From source inspection of `metadrive/engine/base_engine.py` (lines 96, 563–570) and `metadrive/manager/traffic_manager.py` (lines 339–342):
- When `num_scenarios = 1`, `_num_scenarios_per_level = 1`. Any seed passed to `reset(seed=...)` is modulo-ed by 1 to `start_seed`.
- Therefore, to evaluate an explicit `environment_seed` (e.g. `9101`), `start_seed` must be explicitly configured as `start_seed = 9101`.
- When road geometry is fixed via `MapGenerateMethod.PG_MAP_FILE`:
  - The road network, lane boundaries, and route checkpoints remain 100% frozen.
  - Setting `environment_seed` (`start_seed`) seeds the `TrafficManager` PRNG, producing deterministic variations in traffic vehicle spawn positions and spawn lanes under `TrafficMode.Trigger`.

### Empirical Seed Channel Audit (`seed_channel_reproducibility.csv`)
Tested across Medium (`SCXCS` s11), Hard (`SCXOCS` s2), and Extreme (`CrXROSTR` s6) primary canonicals:
1. **Case A (Same Env Seed 9101 across 2 independent runs):**
   - Initial ego position difference: **`0.000000e+00 m`**
   - Initial observation max difference: **`0.000000e+00`**
   - Initial heading theta difference: **`0.000000e+00`**
   - Planned traffic count match: **`True`** (Medium: 9 veh, Hard: 27 veh, Extreme: 70 veh)
   - Traffic state signatures at step 0: **100% match** (Run 1: `e6845188b1d2aebd`, Run 2: `e6845188b1d2aebd`)
   - Traffic state signatures at step 30: **100% match** (Medium: `501467a4ac2d5cdf`, Hard: `9ee1ed52a92647d8`, Extreme: `f70b55c1d264d1c1`)
2. **Case B (Different Env Seeds 9101 vs. 9102 on Fixed Geometry):**
   - Geometry hash & bounding box: **100% identical**
   - Planned traffic count: **Identical**
   - Traffic vehicle spawn positions & step 30 signatures: **Differ significantly**
     - Medium step 30: seed 9101 (`501467a4ac2d5cdf`) vs. seed 9102 (`57a65571f94bebff`)
     - Hard step 30: seed 9101 (`9ee1ed52a92647d8`) vs. seed 9102 (`b8b28aed76bd4dae`)
     - Extreme step 30: seed 9101 (`f70b55c1d264d1c1`) vs. seed 9102 (`f8ef4e373d3add70`)
3. **Case C (Agent Seed Isolation Control - Option A):**
   - Executed two independent MetaDrive simulator instances with the same geometry and `environment_seed = 9101`, holding environment actions identical while assigning separate external agent RNG instances (`agent_seed = 101` vs. `agent_seed = 202`).
   - Initial ego position difference: **`0.000000e+00 m`**
   - Initial observation max difference: **`0.000000e+00`**
   - Traffic state signatures at steps 0 and 30: **100% match** across all tiers.
   - Formal conclusion: Changing an external agent RNG object has no effect when that RNG is not fed into environment configuration/actions. Agent stochasticity is strictly isolated outside MetaDrive.

---

## 5. 240-Geometry Universe Verification

From Gate-2 `candidate_metrics.csv` and `canonical_candidates.json`:
- **12 Sequence Families:**
  - Easy: `SCS`, `SCSS`, `SCCS`
  - Medium: `SCXCS`, `SCTCS`, `SCXCCS`
  - Hard: `SCXOCS`, `SCTXrCS`, `XTOCS`
  - Extreme: `CrXROSTR`, `SCXOCrTYCS`, `SCTXORyCCS`
- **Procedural Generation Seeds:** Exactly 20 seeds ($0 \dots 19$) per sequence.
- **Total Universe:** $12 \times 20 = \mathbf{240\text{ candidate geometries}}$, audited and confirmed with 100% generation success.

---

## 6. Deterministic Stratified Split Algorithm

To prevent manual cherry-picking while guaranteeing that every sequence and tier is evenly represented across splits, Platform V1 uses deterministic SHA-256 stratified assignment:

### Algorithm (`src/platform/protocol.py`):
```text
For each of the 12 sequence families:
  1. Identify the Gate-2 human-reviewed canonical geometry -> Assign to TEST (1 per sequence)
  2. For the remaining 19 procedural seeds:
     Compute sort_key = sha256("platform-v1-geometry-split-v1" + tier + sequence + seed)
  3. Sort the 19 seeds by sort_key
  4. Assign first 15 seeds -> TRAIN
  5. Assign remaining 4 seeds -> VALIDATION
```

### Partition Distribution:

| Split Role | Count per Sequence | Count per Tier | Total Geometries | Percentage | Primary Purpose |
|---|---|---|---|---|---|
| **TRAIN** | 15 | 45 | **180** | 75.0% | Model training, heuristic exploration, policy optimization. |
| **VALIDATION** | 4 | 12 | **48** | 20.0% | Hyperparameter tuning, ablation studies, model checkpoint selection. |
| **TEST** | 1 | 3 | **12** | 5.0% | Final frozen benchmark evaluation. Zero training/tuning permitted. |
| **TOTAL** | **20** | **60** | **240** | **100.0%** | Full audited MapSuiteV1 universe. |

*Manifest artifact:* `results/audits/evaluation_protocol/geometry_split_manifest.csv`.

---

## 7. Geometry Fingerprinting & Zero-Leakage Audit

Every geometry receives a canonical SHA-256 fingerprint:
$$\text{geometry\_sha256} = \text{SHA256}(\text{json.dumps}(\text{exact\_block\_sequence},\ \text{sort\_keys}=\text{True}))$$

### Full-Precision Regeneration & Verification:
- All 240 geometries are re-synthesized via pinned MetaDrive (`85e5dadc6c7436d324348f6e3d8f8e680c06b4db`, v0.4.3).
- Full-precision route lengths are captured directly from agent navigation and compared against Gate-2 `candidate_metrics.csv` values; all match within $\le 0.05\text{ m}$ tolerance (accounting only for 2-decimal rounding).
- For all 12 canonical test geometries, route length is measured from reconstruction of the authoritative `PG_MAP_FILE` geometry.
- Impact on horizons: Recomputing route-aware horizons with full-precision route length changed **0/12 test horizons** and **0/48 validation horizons**.

### Leakage Invariants Verified (`split_integrity.json`):
- Total geometries: exactly 240.
- TRAIN count: 180; VALIDATION count: 48; TEST count: 12.
- Tier balance: 45 Train / 12 Val / 3 Test for each tier.
- Sequence balance: 15 Train / 4 Val / 1 Test for each sequence family.
- Sequence+Seed pair overlap: **`0` pairs** (completely disjoint).
- Canonical test sequestering: All 12 Gate-2 human-reviewed canonical geometries occupy TEST slots; zero canonicals in TRAIN or VALIDATION.
- Test geometry hash in Train: **`0` matches**.
- Test geometry hash in Validation: **`0` matches**.
- Train geometry hash in Validation: **`0` matches**.
- Duplicate geometry groups detected: **`0`** (every geometry in the 240 universe has a unique SHA-256 block fingerprint).
- Split status: **`VALID`** (verified by `tests/test_evaluation_protocol.py` and `scripts/audit_evaluation_protocol.py`).

---

## 8. Canonical Test-Set Policy & Holdout Enforcement

### The 12 Canonical Test Geometries:
- **Easy:** Primary `SCS` s11 ($349.6\text{ m}$); Alternates `SCSS` s9 ($403.2\text{ m}$), `SCCS` s9 ($444.2\text{ m}$).
- **Medium:** Primary `SCXCS` s11 ($523.8\text{ m}$); Alternates `SCTCS` s0 ($521.1\text{ m}$), `SCXCCS` s13 ($688.4\text{ m}$).
- **Hard:** Primary `SCXOCS` s2 ($643.0\text{ m}$); Alternates `SCTXrCS` s1 ($748.8\text{ m}$), `XTOCS` s19 ($496.9\text{ m}$).
- **Extreme:** Primary `CrXROSTR` s6 ($938.6\text{ m}$); Alternates `SCXOCrTYCS` s16 ($1051.8\text{ m}$), `SCTXORyCCS` s4 ($1003.2\text{ m}$).

### Public Repository Holdout Rule:
Because the repository is public, test scenarios cannot be kept secret. Scientific holdout is enforced by **protocol contract**:
- Public availability of test scenarios does **not** permit tuning algorithms to those scenarios.
- No model weights may be updated on test geometries.
- No hyperparameters, planner horizons, or heuristic thresholds may be selected using test outcomes.
- Checkpoint selection must occur strictly on the 96 **VALIDATION** cases.
- Any method using geometry-specific test knowledge is classified as "test-informed" and is non-compliant with the scientific benchmark.

---

## 9. Evaluation Case Suites: Test & Validation

### 1. Final Test Case Manifest (`test_case_manifest.csv`)
Combines the 12 canonical test geometries with 5 predeclared test environment seeds:
$$12\text{ geometries} \times 5\text{ environment seeds} = \mathbf{60\text{ evaluation cases}}$$
- **Test Environment Seeds:** `[9101, 9102, 9103, 9104, 9105]`.
- **Tier Balance:** Exactly 15 test cases per tier ($3\text{ geoms} \times 5\text{ env seeds} = 15$).
- **Paired Protocol Order:** Deterministically shuffled once using `protocol_order_seed = 424242`. All algorithms evaluate cases in this identical sequence.

### 2. Validation Case Manifest (`validation_case_manifest.csv`)
Combines the 48 validation geometries with 2 predeclared validation environment seeds:
$$48\text{ geometries} \times 2\text{ environment seeds} = \mathbf{96\text{ validation cases}}$$
- **Validation Environment Seeds:** `[5101, 5102]`.
- **Tier Balance:** Exactly 24 validation cases per tier ($12\text{ geoms} \times 2\text{ env seeds} = 24$).

---

## 10. Training Seed Derivation

For training learned policies across the 180 TRAIN geometries, environment seeds are not pre-expanded into a rigid table. Instead, algorithms derive infinite reproducible training seeds from `training_run_seed`:

$$\text{env\_seed} = \text{derive\_seed}(\text{training\_run\_seed},\ \text{"environment"},\ \text{episode\_index},\ \text{geometry\_id})$$
$$\text{geom\_order\_seed} = \text{derive\_seed}(\text{training\_run\_seed},\ \text{"geometry-order"},\ \text{epoch\_index})$$

This provides unlimited reproducible training schedules without touching the sequestered validation or test seed pools.

---

## 11. Agent Replicates & Stochasticity Policy

- **Stochastic / Learning Agents (Stages 4–7):**
  - Evaluated across 3 independent training replicates: `training_run_seed` $\in \{\mathbf{101, 202, 303}\}$.
  - Report mean $\pm$ standard deviation across the 3 replicates for all primary metrics.
  - Do not pool episodes across multiple models into a single fake single-model pool.
- **Random Baseline (Stage 0):**
  - Evaluated across 3 independent action RNG seeds: `agent_seed` $\in \{\mathbf{101, 202, 303}\}$.
- **Deterministic Heuristic & Planning Agents (Stages 1–3):**
  - Genuinely deterministic methods report one run per test case (60 runs total).
  - Documented as `agent_stochasticity = deterministic`; does not duplicate identical runs to manufacture a zero standard deviation.

---

## 12. Benchmark Reporting Policy: Per-Tier Scorecards

Benchmark scorecards must report results broken down by tier:

```text
========================================================================================
BENCHMARK SCORECARD: Algorithm X (Mean ± Std over 3 Replicates)
========================================================================================
Tier       Clean Success (%)    Safety Failure (%)    Route Comp (%)    Time-to-Success (s)
----------------------------------------------------------------------------------------
Easy           100.0 ± 0.0          0.0 ± 0.0          100.0 ± 0.0          40.7 ± 0.2
Medium          86.7 ± 4.7         13.3 ± 4.7           92.4 ± 3.1          61.5 ± 1.8
Hard            60.0 ± 6.7         40.0 ± 6.7           78.6 ± 5.4          75.2 ± 2.4
Extreme         26.7 ± 4.7         73.3 ± 4.7           64.2 ± 6.1         114.8 ± 3.5
----------------------------------------------------------------------------------------
MACRO (Equal)   68.3 ± 3.8         31.7 ± 3.8           83.8 ± 3.2           N/A
========================================================================================
```

### Macro Aggregation:
- `macro_clean_success_rate = (Easy + Medium + Hard + Extreme) / 4.0`
- `macro_safety_failure_rate = (Easy + Medium + Hard + Extreme) / 4.0`
- `macro_mean_final_route_completion = (Easy + Medium + Hard + Extreme) / 4.0`
- Equal tier weighting ($0.25$ per tier).

### Explicit Rejection of Geometric-Mean Mega-Score:
A geometric-mean success score $\left(\prod S_i\right)^{1/4}$ is **strictly rejected**:
1. It collapses to 0 if an algorithm fails a single tier (e.g. Extreme).
2. Adding arbitrary epsilon ($\epsilon$) distorts rankings based on epsilon tuning.
3. It conceals the true tier-specific failure pattern.
4. Gate 4 established a multi-metric scorecard; no single scalar "winner score" is permitted.

---

## 13. Benchmark Locking & Contract Hashes

To guarantee that benchmark results are verifiable and mutation-proof, Platform V1 locks all specifications into cryptographic fingerprints:

```json
{
  "benchmark_contract_sha256": "9ddd889b84d8705fae618879e5035556c80d0276a3dc2a58a7963937ebb59f77",
  "evaluation_protocol_core_sha256": "30b78e6e1b2d0b84d2c74326080a59dc57d5a42d0d400a60edd161d76ad5209e",
  "test_manifest_sha256": "0832c38e2e8a0a3bb0c6cafbb2ba63cfcd1dcbbf84b4c7d6b31b9eaf5dc8ec77",
  "geometry_split_manifest_sha256": "3b07e94b99766f409b000455304ae2980a05467156734ee519ccf27068bdc481",
  "validation_case_manifest_sha256": "10afc2afcfd1c3e8e6f201448bdeb79ee5b07b3c0ec92e8e6e1adb7c10f42ac1"
}
```

### Canonical Content Hashing Methodology:
Hashes describe **semantic protocol content**, not local filesystem line endings or indentation formatting:
- **JSON Contracts:** Parsed via `json.load()` and hashed with `canonical_json_sha256(parsed_object)` (`sort_keys=True, separators=(",", ":")`). Completely invariant to CRLF vs. LF line endings and whitespace.
- **CSV Manifests:** Parsed via `csv.DictReader(f)` where each row is normalized into a dictionary with sorted keys, and the list of rows is encoded via `canonical_json_sha256`. Completely invariant to CRLF vs. LF and column order.
- **Protocol Rules Integrity:** `evaluation_protocol_core_sha256` fingerprints all evaluation rules (`paired_evaluation`, `test_set_holdout`, `validation_usage`, `replicate_reporting`, `tier_scorecards`) and case summaries, embedding this fingerprint into `benchmark_contract_sha256`.

The `benchmark_contract_sha256` digest securely incorporates the canonical content hashes of contracts across Gates 1 through 5:
- Gate 1 schema definitions (`observation_schema.json`, `action_schema.json`)
- Gate 2 candidate manifest and canonical configurations (`mapsuite_v1_candidates.json`, `canonical_candidates.json`)
- Gate 3 episode specification (`episode_spec_v1.json`)
- Gate 4 reward and evaluation metric specifications (`reward_spec_v1.json`, `evaluation_metrics_v1.json`)
- Gate 5 scenario split, seed plan, evaluation protocol core, and manifests (`scenario_split_v1.json`, `seed_plan_v1.json`, `evaluation_protocol_core`, `geometry_split_manifest.csv`, `test_case_manifest.csv`, `validation_case_manifest.csv`)

### Circular Hashing Elimination & Single-Run Consistency:
`evaluation_protocol_v1.json` embeds `benchmark_contract_sha256` and is **not** part of its own input hash. After writing all artifacts, `verify_contract_hashes_against_disk()` recomputes all contract hashes from disk and asserts bit-for-bit equality against `protocol_hashes.json`. A single clean run produces a fully self-consistent repository state.

### Version Bump Policy:
Any modification to:
- Test geometries, sequence families, or canonical block parameters;
- Test environment seeds (`9101..9105`);
- Tier assignments or split salt;
- Control frequency or episode horizon;

requires an explicit **protocol version bump** (`EvaluationProtocolV2`), producing a new `benchmark_contract_sha256`. Results across differing contract hashes cannot be directly compared.

---

## 14. Unit Tests & Verification Summary

Implemented in `tests/test_evaluation_protocol.py` (20 pure unit tests executing in $<0.05\text{ s}$ without Panda3D):
- Verified exact 240-geometry universe.
- Verified 180 Train / 48 Validation / 12 Test counts.
- Verified 45/12/3 tier balance and 15/4/1 sequence balance.
- Verified all 12 Gate-2 canonicals in Test only; 0 canonicals in Train/Validation.
- Verified canonical test identity enforcement (fails loudly if canonical missing or leaked).
- Verified zero geometry hash or sequence+seed leakage across pairwise split sets.
- Verified zero duplicate geometry groups across the 240 universe.
- Verified deterministic split assignment reproducibility.
- Verified deterministic `derive_seed()` 31-bit integer generation.
- Verified traffic density propagation and route-aware horizon computation in test/validation case builders.
- Verified uniqueness of all 60 test case IDs and 96 validation case IDs.
- Verified equal-tier macro metric calculation and absence of geometric-mean mega-scores.
- Verified that mutating even a single environment seed changes `test_manifest_sha256`.
- Verified canonical JSON formatting invariance (CRLF vs LF, compact vs indented).
- Verified canonical CSV formatting invariance (CRLF vs LF line endings).
- Verified committed manifest integrity directly from disk.
- Verified that mutating an evaluation rule changes `evaluation_protocol_core_sha256`.
- Verified that case builders use supplied Gate-3 `HorizonPolicy`.
- Verified that varying `safety_margin` dynamically modifies computed horizons.
- Verified that current `EpisodeSpecV1` reproduces the locked 60 test and 96 validation case horizons.

---

## 15. Regression & Integrity Validation
- Pure unit tests: **46 total tests pass across repo** (`test_episode_lifecycle`, `test_reward_metrics`, `test_evaluation_protocol`).
- `CourseEnvV1`: Reset OK (shape `35,`).
- `evaluate_random.py`: Completed 20 evaluation episodes cleanly.
- Gate-1, Gate-2, Gate-3, and Gate-4 contracts remain completely untouched.
- Zero hardcoded drive letters in executable code.

STOP. PR remains open and unmerged.
