# Dataset Card: MetaDrive MapSuite V1

- **Dataset ID:** `mapsuite_v1`
- **Dataset Name:** MetaDrive MapSuite V1
- **Dataset Version:** `1.0.0`
- **Dataset Type:** Procedural Autonomous-Driving Scenario Benchmark Dataset
- **Scientific Status:** `DERIVED_FROM_FROZEN_PLATFORM_V1_BENCHMARK`
- **Upstream Engine:** MetaDrive `0.4.3` (pinned commit `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`)

---

## 1. Executive Summary & Purpose

**MetaDrive MapSuite V1** is the frozen, reproducible procedural map and scenario benchmark dataset for the autonomous-driving research platform (`autonomous-driving-rl`).

Its primary purpose is to provide a **fixed, standardized universe of 240 road geometries** and canonical evaluation suites so that different algorithm families (Random baselines, heuristic rule agents, search/planners, and reinforcement learning policies) can be trained, tuned, and evaluated under identical, controlled geometric conditions.

---

## 2. What this Dataset IS NOT

To prevent common misunderstandings among teammates and external collaborators, note explicitly:

- **NOT an image dataset:** The repository includes 12 representative top-down preview renderings for visual inspection, but the dataset is not a computer-vision or camera-image benchmark.
- **NOT 240 exported 3D mesh files:** The geometries are generated algorithmically on-the-fly inside MetaDrive from sequence strings and seeds.
- **NOT an offline RL transition dataset:** It does not contain state-action-reward tuples or replay buffers.
- **NOT a human demonstration dataset:** It contains zero human driving traces or expert teleoperation logs.
- **NOT an OpenStreetMap (OSM) or real-world road network dataset:** All roads are procedural synthetic corridor compositions.
- **NOT a Vietnam-specific traffic dataset:** Roads follow standardized synthetic geometric configurations, not Vietnamese traffic or roadway geometry.

---

## 3. Dataset Composition & Metrics

MapSuite V1 consists of **240 procedural road geometries** organized into:

- **12 Scenario Families:** 3 distinct block-composition sequence families per difficulty tier.
- **20 Procedural Seeds per Family:** Geometry generation seeds $0 \dots 19$, producing diverse road curvature, branch lengths, and socket connections.
- **4 Intrinsic Difficulty Tiers:**
  - **Easy:** Linear cruising, lane centering, simple curves (`SCS`, `SCSS`, `SCCS`), traffic density = 0.0.
  - **Medium:** Longer distance, single decision block (T- or X-intersection) (`SCXCS`, `SCTCS`, `SCXCCS`), traffic density = 0.08.
  - **Hard:** Multi-block decision networks (Intersections, Roundabouts, Ramps) (`SCXOCS`, `SCTXrCS`, `XTOCS`), traffic density = 0.15.
  - **Extreme:** Heterogeneous composition (Intersections, Roundabouts, Merges, Splits, Ramps) (`CrXROSTR`, `SCXOCrTYCS`, `SCTXORyCCS`), traffic density = 0.25.

---

## 4. Scientific Partitioning & Evaluation Suites

The 240 geometries are partitioned into three strictly disjoint scientific splits:

| Split | Geometry Count | Purpose & Usage Policy |
|---|---|---|
| **TRAIN** | **180** | Open development, exploratory sandbox experiments, and policy/model training. |
| **VALIDATION** | **48** | Hyperparameter selection, checkpoint ablation, and model selection. **96 canonical evaluation cases**. |
| **TEST** | **12** | **Strict holdout benchmark.** Evaluated once for frozen final comparison. **60 canonical evaluation cases**. |

### Canonical Evaluation Manifests
- **Validation Cases (`evaluation_cases/validation_cases.csv`):** 96 paired cases (48 validation geometries × 2 fixed environment seeds).
- **Test Cases (`evaluation_cases/test_cases.csv`):** 60 paired cases (12 test geometries × 5 fixed environment seeds), deterministically ordered via `protocol_order_seed = 424242`.

### Holdout & No-Tuning Policy
The test manifest is checked into this repository to make the benchmark 100% reproducible and auditable. However, **TEST cases must NEVER be used to tune algorithm hyperparameters, select model checkpoints, or redesign scenario parameters**. If an algorithm struggles on TEST, that limitation must be documented as an empirical research finding, not bypassed by changing the benchmark.

---

## 5. Procedural Reconstruction & Integrity

Every geometry in MapSuite V1 is deterministically reconstructible from four parameters:

$$\text{Geometry} = f_{\text{MetaDrive}}(\text{sequence}, \text{geometry\_generation\_seed}, \text{lane\_width}=3.5, \text{base\_lane\_num}=2)$$

1. **`sequence`:** String of block codes (e.g. `SCXCS`) assembled with an implicit initial straight spawn block `I`.
2. **`geometry_generation_seed`:** Integer seed ($0 \dots 19$) that seeds MetaDrive's Block-Intersection Generator (`BIG`).
3. **`geometry_sha256`:** Canonical cryptographic fingerprint of the road network and route checkpoints. This hash allows automated preflight checks to guarantee that an environment instantiated on any machine has zero geometric drift.

---

## 6. Source-of-Truth Rule

This dataset package is **strictly derived** from the frozen Platform V1 source audit artifacts:

- `results/audits/mapsuite/candidate_metrics.csv`
- `results/audits/mapsuite/canonical_candidates.json`
- `results/audits/evaluation_protocol/geometry_split_manifest.csv`
- `results/audits/evaluation_protocol/validation_case_manifest.csv`
- `results/audits/evaluation_protocol/test_case_manifest.csv`

**Do NOT hand-edit the CSV or JSON files in this directory.**
Any update or packaging pass must be run through the deterministic exporter:

```powershell
python scripts/export_mapsuite_v1_dataset.py
python scripts/verify_mapsuite_v1_dataset.py
```

---

## 7. Package Contents & Navigation

```text
datasets/mapsuite_v1/
├── README.md               # This dataset card
├── SCHEMA.md               # Detailed schema definition for every table and column
├── REPRODUCE.md            # Reproduction and export verification instructions
├── CHECKSUMS.sha256        # Cryptographic checksums of all package contents
├── dataset_manifest.json   # Machine-readable dataset manifest and provenance
├── generation_config.json  # Procedural generation parameters and bounds
├── geometries.csv          # Master catalog of all 240 geometries
├── scenario_families.csv   # Summary catalog of the 12 scenario families
├── splits/                 # Partitioned geometry subsets
│   ├── train_geometries.csv        # 180 TRAIN geometries
│   ├── validation_geometries.csv   # 48 VALIDATION geometries
│   └── test_geometries.csv         # 12 TEST geometries
├── evaluation_cases/       # Canonical paired evaluation manifests
│   ├── validation_cases.csv        # 96 canonical validation cases
│   └── test_cases.csv              # 60 canonical test cases
└── previews/               # 12 representative top-down scenario family previews
```
