# MetaDrive MapSuite V1 Schema Documentation

This document describes the schema, data types, value ranges, and semantic definitions for all tabular files in the `mapsuite_v1` dataset package.

---

## 1. Critical Distinction: `geometry_generation_seed` vs. `environment_seed`

Teammates and external users frequently conflate these two concepts. In Platform V1, they represent two completely distinct and orthogonal sources of randomness:

| Seed Concept | Column Name | Range / Values | Component Controlled | Invariance Policy |
|---|---|---|---|---|
| **Geometry Generation Seed** | `geometry_generation_seed` | Integer $0 \dots 19$ | **Static Road Network:** Road curvature, corridor lengths, intersection branching angles, lane topology. | **Static:** Once generated, the physical road layout and centerline geometry never change. |
| **Environment Seed** | `environment_seed` | Integer (e.g. $9101 \dots 9105$) | **Dynamic Simulation:** Initial placement of surrounding dynamic traffic vehicles, spawn timing, and background vehicle behaviors. | **Dynamic:** Evaluating the *same* geometry with *different* environment seeds tests an agent's robustness to varying traffic flow on identical roads. |

*Critical Isolation Rule:* An algorithm's algorithmic seed (`agent_seed`) must **never** be substituted for either `geometry_generation_seed` or `environment_seed`.

---

## 2. Master Catalog: `geometries.csv`

The master catalog contains exactly 240 rows, each describing one unique procedural road geometry.

| Column | Type | Example | Description |
|---|---|---|---|
| `geometry_id` | String | `geom_easy_SCS_seed0` | Unique canonical geometry identifier formatted as `geom_<tier>_<sequence>_seed<id>`. |
| `split` | String | `TRAIN`, `VALIDATION`, `TEST` | Scientific partition assignment (180 TRAIN, 48 VALIDATION, 12 TEST). |
| `tier` | String | `Easy`, `Medium`, `Hard`, `Extreme` | Intrinsic difficulty tier of the scenario. |
| `sequence` | String | `SCXCS` | MetaDrive block sequence string defining the sequence of road blocks. |
| `geometry_generation_seed` | Integer | `11` | Procedural generation seed ($0 \dots 19$) used to initialize MetaDrive's `BIG` algorithm. |
| `candidate_role` | String | `primary_canonical`, `alternate_canonical`, `pool_candidate` | Scientific benchmark role: `primary_canonical` (Gate-2 primary canonical benchmark geometry, 1 per tier), `alternate_canonical` (Gate-2 alternate canonical geometry, 2 per tier), or `pool_candidate` (standard geometry in the training/validation pool). |
| `geometry_sha256` | String | `5999ad11...` | Cryptographic SHA-256 fingerprint of the road network and route checkpoints. |
| `geometry_hash_source` | String | `gate2_stored_exact`, `pinned_regeneration` | Method/origin of the geometry hash: `gate2_stored_exact` (exact hash verified from Gate-2 canonical audit) or `pinned_regeneration` (hash derived from pinned procedural regeneration in Gate 5). |
| `generation_success` | Boolean | `True` | Whether MetaDrive's `BIG` algorithm constructed the map without unresolvable physical collision. |
| `block_ids` | String | `ISCXCS` | Full ordered block string, including the implicit initial spawn straight block `I`. |
| `block_count` | Integer | `6` | Total count of road blocks in the corridor. |
| `route_length_m` | Float | `487.35` | Total centerline route distance in meters from start to destination checkpoint. |
| `bbox_width_m` | Float | `246.50` | Horizontal bounding box span of the road network in meters. |
| `bbox_height_m` | Float | `312.80` | Vertical bounding box span of the road network in meters. |
| `lane_width_m` | Float | `3.5` | Standard lane width in meters (fixed across MapSuite V1). |
| `base_lane_num` | Integer | `2` | Base number of lanes per direction (fixed at 2 across MapSuite V1). |
| `straight_count` | Integer | `3` | Number of straight segments (`S` and `I`) in the corridor. |
| `curve_count` | Integer | `2` | Number of arc curve blocks (`C`) in the corridor. |
| `intersection_count` | Integer | `1` | Number of 4-way standard intersection blocks (`X`). |
| `t_intersection_count` | Integer | `0` | Number of 3-way T-intersection blocks (`T`). |
| `roundabout_count` | Integer | `0` | Number of roundabout blocks (`O`). |
| `ramp_count` | Integer | `0` | Number of on-ramp (`r`) or off-ramp (`R`) blocks. |
| `merge_split_count` | Integer | `0` | Number of lane-merge (`y`) or lane-split (`Y`) bottleneck blocks. |
| `decision_block_count` | Integer | `1` | Count of multi-exit decision blocks requiring routing choice (Intersections, Roundabouts, Ramps). |
| `branching_choice_score` | Integer | `3` | Cumulative number of alternative exit choices across all decision blocks. |
| `traffic_density` | Float | `0.08` | Intrinsic traffic density parameter assigned to this tier/scenario family. |
| `planned_traffic_vehicle_count` | Integer | `9` | Total background traffic vehicles planned for the scenario corridor under nominal density. |
| `episode_budget_seconds` | Float | `100.0` | Nominal time budget in seconds allocated for a standard 1000-step horizon at 10 Hz. |
| `required_avg_speed_kmh_for_horizon_1000` | Float | `17.54` | Minimum average speed (in km/h) required to reach destination within the 1000-step horizon. |

---

## 3. Scenario Families: `scenario_families.csv`

Contains exactly 12 rows describing the 12 canonical scenario families that comprise MapSuite V1.

| Column | Type | Example | Description |
|---|---|---|---|
| `tier` | String | `Medium` | Difficulty tier (`Easy`, `Medium`, `Hard`, `Extreme`). |
| `family_rank_within_tier` | Integer | `1`, `2`, `3` | Relative structural ranking within the difficulty tier. |
| `sequence` | String | `SCXCS` | Representative block sequence defining the family. |
| `geometry_seed_count` | Integer | `20` | Count of procedural seeds in this family (always 20). |
| `geometry_seed_min` | Integer | `0` | Minimum procedural seed index. |
| `geometry_seed_max` | Integer | `19` | Maximum procedural seed index. |
| `traffic_density` | Float | `0.08` | Traffic density assigned to all members of this family. |
| `representative_geometry_seed` | Integer | `11` | Procedural seed selected as the representative canonical exemplar for this family. |
| `representative_preview` | String | `previews/medium_rank1_SCXCS_seed11.png` | Relative path to the top-down visual preview image. |

---

## 4. Split Manifests: `splits/`

Contains three files that partition `geometries.csv`:
- `splits/train_geometries.csv` (180 rows)
- `splits/validation_geometries.csv` (48 rows)
- `splits/test_geometries.csv` (12 rows)

Each split file preserves the identical 29-column schema of `geometries.csv`, filtered deterministically by the `split` column.

---

## 5. Evaluation Manifests: `evaluation_cases/`

Contains the concrete paired evaluation cases evaluated by research agents during benchmark execution:
- `evaluation_cases/validation_cases.csv` (96 rows)
- `evaluation_cases/test_cases.csv` (60 rows)

| Column | Type | Example | Description |
|---|---|---|---|
| `case_index` | Integer | `53` | Deterministic 1-based index assigned during initial grid construction (in tier/sequence/geom_seed/env_seed sorted order) prior to protocol shuffling. |
| `case_id` | String | `test/Extreme/CrXROSTR/geom-6/env-9101` | Unique evaluation case identifier formatted as `<split>/<tier>/<sequence>/geom-<seed>/env-<seed>`. |
| `tier` | String | `Extreme` | Difficulty tier of the evaluated geometry. |
| `sequence` | String | `CrXROSTR` | Scenario family sequence string. |
| `candidate_role` | String | `primary_canonical` | Role of the underlying geometry (`primary_canonical` or `alternate_canonical`). |
| `geometry_generation_seed` | Integer | `6` | Seed used to generate the static road network. |
| `geometry_sha256` | String | `f1a8c3...` | Cryptographic fingerprint verifying road layout integrity. |
| `environment_seed` | Integer | `9101` | Seed used to initialize dynamic traffic positions and initial conditions. |
| `traffic_density` | Float | `0.25` | Traffic density active during this case. |
| `horizon_steps` | Integer | `1224` | Route-aware calibrated step horizon limit for this case. |
| `protocol_order_index` | Integer | `1` | Deterministic 1-based execution order index assigned after shuffling the evaluation grid with `protocol_order_seed = 424242`. All algorithms evaluate cases in this exact sequence. |

---

## 6. MetaDrive Block Code Glossary

The `sequence` and `block_ids` strings use MetaDrive's standard single-letter block identifiers:

| Code | Block Type | Description |
|---|---|---|
| `I` | First Block | Initial straight corridor ($50\text{ m}$) where the vehicle spawns. |
| `S` | Straight | Linear road corridor without intersections ($40\text{ m} \dots 80\text{ m}$). |
| `C` | Curve | Arc curve corridor ($25\text{ m} \dots 60\text{ m}$ radius, $45^\circ \dots 135^\circ$ angle). |
| `X` | Intersection | 4-way standard intersection with 3 branching turn options. |
| `T` | T-Intersection | 3-way T-intersection with 2 branching turn options. |
| `O` | Roundabout | Multi-lane circular rotary with multiple exits. |
| `r` | In-Ramp | Highway on-ramp merging into a straight corridor. |
| `R` | Out-Ramp | Highway off-ramp diverging from a straight corridor. |
| `y` | Merge | Road narrowing bottleneck / lane reduction. |
| `Y` | Split | Road widening bottleneck / lane expansion. |
