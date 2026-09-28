# Platform V1 MapSuite Audit and Difficulty Tier Calibration

> **Status:** AUDIT & DESIGN RECOMMENDATION (GATE 2)  
> **Freeze Status:** NON-FROZEN — RESEARCH CANDIDATE SUITE ONLY  
> **Target Scope:** Universal scenario suite supporting Stages 0 through 7  
> **Date:** September 2026 (Updated with Gate-2 Review Refinements)  

---

## 1. Executive Summary & Purpose

Gate 2 of Research Platform V1 establishes the scenario and map difficulty foundation for the entire 8-stage research pipeline:
- **Stage 0:** Random Baseline
- **Stage 1:** Rule / Heuristic
- **Stage 2:** Planning / Search
- **Stage 3:** Simulation Look-ahead (MCTS)
- **Stage 4:** Imitation Learning
- **Stage 5:** Model-Free RL (PPO/SAC)
- **Stage 6:** Search + Learning
- **Stage 7:** Model-Based RL (World Models)

The mission of Gate 2 is to design, experimentally sweep, and calibrate a **four-tier benchmark scenario suite**:
1. **Easy:** Baseline steering, lane centering, and cruising on clean roads (no multi-exit intersections, 0 traffic).
2. **Medium:** Moderate distance, curve handling, single intersection decision block (T or X intersection), low traffic.
3. **Hard:** Longer distance, multiple decision blocks (Intersection, Roundabout, Ramp), moderate traffic.
4. **Extreme:** Longest distance, heterogeneous multi-block composition (Intersections, Roundabouts, Ramps, Merges, Splits), dense traffic.

**Core Mandates:**
- Difficulty is defined by **intrinsic environment properties** (topology, route length, decision blocks, traffic density), NOT by agent failure rates.
- MapSuiteV1 candidates are **NOT frozen** in Gate 2; they form a validated candidate pool.
- No Stage 1+ agents, rewards, observations, or actions were modified.

---

## 2. Source-of-Truth Verification

Authoritative source reference is the local pinned MetaDrive repository:
- **Local Path:** `D:\SGU\CNTT\TTNTNC\metadrive-src`
- **Pinned Git Commit:** `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`
- **Package Version:** `0.4.3` (`metadrive.constants.VERSION == "0.4.3"`)
- **Git Status:** Verified clean working tree via `git status --porcelain`.
- **Automated Validation:** Verified dynamically by `scripts/audit_mapsuite_candidates.py`.

---

## 3. Procedural Map Generation Audit

Procedural generation was audited from source in:
- `metadrive/component/map/pg_map.py` (`PGMap`, `MapGenerateMethod`)
- `metadrive/component/algorithm/BIG.py` (`BIG`, `NextStep`, `BigGenerateMethod`)
- `metadrive/component/algorithm/blocks_prob_dist.py` (`PGBlockDistConfig`)
- `metadrive/component/pg_space.py` (`BlockParameterSpace`, `Parameter`)
- `metadrive/component/pgblock/*`

### Block Types & Registered IDs
MetaDrive registers block primitives through `PGBlockDistConfig` and `get_metadrive_class()`:

| Block Type | Class Name | ID | Description & Maneuver Capability |
|---|---|---|---|
| **First Block** | `FirstPGBlock` | `I` | Spawn straight block ($50\text{ m}$ default) where vehicle initializes. |
| **Straight** | `Straight` | `S` | Linear road corridor without branches. Parameter: `length` $\in [40, 80]\text{ m}$. |
| **Curve** | `Curve` | `C` | Arc curve corridor. Parameters: `radius` $\in [25, 60]\text{ m}$, `angle` $\in [45^\circ, 135^\circ]$, `dir` $\in \{0, 1\}$. |
| **Intersection** | `StdInterSection` | `X` | 4-way standard intersection. **3 alternative turning choices** per entrance. |
| **T-Intersection** | `StdTInterSection` | `T` | 3-way T-intersection. **2 alternative turning choices** per entrance. |
| **Roundabout** | `Roundabout` | `O` | Multi-lane rotary. Multiple exits, circular weaving. |
| **In-Ramp** | `InRampOnStraight` | `r` | Highway on-ramp merging into straight corridor. |
| **Out-Ramp** | `OutRampOnStraight` | `R` | Highway off-ramp diverging from straight corridor. |
| **Merge** | `Merge` | `y` | Bottleneck narrowing lane reduction. |
| **Split** | `Split` | `Y` | Bottleneck widening lane expansion. |
| **Fork In/Out** | `InFork` / `OutFork` | `f` / `F` | Highway fork diverges / converges. |
| **TollGate** | `TollGate` | `$` | Toll booth barrier constriction. |
| **ParkingLot** | `ParkingLot` | `P` | Parking bay lot. |

### How Seed Affects Map Generation
When generating via a string sequence (e.g. `map="SCXCS"`):
1. The sequence prepends `FirstPGBlock` (`I`), forming `"ISCXCS"`.
2. Each block samples its geometric parameters (length, radius, angle, curvature direction) from `BlockParameterSpace` using NumPy PRNG initialized by `scenario_seed` (`random_seed = self.engine.global_random_seed`).
3. For multi-socket blocks (Intersections, Roundabouts), the downstream connection socket is sampled pseudo-randomly via `socket = self.np_random.choice(blocks[-1].get_socket_indices())`.
4. Therefore, **the scenario seed simultaneously determines block geometry, route turns, socket attachment angles, and traffic vehicle spawn positions**.

### BIG Algorithm & Backtracking Mechanics
`BIG` (Block-Intersection Generator) enforces physical feasibility:
- `MAX_TRIAL = 5`: Up to 5 parameter samples are attempted per block.
- In `construct(block)`: The engine checks whether the candidate block collides with previously placed road polygons or if lane count violates bounds $[1, 5]$.
- **Backtracking:** If collision occurs across all 5 trials, `BIG` transitions to `NextStep.back`, popping the failed block, destructing the previous block, and searching sibling sockets on the predecessor.
- **Stability Finding:** Sequences combining multiple wide, loopy blocks (e.g. Roundabout `O` followed by multiple Curves and Ramps) experience higher backtracking frequency, but under well-structured candidate sequences, 100% of tested seeds successfully constructed valid maps.

---

## 4. MapSuite Architecture: Canonical Scenario + Scenario Family

To reconcile reproducible debugging with statistical generalization across Stages 0–7, MapSuiteV1 adopts a two-layer structure:

```text
MapSuiteV1 Tier (e.g. Medium)
   ├── 1. Primary Canonical Scenario (Fixed Sequence + Fixed Seed, e.g. SCXCS Seed 11)
   │      - Deterministic benchmark reference adopted after structural and visual review
   │      - Used for debugging, trajectory visualization, video rendering, baseline comparisons
   │      - Fully serialized block-level configuration
   │
   ├── 2. Alternate Canonical Scenarios (e.g. SCTCS Seed 0, SCXCCS Seed 13)
   │      - Fully validated alternate reference configurations preserved for diagnostics
   │
   └── 3. Scenario Family (Fixed Sequence + Seed Pool 0..19+)
          - Procedural seed variations of identical structural complexity
          - Used for policy training, cross-validation, and out-of-distribution evaluation
```

---

## 5. Controlled Platform Variables

To ensure map difficulty is unpolluted by confounding physical or perceptual artifacts:
- `lane_width = 3.5` meters (fixed across all tiers).
- `base_lane_num = 2` (fixed across all tiers).
- `random_lane_width = False`, `random_lane_num = False`.
- `gaussian_noise = 0.0`, `dropout_prob = 0.0`.
- `accident_prob = 0.0`.
- Control frequency: $\Delta t = 0.10\text{ s}$ ($10\text{ Hz}$ agent cycle, `decision_repeat=5`, `physics_world_step_size=0.02`).
- Observation / Action contracts: strictly uncompressed Gate-1 contracts.

---

## 6. Tier Intent & Multi-Dimensional Difficulty Concept

Difficulty in MapSuiteV1 is **multi-dimensional**—governed by topology, maneuver-decision blocks, curvature, and traffic density—**not defined solely by route distance**. Consequently, route-length distributions between tiers are allowed to overlap naturally (e.g. a long Hard route vs. a compact Extreme network).

| Tier | Topological Intent | Route Length Target | Decision Blocks (`decision_block_count`) | Branching Score (`branching_choice_score`) | Traffic Density Target | Primary Agent Challenge |
|---|---|---|---|---|---|---|
| **Easy** | Pure corridor following (Straight, Curve). Zero multi-exit branch blocks. | $250\text{ m} - 450\text{ m}$ | 0 | 0 | $0.0$ (no traffic) | Basic lane centering, curvature tracking, smooth throttle control. |
| **Medium** | Single branch decision (T or X intersection) embedded in curves/straights. | $450\text{ m} - 700\text{ m}$ | 1 | $2 - 3$ | $0.08$ (low traffic) | Navigation checkpoint following, intersection navigation, light vehicle clearance. |
| **Hard** | Multiple decision blocks combining Intersections, Roundabouts, and Ramps. | $600\text{ m} - 850\text{ m}$ | $2 - 3$ | $6 - 8$ | $0.15$ (moderate traffic) | Multi-stage routing, roundabout circulating, dynamic gap selection. |
| **Extreme** | Heterogeneous long-horizon composition (X, T, Roundabout, Ramps, Merges, Splits). | $850\text{ m} - 1200\text{ m}$ | $4 - 6$ | $10 - 11$ | $0.25$ (dense traffic) | Long-horizon planning, bottleneck merges, dense traffic interaction. |

---

## 7. Candidate Sequence Sweep Results (240 Scenarios)

12 candidate sequences (3 per tier) were swept across 20 scenario seeds each ($20 \times 12 = 240$ scenarios total):

### Metric Clarifications:
- **`decision_block_count`:** Count of physical branching / maneuver-decision blocks encountered (`X`, `T`, `O`, `r`, `R`, `y`, `Y`).
- **`branching_choice_score`:** Weighted branching complexity score ($X \to 3$, $T \to 2$, $O \to 3$, $r/R \to 1$, $y/Y \to 1$).
- **`planned_traffic_vehicle_count`:** Total traffic vehicles planned across blocks under `TrafficMode.Trigger`.

| Tier | Sequence | Seeds Tested | Success Rate | Route Length Range | Mean Route Length | Decision Blocks | Branch Score | Mean Bounding Box ($W \times H$) |
|---|---|---|---|---|---|---|---|---|
| **Easy** | `SCS` | 20 | **20/20 (100%)** | $209.5\text{ m} - 414.5\text{ m}$ | $324.7\text{ m}$ | 0 | 0 | $193\text{ m} \times 152\text{ m}$ |
| **Easy** | `SCSS` | 20 | **20/20 (100%)** | $251.8\text{ m} - 491.5\text{ m}$ | $382.9\text{ m}$ | 0 | 0 | $228\text{ m} \times 178\text{ m}$ |
| **Easy** | `SCCS` | 20 | **20/20 (100%)** | $280.5\text{ m} - 640.4\text{ m}$ | $454.5\text{ m}$ | 0 | 0 | $235\text{ m} \times 245\text{ m}$ |
| **Medium** | `SCXCS` | 20 | **20/20 (100%)** | $352.0\text{ m} - 625.2\text{ m}$ | $509.1\text{ m}$ | 1 | 3 | $237\text{ m} \times 326\text{ m}$ |
| **Medium** | `SCTCS` | 20 | **20/20 (100%)** | $326.7\text{ m} - 708.8\text{ m}$ | $518.2\text{ m}$ | 1 | 2 | $244\text{ m} \times 304\text{ m}$ |
| **Medium** | `SCXCCS` | 20 | **20/20 (100%)** | $478.6\text{ m} - 827.1\text{ m}$ | $666.4\text{ m}$ | 1 | 3 | $275\text{ m} \times 356\text{ m}$ |
| **Hard** | `SCXOCS` | 20 | **20/20 (100%)** | $556.6\text{ m} - 829.8\text{ m}$ | $649.0\text{ m}$ | 2 | 6 | $298\text{ m} \times 382\text{ m}$ |
| **Hard** | `SCTXrCS` | 20 | **20/20 (100%)** | $612.1\text{ m} - 871.2\text{ m}$ | $742.1\text{ m}$ | 3 | 6 | $326\text{ m} \times 402\text{ m}$ |
| **Hard** | `XTOCS` | 20 | **20/20 (100%)** | $397.9\text{ m} - 659.0\text{ m}$ | $508.2\text{ m}$ | 3 | 8 | $268\text{ m} \times 351\text{ m}$ |
| **Extreme** | `CrXROSTR` | 20 | **20/20 (100%)** | $752.0\text{ m} - 1062.4\text{ m}$ | $919.5\text{ m}$ | 6 | 11 | $385\text{ m} \times 478\text{ m}$ |
| **Extreme** | `SCXOCrTYCS` | 20 | **20/20 (100%)** | $884.9\text{ m} - 1288.7\text{ m}$ | $1054.0\text{ m}$ | 5 | 10 | $412\text{ m} \times 524\text{ m}$ |
| **Extreme** | `SCTXORyCCS` | 20 | **20/20 (100%)** | $823.5\text{ m} - 1153.5\text{ m}$ | $1009.7\text{ m}$ | 5 | 10 | $428\text{ m} \times 495\text{ m}$ |

*Machine-readable evidence:* Full row-by-row scenario telemetry is stored in `results/audits/mapsuite/candidate_metrics.csv`.

---

## 8. Primary Canonical Candidates & Alternate Pool Selection

### Human-Reviewed Selection Rationale
Primary canonical candidates were adopted following thorough structural, visual, and pedagogical review:
1. **Easy Primary (`SCS` Seed 11):** Clean $349.6\text{ m}$ baseline. Zero branching decision blocks. Perfect for early steering and throttle calibration.
2. **Medium Primary (`SCXCS` Seed 11):** Balanced $523.8\text{ m}$ corridor introducing a single 4-way intersection (`X`) with light traffic.
3. **Hard Primary (`SCXOCS` Seed 2):** $643.0\text{ m}$ urban network combining a 4-way cross-intersection and a multi-exit roundabout (`O`).
4. **Extreme Primary (`CrXROSTR` Seed 6):** $938.6\text{ m}$ multi-feature composition with ramps (`r`, `R`), intersections (`X`, `T`), and rotaries (`O`).

The remaining candidates per tier are **preserved as alternate canonicals** in the candidate pool for diagnostic evaluation and future scenario testing.

### Canonical Candidates Manifest

| Tier | Candidate Role | Sequence | Seed | Route Length | Decision Blocks | Branch Score | Traffic Density | Planned Traffic | Top-Down Preview Path |
|---|---|---|---|---|---|---|---|---|---|
| **Easy** | **Primary** | `SCS` | `11` | $349.60\text{ m}$ | 0 | 0 | $0.00$ | 0 | `results/audits/mapsuite/previews/easy_rank1_SCS_seed11.png` |
| **Easy** | Alternate | `SCSS` | `9` | $403.15\text{ m}$ | 0 | 0 | $0.00$ | 0 | `results/audits/mapsuite/previews/easy_rank2_SCSS_seed9.png` |
| **Easy** | Alternate | `SCCS` | `9` | $444.15\text{ m}$ | 0 | 0 | $0.00$ | 0 | `results/audits/mapsuite/previews/easy_rank3_SCCS_seed9.png` |
| **Medium** | **Primary** | `SCXCS` | `11` | $523.75\text{ m}$ | 1 | 3 | $0.08$ | 9 | `results/audits/mapsuite/previews/medium_rank1_SCXCS_seed11.png` |
| **Medium** | Alternate | `SCTCS` | `0` | $521.06\text{ m}$ | 1 | 2 | $0.08$ | 10 | `results/audits/mapsuite/previews/medium_rank2_SCTCS_seed0.png` |
| **Medium** | Alternate | `SCXCCS` | `13` | $688.37\text{ m}$ | 1 | 3 | $0.08$ | 15 | `results/audits/mapsuite/previews/medium_rank3_SCXCCS_seed13.png` |
| **Hard** | **Primary** | `SCXOCS` | `2` | $643.01\text{ m}$ | 2 | 6 | $0.15$ | 24 | `results/audits/mapsuite/previews/hard_rank1_SCXOCS_seed2.png` |
| **Hard** | Alternate | `SCTXrCS` | `1` | $748.76\text{ m}$ | 3 | 6 | $0.15$ | 26 | `results/audits/mapsuite/previews/hard_rank2_SCTXrCS_seed1.png` |
| **Hard** | Alternate | `XTOCS` | `19` | $496.89\text{ m}$ | 3 | 8 | $0.15$ | 18 | `results/audits/mapsuite/previews/hard_rank3_XTOCS_seed19.png` |
| **Extreme** | **Primary** | `CrXROSTR` | `6` | $938.58\text{ m}$ | 6 | 11 | $0.25$ | 44 | `results/audits/mapsuite/previews/extreme_rank1_CrXROSTR_seed6.png` |
| **Extreme** | Alternate | `SCXOCrTYCS` | `16` | $1051.77\text{ m}$ | 5 | 10 | $0.25$ | 46 | `results/audits/mapsuite/previews/extreme_rank2_SCXOCrTYCS_seed16.png` |
| **Extreme** | Alternate | `SCTXORyCCS` | `4` | $1003.23\text{ m}$ | 5 | 10 | $0.25$ | 45 | `results/audits/mapsuite/previews/extreme_rank3_SCTXORyCCS_seed4.png` |

*Exact metadata and serialized block configurations:* Saved in `results/audits/mapsuite/canonical_candidates.json` and `configs/maps/mapsuite_v1_candidates.json`.

---

## 9. Top-Down Visual Previews

All 12 candidates were rendered headlessly at $512 \times 512$ resolution using MetaDrive's official `draw_top_down_map` renderer:

- **Easy Tier:**
  - Primary: `SCS` Seed 11 — Straight spawn, gentle sweeping curve, straight exit.
  - Alternate: `SCSS` Seed 9 — Extended straight cruising with single curve.
  - Alternate: `SCCS` Seed 9 — S-curve sequence with consecutive left/right bends.
- **Medium Tier:**
  - Primary: `SCXCS` Seed 11 — Clean approach curve leading into perpendicular 4-way intersection.
  - Alternate: `SCTCS` Seed 0 — S-corridor with T-junction branching point.
  - Alternate: `SCXCCS` Seed 13 — Higher-speed multi-curve sequence flanking standard intersection.
- **Hard Tier:**
  - Primary: `SCXOCS` Seed 2 — 4-way intersection followed by full roundabout circulation.
  - Alternate: `SCTXrCS` Seed 1 — T-junction, cross intersection, and highway on-ramp merge.
  - Alternate: `XTOCS` Seed 19 — Compact high-density urban core with intersection and rotary.
- **Extreme Tier:**
  - Primary: `CrXROSTR` Seed 6 — Multi-block highway network combining curves, ramps, intersections, and rotaries.
  - Alternate: `SCXOCrTYCS` Seed 16 — Long-distance corridor with bottlenecks, splits, and intersections.
  - Alternate: `SCTXORyCCS` Seed 4 — Complex multi-branch loop with merge and split transitions.

---

## 10. Traffic Profile Observations

1. **Traffic Mode Finding:** MetaDrive operates by default in `TrafficMode.Trigger`. Traffic vehicles are procedurally planned across blocks (`block_triggered_vehicles`) and dynamically activated as ego approaches.
2. **Planned Vehicle Counts:**
   - Easy ($\text{density} = 0.0$): 0 vehicles.
   - Medium ($\text{density} = 0.08$): $9 - 15$ planned vehicles.
   - Hard ($\text{density} = 0.15$): $18 - 26$ planned vehicles.
   - Extreme ($\text{density} = 0.25$): $40 - 50$ planned vehicles.
3. **Rollout Note:** Actual concurrent nearby vehicle counts will be measured during dynamic rollout evaluation in later gates.

---

## 11. Objective Horizon Feasibility Analysis (Critical Gate-3 Finding)

The current exploratory environment operates with `horizon = 1000`. At $10\text{ Hz}$ control frequency ($\Delta t = 0.10\text{ s}$), one episode provides an upper bound budget of:
$$\text{Episode Budget} = 1000 \times 0.10\text{ s} = 100.0\text{ seconds}$$

To avoid relying on an arbitrary boolean threshold, we compute the **minimum required average vehicle speed** to complete the route within 100.0s:
$$\bar{v}_{\text{required}} = \frac{\text{route\_length\_m}}{100.0\text{ s}} \times 3.6\quad [\text{km/h}]$$

### Feasibility Against Reference Average Speeds

| Canonical Scenario | Route Length | Required Avg Speed ($\bar{v}_{\text{req}}$) | $15\text{ km/h}$ Feasible? | $20\text{ km/h}$ Feasible? | $25\text{ km/h}$ Feasible? | $35\text{ km/h}$ Feasible? |
|---|---|---|---|---|---|---|
| **Easy Primary (`SCS` s11)** | $349.6\text{ m}$ | **$12.59\text{ km/h}$** | **Yes** | **Yes** | **Yes** | **Yes** |
| **Easy Alternate (`SCSS` s9)**| $403.2\text{ m}$ | **$14.51\text{ km/h}$** | **Yes** | **Yes** | **Yes** | **Yes** |
| **Medium Primary (`SCXCS` s11)**| $523.8\text{ m}$ | **$18.86\text{ km/h}$** | No (Truncated) | **Yes** | **Yes** | **Yes** |
| **Hard Primary (`SCXOCS` s2)** | $643.0\text{ m}$ | **$23.15\text{ km/h}$** | No (Truncated) | No (Truncated) | **Yes** | **Yes** |
| **Hard Alternate (`SCTXrCS` s1)**| $748.8\text{ m}$| **$26.96\text{ km/h}$** | No (Truncated) | No (Truncated) | No (Truncated) | **Yes** |
| **Extreme Primary (`CrXROSTR` s6)**|$938.6\text{ m}$| **$33.79\text{ km/h}$** | No (Truncated) | No (Truncated) | No (Truncated) | **Yes** |
| **Extreme Alternate (`SCXOCrTYCS` s16)**|$1051.8\text{ m}$| **$37.86\text{ km/h}$** | No (Truncated) | No (Truncated) | No (Truncated) | No (Truncated) |

### Finding & Recommendation for Gate 3:
- In Hard and Extreme tiers, routes exceeding $700\text{ m} - 1050\text{ m}$ require maintaining average cruising speeds above $25 - 38\text{ km/h}$ without pausing at intersections or yielding to dense traffic.
- If an agent drives cautiously at $20\text{ km/h}$ through complex roundabouts, it will be **falsely truncated (`max_step=True`) due to step budget exhaustion rather than a driving failure**.
- **Actionable Input for Gate 3:** Gate 3 (Episode Termination and Reward Design) must resolve the horizon policy (e.g. dynamic distance-proportional horizon or tier-based horizon ceilings). Horizon was **not modified** in Gate 2.

---

## 12. Strengthened Map Round-Trip Reproducibility

To ensure map reconstructions are 100% stable:
1. `PGMap.get_meta_data()["block_sequence"]` serializes exact geometric parameters for every block.
2. In `scripts/audit_mapsuite_candidates.py`, a strengthened verification test compares the procedural map vs. `MapGenerateMethod.PG_MAP_FILE` reconstruction, enforcing all 6 reported invariants in the pass condition:
   - **Block IDs Match:** `True` (`ISCS`)
   - **Serialized Block Configuration Match:** `True` (exact equality across all block parameter dictionaries)
   - **Bounding Box Match:** `True`
   - **Route Length Difference:** $0.0000000000\text{ m}$ (exact float match)
   - **Checkpoints Match:** `True`
   - **Top-Down Render Pixel Equality:** `True` (bit-for-bit identical $512 \times 512$ top-down render arrays)
3. Manifests store both the procedural definition (`sequence`, `seed`) and the exact block dictionary.

---

## 13. Future Race Map Note (Showcase Environment)

Inspection of `metadrive/envs/marl_envs/marl_racing_env.py` reveals:
- **`RacingMap` (`PGMap` subclass):** A custom closed-circuit racing track manually composed of Straight and Curve blocks with `remove_negative_lanes=True` and `side_lane_line_type=PGLineType.GUARDRAIL`.
- **Multi-Agent Architecture:** Powered by `MultiAgentMetaDrive` supporting competitive multi-agent simulation (`num_agents=12`).
- **Benchmark Distinction:** The Race Map is **strictly separated from the scientific benchmark MapSuiteV1**. It serves as an evaluation showcase (Human vs. Agent, Agent vs. Agent competitive racing), not a platform difficulty tier.

---

## 14. Future Research Launcher Roadmap Note

A later stage may introduce a **Research Launcher** interface exposing:
- Map tier selection (Easy, Medium, Hard, Extreme);
- Primary vs. Alternate canonical map selection;
- Agent policy selection (Stage 0 through 7);
- Evaluation seed pool / episode count;
- Rendering & HUD toggle;
- Experiment logging & W&B synchronization.

**Constraint:** In benchmark mode, the Research Launcher must strictly lock scientific platform contracts. No UI dependency (e.g. PySide6) is introduced in Gate 2.

---

## 15. Open Questions Before Platform V1 Freeze

1. **Horizon Policy for Gate 3:** Should Platform V1 use a dynamic distance-proportional horizon ($\text{horizon} = \max(1000, 1.5 \times \text{route\_len} / v_{\text{nominal}}$)) or fixed tier-based step ceilings (e.g. 1000 for Easy/Med, 2000 for Hard/Ext)?
2. **Traffic Respawn Policy:** Should background traffic respawn indefinitely (`TrafficMode.Respawn`) or remain finite block-triggered waves (`TrafficMode.Trigger`)?

---

## 16. Non-Frozen Recommendations

1. **Adopt Four-Tier Primary Canonicals:**
   - **Easy Primary:** `SCS` Seed 11 ($349.6\text{ m}$, 0 decision blocks, 0.0 traffic).
   - **Medium Primary:** `SCXCS` Seed 11 ($523.8\text{ m}$, 1 decision block, 0.08 traffic).
   - **Hard Primary:** `SCXOCS` Seed 2 ($643.0\text{ m}$, 2 decision blocks, 0.15 traffic).
   - **Extreme Primary:** `CrXROSTR` Seed 6 ($938.6\text{ m}$, 6 decision blocks, 0.25 traffic).
2. **Preserve Alternate Canonical Pool:** Maintain the 8 validated alternate candidates across tiers for diagnostic and robustness evaluation.
3. **Preserve Scenario Families:** Maintain the 20-seed candidate pools in `candidate_metrics.csv` for post-canonical generalization testing.
