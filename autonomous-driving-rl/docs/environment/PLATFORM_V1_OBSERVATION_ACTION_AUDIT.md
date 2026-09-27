# Platform V1 Observation, Action, and Control Contract Audit

> **Status:** AUDIT & DESIGN RECOMMENDATION (GATE 1)  
> **Freeze Status:** NON-FROZEN — RECOMMENDATIONS ONLY  
> **Target Scope:** Universal compatibility across Stages 0 through 7  
> **Date:** September 2026 (Updated with Gate-1 Audit Refinements)  

---

## 1. Source-of-Truth Verification

The authoritative reference for this audit is the local pinned MetaDrive source tree:

- **Local Path:** `D:\SGU\CNTT\TTNTNC\metadrive-src`
- **Pinned Git Commit:** `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`
- **Package Name:** `metadrive-simulator`
- **Package Version:** `0.4.3` (`metadrive.constants.VERSION == "0.4.3"`)
- **Automated Verification:** Verified programmatically within `scripts/run_observation_action_audit.py` via `git rev-parse HEAD`, `git status --porcelain`, and package version validation. The script halts with an explicit error if the commit differs or if the working tree is dirty. No source modifications were made.

---

## 2. Observation Source Audit & Discrepancy Findings

The MetaDrive observation pipeline was audited from source code in:
- `metadrive/obs/state_obs.py` (`StateObservation`, `LidarStateObservation`)
- `metadrive/component/navigation_module/node_network_navigation.py` (`NodeNetworkNavigation`)
- `metadrive/component/vehicle/base_vehicle.py` (`BaseVehicle`)
- `metadrive/component/sensors/lidar.py` (`Lidar`)
- `metadrive/component/sensors/distance_detector.py` (`DistanceDetector`)

### Critical Source Discrepancies and Terminology Clarifications

1. **Ego Last-Action Ordering Inversion:**
   - In `metadrive/obs/state_obs.py` lines 39–41, the docstring claims:
     ```python
     # Current steering,
     # Throttle/brake of last frame,
     # Steering of last frame,
     ```
   - However, the actual code implementation in `state_obs.py` lines 116–122 is:
     ```python
     # Current steering
     clip((vehicle.steering / vehicle.MAX_STEERING + 1) / 2, 0.0, 1.0),
     # The normalized actions at last steps
     clip((vehicle.last_current_action[1][0] + 1) / 2, 0.0, 1.0),
     clip((vehicle.last_current_action[1][1] + 1) / 2, 0.0, 1.0)
     ```
   - In `BaseVehicle`, `action[0]` is **steering** and `action[1]` is **throttle/brake**.
   - Therefore, **Index 5 is steering** and **Index 6 is throttle/brake**. The official docstring had these two fields swapped. Empirical tests with `[0.2, 0.6]` confirmed `obs[5] == 0.6` (steering) and `obs[6] == 0.8` (throttle).

2. **Road Border Measurements Are Geometric Abstractions (Not Sensor Rays):**
   - With the default configuration (`vehicle_config["side_detector"]["num_lasers"] == 0`), `dist_to_left_side` and `dist_to_right_side` (Indices 0 and 1) **ARE NOT side-detector raycasts**.
   - They are calculated analytically from vehicle position and the reference lane / route geometry (`_dist_to_route_left_right()`), normalized by `total_width`.
   - Likewise, lane offset (Index 8), heading deviation (Index 2), and navigation checkpoints (Indices 9–18) are **geometric state abstractions** derived directly from simulator and map state.
   - Consequently, default MetaDrive observation (Candidate A) is **not a "pure sensor observation"**; it is MetaDrive's default state-based local observation without explicit surrounding-vehicle ground-truth tracks.

3. **Heading Difference Semantics:**
   - Index 2 (`heading_diff`) is **not a raw angle in radians**.
   - It is a normalized signed heading-deviation proxy computed via the dot product between the vehicle heading vector and the lane lateral unit vector:
     $$\text{heading\_diff} = \frac{\text{clip}\left(\frac{\mathbf{v}_{\text{heading}} \cdot \mathbf{v}_{\text{lane\_lateral}}}{\|\mathbf{v}_{\text{heading}}\| \|\mathbf{v}_{\text{lane\_lateral}}\|}, -1.0, 1.0\right)}{2} + 0.5$$
   - When aligned with the lane: $\mathbf{v}_{\text{heading}} \cdot \mathbf{v}_{\text{lane\_lateral}} = 0 \implies \text{heading\_diff} = 0.5$.
   - A $+90^\circ$ perpendicular deviation yields `1.0`; a $-90^\circ$ perpendicular deviation yields `0.0`.

4. **LiDAR Angular Sweep Direction:**
   - In `state_obs.py` line 199, the docstring states: `"starting from the vehicle head in clockwise direction"`.
   - The actual mathematical implementation in `DistanceDetector._get_lidar_range` and `get_laser_end` is:
     $$\theta_i = \text{heading\_theta} + i \times \frac{2\pi}{N}$$
   - In 2D Cartesian coordinates (+x forward, +y left), an increasing positive angle rotates from +x toward +y, which is **COUNTER-CLOCKWISE**.
   - Empirical obstacle placements at 0°, +90°, 180°, and 270° produced hits at ray 0 (Front), ray 60 (Left), ray 120 (Rear), and ray 180 (Right), confirming a counter-clockwise sweep.

---

## 3. Exact Observation Index Table

Authoritative field-level definition for Candidate B (275D) and Candidate A (259D):

| Index / Slice | Field Name | Category / Abstraction Type | Semantic Meaning | Raw Units | Normalization Formula | Neutral / Zero Value | Privileged State? |
|---|---|---|---|---|---|---|---|
| `[0]` | `dist_to_left_side` | State abstraction (lane geometry) | Distance to left road border | meters | `clip(dist / total_width, 0, 1)` | `0.0` = touching/beyond left curb | No |
| `[1]` | `dist_to_right_side` | State abstraction (lane geometry) | Distance to right road border | meters | `clip(dist / total_width, 0, 1)` | `0.0` = touching/beyond right curb | No |
| `[2]` | `heading_diff` | State abstraction (lane geometry) | Signed heading-deviation proxy (dot product with lane lateral vector) | scalar proxy | `clip(cos(heading, lane_lat), -1, 1)/2 + 0.5` | `0.5` = aligned with lane heading; `1.0` = +90°, `0.0` = -90° | No |
| `[3]` | `current_speed` | Proprioceptive sensor | Vehicle forward speed | km/h | `clip((speed_km_h + 1)/(max_speed_km_h + 1), 0, 1)` | `~0.0123` = stationary ($0\text{ km/h}$) | No |
| `[4]` | `current_steering` | Proprioceptive sensor | Front wheel turn angle | degrees | `clip((steering / MAX_STEERING + 1)/2, 0, 1)` | `0.5` = wheels straight; `0.0` = max left, `1.0` = max right | No |
| `[5]` | `last_action_steering` | Internal actuator feedback | Steering command from last step | normalized `[-1, 1]` | `clip((action[0] + 1)/2, 0, 1)` | `0.5` = zero steering command | No |
| `[6]` | `last_action_throttle` | Internal actuator feedback | Throttle/brake command from last step | normalized `[-1, 1]` | `clip((action[1] + 1)/2, 0, 1)` | `0.5` = coast/idle command ($0.0$) | No |
| `[7]` | `yaw_rate` | Proprioceptive sensor | Vehicle angular velocity magnitude | rad/s | `clip(arccos(clip(h_now . h_last, 0, 1))/0.1, 0, 1)` | `0.0` = zero angular turn rate (straight line) | No |
| `[8]` | `lane_lateral_offset` | State abstraction (lane geometry) | Lateral offset from current lane centerline | meters | `clip((lateral * 2 / max_lane_width + 1.0)/2.0, 0, 1)` | `0.5` = centered on lane; `<0.5` right, `>0.5` left | No |
| `[9]` | `navi_ckpt1_long` | Navigation abstraction | Projected distance to Checkpoint 1 along heading | meters | `clip((dx / 50.0 + 1)/2, 0, 1)` | `0.5` = checkpoint laterally aligned | No |
| `[10]` | `navi_ckpt1_lat` | Navigation abstraction | Projected distance to Checkpoint 1 along RHS | meters | `clip((dy / 50.0 + 1)/2, 0, 1)` | `0.5` = checkpoint directly ahead/behind | No |
| `[11]` | `navi_ckpt1_bend_radius` | Navigation abstraction | Bending radius of lane to Checkpoint 1 | normalized | `clip(radius / (CURVE_MAX + lane_num * lane_width), 0, 1)` | `0.0` = straight lane | No |
| `[12]` | `navi_ckpt1_bend_dir` | Navigation abstraction | Curvature direction of lane to Checkpoint 1 | sign | `clip((-ref_lane.direction + 1)/2, 0, 1)` | `0.5` = straight lane; `1.0` clockwise, `0.0` CCW | No |
| `[13]` | `navi_ckpt1_lane_angle` | Navigation abstraction | Total angle change of lane to Checkpoint 1 | degrees | `clip((deg(angle)/ANGLE_MAX + 1)/2, 0, 1)` | `0.5` = straight lane | No |
| `[14]` | `navi_ckpt2_long` | Navigation abstraction | Projected distance to Checkpoint 2 along heading | meters | `clip((dx / 50.0 + 1)/2, 0, 1)` | `0.5` = checkpoint laterally aligned | No |
| `[15]` | `navi_ckpt2_lat` | Navigation abstraction | Projected distance to Checkpoint 2 along RHS | meters | `clip((dy / 50.0 + 1)/2, 0, 1)` | `0.5` = checkpoint directly ahead/behind | No |
| `[16]` | `navi_ckpt2_bend_radius` | Navigation abstraction | Bending radius of lane to Checkpoint 2 | normalized | `clip(radius / (CURVE_MAX + lane_num * lane_width), 0, 1)` | `0.0` = straight lane | No |
| `[17]` | `navi_ckpt2_bend_dir` | Navigation abstraction | Curvature direction of lane to Checkpoint 2 | sign | `clip((-ref_lane.direction + 1)/2, 0, 1)` | `0.5` = straight lane | No |
| `[18]` | `navi_ckpt2_lane_angle` | Navigation abstraction | Total angle change of lane to Checkpoint 2 | degrees | `clip((deg(angle)/ANGLE_MAX + 1)/2, 0, 1)` | `0.5` = straight lane | No |
| `[19:35]` | `surrounding_vehicles` | Privileged physics query | 4 nearest detected vehicles (`rel_x`, `rel_y`, `rel_vx`, `rel_vy`) | m, km/h | `(pos/50.0 + 1)/2`, `(vel/max_speed + 1)/2` | **`0.0` = absent/padded vehicle**; `0.5` = co-located & zero relative speed | **YES (Privileged)** |
| `[35:275]` (or `[19:259]`) | `lidar_cloud_points` | Active range sensor | 240 LiDAR distance rays sweeping 360° counter-clockwise | fraction of 50m | `hit_distance / 50.0` | `1.0` = clear path ($50\text{ m}$); `0.0` = contact or dropped ray | No |

---

## 4. LiDAR Semantics

- **Ray Count & Range:** 240 rays, max distance $50.0\text{ m}$, angular resolution $1.5^\circ$.
- **Ray 0 Orientation & Sweep:** Ray 0 points directly along the vehicle heading vector ($0^\circ$). Increasing ray indices advance **COUNTER-CLOCKWISE**:
  - Ray 0 ($0^\circ$): Forward / Heading
  - Ray 60 ($90^\circ$): Direct Left
  - Ray 120 ($180^\circ$): Direct Rear
  - Ray 180 ($270^\circ$): Direct Right
- **Normalization:** Hit fraction $\in [0.0, 1.0]$. `1.0` means clear path ($50\text{ m}$); `0.0` means immediate contact ($0\text{ m}$) or dropped ray.
- **Physical Sensing Targets:** Dynamic vehicles and static traffic obstacles (`TrafficCone`, `TrafficBarrier`) reside in `physics_world.dynamic_world` and are sensed. **Static road borders, curbs, sidewalks, and lane markings reside in `physics_world.static_world` and ARE NOT DETECTED by LiDAR**.
- **Noise Characteristics:** `gaussian_noise` perturbs hit fractions with $\mathcal{N}(0, \sigma^2)$. `dropout_prob` forces rays to `0.0` (which semantically mimics an obstacle at contact distance).

---

## 5. Surrounding-Vehicle Semantics (`num_others = 4`)

1. **Dimensions:** `num_others=0` $\to$ 259D; `num_others=1` $\to$ 263D; `num_others=4` $\to$ 275D.
2. **Vehicle Ordering:** Sorted ascending by Euclidean distance from ego vehicle.
3. **Four Features per Vehicle:** `rel_x`, `rel_y`, `rel_vx`, `rel_vy`.
4. **Padding Semantics:** Absent slots are filled with `[0.0, 0.0, 0.0, 0.0]`.  
   *Note:* `0.0` indicates an absent/dummy vehicle, whereas `0.5` represents an active vehicle at zero relative position and zero relative velocity.
5. **Privileged State Leakage:** Vehicles are collected via a 50m broad-phase cylinder query in the physics world. This query **bypasses LiDAR ray occlusion completely** (detects cars behind opaque obstacles) and reads simulator internal coordinates and velocities directly.

---

## 6. Observation Candidate Comparison

We evaluate four observation candidates (Candidates A, B, C, and new Candidate D):

- **Candidate A (259D):** MetaDrive default local state + 240 LiDAR rays. State-based local observation without explicit surrounding vehicle tracks.
- **Candidate B (275D):** MetaDrive state + navigation + 4 surrounding vehicles (16D) + 240 LiDAR rays.
- **Candidate C (35D):** CourseEnv experimental representation (9 ego + 10 navigation + 16 min-pooled LiDAR sectors).
- **Candidate D (Structured AgentInput Concept — Open Candidate):** Decoupled architecture separating CoreObservation (259D), optional TrafficContext (structured surrounding vehicle state with explicit validity mask), TaskContext (read-only map context), and private EvaluatorTelemetry.

### Comparison Matrix

| Criterion | Candidate A (259D) | Candidate B (275D) | Candidate C (35D) | Candidate D (Structured Concept) |
|---|---|---|---|---|
| **Information Sufficiency** | Moderate (full spatial occupancy; lacks obstacle velocity) | High (explicit bounding-box relative positions & velocities) | Low (coarse 16-sector spatial pooling; no velocities) | **Very High** (full uncompressed observation + structured contexts) |
| **Dimensionality** | Moderate (259D vector) | Moderate (275D vector) | Low (35D vector) | Modular (259D Core + typed contexts) |
| **Algorithm Neutrality** | High | Medium (bakes privileged tracker into flat vector) | Low (bakes in specific min-pooling) | **Very High** (agents consume only what their contract permits) |
| **Dynamic Obstacle Velocity** | Requires frame stacking / recurrent models | Explicitly provided in observation | Missing | Provided in optional `TrafficContext` with validity mask |
| **Ability to Support Learning (RL/IL)** | Strong; standard for MetaDrive benchmarks | Strong; faster sample efficiency | Good for toy baselines, poor for complex maneuvers | **Strong**; clean separation prevents policy cheating |
| **Ability to Support Planning/Search** | Requires ray triangulation for local collision checking | High for local avoidance, but still lacks global map topology | Unusable for collision-free trajectory optimization | **Ideal**; cleanly pairs local obstacle checks with global `TaskContext` |
| **Risk of Information Loss** | Minimal | None | **High** (averages/min-pools 15 rays into 1 sector) | None |
| **Privileged State Leakage** | Low (only ego telemetry & map checkpoints) | **High** (direct physics-world query of other vehicles) | Low | **Strictly Managed** (traffic tracker is explicit and optional) |
| **Computational Cost** | Low | Low | Very Low | Low |

---

## 7. Stage 0–7 Compatibility Matrix & Planning Nuance

### Conceptual Disentanglement of Planning Requirements
To avoid overclaiming that a single flat observation vector "solves" or "breaks" planning:
1. **Global Route Planning:** Requires topological road connectivity, lane graph connections, and route goal checkpoints. Neither 259D nor 275D provides global topology; both require a standardized, read-only `TaskContext` / `MapContext`.
2. **Local Motion / Trajectory Planning:** Requires drivable corridor boundaries and local obstacle occupancy. Candidate A (259D) provides spatial occupancy via 240 LiDAR rays, enabling local path generation, though without obstacle velocities. Candidate B (275D) adds explicit relative positions and velocities.
3. **Dynamic Obstacle Interaction:** Crucial for predictive collision checking (e.g. time-to-collision heuristics, lattice rollouts). Candidate B and Candidate D supply dynamic velocities; Candidate A requires temporal estimation.

### Stage-by-Stage Compatibility Matrix

| Project Stage | Candidate A (259D) | Candidate B (275D) | Candidate C (35D) | Candidate D (Structured Concept) | Architectural Recommendation |
|---|---|---|---|---|---|
| **Stage 0: Random Baseline** | Compatible | Compatible | Compatible | Compatible | Any candidate functions; 35D has minimal overhead. |
| **Stage 1: Rule / Heuristic** | Moderate (requires ray grouping for TTC) | Suitable (TTC computed directly) | Degraded (no velocity, coarse sectors) | **Ideal** (rules can inspect CoreObservation + optional TrafficContext) | Rule agents benefit from explicit obstacle distance and velocity. |
| **Stage 2: Planning / Search** | Partially Suitable (local obstacle checks via LiDAR; needs map context) | Suitable for local collision; lacks global topology | Insufficient (cannot plan fine trajectories through gaps) | **Ideal** (local planning from Core, global planning from TaskContext) | Global planning requires `TaskContext`; local planning needs fine occupancy. |
| **Stage 3: Simulation Look-ahead (MCTS)** | Partially Suitable (high ray dimension for rollouts) | Suitable (structured states for forward collision checks) | Insufficient (coarse sectors cause collision blindness) | **Ideal** (structured rollouts with explicit state representation) | MCTS forward simulation benefits from structured obstacle tracking. |
| **Stage 4: Imitation Learning** | Suitable | Suitable | Degraded | **Suitable** | Expert behavior correlates strongly with surrounding traffic dynamics. |
| **Stage 5: Model-Free RL (PPO/SAC)** | **Highly Suitable** (clean, standard benchmark baseline) | Suitable (faster sample efficiency; risk of privileged dependence) | Suitable for fast initial smoke test | **Highly Suitable** (policy trains on CoreObservation) | PPO learns robustly on 259D without relying on privileged physics tracks. |
| **Stage 6: Search + Learning** | Partially Suitable | Suitable | Insufficient | **Ideal** (learning uses CoreObservation, search uses TaskContext) | Hybrid systems require clean architectural separation. |
| **Stage 7: Model-Based RL / World Models** | **Suitable** (latent dynamics reconstruct raw LiDAR) | Suitable | Degraded | **Suitable** | World models (e.g. Dreamer) model temporal sequences of raw observations. |

---

## 8. Agent-Visible Information vs. Privileged Telemetry & Candidate D

### Proposed Information Boundary

```text
+-----------------------------------------------------------------------------------------+
|                                    SIMULATOR STATE                                      |
+-----------------------------------------------------------------------------------------+
       |                                                                |
       v                                                                v
[AGENT-VISIBLE / PUBLIC INTERFACE]                    [EVALUATOR / LOGGER-ONLY INTERFACE]
- CoreObservation: ego kinematics, lane offsets,      - Ground-truth global trajectory
  navigation checkpoints, 240-ray LiDAR               - Route completion percentage
- Optional TrafficContext: structured surrounding     - Collision flags (crash_vehicle, crash_object)
  vehicle tracks + explicit boolean validity mask      - Out-of-road termination signals
- Non-frozen TaskContext: read-only goal & route       - Simulation internal RNG seeds & object pointers
```

### Telemetry Field Classification

| Field / Concept | Classification | Justification |
|---|---|---|
| `current_speed`, `steering`, `yaw_rate` | **A. Agent-Visible** | Proprioceptive vehicle sensors (encoders, IMU). |
| `dist_to_left_side`, `dist_to_right_side` | **A. Agent-Visible** | Local lane boundary state abstractions. |
| `lane_lateral_offset`, `heading_diff` | **A. Agent-Visible** | Lane-centering perception abstractions. |
| `navi_ckpt1_*`, `navi_ckpt2_*` | **A. Agent-Visible** | Local waypoint navigator outputs. |
| `lidar_cloud_points` (240 rays) | **A. Agent-Visible** | Uncompressed physical range sensor perception. |
| `surrounding_vehicles` (16D) | **B. Optional Public Context / Tracker** | If exposed, must include an explicit validity mask and be documented as a perception tracker abstraction rather than raw sensing. |
| `route_completion` | **C. Evaluator-Only Telemetry** | Global evaluation metric. Providing it to policies creates reward/progress hacking. |
| `arrive_dest`, `crash_*`, `out_of_road` | **C. Evaluator-Only Telemetry** | Objective benchmark evaluation flags. |
| `env_seed` | **C. Evaluator-Only Telemetry** | Scenario indexing for reproducible evaluation logging. |
| `vehicle.position` (global `(x, y)`) | **C. Evaluator-Only Telemetry** | Absolute world coordinates are privileged; agents operate in ego-centric local frames. |
| Internal map graphs / engine pointers | **C. Evaluator-Only Telemetry** | Raw simulator internals must remain strictly inaccessible to policies. |

### Proposed Open Candidate D: Structured AgentInput Concept (NON-FROZEN)
Rather than forcing all algorithms into a monolithic 275D flat vector where privileged tracks are invisibly mixed with raw rays, Candidate D proposes a structured container:
- **`CoreObservation`**: 259D uncompressed local state, navigation, and 240 LiDAR rays.
- **`TrafficContext`**: Up to $N$ tracked vehicle bounding boxes with relative $(x, y, v_x, v_y)$ and an explicit boolean `validity_mask` (eliminating the ambiguous `0.0` padding).
- **`TaskContext`**: A read-only interface providing reference path geometry and speed limits for planners.
- **`EvaluatorTelemetry`**: Private dictionary for benchmark metrics.

*Status: Open concept for Platform V1 consideration; NOT frozen and NOT declared winner.*

---

## 9. Native Action Semantics

- **Action Space:** Continuous `Box(-1.0, 1.0, shape=(2,), dtype=np.float32)`.
- **Vector Format:** `[steering, throttle_brake]`.
- **Steering (`action[0]`):** Continuous $[-1.0, 1.0]$ mapped to wheel angle $\delta = \text{action}[0] \times 60^\circ$. Negative is **LEFT**, positive is **RIGHT**, `0.0` is straight.
- **Throttle / Brake (`action[1]`):** Continuous $[-1.0, 1.0]$:
  - $\ge 0$: Engine tractive force $F_{\text{engine}} = \text{action}[1] \times \text{max\_engine_force}$.
  - $< 0$: Mechanical braking force $F_{\text{brake}} = |\text{action}[1]| \times \text{max\_brake_force}$.
  - $== 0$: Coasting / idle (zero engine force, rolling resistance only).

---

## 10. Discrete Mapping Semantics

MetaDrive provides native discrete action mapping in `metadrive/policy/env_input_policy.py`:
- Config keys: `discrete_action=True`, `discrete_steering_dim=5`, `discrete_throttle_dim=5`.
- Space: `Discrete(25)` ($5 \times 5$).
- **Conversion Formula:**
  $$\text{steering} = (\text{action} \pmod 5) \times 0.5 - 1.0$$
  $$\text{throttle} = (\text{action} // 5) \times 0.5 - 1.0$$
- candidate steering values: `[-1.0, -0.5, 0.0, 0.5, 1.0]`.
- candidate throttle values: `[-1.0, -0.5, 0.0, 0.5, 1.0]`.
- Perfectly symmetric and covers all four quadrants of the control rectangle.

---

## 11. Current CourseEnvV1 Discrete(5) Limitations

`CourseEnvV1` implements an exploratory 5-action discrete mapping:
```python
0: [-0.35,  0.35]  # LEFT
1: [ 0.00,  0.40]  # STRAIGHT
2: [ 0.35,  0.35]  # RIGHT
3: [ 0.00,  0.80]  # ACCELERATE
4: [ 0.00, -0.80]  # BRAKE
```

### Objective Limitations:
1. **Coupled Steering and Throttle:** Steering always forces $+0.35$ throttle.
2. **Missing Evasive Braking:** Cannot steer while decelerating.
3. **Missing Idle / Coasting:** No action provides $(0.0, 0.0)$.
4. **Locked Steering Magnitude:** Limited to $|\delta| = 0.35$; sharp turns are physically impossible.
5. **Asymmetric Envelope Coverage:** Restricts reachable vehicle dynamics.

---

## 12. Candidate Action Comparison

| Candidate Space | Type | Branching Factor | Expressiveness | Suitability for DQN | Suitability for PPO / SAC | Suitability for MCTS / Search |
|---|---|---|---|---|---|---|
| **A. CourseEnv Discrete(5)** | Discrete | Very Low (5) | Very Poor | Quick toy test only | Incompatible | High rollout depth, poor vehicle maneuverability |
| **B. Low-Branching Adapter (Discrete 9)** | Discrete | Low (9) | Moderate (steer $\in \{-0.6, 0, 0.6\}$, throttle $\in \{-0.8, 0, 0.6\}$) | Good | Incompatible | Balanced branching factor; primitives non-frozen |
| **C. MetaDrive Discrete(25)** | Discrete | Moderate (25) | High (covers full $[-1, 1]^2$ grid) | Good | Incompatible | Higher branching factor limits tree depth |
| **D. Native Continuous Box(2)** | Continuous | Infinite | Full ($\mathcal{C}^0$ envelope) | Incompatible without adapter | **Native / Ideal** | Requires action sampling / progressive widening |

---

## 13. Control Frequency Findings

Audited from `metadrive/envs/base_env.py` and empirical measurements:
- `physics_world_step_size`: $0.02\text{ s}$ ($50\text{ Hz}$ physics integration).
- `decision_repeat`: $5$ substeps per env step.
- Nominal step duration: $\Delta t = 0.02\text{ s} \times 5 = 0.10\text{ s}$ ($10\text{ Hz}$ decision frequency).
- **Empirical Confirmation:** $\Delta \text{pos} / v_{\text{avg}} = 0.0955\text{ s}$ at steady cruise.
- $10\text{ Hz}$ is well-conditioned for RL (PPO/SAC) and world models, while search algorithms will require compact discrete primitives or sampling.

---

## 14. Action-Response Calibration Experiments

All calibration tests were executed on a deterministic straight road (`map='S'`, `traffic_density=0.0`).

### 1. Symmetric Early Steering Calibration (15 Steps Horizon)
To eliminate road-boundary spawn asymmetry from previous tests, steering response was evaluated over an identical 15-step duration ($1.5\text{ s}$) before reaching road edges:

| Action Label | Steering Command | Throttle Command | Speed at Step 15 | Heading Change | Lateral Offset | Symmetry Check |
|---|---|---|---|---|---|---|
| `slight_left_15steps` | `-0.25` | `0.40` | $5.92\text{ km/h}$ | $-4.582^\circ$ | $+0.162\text{ m}$ | Identical magnitude to right |
| `slight_right_15steps`| `+0.25` | `0.40` | $5.92\text{ km/h}$ | $+4.582^\circ$ | $-0.162\text{ m}$ | Identical magnitude to left |
| `hard_left_15steps` | `-0.50` | `0.40` | $5.73\text{ km/h}$ | $-8.934^\circ$ | $+0.316\text{ m}$ | Identical magnitude to right |
| `hard_right_15steps` | `+0.50` | `0.40` | $5.73\text{ km/h}$ | $+8.934^\circ$ | $-0.316\text{ m}$ | Identical magnitude to left |

*Finding:* The vehicle actuator responds with **exact bilateral symmetry** in speed, heading rate, and lateral displacement. Earlier asymmetric termination (22 vs 49 steps) was purely an artifact of unequal distance from the spawn lane to the left versus right road edges.

### 2. Controlled Braking Calibration (Equal Initial Pre-Brake Speed)
The vehicle was accelerated for 20 steps at full throttle (`[0.0, 1.0]`) to reach an identical pre-brake speed of **$19.88\text{ km/h}$**, followed by independent brake commands until stopped ($v < 0.1\text{ km/h}$):

| Test Label | Brake Command | Pre-Brake Speed | Stopping Steps | Stopping Time | Stopping Distance |
|---|---|---|---|---|---|
| `controlled_brake_25pct` | `throttle_brake = -0.25` | $19.88\text{ km/h}$ | 14 steps | $1.40\text{ s}$ | $3.92\text{ m}$ |
| `controlled_brake_50pct` | `throttle_brake = -0.50` | $19.88\text{ km/h}$ | 10 steps | $1.00\text{ s}$ | $2.06\text{ m}$ |
| `controlled_brake_100pct`| `throttle_brake = -1.00` | $19.88\text{ km/h}$ | 9 steps | $0.90\text{ s}$ | $1.48\text{ m}$ |

*Finding:* Mechanical braking exhibits clean monotonic deceleration and stopping distance scaling with brake magnitude.

*Machine-readable evidence:* Summary rows for all calibration tests are recorded in `results/audits/observation_action/control_response.csv`.

---

## 15. Determinism & Repeatability Findings

Evaluated across repeated independent executions on multiple scenario seeds (seeds 42 and 101) using a fixed multi-action sequence:
- **Full Observation Vector Discrepancy:** $0.00\times 10^0$ (exact element-wise match across all 275 dimensions).
- **Reward Discrepancy:** $0.00\times 10^0$.
- **Position Discrepancy:** $0.00\times 10^0\text{ m}$.
- **Heading Discrepancy:** $0.00\times 10^0\text{ rad}$.
- **Velocity Discrepancy:** $0.00\times 10^0\text{ km/h}$.
- **Route Completion Discrepancy:** $0.00\times 10^0$.
- **Termination / Truncation Agreement:** 100% identical.

### Determinism Claim Formulation
> **Verified Finding:** Exact repeatability was observed under the tested configuration.  
> *Scope Limitation:* This evidence confirms single-platform repeatability within the pinned software environment. It does not assert universal cross-hardware or cross-platform determinism across diverse OS/CPU architectures.

### Decoupling Scenario Seed from Agent Stochasticity
The platform benchmark harness must formally separate:
1. **Scenario Seed (`env_seed`):** Determines road geometry, procedural block layout, and initial traffic placement.
2. **Agent RNG Seed:** Controls policy stochasticity, action noise, and dropout perturbations.

*Comparative step records are saved in `results/audits/observation_action/determinism.csv`.*

---

## 16. Risks

1. **Privileged State Contamination:** Baking 16 surrounding vehicle values directly into a flat observation vector leaks occlusion-free physics state to policies.
2. **Ray Downsampling Blindness:** Irreversibly compressing 240 rays into 16 sectors prevents rule agents and planners from identifying narrow drivable gaps.
3. **Actuator Space Fragmentation:** Disparate action contracts between algorithms (e.g. Discrete(5) for DQN vs Box(2) for PPO) undermine fair scientific benchmark comparison across Stages 0–7.
4. **Dropped Ray Misclassification:** With `dropout_prob > 0`, dropped rays read as `0.0` (immediate collision), which could trigger false-positive emergency braking in heuristic or search policies.

---

## 17. Open Questions

1. **Surrounding Vehicles Contract:** Should surrounding vehicles be exposed as an optional structured tracker (`TrafficContext` with validity mask) or kept strictly out of the default policy interface?
2. **Discrete Primitives Calibration:** For discrete planning and RL, what exact steering and throttle discrete primitives provide optimal coverage without excessive branching?
3. **Canonical Benchmark Noise:** Should default evaluation runs feature clean LiDAR (`noise=0.0`, `dropout=0.0`) with noise reserved for robustness stress tests?
4. **Lane Line / Side Detector Inclusions:** Should static road border point clouds (`side_detector`, `lane_line_detector`) be evaluated in Platform V1?

---

## 18. Recommendations — Explicitly Non-Frozen

> **NOTICE:** All recommendations below are non-binding proposals resulting from this Gate 1 audit. **No specifications or contracts are frozen in this task.**

1. **Canonical Observation Recommendation:**
   - Retain **Candidate A (259D)** as the uncompressed root observation contract emitted by the environment.
   - Maintain **Candidate D (Structured AgentInput Concept)** as an active candidate separating CoreObservation, optional TrafficContext, and TaskContext.
   - Prohibit freezing a 35D compressed observation at the environment layer.
2. **Canonical Action Recommendation:**
   - Adopt native continuous **`Box(-1.0, 1.0, shape=(2,), dtype=np.float32)`** as the foundational actuator interface for all stages.
   - Provide an optional, certified discrete adapter (e.g. `Discrete(25)` or `candidate_low_branching_adapter_9`) for discrete-only methods.
3. **Telemetry & Task Context:**
   - Strictly sequester `route_completion`, collision flags, and raw world coordinates in `info` as evaluator-only telemetry.
   - Develop a read-only `TaskContext` / `MapContext` specification for global planning algorithms.
4. **Control Frequency Contract:**
   - Preserve `physics_world_step_size=0.02` and `decision_repeat=5` ($10\text{ Hz}$ agent cycle).
