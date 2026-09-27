# Platform V1 Observation, Action, and Control Contract Audit

> **Status:** AUDIT & DESIGN RECOMMENDATION (GATE 1)  
> **Freeze Status:** NON-FROZEN — RECOMMENDATIONS ONLY  
> **Target Scope:** Universal compatibility across Stages 0 through 7  
> **Date:** September 2026  

---

## 1. Source-of-Truth Verification

The authoritative reference for this audit is the local pinned MetaDrive source tree:

- **Local Path:** `D:\SGU\CNTT\TTNTNC\metadrive-src`
- **Pinned Git Commit:** `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`
- **Package Name:** `metadrive-simulator`
- **Package Version:** `0.4.3` (`metadrive.constants.VERSION == "0.4.3"`)
- **Package Location:** Verified resolving to `D:\SGU\CNTT\TTNTNC\metadrive-src\metadrive\__init__.py`
- **Git Tree Status:** Clean, verified via `git rev-parse HEAD` and `git status`. No source modifications made.

---

## 2. Observation Source Audit

The MetaDrive observation pipeline was audited from source code in:
- `metadrive/obs/state_obs.py` (`StateObservation`, `LidarStateObservation`)
- `metadrive/component/navigation_module/node_network_navigation.py` (`NodeNetworkNavigation`)
- `metadrive/component/vehicle/base_vehicle.py` (`BaseVehicle`)
- `metadrive/component/sensors/lidar.py` (`Lidar`)
- `metadrive/component/sensors/distance_detector.py` (`DistanceDetector`)

### Critical Source Discrepancies Uncovered

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

2. **LiDAR Angular Sweep Direction Inversion:**
   - In `state_obs.py` line 199, the docstring states:
     `"starting from the vehicle head in clockwise direction"`.
   - The actual implementation in `DistanceDetector._get_lidar_range` and `get_laser_end` is:
     $$\text{angle} = \text{heading\_theta} + i \times \frac{2\pi}{N}$$
   - In standard 2D Cartesian vehicle coordinates (+x forward, +y left), an increasing positive angle rotates from +x toward +y, which is **COUNTER-CLOCKWISE**.
   - Experimental obstacle placements at 0°, +90°, 180°, and 270° produced hits at ray 0 (Front), ray 60 (Left), ray 120 (Rear), and ray 180 (Right). The sweep is mathematically and empirically **counter-clockwise**.

---

## 3. Exact Observation Index Table

The authoritative field-level definition for the candidate 275D observation (`num_others=4`) and default 259D observation (`num_others=0`) is detailed below:

| Index / Slice | Field Name | Semantic Meaning | Raw Units | Normalization Formula | Neutral / Zero Value | Privileged State? | Map Stability |
|---|---|---|---|---|---|---|---|
| `[0]` | `dist_to_left_side` | Distance to left road curb/sidewalk | meters | `clip(dist / total_width, 0, 1)` where `total_width = (MAX_LANE_NUM+1)*MAX_LANE_WIDTH` | `0.0` = touching/beyond left border | No | Depends on map width definition |
| `[1]` | `dist_to_right_side` | Distance to right road curb/sidewalk | meters | `clip(dist / total_width, 0, 1)` | `0.0` = touching/beyond right border | No | Depends on map width definition |
| `[2]` | `heading_diff` | Heading difference relative to current reference lane | radians | `clip(cos(heading, lane_lat), -1, 1)/2 + 0.5` | `0.5` = perfectly aligned with lane heading; `1.0` = +90°, `0.0` = -90° | No | Stable |
| `[3]` | `current_speed` | Vehicle forward velocity | km/h | `clip((speed_km_h + 1)/(max_speed_km_h + 1), 0, 1)` | `~0.0123` = stationary (0 km/h with 80 km/h max) | No | Stable |
| `[4]` | `current_steering` | Front wheel turn angle | degrees | `clip((steering / MAX_STEERING + 1)/2, 0, 1)` | `0.5` = wheels straight; `0.0` = max left, `1.0` = max right | No | Stable |
| `[5]` | `last_action_steering` | Steering input applied at last step | normalized `[-1, 1]` | `clip((action[0] + 1)/2, 0, 1)` | `0.5` = zero steering command | No | Stable |
| `[6]` | `last_action_throttle` | Throttle/brake input applied at last step | normalized `[-1, 1]` | `clip((action[1] + 1)/2, 0, 1)` | `0.5` = idle/coast command (0.0 continuous) | No | Stable |
| `[7]` | `yaw_rate` | Vehicle angular velocity magnitude | rad/s | `clip(arccos(clip(h_now . h_last, 0, 1))/0.1, 0, 1)` | `0.0` = zero angular turn rate (straight line) | No | Stable |
| `[8]` | `lane_lateral_offset` | Offset from current lane centerline | meters | `clip((lateral * 2 / max_lane_width + 1.0)/2.0, 0, 1)` | `0.5` = centered on lane; `<0.5` right of center, `>0.5` left of center | No | Stable |
| `[9]` | `navi_ckpt1_long` | Longitudinal distance to Checkpoint 1 | meters | `clip((dx / 50.0 + 1)/2, 0, 1)` | `0.5` = checkpoint is laterally aligned | No | Stable |
| `[10]` | `navi_ckpt1_lat` | Lateral distance to Checkpoint 1 | meters | `clip((dy / 50.0 + 1)/2, 0, 1)` | `0.5` = checkpoint directly ahead/behind | No | Stable |
| `[11]` | `navi_ckpt1_bend_radius` | Bending radius of lane leading to Checkpoint 1 | normalized | `clip(radius / (CURVE_MAX + lane_num * lane_width), 0, 1)` | `0.0` = straight lane | No | Stable |
| `[12]` | `navi_ckpt1_bend_dir` | Curvature direction of lane to Checkpoint 1 | sign | `clip((-ref_lane.direction + 1)/2, 0, 1)` | `0.5` = straight lane; `1.0` clockwise, `0.0` counter-clockwise | No | Stable |
| `[13]` | `navi_ckpt1_lane_angle` | Total angle change of lane to Checkpoint 1 | degrees | `clip((deg(angle)/ANGLE_MAX + 1)/2, 0, 1)` | `0.5` = straight lane | No | Stable |
| `[14]` | `navi_ckpt2_long` | Longitudinal distance to Checkpoint 2 | meters | `clip((dx / 50.0 + 1)/2, 0, 1)` | `0.5` = checkpoint is laterally aligned | No | Stable |
| `[15]` | `navi_ckpt2_lat` | Lateral distance to Checkpoint 2 | meters | `clip((dy / 50.0 + 1)/2, 0, 1)` | `0.5` = checkpoint directly ahead/behind | No | Stable |
| `[16]` | `navi_ckpt2_bend_radius` | Bending radius of lane leading to Checkpoint 2 | normalized | `clip(radius / (CURVE_MAX + lane_num * lane_width), 0, 1)` | `0.0` = straight lane | No | Stable |
| `[17]` | `navi_ckpt2_bend_dir` | Curvature direction of lane to Checkpoint 2 | sign | `clip((-ref_lane.direction + 1)/2, 0, 1)` | `0.5` = straight lane | No | Stable |
| `[18]` | `navi_ckpt2_lane_angle` | Total angle change of lane to Checkpoint 2 | degrees | `clip((deg(angle)/ANGLE_MAX + 1)/2, 0, 1)` | `0.5` = straight lane | No | Stable |
| `[19:35]` | `surrounding_vehicles` | 4 nearest detected vehicles (4 values each: `rel_x`, `rel_y`, `rel_vx`, `rel_vy`) | m, km/h | `(pos/50.0 + 1)/2`, `(vel/max_speed + 1)/2` | **`0.0` = absent/padded vehicle**; `0.5` = co-located & zero relative speed | **YES (Privileged)** | Dynamic traffic |
| `[35:275]` (or `[19:259]`) | `lidar_cloud_points` | 240 LiDAR distance rays sweeping 360° counter-clockwise | fraction of 50m | `hit_distance / 50.0` | `1.0` = no obstacle / max range (50m); `0.0` = contact or dropped ray | No | Dynamic & static objects |

---

## 4. LiDAR Semantics

- **Number of Rays:** 240 rays.
- **Maximum Range:** 50.0 meters (`distance=50.0`).
- **Ray Angular Resolution:** $360^\circ / 240 = 1.5^\circ$ per ray.
- **Ray 0 Orientation:** $0^\circ$, pointing straight ahead along the vehicle's heading vector (`[cos(heading), sin(heading)]`).
- **Ray Angular Ordering:** Counter-Clockwise around the vehicle:
  - Ray 0 ($0^\circ$): Front / Heading
  - Ray 60 ($90^\circ$): Direct Left
  - Ray 120 ($180^\circ$): Direct Rear
  - Ray 180 ($270^\circ$ or $-90^\circ$): Direct Right
  - Ray 239 ($358.5^\circ$): Right-Front
- **Normalization Convention:** Normalized hit fraction $\in [0.0, 1.0]$:
  - `1.0`: No obstacle detected within 50.0 m (clear path).
  - `0.0`: Immediate contact ($0.0\text{ m}$) or ray dropped by noise.
  - Intermediate values represent distance $d = \text{val} \times 50.0\text{ m}$.
- **Static vs Dynamic Interaction:**
  - **Dynamic Vehicles:** Detected when within line of sight.
  - **Static Traffic Objects:** Cones (`TrafficCone`), barriers (`TrafficBarrier`), and warning signs are in `physics_world.dynamic_world`, and therefore **ARE detected**.
  - **Road Boundaries & Sidewalks:** Road geometry is in `physics_world.static_world`. Standard MetaDrive `Lidar` queries only `dynamic_world`. Therefore, **road boundaries, lane markings, and sidewalks ARE NOT VISIBLE to LiDAR**. (Road edges are only sensed via `dist_to_left_side` / `dist_to_right_side` from `side_detector`).
- **Sensor Noise Effects:**
  - `gaussian_noise`: Adds zero-mean Gaussian perturbation $\mathcal{N}(0, \sigma^2)$ to hit fractions, clipped to $[0.0, 1.0]$.
  - `dropout_prob`: Randomly sets hit fractions to `0.0`. Notice this creates semantic ambiguity: a dropped ray reads as an obstacle at $0\text{ m}$ distance.

---

## 5. Surrounding-Vehicle Semantics (`num_others = 4`)

When `vehicle_config["lidar"]["num_others"] = 4`:
1. **Observation Dimensions:**
   - `num_others = 0`: 259D ($9 + 10 + 240$)
   - `num_others = 1`: 263D ($9 + 10 + 4 + 240$)
   - `num_others = 4`: 275D ($9 + 10 + 16 + 240$)
2. **Vehicle Ordering:**
   - Sorted ascending by Euclidean distance: $\sqrt{(x_{\text{ego}} - x_v)^2 + (y_{\text{ego}} - y_v)^2}$.
   - The closest 4 vehicles within 50 meters are selected.
3. **Four Features per Vehicle:**
   - `rel_longitudinal_pos`: `clip((dx / 50.0 + 1) / 2, 0.0, 1.0)`
   - `rel_lateral_pos`: `clip((dy / 50.0 + 1) / 2, 0.0, 1.0)`
   - `rel_longitudinal_vel`: `clip((dvx / max_speed + 1) / 2, 0.0, 1.0)`
   - `rel_lateral_vel`: `clip((dvy / max_speed + 1) / 2, 0.0, 1.0)`
4. **Padding Semantics:**
   - If fewer than 4 vehicles exist within 50 meters (or if `traffic_density == 0.0`), absent vehicle slots are padded with `[0.0, 0.0, 0.0, 0.0]`.
   - **Critical Semantic Distinction:** A detected vehicle matched in position and velocity with ego yields normalized values of `[0.5, 0.5, 0.5, 0.5]`. An absent vehicle yields `[0.0, 0.0, 0.0, 0.0]`.
5. **Privileged State Leakage Analysis:**
   - **Physics World Query:** Surrounding vehicles are gathered via `get_surrounding_objects()` using a `BulletGhostNode` cylinder of radius 50m.
   - **Occlusion Bypass:** Vehicles completely occluded behind other vehicles or obstacles are still included if within the 50m radius.
   - **Direct Simulation Coordinates:** Ground truth position and velocity are extracted directly from simulator internals (`vehicle.position`, `vehicle.velocity_km_h`).
   - **Conclusion:** These 16 features constitute **privileged simulator state**, not raw physical sensor perception.

---

## 6. Observation Candidate Comparison

We evaluate three observation candidates:
- **Candidate A (259D):** MetaDrive default raw state + 240 LiDAR rays.
- **Candidate B (275D):** MetaDrive raw state + navigation + 4 surrounding vehicles (16D) + 240 LiDAR rays.
- **Candidate C (35D):** CourseEnv experimental representation (9 ego + 10 navigation + 16 min-pooled LiDAR sectors).

### Comparison Matrix

| Criterion | Candidate A (259D) | Candidate B (275D) | Candidate C (35D) |
|---|---|---|---|
| **Information Sufficiency** | Moderate (spatial geometry present, but obstacle velocity must be inferred across frames) | High (explicit bounding-box relative positions and velocities) | Low (coarse 16-sector spatial pooling, no dynamic velocity) |
| **Dimensionality** | Moderate (259D vector) | Moderate (275D vector) | Low (35D compact vector) |
| **Algorithm Neutrality** | High (pure sensor + navigation signals) | Medium (contains privileged state assumptions) | Low (bakes in specific min-pooling heuristic) |
| **Dynamic Object Velocity** | Requires frame stacking / recurrence (LSTM/GRU/Transformer) | Explicitly provided in observation | Missing; requires frame stacking |
| **Ability to Support Learning (RL/IL)** | Strong; standard for MetaDrive benchmarks | Strong; faster sample efficiency due to explicit traffic state | Good for fast toy prototyping, poor for advanced maneuvers |
| **Ability to Support Planning/Search** | Difficult to extract obstacle positions without ray triangulation | Highly convenient (structured bounding-box inputs) | Unusable for collision-free trajectory optimization |
| **Risk of Information Loss** | Minimal | None | **High** (averages/min-pools 15 rays into 1 sector, losing angular resolution) |
| **Privileged State Leakage** | Low (only ego telemetry and map checkpoints) | **High** (direct physics-world query of other vehicles) | Low (only aggregated sectors) |
| **Computational Cost** | Low | Low | Very Low |

---

## 7. Stage 0–7 Compatibility Matrix

| Project Stage | Candidate A (259D) | Candidate B (275D) | Candidate C (35D) | Evaluation & Architecture Recommendation |
|---|---|---|---|---|
| **Stage 0: Random Baseline** | Compatible | Compatible | Compatible | Any candidate works; 35D has no advantage beyond marginally smaller memory. |
| **Stage 1: Rule / Heuristic** | Difficult | **Highly Suitable** | Partially Suitable | Rule agents (e.g. IDM, TTC heuristics) require obstacle distance and relative velocity. 275D provides this directly; 35D lacks velocity; 259D requires ray tracking. |
| **Stage 2: Planning / Search** | Impractical | **Highly Suitable** | Insufficient | Planners (A*, lattice planners) need obstacle state coordinates to check bounding boxes against candidate paths. 275D gives structured states; 35D is too coarse. |
| **Stage 3: Simulation-Based Planning** | Impractical | **Highly Suitable** | Insufficient | MCTS / forward rollout search requires structured obstacle positions for forward state rollout collision checks. |
| **Stage 4: Imitation Learning** | Suitable | **Highly Suitable** | Degraded | Expert demonstrations driving among traffic correlate strongly with surrounding vehicle relative velocities. 35D loses fine guidance. |
| **Stage 5: Model-Free RL (PPO/SAC)** | **Highly Suitable** | Suitable (with caution on privileged leakage) | Suitable for fast baseline | PPO/SAC learn effectively on both 259D and 275D. 275D converges faster in multi-vehicle traffic. |
| **Stage 6: Search + Learning** | Difficult for search part | **Highly Suitable** | Insufficient | Hybrid architectures require the search component to have structured geometric entities. |
| **Stage 7: Model-Based RL / World Models** | **Suitable** | Suitable | Degraded | World models (Dreamer-style) can model 259D or 275D. In 259D, latent state reconstructs LiDAR rays; in 275D, dynamics predict surrounding vehicle tracks. |

### Evaluation of the Canonical Observation Architecture Principle

> **Design Principle:** The canonical environment exposes the richer raw observation contract, while individual agents or adapter wrappers apply deterministic transformations.

**Conclusion:** **CONFIRMED VALID.**  
If the platform freezes a 35D observation at the environment root, Stage 1 (Rules), Stage 2 (Planning), and Stage 3 (Look-ahead) are permanently impaired because ray pooling irreversibly destroys obstacle angular resolution and dynamic velocities cannot be recovered. Conversely, an agent desiring a 35D representation can deterministically downsample a 275D observation in microseconds.

---

## 8. Agent-Visible Observation vs. Privileged Telemetry Proposal

To maintain strict scientific integrity across learning and search benchmarks, the platform must formalize an information boundary:

```text
+--------------------------------------------------------------------------------+
|                             SIMULATOR STATE                                    |
+--------------------------------------------------------------------------------+
       |                                                  |
       v                                                  v
[AGENT-VISIBLE INTERFACE]                       [EVALUATOR / PRIVILEGED INTERFACE]
- Ego kinematics (speed, steering, yaw)          - Ground-truth global trajectory
- Local lane offsets & heading difference        - Route completion percentage
- Waypoint / checkpoint guidance (local)         - Collision flags (crash_vehicle, crash_object)
- 240-ray LiDAR sensor data                      - Out-of-road termination signals
- [Optional] Read-only MapContext / TrackContext  - Simulator internal seeds & object pointers
```

### Proposed Classification of Telemetry Fields

| Field / Concept | Classification | Justification |
|---|---|---|
| `current_speed`, `steering`, `yaw_rate` | **A. Agent-Visible** | Realistic on-board proprioceptive sensors (wheel encoders, IMU). |
| `dist_to_left_side`, `dist_to_right_side` | **A. Agent-Visible** | Ultrasonic / short-range lane boundary sensors. |
| `lane_lateral_offset`, `heading_diff` | **A. Agent-Visible** | Standard lane-detection camera / lane centering output. |
| `navi_ckpt1_*`, `navi_ckpt2_*` | **A. Agent-Visible** | GPS / local navigation path planner output. |
| `lidar_cloud_points` (240 rays) | **A. Agent-Visible** | Legitimate active range sensor. |
| `surrounding_vehicles` (16D) | **B. Optional Public Context / Track Filter** | Can be viewed as an onboard perception module (bounding-box tracker) OR as privileged state. Must be documented if used. |
| `route_completion` | **C. Evaluator-Only Telemetry** | Global evaluation metric. If fed to the policy, the agent could exploit progress hacking. |
| `arrive_dest`, `crash_*`, `out_of_road` | **C. Evaluator-Only Telemetry** | Objective evaluation outcomes and termination monitors. |
| `env_seed` | **C. Evaluator-Only Telemetry** | Scenario indexing for reproducible evaluation logging. |
| `vehicle.position` (global `(x, y)`) | **C. Evaluator-Only Telemetry** | Absolute world coordinates are privileged. Agents should operate in ego-centric coordinates. |
| `vehicle.navigation.map` graph / internal nodes | **C. Evaluator-Only Telemetry** | Raw simulator internal objects must not be modified by agents. |

### Proposed Read-Only `MapContext` Interface (Future Concept)
For global planning and search algorithms (Stages 2, 3, 6) requiring road topology without granting backdoor access to the simulator engine:
- A frozen, read-only data transfer object: `MapContext(lanes: Sequence[LaneGeometry], reference_path: Polyline, speed_limit: float)`.
- Available equally to all algorithms through a standardized public getter (`env.get_task_context()`), ensuring fair evaluation.

---

## 9. Native Action Semantics

From `metadrive/component/vehicle/base_vehicle.py` (`_set_action`, `_apply_throttle_brake`):
- Action space: `Box(-1.0, 1.0, shape=(2,), dtype=np.float32)`.
- Vector format: `[steering, throttle_brake]`.
- **Steering (`action[0]`):**
  - Continuous range: $[-1.0, 1.0]$.
  - Scales wheel steering angle: $\theta_{\text{steer}} = \text{action}[0] \times \text{max\_steering}$ (where $\text{max\_steering} = 60^\circ$).
  - Negative values steer **LEFT**; positive values steer **RIGHT**; `0.0` is straight.
- **Throttle / Brake (`action[1]`):**
  - Continuous range: $[-1.0, 1.0]$.
  - When $\text{action}[1] \ge 0$: Applies engine force $F_{\text{engine}} = \text{max\_engine_force} \times \text{action}[1]$ across wheels (capped if speed exceeds `max_speed_km_h`). Wheel brake is set to a nominal rolling resistance of $2.0\text{ N}$.
  - When $\text{action}[1] < 0$: Engine force is $0.0$; applies wheel brake torque $F_{\text{brake}} = |\text{action}[1]| \times \text{max\_brake_force}$.
  - When $\text{action}[1] == 0$: Coasting (no engine force, minimal rolling brake).

---

## 10. Discrete Mapping Semantics

MetaDrive provides native discrete action mapping in `metadrive/policy/env_input_policy.py`:
- Config keys: `discrete_action=True`, `discrete_steering_dim=5`, `discrete_throttle_dim=5`.
- Space: `Discrete(25)` ($5 \times 5$).
- **Conversion Formula:**
  $$\text{steering\_unit} = \frac{2.0}{\text{discrete\_steering\_dim} - 1} = \frac{2.0}{4} = 0.5$$
  $$\text{throttle\_unit} = \frac{2.0}{\text{discrete\_throttle\_dim} - 1} = \frac{2.0}{4} = 0.5$$
  $$\text{steering} = (\text{action} \pmod 5) \times 0.5 - 1.0$$
  $$\text{throttle} = (\text{action} // 5) \times 0.5 - 1.0$$

### Discrete(25) Action Lookup Table

| Index | Steering Index | Throttle Index | Continuous Steering | Continuous Throttle | Semantic Meaning |
|---|---|---|---|---|---|
| `0` | 0 | 0 | -1.0 | -1.0 | Full Left, Full Brake |
| `2` | 2 | 0 | 0.0 | -1.0 | Straight, Full Brake |
| `4` | 4 | 0 | +1.0 | -1.0 | Full Right, Full Brake |
| `10` | 0 | 2 | -1.0 | 0.0 | Full Left, Coast |
| `12` | 2 | 2 | 0.0 | 0.0 | Straight, Coast / Idle |
| `14` | 4 | 2 | +1.0 | 0.0 | Full Right, Coast |
| `20` | 0 | 4 | -1.0 | +1.0 | Full Left, Full Throttle |
| `22` | 2 | 4 | 0.0 | +1.0 | Straight, Full Throttle |
| `24` | 4 | 4 | +1.0 | +1.0 | Full Right, Full Throttle |

---

## 11. Current CourseEnvV1 Discrete(5) Shortcomings

`CourseEnvV1` implements an exploratory 5-action discrete mapping:
```python
0: [-0.35,  0.35]  # LEFT
1: [ 0.00,  0.40]  # STRAIGHT
2: [ 0.35,  0.35]  # RIGHT
3: [ 0.00,  0.80]  # ACCELERATE
4: [ 0.00, -0.80]  # BRAKE
```

### Objective Shortcoming Analysis:
1. **Coupled Steering and Throttle:** Commands 0 and 2 automatically command $+0.35$ throttle. The agent cannot steer without simultaneously accelerating.
2. **Missing Evasive Braking:** There is no command for steering while braking (e.g., swerving while slowing down to avoid an obstacle).
3. **Missing Idle / Coasting:** No action provides $(0.0, 0.0)$. The vehicle is either forced into heavy acceleration, cruising, or abrupt braking.
4. **Restricted Steering Magnitude:** Steering is locked at $|\delta| = 0.35$. Hard cornering or fast evasive swerving requiring $|\delta| \in [0.6, 1.0]$ is physically impossible.
5. **Asymmetric Coverage:** The actuator envelope covers only a tiny subset of the reachable control space, creating unfairness for algorithms that could otherwise find optimal trajectory plans.

---

## 12. Candidate Action Comparison

| Candidate Space | Type | Branching Factor | Expressiveness | Suitability for DQN | Suitability for PPO / SAC | Suitability for MCTS / Search |
|---|---|---|---|---|---|---|
| **A. CourseEnv Discrete(5)** | Discrete | Very Low (5) | Very Poor | Good for quick toy test | Incompatible (requires discrete policy head) | High rollout depth, poor vehicle maneuverability |
| **B. Symmetric Discrete(9)** ($3 \times 3$) | Discrete | Low (9) | Moderate (steer $\in \{-0.6, 0, 0.6\}$, throttle $\in \{-0.8, 0, 0.6\}$) | Good | Incompatible (requires discrete policy head) | Good balance of search horizon and independent control |
| **C. MetaDrive Discrete(25)** ($5 \times 5$) | Discrete | Moderate (25) | High (covers full $[-1, 1]^2$ grid) | Good | Incompatible without discretization wrapper | High branching factor (limits tree depth) |
| **D. Native Continuous Box(2)** | Continuous | Infinite | Full ($\mathcal{C}^0$ envelope) | Incompatible (requires discretization) | **Native / Ideal** | Requires action sampling / progressive widening |

---

## 13. Control Frequency Findings

Audited from `metadrive/envs/base_env.py` and empirical stepping:
- `physics_world_step_size`: $0.02\text{ s}$ ($50\text{ Hz}$ physics integration).
- `decision_repeat`: $5$ physics substeps per agent decision step.
- Nominal agent step duration:
  $$\Delta t = 0.02\text{ s} \times 5 = 0.10\text{ s} \implies 10\text{ Hz decision frequency}$$
- **Empirical Measurement:**  
  At steady speed, measured $\Delta \text{pos} / v_{\text{avg}} = 0.0955\text{ s}$ (within expected numerical tolerance of discrete physics integration).

### Implications Across Stages:
- **Rule / PID Control:** $10\text{ Hz}$ is adequate for highway and urban cruise control, but aggressive dynamic stabilization requires smooth PID output.
- **Search Depth (MCTS / Lattice):** At $10\text{ Hz}$, a 2-second planning horizon requires 20 steps. At branching factor $b=25$, exhaustive search $25^{20}$ is impossible; sampling-based search or $b \le 9$ is necessary.
- **Model-Free RL (PPO/SAC):** $10\text{ Hz}$ is the standard, well-conditioned decision rate for MetaDrive. Transitions are informative without extreme step latency.
- **World Models:** $0.1\text{ s}$ transitions provide sufficiently significant state deltas for latent transition predictors (e.g. RSSM) without suffering from single-step vanishing deltas.

---

## 14. Action-Response Calibration Experiments

All experiments were executed on a deterministic straight road (`map='S'`, `traffic_density=0.0`) starting from a stationary vehicle at spawn.

### Calibration Data Summary (50 Steps / 5.0 Seconds Duration)

| Test Label | Action `[steer, throttle]` | Steps | Final Speed | Distance Traveled | Heading Change | Lateral Offset | Termination |
|---|---|---|---|---|---|---|---|
| `idle` | `[0.0, 0.0]` | 50 | $0.0\text{ km/h}$ | $0.0\text{ m}$ | $0.0^\circ$ | $0.00\text{ m}$ | None |
| `low_throttle` | `[0.0, 0.25]` | 50 | $12.4\text{ km/h}$ | $8.5\text{ m}$ | $0.0^\circ$ | $0.00\text{ m}$ | None |
| `mid_throttle` | `[0.0, 0.50]` | 50 | $24.9\text{ km/h}$ | $17.1\text{ m}$ | $0.0^\circ$ | $0.00\text{ m}$ | None |
| `full_throttle` | `[0.0, 1.00]` | 50 | $49.7\text{ km/h}$ | $34.1\text{ m}$ | $0.0^\circ$ | $0.00\text{ m}$ | None |
| `gentle_brake` | `[0.0, -0.25]` | 50 | $0.0\text{ km/h}$ | $0.0\text{ m}$ | $0.0^\circ$ | $0.00\text{ m}$ | None |
| `mid_brake` | `[0.0, -0.50]` | 50 | $0.0\text{ km/h}$ | $0.0\text{ m}$ | $0.0^\circ$ | $0.00\text{ m}$ | None |
| `full_brake` | `[0.0, -1.00]` | 50 | $0.0\text{ km/h}$ | $0.0\text{ m}$ | $0.0^\circ$ | $0.00\text{ m}$ | None |
| `slight_left_throttle` | `[-0.25, 0.40]` | 49 | $19.2\text{ km/h}$ | $12.5\text{ m}$ | $-51.3^\circ$ | $-0.59\text{ m}$ | Out of Road |
| `hard_left_throttle` | `[-0.50, 0.40]` | 42 | $15.5\text{ km/h}$ | $8.5\text{ m}$ | $-72.2^\circ$ | $-0.78\text{ m}$ | Out of Road |
| `slight_right_throttle`| `[0.25, 0.40]` | 22 | $8.7\text{ km/h}$ | $2.6\text{ m}$ | $+10.1^\circ$ | $-0.47\text{ m}$ | Out of Road |
| `hard_right_throttle` | `[0.50, 0.40]` | 17 | $6.5\text{ km/h}$ | $1.5\text{ m}$ | $+11.6^\circ$ | $-0.44\text{ m}$ | Out of Road |
| `accel20_then_full_brake` | Accel 20 steps $\to$ full brake | $20 + 9$ | $0.0\text{ km/h}$ | Stopping dist: $1.48\text{ m}$ | $0.0^\circ$ | $0.00\text{ m}$ | Stopped in 0.9s from $19.9\text{ km/h}$ |

*Full step-by-step telemetry saved in `results/audits/observation_action/control_response.csv`.*

---

## 15. Determinism Findings

Tested across 3 independent environment instances with identical scenario seed (`seed=42`) and identical non-trivial action sequence (30 steps):
- **Position Discrepancy (Run 0 vs Run 1):** $0.0000000000\text{ m}$ (Exact float match).
- **Position Discrepancy (Run 0 vs Run 2):** $0.0000000000\text{ m}$ (Exact float match).
- **Heading Discrepancy:** $0.0000000000\text{ rad}$.
- **Velocity Discrepancy:** $0.0000000000\text{ km/h}$.
- **Route Completion Discrepancy:** $0.0000000000$.

### Concept Separation: Scenario Seed vs. Agent RNG Seed
- **Scenario Seed (`env_seed`):** Determines road geometry, procedural block generation, obstacle placements, and initial traffic vehicle spawns.
- **Agent RNG Seed:** Governs stochastic action sampling, policy exploratory noise, and dropout perturbations.
- **Platform V1 Requirement:** The benchmark harness must explicitly decouple these two concepts to ensure that agent comparisons evaluate policy capability across identical scenario seeds.

*Full comparative run data saved in `results/audits/observation_action/determinism.csv`.*

---

## 16. Risks

1. **Silent Privileged State Leakage:** If `num_others=4` is enabled in candidate observations, policies will train on physics-world cheat data (occlusion-free coordinates and velocities) that are unrepresentative of physical sensors.
2. **LiDAR Sector Compression Blindness:** Min-pooling 240 rays into 16 sectors permanently destroys spatial resolution, preventing rule agents and planners from navigating narrow gaps between obstacles.
3. **Actuator Space Fragmentation:** Disparate action contracts between algorithms (e.g. Discrete(5) for DQN vs Box(2) for PPO) undermine fair scientific benchmark comparison across Stages 0–7.
4. **Dropped Ray Misclassification:** With `dropout_prob > 0`, dropped rays read as `0.0` (immediate collision), which could trigger false-positive emergency braking in heuristic or search policies.

---

## 17. Open Questions

1. **Surrounding Vehicles: Sensor vs. Privileged Contract?**
   Should `num_others=4` be considered an accepted "onboard perception tracker module" (analogous to Mobileye / YOLO bounding box tracking), or should it be removed from the canonical contract in favor of pure 259D raw perception?
2. **Standardizing the Discrete Adapter:**
   If continuous `Box(2)` is the canonical underlying actuator contract, should discrete algorithms (DQN, MCTS) use MetaDrive's native `Discrete(25)` or a leaner symmetric `Discrete(9)` to reduce tree search branching factor?
3. **LiDAR Noise in Canonical Benchmarks:**
   Should canonical evaluation seeds feature clean LiDAR (`gaussian_noise=0.0`, `dropout_prob=0.0`) with noise reserved for robustness stress-testing, or should non-zero noise be part of the default environment?
4. **Lane Line / Side Detector Inclusions:**
   Currently, `side_detector` and `lane_line_detector` point clouds are disabled by default (scalar left/right distances are used instead). Should full border ray point clouds be explored for Platform V1?

---

## 18. Recommendations — Non-Frozen

> **NOTICE:** The recommendations below are strictly proposals resulting from this Gate 1 audit. **No specifications or contracts are frozen in this task.**

1. **Canonical Environment Observation Recommendation:**
   - Retain **Candidate B (275D)** or **Candidate A (259D)** as the uncompressed root observation contract emitted by the environment.
   - Prohibit freezing a 35D compressed observation at the environment layer; allow agents to apply deterministic feature extraction wrappers locally.
2. **Canonical Environment Action Recommendation:**
   - Adopt native continuous **`Box(-1.0, 1.0, shape=(2,), dtype=np.float32)`** as the foundational actuator interface for all stages.
   - Provide an optional, certified symmetric discrete adapter (e.g. `Discrete(25)` or `Discrete(9)`) for discrete-only methods (DQN, discrete MCTS).
3. **Telemetry Formalization:**
   - Strictly sequester `route_completion`, collision flags, and raw world coordinates in `info` as evaluator-only telemetry.
4. **Control Frequency Contract:**
   - Preserve `physics_world_step_size=0.02` and `decision_repeat=5` ($10\text{ Hz}$ agent cycle), which is thoroughly validated across physics, actuation, and determinism.
