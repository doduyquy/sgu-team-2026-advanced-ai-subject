# Platform V1 Episode Lifecycle, Termination, and Horizon Specification Audit

> **Status:** AUDIT & DESIGN RECOMMENDATION (GATE 3)  
> **Freeze Status:** NON-FROZEN — RESEARCH SPECIFICATION ONLY  
> **Target Scope:** Universal episode lifecycle contracts supporting Stages 0 through 7  
> **Date:** September 2026 (Updated with Gate-3 Audit-Integrity Refinements)  

---

## 1. Executive Summary & Purpose

Gate 3 of Research Platform V1 establishes the scientific **episode lifecycle, reset reproducibility, termination, and horizon contracts** shared across all research stages (Stages 0–7):
- **Stage 0:** Random Baseline
- **Stage 1:** Rule / Heuristic
- **Stage 2:** Planning / Search
- **Stage 3:** Simulation Look-ahead (MCTS)
- **Stage 4:** Imitation Learning
- **Stage 5:** Model-Free RL (PPO/SAC)
- **Stage 6:** Search + Learning
- **Stage 7:** Model-Based RL (World Models)

### Core Evidence Distinction:
To maintain rigorous scientific precision, this report explicitly separates:
1. **Source-Verified Behavior:** Simulator mechanics confirmed directly from local MetaDrive 0.4.3 source code.
2. **Empirically Observed Behavior:** Measurable experimental data collected under tested simulator runs.
3. **Platform Design Policy:** Recommended Platform V1 contracts and architectural principles (all non-frozen).

### Core Mandates:
- Standardize the Gymnasium episode interface: separate task-level **termination** (`terminated=True`) from step-budget **truncation** (`truncated=True`).
- Formulate a deterministic, **safety-first outcome taxonomy** prioritizing physical collision and boundary failures over task arrival.
- Calibrate a **route-aware horizon policy** eliminating false-negative timeouts on longer canonical scenarios under actual MapSuite traffic.
- Ensure strict **reset reproducibility** and zero state leakage across episode boundaries following terminal failures.
- Provide pure, unit-testable helpers in `src/platform/` runnable without starting the 3D graphics simulator.
- Keep reward design strictly deferred to **Gate 4**. `CourseEnvV1` remains the historical exploratory baseline.

---

## 2. Source-of-Truth Verification

Authoritative source reference is the local pinned MetaDrive repository:
- **Local Path:** `D:\SGU\CNTT\TTNTNC\metadrive-src`
- **Pinned Git Commit:** `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`
- **Package Version:** `0.4.3` (`metadrive.constants.VERSION == "0.4.3"`)
- **Git Status:** Verified clean working tree via `git status --porcelain`.
- **Automated Validation:** Verified dynamically by `scripts/audit_episode_contract.py`.

---

## 3. Reset Semantics & State Cleansing

### Source-Verified Behavior (`metadrive/envs/base_env.py` lines 512–544):
1. **Global Seed Assignment:** `self._reset_global_seed(seed)` sets `self.current_seed = seed`.
2. **Engine Reset:** `self.engine.reset()` builds or swaps the road network, samples traffic parameters, and initializes agents.
3. **Sensor Flush:** `self.reset_sensors()` flushes depth, LiDAR, and camera buffers.
4. **State Cleansing:** Step counters, cumulative rewards, and termination flags are explicitly re-initialized:
   ```python
   self.dones = {agent_id: False for agent_id in self.agents.keys()}
   self.episode_rewards = defaultdict(float)
   self.episode_lengths = defaultdict(int)
   ```

### Empirically Observed Behavior:
- Captured the true reset observation vector immediately upon `env.reset(seed=42)` before any actions were executed.
- In Run 0, the vehicle was deliberately driven into a real terminal failure (`out_of_road=True` at step 27).
- Consecutive resets on the same environment instance (`seed=42`) yielded:
  - Observation max absolute difference: $0.00\times 10^0$
  - Position difference: $0.0000000000\text{ m}$
  - Heading difference: $0.0000000000\text{ rad}$
  - Speed difference: $0.0000000000\text{ km/h}$
  - Route checkpoints & block sequence: 100% identical
  - Reset flags: `arrive_dest=False`, `out_of_road=False`, `crash=False`
- **Zero state leakage** was confirmed across episode boundaries.

---

## 4. Termination Semantics (`done_function()`)

### Source-Verified Behavior (`metadrive/envs/metadrive_env.py` lines 133–205):
At every simulation step, `done_function(vehicle_id)` evaluates:

```python
done_info = {
    TerminationState.CRASH_VEHICLE: vehicle.crash_vehicle,
    TerminationState.CRASH_OBJECT: vehicle.crash_object,
    TerminationState.CRASH_BUILDING: vehicle.crash_building,
    TerminationState.CRASH_HUMAN: vehicle.crash_human,
    TerminationState.CRASH_SIDEWALK: vehicle.crash_sidewalk,
    TerminationState.OUT_OF_ROAD: self._is_out_of_road(vehicle),
    TerminationState.SUCCESS: self._is_arrive_destination(vehicle),
    TerminationState.MAX_STEP: max_step,
    TerminationState.ENV_SEED: self.current_seed,
}
```

### Direct Termination Triggers:
- `SUCCESS` (`arrive_dest`): Directly sets `done = True`.
- `OUT_OF_ROAD`: Directly sets `done = True` (when `out_of_road_done=True`, default True).
- `CRASH_VEHICLE`: Directly sets `done = True` (when `crash_vehicle_done=True`, default True).
- `CRASH_OBJECT`: Directly sets `done = True` (when `crash_object_done=True`, default True).
- `CRASH_BUILDING`: Directly sets `done = True` (always, unconditional).
- `CRASH_HUMAN`: Directly sets `done = True` (when `crash_human_done=True`, default True).
- `MAX_STEP`: Sets `done = True` **only if** `truncate_as_terminate=True`.

---

## 5. Truncation vs. Termination Construction

### Source-Verified Behavior (`metadrive/envs/base_env.py` lines 624–625):
```python
truncateds = {k: step_infos[k].get(TerminationState.MAX_STEP, False) for k in self.agents.keys()}
terminateds = {k: self.dones[k] for k in self.agents.keys()}
```

### Empirically Observed Behavior:
1. **Case 4 (`truncate_as_terminate = False`):**
   - Step budget timeout ($15\text{ steps}$) produced:
     $$\text{max\_step} = \text{True},\quad \text{terminated} = \text{False},\quad \text{truncated} = \text{True}$$
   - Conforms strictly to standard Gymnasium semantics.
2. **Case 4b Control Test (`truncate_as_terminate = True`):**
   - Step budget timeout ($15\text{ steps}$) produced:
     $$\text{max\_step} = \text{True},\quad \text{terminated} = \text{True},\quad \text{truncated} = \text{True}$$
   - Simulator logged internal warning: `"When reaching max steps, both 'terminate' and 'truncate will be True. Generally, only the 'truncate' should be 'True'."`

### Platform Design Policy:
Platform V1 adopts **`truncate_as_terminate = False`** as the default contract.

---

## 6. Task Success Semantics (`_is_arrive_destination()`)

### Source-Verified Behavior (`metadrive/envs/metadrive_env.py` lines 227–232):
```python
long, lat = vehicle.navigation.final_lane.local_coordinates(vehicle.position)
flag = (vehicle.navigation.final_lane.length - 5 < long < vehicle.navigation.final_lane.length + 5) and (
    vehicle.navigation.get_current_lane_width() / 2 >= lat >=
    (0.5 - vehicle.navigation.get_current_lane_num()) * vehicle.navigation.get_current_lane_width()
)
```

### Exact Arrival Boundary:
- `_is_arrive_destination()` does **NOT** explicitly test `vehicle.lane == final_lane`.
- Instead, arrival is true when the vehicle position projects within the destination window near the end of `final_lane` ($\pm 5.0\text{ m}$ from lane terminus) and lies inside the allowed final-road lateral corridor:
  $$\text{final\_lane.length} - 5.0\text{ m} < \text{long} < \text{final\_lane.length} + 5.0\text{ m}$$
  $$\frac{w}{2} \ge \text{lat} \ge (0.5 - N_{\text{lanes}}) \times w$$
- Decoupled from cumulative reward thresholds or return sums.

---

## 7. Out-of-Road, Continuous Line, and Sidewalk Audit

### Source-Verified Behavior (`metadrive/envs/metadrive_env.py` lines 234–245):
```python
def _is_out_of_road(self, vehicle):
    ret = not vehicle.on_lane
    if self.config["out_of_route_done"]:
        ret = ret or vehicle.out_of_route
    elif self.config["on_continuous_line_done"]:
        ret = ret or vehicle.on_yellow_continuous_line or vehicle.on_white_continuous_line or vehicle.crash_sidewalk
    if self.config["on_broken_line_done"]:
        ret = ret or vehicle.on_broken_line
    return ret
```

### Empirical & Source Discoveries:
1. **`crash_sidewalk` is NOT directly checked in `done_function()`.**
2. **`crash_sidewalk` is terminal indirectly through `out_of_road`:**  
   Because `on_continuous_line_done = True` by default, `_is_out_of_road()` sets `ret = True` whenever `vehicle.crash_sidewalk` occurs.
3. **Solid Border Crossing:** Steering across the solid outer lane marking triggers `on_white_continuous_line = True` $\implies$ `out_of_road = True` $\implies$ `terminated = True`.

---

## 8. Crash Termination Semantics

- **Vehicle Collision (`crash_vehicle`):** Triggered when Bullet physics reports contact with `CollisionGroup.Vehicle`. Directly sets `done = True`.
- **Traffic Object Collision (`crash_object`):** Triggered on contact with static traffic objects (`TrafficCone`, `TrafficBarrier`). Directly sets `done = True`.
- **Building Collision (`crash_building`):** Unconditionally sets `done = True`.
- **Pedestrian Collision (`crash_human`):** Directly sets `done = True`.

---

## 9. Simultaneous Terminal Event Handling & Safety-First Precedence

In autonomous driving, multiple raw flags can fire on the final step (e.g. crossing the destination line while crashing into a vehicle).

### Platform Design Policy (Precedence Architecture):
To prevent "success hacking" where unsafe policies claim arrival despite catastrophic collisions, Platform V1 enforces **Safety-First Outcome Precedence**:

```text
1. CRASH_HUMAN         (Critical safety failure)
2. CRASH_VEHICLE       (Multi-agent vehicle collision)
3. CRASH_OBJECT        (Traffic obstacle impact)
4. CRASH_BUILDING      (Off-road structure impact)
5. CRASH_SIDEWALK      (Pedestrian refuge/curb collision)
6. OUT_OF_ROAD         (Drivable boundary departure)
7. SUCCESS             (Clean destination arrival)
8. TIMEOUT             (Horizon step budget exhaustion)
9. UNKNOWN_TERMINATION (Unclassified termination fallback)
10. UNDETERMINED       (Ongoing active step)
```

### Single-Source-of-Truth Implementation:
In `src/platform/episode.py`, `classify_episode_outcome()` evaluates dynamically against `SAFETY_FIRST_PRECEDENCE` and `REASON_TO_RAW_FLAG` to prevent code drift.

### Clean Success Semantics:
$$\text{clean\_success} = \text{arrive\_dest} \land \neg(\text{crash\_human} \lor \text{crash\_vehicle} \lor \text{crash\_object} \lor \text{crash\_building} \lor \text{crash\_sidewalk} \lor \text{out\_of\_road})$$

- If an agent reaches the destination line while simultaneously contacting a vehicle or curb, `primary_reason` is classified as `CRASH_VEHICLE` (or `CRASH_SIDEWALK`) and `clean_success = False`.
- **Raw Flag Preservation:** All original simulator flags are preserved intact in `EpisodeOutcome.raw_flags`.

---

## 10. Normalized Outcome Taxonomy & Implementation

Implemented in `src/platform/episode.py`:
- `TerminalReason`: Enum representing all normalized outcomes.
- `EpisodeOutcome`: Immutable dataclass container storing `terminated`, `truncated`, `primary_reason`, `clean_success`, and `raw_flags`.
- `compute_route_aware_horizon()`: Hardened helper with strict input validation (raises `ValueError` on negative route length, non-positive speeds, or invalid bounds).

*Unit-tested:* 10 pure unit tests in `tests/test_episode_lifecycle.py` execute in $<0.01\text{ s}$ without Panda3D.

---

## 11. Horizon Calibration Across 12 Canonical Scenarios (Actual MapSuite Traffic)

We evaluated all 12 canonical and alternate MapSuite scenarios using **actual Gate-2 benchmark traffic** (`traffic_density = cand["traffic_density"]`, `traffic_mode = "trigger"`) and reconstructed exact geometry via `MapGenerateMethod.PG_MAP_FILE`:

### Empirical Rollout Results (Actual MapSuite Benchmark Traffic)

| Tier | Role | Sequence | Seed | Route Length | Traffic Density | Planned Traffic | Active Mean | Active Max | Unique Activated | Reference Outcome | IDM Speed | Counterfactual 1000 Status | Proposed Horizon | Budget Seconds | Margin over IDM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Easy** | Primary | `SCS` | `11` | $349.6\text{ m}$ | 0.00 | 0 | 0.0 | 0 | 0 | SUCCESS @ 407 steps | $29.0\text{ km/h}$ | COMPLETED_WITHIN_1000 | **1049** | $104.9\text{ s}$ | $2.58\times$ |
| **Easy** | Alternate | `SCSS` | `9` | $403.1\text{ m}$ | 0.00 | 0 | 0.0 | 0 | 0 | SUCCESS @ 478 steps | $29.1\text{ km/h}$ | COMPLETED_WITHIN_1000 | **1210** | $121.0\text{ s}$ | $2.53\times$ |
| **Easy** | Alternate | `SCCS` | `9` | $444.1\text{ m}$ | 0.00 | 0 | 0.0 | 0 | 0 | SUCCESS @ 533 steps | $29.2\text{ km/h}$ | COMPLETED_WITHIN_1000 | **1333** | $133.3\text{ s}$ | $2.50\times$ |
| **Medium** | Primary | `SCXCS` | `11` | $523.8\text{ m}$ | 0.08 | 9 | 4.51 | 8 | 8 | OUT_OF_ROAD @ 383 steps | $28.8\text{ km/h}$ | OUT_OF_ROAD_AT_STEP_383 | **1572** | $157.2\text{ s}$ | N/A |
| **Medium** | Alternate | `SCTCS` | `0` | $521.1\text{ m}$ | 0.08 | 7 | 3.57 | 5 | 7 | SUCCESS @ 629 steps | $29.3\text{ km/h}$ | COMPLETED_WITHIN_1000 | **1564** | $156.4\text{ s}$ | $2.49\times$ |
| **Medium** | Alternate | `SCXCCS` | `13` | $688.4\text{ m}$ | 0.08 | 12 | 5.62 | 9 | 12 | SUCCESS @ 844 steps | $28.9\text{ km/h}$ | COMPLETED_WITHIN_1000 | **2066** | $206.6\text{ s}$ | $2.45\times$ |
| **Hard** | Primary | `SCXOCS` | `2` | $643.0\text{ m}$ | 0.15 | 27 | 5.56 | 9 | 9 | CRASH_VEHICLE @ 249 steps | $27.5\text{ km/h}$ | CRASH_VEHICLE_AT_STEP_249 | **1930** | $193.0\text{ s}$ | N/A |
| **Hard** | Alternate | `SCTXrCS` | `1` | $748.8\text{ m}$ | 0.15 | 28 | 12.76 | 23 | 28 | CRASH_VEHICLE @ 780 steps | $27.7\text{ km/h}$ | CRASH_VEHICLE_AT_STEP_780 | **2247** | $224.7\text{ s}$ | N/A |
| **Hard** | Alternate | `XTOCS` | `19` | $496.9\text{ m}$ | 0.15 | 17 | 7.35 | 12 | 17 | SUCCESS @ 678 steps | $26.0\text{ km/h}$ | COMPLETED_WITHIN_1000 | **1491** | $149.1\text{ s}$ | $2.20\times$ |
| **Extreme** | Primary | `CrXROSTR` | `6` | $938.6\text{ m}$ | 0.25 | 70 | 16.42 | 26 | 26 | CRASH_VEHICLE @ 563 steps | $12.4\text{ km/h}$ | CRASH_VEHICLE_AT_STEP_563 | **2816** | $281.6\text{ s}$ | N/A |
| **Extreme** | Alternate | `SCXOCrTYCS` | `16` | $1051.8\text{ m}$ | 0.25 | 79 | 20.20 | 35 | 35 | CRASH_VEHICLE @ 618 steps | $20.9\text{ km/h}$ | CRASH_VEHICLE_AT_STEP_618 | **3156** | $315.6\text{ s}$ | N/A |
| **Extreme** | Alternate | `SCTXORyCCS` | `4` | $1003.2\text{ m}$ | 0.25 | 58 | 15.09 | 19 | 19 | CRASH_VEHICLE @ 405 steps | $26.5\text{ km/h}$ | CRASH_VEHICLE_AT_STEP_405 | **3010** | $301.0\text{ s}$ | N/A |

### Secondary Zero-Traffic Reference Traversal Times (`horizon_reference_no_traffic.csv`)

| Tier | Role | Sequence | Seed | Route Length | Reference Outcome | Completion Steps | Completion Time | IDM Mean Speed | Counterfactual 1000 Status |
|---|---|---|---|---|---|---|---|---|---|
| **Easy** | Primary | `SCS` | `11` | $349.6\text{ m}$ | SUCCESS | 407 | $40.7\text{ s}$ | $29.0\text{ km/h}$ | COMPLETED_WITHIN_1000 |
| **Easy** | Alternate | `SCSS` | `9` | $403.1\text{ m}$ | SUCCESS | 478 | $47.8\text{ s}$ | $29.1\text{ km/h}$ | COMPLETED_WITHIN_1000 |
| **Easy** | Alternate | `SCCS` | `9` | $444.1\text{ m}$ | SUCCESS | 533 | $53.3\text{ s}$ | $29.2\text{ km/h}$ | COMPLETED_WITHIN_1000 |
| **Medium** | Primary | `SCXCS` | `11` | $523.8\text{ m}$ | OUT_OF_ROAD | 383 | $38.3\text{ s}$ | $28.8\text{ km/h}$ | OUT_OF_ROAD_AT_STEP_383 |
| **Medium** | Alternate | `SCTCS` | `0` | $521.1\text{ m}$ | SUCCESS | 629 | $62.9\text{ s}$ | $29.3\text{ km/h}$ | COMPLETED_WITHIN_1000 |
| **Medium** | Alternate | `SCXCCS` | `13` | $688.4\text{ m}$ | SUCCESS | 830 | $83.0\text{ s}$ | $29.5\text{ km/h}$ | COMPLETED_WITHIN_1000 |
| **Hard** | Primary | `SCXOCS` | `2` | $643.0\text{ m}$ | SUCCESS | 779 | $77.9\text{ s}$ | $29.4\text{ km/h}$ | COMPLETED_WITHIN_1000 |
| **Hard** | Alternate | `SCTXrCS` | `1` | $748.8\text{ m}$ | SUCCESS | 897 | $89.7\text{ s}$ | $29.5\text{ km/h}$ | COMPLETED_WITHIN_1000 |
| **Hard** | Alternate | `XTOCS` | `19` | $496.9\text{ m}$ | SUCCESS | 599 | $59.9\text{ s}$ | $29.2\text{ km/h}$ | COMPLETED_WITHIN_1000 |
| **Extreme** | Primary | `CrXROSTR` | `6` | $938.6\text{ m}$ | SUCCESS | 1123 | $112.3\text{ s}$ | $29.4\text{ km/h}$ | **WOULD_TRUNCATE_UNDER_1000** |
| **Extreme** | Alternate | `SCXOCrTYCS` | `16` | $1051.8\text{ m}$ | SUCCESS | 1255 | $125.5\text{ s}$ | $29.6\text{ km/h}$ | **WOULD_TRUNCATE_UNDER_1000** |
| **Extreme** | Alternate | `SCTXORyCCS` | `4` | $1003.2\text{ m}$ | SUCCESS | 1218 | $121.8\text{ s}$ | $29.6\text{ km/h}$ | **WOULD_TRUNCATE_UNDER_1000** |

### Traffic Statistics Analysis:
- Under `TrafficMode.Trigger`, planned traffic is dynamically activated in waves as ego approaches.
- In Hard and Extreme tiers, active concurrent traffic averages $5.6 - 20.2$ vehicles with peak concurrent loads reaching up to $35$ vehicles.
- In dense traffic, IDMPolicy lane-tracking struggles with sharp intersection turns and merge bottlenecks, clipping curbs or colliding with traffic at complex junctures. This is reported honestly per Section 12 instructions.
- In the secondary zero-traffic calibration, IDMPolicy cleanly reached the goal on all 3 Extreme scenarios at steps 1123, 1255, and 1218.

### Scoped Horizon Finding:
Under a fixed 1000-step budget ($100.0\text{ s}$), reference rollouts on Extreme routes ($>900\text{ m}$) **would be truncated prior to arrival** even when driving at nominal cruising speeds ($\sim 29.5\text{ km/h}$) under clean conditions.

---

## 12. Selected Non-Frozen Route-Aware Horizon Policy

To treat the horizon strictly as an **emergency safety cutoff** rather than an aggressive speed target, we adopt a deterministic route-aware formula:

$$\text{horizon\_steps} = \max\left(\text{min\_steps},\ \min\left(\text{max\_steps},\ \left\lceil \frac{\text{route\_length\_m}}{v_{\text{floor\_mps}}} \times M \times f_{\text{control}} \right\rceil\right)\right)$$

### Calibrated Parameters:
- **Reference Floor Speed ($v_{\text{floor}}$):** $18.0\text{ km/h} = 5.0\text{ m/s}$. (Allows cautious navigation through roundabouts and intersections).
- **Safety Margin ($M$):** $1.5\times$.
- **Control Frequency ($f_{\text{control}}$):** $10\text{ Hz}$.
- **Minimum Horizon Floor ($\text{min\_steps}$):** $1000\text{ steps}$ ($100.0\text{ seconds}$).
- **Maximum Safety Cap ($\text{max\_steps}$):** $4000\text{ steps}$ ($400.0\text{ seconds}$).

### Formula Simplification:
$$\text{horizon\_steps} = \max\left(1000,\ \lceil \text{route\_length\_m} \times 3.0 \rceil\right)$$

### Horizon Policy Justification:
1. **Design Rationale:** The horizon is an emergency safety cutoff, not a speed performance metric. Route length is static scenario metadata known prior to episode launch. Setting $v_{\text{floor}} = 18\text{ km/h}$ with a $1.5\times$ margin provides sufficient time for cautious driving, queuing at intersections, and yielding to traffic.
2. **Empirical Grounding:** In successful zero-traffic reference rollouts across all tiers and successful actual-traffic scenarios in Easy, Medium, and Hard, the proposed formula provides a $2.20\times - 2.58\times$ buffer over nominal reference completion time.
3. **Coverage Limitation:** IDMPolicy is not a robust expert under dense traffic on complex geometries; scenarios where IDM failed do not provide a measured empirical completion-time margin and are documented as `N/A`. The proposed horizon remains non-frozen and will be verified by future learned/planning agents.

---

## 13. Traffic Lifecycle: `TrafficMode.Trigger` vs. `TrafficMode.Respawn`

### Source-Verified Behavior:
- In `TrafficMode.Trigger`, traffic vehicles are pre-allocated across road blocks and triggered once when ego enters the trigger road.
- In `TrafficMode.Respawn`, source code shows vehicles are continuously replaced upon reaching their destinations.

### Empirically Observed Behavior:
- Across two independent 60-step runs of scenario seed `42`:
  - Planned traffic count match: `True` (9 planned vehicles).
  - Step-by-step active vehicle count trace match: `True`.
  - First traffic vehicle position equality: `True` (exact coordinate match).
- Under `TrafficMode.Respawn`, active vehicles remain continuously sustained at peak capacity.
- **Finding:** TrafficMode.Trigger provides a finite, preplanned traffic population and repeatable initialization/activation under the tested same-seed, same-policy configuration.
- *Machine-readable trace:* Recorded in `results/audits/episode/traffic_lifecycle.csv`.

---

## 14. Reset Reproducibility & Zero State Leakage

Tested over sequential resets of scenario seed `42` where Episode 0 was driven into a real terminal failure (`out_of_road=True` at step 27):
- **Initial Position Difference:** $0.0000000000\text{ m}$ (exact float match).
- **Initial Heading Difference:** $0.0000000000\text{ rad}$.
- **Initial Velocity Difference:** $0.0000000000\text{ km/h}$.
- **Initial Observation Max Difference:** $0.00\times 10^0$.
- **Route Checkpoints Match:** `True`.
- **Block Sequence Match:** `True` (`ISCXCS`).
- **Planned Traffic Match:** `True` (9 vehicles).
- **Flag Cleansing:** Subsequent resets yield `arrive_dest=False`, `out_of_road=False`, `crash=False`.
- **Finding:** Zero state leakage across episode boundaries following terminal failures.

---

## 15. Research Launcher Roadmap Note

A future Research Launcher UI may expose:
- Scenario tier selection (Easy, Medium, Hard, Extreme);
- Primary vs. Alternate map selection;
- Agent policy selection (Stage 0 through 7);
- Dynamic route-aware horizon display;
- Headless vs. interactive 3D rendering.

*Status:* Roadmap note only. No GUI dependencies (e.g. PySide6) are implemented in Gate 3.

---

## 16. Risks & Unresolved Points

1. **IDM Dense Traffic Bottlenecks:** Under dense traffic ($0.15 - 0.25$), IDMPolicy exhibits lane-edge clipping when traffic clusters at intersection entries, underscoring the necessity of Stage 2+ planning algorithms capable of multi-lane negotiation.
2. **Pedestrian/Human Collision Spawning:** `crash_human` is verified in the taxonomy and classifier, but standard MapSuite scenarios keep `accident_prob = 0.0`. Human collisions will be stress-tested in later robustness benchmarks.
3. **Reward Alignment:** Reward shaping remains completely unaddressed in Gate 3 and is explicitly deferred to **Gate 4**.

---

## 17. Non-Frozen EpisodeSpecV1 Recommendation

```json
{
  "spec_version": "1.0.0",
  "control_frequency_hz": 10,
  "truncate_as_terminate": false,
  "traffic_mode": "trigger",
  "horizon_policy_type": "route_aware",
  "reference_floor_speed_kmh": 18.0,
  "safety_margin": 1.5,
  "min_horizon_steps": 1000,
  "max_horizon_steps": 4000
}
```

---

## 18. Explicit Confirmation: Reward Design is Gate 4

Reward redesign, potential penalties, progress bonuses, and lateral shaping are strictly out of scope for Gate 3 and remain reserved for **Gate 4**. `CourseEnvV1` remains unchanged.
