# Platform V1 Episode Lifecycle, Termination, and Horizon Specification Audit

> **Status:** AUDIT & DESIGN RECOMMENDATION (GATE 3)  
> **Freeze Status:** NON-FROZEN — RESEARCH SPECIFICATION ONLY  
> **Target Scope:** Universal episode lifecycle contracts supporting Stages 0 through 7  
> **Date:** September 2026  

---

## 1. Executive Summary & Purpose

Gate 3 of Research Platform V1 establishes the scientific **episode lifecycle and termination contracts** shared across all research stages (Stages 0–7):
- **Stage 0:** Random Baseline
- **Stage 1:** Rule / Heuristic
- **Stage 2:** Planning / Search
- **Stage 3:** Simulation Look-ahead (MCTS)
- **Stage 4:** Imitation Learning
- **Stage 5:** Model-Free RL (PPO/SAC)
- **Stage 6:** Search + Learning
- **Stage 7:** Model-Based RL (World Models)

### Core Mandates:
- Standardize the Gymnasium episode interface: separate task-level **termination** (`terminated=True`) from step-budget **truncation** (`truncated=True`).
- Formulate a deterministic, **safety-first outcome taxonomy** prioritizing physical collision and boundary failures over task arrival.
- Calibrate a **route-aware horizon policy** eliminating the false-negative timeout failures discovered in Gate 2 for longer maps.
- Ensure strict **reset reproducibility** and zero state leakage across episode boundaries.
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

## 3. Exact Reset Semantics (`BaseEnv.reset()`)

From source inspection of `metadrive/envs/base_env.py` (lines 512–544):
1. **Global Seed Assignment:** `self._reset_global_seed(seed)` sets `self.current_seed = seed`.
2. **Engine Reset:** `self.engine.reset()` triggers map generation (if unbuilt), obstacle placement, and traffic initialization.
3. **Sensor Reset:** `self.reset_sensors()` flushes depth, LiDAR, and camera buffers.
4. **Task Manager Step:** `self.engine.taskMgr.step()` renders the initial frame.
5. **State Initialization:**
   ```python
   self.dones = {agent_id: False for agent_id in self.agents.keys()}
   self.episode_rewards = defaultdict(float)
   self.episode_lengths = defaultdict(int)
   ```
6. **State Cleansing:** All internal step counters, cumulative rewards, and termination flags are explicitly reset to zero/False. No previous episode terminal flags leak into the initial step.

---

## 4. Exact MetaDrive Termination Semantics (`done_function()`)

From `metadrive/envs/metadrive_env.py` (lines 133–205):
MetaDrive evaluates terminal conditions at every step in `done_function(vehicle_id)`:

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
- `SUCCESS` (`arrive_dest`): Sets `done = True`.
- `OUT_OF_ROAD`: Sets `done = True` (if `out_of_road_done=True`, default True).
- `CRASH_VEHICLE`: Sets `done = True` (if `crash_vehicle_done=True`, default True).
- `CRASH_OBJECT`: Sets `done = True` (if `crash_object_done=True`, default True).
- `CRASH_BUILDING`: Sets `done = True` (always, unconditional).
- `CRASH_HUMAN`: Sets `done = True` (if `crash_human_done=True`, default True).
- `MAX_STEP`: Sets `done = True` **only if** `truncate_as_terminate=True`.

---

## 5. Truncation vs. Termination Construction

In `metadrive/envs/base_env.py` (lines 624–625):
```python
truncateds = {k: step_infos[k].get(TerminationState.MAX_STEP, False) for k in self.agents.keys()}
terminateds = {k: self.dones[k] for k in self.agents.keys()}
```

### Critical Gymnasium Conformance Finding:
1. When `truncate_as_terminate = False` (Recommended Platform V1 Contract):
   - At step budget exhaustion (`episode_length >= horizon`), `max_step` is True.
   - `done` is NOT set to True in `done_function()`.
   - Result: **`terminated = False`, `truncated = True`**.
   - This adheres to the standard Gymnasium specification: timeout is an environmental horizon truncation, not an episodic policy failure.
2. When `truncate_as_terminate = True`:
   - Both `done` and `max_step` evaluate to True.
   - Result: `terminated = True`, `truncated = True`.
   - The simulator issues an explicit internal warning: `"When reaching max steps, both 'terminate' and 'truncate will be True. Generally, only the 'truncate' should be 'True'."`

*Platform V1 Contract:* Sets **`truncate_as_terminate = False`** as the default.

---

## 6. Task Success Semantics (`_is_arrive_destination()`)

From `metadrive/envs/metadrive_env.py` (lines 227–232):
```python
long, lat = vehicle.navigation.final_lane.local_coordinates(vehicle.position)
flag = (vehicle.navigation.final_lane.length - 5 < long < vehicle.navigation.final_lane.length + 5) and (
    vehicle.navigation.get_current_lane_width() / 2 >= lat >=
    (0.5 - vehicle.navigation.get_current_lane_num()) * vehicle.navigation.get_current_lane_width()
)
```

### Exact Arrival Boundary:
- The ego vehicle must be localized on the designated `final_lane`.
- Longitudinal position must be within a $\pm 5.0\text{ m}$ window of the lane terminus:
  $$\text{lane.length} - 5.0\text{ m} < \text{long} < \text{lane.length} + 5.0\text{ m}$$
- Lateral position must remain within the drivable road boundary of the final corridor:
  $$\frac{w}{2} \ge \text{lat} \ge (0.5 - N_{\text{lanes}}) \times w$$
- **Platform Success Definition:** Success is a pure spatial task-arrival event. It is decoupled from reward thresholds or episode return sums.

---

## 7. Out-of-Road, Continuous Line, and Sidewalk Audit

From `metadrive/envs/metadrive_env.py` (lines 234–245):
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

### Critical Empirical Discoveries:
1. **`crash_sidewalk` is NOT Directly Terminal:**  
   Inspection of `done_function()` confirms there is no direct check for `done_info[TerminationState.CRASH_SIDEWALK]`.
2. **`crash_sidewalk` Triggers Termination Indirectly Through `out_of_road`:**  
   Because `on_continuous_line_done=True` by default, `_is_out_of_road()` sets `ret = True` whenever `vehicle.crash_sidewalk` or `vehicle.on_white_continuous_line` is detected.
3. **Solid Border Crossing:** Steering across the solid outer lane marking triggers `on_white_continuous_line=True` $\implies$ `out_of_road=True` $\implies$ `terminated=True`.

---

## 8. Crash Termination Semantics

- **Vehicle Collision (`crash_vehicle`):** Triggered when the Bullet physics engine reports collision contact with any other vehicle (`CollisionGroup.Vehicle`). Sets `done = True` directly.
- **Traffic Object Collision (`crash_object`):** Triggered on contact with static traffic objects (`TrafficCone`, `TrafficBarrier`). Sets `done = True` directly.
- **Building Collision (`crash_building`):** Unconditionally sets `done = True`.
- **Pedestrian Collision (`crash_human`):** Triggered on contact with traffic participants. Sets `done = True`.

---

## 9. Simultaneous Terminal Event Handling & Safety-First Precedence

In autonomous driving, multiple raw flags can fire on the final step (e.g. crossing the destination line while crashing into a vehicle, or drifting off-road at the exact step of timeout).

### Precedence Architecture:
To prevent "success hacking" where unsafe policies claim arrival despite catastrophic collisions, Platform V1 enforces **Safety-First Outcome Precedence**:

```text
1. CRASH_HUMAN        (Highest severity safety violation)
2. CRASH_VEHICLE      (Multi-agent vehicle collision)
3. CRASH_OBJECT       (Traffic obstacle impact)
4. CRASH_BUILDING     (Off-corridor structure impact)
5. CRASH_SIDEWALK     (Pedestrian refuge/curb collision)
6. OUT_OF_ROAD        (Drivable boundary departure)
7. SUCCESS            (Clean destination arrival)
8. TIMEOUT            (Horizon step budget exhaustion)
9. UNKNOWN_TERMINATION(Unclassified terminal fallback)
10. UNDETERMINED      (Ongoing active step)
```

### Clean Success Semantics:
$$\text{clean\_success} = \text{arrive\_dest} \land \neg(\text{crash\_human} \lor \text{crash\_vehicle} \lor \text{crash\_object} \lor \text{crash\_building} \lor \text{crash\_sidewalk} \lor \text{out\_of\_road})$$

- If an agent reaches the destination line while simultaneously contacting a vehicle or curb, `primary_reason` is classified as `CRASH_VEHICLE` (or `CRASH_SIDEWALK`) and `clean_success = False`.
- **Raw Flag Preservation:** All original simulator flags are preserved intact in `EpisodeOutcome.raw_flags`.

---

## 10. Normalized Outcome Taxonomy & Implementation

Implemented in `src/platform/episode.py` as a pure, unit-testable module:

```python
@dataclass(frozen=True)
class EpisodeOutcome:
    terminated: bool
    truncated: bool
    primary_reason: TerminalReason
    clean_success: bool
    raw_flags: Dict[str, Any]

    @property
    def is_done(self) -> bool:
        return self.terminated or self.truncated
```

*Unit-tested:* 9 pure unit tests in `tests/test_episode_lifecycle.py` execute in $<0.01\text{ s}$ without Panda3D.

---

## 11. Horizon Calibration Methodology & Reference Driving

In Gate 2, we identified that the default MetaDrive `horizon = 1000` (100.0s at 10 Hz) causes false-negative timeouts on Extreme scenarios. To establish a scientifically calibrated horizon, we deployed MetaDrive's **`IDMPolicy`** as a deterministic calibration instrument across all 12 canonical and alternate MapSuite scenarios (with a generous audit ceiling of 3500 steps).

### Calibration Findings Across 12 Scenarios

| Tier | Candidate Role | Sequence | Seed | Route Length | IDM Result | IDM Steps | IDM Time | IDM Mean Speed | Default Horizon 1000 Status | Proposed Route Horizon | Budget Seconds | Margin over IDM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Easy** | Primary | `SCS` | `11` | $349.6\text{ m}$ | ARRIVED | 407 | $40.7\text{ s}$ | $29.0\text{ km/h}$ | FEASIBLE | **1049** | $104.9\text{ s}$ | $2.58\times$ |
| **Easy** | Alternate | `SCSS` | `9` | $403.1\text{ m}$ | ARRIVED | 478 | $47.8\text{ s}$ | $29.1\text{ km/h}$ | FEASIBLE | **1210** | $121.0\text{ s}$ | $2.53\times$ |
| **Easy** | Alternate | `SCCS` | `9` | $444.1\text{ m}$ | ARRIVED | 533 | $53.3\text{ s}$ | $29.2\text{ km/h}$ | FEASIBLE | **1333** | $133.3\text{ s}$ | $2.50\times$ |
| **Medium** | Primary | `SCXCS` | `11` | $523.8\text{ m}$ | OUT_OF_ROAD* | 383 | $38.3\text{ s}$ | $28.8\text{ km/h}$ | TIMEOUT_OR_FAIL | **1572** | $157.2\text{ s}$ | $2.50\times$ (nom) |
| **Medium** | Alternate | `SCTCS` | `0` | $521.1\text{ m}$ | ARRIVED | 629 | $62.9\text{ s}$ | $29.3\text{ km/h}$ | FEASIBLE | **1564** | $156.4\text{ s}$ | $2.49\times$ |
| **Medium** | Alternate | `SCXCCS` | `13` | $688.4\text{ m}$ | ARRIVED | 830 | $83.0\text{ s}$ | $29.5\text{ km/h}$ | FEASIBLE | **2066** | $206.6\text{ s}$ | $2.49\times$ |
| **Hard** | Primary | `SCXOCS` | `2` | $643.0\text{ m}$ | ARRIVED | 779 | $77.9\text{ s}$ | $29.4\text{ km/h}$ | FEASIBLE | **1930** | $193.0\text{ s}$ | $2.48\times$ |
| **Hard** | Alternate | `SCTXrCS` | `1` | $748.8\text{ m}$ | ARRIVED | 897 | $89.7\text{ s}$ | $29.5\text{ km/h}$ | FEASIBLE | **2247** | $224.7\text{ s}$ | $2.50\times$ |
| **Hard** | Alternate | `XTOCS` | `19` | $496.9\text{ m}$ | ARRIVED | 599 | $59.9\text{ s}$ | $29.2\text{ km/h}$ | FEASIBLE | **1491** | $149.1\text{ s}$ | $2.49\times$ |
| **Extreme** | Primary | `CrXROSTR` | `6` | $938.6\text{ m}$ | ARRIVED | **1123** | $112.3\text{ s}$ | $29.4\text{ km/h}$ | **TIMEOUT (Defect)** | **2816** | $281.6\text{ s}$ | $2.51\times$ |
| **Extreme** | Alternate | `SCXOCrTYCS` | `16` | $1051.8\text{ m}$ | ARRIVED | **1255** | $125.5\text{ s}$ | $29.6\text{ km/h}$ | **TIMEOUT (Defect)** | **3156** | $315.6\text{ s}$ | $2.51\times$ |
| **Extreme** | Alternate | `SCTXORyCCS` | `4` | $1003.2\text{ m}$ | ARRIVED | **1218** | $121.8\text{ s}$ | $29.6\text{ km/h}$ | **TIMEOUT (Defect)** | **3010** | $301.0\text{ s}$ | $2.47\times$ |

*\*Note on SCXCS s11:* The baseline IDMPolicy clipped the sharp 90-degree intersection curb at step 383 due to known IDM lateral tracking constraints at 29 km/h. Per Section 12 instructions, we report this honestly and rely on the successful alternate `SCTCS` s0 ($521\text{ m}$, arrived at step 629) to calibrate nominal Medium completion.

*Critical Finding on Extreme Tier:* Under the old `horizon=1000`, **100% of Extreme scenarios fail due to timeout truncation**, even when driven by an expert reference driver at $29.5\text{ km/h}$.

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

### Fairness Rules Enforced:
1. Computed strictly from static scenario metadata **before** the episode begins.
2. Completely independent of agent identity, policy type, or research Stage.
3. Never adapts dynamically during the episode.

---

## 13. Traffic Lifecycle: `TrafficMode.Trigger` vs. `TrafficMode.Respawn`

Empirical testing compared MetaDrive's two primary traffic modes:
1. **`TrafficMode.Trigger` (Adopted Platform Contract):**
   - Traffic vehicles are pre-allocated across road blocks (`block_triggered_vehicles`).
   - Vehicles are activated once when ego approaches their road block and disappear upon reaching their destination.
   - **Reproducibility:** Tested across independent identical runs; active vehicle traces matched bit-for-bit ($0$ count difference).
   - **Fairness:** Slower policies face the exact same finite traffic encounters as faster policies.
2. **`TrafficMode.Respawn` (Rejected for Scientific Benchmark):**
   - Vehicles continually respawn at destination arrivals.
   - Slower policies take more simulation steps, accumulating a higher cumulative traffic exposure and consuming unpredictable PRNG steps.

*Recommendation:* Adopt **`TrafficMode.Trigger`** as the scientific benchmark contract.

---

## 14. Reset Reproducibility & Zero State Leakage

Tested over sequential resets of scenario seed `42` where Episode 0 was driven into a terminal failure (`out_of_road=True`):
- **Initial Position Difference:** $0.0000000000\text{ m}$ (exact float match).
- **Initial Heading Difference:** $0.0000000000\text{ rad}$.
- **Initial Velocity Difference:** $0.0000000000\text{ km/h}$.
- **Initial Observation Max Difference:** $0.00\times 10^0$.
- **Planned Traffic Initialization:** 9 vehicles (exact match).
- **Flag Cleansing:** Subsequent reset yields `arrive_dest=False`, `out_of_road=False`, `crash=False`.
- **Finding:** Zero state leakage across episode boundaries under the tested configuration.

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

1. **IDM Intersection Edge-Clipping:** IDMPolicy lacks look-ahead corner-cutting avoidance for sharp 90-degree intersection turns at high speeds ($29\text{ km/h}$), requiring future planning agents to modulate speed appropriately.
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
