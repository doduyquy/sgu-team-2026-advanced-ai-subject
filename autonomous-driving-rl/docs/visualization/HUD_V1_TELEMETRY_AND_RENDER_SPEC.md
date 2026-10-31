# HUD V1 — Telemetry Data Model and Visual Render Specification

- **Document ID:** `SPEC-HUD-V1-TELEMETRY-RENDER`
- **Version:** `1.0.0`
- **Status:** `DESIGN-SPEC / READY FOR REVIEW`
- **Date:** 2026-10-31
- **Lead / Stage 5 + GUI:** Hữu Tri
- **Baseline Git Commit:** `581dc49570ceb00fff13cf8df073a76e654735df`
- **Subproject Root:** `autonomous-driving-rl/`

---

## 1. Design Philosophy & Feature Boundary

The HUD V1 design adheres strictly to the user-approved **Minimal Transparent Racing HUD** style. It delivers real-time situational awareness during interactive simulation (`render_mode="NATIVE"` in `SANDBOX`) while preserving an uncluttered view of the 3D driving environment.

### 1.1 The Seven Locked Components
The HUD comprises exactly seven visual elements:
1. **Live Minimap:** Top-left corner showing the global road geometry, vehicle position, heading vector, and historical trajectory trail.
2. **Route Progress Bar:** Slender horizontal bar centered at the top edge showing percentage completion of the current route.
3. **Digital Speedometer:** Circular arc gauge and prominent digital readout at the bottom-right displaying vehicle speed in km/h.
4. **Left Turn Indicator:** Leftward chevron at the bottom-left that pulses when the agent outputs a left steering command.
5. **Right Turn Indicator:** Rightward chevron adjacent to the left indicator that pulses when the agent outputs a right steering command.
6. **Brake Meter:** Vertical fill bar left of the speedometer indicating deceleration/braking command intensity.
7. **Throttle Meter:** Vertical fill bar adjacent to the brake meter indicating acceleration/throttle command intensity.

### 1.2 Prohibited Elements in HUD V1
To prevent UI clutter and maintain scientific integrity, the following elements are **explicitly excluded** from HUD V1:
| Excluded Element | Rationale |
|---|---|
| *AI Active / Autopilot Badge* | Redundant; the simulator is always autonomous in research evaluation. |
| *Live System Status / Ping* | Simulation runs synchronously with the physics engine; network ping is non-existent. |
| *Standalone Steering Wheel Angle Gauge* | Turn indicators and vehicle heading already provide clear directional feedback; a separate wheel graphic clutters the bottom-left. |
| *Engine RPM & Gear Readout* | MetaDrive vehicles use a continuous physical powertrain model; artificial gear/RPM values would be fictitious. |
| *Cumulative Reward & Score* | Reward is an internal RL training signal, never a primary benchmark metric. Displaying live reward in the HUD misleads researchers into treating it as a performance score. |
| *Interactive Pause / Camera / Play Buttons* | The HUD is an overlay for observation, not an interactive game menu. Execution control belongs to CLI/Workbench. |
| *Predictive Collision Warnings without Sensors* | Fabricating predictive alerts without an underlying active perception sensor violates scientific truthfulness. |

---

## 2. Telemetry Snapshot Data Model

To enforce complete decoupling between the simulation engine and the visual overlay, the HUD consumes an immutable snapshot object created on each simulation step.

### 2.1 Pre-Existing Telemetry Defect Note
Source audit of `src/launcher/executor.py` lines 413–414 reveals an existing defect:
```python
speed_kmh = float(env.agent.speed_kmh) if hasattr(env, "agent") and hasattr(env.agent, "speed_kmh") else 0.0
```
In MetaDrive 0.4.3 (`DefaultVehicle`), the speed attribute is named **`speed_km_h`** (with an underscore), not `speed_kmh`. Because `hasattr(env.agent, "speed_kmh")` evaluates to `False`, the executor silently falls back to `0.0` km/h.
*HUD Architecture Requirement:* In HUD-2, the telemetry extraction helper must access the authoritative pinned attribute `getattr(env.agent, "speed_km_h", None)`. If `env.agent` lacks this attribute, `speed_kmh` must evaluate to `None` (missing telemetry), triggering `"---"` on the speedometer rather than silently fabricating `0.0`. A regression test must be added in HUD-2 to assert that moving vehicles produce non-null, non-zero speed.

### 2.2 Proposed Snapshot Schema (`HudTelemetrySnapshotV1`)
```python
@dataclass(frozen=True)
class HudTelemetrySnapshotV1:
    """Immutable per-step telemetry snapshot consumed strictly by the HUD renderer."""
    step_index: int                               # Current simulation step (0..horizon)
    simulation_time_s: float                      # Elapsed simulation time (step_index * control_dt_s)
    world_position: Optional[Tuple[float, float]] # Ego vehicle (x, y) in meters (None if unavailable)
    heading_theta_rad: Optional[float]            # Ego yaw angle in radians [-pi, pi] (None if unavailable)
    speed_kmh: Optional[float]                    # Current speed in km/h [0.0, 150.0] (None if unavailable)
    route_completion: Optional[float]             # Route completion fraction [0.0, 1.0] (None if unavailable)
    canonical_steering: Optional[float]           # Physical steering [-1.0, 1.0] (None if unavailable)
    canonical_throttle_brake: Optional[float]     # Physical throttle/brake [-1.0, 1.0] (None if unavailable)
    terminal_outcome_status: str                  # "UNDETERMINED" during episode; TerminalReason upon end
    is_clean_success: Optional[bool]              # None during intermediate steps; bool only upon completion
```

### 2.3 Missing Data & Fallback Semantics
To maintain scientific truthfulness, the HUD never fabricates synthetic numbers (such as substituting 0.0 or North) for missing data:
- **Missing Speed (`speed_kmh is None`):** Render digits as `"---"` and suppress arc fill. This is visually distinct from `0.0` km/h (a validly stationary vehicle with numeric readout `"000"`).
- **Missing Route Completion (`route_completion is None`):** Render progress bar in dim gray with text `"---"`. This is visually distinct from `0.0` (valid vehicle position at the starting line).
- **Missing Action Telemetry vs Neutral Coasting:**
  - *Valid Neutral Action (`canonical_throttle_brake = 0.0`):* Meter frames remain active/illuminated with $0\%$ bar fill, visually indicating an active coasting command.
  - *Missing Action (`canonical_throttle_brake is None`):* Throttle and brake meters render in a dimmed, disabled state (`UNAVAILABLE`), signaling absence of command data without adding extraneous widgets.
  - *Turn Indicators:* Left and right chevrons glow amber when active, dim to dark slate-blue (`INDICATOR_IDLE`) when steering is neutral ($[-0.15, 0.15]$), and fade to low-alpha disabled state when `canonical_steering is None`.
- **Missing Position / Heading:**
  - If `world_position is None`: Hide ego marker completely and preserve existing trajectory trail without appending invalid points.
  - If `world_position` is valid but `heading_theta_rad is None`: Render a non-directional circular dot at the known position without an oriented tip. Never fabricate an arbitrary North orientation ($\theta = 0.0$).
- **Intermediate Episode Status:** `is_clean_success` is strictly `None` during intermediate steps; clean arrival is an episodic terminal classification that cannot be known prematurely.

---

## 3. Authoritative Telemetry Source Mapping

Every HUD widget derives its state strictly from verified simulator and action objects:

| HUD Widget | Source Object / API in Code | Physical Unit | Valid Data Range | Display Range | Fallback / Missing Data Behavior |
|---|---|---|---|---|---|
| **Speedometer (Digits)** | `getattr(env.agent, "speed_km_h", None)` | km/h | $[0.0, 150.0]$ | $0..999$ | Render `"---"` if None; render `"000"` if $0.0$ |
| **Speedometer (Arc)** | Normalized: `speed_kmh / 100.0` | Fraction | $[0.0, 1.5]$ | $0.0..1.0$ (sweeps $240^\circ$) | Arc fill clamped at $1.0$ ($100\,\text{km/h}$); empty if None |
| **Route Progress** | `step_info.get("route_completion", None)` | Fraction | $[0.0, 1.0]$ | $0\%..100\%$ | Render `"---"` and empty fill if None |
| **Left Indicator** | `canonical_action.steering < -0.15` | Boolean | `{True, False}` | Amber pulse / Dim gray | Disabled state if action is None; dim gray if neutral |
| **Right Indicator** | `canonical_action.steering > 0.15` | Boolean | `{True, False}` | Amber pulse / Dim gray | Disabled state if action is None; dim gray if neutral |
| **Throttle Meter** | `max(0.0, canonical_action.throttle_brake)` | Fraction | $[0.0, 1.0]$ | Vertical fill $0\%..100\%$ | Disabled state if None; active frame with empty fill if $0.0$ |
| **Brake Meter** | `max(0.0, -canonical_action.throttle_brake)` | Fraction | $[0.0, 1.0]$ | Vertical fill $0\%..100\%$ | Disabled state if None; active frame with empty fill if $0.0$ |
| **Minimap (Ego Pos)** | `env.agent.position` | Meters $(x, y)$ | Route bounding box | Minimap viewport | Hide marker if unavailable |
| **Minimap (Heading)** | `env.agent.heading_theta` | Radians | $[-\pi, \pi]$ | Oriented chevron | Render non-directional position dot if None (no directional tip) |
| **Minimap (Roads)** | `env.current_map.get_map_features(2.0)` | Polylines | World coordinate space | Top-down scene | Fallback to agent bounding box |

---

## 4. Minimap Mathematical Formulation

### 4.1 Bounding Box & Orthographic Projection
Let $\mathcal{P} = \{\mathbf{p}_i = (x_i, y_i)\}_{i=1}^M$ be the set of road polyline vertices extracted from `env.current_map.get_map_features(interval=2.0)`.
1. **Map Extents:**
   $$x_{\min} = \min_i(x_i) - d_{\text{pad}}, \quad x_{\max} = \max_i(x_i) + d_{\text{pad}}$$
   $$y_{\min} = \min_i(y_i) - d_{\text{pad}}, \quad y_{\max} = \max_i(y_i) + d_{\text{pad}}$$
   where $d_{\text{pad}} = 12.0\,\text{m}$ provides visual clearance around peripheral roads.
2. **Camera Center & Dimensions:**
   $$c_x = \frac{x_{\min} + x_{\max}}{2}, \quad c_y = \frac{y_{\min} + y_{\max}}{2}$$
   $$W_{\text{map}} = x_{\max} - x_{\min}, \quad H_{\text{map}} = y_{\max} - y_{\min}$$
3. **Aspect-Ratio Locking:**
   Let $r_{\text{region}} = \frac{W_{\text{pixel}}}{H_{\text{pixel}}}$ be the aspect ratio of the minimap screen display region.
   To prevent geometric stretching of roads:
   $$\text{If } \frac{W_{\text{map}}}{H_{\text{map}}} > r_{\text{region}}: \quad \text{film\_width} = W_{\text{map}}, \quad \text{film\_height} = \frac{W_{\text{map}}}{r_{\text{region}}}$$
   $$\text{Otherwise}: \quad \text{film\_height} = H_{\text{map}}, \quad \text{film\_width} = H_{\text{map}} \times r_{\text{region}}$$
   The `OrthographicLens` film size is set to `(film_width, film_height)`.

### 4.2 Heading Direction Vector
In MetaDrive, vehicle heading angle $\theta$ is defined such that $\theta = 0$ points along the $+X$ axis, and $\theta = \frac{\pi}{2}$ points along the $+Y$ axis (counter-clockwise positive yaw).
In Panda3D screen space (`aspect2d`), $+X$ is rightward and $+Z$ is upward.
The minimap ego marker is rendered as an oriented triangle with vertices:
$$\mathbf{v}_{\text{tip}} = (x_e + L \cos\theta, \; y_e + L \sin\theta)$$
$$\mathbf{v}_{\text{left}} = (x_e + W \cos(\theta + \frac{5\pi}{6}), \; y_e + W \sin(\theta + \frac{5\pi}{6}))$$
$$\mathbf{v}_{\text{right}} = (x_e + W \cos(\theta - \frac{5\pi}{6}), \; y_e + W \sin(\theta - \frac{5\pi}{6}))$$
where $L = 5.0\,\text{m}$ and $W = 3.5\,\text{m}$ in world space coordinates.
*Validation Requirement:* In HUD-1, heading marker rotation must be asserted at cardinal headings: East ($\theta = 0$, pointing right), North ($\theta = \frac{\pi}{2}$, pointing up), West ($\theta = \pi$, pointing left), and South ($\theta = -\frac{\pi}{2}$, pointing down).

### 4.3 Trajectory History Buffer
- **Ring Buffer Specification:** Stores an in-memory trailing visual history of up to $K \le 500$ retained waypoints. When the buffer reaches capacity, it functions as a FIFO sliding window retaining the most recent 500 points (a visual trailing wake of traveled path); it is not an exhaustive recording of an arbitrarily long entire route.
- **Minimum Distance Threshold (Not an Upper Bound):** A new vehicle coordinate $(x_t, y_t)$ is appended to the trail if and only if its Euclidean distance from the preceding retained point satisfies the minimum distance threshold:
  $$\Delta d = \sqrt{(x_t - x_{\text{last}})^2 + (y_t - y_{\text{last}})^2} \ge d_{\text{thresh}} = 0.5\,\text{m}$$
  *Clarification on Vertex Spacing:* The threshold $0.5\,\text{m}$ is a lower bound designed to filter out stationary vehicle jitter and prevent redundant vertex accumulation when stopped; it is **not an upper bound** on vertex spacing. For example, if a vehicle travels at $72\,\text{km/h}$ ($20.0\,\text{m/s}$), in one $100\,\text{ms}$ simulation step it covers $2.0\,\text{m}$. Two consecutive retained points will be spaced $2.0\,\text{m}$ apart, which is completely valid.
- **Piecewise Linear Polyline Representation:** Consecutive retained waypoints are connected via `LineSegs` segments. The resulting polyline is a piecewise linear visual approximation of the traveled path. Retained waypoint coordinates match raw simulator positions at the instant of sampling within numerical precision ($< 1\text{e-4}\,\text{m}$), but the connecting line segments do not claim an unconditional bound on continuous curve interpolation error between distant samples.
- **Reset Invariant & Scientific Boundary:** When an episode terminates or `env.reset()` is invoked, the trajectory buffer is cleared to 0 points. Old trajectory trails never leak into subsequent episodes. Trajectory data is ephemeral visualization state held exclusively in memory; it is never persisted into Gate 7 scientific run artifacts (`episodes.csv`, `summary.json`), and no replay subsystem is added.

---

## 5. Visual Styling and Layout Specification

### 5.1 Screen Coordinate System (`aspect2d`)
Panda3D's `aspect2d` coordinates are normalized to the screen height:
- Vertical axis $Z \in [-1.0, 1.0]$ ($Z = 1.0$ is screen top, $Z = -1.0$ is screen bottom).
- Horizontal axis $X \in [-\alpha, \alpha]$, where $\alpha = \frac{\text{width}}{\text{height}}$ (for 16:9 aspect ratio, $\alpha = \frac{16}{9} \approx 1.7778$, giving a total screen width of $2\alpha \approx 3.5556$ units).
- Consistent Sizing Formula: To span a screen width fraction $f_{\text{width}} \in [0.0, 1.0]$ on 16:9:
  $$\text{width}_{\text{aspect2d}} = 2 \times \frac{16}{9} \times f_{\text{width}} \approx 3.5556 \times f_{\text{width}}$$
- All 2D widgets anchor relative to screen edges using $\alpha$:
  - Left edge: $X = -\alpha + x_{\text{margin}}$
  - Right edge: $X = \alpha - x_{\text{margin}}$
  - Top edge: $Z = 1.0 - z_{\text{margin}}$
  - Bottom edge: $Z = -1.0 + z_{\text{margin}}$

```text
+-----------------------------------------------------------------------------------+
| [Minimap]                                [ Route Progress Bar ]                   |
| 17% width                                      35% width                          |
| Top-Left                                      Top-Center                          |
|                                                                                   |
|                                                                                   |
|                                                                                   |
|                         [ Unobstructed Driving View ]                             |
|                                                                                   |
|                                                                                   |
|                                                                                   |
| [Turn Indicators]                                       [Pedals]    [Speedometer] |
|   <Left   Right>                                       |Brk| |Thr|     048 km/h   |
|   Bottom-Left                                          Bottom-Right  Bottom-Right |
+-----------------------------------------------------------------------------------+
```

### 5.2 Color Tokens & Material Design

| Token Name | Hex Code | RGBA (Panda3D Scale 0..1) | Role / Usage |
|---|---|---|---|
| `BG_GLASS_DARK` | `#0B131B` | `(0.043, 0.075, 0.106, 0.75)` | Translucent glass backing behind minimap & meters |
| `BORDER_GLASS` | `#E0F0FF` | `(0.878, 0.941, 1.000, 0.35)` | Thin 1.5 px subtle border framing widgets |
| `ACCENT_CYAN` | `#00E5FF` | `(0.000, 0.898, 1.000, 1.00)` | Speedometer arc, active progress bar, trail glow |
| `PEDAL_THROTTLE`| `#00E676` | `(0.000, 0.902, 0.463, 0.85)` | Green throttle meter fill |
| `PEDAL_BRAKE` | `#FF1744` | `(1.000, 0.090, 0.267, 0.85)` | Bright red brake meter fill |
| `INDICATOR_ACTIVE`| `#FFD600` | `(1.000, 0.839, 0.000, 1.00)` | Bright amber glow when turn indicator is pulsed |
| `INDICATOR_IDLE` | `#37474F` | `(0.216, 0.278, 0.310, 0.40)` | Muted gray-blue when turn indicator is inactive |
| `TEXT_PRIMARY` | `#FFFFFF` | `(1.000, 1.000, 1.000, 0.95)` | Speed digits, unit label, progress percentage |

### 5.3 Widget Dimensions & Layout Parameters (16:9 Baseline)
1. **Minimap Viewport (`DisplayRegion`):**
   - Coordinates: `[0.015, 0.185, 0.745, 0.985]` (Width fraction = $17\%$, Height fraction = $24\%$).
   - Pixel dimensions at $1920 \times 1080$: $326.4 \times 259.2\,\text{px}$ (aspect ratio $1.26$). The internal `OrthographicLens` camera uses aspect-ratio locking so the map is never distorted within this rectangular frame. *(Alternative pixel-square viewport: `[0.015, 0.150, 0.745, 0.985]` yielding $260 \times 260\,\text{px}$ at $1920 \times 1080$)*.
   - Glass backing & border: 1.5 px border with `BORDER_GLASS`. Translucent glass backing configured with Panda3D `TransparencyAttrib.MAlpha`.
2. **Route Progress Bar:**
   - Anchor: Center $X = 0.0$, Top $Z = 0.92$.
   - Dimensions: Width = $1.244$ units (exactly $35\%$ of 16:9 screen width), Height = $0.024$ units ($\approx 1.2\%$ of screen height). Extends from $X = -0.622$ to $X = +0.622$.
   - Fill: Cyan fill bar scaling from left to right as `route_completion` advances from $0.0$ to $1.0$. If `route_completion is None`, bar is dim gray with label `"---"`.
3. **Speedometer Gauge:**
   - Anchor: Center $X = \alpha - 0.32$, Bottom $Z = -0.72$.
   - Digits: 3-digit bold font (`64 pt`), format `f"{int(speed_kmh):03d}"` (or `"---"` if None), followed by `"km/h"` in `16 pt`.
   - Outer Arc: 240-degree circular arc ($120^\circ$ sweep to $-120^\circ$ sweep), filling proportionally up to $100\,\text{km/h}$. Clamped at $100\,\text{km/h}$ for visual arc fill, with digits showing full un-clamped speed up to $150\,\text{km/h}$.
4. **Pedal Meters (Brake & Throttle):**
   - Anchor: $X = \alpha - 0.62$, Bottom $Z = -0.72$.
   - Two vertical bars: Width = $0.035$ units, Height = $0.20$ units, separated by $0.02$ units.
   - Left Bar: Brake (fills bottom-to-top with `PEDAL_BRAKE`).
   - Right Bar: Throttle (fills bottom-to-top with `PEDAL_THROTTLE`).
5. **Turn Indicators:**
   - Anchor: $X = -\alpha + 0.20$, Bottom $Z = -0.80$.
   - Left Chevron (`<`): Glows `INDICATOR_ACTIVE` when steering $< -0.15$; `INDICATOR_IDLE` when neutral.
   - Right Chevron (`>`): Glows `INDICATOR_ACTIVE` when steering $> +0.15$; `INDICATOR_IDLE` when neutral.

---

## 6. Scientific Isolation & Timing Design Invariants

*Status Classification:* These principles are **DESIGN INVARIANTS** established at HUD-0 to govern implementation in HUD-1/2/3. Because HUD-0 is documentation-only, runtime isolation must be empirically verified via code path audits and regression testing in HUD-3 before being classified as `VERIFIED`.

### 6.1 Timing Boundary Design Principle
To isolate agent decision latency measurements in scientific benchmarks:
- Monotonic timing of agent decision latency in `src/launcher/executor.py` strictly measures `agent.act()`:
  ```python
  t0 = time.perf_counter_ns()
  decision = agent.act(agent_input)
  t1 = time.perf_counter_ns()
  latencies_ms.append((t1 - t0) / 1e6)
  ```
- **Design Invariant:** HUD telemetry extraction and visualizer updates execute strictly *after* physics stepping and outcome classification. Under no circumstances may any HUD update code, scene graph traversal, or drawing call execute between `t0` and `t1`.
- *Caveat on System Contention:* While timing-boundary isolation ensures that HUD update logic does not execute inside the `t0..t1` measurement bracket, it does not categorically eliminate shared OS thread scheduling or GPU queue contention on multi-threaded runtimes. Consequently, potential background contention must be evaluated via controlled comparison in HUD-3 rather than assumed to be zero.

### 6.2 Read-Only Environment Access (Design Invariant)
- The HUD renderer receives strictly an immutable `HudTelemetrySnapshotV1` copy.
- The visualizer holds zero references to `env.step()`, `env.reset()`, or action adapters.
- It cannot mutate physics state, spawn positions, traffic manager queues, or episode outcome classifications.
- Frozen platform contracts, action adapters, observation rights, reward specs, metrics, seed protocols, and benchmark manifests remain completely untouched.

### 6.3 Headless Benchmark Execution Gating (Verification Criterion)
- Benchmark suites (`VALIDATION`, `TEST`) strictly enforce `render_mode="OFF"` in preflight.
- **Acceptance Criterion for HUD-3:** In `render_mode="OFF"`, code path audits and unit tests must verify that no secondary DisplayRegions are allocated, no visualizer nodes are attached to scene graphs, and zero visualization draw calls or background tasks are spawned. Headless benchmark simulations must execute identically without initializing visualizer infrastructure.
