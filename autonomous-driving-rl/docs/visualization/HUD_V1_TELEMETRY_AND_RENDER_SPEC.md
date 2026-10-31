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

### 2.1 Proposed Snapshot Schema (`HudTelemetrySnapshotV1`)
```python
@dataclass(frozen=True)
class HudTelemetrySnapshotV1:
    """Immutable per-step telemetry snapshot consumed strictly by the HUD renderer."""
    step_index: int                    # Current simulation step (0..horizon)
    simulation_time_s: float           # Elapsed simulation time (step_index * 0.1s)
    world_position: Tuple[float, float]# Ego vehicle (x, y) coordinates in meters
    heading_theta_rad: float           # Ego heading angle in radians [-pi, pi]
    speed_kmh: float                   # Current vehicle speed in km/h [0.0, 150.0]
    route_completion: float            # Fraction of route completed [0.0, 1.0]
    canonical_steering: float          # Physical steering command [-1.0, 1.0] (negative=left, positive=right)
    canonical_throttle_brake: float    # Physical throttle/brake [-1.0, 1.0] (positive=throttle, negative=brake)
    is_clean_success: bool             # Destination reached with zero violations
    is_terminated: bool                # Episode has reached a terminal state
```

---

## 3. Authoritative Telemetry Source Mapping

Every HUD widget derives its state strictly from verified simulator and action objects:

| HUD Widget | Source Object / API in Code | Physical Unit | Valid Data Range | Update Timing | Fallback / Missing Data Behavior |
|---|---|---|---|---|---|
| **Speedometer (Value)** | `env.agent.speed_km_h` | km/h | $[0.0, 150.0]$ | 10 Hz (every step) | Clamp to $0.0$; render as `"0"` |
| **Speedometer (Arc)** | Normalized: `speed_kmh / 100.0` | Fraction | $[0.0, 1.0]$ | 10 Hz | Clamp to $[0.0, 1.0]$; $0\%$ arc fill |
| **Route Progress** | `step_info.get("route_completion", 0.0)` | Fraction | $[0.0, 1.0]$ | 10 Hz | Clamp to $[0.0, 1.0]$; empty bar |
| **Left Indicator** | `canonical_action.steering < -0.15` | Boolean | `{True, False}` | 10 Hz | Inactive state (dim gray) |
| **Right Indicator** | `canonical_action.steering > 0.15` | Boolean | `{True, False}` | 10 Hz | Inactive state (dim gray) |
| **Throttle Meter** | `max(0.0, canonical_action.throttle_brake)` | Fraction | $[0.0, 1.0]$ | 10 Hz | $0.0$ fill (empty meter) |
| **Brake Meter** | `max(0.0, -canonical_action.throttle_brake)` | Fraction | $[0.0, 1.0]$ | 10 Hz | $0.0$ fill (empty meter) |
| **Minimap (Ego Pos)** | `env.agent.position` | Meters $(x, y)$ | Route bounding box | 10 Hz | Hide ego marker |
| **Minimap (Heading)** | `env.agent.heading_theta` | Radians | $[-\pi, \pi]$ | 10 Hz | Point northward ($0.0$) |
| **Minimap (Roads)** | `env.current_map.get_map_features(2.0)` | Polylines | World coordinate space | Once at episode start | Fallback to agent bounding box |

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
In MetaDrive, vehicle heading angle $\theta$ is defined such that $\theta = 0$ points along the $+X$ axis, and $\theta = \frac{\pi}{2}$ points along the $+Y$ axis.
The minimap ego marker is rendered as an oriented triangle with vertices:
$$\mathbf{v}_{\text{tip}} = (x_e + L \cos\theta, \; y_e + L \sin\theta)$$
$$\mathbf{v}_{\text{left}} = (x_e + W \cos(\theta + \frac{5\pi}{6}), \; y_e + W \sin(\theta + \frac{5\pi}{6}))$$
$$\mathbf{v}_{\text{right}} = (x_e + W \cos(\theta - \frac{5\pi}{6}), \; y_e + W \sin(\theta - \frac{5\pi}{6}))$$
where $L = 5.0\,\text{m}$ and $W = 3.5\,\text{m}$ in world space coordinates.

### 4.3 Trajectory History Buffer
- **Ring Buffer Specification:** Stores the last $K \le 500$ points.
- **Distance Filtering:** A point $(x_t, y_t)$ is appended to the trail only if Euclidean distance from the preceding point exceeds $d_{\text{thresh}} = 0.5\,\text{m}$:
  $$\sqrt{(x_t - x_{t-1})^2 + (y_t - y_{t-1})^2} \ge 0.5\,\text{m}$$
  This prevents redundant memory allocation when the vehicle is stationary or stopped at an intersection.
- **Reset Invariant:** When an episode terminates or `env.reset()` is invoked, the trajectory buffer is cleared to 0 points. Old trajectory trails never leak into subsequent episodes.

---

## 5. Visual Styling and Layout Specification

### 5.1 Screen Coordinate System (`aspect2d`)
Panda3D's `aspect2d` coordinates are normalized to the screen height:
- Vertical axis $Z \in [-1.0, 1.0]$ ($Z = 1.0$ is screen top, $Z = -1.0$ is screen bottom).
- Horizontal axis $X \in [-\alpha, \alpha]$, where $\alpha = \frac{\text{width}}{\text{height}}$ (e.g. $\alpha = 1.777$ for 16:9 aspect ratio).
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
   - Coordinates: `[0.015, 0.185, 0.745, 0.985]` (Width = 17%, Height = 24%).
   - Frame: 1.5 px border with `BORDER_GLASS` color.
2. **Route Progress Bar:**
   - Anchor: Center $X = 0.0$, Top $Z = 0.92$.
   - Dimensions: Width = $0.70$ units ($\approx 35\%$ of screen), Height = $0.024$ units ($\approx 1.2\%$).
   - Fill: Cyan fill bar scaling from left to right as `route_completion` advances from $0.0$ to $1.0$.
3. **Speedometer Gauge:**
   - Anchor: Center $X = \alpha - 0.32$, Bottom $Z = -0.72$.
   - Digits: 3-digit bold font (`64 pt`), format `f"{int(speed_kmh):03d}"`, followed by `"km/h"` in `16 pt`.
   - Outer Arc: 240-degree circular arc ($120^\circ$ sweep to $-120^\circ$ sweep), filling proportionally up to $100\,\text{km/h}$.
4. **Pedal Meters (Brake & Throttle):**
   - Anchor: $X = \alpha - 0.62$, Bottom $Z = -0.72$.
   - Two vertical bars: Width = $0.035$ units, Height = $0.20$ units, separated by $0.02$ units.
   - Left Bar: Brake (fills bottom-to-top with `PEDAL_BRAKE`).
   - Right Bar: Throttle (fills bottom-to-top with `PEDAL_THROTTLE`).
5. **Turn Indicators:**
   - Anchor: $X = -\alpha + 0.20$, Bottom $Z = -0.80$.
   - Left Chevron (`<`): Glows `INDICATOR_ACTIVE` when steering $< -0.15$.
   - Right Chevron (`>`): Glows `INDICATOR_ACTIVE` when steering $> +0.15$.

---

## 6. Scientific Isolation & Timing Invariants

### 6.1 Absolute Timing Boundary Protection
To guarantee that rendering the HUD introduces **zero measurement distortion** into scientific latency benchmarks:
- Monotonic timing of agent decision latency in `src/launcher/executor.py` strictly measures `agent.act()`:
  ```python
  t0 = time.perf_counter_ns()
  decision = agent.act(agent_input)
  t1 = time.perf_counter_ns()
  latencies_ms.append((t1 - t0) / 1e6)
  ```
- **HUD telemetry capture and rendering updates occur strictly after physics stepping and outcome classification.**
- Under no circumstances may any HUD update code, scene graph traversal, or OpenGL drawing call execute between `t0` and `t1`.

### 6.2 Zero Environment Mutation
- The HUD renderer has strictly read-only access to `HudTelemetrySnapshotV1`.
- It possesses zero references to `env.step()`, `env.reset()`, or action adapters.
- It cannot alter physics state, spawn locations, or background traffic.

### 6.3 Headless Benchmark Isolation
- `VALIDATION` and `TEST` launcher modes enforce `render_mode="OFF"`.
- When `render_mode="OFF"`, the HUD system is completely bypassed (0 display regions allocated, 0 nodes attached, 0 draw calls emitted).
- Benchmark results are 100% byte-identical whether the HUD subsystem is installed or uninstalled.
