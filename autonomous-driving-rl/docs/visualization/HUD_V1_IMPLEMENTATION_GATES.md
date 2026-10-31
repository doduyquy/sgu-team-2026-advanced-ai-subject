# HUD V1 — Phased Implementation Gates and Test Matrix

- **Document ID:** `PLAN-HUD-V1-IMPLEMENTATION-GATES`
- **Version:** `1.0.0`
- **Status:** `GATES-PLAN / READY FOR REVIEW`
- **Date:** 2026-10-31
- **Lead / Stage 5 + GUI:** Hữu Tri
- **Baseline Git Commit:** `581dc49570ceb00fff13cf8df073a76e654735df`
- **Subproject Root:** `autonomous-driving-rl/`

---

## 1. Overview and Phasing Strategy

Following the architectural decisions and visual specifications approved in **HUD-0**, the implementation of the Minimal Transparent Racing HUD is partitioned into three strictly sequential, independently verifiable engineering gates:

```text
+-----------------------------------------------------------------------------------+
| HUD-0: Architecture ADR & Existing Minimap Audit (Docs Only) [CURRENT GATE]       |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| HUD-1: Live Minimap Implementation                                                |
| - Adopt Khoa's Panda3D DisplayRegion minimap baseline from origin/khoa            |
| - Add directional heading chevron and historical trajectory trail                 |
| - Implement glass frame and step-driven update API                                |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| HUD-2: Transparent Racing HUD Implementation                                      |
| - Implement 6 remaining widgets on base.aspect2d: Progress Bar, Speedometer,      |
|   Left/Right Indicators, Brake/Throttle Meters                                    |
| - Apply glass-morphism aesthetic (transparent dark, cyan accents, active pulse)  |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| HUD-3: Runtime Integration & Scientific Isolation Audit                          |
| - Wire HUD into src/launcher/executor.py strictly when render_mode="NATIVE"      |
| - Benchmark runtime performance (<1.0 ms overhead) on SANDBOX                     |
| - Prove zero scientific contamination and complete headless isolation             |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| Return to Core Research: Open Decisions Sign-Off & Stage 5 Model-Free RL         |
+-----------------------------------------------------------------------------------+
```

---

## 2. Gate-by-Gate Specification

---

### Gate HUD-1: Live Minimap Implementation

#### 2.1 Scope & Objectives
- Introduce `src/visualization/minimap_overlay.py` by cleanly adopting and evolving the existing `MiniMapOverlay` from `origin/khoa` (commit `271d704`).
- Add vehicle heading orientation via an oriented triangle/chevron.
- Add historical trajectory trail via a bounded ring buffer ($K \le 500$) drawn with `LineSegs`.
- Replace opaque olive background with translucent dark glass (`#0B131B` @ 0.75 alpha) and subtle glass border.
- Provide a clean step-driven update method: `update(world_pos, heading_rad)` without thread racing in `taskMgr`.
- **Standalone Demo Harness:** Implement `scripts/demo_minimap_native.py` allowing interactive testing and visual inspection of the minimap in SANDBOX mode without modifying production `src/launcher/executor.py` before HUD-3.

#### 2.2 Expected Files
- `src/visualization/__init__.py`: Package export for visualizer components.
- `src/visualization/minimap_overlay.py`: Hardware-accelerated minimap overlay class.
- `scripts/demo_minimap_native.py`: Standalone sandbox demo harness for visual verification.
- `tests/test_minimap_overlay.py`: Unit and mock tests verifying camera creation, bounding box calculation, trajectory appending, and node destruction.

#### 2.3 Dependencies & Interfaces
- Dependencies: `panda3d.core`, `metadrive.constants.CamMask`.
- Interface:
  ```python
  class MiniMapOverlay:
      def __init__(self, env: MetaDriveEnv, padding_m: float = 12.0) -> None: ...
      def update(self, world_x: float, world_y: float, heading_rad: float) -> None: ...
      def reset(self) -> None: ...
      def destroy(self) -> None: ...
  ```

#### 2.4 Acceptance Criteria
1. Minimap renders road geometry cleanly in top-left `DisplayRegion` (`[0.015, 0.185, 0.745, 0.985]`).
2. Ego marker points in the exact direction of vehicle travel.
3. Trajectory trail buffer stores the sliding window of the most recent $K \le 500$ sampled waypoints; retained coordinates match raw vehicle positions within numerical floating-point precision ($< 1\text{e-4}\,\text{m}$); buffer resets cleanly to 0 points on episode restart.
4. `destroy()` cleans up camera, display region, and nodes with zero resource leaks.
5. All 402 existing unit tests continue to pass without regression.

#### 2.5 Rollback Strategy
If Panda3D camera allocation causes stability issues on certain platforms, `src/visualization/minimap_overlay.py` can be removed or disabled via a single boolean toggle without affecting simulation or logging.

---

### Gate HUD-2: Transparent Racing HUD Implementation

#### 2.1 Scope & Objectives
- Implement `src/visualization/racing_hud.py` containing the remaining six widgets attached to `base.aspect2d`:
  1. Route Progress Bar (top-center, $Z = 0.92$, width $35\%$).
  2. Large Digital Speedometer (bottom-right, $Z = -0.72$, bold numerals + 240-degree outer arc).
  3. Left Turn Indicator (bottom-left, chevrons pulsing amber when steering $< -0.15$).
  4. Right Turn Indicator (bottom-left, chevrons pulsing amber when steering $> +0.15$).
  5. Brake Meter (bottom-right, vertical fill bar glowing red on deceleration).
  6. Throttle Meter (bottom-right, vertical fill bar glowing green on acceleration).
- Apply uniform visual tokens: dark glass (`#0B131B`), subtle white frame borders (`#E0F0FF`), and electric cyan accents (`#00E5FF`).
- Enforce responsive scaling in `aspect2d` across window dimensions without geometric distortion.
- **Standalone Demo Harness:** Implement `scripts/demo_racing_hud_native.py` allowing visual inspection of all six 2D widgets in SANDBOX mode without touching production `executor.py`.

#### 2.2 Expected Files
- `src/visualization/racing_hud.py`: Procedural 2D racing HUD class.
- `src/visualization/telemetry_model.py`: `HudTelemetrySnapshotV1` dataclass.
- `scripts/demo_racing_hud_native.py`: Standalone sandbox demo harness for HUD-2 verification.
- `tests/test_racing_hud.py`: Unit tests verifying widget positioning, clamp math, indicator threshold triggers, and clean node disposal.

#### 2.3 Dependencies & Interfaces
- Dependencies: `panda3d.core`, `direct.gui.OnscreenText`, `direct.gui.OnscreenImage`.
- Interface:
  ```python
  class RacingHud:
      def __init__(self, engine: Any) -> None: ...
      def update(self, snapshot: HudTelemetrySnapshotV1) -> None: ...
      def reset(self) -> None: ...
      def destroy(self) -> None: ...
  ```

#### 2.4 Acceptance Criteria
1. All six widgets render crisp, anti-aliased geometry in their designated screen quadrants.
2. Speedometer readout reflects exact speed in km/h; progress bar reflects exact route completion.
3. Turn indicators and pedal meters respond dynamically to AI action payloads with correct polarity.
4. Window resizing preserves 16:9 circular proportions without warping gauge arcs into ellipses.
5. `destroy()` completely clears all `aspect2d` child nodes.

---

### Gate HUD-3: Runtime Integration & Scientific Isolation Audit

#### 3.1 Scope & Objectives
- Wire `MiniMapOverlay` and `RacingHud` into `src/launcher/executor.py` under an integrated `RacingHudManager`:
  - Instantiated strictly when `self.plan.render_mode.upper() == "NATIVE"`.
  - Updated strictly *after* `env.step()`, preserving absolute timing isolation around `agent.act()`.
  - Cleaned up safely in `finally:` block of `executor.py`.
- Benchmark runtime performance: verify that rendering the HUD adds $<1.0\,\text{ms}$ per step ($<10\%$ of the 100 ms step budget), maintaining 60 FPS simulator performance.
- Perform strict Scientific Isolation Audit: prove that canonical benchmark modes (`VALIDATION`, `TEST`) remain 100% headless with zero HUD invocation.
- Document full integration evidence in `docs/environment/PLATFORM_V1_HUD_INTEGRATION_AUDIT.md`.

#### 3.2 Expected Files
- `src/launcher/executor.py`: Conditional invocation of `RacingHudManager` on `NATIVE` mode.
- `src/visualization/hud_manager.py`: Coordination facade joining Minimap and Racing HUD.
- `tests/test_hud_runtime_integration.py`: End-to-end integration and isolation test suite.
- `docs/environment/PLATFORM_V1_HUD_INTEGRATION_AUDIT.md`: Formal verification audit.

#### 3.3 Acceptance Criteria
1. `render_mode="NATIVE"` in SANDBOX launches MetaDrive with full transparent racing HUD and live minimap.
2. `render_mode="OFF"` in SANDBOX, VALIDATION, and TEST launches headless with zero HUD node or camera allocations.
3. Controlled decision latency verification: timing boundary strictly measures `agent.act()` and completely excludes HUD overhead; under identical seed and deterministic policy, agent actions are identical; latency distributions (p50, p95, p99) are compared across repeated runs with reported confidence intervals rather than assuming unproven OS noise equivalence.
4. An empirical desktop screenshot captured in a live native desktop session is committed in `docs/environment/` as verified visual evidence (cannot be fulfilled by offscreen mock).
5. Simulator exceptions trigger safe cleanup without orphaned DisplayRegions or Panda3D window hangs.
6. All repository unit tests pass (100% OK).

---

## 3. Comprehensive Test & Validation Matrix

The following test matrix must be implemented across HUD-1, HUD-2, and HUD-3:

| # | Test Scenario / Assertion | Testing Mechanism | Target Gate | Acceptance Criteria |
|---|---|---|---|---|
| **T01** | **Headless Import & Execution Independence** | Headless unittest | HUD-1, 2, 3 | `import src.launcher.executor` and headless execution allocate zero HUD nodes or cameras. |
| **T02** | **NATIVE HUD Initialization** | Mock/Subprocess launch | HUD-3 | HUD initializes when `render_mode="NATIVE"`; display region and aspect2d nodes attach. |
| **T03** | **Decision Payload & Timing Non-Interference** | Controlled unit test with fixture agent | HUD-3 | Timing boundary excludes HUD; agent action sequences are identical; latency distributions (p50, p95, p99) are measured and reported across repeated runs. |
| **T04** | **Minimap Geometry Extents** | Polyline bounding test | HUD-1 | Orthographic film size matches calculated map extents plus padding across tested tiers. |
| **T05** | **Ego Heading Correctness** | Heading vector unit test | HUD-1 | Heading marker rotates counter-clockwise with positive yaw; tested at $0^\circ, 90^\circ, 180^\circ, 270^\circ$. |
| **T06** | **Trajectory Trail Precision** | Waypoint append test | HUD-1 | Retained trail waypoints match sampled raw coordinates within floating-point precision ($< 1\text{e-4}\,\text{m}$); $0.5\,\text{m}$ filter bounds vertex spacing; ring buffer maintains trailing window of up to 500 points. |
| **T07** | **Zero Cross-Episode Trail Leakage** | Sequential episode test | HUD-1 | `reset()` clears trajectory points to 0; Episode 2 does not display Episode 1's trail. |
| **T08** | **Exception Teardown Resilience** | Injection of simulated runtime crash | HUD-1, 2, 3 | `destroy()` safely removes DisplayRegion and nodes even if episode crashes mid-loop. |
| **T09** | **Route Completion Truthfulness** | Value injection test | HUD-2 | Progress bar percentage matches `step_info["route_completion"]` without divergence. |
| **T10** | **Indicator & Meter Polarity** | Action payload matrix | HUD-2 | Steering $< -0.15$ triggers Left; $> +0.15$ triggers Right; Throttle/Brake map strictly to positive/negative values; missing action yields disabled state. |
| **T11** | **Speedometer Unit Parity** | Speed comparison test | HUD-2 | Speed digits reflect `env.agent.speed_km_h` in km/h (not m/s or internal physics units); missing data renders `"---"`. |
| **T12** | **16:9 Aspect Ratio Resilience** | Window resize simulation | HUD-2, 3 | Circular speedometer and minimap do not warp into ellipses when window is resized. |
| **T13** | **Interactive Screenshot Verification** | Visual artifact capture | HUD-3 | Real native desktop Panda3D screenshot captured and committed as empirical evidence in audit docs. |
| **T14** | **Performance Overhead Benchmark** | Step time measurement on SANDBOX | HUD-3 | Dedicated profiling timer measures `hud_update_ms` independently; design target: mean $< 1.0\,\text{ms}$, p95 $< 2.0\,\text{ms}$, 60 FPS target. |
| **T15** | **TEST/VALIDATION Headless Gating** | Preflight negative test | HUD-3 | Launcher strictly rejects `render_mode="NATIVE"` on benchmark suites; HUD remains dormant. |

---

## 4. Strict Out-of-Scope Boundaries

The following activities are strictly prohibited throughout all HUD gates:
- No changes to reinforcement learning algorithms (PPO, SAC, DQN).
- No modifications to frozen MapSuite V1 geometries, seed plans, or benchmark manifests.
- No modifications to `configs/platform/` or frozen platform contract hashes.
- No inclusion of manual driving controls, game menus, or camera switching controls in the HUD.
- No execution of canonical TEST benchmark suites for HUD testing.
- No attempt to persist visualizer telemetry into Gate 7 canonical scientific run artifacts.
