# Platform V1 — Gate HUD-1 Live Minimap Implementation Audit

- **Document ID:** `AUDIT-HUD1-LIVE-MINIMAP`
- **Version:** `1.0.0`
- **Status:** `AUDIT-COMPLETE / READY FOR INDEPENDENT RE-AUDIT (R2)`
- **Date:** 2026-10-31
- **Lead / Stage 5 + GUI:** **Hữu Tri**
- **Base Commit (`origin/main`):** `c5107b84d9e166e276e34ece1a90f11675180aa0` (PR #30 Merge)
- **Branch:** `feat/hud1-live-minimap`
- **Subproject Root:** `autonomous-driving-rl/`

---

## 1. Executive Summary & Objectives

Gate HUD-1 delivers the live, hardware-accelerated top-down minimap overlay for the MetaDrive NATIVE window in `autonomous-driving-rl/`, implementing the specifications and architectural decisions established in Gate HUD-0.

### Key Deliverables & Milestones:
1. **Existing Minimap Adoption & Attribution:** Adopted the foundational Panda3D camera and `DisplayRegion` architecture developed by Đặng Khoa on branch `origin/khoa` (commit `271d704bdf04c64c869a48237375a36bbae05750`).
2. **Directional Heading Orientation:** Replaced the static circle with an electric cyan chevron marker that dynamically rotates to match the vehicle's real-time yaw angle (`heading_theta`), with a non-directional dot fallback when heading is unavailable.
3. **Historical Trajectory Trail:** Implemented an in-memory bounded ring buffer ($K \le 500$) with a $0.5\,\text{m}$ spatial distance filter, rendered via Panda3D `LineSegs` to show the traveled route without leaking across episodes.
4. **Decoupled Step-Driven API (HUD1-R1-01):** Formulated the canonical interface `update(world_x, world_y, heading_rad)` invoked after simulation steps, strictly using passed arguments with zero implicit fallback to `env.agent`.
5. **Marker Geometry Hardware Clipping (HUD1-R2-01):** Applied `ScissorEffect.makeScreen(scissor_box)` to `marker_root`, enforcing GPU-level pixel clipping at the exact DisplayRegion boundaries. Tested at edge/corner coordinates and heading angles pointing outward.
6. **Immediate DisplayRegion Tracking & Robust Rollback (HUD1-R2-02):** Captured `display_region` reference immediately after `makeCamera()`. `destroy()` checks both `self.display_region` and `camera.node().getDisplayRegion(0)` to guarantee clean teardown on partial failure.
7. **Resize Consistency across Camera, Backing, and Marker (HUD1-R1-02 & HUD1-R2-03):** `_check_window_resize()` recalculates film size, updates `lens.setFilmSize()`, dynamically scales `self.background.setScale()`, and updates marker scale without recreating cameras or DisplayRegions.
8. **Trajectory Sampling Independence (HUD1-R1-04):** Position validation and trajectory appending execute prior to camera projection and screen clipping, preserving traveled path even when the vehicle temporarily leaves the viewport.
9. **Standalone Sandbox Demo (HUD1-R1-06):** Implemented `scripts/demo_minimap_native.py` with cleanly labeled scripted constant driving (`continuous_box2_v1`), without creating unused `AgentPolicy` instances.
10. **Visual Evidence (HUD1-R1-07):** Captured and committed two real desktop Panda3D screenshots under `docs/visualization/evidence/` demonstrating near-start and after-movement states.
11. **Unit Test Suite (HUD1-R1-05):** Implemented `tests/test_minimap_overlay.py` with 22 dedicated test cases exercising actual component methods under a mock Panda3D environment; full repository test suite verified at **424 tests (423 passed, 1 skipped informational, 0 failures, 0 errors — 100% OK)**.
12. **Complete Scientific Isolation:** Zero modifications to `src/launcher/executor.py`, zero modifications to `configs/platform/` or frozen contracts, and strict headless isolation for benchmark suites.

---

## 2. Adoption of Existing Minimap & Attribution

### 2.1 Attribution to Đặng Khoa
In accordance with Gate HUD-0 findings, HUD-1 adopts and enhances the core rendering architecture designed by Đặng Khoa on branch `origin/khoa` (commit `271d704`):
- Secondary Panda3D camera with `OrthographicLens` and `CamMask.MiniMap`.
- Top-left `DisplayRegion` viewport.
- Dynamic road feature polyline bounding box calculation.
- Safe resource disposal on environment close.

### 2.2 Resolved Deficiencies & Enhancements Implemented
| Component | Khoa Baseline (`origin/khoa:271d704`) | Gate HUD-1 Enhanced Implementation |
|---|---|---|
| **Package Imports** | Prematurely imported `FinishFloatingHologram` causing `ModuleNotFoundError`. | Clean `__init__.py` exporting strictly verified visualizer modules. |
| **Heading Marker** | Static red circle (`radius_px=3.0`); no orientation feedback. | Directional cyan chevron rotating to `heading_theta`; non-directional dot fallback when heading is None. |
| **Trajectory Trail** | Not implemented. | Bounded ring buffer ($K \le 500$) with $0.5\,\text{m}$ minimum distance filter rendered via `LineSegs`. |
| **Update Mechanism** | Asynchronous task spawned in Panda3D `taskMgr`. | Explicit step-driven `update(world_x, world_y, heading_rad)` without thread contention. |
| **Background Styling**| Opaque olive green/gray (`0.16, 0.20, 0.16`). | Translucent dark glass (`#0B131B` @ 0.75 alpha) per approved HUD-0 brief. |
| **Math Testability** | Embedded directly in Panda3D node construction. | Factored pure math into `minimap_math.py` for headless unit testability. |

---

## 3. Canonical HUD-1 Interface & Architecture

### 3.1 Class Signature (`src/visualization/minimap_overlay.py`)
```python
class MiniMapOverlay:
    REGION = (0.015, 0.185, 0.745, 0.985)  # Top-Left Normalized Screen Coordinates

    def __init__(self, env: Any, padding_m: float = 12.0) -> None:
        """Configures OrthographicLens camera, DisplayRegion, background, and markers."""
        ...

    def update(
        self,
        world_x: Optional[float],
        world_y: Optional[float],
        heading_rad: Optional[float] = None
    ) -> None:
        """
        Step-driven decoupled pose update:
        - Only uses the passed arguments (strictly zero implicit fallback to env.agent).
        - If world_x or world_y is None/non-finite: hides marker without appending to trajectory.
        - Trajectory sampling is independent of camera projection / screen clipping.
        - If position is valid: evaluates trajectory distance filter (>= 0.5m) and updates trail buffer.
        - If position is within viewport: displays marker (oriented chevron if heading is valid, non-directional dot if None).
        - If position is outside viewport: hides marker.
        """
        ...

    def reset(self) -> None:
        """Clears trajectory buffer to 0 points; removes trail geometry; hides marker."""
        ...

    def destroy(self) -> None:
        """Safely and idempotently tears down DisplayRegion, camera, and scene nodes."""
        ...
```

### 3.2 Visual & Mathematical Invariants
1. **Heading Direction Mapping:**
   In MetaDrive, heading $\theta = 0$ is along $+X$ (East/Right), and $\theta = \frac{\pi}{2}$ is along $+Y$ (North/Up).
   In Panda3D `render2d`, $+X$ is rightward and $+Z$ is upward.
   Heading rotation maps directly to Panda3D roll via `calculate_heading_rotation_deg(theta) = -math.degrees(theta)`, verified at cardinal angles:
   - East ($\theta = 0^\circ$): Tip at $(+1.8 r_x, 0)$, points Right.
   - North ($\theta = 90^\circ$): Rotates $-90^\circ$, points Up.
   - West ($\theta = 180^\circ$): Rotates $-180^\circ$, points Left.
   - South ($\theta = -90^\circ$): Rotates $+90^\circ$, points Down.
2. **Trajectory Trail Buffer:**
   - Implemented via `TrajectoryTrailBuffer(max_capacity=500)` in `minimap_math.py`.
   - New samples are appended if Euclidean distance from the preceding point is $\ge 0.5\,\text{m}$.
   - Retained waypoint coordinates match raw simulator coordinates within floating-point precision ($< 1\text{e-4}\,\text{m}$).
   - The $0.5\,\text{m}$ threshold is a lower bound to filter stationary vehicle jitter; it does not constrain high-speed vertex separation.
   - When the buffer reaches 500 points, it functions as a sliding FIFO window of the most recent 500 points.
3. **Screen Clipping & Sampling Independence (HUD1-R1-04):**
   Vehicle world positions are projected through the orthographic camera lens. Trajectory trail sampling occurs immediately upon position validation, independent of whether the vehicle marker is inside or outside the minimap viewport. If the projected coordinate falls outside the minimap `[0.0, 1.0] \times [0.0, 1.0]` viewport, the marker is hidden, preventing visual spillover into the main 3D view while preserving the continuity of the traveled path.

---

## 4. Standalone Sandbox Demo & Real Visual Evidence

### 4.1 Execution Details
- Script: `scripts/demo_minimap_native.py`
- Driving Policy: Scripted Constant Drive (steering=0.0, throttle=0.4 via `continuous_box2_v1`; zero unused `AgentPolicy` instantiation).
- Target Case: `sandbox/Easy/SCCS/0/0` (TRAIN split, Easy tier, SCCS geometry).
- Environment: MetaDrive 0.4.3 (commit `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`), Panda3D 1.10.14, Python 3.10.
- Window Resolution: $1200 \times 900$ (default MetaDrive NATIVE window).

### 4.2 Runtime Step Log:
```text
=================================================================
XETUHANH — HUD-1 LIVE MINIMAP STANDALONE DEMO (SANDBOX)
=================================================================
  Launcher Mode:       SANDBOX (Exploratory TRAIN only)
  Render Mode:         NATIVE
  Difficulty Tier:     Easy
  Driving Policy:      Scripted Constant Drive (steering=0.0, throttle=0.4)
  Action Adapter:      continuous_box2_v1
  Max Steps:           40
  Resolved Case ID:    sandbox/Easy/SCCS/0/0 (Split=TRAIN)

--- Initializing MetaDrive Environment ---
--- Initializing MiniMapOverlay in Panda3D Window ---
  Minimap Viewport:    (0.015, 0.185, 0.745, 0.985)
  Calculated Bounds:   (-32.77344269042748, 196.3179421956629, -193.15448628403567, 19.25)
  Film Dimensions:     253.1m x 268.0m

--- Stepping Simulation (40 steps @ 10 Hz) ---
  Step   1/40 | Pos=(   5.0,    7.0) | Heading=  +0.0° | Speed=  0.4 km/h | Route= 0.9% | TrailPts=  1
  Step  20/40 | Pos=(   7.1,    7.0) | Heading=  +0.0° | Speed=  8.0 km/h | Route= 1.2% | TrailPts=  5
  Step  40/40 | Pos=(  13.7,    7.0) | Heading=  +0.0° | Speed= 15.9 km/h | Route= 2.3% | TrailPts= 14

[DEMO COMPLETE] Stepped 40 simulation steps successfully.
  [CLEANUP] MiniMapOverlay destroyed safely.
  [CLEANUP] MetaDriveEnv closed safely.
```

### 4.3 Visual Verification from Real Screen Captures (Committed Evidence)
Two real desktop Panda3D screenshots captured directly from the live native window session are committed under `docs/visualization/evidence/`:

1. **Near Start State (`docs/visualization/evidence/hud1_minimap_near_start.png`, 1,172,608 bytes):**
   - Captured at Step 2 (`Pos=(5.0, 7.0)`, heading `0.0°`).
   - Minimap in top-left displays full SCCS track loop.
   - Electric cyan directional chevron marker is positioned at the start line, pointing rightward along the track.
   - Trajectory trail buffer contains 1 initial sample (no extended line yet).

2. **After Movement State (`docs/visualization/evidence/hud1_minimap_after_movement.png`, 1,219,377 bytes):**
   - Captured at Step 40 (`Pos=(13.7, 7.0)`, speed $15.9\,\text{km/h}$).
   - Cyan chevron marker has progressed forward along the road lane.
   - An extended, glowing cyan polyline (14 distance-filtered trajectory vertices) stretches behind the vehicle, clearly visualizing traveled route history.
   - On-screen display readout: **60.9 FPS** (observed onscreen display rate; distinguished from headless benchmark throughput).
   - Glass Transparency Note: The minimap background quad renders with a translucent dark tint (`#0B131B` @ 0.75 alpha). Multi-layer glass compositing and border styling will be further refined in HUD-2.

---

## 5. Unit Test Suite Results

### 5.1 Dedicated HUD-1 Test Suite (`tests/test_minimap_overlay.py`)
All 22 required test cases passed in 0.098s:
1. `test_01_import_visualization_package_has_no_hologram_dependency`: **PASS**
2. `test_02_headless_import_creates_no_engine_or_window`: **PASS**
3. `test_03_map_bounds_extraction_from_polylines`: **PASS**
4. `test_04_empty_and_malformed_features_fallback`: **PASS**
5. `test_05_nan_and_inf_geometry_vertices_rejected`: **PASS**
6. `test_06_orthographic_film_size_preserves_aspect_ratio`: **PASS**
7. `test_07_screen_to_world_projection_math`: **PASS**
8. `test_08_heading_orientation_cardinal_directions`: **PASS**
9. `test_09_valid_position_with_missing_heading_renders_dot`: **PASS**
10. `test_10_missing_position_hides_marker`: **PASS**
11. `test_11_first_trajectory_sample_appended`: **PASS**
12. `test_12_minimum_distance_threshold_filters_stationary_jitter`: **PASS**
13. `test_13_large_displacement_retained_correctly`: **PASS**
14. `test_14_ring_buffer_fifo_bounded_at_500_points`: **PASS**
15. `test_15_reset_clears_trajectory_buffer_zero_cross_episode_leakage`: **PASS**
16. `test_16_no_unbounded_panda3d_nodes_on_repeated_updates`: **PASS**
17. `test_17_renderer_screen_clipping_bounds_check`: **PASS**
18. `test_18_window_resize_film_recalculation`: **PASS**
19. `test_19_destroy_idempotency_repeated_calls_safe`: **PASS**
20. `test_20_partial_initialization_cleanup_on_exception`: **PASS**
21. `test_21_no_background_taskmgr_update_task_spawned`: **PASS**
22. `test_22_scientific_contracts_and_hashes_unmutated`: **PASS**

### 5.2 Full Repository Regression Test Pass
Ran all 12 test modules in active conda `metadrive` environment:
- Total Tests: **424**
- Passed: **423**
- Skipped: **1** (`test_44_100_runs_discovery_performance_informational_only` in `test_workbench_hardening` — informational benchmark test)
- Failures: **0**
- Errors: **0**
- Result: **100% PASS (Zero Regressions)**

---

## 6. Scientific Integrity & Boundary Preservation

- **Production Launcher Untouched:** `src/launcher/executor.py` was NOT modified in Gate HUD-1. It remains in its clean Gate R0 state until HUD-3.
- **Frozen Contract Immutability:** Verified zero changes to `configs/platform/`, `datasets/mapsuite_v1/`, and locked `results/audits/`.
- **Benchmark Protection:** Canonical `TEST` and `VALIDATION` modes remain strictly headless (`render_mode="OFF"`). The minimap is instantiated only in `SANDBOX` mode with `render_mode="NATIVE"`.
- **Artifact Protection:** Trajectory trail coordinates are held strictly in memory and are never persisted to `episodes.csv`, `timing.csv`, or `summary.json`.

---

## 7. Residual Risks & Next Steps (Handoff to HUD-2)

1. **HUD-2 Scope:** Gate HUD-2 will implement the remaining six widgets on `base.aspect2d`: Route Progress Bar, Large Digital Speedometer, Left/Right Turn Indicators, and Brake/Throttle Meters.
2. **Executor Speed Defect Resolution:** In HUD-2, the telemetry extraction helper will defensively query `getattr(env.agent, "speed_km_h", None)` to resolve the pre-existing executor `speed_kmh` defect.
3. **Panda3D Window Native Testing:** Standalone demo script `scripts/demo_minimap_native.py` provides the mechanism for visual verification during interactive desktop sessions.

---

## 8. Final Verdict

```
========================================================================================
VERDICT: HUD-1 IMPLEMENTED — READY FOR INDEPENDENT REVIEW
========================================================================================
- Branch: feat/hud1-live-minimap
- Base Commit: c5107b84d9e166e276e34ece1a90f11675180aa0 (PR #30 Merge)
- Implementation PR: https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/pull/31
- Status: DRAFT / UNMERGED — Ready for independent audit by reviewer.
========================================================================================
```
