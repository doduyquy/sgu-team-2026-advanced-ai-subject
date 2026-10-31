# Platform V1 — Existing Minimap Feasibility Audit

- **Document ID:** `AUDIT-MINIMAP-FEASIBILITY-V1`
- **Version:** `1.0.0`
- **Status:** `DESIGN-AUDIT / READY FOR REVIEW`
- **Date:** 2026-10-31
- **Lead / Stage 5 + GUI:** Hữu Tri
- **Baseline Git Commit:** `581dc49570ceb00fff13cf8df073a76e654735df` (PR #29 Merge on `main`)
- **Audited Source Commit:** `271d704bdf04c64c869a48237375a36bbae05750` on `origin/khoa`
- **Subproject Root:** `autonomous-driving-rl/`

---

## 1. Executive Summary

This feasibility audit inspects existing visualizer and minimap implementations across accessible branches to prevent redundant work, preserve team effort (specifically work by Đặng Khoa on Stage 0), and evaluate technical feasibility for the upcoming **HUD-1 (Live Minimap)** gate.

### Key Audit Findings:
1. **Source Discovery:** An existing `MiniMapOverlay` implementation was successfully located on remote branch `origin/khoa` at commit `271d704bdf04c64c869a48237375a36bbae05750` (`autonomous-driving-rl/src/visualization/minimap_overlay.py`, verified exactly 372 lines in source).
2. **Package Dependency Trap:** In commit `271d704`, `src/visualization/__init__.py` attempted to import `FinishFloatingHologram`, which was not committed until a later commit (`db68db0`). Blindly importing or porting the package at commit `271d704` triggers `ModuleNotFoundError`. HUD-1 must isolate `minimap_overlay.py` without coupling to unneeded hologram assets.
3. **Branch Provenance:** Commit `271d704` is an intermediate development commit; the latest HEAD on `origin/khoa` is `9a36c313eb3ccdbc671842a81ddb1769bd6ae747` (containing subsequent agent development and experiment archiving).
4. **Current `main` State:** `src/visualization/` was not present on `main` when Gate R0 executed; Gate R0 correctly eliminated the dangling import from `src/launcher/executor.py` to restore test and execution integrity.
5. **Reusability Assessment:** The existing implementation provides a viable foundational architecture: a secondary Panda3D GPU camera (`OrthographicLens`) rendering directly into a `DisplayRegion` on the native window. However, it requires substantive enhancements for heading orientation, trajectory trail, transparent glass styling, and step synchronization.
6. **Required Enhancements for HUD-1:**
   - Add vehicle heading orientation (replace static circular dot with directional chevron/arrow).
   - Add historical trajectory trail (FIFO point buffer).
   - Upgrade visual styling from opaque olive-gray to transparent dark glass with cyan accents per the approved design brief.
   - Synchronize updates with simulator step cycle rather than autonomous background tasks.

---

## 2. Minimap Inventory & Provenance

### 2.1 Branch & Commit Inventory
A full scan across git history and remote branches identified the following artifacts:

| Branch / Location | Commit SHA | Date & Author | Artifact Path | Description |
|---|---|---|---|---|
| `origin/khoa` (Intermediate) | `271d704bdf04c64c869a48237375a36bbae05750` | 2026-10-31 11:39:12<br>Khoa `<khoanguyen12062005@gmail.com>` | `src/visualization/minimap_overlay.py` (372 lines)<br>`src/visualization/__init__.py` | Original `MiniMapOverlay` implementation using Panda3D `DisplayRegion` and `OrthographicLens`. *(Note: `__init__.py` in this commit had a broken import referencing `FinishFloatingHologram` before that file was committed)*. |
| `origin/khoa` (Intermediate) | `db68db00006f9dbb303d70fc6fb915006a6855e6` | 2026-10-31 11:41:39<br>Khoa | `src/visualization/finish_floating_hologram.py` | 3D floating finish line hologram banner placed at route terminus. |
| `origin/khoa` (Latest HEAD) | `9a36c313eb3ccdbc671842a81ddb1769bd6ae747` | 2026-10-31 13:31:58<br>Khoa | Multiple directories | Branch HEAD on `khoa` containing subsequent agent maintenance, archived development snapshots, and reports. |
| `main` | `581dc49570ceb00fff13cf8df073a76e654735df` | 2026-10-31<br>PR #29 Merge | *None (Clean)* | Clean baseline. `src/visualization/` is absent, unblocking headless tests and CI. |

*Handoff ZIP Note:* File `XeTuHanh_HUD_Handoff_2026-10-31.zip` was not present on the local execution machine filesystem. All design requirements and visual constraints are derived directly from the user-approved brief specified in the HUD-0 charter.

---

## 3. Technical Audit of Existing `MiniMapOverlay`

### 3.1 Class Architecture & Design
The implementation on `origin/khoa` defines `class MiniMapOverlay`:

```text
MiniMapOverlay
├── REGION = (0.015, 0.18, 0.755, 0.975)  [Normalized screen coords: Top-Left]
├── __init__(env, padding_m=12.0)
│   ├── _calculate_map_bounds(env)         [Extracts road bounds from polylines]
│   ├── _create_camera(bounds, padding_m)  [Configures Panda3D OrthographicLens]
│   ├── _create_background(...)           [Creates background quad with CamMask]
│   └── _create_vehicle_marker(env)        [Creates 2D screen marker in render2d]
├── update()                               [Projects ego world (x,y) to screen marker]
└── destroy()                              [Cleans up taskMgr, displayRegion, camera]
```

### 3.2 Detailed Mechanism Breakdown

1. **Screen Placement & Aspect Ratio:**
   - Defined as `REGION = (0.015, 0.18, 0.755, 0.975)` in normalized display region coordinates (`left, right, bottom, top` in $[0, 1]$).
   - Screen footprint: Width $\approx 16.5\%$ of window, Height $\approx 22\%$ of window.
   - Location: **Top-Left corner**, exactly matching Item 1 of the user-approved design brief.
   - Calculates region aspect ratio against map aspect ratio, dynamically adjusting the orthographic film size (`film_width`, `film_height`) so the map is never distorted or stretched.

2. **Map Feature Extraction:**
   - Queries `env.current_map.get_map_features(interval=2.0)`.
   - Concatenates polylines of road lanes and dividers to compute bounding box `(min_x, max_x, min_y, max_y)`.
   - Fallback: Uses `env.agent.position \pm 50.0\,\text{m}` if polylines are empty.

3. **Camera & Scene Masking:**
   - Creates camera with `self.engine.makeCamera(win, sort=5, displayRegion=REGION)`.
   - Uses `OrthographicLens` pointing straight down: `setPos(center_x, center_y, height)`, `setHpr(0.0, -90.0, 0.0)`.
   - **Critical Finding:** Uses `lens.setCameraMask(CamMask.MiniMap)`. In MetaDrive 0.4.3, `CamMask.MiniMap` isolates road surface geometries, preventing terrain and skybox artifacts from cluttering the minimap view.

4. **Ego Marker Projection:**
   - Projects 3D world coordinate `(x, y, 0)` into camera space, then uses `lens.project()` to obtain normalized device coordinates `(projected.x, projected.y)`.
   - Maps projection to screen coordinate space and sets `marker.setPos(screen_x, 0.0, screen_z)` on a 2D node attached to `render2d`.
   - Sort order `bin="fixed", draw_order=221` ensures the marker floats visibly above the 3D scene.

5. **Teardown & Cleanup:**
   - Removes Panda3D task `_marker_task_name` from `taskMgr`.
   - Removes `marker` and `background` nodes.
   - Calls `win.removeDisplayRegion(display_region)` and removes camera from `camList`.

---

## 4. Evaluation: What Works vs What Needs Work

| Architectural Dimension | Existing Implementation (`origin/khoa`) | Assessment | Required Action for HUD-1 |
|---|---|---|---|
| **Top-Left Screen Anchor** | `REGION = (0.015, 0.18, 0.755, 0.975)` | **PASS** | Keep exact anchor; wrap with glass border styling. |
| **GPU Hardware Acceleration** | Direct secondary camera in Panda3D | **EXCELLENT** | Preserve architecture. No Pygame software blitting needed. |
| **Aspect Ratio Locking** | Dynamic orthographic film scaling | **PASS** | Preserve aspect locking logic. |
| **Map Feature Bounding** | Polylines from `get_map_features(2.0)` | **QUALIFIED PASS** | Compatible with polyline road extraction in tested scenarios; requires runtime verification across MapSuite geometry archetypes in HUD-1. |
| **Camera Masking** | `CamMask.MiniMap` | **PASS** | Verified correct for MetaDrive 0.4.3. |
| **Vehicle Heading** | Static 2D circle (`radius_px=3.0`) | **NEEDS WORK** | Replace circle with directional arrow/chevron rotated by heading angle. |
| **Trajectory Trail** | Not implemented | **MISSING** | Add FIFO buffer of traveled coordinates rendered via `LineSegs`. |
| **Visual Aesthetic** | Opaque olive background `(0.16, 0.20, 0.16)` | **NEEDS WORK** | Replace with semi-transparent dark glass (`#0B131B` @ 0.75 alpha) and cyan border. |
| **Update Synchronization** | Spawns background task in `taskMgr` | **RISK** | Switch to explicit step-driven update inside simulation loop to avoid thread races. |
| **Scientific Isolation** | Only active when `use_render=True` | **PASS** | Completely disabled in headless mode (`render_mode="OFF"`). |

---

## 5. Adoption & Enhancement Plan for HUD-1

### 5.1 Architecture Recommendation
Rather than rewriting the minimap from scratch, **HUD-1 should adopt Khoa's `MiniMapOverlay` as the foundational baseline** and apply four targeted enhancements:
1. **Directional Heading Pointer:** Replace the static circle with a 3-vertex directional triangle oriented by `env.agent.heading_theta` in minimap space.
2. **Path Trail Buffer:** Add an in-memory ring buffer (maximum 500 points) recording `(x, y)` coordinates, rendered into `render2d` via `LineSegs` with a fading cyan gradient.
3. **Transparent Dark Glass Theme:** Update the background quad color and shader to translucent dark slate with a 1.5 px cyan/white frame border per the approved visual brief.
4. **Step-Driven API:** Expose `update(telemetry_snapshot)` called explicitly by the simulation harness after physics step, eliminating background `taskMgr` overhead.

---

### 5.2 Team Coordination Checklist with Khoa
To ensure smooth integration and prevent duplicate work or branch conflicts:
1. **Targeted Extraction:** In HUD-1, port strictly `minimap_overlay.py` as a standalone module into `src/visualization/`. Do NOT import or depend on `finish_floating_hologram.py` or experimental Stage 0 components.
2. **Clean Package Interface:** Ensure `src/visualization/__init__.py` exports strictly verified modules (`MiniMapOverlay`), eliminating the premature `FinishFloatingHologram` import present in commit `271d704`.
3. **Branch Awareness:** Acknowledge that branch `origin/khoa` has evolved beyond commit `271d704` to HEAD `9a36c31`. Coordinate with Khoa before modifying shared agent or experiment paths.
4. **Empirical MapSuite Testing:** Test minimap polyline extraction across representative geometry archetypes (straight, curve, intersection, roundabout) rather than assuming all 240 geometries render identically without verification.

---

## 6. Audit Verdict

```
========================================================================================
MINIMAP AUDIT VERDICT: EXISTING IMPLEMENTATION VERIFIED ON ORIGIN/KHOA (COMMIT 271d704)
========================================================================================
- Structural Viability: High reusability for core camera, bounding, and DisplayRegion logic.
- Strategy: Adopt and enhance in HUD-1; do NOT reimplement from scratch.
- Current main: Remains clean; implementation will be introduced via dedicated PR for HUD-1.
========================================================================================
```
