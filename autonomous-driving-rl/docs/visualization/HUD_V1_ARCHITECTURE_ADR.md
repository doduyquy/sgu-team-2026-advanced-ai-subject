# ADR: Native Racing HUD Rendering Architecture (HUD V1)

- **Document ID:** `ADR-HUD-V1-ARCHITECTURE`
- **Version:** `1.0.0`
- **Status:** `PROPOSED ARCHITECTURE / READY FOR REVIEW`
- **Date:** 2026-10-31
- **Lead / Stage 5 + GUI:** Hữu Tri
- **Baseline Git Commit:** `581dc49570ceb00fff13cf8df073a76e654735df`
- **Subproject Root:** `autonomous-driving-rl/`

---

## 1. Context and Problem Statement

When evaluating autonomous driving policies in MetaDrive native 3D mode (`render_mode="NATIVE"` in `SANDBOX`), human researchers and evaluators need clear real-time situational awareness. The user-approved design brief mandates a **Minimal Transparent Racing HUD** comprising seven specific components:
1. Live Minimap (top-left)
2. Route Progress Bar (top-center)
3. Large Digital Speedometer (bottom-right)
4. Left Turn Indicator (bottom-left)
5. Right Turn Indicator (bottom-left)
6. Brake Pedal Meter (bottom-right, left of speed)
7. Throttle Pedal Meter (bottom-right, left of speed)

We must select the rendering architecture for this HUD that achieves high visual fidelity, target 60 FPS performance, strict scientific isolation, and maximum reuse of existing team implementations (specifically the `MiniMapOverlay` developed by Đặng Khoa).

---

## 2. Decision Candidates

We evaluate three architectural candidates:

### Option A: Native Panda3D 2D Scene Graph Overlay
- Render the entire HUD (gauges, progress, meters) directly inside the MetaDrive Panda3D window using Panda3D's built-in 2D scene graphs:
  - `base.aspect2d`: For aspect-ratio-corrected 2D HUD widgets (progress bar, speedometer, pedal meters, turn indicators).
  - `base.win.makeDisplayRegion()`: A dedicated sub-region in the top-left corner coupled with a secondary top-down `OrthographicLens` camera for the minimap.
- Vector rendering via `panda3d.core.LineSegs`, text via `direct.gui.OnscreenText`, panels via `panda3d.core.CardMaker`.

### Option B: MetaDrive Top-Down Renderer Reuse (Pygame-Based)
- Utilize `metadrive.engine.top_down_renderer.TopDownRenderer` and `draw_top_down_map_native`.
- Rasterize map and vehicle trails into Pygame software surfaces on the CPU, then either open an auxiliary Pygame window or upload the Pygame surface each frame into a Panda3D `Texture` quad.

### Option C: External PySide6 / Qt Transparent Overlay
- Instantiate a frameless, transparent `QMainWindow` or `QWidget` with `Qt.WindowStaysOnTopHint` and `Qt.WA_TranslucentBackground`, positioned precisely over the OS window handle of the MetaDrive Panda3D window.
- Draw HUD widgets using Qt `QPainter` or QML.

---

## 3. Comparative Evaluation Matrix

| Evaluation Criterion | Weight | Option A: Native Panda3D | Option B: Top-Down / Pygame | Option C: External Qt Overlay |
|---|---|---|---|---|
| **MetaDrive Pinned Parity** | High | **EXCELLENT** (Uses Panda3D 1.10 directly; 100% native) | **MEDIUM** (MetaDrive's Pygame renderer is designed for separate 2D window) | **POOR** (Cross-process / cross-toolkit window coupling) |
| **Existing Minimap Reuse** | High | **EXCELLENT** (Directly adopts Khoa's `MiniMapOverlay` on `origin/khoa`) | **POOR** (Cannot reuse Khoa's Panda3D camera overlay) | **POOR** (Requires complete rewrite in Qt QPainter) |
| **Runtime Performance** | High | **DESIGN TARGET** (GPU-accelerated; zero CPU surface copies; <1.0 ms target) | **INFERRED POOR** (CPU software blitting, texture upload bottleneck across PCI-e bus) | **INFERRED MEDIUM** (OS compositing overhead, dual event loops) |
| **Scientific Isolation** | Critical | **EXCELLENT** (Strictly active on `NATIVE`; zero presence in `OFF` mode) | **EXCELLENT** (Can be gated by `NATIVE`) | **RISKY** (Requires inter-process synchronization if worker is isolated) |
| **Cross-Platform Resilience** | Medium | **EXCELLENT** (Panda3D manages single OpenGL/DirectX window across OS) | **GOOD** (Pygame is cross-platform, but multi-window causes focus bugs) | **POOR** (Transparent window layering fails frequently on Linux/Wayland) |
| **Visual Aesthetics** | Medium | **EXCELLENT** (True alpha blending, cyan glow, vector anti-aliasing) | **POOR** (Pixelated 2D blit; no modern transparent glass styling) | **EXCELLENT** (Qt stylesheet / QPainter supports rich styling) |
| **Maintenance & Simplicity** | Medium | **EXCELLENT** (Single engine, single process, clean teardown on `env.close()`) | **POOR** (Dual graphics pipelines: Panda3D + Pygame) | **POOR** (Window position tracking, focus stealing, OS z-order glitches) |

---

## 4. Detailed Technical Analysis of Candidates

### 4.1 Option A (Native Panda3D Overlay) — Technical Deep Dive
- **Scene Graph Anchors:**
  - `render2d`: Normalized 2D space $[-1.0, 1.0] \times [-1.0, 1.0]$. Ideal for full-window bounding and screen-aligned quads.
  - `aspect2d`: Uniformly scaled 2D space where $Y \in [-1.0, 1.0]$ and $X \in [-\text{aspect}, \text{aspect}]$ (e.g. $[-1.777, 1.777]$ in 16:9). Circles remain true circles and arcs do not deform when the window resizes.
- **Minimap Integration:**
  - Khoa's existing `MiniMapOverlay` on `origin/khoa` proves the feasibility of creating a secondary camera (`camName="project_minimap_camera"`) assigned to a sub-region `REGION = (0.015, 0.18, 0.755, 0.975)`.
  - The road mesh is already in GPU memory. Using `CamMask.MiniMap`, Panda3D renders the road network orthographically with hardware acceleration.
  - *Rendering Cost & Compositing Note:* A secondary camera and display region is not zero-cost; it incurs scene graph traversal, culling, transformation, and additional draw calls on the GPU. However, because geometry is resident in GPU memory and rendered via hardware rasterization, it avoids CPU software rasterization and texture upload cycles across the system bus. Furthermore, achieving true semi-transparent glass backing over the main 3D scene requires explicit framebuffer alpha configuration (`DisplayRegion.setClearColorActive(True)` with alpha or disabling color clear and using `TransparencyAttrib.MAlpha` on background quads); this compositing must be empirically verified on the pinned Panda3D runtime in HUD-1/2.
- **Teardown & Leaks:**
  - When an episode completes or the environment closes, nodes attached to `aspect2d` or `render2d` are removed cleanly with `.removeNode()`, and the secondary display region is removed via `win.removeDisplayRegion()`.

### 4.2 Why Option B (MetaDrive Pygame Renderer) is Rejected
- MetaDrive's `TopDownRenderer` relies on CPU software rasterization via Pygame onto a `pygame.Surface`.
- Blitting and transferring large software pixel surfaces across the system bus from CPU RAM to GPU VRAM on every simulation step introduces CPU-to-GPU transfer bottlenecks and memory bandwidth pressure, creating high risk of frame drops during interactive simulation.
- Pygame software surfaces do not natively integrate with Panda3D's 2D scene graph or shader-based glass-morphism aesthetic.

### 4.3 Why Option C (External Qt Overlay) is Rejected
- Creating a transparent PySide6 overlay on top of an external native Panda3D window requires tracking the exact native window coordinates, handling window move/minimize/restore events, and resolving OS-level click/focus conflicts.
- In multi-process architectures (such as the Workbench `ProcessRunner`), IPC latency between the simulation process and the GUI overlay creates visible telemetry lag.

---

## 5. Architectural Decision

**DECISION:** Adopt **Option A (Native Panda3D 2D Scene Graph Overlay)** as the official rendering architecture for HUD V1.

### Performance Budgets & Frequency Targets (Falsifiable Hypotheses):
- **Simulation Control Frequency:** Strictly 10.0 Hz nominal ($\text{dt} = 0.1\,\text{s}$ of simulated physical progression per decision step).
- **Display Refresh Target:** 60 FPS target refresh rate in native desktop Panda3D window.
- **HUD Overhead Target:** The combined telemetry extraction and HUD rendering update has an independent wall-clock design target of mean $< 1.0\,\text{ms}$ and p95 $< 2.0\,\text{ms}$ per step.
*Strict Separation of Time Domains:* A simulation timestep $\text{dt} = 0.1\,\text{s}$ represents physical simulation progression inside the virtual environment; it does not define a wall-clock execution deadline. Wall-clock execution speed depends entirely on underlying hardware throughput and simulator scheduling. Whether interactive NATIVE mode synchronizes with real-time clock pacing is a runtime behavior to be verified, not assumed. Frame rendering time, HUD update wall-clock time (`hud_update_ms`), and agent decision latency (`agent.act()`) must be instrumented and measured as separate, independent metrics in HUD-3.

### Concrete Structural Blueprint:
```text
MetaDrive Native Window (engine.win)
│
├── DisplayRegion [0.0, 1.0, 0.0, 1.0] (Main 3D Scene)
│   └── Camera: Perspective (Panda3D default render)
│       └── World: 3D vehicles, road blocks, buildings, sky
│
├── DisplayRegion [0.015, 0.185, 0.745, 0.985] (Minimap Viewport)
│   └── Camera: Orthographic (CamMask.MiniMap)
│       ├── Top-Down Road Geometry (Hardware Rendered)
│       └── Minimap Background Quad (#0B131B @ 0.75 alpha)
│
└── 2D HUD Layer (Attached to base.aspect2d, sort=50)
    ├── Minimap Glass Frame & Heading Marker
    ├── Route Progress Bar (Top-Center)
    ├── Speedometer Gauge & Numerical Readout (Bottom-Right)
    ├── Brake & Throttle Meters (Bottom-Right)
    └── Left & Right Turn Indicators (Bottom-Left)
```

---

## 6. Implementation Status Classifications

| Component / Subsystem | Status | Evidence / Notes |
|---|---|---|
| Pinned MetaDrive Native Rendering (`use_render=True`) | `VERIFIED IN SOURCE / REQUIRES RUNTIME DISPLAY TEST` | Flag verified in MetaDrive source; visual window test requires desktop session. |
| Panda3D 2D Primitives (`LineSegs`, `CardMaker`, `OnscreenText`) | `VERIFIED IN CODE` | Verified present and accessible in project Python environment. |
| Top-Left DisplayRegion Minimap | `VERIFIED IN SOURCE (origin/khoa:271d704)` | Verified in `origin/khoa:src/visualization/minimap_overlay.py` (372 lines). |
| Trajectory Trail via `LineSegs` | `PROPOSED` | To be implemented and verified in HUD-1. |
| Directional Heading Chevron | `PROPOSED` | To be implemented and verified in HUD-1. |
| Transparent Glass HUD Styling | `PROPOSED` | To be implemented and verified in HUD-2. |
| Performance Budget (<1.0 ms step overhead) | `DESIGN TARGET / REQUIRES HUD-3 TEST` | To be benchmarked empirically in HUD-3 across p50, p95, p99. |

---

## 7. Consequences and Compliance

- **Positive:**
  - Hardware-accelerated GPU rendering without CPU-to-GPU software surface transfers.
  - Zero new external runtime dependencies (Panda3D is already installed).
  - Maximizes architectural reuse from Khoa's existing minimap commit.
  - Scientific isolation by design: no HUD display regions, cameras, or nodes are allocated when `render_mode="OFF"` (to be verified as an acceptance criterion in HUD-3).
- **Negative / Constraints:**
  - Requires writing procedural 2D UI geometry using Panda3D primitives (`LineSegs`, `CardMaker`, `OnscreenText`).
  - True glass transparency requires validating DisplayRegion alpha clearing and blending on the pinned Panda3D engine.
  - Interactive verification requires a desktop session with an active display window.
