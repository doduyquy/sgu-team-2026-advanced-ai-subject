"""
Racing HUD Manager Facade for Runtime Simulation Integration (Gate HUD-3).

Coordinates the complete 7-component visual experience:
- MiniMapOverlay (HUD-1): Top-left orthographic road geometry, ego marker, and trajectory trail.
- RacingHud (HUD-2): Route progress, digital speedometer, turn indicators, and pedal meters.

Adheres strictly to the Gate HUD-3 architectural invariants:
- Instantiated strictly when render_mode == "NATIVE" in SANDBOX mode.
- Consumes strictly read-only telemetry extracted post-step (after env.step()).
- Preserves absolute timing isolation around agent.act().
- Cleaned up safely in executor finally block prior to env.close().
- Records empirical update latencies for performance verification.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.visualization.minimap_overlay import MiniMapOverlay
from src.visualization.racing_hud import RacingHud
from src.visualization.telemetry_model import (
    HudTelemetrySnapshotV1,
    create_telemetry_snapshot,
)


class RacingHudManager:
    """
    Facade managing both MiniMapOverlay and RacingHud instances for an active simulation episode.
    """

    def __init__(self, env: Any) -> None:
        """
        Initializes the integrated HUD facade for an active MetaDrive environment.

        Args:
            env: Active MetaDriveEnv instance with an active window and engine.
        """
        if env is None:
            raise RuntimeError("Environment is None; RacingHudManager requires active MetaDriveEnv.")

        self.env = env
        self.minimap: Optional[MiniMapOverlay] = None
        self.hud: Optional[RacingHud] = None
        self._is_destroyed = False
        self._update_latencies_ms: List[float] = []
        self._last_snapshot: Optional[HudTelemetrySnapshotV1] = None

        # Wrap visualizer instantiation in an exception-safe rollback block
        try:
            self.minimap = MiniMapOverlay(env)
            engine = getattr(env, "engine", None)
            if engine is None:
                raise RuntimeError("MetaDrive engine is not available on environment.")
            self.hud = RacingHud(engine)
        except Exception:
            self.destroy()
            raise

    # ==========================================================================
    # Update Interface
    # ==========================================================================

    def update(
        self,
        step_idx: int,
        control_dt_s: float,
        canonical_steering: Optional[float] = None,
        canonical_throttle_brake: Optional[float] = None,
        step_info: Optional[Dict[str, Any]] = None,
        outcome: Optional[Any] = None,
    ) -> HudTelemetrySnapshotV1:
        """
        Extracts authoritative telemetry from simulator state, creates an immutable
        HudTelemetrySnapshotV1, routes to visualizers, and tracks update duration.

        Executed strictly after physics step and outcome classification, completely
        outside the agent decision latency timing bracket.
        """
        if self._is_destroyed:
            raise RuntimeError("Cannot update a destroyed RacingHudManager.")

        t0 = time.perf_counter_ns()

        # 1. Extract authoritative telemetry without fabricating fallbacks
        pos = getattr(self.env.agent, "position", None) if hasattr(self.env, "agent") else None
        heading = getattr(self.env.agent, "heading_theta", None) if hasattr(self.env, "agent") else None
        speed = getattr(self.env.agent, "speed_km_h", None) if hasattr(self.env, "agent") else None
        route_comp = step_info.get("route_completion") if isinstance(step_info, dict) else None
        sim_time = step_idx * float(control_dt_s)

        terminal_status = "UNDETERMINED"
        clean_succ = None
        if outcome is not None:
            reason = getattr(outcome, "primary_reason", None)
            if reason is not None and hasattr(reason, "value"):
                terminal_status = str(reason.value)
            elif reason is not None:
                terminal_status = str(reason)
            else:
                terminal_status = "TERMINAL"
            clean_succ = getattr(outcome, "clean_success", None)

        # 2. Build immutable snapshot
        snapshot = create_telemetry_snapshot(
            step_index=step_idx,
            simulation_time_s=sim_time,
            world_position=pos,
            heading_theta_rad=heading,
            speed_kmh=speed,
            route_completion=route_comp,
            canonical_steering=canonical_steering,
            canonical_throttle_brake=canonical_throttle_brake,
            terminal_outcome_status=terminal_status,
            is_clean_success=clean_succ,
        )
        self._last_snapshot = snapshot

        # 3. Route to visualizers
        self._route_snapshot(snapshot)

        t1 = time.perf_counter_ns()
        self._update_latencies_ms.append((t1 - t0) / 1e6)

        return snapshot

    def update_with_snapshot(self, snapshot: HudTelemetrySnapshotV1) -> None:
        """Directly applies a pre-constructed snapshot to visualizers (e.g. for testing/playback)."""
        if self._is_destroyed:
            raise RuntimeError("Cannot update a destroyed RacingHudManager.")

        t0 = time.perf_counter_ns()
        self._last_snapshot = snapshot
        self._route_snapshot(snapshot)
        t1 = time.perf_counter_ns()
        self._update_latencies_ms.append((t1 - t0) / 1e6)

    def _route_snapshot(self, snapshot: HudTelemetrySnapshotV1) -> None:
        """Dispatches snapshot data to MiniMapOverlay and RacingHud."""
        if self.hud is not None:
            self.hud.update(snapshot)

        if self.minimap is not None:
            wx = snapshot.world_position[0] if snapshot.world_position is not None else None
            wy = snapshot.world_position[1] if snapshot.world_position is not None else None
            self.minimap.update(world_x=wx, world_y=wy, heading_rad=snapshot.heading_theta_rad)

    # ==========================================================================
    # Lifecycle Interface
    # ==========================================================================

    def reset(self) -> None:
        """Resets both visualizers for a new episode without destroying nodes."""
        if self._is_destroyed:
            return
        if self.minimap is not None:
            self.minimap.reset()
        if self.hud is not None:
            self.hud.reset()
        self._last_snapshot = None

    def destroy(self) -> None:
        """
        Idempotently and cleanly tears down both visualizers.
        Guaranteed to leave zero orphaned nodes or tasks.
        """
        self._is_destroyed = True

        if self.hud is not None:
            try:
                self.hud.destroy()
            except Exception:
                pass
            self.hud = None

        if self.minimap is not None:
            try:
                self.minimap.destroy()
            except Exception:
                pass
            self.minimap = None

    # ==========================================================================
    # Performance & Diagnostics
    # ==========================================================================

    @property
    def update_latencies_ms(self) -> List[float]:
        """Returns the list of per-step HUD update durations in milliseconds."""
        return list(self._update_latencies_ms)

    @property
    def last_snapshot(self) -> Optional[HudTelemetrySnapshotV1]:
        """Returns the most recent telemetry snapshot consumed."""
        return self._last_snapshot

    def get_latency_statistics(self) -> Dict[str, float]:
        """Calculates mean, p50, p95, and p99 update latencies in milliseconds."""
        if not self._update_latencies_ms:
            return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
        arr = np.array(self._update_latencies_ms, dtype=np.float64)
        return {
            "mean_ms": float(np.mean(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(np.max(arr)),
        }
