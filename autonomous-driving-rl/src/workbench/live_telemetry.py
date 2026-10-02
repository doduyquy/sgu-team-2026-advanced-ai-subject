"""
Live Telemetry Buffer and Matplotlib Integration Models (Gate 7.5B Pass B2).

Maintains in-memory live telemetry traces purely from LauncherEventV1 messages:
- LiveTelemetryBufferV1: strictly consumes LauncherEventV1 payloads.
- Reset semantics:
  - RUN_STARTED resets entire run buffer.
  - EPISODE_STARTED resets current episode trace.
  - EPISODE_PROGRESS appends (step_index, route_completion, speed_kmh).
  - EPISODE_FINISHED finalizes episode.
  - RUN_FINISHED / RUN_FAILED / RUN_INTERRUPTED marks run completion state.
- Zero direct access to simulator or agent objects.
- Zero free-text message parsing for scientific outcomes.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EpisodeTelemetryTrace:
    """Sampled time-series telemetry for a single episode."""
    episode_index: int
    step_indices: List[int] = field(default_factory=list)
    route_completions: List[float] = field(default_factory=list)
    speeds_kmh: List[float] = field(default_factory=list)
    is_finished: bool = False
    final_status: Optional[str] = None


class LiveTelemetryBufferV1:
    """
    Presentation buffer storing live sampled telemetry emitted via LauncherEventV1.
    Decoupled from simulation execution.
    """

    def __init__(self):
        self._current_run_id: Optional[str] = None
        self._run_status: str = "IDLE"
        self._current_episode_trace: Optional[EpisodeTelemetryTrace] = None
        self._completed_episode_traces: List[EpisodeTelemetryTrace] = []

    @property
    def current_run_id(self) -> Optional[str]:
        return self._current_run_id

    @property
    def run_status(self) -> str:
        return self._run_status

    @property
    def current_episode_trace(self) -> Optional[EpisodeTelemetryTrace]:
        return self._current_episode_trace

    @property
    def completed_episode_traces(self) -> List[EpisodeTelemetryTrace]:
        return list(self._completed_episode_traces)

    def reset(self) -> None:
        """Resets all live run telemetry."""
        self._current_run_id = None
        self._run_status = "IDLE"
        self._current_episode_trace = None
        self._completed_episode_traces.clear()

    def handle_event(self, event_dict: Dict[str, Any]) -> None:
        """
        Consumes a single LauncherEventV1 payload dictionary.
        Does NOT parse scientific outcome states from free-text 'message'.
        """
        event = event_dict.get("event", event_dict)
        etype = event.get("event_type")
        run_id = event.get("run_id")

        if etype == "RUN_STARTED":
            self.reset()
            self._current_run_id = run_id
            self._run_status = "RUNNING"

        elif etype == "EPISODE_STARTED":
            ep_idx = int(event.get("episode_index", 0))
            self._current_episode_trace = EpisodeTelemetryTrace(episode_index=ep_idx)

        elif etype == "EPISODE_PROGRESS":
            if self._current_episode_trace is not None:
                step = event.get("step_index")
                route = event.get("route_completion")
                speed = event.get("speed_kmh")
                if step is not None and route is not None:
                    self._current_episode_trace.step_indices.append(int(step))
                    self._current_episode_trace.route_completions.append(float(route))
                    self._current_episode_trace.speeds_kmh.append(float(speed or 0.0))

        elif etype == "EPISODE_FINISHED":
            if self._current_episode_trace is not None:
                self._current_episode_trace.is_finished = True
                self._current_episode_trace.final_status = event.get("status")
                self._completed_episode_traces.append(self._current_episode_trace)

        elif etype == "RUN_FINISHED":
            self._run_status = "COMPLETE"

        elif etype == "RUN_FAILED":
            self._run_status = "FAILED"

        elif etype == "RUN_INTERRUPTED":
            self._run_status = "INTERRUPTED"
