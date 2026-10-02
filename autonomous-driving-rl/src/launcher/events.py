"""
Event Telemetry Interface for Research Platform V1 Launcher (Gate 7.5A).

This module defines standard observer events emitted by ExperimentExecutor:
- Provides clean decoupling between core execution and future Gate 7.5B GUI / progress listeners.
- JSON-safe data structures suitable for inter-process queues or UI callbacks.
- Strictly isolated: Events are evaluator/UI telemetry and are NEVER exposed to AgentPolicy.
"""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

from src.platform import get_utc_now_iso


class LauncherEventType(str, Enum):
    """Lifecycle event types emitted during launcher execution."""
    RUN_STARTED = "RUN_STARTED"
    EPISODE_STARTED = "EPISODE_STARTED"
    EPISODE_PROGRESS = "EPISODE_PROGRESS"
    EPISODE_FINISHED = "EPISODE_FINISHED"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_INTERRUPTED = "RUN_INTERRUPTED"
    RUN_FAILED = "RUN_FAILED"


@dataclass(frozen=True)
class LauncherEventV1:
    """
    Structured observer event emitted by the execution engine.
    Safe for JSON serialization and future GUI consumption.
    """
    event_type: LauncherEventType
    run_id: str
    episode_index: int = 0
    total_episodes: int = 0
    step_index: int = 0
    route_completion: float = 0.0
    speed_kmh: float = 0.0
    status: str = ""
    message: str = ""
    timestamp_utc: str = ""

    @classmethod
    def create(
        cls,
        event_type: LauncherEventType,
        run_id: str,
        episode_index: int = 0,
        total_episodes: int = 0,
        step_index: int = 0,
        route_completion: float = 0.0,
        speed_kmh: float = 0.0,
        status: str = "",
        message: str = ""
    ) -> "LauncherEventV1":
        return cls(
            event_type=event_type,
            run_id=run_id,
            episode_index=int(episode_index),
            total_episodes=int(total_episodes),
            step_index=int(step_index),
            route_completion=float(route_completion),
            speed_kmh=float(speed_kmh),
            status=str(status),
            message=str(message),
            timestamp_utc=get_utc_now_iso()
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d


EventCallback = Callable[[LauncherEventV1], None]
