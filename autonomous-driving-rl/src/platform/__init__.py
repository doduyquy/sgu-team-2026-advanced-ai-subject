"""
Platform V1 Specification and Lifecycle Modules.
Gate 3 of Research Platform V1.
"""

from src.platform.episode import (
    EpisodeOutcome,
    SAFETY_CRITICAL_FLAGS,
    SAFETY_FIRST_PRECEDENCE,
    TerminalReason,
    classify_episode_outcome,
)
from src.platform.specs import (
    EpisodeSpecV1,
    compute_route_aware_horizon,
)

__all__ = [
    "EpisodeOutcome",
    "EpisodeSpecV1",
    "SAFETY_CRITICAL_FLAGS",
    "SAFETY_FIRST_PRECEDENCE",
    "TerminalReason",
    "classify_episode_outcome",
    "compute_route_aware_horizon",
]
