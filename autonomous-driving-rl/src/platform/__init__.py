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
from src.platform.reward import (
    RewardBreakdown,
    RewardSpecV1,
)
from src.platform.metrics import (
    AggregateMetrics,
    EpisodeRecord,
    compute_aggregate_metrics,
)

__all__ = [
    "AggregateMetrics",
    "EpisodeOutcome",
    "EpisodeRecord",
    "EpisodeSpecV1",
    "RewardBreakdown",
    "RewardSpecV1",
    "SAFETY_CRITICAL_FLAGS",
    "SAFETY_FIRST_PRECEDENCE",
    "TerminalReason",
    "classify_episode_outcome",
    "compute_aggregate_metrics",
    "compute_route_aware_horizon",
]
