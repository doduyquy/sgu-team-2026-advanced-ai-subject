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
from src.platform.protocol import (
    EvaluationCase,
    GeometryRecord,
    ScenarioSplitV1,
    SeedPlanV1,
    SplitRole,
    assign_geometry_splits,
    build_test_cases,
    build_validation_cases,
    canonical_json_sha256,
    compute_macro_metrics,
    compute_manifest_sha256,
    derive_seed,
    validate_split_integrity,
)

__all__ = [
    "AggregateMetrics",
    "EpisodeOutcome",
    "EpisodeRecord",
    "EpisodeSpecV1",
    "EvaluationCase",
    "GeometryRecord",
    "RewardBreakdown",
    "RewardSpecV1",
    "SAFETY_CRITICAL_FLAGS",
    "SAFETY_FIRST_PRECEDENCE",
    "ScenarioSplitV1",
    "SeedPlanV1",
    "SplitRole",
    "TerminalReason",
    "assign_geometry_splits",
    "build_test_cases",
    "build_validation_cases",
    "canonical_json_sha256",
    "classify_episode_outcome",
    "compute_aggregate_metrics",
    "compute_macro_metrics",
    "compute_manifest_sha256",
    "compute_route_aware_horizon",
    "derive_seed",
    "validate_split_integrity",
]
