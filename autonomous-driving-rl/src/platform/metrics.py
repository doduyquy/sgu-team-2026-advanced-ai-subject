"""
Platform V1 Evaluation Metrics, Episode Records, and Aggregation Helpers.
Gate 4 of Research Platform V1.

Pure, unit-testable module independent of Panda3D / simulator engine state.
"""

from dataclasses import asdict, dataclass, field
import math
import statistics
from typing import Any, Dict, List, Optional, Union

from src.platform.episode import TerminalReason


@dataclass(frozen=True)
class EpisodeRecord:
    """
    Standardized per-episode evaluation telemetry container.
    Decoupled from reward functions and algorithm identity.
    """
    tier: str
    sequence: str
    scenario_seed: int
    terminated: bool
    truncated: bool
    primary_reason: TerminalReason
    raw_arrival: bool
    clean_success: bool
    final_route_completion: float
    max_route_completion: float
    raw_crash_vehicle: bool
    raw_crash_object: bool
    raw_crash_building: bool
    raw_crash_human: bool
    raw_crash_sidewalk: bool
    raw_out_of_road: bool
    episode_steps: int
    simulation_time_s: float
    mean_speed_kmh: float
    max_speed_kmh: float
    episode_return: float  # Diagnostic training signal, NOT primary benchmark score
    time_to_clean_success_s: Optional[float] = None
    agent_seed: Optional[int] = None

    def __post_init__(self):
        # Normalize primary_reason to TerminalReason enum if passed as string
        if isinstance(self.primary_reason, str):
            try:
                object.__setattr__(self, "primary_reason", TerminalReason(self.primary_reason))
            except ValueError:
                raise ValueError(f"Unknown primary_reason string: {self.primary_reason}")

        if not isinstance(self.primary_reason, TerminalReason):
            raise ValueError(f"primary_reason must be a TerminalReason, got {type(self.primary_reason)}")

        if self.primary_reason == TerminalReason.UNDETERMINED:
            raise ValueError("Completed EpisodeRecord cannot have primary_reason=UNDETERMINED")

        if not math.isfinite(self.final_route_completion):
            raise ValueError(f"final_route_completion must be finite, got {self.final_route_completion}")
        if not math.isfinite(self.max_route_completion):
            raise ValueError(f"max_route_completion must be finite, got {self.max_route_completion}")

        if self.episode_steps < 0:
            raise ValueError(f"episode_steps must be non-negative, got {self.episode_steps}")
        if self.simulation_time_s < 0:
            raise ValueError(f"simulation_time_s must be non-negative, got {self.simulation_time_s}")

        # Semantic Consistency Invariants
        if self.clean_success:
            if not self.raw_arrival or self.primary_reason != TerminalReason.SUCCESS:
                raise ValueError(
                    f"clean_success=True requires raw_arrival=True and primary_reason=SUCCESS (got arrival={self.raw_arrival}, reason={self.primary_reason})"
                )
            if self.time_to_clean_success_s is None or not math.isfinite(self.time_to_clean_success_s) or self.time_to_clean_success_s < 0:
                raise ValueError(
                    f"clean_success=True requires a non-null, finite, non-negative time_to_clean_success_s (got {self.time_to_clean_success_s})"
                )
        else:
            if self.primary_reason == TerminalReason.SUCCESS:
                raise ValueError(
                    "primary_reason=SUCCESS requires clean_success=True and raw_arrival=True in Platform V1 taxonomy"
                )
            if self.time_to_clean_success_s is not None:
                raise ValueError(
                    f"Non-success record must have time_to_clean_success_s=None (got {self.time_to_clean_success_s})"
                )

        if self.primary_reason == TerminalReason.TIMEOUT:
            if not self.truncated:
                raise ValueError("primary_reason=TIMEOUT requires truncated=True")

        if self.primary_reason in (
            TerminalReason.CRASH_HUMAN,
            TerminalReason.CRASH_VEHICLE,
            TerminalReason.CRASH_OBJECT,
            TerminalReason.CRASH_BUILDING,
            TerminalReason.CRASH_SIDEWALK,
            TerminalReason.OUT_OF_ROAD,
        ):
            if not self.terminated:
                raise ValueError(f"Safety primary reason {self.primary_reason} requires terminated=True")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["primary_reason"] = self.primary_reason.value
        return data


@dataclass(frozen=True)
class AggregateMetrics:
    """
    Scientific benchmark summary aggregated across evaluation episodes.
    Preserves full Python float precision; rounding belongs in formatting/display layers.
    """
    total_episodes: int

    # PRIMARY BENCHMARK METRICS
    clean_success_rate: float
    safety_failure_rate: float
    mean_final_route_completion: float
    median_final_route_completion: float
    mean_time_to_clean_success_s: Optional[float]

    # MUTUALLY EXCLUSIVE PRIMARY OUTCOME BREAKDOWN (Sums to 1.0)
    success_rate: float
    timeout_rate: float
    crash_human_rate: float
    crash_vehicle_rate: float
    crash_object_rate: float
    crash_building_rate: float
    crash_sidewalk_rate: float
    out_of_road_rate: float
    unknown_termination_rate: float

    # SECONDARY / DIAGNOSTIC METRICS
    raw_arrival_rate: float
    median_time_to_clean_success_s: Optional[float]
    mean_max_route_completion: float
    raw_crash_vehicle_rate: float
    raw_crash_object_rate: float
    raw_crash_building_rate: float
    raw_crash_human_rate: float
    raw_crash_sidewalk_rate: float
    raw_out_of_road_rate: float
    raw_any_safety_event_rate: float
    mean_speed_kmh: float
    max_speed_kmh: float

    # TRAINING / RUNTIME DIAGNOSTICS (NOT BENCHMARK RANKING SCORES)
    mean_episode_return: float
    mean_episode_steps: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_aggregate_metrics(records: List[EpisodeRecord]) -> AggregateMetrics:
    """
    Aggregates a list of EpisodeRecord instances into standardized evaluation metrics.
    Maintains full float precision throughout. Handles empty success pools with explicit None semantics.
    """
    if not records:
        raise ValueError("Cannot aggregate empty list of episode records.")

    n = len(records)

    # Primary task rates (full float precision)
    clean_successes = sum(1 for r in records if r.clean_success)
    clean_success_rate = clean_successes / n

    raw_arrivals = sum(1 for r in records if r.raw_arrival)
    raw_arrival_rate = raw_arrivals / n

    # Mutually exclusive primary outcome rates driven by TerminalReason enum
    reasons = [r.primary_reason for r in records]
    success_rate = reasons.count(TerminalReason.SUCCESS) / n
    timeout_rate = reasons.count(TerminalReason.TIMEOUT) / n
    crash_human_rate = reasons.count(TerminalReason.CRASH_HUMAN) / n
    crash_vehicle_rate = reasons.count(TerminalReason.CRASH_VEHICLE) / n
    crash_object_rate = reasons.count(TerminalReason.CRASH_OBJECT) / n
    crash_building_rate = reasons.count(TerminalReason.CRASH_BUILDING) / n
    crash_sidewalk_rate = reasons.count(TerminalReason.CRASH_SIDEWALK) / n
    out_of_road_rate = reasons.count(TerminalReason.OUT_OF_ROAD) / n
    unknown_rate = reasons.count(TerminalReason.UNKNOWN_TERMINATION) / n

    # Overall safety failure rate (any safety failure as primary reason)
    safety_failures = sum(
        1 for r in reasons
        if r in (
            TerminalReason.CRASH_HUMAN,
            TerminalReason.CRASH_VEHICLE,
            TerminalReason.CRASH_OBJECT,
            TerminalReason.CRASH_BUILDING,
            TerminalReason.CRASH_SIDEWALK,
            TerminalReason.OUT_OF_ROAD,
        )
    )
    safety_failure_rate = safety_failures / n

    # Raw individual event rates (can overlap / sum > 1.0)
    raw_crash_vehicle_rate = sum(1 for r in records if r.raw_crash_vehicle) / n
    raw_crash_object_rate = sum(1 for r in records if r.raw_crash_object) / n
    raw_crash_building_rate = sum(1 for r in records if r.raw_crash_building) / n
    raw_crash_human_rate = sum(1 for r in records if r.raw_crash_human) / n
    raw_crash_sidewalk_rate = sum(1 for r in records if r.raw_crash_sidewalk) / n
    raw_out_of_road_rate = sum(1 for r in records if r.raw_out_of_road) / n
    raw_any_safety_event_rate = sum(
        1 for r in records
        if (
            r.raw_crash_vehicle
            or r.raw_crash_object
            or r.raw_crash_building
            or r.raw_crash_human
            or r.raw_crash_sidewalk
            or r.raw_out_of_road
        )
    ) / n

    # Route completion statistics
    final_completions = [r.final_route_completion for r in records]
    max_completions = [r.max_route_completion for r in records]
    mean_final_rc = statistics.mean(final_completions)
    median_final_rc = statistics.median(final_completions)
    mean_max_rc = statistics.mean(max_completions)

    # Conditional efficiency (successful episodes only)
    success_times = [
        r.time_to_clean_success_s
        for r in records
        if r.clean_success and r.time_to_clean_success_s is not None
    ]
    if success_times:
        mean_time_to_success = statistics.mean(success_times)
        median_time_to_success = statistics.median(success_times)
    else:
        mean_time_to_success = None
        median_time_to_success = None

    # Motion & Diagnostics
    mean_speeds = [r.mean_speed_kmh for r in records]
    max_speeds = [r.max_speed_kmh for r in records]
    mean_spd = statistics.mean(mean_speeds)
    max_spd = max(max_speeds)

    returns = [r.episode_return for r in records]
    steps_list = [r.episode_steps for r in records]
    mean_ret = statistics.mean(returns)
    mean_steps = statistics.mean(steps_list)

    return AggregateMetrics(
        total_episodes=n,
        clean_success_rate=clean_success_rate,
        safety_failure_rate=safety_failure_rate,
        mean_final_route_completion=mean_final_rc,
        median_final_route_completion=median_final_rc,
        mean_time_to_clean_success_s=mean_time_to_success,
        success_rate=success_rate,
        timeout_rate=timeout_rate,
        crash_human_rate=crash_human_rate,
        crash_vehicle_rate=crash_vehicle_rate,
        crash_object_rate=crash_object_rate,
        crash_building_rate=crash_building_rate,
        crash_sidewalk_rate=crash_sidewalk_rate,
        out_of_road_rate=out_of_road_rate,
        unknown_termination_rate=unknown_rate,
        raw_arrival_rate=raw_arrival_rate,
        median_time_to_clean_success_s=median_time_to_success,
        mean_max_route_completion=mean_max_rc,
        raw_crash_vehicle_rate=raw_crash_vehicle_rate,
        raw_crash_object_rate=raw_crash_object_rate,
        raw_crash_building_rate=raw_crash_building_rate,
        raw_crash_human_rate=raw_crash_human_rate,
        raw_crash_sidewalk_rate=raw_crash_sidewalk_rate,
        raw_out_of_road_rate=raw_out_of_road_rate,
        raw_any_safety_event_rate=raw_any_safety_event_rate,
        mean_speed_kmh=mean_spd,
        max_speed_kmh=max_spd,
        mean_episode_return=mean_ret,
        mean_episode_steps=mean_steps,
    )
