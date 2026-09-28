"""
Platform V1 Episode Lifecycle, Terminal Event Taxonomy, and Outcome Classifier.
Gate 3 of Research Platform V1.

Pure, unit-testable module independent of Panda3D / simulator engine state.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TerminalReason(str, Enum):
    """Normalized terminal outcome categories."""
    SUCCESS = "SUCCESS"
    CRASH_HUMAN = "CRASH_HUMAN"
    CRASH_VEHICLE = "CRASH_VEHICLE"
    CRASH_OBJECT = "CRASH_OBJECT"
    CRASH_BUILDING = "CRASH_BUILDING"
    CRASH_SIDEWALK = "CRASH_SIDEWALK"
    OUT_OF_ROAD = "OUT_OF_ROAD"
    TIMEOUT = "TIMEOUT"
    UNKNOWN_TERMINATION = "UNKNOWN_TERMINATION"
    UNDETERMINED = "UNDETERMINED"


# Mapping from normalized reason to simulator raw flag key
REASON_TO_RAW_FLAG: Dict[TerminalReason, str] = {
    TerminalReason.CRASH_HUMAN: "crash_human",
    TerminalReason.CRASH_VEHICLE: "crash_vehicle",
    TerminalReason.CRASH_OBJECT: "crash_object",
    TerminalReason.CRASH_BUILDING: "crash_building",
    TerminalReason.CRASH_SIDEWALK: "crash_sidewalk",
    TerminalReason.OUT_OF_ROAD: "out_of_road",
    TerminalReason.SUCCESS: "arrive_dest",
}

# Single source of truth for terminal event precedence:
# Safety-critical violations take absolute precedence over task arrival.
# Arrival takes precedence over step truncation/timeout.
SAFETY_FIRST_PRECEDENCE: List[TerminalReason] = [
    TerminalReason.CRASH_HUMAN,
    TerminalReason.CRASH_VEHICLE,
    TerminalReason.CRASH_OBJECT,
    TerminalReason.CRASH_BUILDING,
    TerminalReason.CRASH_SIDEWALK,
    TerminalReason.OUT_OF_ROAD,
    TerminalReason.SUCCESS,
    TerminalReason.TIMEOUT,
]

SAFETY_CRITICAL_FLAGS: List[str] = [
    "crash_human",
    "crash_vehicle",
    "crash_object",
    "crash_building",
    "crash_sidewalk",
    "out_of_road",
]


@dataclass(frozen=True)
class EpisodeOutcome:
    """
    Standardized, normalized container representing the outcome of an episode step or termination.
    Preserves all raw simulator telemetry while classifying a deterministic primary outcome.
    """
    terminated: bool
    truncated: bool
    primary_reason: TerminalReason
    clean_success: bool
    raw_flags: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_done(self) -> bool:
        """True if the episode has finished either by termination or truncation."""
        return self.terminated or self.truncated


def classify_episode_outcome(
    raw_flags: Dict[str, Any],
    terminated: bool = False,
    truncated: bool = False,
) -> EpisodeOutcome:
    """
    Classifies raw environment flags into a deterministic, safety-first primary outcome.

    Precedence Policy (driven directly by SAFETY_FIRST_PRECEDENCE):
    1. Critical safety failures (crash human, vehicle, object, building, sidewalk, out-of-road).
    2. Clean task arrival (arrive_dest).
    3. Truncation / timeout (max_step or truncated).
    4. Unknown termination (terminated=True without recognized flag).
    5. Undetermined (step is ongoing).

    Clean Success Semantics:
    clean_success is True IF AND ONLY IF arrive_dest is True AND no safety violation flags occurred.
    """
    flags = dict(raw_flags) if raw_flags else {}
    primary = None

    # Evaluate precedence driven by SAFETY_FIRST_PRECEDENCE to avoid code drift
    for reason in SAFETY_FIRST_PRECEDENCE:
        if reason == TerminalReason.TIMEOUT:
            if truncated or bool(flags.get("max_step", False)):
                primary = reason
                break
        else:
            raw_key = REASON_TO_RAW_FLAG.get(reason)
            if raw_key and bool(flags.get(raw_key, False)):
                primary = reason
                break

    if primary is None:
        if terminated:
            primary = TerminalReason.UNKNOWN_TERMINATION
        else:
            primary = TerminalReason.UNDETERMINED

    # Determine clean success: arrival must not be accompanied by any safety violation
    has_safety_violation = any(bool(flags.get(f, False)) for f in SAFETY_CRITICAL_FLAGS)
    clean_success = bool(flags.get("arrive_dest", False)) and not has_safety_violation

    # Standardize terminated / truncated flags
    is_terminal = (
        terminated
        or primary in (
            TerminalReason.CRASH_HUMAN,
            TerminalReason.CRASH_VEHICLE,
            TerminalReason.CRASH_OBJECT,
            TerminalReason.CRASH_BUILDING,
            TerminalReason.CRASH_SIDEWALK,
            TerminalReason.OUT_OF_ROAD,
            TerminalReason.SUCCESS,
            TerminalReason.UNKNOWN_TERMINATION,
        )
    )
    is_truncated = truncated or (primary == TerminalReason.TIMEOUT and not is_terminal)

    return EpisodeOutcome(
        terminated=is_terminal,
        truncated=is_truncated,
        primary_reason=primary,
        clean_success=clean_success,
        raw_flags=flags,
    )
