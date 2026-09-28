"""
Platform V1 Reward Specification, Breakdown, and Step-Reward Computation.
Gate 4 of Research Platform V1.

Pure, unit-testable module independent of Panda3D / simulator engine state.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from src.platform.episode import EpisodeOutcome, TerminalReason


@dataclass(frozen=True)
class RewardBreakdown:
    """
    Detailed breakdown of reward components for training telemetry and debugging.
    Distinguishes raw unclipped reward sum from optionally clamped total reward.
    """
    progress_reward: float
    time_cost: float
    terminal_reward: float
    raw_total_reward: float
    total_reward: float
    clipping_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RewardSpecV1:
    """
    Versioned reward contract defining the scalar RL training signal (Candidate C).
    Decoupled from scientific evaluation benchmark metrics.

    Design:
        raw_total = w_progress * delta_route_completion - (time_penalty_budget / horizon_steps) + r_terminal
        total_reward = clip(raw_total, clip_min, clip_max) if clipping enabled else raw_total
    """
    spec_version: str = "1.0.0"
    status: str = "NON-FROZEN"
    progress_weight: float = 1.0
    time_penalty_budget: float = 0.25
    success_bonus: float = 1.0
    safety_penalty: float = 1.0
    timeout_penalty: float = 0.0
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None

    def __post_init__(self):
        if self.progress_weight < 0:
            raise ValueError(f"progress_weight must be non-negative, got {self.progress_weight}")
        if self.time_penalty_budget < 0:
            raise ValueError(f"time_penalty_budget must be non-negative, got {self.time_penalty_budget}")
        if self.success_bonus < 0:
            raise ValueError(f"success_bonus must be non-negative, got {self.success_bonus}")
        if self.safety_penalty < 0:
            raise ValueError(f"safety_penalty must be non-negative, got {self.safety_penalty}")
        if self.timeout_penalty < 0:
            raise ValueError(f"timeout_penalty must be non-negative, got {self.timeout_penalty}")
        if self.clip_min is not None and self.clip_max is not None and self.clip_min > self.clip_max:
            raise ValueError(
                f"clip_min ({self.clip_min}) cannot be greater than clip_max ({self.clip_max})"
            )

    def compute_step_reward(
        self,
        delta_route_completion: float,
        horizon_steps: int,
        outcome: Optional[EpisodeOutcome] = None,
    ) -> RewardBreakdown:
        """
        Computes the step reward and component breakdown at full float precision.

        Args:
            delta_route_completion: Progress delta (route_completion_t - route_completion_{t-1}).
            horizon_steps: Episode horizon step budget.
            outcome: Optional classified EpisodeOutcome for terminal transitions.

        Returns:
            RewardBreakdown with progress, time cost, terminal, raw total, and clamped total.
        """
        if horizon_steps <= 0:
            raise ValueError(f"horizon_steps must be strictly positive, got {horizon_steps}")

        # 1. Progress component
        progress_rew = self.progress_weight * float(delta_route_completion)

        # 2. Time cost component (subtracted)
        time_cost = self.time_penalty_budget / float(horizon_steps)

        # 3. Terminal outcome component
        terminal_rew = 0.0
        if outcome is not None and outcome.is_done:
            if outcome.primary_reason == TerminalReason.SUCCESS and outcome.clean_success:
                terminal_rew = self.success_bonus
            elif outcome.primary_reason in (
                TerminalReason.CRASH_HUMAN,
                TerminalReason.CRASH_VEHICLE,
                TerminalReason.CRASH_OBJECT,
                TerminalReason.CRASH_BUILDING,
                TerminalReason.CRASH_SIDEWALK,
                TerminalReason.OUT_OF_ROAD,
                TerminalReason.UNKNOWN_TERMINATION,
            ):
                terminal_rew = -self.safety_penalty
            elif outcome.primary_reason == TerminalReason.TIMEOUT:
                terminal_rew = -self.timeout_penalty

        raw_total = progress_rew - time_cost + terminal_rew

        # 4. Optional clipping (default is None/unclipped)
        clamped_total = raw_total
        clipping_applied = False
        if self.clip_min is not None and clamped_total < self.clip_min:
            clamped_total = self.clip_min
            clipping_applied = True
        if self.clip_max is not None and clamped_total > self.clip_max:
            clamped_total = self.clip_max
            clipping_applied = True

        return RewardBreakdown(
            progress_reward=progress_rew,
            time_cost=time_cost,
            terminal_reward=terminal_rew,
            raw_total_reward=raw_total,
            total_reward=clamped_total,
            clipping_applied=clipping_applied,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
