"""Seeded random baseline with timed and junction-triggered decisions."""

import math
from typing import Optional, Tuple

import numpy as np

from src.platform.agent import (
    AgentDecision,
    AgentDescriptor,
    AgentPublicEpisodeContext
)

from src.platform.agent_input import AgentInputV1

class Stage0RandomNaiveAgent:
    """
    Stage 0 Random / Naive research baseline.

    Hold controls for 2–4 simulation seconds. A nearby real road fork overrides
    the timer once and samples uniformly from its available turn directions.
    Both triggers restart the entire hold period.

    Development state:
    - Registered for SANDBOX/AUDIT only.
    - Benchmark eligibility remains disabled until reviewed.
    """

    # Physical MetaDrive sign: positive steering turns left.
    _TURN_TO_STEERING = {"left": 2, "straight": 1, "right": 0}

    def __init__(
        self,
        min_hold_s: float = 2.0,
        max_hold_s: float = 4.0,
        junction_trigger_distance_m: float = 8.0,
    ) -> None:
        if not (math.isfinite(min_hold_s) and math.isfinite(max_hold_s)
                and 0.0 < min_hold_s <= max_hold_s):
            raise ValueError("Hold durations must be finite and 0 < min <= max.")
        if not math.isfinite(junction_trigger_distance_m) or junction_trigger_distance_m < 0:
            raise ValueError("Junction trigger distance must be finite and non-negative.")
        self._min_hold_s = min_hold_s
        self._max_hold_s = max_hold_s
        self._junction_trigger_distance_m = junction_trigger_distance_m
        self._rng: Optional[np.random.RandomState] = None

        self._current_action = 7  # Straight + throttle.
        self._step_counter = 0
        self._last_decision_point_id: Optional[str] = None
        self._junction_clear_steps = 0
        self._control_dt_s = 0.1

        self._steps_until_resample: int = 0
        self._min_hold_steps: int = 20
        self._max_hold_steps: int = 40

        self._descriptor = AgentDescriptor(
            agent_id="stage0_random_naive",
            agent_version="0.2.0",
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="discrete9_lowbranch_v1",
            inference_stochasticity="stochastic",
            stateful_within_episode=True,
            method_family="RANDOM",
        )

    @property
    def descriptor(self) -> AgentDescriptor:
        return self._descriptor

    def reset(
            self,
            public_context: AgentPublicEpisodeContext,
            agent_seed: Optional[int] = None,
    ) -> None:
        seed = 42 if agent_seed is None else int(agent_seed)

        self._rng = np.random.RandomState(seed)

        self._control_dt_s = public_context.control_dt_s
        self._min_hold_steps = max(1, math.ceil(self._min_hold_s / self._control_dt_s))
        self._max_hold_steps = max(
            self._min_hold_steps, math.floor(self._max_hold_s / self._control_dt_s)
        )
        self._current_action = 7
        self._steps_until_resample = 0
        self._step_counter = 0
        self._last_decision_point_id = None
        self._junction_clear_steps = 0

    def _sample_random_action(self, available_turns: Tuple[str, ...] = ()) -> int:
        if self._rng is None:
            raise RuntimeError("Agent not initialized. Call reset() first.")

        if available_turns:
            turn = str(self._rng.choice(available_turns))
            steering_index = self._TURN_TO_STEERING[turn]
        else:
            steering_index = int(
                self._rng.choice([0, 1, 2], p=[0.2, 0.6, 0.2])
            )

        # 0=phanh, 1=thả trôi, 2=tăng ga
        throttle_index = int(
            self._rng.choice(
                [0, 1, 2],
                p = [0.1, 0.15, 0.75],
            )
        )

        return throttle_index * 3 + steering_index

    def act(self, agent_input: AgentInputV1) -> AgentDecision:
        """Hold a random action until timeout or entry into a new real junction."""
        if self._rng is None:
            raise RuntimeError("Agent must be reset before act().")

        self._step_counter += 1
        point = agent_input.task_context.decision_point
        at_junction = point is not None and point.distance_m <= self._junction_trigger_distance_m
        if at_junction:
            self._junction_clear_steps = 0
        else:
            self._junction_clear_steps += 1
            # Re-arm after leaving; tolerate a brief localization flicker.
            if self._junction_clear_steps >= 3:
                self._last_decision_point_id = None
        junction_trigger = at_junction and point.point_id != self._last_decision_point_id

        reason = None
        if junction_trigger:
            reason = "junction"
            self._last_decision_point_id = point.point_id
        elif self._steps_until_resample <= 0:
            reason = "initial" if self._step_counter == 1 else "timer"

        if reason is not None:
            self._current_action = self._sample_random_action(
                point.available_turns if junction_trigger else ()
            )

            self._steps_until_resample = int(
                self._rng.randint(
                    self._min_hold_steps,
                    self._max_hold_steps + 1,
                )
            )

        self._steps_until_resample -= 1

        return AgentDecision(
            action_payload=self._current_action,
            diagnostics={
                "step_counter": self._step_counter,
                "selected_action": self._current_action,
                "action_resample": reason is not None,
                "resample_reason": reason,
                "steps_until_resample": self._steps_until_resample,
                "seconds_until_resample": self._steps_until_resample * self._control_dt_s,
                "decision_point_id": point.point_id if point else None,
                "junction_trigger": junction_trigger,
            },
        )

    def close(self) -> None:
        """Clean up resources."""
        self._rng = None
