"""Stage 0 Random / Naive research baseline."""

from typing import Optional

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

    Development state:
    - Registered for SANDBOX/AUDIT only.
    - Benchmark eligibility remains disabled until reviewed.
    """

    def __init__(self) ->None:
        self._rng: Optional[np.random.RandomState] = None

        # Dự kiến  Agent sẽ nhớ action đã chọn cho đến decision tiếp theo.
        self._current_action: int = 7  # mặc định là thả trôi
        self._step_couter: int = 0

        self._steps_until_resample: int = 10
        self._min_hold_steps: int = 20
        self._max_hold_steps: int = 40

        self._descriptor = AgentDescriptor(
            agent_id="stage0_random_naive",
            agent_version="0.1.0",
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

        self._current_action = 7
        self._steps_until_resample = 20
        self._step_counter = 0

    def _sample_random_action(self) -> int:
        if self._rng is None:
            raise RuntimeError("Agent not initialized. Call reset() first.")

        # 0=trái, 1=thẳng, 2=phải
        steering_index = int(
            self._rng.choice(
                [0, 1, 2],
                p = [0.2, 0.6, 0.2],
            )
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
        """
        Placeholder trong giai đoạn dựng khung.

        Logic nhận diện decision point và random action sẽ được
        triển khai ở bước tiếp theo.
        """
        if self._rng is None:
            raise RuntimeError("Agent must be reset before act().")
        
        self._step_couter += 1
        action_resample = False

        if self._steps_until_resample <= 0:
            self._current_action = self._sample_random_action()

            self._steps_until_resample = int(
                self._rng.randint(
                    self._min_hold_steps,
                    self._max_hold_steps + 1,
                )
            )

            action_resample = True

        self._steps_until_resample -= 1

        return AgentDecision(
            action_payload=self._current_action,
            diagnostics={
                "step_counter" : self._step_couter,
                "selected_action" : self._current_action,
                "action_resample" : action_resample,
                "steps_until_resample" : self._steps_until_resample,
            },
        )
    def close(self) -> None:
        """Clean up resources."""
        self._rng = None
    