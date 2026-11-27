"""Policy-side terminal route preview for the Stage 5 SAC development candidate.

This adapter uses only the public, pre-transform AgentInputV1 encoding. It does
not alter F09, simulator navigation, route completion, rewards, or termination.
The generated tail points are explicitly marked as synthetic in every decision
diagnostic and must not be treated as authoritative route geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from src.platform import AgentDecision, AgentDescriptor, AgentInputV1, AgentPolicy, AgentPublicEpisodeContext
from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER


STAGE5_TERMINAL_PREVIEW_V1 = {
    "id": "stage5_terminal_preview_v1",
    "observation_dim": 409,
    "waypoint_slice": [325, 385],
    "waypoints": 20,
    "features_per_waypoint": ["relative_x_m", "relative_y_m", "relative_heading_rad"],
    "validity_mask_slice": [385, 405],
    "route_end_flag_index": 408,
    "lookahead_spacing_m": 2.5,
    "trigger": "route_end_flag_true_and_at_least_one_contiguous_invalid_tail_slot",
    "extension": "last_real_waypoint_plus_spacing_along_its_relative_heading; preserve_heading",
    "route_end_flag_after_adaptation": "unchanged",
    "other_features": "identity",
    "input_source": "public_AgentInputV1_encoded_409d_before_checkpoint_transform",
    "evaluator_private_fields": "never_read",
}


def terminal_preview_fingerprint() -> str:
    payload = json.dumps(STAGE5_TERMINAL_PREVIEW_V1, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TerminalPreviewResult:
    observation: np.ndarray
    activated: bool
    route_end_within_lookahead: bool
    real_waypoint_count: int
    synthetic_waypoint_count: int
    last_real_waypoint_index: int


class Stage5TerminalPreviewAdapterV1:
    """Extend only masked tail slots when the public route-end bit is set."""

    adapter_id = STAGE5_TERMINAL_PREVIEW_V1["id"]
    fingerprint = terminal_preview_fingerprint()

    def apply(self, observation_409d: np.ndarray) -> TerminalPreviewResult:
        source = np.asarray(observation_409d, dtype=np.float32)
        if source.shape != (409,):
            raise ValueError(f"Expected one raw 409D observation, got {source.shape}")
        if not np.all(np.isfinite(source)):
            raise ValueError("Observation contains non-finite values")

        result = np.array(source, dtype=np.float32, copy=True)
        waypoints = result[325:385].reshape(20, 3)
        mask = result[385:405]
        if np.any((np.abs(mask) > 1e-6) & (np.abs(mask - 1.0) > 1e-6)):
            raise ValueError("Task waypoint mask must contain only 0/1 values")
        if abs(float(result[408])) > 1e-6 and abs(float(result[408]) - 1.0) > 1e-6:
            raise ValueError("Public route-end flag must contain only 0/1 values")
        valid = mask > 0.5
        route_end = bool(result[408] > 0.5)
        real_count = int(valid.sum())
        last_real = int(np.flatnonzero(valid)[-1]) if real_count else -1

        if not route_end or last_real < 0:
            return TerminalPreviewResult(result, False, route_end, real_count, 0, last_real)
        # F09 masks must be a valid prefix followed by a zero-padded suffix.
        # Reject holes (e.g. 111011) rather than treating the later valid point
        # as the extrapolation anchor and silently bridging missing route data.
        if real_count and not np.array_equal(valid, np.arange(20) <= last_real):
            raise ValueError("Task validity mask is not a contiguous valid prefix; refusing to invent route points")

        synthetic_count = 20 - last_real - 1
        if synthetic_count <= 0:
            return TerminalPreviewResult(result, False, route_end, real_count, 0, last_real)

        # F09 emits a valid prefix followed by zero-padded invalid slots. Extend
        # that suffix in the ego frame using only the last real waypoint heading.
        for index in range(last_real + 1, 20):
            previous = waypoints[index - 1].copy()
            heading = float(previous[2])
            waypoints[index, 0] = previous[0] + 2.5 * np.cos(heading)
            waypoints[index, 1] = previous[1] + 2.5 * np.sin(heading)
            waypoints[index, 2] = heading
            mask[index] = 1.0

        return TerminalPreviewResult(result, True, route_end, real_count, synthetic_count, last_real)


class Stage5TerminalPreviewSacAgent(AgentPolicy):
    """Frozen SAC inference wrapper with the versioned public-input adapter."""

    agent_id = "stage5_sac_terminal_preview"
    agent_version = "1.0.0"

    def __init__(
        self,
        checkpoint_path: Union[Path, str],
        expected_sha256: Optional[str] = None,
        *,
        task_context_producer_id: str = REFERENCE_TASK_PRODUCER,
    ) -> None:
        from src.agents.stage5_sac_agent import Stage5SacAgent

        self._base = Stage5SacAgent(
            checkpoint_path=checkpoint_path,
            expected_sha256=expected_sha256,
            task_context_producer_id=task_context_producer_id,
        )
        self.adapter = Stage5TerminalPreviewAdapterV1()
        self._descriptor = AgentDescriptor(
            agent_id=self.agent_id,
            agent_version=self.agent_version,
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic",
            stateful_within_episode=False,
            method_family="MODEL_FREE_RL_WITH_POLICY_INPUT_ADAPTER",
        )

    @property
    def descriptor(self) -> AgentDescriptor:
        return self._descriptor

    @property
    def checkpoint_provenance(self) -> Optional[dict[str, Any]]:
        return self._base.checkpoint_provenance

    @property
    def task_context_producer_id(self) -> str:
        return self._base.task_context_producer_id

    def reset(self, public_context: AgentPublicEpisodeContext, agent_seed: Optional[int] = None) -> None:
        self._base.reset(public_context, agent_seed)

    def act(self, agent_input: AgentInputV1, *, task_context_producer_id: str = REFERENCE_TASK_PRODUCER) -> AgentDecision:
        if not self._base._ready or self._base._model is None:
            raise RuntimeError("Terminal-preview SAC agent must be reset before act()")
        if task_context_producer_id != self.task_context_producer_id:
            raise ValueError("Policy input TaskContext producer mismatch")

        raw = self._base.encoder.encode(agent_input)
        adapted = self.adapter.apply(raw)
        policy_observation = adapted.observation
        if self._base._observation_transform is not None:
            from src.training.stage5.pilot_observation_transform import apply_observation_transform

            policy_observation = apply_observation_transform(
                policy_observation,
                self._base._observation_transform,
            )
        action, _ = self._base._model.predict(policy_observation, deterministic=True)
        canonical = self._base.action_adapter.to_canonical(action)
        return AgentDecision(
            action_payload=[float(canonical.steering), float(canonical.throttle_brake)],
            diagnostics={
                "deterministic": True,
                "step_index": int(agent_input.step_index),
                "model_sha256": (self._base._metadata or {}).get("model_sha256", "unknown"),
                "policy_input_adapter_id": self.adapter.adapter_id,
                "policy_input_adapter_fingerprint": self.adapter.fingerprint,
                "terminal_preview_activated": adapted.activated,
                "route_end_within_lookahead": adapted.route_end_within_lookahead,
                "real_waypoint_count": adapted.real_waypoint_count,
                "synthetic_waypoint_count": adapted.synthetic_waypoint_count,
                "last_real_waypoint_index": adapted.last_real_waypoint_index,
            },
        )

    def close(self) -> None:
        self._base.close()
