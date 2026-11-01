"""
Stage 5 Soft Actor-Critic (SAC) Inference Agent (Gate S5.2).

Wraps a trained Stable-Baselines3 SAC policy into an authentic Platform V1
AgentPolicy consuming public AgentInputV1 and emitting continuous Box(2) actions.
"""

from __future__ import annotations

# Windows DLL initialization order: import torch before Panda3D
try:
    import torch
except ImportError:
    pass

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.platform import (
    AgentDecision,
    AgentDescriptor,
    AgentInputV1,
    AgentPolicy,
    AgentPublicEpisodeContext,
    InputProfileId,
    get_action_adapter,
)


class Stage5SacAgent(AgentPolicy):
    """
    Inference policy for Stage 5 Soft Actor-Critic (SAC) models.
    """

    agent_id = "stage5_sac"
    agent_version = "0.1.0"
    method_family = "MODEL_FREE_RL"

    def __init__(
        self,
        checkpoint_path: Optional[Union[Path, str]] = None,
        expected_sha256: Optional[str] = None,
    ) -> None:
        self._descriptor = AgentDescriptor(
            agent_id=self.agent_id,
            agent_version=self.agent_version,
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic",
            stateful_within_episode=False,
            method_family=self.method_family,
        )
        from src.training.stage5.observation_encoder import Stage5ObservationEncoder
        self.encoder = Stage5ObservationEncoder(core_only=False)
        self.action_adapter = get_action_adapter("continuous_box2_v1")
        self._model = None
        self._metadata: Optional[Dict[str, Any]] = None
        self._ready = False

        if checkpoint_path is None:
            # Check default candidate checkpoint path if present
            from src.launcher.cases import get_default_project_root
            default_ckpt = get_default_project_root() / "runs" / "checkpoints" / "stage5_sac_smoke"
            if default_ckpt.exists() and (default_ckpt / "model.zip").exists():
                checkpoint_path = default_ckpt

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path, expected_sha256=expected_sha256)

    @property
    def descriptor(self) -> AgentDescriptor:
        return self._descriptor

    def load_checkpoint(self, path: Union[Path, str], expected_sha256: Optional[str] = None) -> None:
        """
        Loads and verifies a trained SAC checkpoint bundle.
        Enforces cryptographic digest verification and observation schema fingerprint matching.
        """
        from src.training.stage5.checkpoint import load_sac_checkpoint
        ckpt_path = Path(path).resolve()
        expected_fp = self.encoder.get_schema_fingerprint()
        model, metadata = load_sac_checkpoint(
            checkpoint_dir=ckpt_path,
            device="cpu",
            expected_sha256=expected_sha256,
            expected_encoder_fingerprint=expected_fp,
            load_replay_buffer=False,
        )
        self._model = model
        self._metadata = metadata

    def reset(self, public_context: AgentPublicEpisodeContext, agent_seed: Optional[int] = None) -> None:
        """Resets the agent before a new episode begins."""
        if self._model is None:
            raise RuntimeError(
                f"Agent '{self.agent_id}' requires a loaded checkpoint before inference. "
                "Call load_checkpoint() before reset()."
            )
        self._ready = True

    def act(self, agent_input: AgentInputV1) -> AgentDecision:
        """
        Produces a deterministic decision for the current 10 Hz control cycle.
        Operates strictly in inference mode without gradient updates.
        """
        if not self._ready or self._model is None:
            raise RuntimeError(f"Agent '{self.agent_id}' must be reset with a loaded model before act().")

        # 1. Encode public AgentInputV1 into 409D observation vector
        obs_vec = self.encoder.encode(agent_input)

        # 2. Predict action deterministically using frozen policy network
        raw_action, _ = self._model.predict(obs_vec, deterministic=True)

        # 3. Validate action through canonical continuous_box2_v1 adapter
        canonical_act = self.action_adapter.to_canonical(raw_action)

        return AgentDecision(
            action_payload=[float(canonical_act.steering), float(canonical_act.throttle_brake)],
            diagnostics={
                "deterministic": True,
                "step_index": int(agent_input.step_index),
                "model_sha256": self._metadata.get("model_sha256", "unknown") if self._metadata else "unknown",
            },
        )

    def close(self) -> None:
        """Closes policy and releases model resources."""
        self._ready = False
        self._model = None
        self._metadata = None
