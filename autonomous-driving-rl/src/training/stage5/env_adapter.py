"""
Gymnasium Training Environment Adapter for Stage 5 Reinforcement Learning (Gate S5.1).

Wraps the pinned MetaDrive 0.4.3 simulator into a standard Gymnasium environment
strictly aligned with Platform V1 contracts and semantics:
- Geometry Universe: Confined 100% to MapSuite V1 TRAIN split (180 geometries).
- Action Contract: Continuous Box(2) [steering, throttle_brake] in [-1.0, 1.0]^2 via continuous_box2_v1.
- Observation Contract: Flat float32 vector encoded from public AgentInputV1 (409D default).
- Reward Contract: Exact RewardSpecV1 step-reward computation with signed progress delta.
- Lifecycle: 10 Hz control frequency, safety-first termination, and leak-free teardown.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from metadrive.envs.metadrive_env import MetaDriveEnv

from src.launcher.cases import (
    get_default_project_root,
    load_locked_geometry_block_sequence,
)
from src.launcher.executor import (
    build_metadrive_case_config,
    compute_step_route_progress_and_reward,
    read_vehicle_speed_kmh,
)
from src.launcher.models import ResolvedCaseV1
from src.platform import (
    AgentInputV1,
    EpisodeOutcome,
    InputProfileId,
    RewardBreakdown,
    RewardSpecV1,
    TerminalReason,
    classify_episode_outcome,
    get_action_adapter,
)
from src.platform.agent_input import build_agent_input
from src.training.stage5.observation_encoder import (
    Stage5ObservationEncoder,
)
from src.training.stage5.train_sampler import (
    SamplerStrategy,
    TrainCaseSampler,
)


class Stage5TrainingEnv(gym.Env):
    """
    Standard Gymnasium RL training environment wrapping MetaDrive under Platform V1 contracts.
    """

    metadata = {"render_modes": ["none"], "render_fps": 10}

    def __init__(
        self,
        sampler: Optional[TrainCaseSampler] = None,
        encoder: Optional[Stage5ObservationEncoder] = None,
        reward_spec: Optional[RewardSpecV1] = None,
        training_root_seed: int = 101,
        sampler_strategy: str | SamplerStrategy = SamplerStrategy.ROUND_ROBIN,
        tier_filter: Optional[str] = None,
        sequence_filter: Optional[str] = None,
        fixed_geometry_seed: Optional[int] = None,
        core_only: bool = False,
        project_root: Optional[Path] = None,
    ) -> None:
        """
        Initializes the Stage 5 Training Environment.

        Args:
            sampler: Optional pre-configured TrainCaseSampler.
            encoder: Optional pre-configured Stage5ObservationEncoder.
            reward_spec: Optional RewardSpecV1 contract (default: standard Platform V1 spec).
            training_root_seed: Master root seed for sampler reproducibility.
            sampler_strategy: Case selection strategy (ROUND_ROBIN, UNIFORM_RANDOM, TIER_BALANCED, FIXED).
            tier_filter: Optional filter ("Easy", "Medium", "Hard", "Extreme").
            sequence_filter: Optional sequence filter.
            fixed_geometry_seed: Geometry seed if using FIXED strategy.
            core_only: If True, uses 259D core observation instead of full 409D.
            project_root: Subproject root directory.
        """
        super().__init__()
        self.project_root = project_root or get_default_project_root()

        # 1. Sampler (strictly TRAIN split)
        self.sampler = sampler or TrainCaseSampler(
            training_root_seed=training_root_seed,
            strategy=sampler_strategy,
            tier_filter=tier_filter,
            sequence_filter=sequence_filter,
            fixed_geometry_seed=fixed_geometry_seed,
            project_root=self.project_root,
        )

        # 2. Observation Encoder
        self.encoder = encoder or Stage5ObservationEncoder(core_only=core_only)
        self.observation_space = self.encoder.observation_space

        # 3. Action Space: Box(2) continuous in [-1.0, 1.0]
        self.action_adapter = get_action_adapter("continuous_box2_v1")
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # 4. Reward Contract
        self.reward_spec = reward_spec or RewardSpecV1()

        # 5. Internal Simulator State
        self._env: Optional[MetaDriveEnv] = None
        self._current_case: Optional[ResolvedCaseV1] = None
        self._current_sim_key: Optional[Tuple[str, str, int, int]] = None
        self._blocks_cache: Dict[Tuple[str, int], List[Any]] = {}

        # Episode state variables
        self._step_idx = 0
        self._last_route_completion = 0.0
        self._max_route_completion = 0.0
        self._cumulative_reward = 0.0
        self._terminated = False
        self._truncated = False
        self._env_flags: Dict[str, Any] = {}
        self._last_outcome: Optional[EpisodeOutcome] = None

    # ==========================================================================
    # Gymnasium API: reset()
    # ==========================================================================

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets environment to a fresh training episode.

        Args:
            seed: Optional explicit simulator seed override. If provided, overrides derived env seed.
            options: Optional dictionary (e.g. {"case": ResolvedCaseV1}).

        Returns:
            Tuple of (encoded_observation, info_dictionary).
        """
        super().reset(seed=seed)

        # 1. Reseed sampler if explicit seed is provided
        if seed is not None:
            self.sampler.reset_seed(seed)

        # 2. Determine target case with strict holdout verification
        if options and "case" in options:
            raw_case = options["case"]
            if not isinstance(raw_case, ResolvedCaseV1):
                raise TypeError(
                    f"options['case'] must be a ResolvedCaseV1 instance, got {type(raw_case).__name__}"
                )
            self.sampler.validate_train_case(raw_case)
            case = raw_case
        else:
            case = self.sampler.sample_case()

        self._current_case = case
        geom_key = (case.tier, case.sequence, case.geometry_generation_seed)

        # 2. Acquire locked block sequence (using cache to avoid repeated file extraction)
        cache_key = (case.sequence, case.geometry_generation_seed)
        if cache_key not in self._blocks_cache:
            # If a simulator is already open, close it first to avoid singleton engine collision
            if self._env is not None:
                self._env.close()
                self._env = None
                self._current_sim_key = None

            blocks, _, _ = load_locked_geometry_block_sequence(case.to_dict(), self.project_root)
            self._blocks_cache[cache_key] = blocks
        blocks = self._blocks_cache[cache_key]

        # 3. Simulator instance management and initial step observation in exception-safe block
        sim_key = (case.tier, case.sequence, case.geometry_generation_seed, case.environment_seed)
        try:
            # Recreate simulator if simulator is None or if geometry/environment seed changed
            if self._env is None or self._current_sim_key != sim_key:
                if self._env is not None:
                    self._env.close()
                    self._env = None

                env_config = build_metadrive_case_config(case, blocks, render_mode="OFF")
                self._env = MetaDriveEnv(env_config)
                self._current_sim_key = sim_key

            raw_obs, info = self._env.reset(seed=case.environment_seed)

            # 4. Reset episodic tracking variables
            self._step_idx = 0
            self._last_route_completion = 0.0
            self._max_route_completion = 0.0
            self._cumulative_reward = 0.0
            self._terminated = False
            self._truncated = False
            self._env_flags = {}
            self._last_outcome = None

            # 5. Build public AgentInputV1 and encode observation
            agent_input = self._build_agent_input(raw_obs, step_idx=0)
            obs = self.encoder.encode(agent_input)

            # 6. Diagnostic info (strictly for training instrumentation, not passed to policy)
            info_dict: Dict[str, Any] = {
                "step_index": 0,
                "case_id": case.case_id,
                "tier": case.tier,
                "sequence": case.sequence,
                "geometry_seed": case.geometry_generation_seed,
                "environment_seed": case.environment_seed,
                "horizon_steps": case.horizon_steps,
                "traffic_density": case.traffic_density,
            }

            return obs, info_dict
        except Exception:
            self.close()
            raise

    # ==========================================================================
    # Gymnasium API: step()
    # ==========================================================================

    def step(
        self,
        action: Union[np.ndarray, Sequence[float]],
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes one 10 Hz physical control cycle.

        Args:
            action: Box(2) continuous actuator command [steering, throttle_brake] in [-1.0, 1.0]^2.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        if self._env is None or self._current_case is None:
            raise RuntimeError("Environment is not initialized. Call reset() before step().")

        if self._terminated or self._truncated:
            raise RuntimeError("Episode has already terminated. Call reset() before calling step() again.")

        # 1. Action validation and mapping via Platform V1 continuous_box2_v1 adapter
        # Raises InvalidActionError on NaN, Inf, wrong shape, or out-of-bounds (no silent clipping!)
        canonical_act = self.action_adapter.to_canonical(action)

        # 2. Simulator physics step, outcome, reward, and observation encoding in exception-safe block
        try:
            raw_obs, r_env, term_step, trunc_step, step_info = self._env.step(canonical_act.to_numpy())
            self._step_idx += 1

            # 3. Update safety and task flags from step_info
            for k in (
                "arrive_dest", "out_of_road", "crash_vehicle", "crash_object",
                "crash_building", "crash_human", "crash_sidewalk", "max_step"
            ):
                if bool(step_info.get(k, False)):
                    self._env_flags[k] = True

            # 4. Classify outcome adhering strictly to safety-first precedence
            outcome = classify_episode_outcome(
                raw_flags=self._env_flags,
                terminated=term_step,
                truncated=trunc_step or (self._step_idx >= self._current_case.horizon_steps),
            )
            self._last_outcome = outcome
            self._terminated = outcome.terminated
            self._truncated = outcome.truncated

            # 5. Compute authoritative step reward via Platform V1 RewardSpecV1
            curr_route = float(step_info.get("route_completion", 0.0))
            if curr_route > self._max_route_completion:
                self._max_route_completion = curr_route

            delta_route, self._last_route_completion, breakdown = compute_step_route_progress_and_reward(
                reward_spec=self.reward_spec,
                current_route_completion=curr_route,
                last_route_completion=self._last_route_completion,
                horizon_steps=self._current_case.horizon_steps,
                outcome=outcome if (self._terminated or self._truncated) else None,
            )
            step_reward = float(breakdown.total_reward)
            self._cumulative_reward += step_reward

            # 6. Build authoritative AgentInputV1 and encode observation
            agent_input = self._build_agent_input(raw_obs, step_idx=self._step_idx)
            obs = self.encoder.encode(agent_input)

            # 7. Construct info dictionary
            speed_kmh = read_vehicle_speed_kmh(self._env.agent)
            info_dict: Dict[str, Any] = {
                "step_index": self._step_idx,
                "speed_km_h": speed_kmh,
                "route_completion": curr_route,
                "delta_route": delta_route,
                "reward_breakdown": breakdown.to_dict(),
            }

            if self._terminated or self._truncated:
                info_dict["outcome"] = {
                    "terminated": outcome.terminated,
                    "truncated": outcome.truncated,
                    "primary_reason": outcome.primary_reason.value,
                    "clean_success": bool(outcome.clean_success),
                    "raw_flags": dict(outcome.raw_flags),
                }
                info_dict["clean_success"] = bool(outcome.clean_success)
                info_dict["primary_reason"] = outcome.primary_reason.value
                info_dict["cumulative_reward"] = self._cumulative_reward
                info_dict["total_steps"] = self._step_idx

            return obs, step_reward, self._terminated, self._truncated, info_dict
        except Exception:
            self.close()
            raise

    # ==========================================================================
    # Private Simulator Helpers
    # ==========================================================================

    def _build_agent_input(self, raw_obs: np.ndarray, step_idx: int) -> AgentInputV1:
        """Constructs an authoritative AgentInputV1 from current simulator state."""
        tm = getattr(self._env.engine, "traffic_manager", None)
        traffic_v = list(getattr(tm, "traffic_vehicles", [])) if tm and hasattr(tm, "traffic_vehicles") else []

        profile_id = InputProfileId.CORE_ONLY_V1 if self.encoder.is_core_only else InputProfileId.STATE_DECISION_V1

        return build_agent_input(
            raw_obs=raw_obs,
            ego_vehicle=self._env.agent,
            traffic_vehicles=traffic_v,
            navigation=self._env.agent.navigation,
            road_network=self._env.current_map.road_network,
            step_index=step_idx,
            profile_id=profile_id,
        )

    # ==========================================================================
    # Gymnasium API: close() and render()
    # ==========================================================================

    def close(self) -> None:
        """Closes the underlying MetaDrive simulator safely and releases resources."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
        self._current_sim_key = None

    def render(self) -> None:
        """Render is disabled in training environment (strictly headless)."""
        pass

    # ==========================================================================
    # Properties
    # ==========================================================================

    @property
    def current_case(self) -> Optional[ResolvedCaseV1]:
        """Returns the currently active training case."""
        return self._current_case

    @property
    def current_step(self) -> int:
        """Returns the current step index within the episode."""
        return self._step_idx

    @property
    def cumulative_reward(self) -> float:
        """Returns total accumulated reward for the current episode."""
        return self._cumulative_reward

    @property
    def last_outcome(self) -> Optional[EpisodeOutcome]:
        """Returns the terminal outcome if episode has completed, else None."""
        return self._last_outcome
