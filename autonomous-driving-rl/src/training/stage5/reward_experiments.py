"""Versioned, training-only reward variants for the bounded S5.5 study.

RewardSpecV1 remains authoritative and unchanged.  This module computes parallel
experimental signals from the exact signed route-progress delta and terminal
breakdown produced by Stage5TrainingEnv.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping

import gymnasium as gym
import numpy as np


S5_5_REWARD_EXPERIMENT_ID = "stage5_reward_meter_v1"
S5_5_REWARD_VARIANTS = ("V1", "SCALE", "METER")
S5_5_TIME_PENALTY_BUDGET = 0.25
S5_5_PROGRESS_METERS_PER_UNIT = 10.0


@dataclass(frozen=True)
class RewardScale:
    """Frozen scale derived from the exact median route length of the train pool."""

    median_route_length_m: float
    scale_k: float

    @classmethod
    def from_route_lengths(cls, route_lengths_m: list[float]) -> "RewardScale":
        if not route_lengths_m:
            raise ValueError("At least one route length is required")
        values = [float(value) for value in route_lengths_m]
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Route lengths must be positive finite metres")
        median = float(np.median(np.asarray(values, dtype=np.float64)))
        return cls(median_route_length_m=median, scale_k=median / S5_5_PROGRESS_METERS_PER_UNIT)

    def to_dict(self) -> Dict[str, float]:
        return {
            "median_route_length_m": self.median_route_length_m,
            "scale_k": self.scale_k,
            "equivalent_route_length_m": self.scale_k * S5_5_PROGRESS_METERS_PER_UNIT,
        }


def compute_s5_5_reward_variants(
    *,
    delta_route: float,
    horizon_steps: int,
    terminal_reward: float,
    effective_route_length_m: float,
    scale_k: float,
    time_penalty_budget: float = S5_5_TIME_PENALTY_BUDGET,
) -> Dict[str, float]:
    """Compute V1, global-scale, and signed route-metre rewards for one step."""
    vals = (delta_route, terminal_reward, effective_route_length_m, scale_k, time_penalty_budget)
    if any(not math.isfinite(float(value)) for value in vals):
        raise ValueError("Reward inputs must be finite")
    if int(horizon_steps) <= 0:
        raise ValueError("horizon_steps must be positive")
    if float(effective_route_length_m) <= 0 or float(scale_k) <= 0:
        raise ValueError("effective route length and reward scale must be positive")
    if float(time_penalty_budget) < 0:
        raise ValueError("time_penalty_budget must be non-negative")

    delta_p = float(delta_route)
    terminal = float(terminal_reward)
    length_m = float(effective_route_length_m)
    k = float(scale_k)
    time_cost = float(time_penalty_budget) / int(horizon_steps)
    delta_s_m = length_m * delta_p
    reward_v1 = delta_p - time_cost + terminal
    reward_scale = k * reward_v1
    reward_meter = delta_s_m / S5_5_PROGRESS_METERS_PER_UNIT + k * (terminal - time_cost)
    return {
        "delta_route": delta_p,
        "delta_s_m": delta_s_m,
        "effective_route_length_m": length_m,
        "time_cost": time_cost,
        "terminal_reward": terminal,
        "reward_v1": reward_v1,
        "reward_scale": reward_scale,
        "reward_meter": reward_meter,
    }


class Stage5ExperimentalRewardWrapper(gym.Wrapper):
    """Replace only the reward returned to an RL learner; preserve V1 in info.

    The wrapped environment must be Stage5TrainingEnv.  This wrapper never changes
    observation, action, termination, truncation, or any frozen Platform contract.
    """

    def __init__(self, env: gym.Env, variant: str, scale: RewardScale) -> None:
        super().__init__(env)
        variant = str(variant).upper()
        if variant not in S5_5_REWARD_VARIANTS:
            raise ValueError(f"Unsupported S5.5 reward variant: {variant}")
        base = env.unwrapped
        spec = getattr(base, "reward_spec", None)
        if spec is None:
            raise TypeError("S5.5 reward wrapper requires Stage5TrainingEnv with RewardSpecV1")
        if spec.clip_min is not None or spec.clip_max is not None:
            raise ValueError("S5.5 V1 protocol requires unclipped RewardSpecV1")
        if (float(spec.progress_weight) != 1.0
                or float(spec.time_penalty_budget) != S5_5_TIME_PENALTY_BUDGET):
            raise ValueError("S5.5 reward formula requires the registered RewardSpecV1 progress/time values")
        self.variant = variant
        self.scale = scale
        self.effective_route_length_m: float | None = None
        self.initial_route_completion: float | None = None

    @staticmethod
    def _navigation_length_m(base_env: Any) -> float:
        sim = getattr(base_env, "_env", None)
        agent = getattr(sim, "agent", None)
        navigation = getattr(agent, "navigation", None)
        if navigation is None:
            raise RuntimeError("MetaDrive navigation is unavailable; cannot identify route-completion denominator")
        # MetaDrive 0.4.3 NodeNetworkNavigation uses travelled_length / total_length.
        if hasattr(navigation, "total_length"):
            length_m = float(navigation.total_length)
        else:
            trajectory = getattr(navigation, "reference_trajectory", None)
            length_m = float(getattr(trajectory, "length", float("nan")))
        if not math.isfinite(length_m) or length_m <= 0:
            raise RuntimeError(f"Invalid MetaDrive effective route length: {length_m}")
        return length_m

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        base = self.env.unwrapped
        self.effective_route_length_m = self._navigation_length_m(base)
        self.initial_route_completion = float(info["initial_route_completion"])
        info = dict(info)
        info["s5_5_effective_route_length_m"] = self.effective_route_length_m
        info["s5_5_initial_route_completion"] = self.initial_route_completion
        return observation, info

    def step(self, action: Any):
        observation, reward_v1_env, terminated, truncated, info = self.env.step(action)
        if self.effective_route_length_m is None:
            raise RuntimeError("reset() must be called before S5.5 reward step")
        breakdown = info.get("reward_breakdown")
        if not isinstance(breakdown, Mapping):
            raise RuntimeError("Stage5TrainingEnv omitted its RewardSpecV1 breakdown")
        values = compute_s5_5_reward_variants(
            delta_route=float(info["delta_route"]),
            horizon_steps=int(info["horizon_steps"]) if "horizon_steps" in info else int(self.unwrapped._current_case.horizon_steps),
            terminal_reward=float(breakdown["terminal_reward"]),
            effective_route_length_m=self.effective_route_length_m,
            scale_k=self.scale.scale_k,
            time_penalty_budget=float(self.unwrapped.reward_spec.time_penalty_budget),
        )
        if not math.isclose(values["reward_v1"], float(reward_v1_env), rel_tol=0.0, abs_tol=1e-8):
            raise RuntimeError(
                "S5.5 V1 reward parity failed: computed "
                f"{values['reward_v1']}, Stage5TrainingEnv returned {reward_v1_env}"
            )
        reward_used = values[f"reward_{self.variant.lower()}"]
        enriched = dict(info)
        enriched["s5_5_reward_experiment_id"] = S5_5_REWARD_EXPERIMENT_ID
        enriched["s5_5_reward_variant"] = self.variant
        enriched["s5_5_rewards"] = {**values, "reward_used": reward_used}
        return observation, float(reward_used), terminated, truncated, enriched
