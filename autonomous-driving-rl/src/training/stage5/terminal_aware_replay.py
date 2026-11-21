"""Terminal-aware critic replay sampling for the bounded P2-H ablation.

This module is deliberately independent of SAC and does not change replay
storage, Bellman targets, action collection, or the actor/temperature batch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TerminalAwareBatch:
    indices: np.ndarray
    probabilities: np.ndarray
    normalized_importance_weights: np.ndarray
    terminal_mask: np.ndarray
    effective_sample_size: float
    terminal_candidates: int


class TerminalAwareSampler:
    """Uniform mixture plus capped-TD-priority sampling of real terminals.

    Each draw uses ``(1-mixture)*Uniform(replay) + mixture*Priority(terminals)``.
    Timeout truncations are excluded by the caller's true-terminal mask.
    """

    def __init__(self, mixture: float = 0.20, priority_alpha: float = 0.60,
                 priority_beta_start: float = 0.40, priority_beta_end: float = 1.0,
                 priority_beta_steps: int = 9000, td_error_cap: float = 10.0,
                 epsilon: float = 1e-3, seed: int = 101) -> None:
        if not 0.0 < mixture < 1.0:
            raise ValueError("mixture must be strictly between zero and one")
        if not 0.0 <= priority_alpha <= 1.0 or not 0.0 <= priority_beta_start <= priority_beta_end <= 1.0:
            raise ValueError("invalid alpha/beta schedule")
        if priority_beta_steps < 1 or td_error_cap <= 0 or epsilon <= 0:
            raise ValueError("invalid priority cap, epsilon, or schedule")
        self.mixture = float(mixture)
        self.priority_alpha = float(priority_alpha)
        self.beta_start = float(priority_beta_start)
        self.beta_end = float(priority_beta_end)
        self.beta_steps = int(priority_beta_steps)
        self.td_error_cap = float(td_error_cap)
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(int(seed))
        self.draws = 0
        self.terminal_draws = 0
        self.safety_draws = 0
        self.clean_draws = 0
        self.timeout_draws = 0

    def beta(self, optimizer_update: int) -> float:
        fraction = min(max(int(optimizer_update), 0) / self.beta_steps, 1.0)
        return self.beta_start + fraction * (self.beta_end - self.beta_start)

    def sample(self, *, size: int, priorities: np.ndarray, true_terminal: np.ndarray,
               kinds: np.ndarray, optimizer_update: int) -> TerminalAwareBatch:
        priorities = np.asarray(priorities, dtype=np.float64).reshape(-1)
        true_terminal = np.asarray(true_terminal, dtype=bool).reshape(-1)
        kinds = np.asarray(kinds, dtype=np.int8).reshape(-1)
        n = len(priorities)
        if size < 1 or n < 1 or len(true_terminal) != n or len(kinds) != n:
            raise ValueError("invalid replay arrays or sample size")
        eligible = np.flatnonzero(true_terminal & np.isin(kinds, (1, 2)))
        probs_all = np.full(n, 1.0 / n, dtype=np.float64)
        if len(eligible):
            capped = np.minimum(np.maximum(priorities[eligible], 0.0), self.td_error_cap)
            terminal_prob = np.power(capped + self.epsilon, self.priority_alpha)
            terminal_prob /= terminal_prob.sum()
            probs_all *= 1.0 - self.mixture
            probs_all[eligible] += self.mixture * terminal_prob
        # With no eligible terminal, treatment safely falls back to uniform.
        probs_all /= probs_all.sum()
        indices = self.rng.choice(n, size=int(size), replace=True, p=probs_all)
        sampled_p = probs_all[indices]
        beta = self.beta(optimizer_update)
        weights = np.power(n * sampled_p, -beta)
        # Standard PER max-normalization: avoids unbounded loss scaling.
        max_weight = float(np.power(n * probs_all.min(), -beta))
        weights = (weights / max_weight).astype(np.float32)
        sampled_terminal = true_terminal[indices] & np.isin(kinds[indices], (1, 2))
        self.draws += len(indices)
        self.terminal_draws += int(sampled_terminal.sum())
        self.safety_draws += int(np.sum(sampled_terminal & (kinds[indices] == 1)))
        self.clean_draws += int(np.sum(sampled_terminal & (kinds[indices] == 2)))
        self.timeout_draws += int(np.sum(kinds[indices] == 3))
        ess = float(np.square(weights.sum()) / np.square(weights).sum())
        return TerminalAwareBatch(indices.astype(np.int64), probs_all.astype(np.float64),
                                  weights, sampled_terminal, ess, int(len(eligible)))

    def state_dict(self) -> dict:
        return {"rng_state": self.rng.bit_generator.state, "draws": self.draws,
                "terminal_draws": self.terminal_draws, "safety_draws": self.safety_draws,
                "clean_draws": self.clean_draws, "timeout_draws": self.timeout_draws}
