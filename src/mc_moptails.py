#!/usr/bin/env python3

"""
mc_moptails.py

Monte Carlo + Mixture-of-Polynomial Tails (MoP-Tails) hybrid estimator.
Provides:
- MoPTailsDistribution: polynomial body + Pareto tail
- importance_sampling_expectation: MC with importance weights
- demo_tail_probability: estimate P(X > x0) under heavy tails

Intended for physics/finance tail-risk experiments.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Marlon J. Broussard de Beausoleil

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass
class MoPTailsDistribution:
    """
    Mixture-of-polynomial-with-tail approximation.

    Core: polynomial density (here, approximated via truncated normal)
    Tail: Pareto with shape alpha, threshold tau.

    Parameters
    ----------
    tau : float
        Threshold separating core and tail (x > tau => tail).
    alpha : float
        Pareto tail index (> 0). Smaller => heavier tail.
    core_mean : float
        Mean of the core proposal distribution.
    core_std : float
        Std of the core proposal distribution.
    tail_weight : float
        Mixture weight for tail component (0 < tail_weight < 1).
    rng : np.random.Generator
        Random number generator.
    """
    tau: float = 2.0
    alpha: float = 1.5
    core_mean: float = 0.0
    core_std: float = 1.0
    tail_weight: float = 0.1
    rng: np.random.Generator = np.random.default_rng()

    def sample_proposal(self, n: int) -> np.ndarray:
        """
        Sample from the proposal q(x), a mixture:
        - with prob 1 - tail_weight: N(core_mean, core_std^2)
        - with prob tail_weight: Pareto(alpha, tau)
        """
        mix_choice = self.rng.uniform(size=n) < self.tail_weight
        x = np.empty(n)
        # core samples
        idx_core = ~mix_choice
        x[idx_core] = self.rng.normal(loc=self.core_mean,
                                      scale=self.core_std,
                                      size=idx_core.sum())
        # Pareto tail samples: X = tau * (1 - U)^(-1/alpha)
        idx_tail = mix_choice
        u = self.rng.uniform(size=idx_tail.sum())
        x[idx_tail] = self.tau * (1 - u) ** (-1.0 / self.alpha)
        return x

    def core_pdf(self, x: np.ndarray) -> np.ndarray:
        """Gaussian core density q_core(x)."""
        coef = 1.0 / (self.core_std * np.sqrt(2.0 * np.pi))
        z = (x - self.core_mean) / self.core_std
        return coef * np.exp(-0.5 * z ** 2)

    def tail_pdf(self, x: np.ndarray) -> np.ndarray:
        """Pareto tail density t(x) for x >= tau."""
        x = np.asarray(x)
        pdf = np.zeros_like(x, dtype=float)
        mask = x >= self.tau
        pdf[mask] = (self.alpha * self.tau ** self.alpha) / (x[mask] ** (self.alpha + 1.0))
        return pdf

    def target_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Target density p(x) ≈ mixture of core polynomial + tail.
        For simplicity we use the same functional form as q(x),
        but in practice you plug in your calibrated tail model.
        """
        q_core = self.core_pdf(x)
        q_tail = self.tail_pdf(x)
        return (1.0 - self.tail_weight) * q_core + self.tail_weight * q_tail

    def proposal_pdf(self, x: np.ndarray) -> np.ndarray:
        """Proposal density q(x). Here we use same mixture as target."""
        return self.target_pdf(x)


def importance_sampling_expectation(
    dist: MoPTailsDistribution,
    f: Callable[[np.ndarray], np.ndarray],
    n_samples: int
) -> Tuple[float, float]:
    """
    Estimate E_p[f(X)] via importance sampling with proposal q.

    Parameters
    ----------
    dist : MoPTailsDistribution
        Distribution object with sampling and pdf methods.
    f : callable
        Function f(x) whose expectation under p we want.
    n_samples : int
        Number of Monte Carlo samples.

    Returns
    -------
    est : float
        Monte Carlo estimate of E_p[f(X)].
    var_est : float
        Monte Carlo variance estimate of the estimator.
    """
    x = dist.sample_proposal(n_samples)
    p = dist.target_pdf(x)
    q = dist.proposal_pdf(x)
    w = p / (q + 1e-16)  # guard for numerical stability

    fx = f(x)
    wf = w * fx
    est = np.mean(wf)
    var_est = np.var(wf, ddof=1) / n_samples
    return est, var_est


def demo_tail_probability():
    """
    Demo: estimate P(X > x0) under a heavy-tailed mixture.

    You can adapt this function for finance tail risk or physics extremes.
    """
    dist = MoPTailsDistribution(tau=2.0, alpha=1.5, tail_weight=0.2)
    x0 = 5.0

    def indicator(x):
        return (x > x0).astype(float)

    est, var_est = importance_sampling_expectation(dist, indicator, n_samples=50_000)
    print(f"Estimated P(X > {x0}) ≈ {est:.4e}, Var(est) ≈ {var_est:.2e}")


if __name__ == "__main__":
    demo_tail_probability()

