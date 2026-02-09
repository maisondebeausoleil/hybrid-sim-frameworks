#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
mcmc_homer.py

Generic MCMC sampler + HOMER-style cost function coupling.
Use for experiments where:
- X: uncertain state (e.g., demand, renewable output)
- cost_model(X, params): HOMER-like dispatch/cost simulator

This module focuses on the Markov + Monte Carlo logic; plug in your real HOMER model via the cost_model callback.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass
class MCMCSampler:
    """
    Simple Metropolis-Hastings sampler on R^d with Gaussian random walk.

    Parameters
    ----------
    log_target : callable
        log p(x), up to a constant.
    step_size : float
        Std deviation of proposal distribution.
    rng : np.random.Generator
        Random number generator.
    """
    log_target: Callable[[np.ndarray], float]
    step_size: float = 0.5
    rng: np.random.Generator = np.random.default_rng()

    def sample(self, x0: np.ndarray, n_samples: int, burn_in: int = 1000) -> np.ndarray:
        d = x0.size
        x = x0.copy()
        samples = []

        current_logp = self.log_target(x)

        for i in range(n_samples + burn_in):
            proposal = x + self.step_size * self.rng.normal(size=d)
            logp_prop = self.log_target(proposal)

            log_alpha = logp_prop - current_logp
            if np.log(self.rng.uniform()) < log_alpha:
                x = proposal
                current_logp = logp_prop

            if i >= burn_in:
                samples.append(x.copy())

        return np.array(samples)


def homer_style_cost(state: np.ndarray, params: dict) -> float:
    """
    Surrogate HOMER cost model.

    state: vector [demand, wind, solar].
    params: dict with cost coefficients.

    This is a simple quadratic+linear cost; replace with your real HOMER integration.
    """
    demand, wind, solar = state
    # Negative contributions reduce net demand
    net_load = max(demand - wind - solar, 0.0)

    c_grid = params.get("c_grid", 0.2)      # grid cost per kWh
    c_unmet = params.get("c_unmet", 2.0)    # penalty per kWh unmet
    c_renew = params.get("c_renew", 0.05)   # renewable cost per kWh

    # cost components (toy model)
    cost_grid = c_grid * net_load
    cost_unmet = c_unmet * max(net_load - params.get("grid_cap", 10.0), 0.0)
    cost_renew = c_renew * (wind + solar)

    return cost_grid + cost_unmet + cost_renew


def estimate_expected_cost(
    sampler: MCMCSampler,
    x0: np.ndarray,
    cost_model: Callable[[np.ndarray, dict], float],
    cost_params: dict,
    n_samples: int = 10_000
) -> Tuple[float, float]:
    """
    Estimate E[cost(X)] where X ~ p(x) using MCMC samples.

    Returns
    -------
    est : float
        Monte Carlo estimate of expected cost.
    var_est : float
        Variance estimate of that estimator (ignores autocorrelation, for simplicity).
    """
    samples = sampler.sample(x0, n_samples=n_samples)
    costs = np.array([cost_model(s, cost_params) for s in samples])
    est = costs.mean()
    var_est = costs.var(ddof=1) / n_samples
    return est, var_est


def demo_mcmc_homer():
    """
    Demo: uncertain demand/wind/solar with log-normal prior;
    sample with MCMC and evaluate expected cost.
    """

    def log_target(x: np.ndarray) -> float:
        # independent lognormal-like target on positive states
        if np.any(x <= 0):
            return -np.inf
        mu = np.array([3.0, 2.0, 2.0])
        sigma = np.array([0.5, 0.5, 0.5])
        z = (np.log(x) - mu) / sigma
        return -0.5 * np.sum(z ** 2)

    sampler = MCMCSampler(log_target=log_target, step_size=0.3)
    x0 = np.array([20.0, 5.0, 5.0])

    cost_params = dict(c_grid=0.2, c_unmet=2.0, c_renew=0.05, grid_cap=15.0)
    est, var_est = estimate_expected_cost(
        sampler, x0, homer_style_cost, cost_params, n_samples=5000
    )

    print(f"Estimated expected cost ≈ {est:.3f}, Var(est) ≈ {var_est:.3e}")


if __name__ == "__main__":
    demo_mcmc_homer()

