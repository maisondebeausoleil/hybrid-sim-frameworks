#!/usr/bin/env python3

"""
hmc_pdmp.py

Hybrid Hamiltonian Monte Carlo (HMC) + Piecewise Deterministic Markov Process (PDMP)
toy implementation.

- HMC: efficient proposals via leapfrog integration.
- PDMP: deterministic ODE flow with random jump times and Markovian state changes.

Use this for social dynamics / physics toy models where you want
continuous evolution interrupted by discrete jumps.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Marlon J. Broussard de Beausoleil

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple


# ------------------------------
# Hamiltonian Monte Carlo
# ------------------------------

@dataclass
class HMCSampler:
    """
    Simple HMC sampler in R^d.

    Parameters
    ----------
    log_target : callable
        log p(q) up to constant.
    grad_log_target : callable
        gradient of log_target wrt q.
    step_size : float
        Leapfrog step size (epsilon).
    n_steps : int
        Number of leapfrog steps per proposal (L).
    rng : np.random.Generator
        Random generator.
    """
    log_target: Callable[[np.ndarray], float]
    grad_log_target: Callable[[np.ndarray], np.ndarray]
    step_size: float = 0.1
    n_steps: int = 10
    rng: np.random.Generator = np.random.default_rng()

    def _leapfrog(self, q: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform L leapfrog steps."""
        eps = self.step_size
        q_new = q.copy()
        p_new = p.copy()

        # half step momentum
        p_new += 0.5 * eps * self.grad_log_target(q_new)

        for _ in range(self.n_steps):
            # full step position
            q_new += eps * p_new
            # full step momentum, except last iteration we break early
            if _ != self.n_steps - 1:
                p_new += eps * self.grad_log_target(q_new)

        # final half step momentum
        p_new += 0.5 * eps * self.grad_log_target(q_new)

        # negate momentum to make proposal symmetric
        p_new = -p_new
        return q_new, p_new

    def sample(self, q0: np.ndarray, n_samples: int, burn_in: int = 500) -> np.ndarray:
        d = q0.size
        q = q0.copy()
        samples = []

        def hamiltonian(q_, p_):
            return -self.log_target(q_) + 0.5 * np.dot(p_, p_)

        current_logp = self.log_target(q)

        for i in range(n_samples + burn_in):
            # sample momentum ~ N(0, I)
            p0 = self.rng.normal(size=d)
            q_prop, p_prop = self._leapfrog(q, p0)

            current_H = hamiltonian(q, p0)
            prop_H = hamiltonian(q_prop, p_prop)

            log_alpha = -prop_H + current_H
            if np.log(self.rng.uniform()) < log_alpha:
                q = q_prop
                current_logp = self.log_target(q)

            if i >= burn_in:
                samples.append(q.copy())

        return np.array(samples)


def demo_hmc():
    """
    Demo HMC sampling from a 2D correlated Gaussian.
    """

    cov = np.array([[1.0, 0.8], [0.8, 1.5]])
    inv_cov = np.linalg.inv(cov)

    def log_target(q: np.ndarray) -> float:
        return -0.5 * q.T @ inv_cov @ q

    def grad_log_target(q: np.ndarray) -> np.ndarray:
        return -inv_cov @ q

    hmc = HMCSampler(log_target=log_target, grad_log_target=grad_log_target,
                     step_size=0.1, n_steps=20)
    q0 = np.array([0.0, 0.0])
    samples = hmc.sample(q0, n_samples=5000)

    print("HMC demo: mean ≈", samples.mean(axis=0))
    print("HMC demo: cov ≈")
    print(np.cov(samples.T))


# ------------------------------
# PDMP: deterministic flow + jumps
# ------------------------------

@dataclass
class PDMPProcess:
    """
    Simple PDMP with:
    - deterministic flow dx/dt = b(x)
    - jump intensity lambda(x)
    - jump kernel J(x): new state after jump

    We simulate with thinning for a finite horizon T.
    """

    drift: Callable[[float, np.ndarray], np.ndarray]
    intensity: Callable[[float, np.ndarray], float]
    jump: Callable[[float, np.ndarray], np.ndarray]
    rng: np.random.Generator = np.random.default_rng()

    def simulate(self, x0: np.ndarray, t0: float, t_end: float, dt: float = 0.01):
        """
        Simulate PDMP trajectories from t0 to t_end.

        Returns
        -------
        times : np.ndarray
        states : np.ndarray
        """
        t = t0
        x = x0.copy()
        times = [t]
        states = [x.copy()]

        while t < t_end:
            # deterministic Euler step
            b = self.drift(t, x)
            x = x + dt * b
            t = t + dt

            # jump with probability lambda(x) * dt
            lam = self.intensity(t, x)
            if self.rng.uniform() < lam * dt:
                x = self.jump(t, x)

            times.append(t)
            states.append(x.copy())

        return np.array(times), np.array(states)


def demo_pdmp():
    """
    Demo PDMP: scalar opinion x(t) drifting towards 0 with occasional jumps
    to +/-1 representing sudden adoption / abandonment.
    """

    def drift(t, x):
        # drift towards 0: dx/dt = -k x
        k = 0.5
        return -k * x

    def intensity(t, x):
        # higher intensity when |x| is small (people are undecided)
        base = 0.1
        return base + 0.5 * np.exp(-np.abs(x))

    def jump(t, x):
        # jump to +1 or -1 with equal probability
        return np.array([1.0]) if np.random.rand() < 0.5 else np.array([-1.0])

    pdmp = PDMPProcess(drift=drift, intensity=intensity, jump=jump)
    t, x = pdmp.simulate(x0=np.array([0.0]), t0=0.0, t_end=10.0, dt=0.01)

    print("PDMP demo: final state:", x[-1])


if __name__ == "__main__":
    print("=== HMC demo ===")
    demo_hmc()
    print("\n=== PDMP demo ===")
    demo_pdmp()

