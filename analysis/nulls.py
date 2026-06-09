"""Null-distribution generators for the four-way bootstrap.

All four take a single agent trajectory (N, D) and a `seed`, returning
either a single surrogate trajectory or a batch of `n_trials` surrogates.
No module-level RNG state — every call is reproducible from its seed.
"""

from __future__ import annotations

import numpy as np


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def gaussian_null(agent_traj: np.ndarray, n_trials: int = 150, seed: int = 0) -> np.ndarray:
    """i.i.d. Gaussian in D-dimensional space, matched mean and per-coordinate scale.

    Returns shape (n_trials, N, D).
    """
    rng = _rng(seed)
    mean = agent_traj.mean(axis=0)
    std = float(agent_traj.std(axis=0).mean()) or 1e-3
    n, d = agent_traj.shape
    return mean + rng.normal(0.0, std, size=(n_trials, n, d))


def random_walk_null(agent_traj: np.ndarray, n_trials: int = 150, seed: int = 0) -> np.ndarray:
    """D-dimensional random walk with step size matched to mean day-over-day step.

    Returns shape (n_trials, N, D).
    """
    rng = _rng(seed)
    n, d = agent_traj.shape
    steps = np.diff(agent_traj, axis=0)
    step_scale = float(np.linalg.norm(steps, axis=1).mean()) if steps.size else 1e-3
    out = np.zeros((n_trials, n, d), dtype=agent_traj.dtype)
    for t in range(n_trials):
        walk = np.zeros_like(agent_traj)
        walk[0] = agent_traj[0]
        inc = rng.normal(0.0, 1.0, size=(n - 1, d))
        inc = inc / (np.linalg.norm(inc, axis=1, keepdims=True) + 1e-12) * step_scale
        for i in range(1, n):
            walk[i] = walk[i - 1] + inc[i - 1]
        out[t] = walk
    return out


def day_shuffle_null(agent_traj: np.ndarray, n_trials: int = 150, seed: int = 0) -> np.ndarray:
    """Uniform random permutation of the agent's day labels.

    Returns shape (n_trials, N, D).
    """
    rng = _rng(seed)
    n = agent_traj.shape[0]
    out = np.empty((n_trials,) + agent_traj.shape, dtype=agent_traj.dtype)
    for t in range(n_trials):
        out[t] = agent_traj[rng.permutation(n)]
    return out


def _fit_ar1(traj: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-coordinate AR(1) fit by OLS.

    v_{t+1}[d] = intercept[d] + alpha[d] * v_t[d] + eps[d].
    Returns (alpha, intercept, residual_std, start) — first three shape (D,),
    start shape (D,).
    """
    T, D = traj.shape
    if T < 3:
        return np.zeros(D), traj.mean(axis=0), np.ones(D) * 1e-3, traj[0]

    x = traj[:-1]
    y = traj[1:]
    mx = x.mean(axis=0)
    my = y.mean(axis=0)
    cov_xy = ((x - mx) * (y - my)).mean(axis=0)
    var_x = ((x - mx) ** 2).mean(axis=0)
    alpha = np.where(var_x > 1e-15, cov_xy / var_x, 0.0)
    alpha = np.clip(alpha, -0.999, 0.999)
    intercept = my - alpha * mx
    residuals = y - (intercept + alpha * x)
    res_std = np.maximum(residuals.std(axis=0), 1e-8)
    return alpha, intercept, res_std, traj[0]


def ar1_surrogate(agent_traj: np.ndarray, n_trials: int = 150, seed: int = 0) -> np.ndarray:
    """Per-coordinate AR(1) surrogate. Fit OLS once, generate n_trials trajectories.

    Returns shape (n_trials, N, D).
    """
    rng = _rng(seed)
    T, D = agent_traj.shape
    alpha, intercept, res_std, start = _fit_ar1(agent_traj)
    out = np.zeros((n_trials, T, D), dtype=np.float32)
    for k in range(n_trials):
        out[k, 0] = start
        for t in range(1, T):
            noise = rng.normal(0.0, res_std)
            out[k, t] = intercept + alpha * out[k, t - 1] + noise
    return out


NULL_REGISTRY = {
    "gaussian": gaussian_null,
    "random_walk": random_walk_null,
    "day_shuffle": day_shuffle_null,
    "ar1": ar1_surrogate,
}
