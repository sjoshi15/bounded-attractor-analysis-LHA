"""Top-level four-null bootstrap entry point.

This is the single function paper consumers will call. Feed it a daily
trajectory (N, D) high-dimensional embedding, a statistic, and a null
name, and it returns the real value, the null distribution summary, a
95% CI, the z-score, and a two-sided permutation p-value.

The statistic is invoked as `statistic(traj_hd)` and must return a scalar
float (e.g. box-counting D, correlation D2). The default statistic is
2-D box-counting after a local PCA projection — that is what the paper
uses.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import nulls as _nulls
from .box_counting import box_count_scaling, interpolate_segments, pca_2d


def default_box_counting_statistic(traj_hd: np.ndarray) -> float:
    """Local PCA -> dense polyline -> box-counting D. The paper's main statistic."""
    if traj_hd.shape[0] < 8:
        return float("nan")
    coords_2d, _, _ = pca_2d(traj_hd)
    dense = interpolate_segments(coords_2d, pts_per_segment=25)
    return float(box_count_scaling(dense)["global_D"])


def _resolve_null(null: str | Callable) -> Callable:
    if callable(null):
        return null
    if null not in _nulls.NULL_REGISTRY:
        raise ValueError(f"unknown null {null!r}; choose from {sorted(_nulls.NULL_REGISTRY)}")
    return _nulls.NULL_REGISTRY[null]


def run_bootstrap(
    agent_traj: np.ndarray,
    statistic: Callable[[np.ndarray], float] = default_box_counting_statistic,
    null: str | Callable = "day_shuffle",
    n_trials: int = 150,
    seed: int = 0,
) -> dict:
    """Bootstrap one agent against one null.

    Parameters
    ----------
    agent_traj : (N, D) high-dimensional trajectory.
    statistic  : callable mapping (N, D) -> float.
    null       : key in NULL_REGISTRY ("gaussian", "random_walk",
                 "day_shuffle", "ar1") or a callable
                 (traj, n_trials, seed) -> (n_trials, N, D).
    n_trials   : number of surrogate trajectories to draw.
    seed       : RNG seed for the null generator.

    Returns a dict with keys: real, null_mean, null_std, ci, z, p,
    n_trials_used, null_values.
    """
    null_fn = _resolve_null(null)
    real = float(statistic(agent_traj))

    surrogates = null_fn(agent_traj, n_trials=n_trials, seed=seed)
    values = np.array([statistic(s) for s in surrogates])
    finite = values[np.isfinite(values)]

    if finite.size < 5:
        return {"real": real, "null_mean": float("nan"),
                "null_std": float("nan"), "ci": [float("nan"), float("nan")],
                "z": float("nan"), "p": float("nan"),
                "n_trials_used": int(finite.size),
                "null_values": values.tolist()}

    mean = float(finite.mean())
    sd = float(finite.std(ddof=1))
    lo, hi = np.quantile(finite, [0.025, 0.975])
    z = (real - mean) / (sd + 1e-12) if np.isfinite(real) else float("nan")
    p = min(1.0, 2 * min(float((finite >= real).mean()), float((finite <= real).mean()))) \
        if np.isfinite(real) else float("nan")
    return {
        "real": real,
        "null_mean": mean,
        "null_std": sd,
        "ci": [float(lo), float(hi)],
        "z": float(z),
        "p": float(p),
        "n_trials_used": int(finite.size),
        "null_values": values.tolist(),
    }
