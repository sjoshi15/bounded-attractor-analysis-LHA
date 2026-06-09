"""2-D correlation dimension D2 via pairwise distance distribution.

Implements the Grassberger-Procaccia estimator restricted to a 2-D point
cloud (the PCA projection of the high-dimensional trajectory). The slope
is fit on the central 60% of the log-log distance distribution.
"""

from __future__ import annotations

import numpy as np


def correlation_dim_2d(coords_2d: np.ndarray) -> dict:
    """Return D2 (slope of log C(eps) vs log eps) and the fit R^2.

    coords_2d : (N, 2) array. N must be >= 8 to attempt a fit.
    """
    n = coords_2d.shape[0]
    if n < 8:
        return {"D2": float("nan"), "r2": float("nan")}
    diffs = coords_2d[:, None, :] - coords_2d[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = d[d > 0]
    if d.size == 0:
        return {"D2": float("nan"), "r2": float("nan")}
    eps = np.geomspace(d.min() * 1.01, d.max() * 0.99, 40)
    total = n * (n - 1) / 2.0
    C = np.array([(d < e).sum() / total for e in eps])
    mask = (C > 0) & (C < 1.0)
    if mask.sum() < 5:
        return {"D2": float("nan"), "r2": float("nan")}
    log_eps = np.log(eps[mask])
    log_C = np.log(C[mask])
    q_lo, q_hi = np.quantile(log_eps, [0.2, 0.8])
    fit_mask = (log_eps >= q_lo) & (log_eps <= q_hi)
    if fit_mask.sum() < 4:
        fit_mask = np.ones_like(log_eps, dtype=bool)
    slope, inter = np.polyfit(log_eps[fit_mask], log_C[fit_mask], 1)
    yhat = slope * log_eps[fit_mask] + inter
    ss_res = ((log_C[fit_mask] - yhat) ** 2).sum()
    ss_tot = ((log_C[fit_mask] - log_C[fit_mask].mean()) ** 2).sum()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {"D2": float(slope), "r2": float(r2)}
