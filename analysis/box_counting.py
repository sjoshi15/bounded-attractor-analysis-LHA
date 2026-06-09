"""Box-counting dimension estimator with dense-sampling and central-60% fit.

The estimator measures the trajectory curve, not just its vertices.
Synthetic sanity check helpers reproduce the unit-segment, Koch-curve, and
unit-square reference values used in the paper's audit (Section 5.1).
"""

from __future__ import annotations

import numpy as np


def pca_2d(points_nd: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project an (N, D) point cloud onto its top-2 principal components.

    Returns (coords_2d, mean_row, basis) where basis has shape (2, D).
    """
    mean = points_nd.mean(axis=0, keepdims=True)
    centered = points_nd - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:2].T, mean, Vt[:2]


def interpolate_segments(coords: np.ndarray, pts_per_segment: int = 25) -> np.ndarray:
    """Dense-sample the polyline so box-counting measures the curve, not the dots."""
    if coords.shape[0] < 2:
        return coords
    out = [coords[0:1]]
    ts = np.linspace(0, 1, pts_per_segment, endpoint=False)[1:]
    for a, b in zip(coords[:-1], coords[1:]):
        out.append(a + (b - a) * ts[:, None])
        out.append(b[None, :])
    return np.vstack(out)


def box_count_scaling(coords_2d: np.ndarray, n_scales: int = 14) -> dict:
    """Return log ε, log N(ε) and the locally-fit D at each scale.

    The global D is fit on the central 60% of scales (q20 to q80 of log eps).
    """
    if coords_2d.shape[0] < 8:
        return {"log_eps": [], "log_N": [], "local_D": [],
                "global_D": float("nan"), "global_r2": float("nan")}
    xs = coords_2d[:, 0]
    ys = coords_2d[:, 1]
    span_x = xs.max() - xs.min()
    span_y = ys.max() - ys.min()
    span = max(span_x, span_y, 1e-9)
    eps_small = span / (2 ** n_scales)
    eps = np.geomspace(eps_small, span * 1.1, n_scales)
    log_eps = []
    log_N = []
    for e in eps:
        ix = np.floor((xs - xs.min()) / e).astype(int)
        iy = np.floor((ys - ys.min()) / e).astype(int)
        cells = set(zip(ix.tolist(), iy.tolist()))
        N = len(cells)
        if N > 0:
            log_eps.append(np.log(e))
            log_N.append(np.log(N))

    log_eps = np.array(log_eps)
    log_N = np.array(log_N)

    local_D = []
    for i in range(len(log_eps)):
        lo = max(0, i - 1)
        hi = min(len(log_eps), i + 2)
        if hi - lo >= 2:
            slope, _ = np.polyfit(log_eps[lo:hi], log_N[lo:hi], 1)
            local_D.append(-float(slope))
        else:
            local_D.append(float("nan"))

    if len(log_eps) >= 4:
        q_lo, q_hi = np.quantile(log_eps, [0.2, 0.8])
        mask = (log_eps >= q_lo) & (log_eps <= q_hi)
        if mask.sum() < 3:
            mask = np.ones_like(log_eps, dtype=bool)
        slope, inter = np.polyfit(log_eps[mask], log_N[mask], 1)
        yhat = slope * log_eps[mask] + inter
        ss_res = ((log_N[mask] - yhat) ** 2).sum()
        ss_tot = ((log_N[mask] - log_N[mask].mean()) ** 2).sum()
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        gD = -float(slope)
    else:
        gD, r2 = float("nan"), float("nan")

    return {"log_eps": log_eps.tolist(), "log_N": log_N.tolist(),
            "local_D": local_D, "global_D": gD, "global_r2": float(r2)}


def unit_segment_dim(n_points: int = 1500) -> float:
    """Reference: dense-sampled unit segment. Expected D ~ 1.0."""
    t = np.linspace(0, 1, n_points)
    line = np.column_stack([t, np.zeros_like(t)])
    return box_count_scaling(line)["global_D"]


def koch_curve(n_iter: int = 6) -> np.ndarray:
    """Generate the Koch curve by n_iter L-system iterations on [0,1]."""
    pts = np.array([[0.0, 0.0], [1.0, 0.0]])
    R60 = np.array([[np.cos(np.pi / 3), -np.sin(np.pi / 3)],
                    [np.sin(np.pi / 3),  np.cos(np.pi / 3)]])
    for _ in range(n_iter):
        new = []
        for a, b in zip(pts[:-1], pts[1:]):
            v = b - a
            p1 = a + v / 3
            p2 = p1 + R60 @ (v / 3)
            p3 = a + 2 * v / 3
            new.extend([a, p1, p2, p3])
        new.append(pts[-1])
        pts = np.array(new)
    return pts


def koch_curve_dim(n_iter: int = 6) -> float:
    """Reference: Koch curve. Theoretical D = log 4 / log 3 ≈ 1.262."""
    return box_count_scaling(koch_curve(n_iter))["global_D"]


def unit_square_dim(n_points: int = 1500, seed: int = 0) -> float:
    """Reference: uniformly-filled unit square. Expected D close to 2.0
    (finite-sample bias usually pushes it down slightly).
    """
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 1, size=(n_points, 2))
    return box_count_scaling(pts)["global_D"]
