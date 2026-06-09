"""Reproduces Section 5.2 of the paper: synthetic-shape sanity check + a
200-trial null bootstrap for one agent against gaussian, random_walk, and
day_shuffle nulls in 2-D, on both D_box and D2 statistics.

Default agent: kira_nakamura. Default seed: 7.

Outputs:
    figures/output/audit_bootstrap_<agent>.png
    data/audit_bootstrap_<agent>.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.box_counting import (
    box_count_scaling,
    interpolate_segments,
    pca_2d,
    unit_segment_dim,
    koch_curve_dim,
    unit_square_dim,
)
from analysis.correlation_dimension import correlation_dim_2d
from analysis.nulls import gaussian_null, random_walk_null, day_shuffle_null
from analysis.data import (
    REPO_ROOT,
    DATA_DIR,
    load_embeddings,
    agent_trajectory,
    agent_slug,
)

FIG_OUT = REPO_ROOT / "figures" / "output"


def check_synthetic_shapes() -> dict:
    print("=" * 70)
    print("AUDIT A: box-counting on known shapes")
    print("=" * 70)
    d_line = unit_segment_dim()
    d_square = unit_square_dim(seed=0)
    d_koch = koch_curve_dim(n_iter=6)
    print(f"  unit line segment      expected D~1.00   got D={d_line:.3f}")
    print(f"  filled unit square     expected D~2.00   got D={d_square:.3f}")
    print(f"  Koch curve (6 iter)    expected D~1.26   got D={d_koch:.3f}")
    return {"unit_segment": d_line, "unit_square": d_square, "koch_6": d_koch}


def bootstrap_one_agent(agent: str, seed: int, n_trials: int = 200) -> dict:
    print("\n" + "=" * 70)
    print(f"AUDIT B: {n_trials}-trial null distribution for {agent}")
    print("=" * 70)

    df = load_embeddings("full_state")
    traj_hd = agent_trajectory(df, agent, alive_only=True)
    if traj_hd is None:
        raise RuntimeError(f"agent {agent!r} not found or too few alive days")

    coords_2d, mean_hd, basis = pca_2d(traj_hd)
    coords_dense = interpolate_segments(coords_2d, pts_per_segment=25)
    real_D = float(box_count_scaling(coords_dense)["global_D"])
    real_D2 = float(correlation_dim_2d(coords_2d)["D2"])
    print(f"  real: D_box={real_D:.3f}  D2_2d={real_D2:.3f}")

    gauss_traj = gaussian_null(traj_hd, n_trials=n_trials, seed=seed)
    walk_traj = random_walk_null(traj_hd, n_trials=n_trials, seed=seed + 1)
    shuf_traj = day_shuffle_null(traj_hd, n_trials=n_trials, seed=seed + 2)

    def evaluate(surrogates: np.ndarray):
        dbox, d2 = [], []
        for s_hd in surrogates:
            s_2d = (s_hd - mean_hd) @ basis.T
            dbox.append(box_count_scaling(interpolate_segments(s_2d))["global_D"])
            d2.append(correlation_dim_2d(s_2d)["D2"])
        return np.array(dbox), np.array(d2)

    g_d, g_d2 = evaluate(gauss_traj)
    w_d, w_d2 = evaluate(walk_traj)
    s_d, s_d2 = evaluate(shuf_traj)

    summary = {"agent": agent, "seed": seed, "n_trials": n_trials,
               "real_D_box": real_D, "real_D2_2d": real_D2}

    def summ(real_val, arr, label):
        a = np.asarray([x for x in arr if np.isfinite(x)])
        if a.size < 5:
            return None
        mean = float(a.mean())
        lo, hi = np.quantile(a, [0.025, 0.975])
        sd = float(a.std(ddof=1))
        z = (real_val - mean) / (sd + 1e-12) if np.isfinite(real_val) else float("nan")
        p = min(1.0, 2 * min(float((a >= real_val).mean()), float((a <= real_val).mean()))) \
            if np.isfinite(real_val) else float("nan")
        print(f"  {label:<22}  null mean={mean:.3f}  "
              f"95% CI=[{lo:.3f}, {hi:.3f}]  z={z:+.2f}  p={p:.3f}")
        return {"mean": mean, "ci": [float(lo), float(hi)],
                "z": float(z), "p": float(p)}

    summary["D_box_gaussian"] = summ(real_D, g_d, "D_box vs gaussian")
    summary["D_box_random_walk"] = summ(real_D, w_d, "D_box vs random_walk")
    summary["D_box_day_shuffle"] = summ(real_D, s_d, "D_box vs day_shuffle")
    summary["D2_2d_gaussian"] = summ(real_D2, g_d2, "D2_2d vs gaussian")
    summary["D2_2d_random_walk"] = summ(real_D2, w_d2, "D2_2d vs random_walk")
    summary["D2_2d_day_shuffle"] = summ(real_D2, s_d2, "D2_2d vs day_shuffle")

    slug = agent_slug(agent)
    out_json = DATA_DIR / f"audit_bootstrap_{slug}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n  -> {out_json.relative_to(REPO_ROOT)}")

    FIG_OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#101216")
    arrs_box = {"gaussian": g_d, "random_walk": w_d, "day_shuffle": s_d}
    arrs_d2 = {"gaussian": g_d2, "random_walk": w_d2, "day_shuffle": s_d2}
    colors = {"gaussian": "#E06C75", "random_walk": "#E5C07B", "day_shuffle": "#C678DD"}

    for ax, (metric, real_val, arrs) in zip(
        axes, [("D_box", real_D, arrs_box), ("D2_2d", real_D2, arrs_d2)]
    ):
        ax.set_facecolor("#14171C")
        for name, arr in arrs.items():
            a = arr[np.isfinite(arr)]
            ax.hist(a, bins=30, alpha=0.55, color=colors[name], label=name,
                    edgecolor="none")
        ax.axvline(real_val, color="#4C9AFF", lw=2.2, label=f"real = {real_val:.2f}")
        ax.set_title(f"{metric} — {agent} vs {n_trials} null trials",
                     color="white", fontsize=11)
        ax.tick_params(colors="#c0c4cc", labelsize=9)
        for s in ax.spines.values():
            s.set_color("#3a3f46")
        ax.grid(alpha=0.15, color="#6b7280")
        leg = ax.legend(facecolor="#14171C", edgecolor="#3a3f46", fontsize=8)
        for t in leg.get_texts():
            t.set_color("white")

    fig.suptitle(f"Audit — bootstrap null distribution for {agent}",
                 color="white", fontsize=13)
    out_png = FIG_OUT / f"audit_bootstrap_{slug}.png"
    fig.savefig(out_png, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_png.relative_to(REPO_ROOT)}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="kira_nakamura")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--skip-synthetic", action="store_true")
    args = ap.parse_args()

    if not args.skip_synthetic:
        check_synthetic_shapes()
    bootstrap_one_agent(args.agent, seed=args.seed, n_trials=args.n_trials)


if __name__ == "__main__":
    main()
