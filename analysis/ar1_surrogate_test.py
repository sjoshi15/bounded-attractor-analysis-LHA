"""Reproduces Section 5.3 of the paper: AR(1) surrogate test on all 17
long-trajectory agents. For each agent, fit a per-coordinate AR(1) and
ask whether the real trajectory's box-counting dimension is below the
surrogate distribution (z < -2 => identity is more constrained than
first-order conditioning predicts).

Default seed: 42. Uses the full-personality-state embeddings.

Outputs:
    data/ar1_surrogate_results.json
    figures/output/ar1_surrogate_comparison.png
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

from analysis.box_counting import box_count_scaling, interpolate_segments, pca_2d
from analysis.nulls import ar1_surrogate, _fit_ar1
from analysis.data import (
    REPO_ROOT,
    DATA_DIR,
    load_embeddings,
    agent_trajectory,
    long_agents,
    list_scenarios,
)

FIG_OUT = REPO_ROOT / "figures" / "output"
MIN_ALIVE_DAYS = 40
DEFAULT_N_TRIALS = 150


def bootstrap_ar1_one(traj_hd: np.ndarray, n_trials: int, seed: int) -> dict:
    coords_2d, mean_hd, basis = pca_2d(traj_hd)
    dense = interpolate_segments(coords_2d, pts_per_segment=25)
    real_D = float(box_count_scaling(dense)["global_D"])

    alpha, _intercept, _res_std, _start = _fit_ar1(traj_hd)
    surrogates = ar1_surrogate(traj_hd, n_trials=n_trials, seed=seed)

    ar1_D = []
    for fake_hd in surrogates:
        fake_2d = (fake_hd - mean_hd) @ basis.T
        ar1_D.append(box_count_scaling(interpolate_segments(fake_2d))["global_D"])

    arr = np.array([x for x in ar1_D if np.isfinite(x)])
    if arr.size < 5:
        return {"real_D": real_D, "ar1_mean": float("nan"), "ar1_std": float("nan"),
                "z": float("nan"), "p": float("nan"),
                "alpha_mean": float(alpha.mean()), "alpha_median": float(np.median(alpha)),
                "n_days": int(traj_hd.shape[0]), "n_trials": int(arr.size)}

    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    z = (real_D - mean) / (sd + 1e-12)
    p = 2 * min(float((arr >= real_D).mean()), float((arr <= real_D).mean()))
    return {"real_D": real_D, "ar1_mean": mean, "ar1_std": sd,
            "z": float(z), "p": float(p),
            "alpha_mean": float(alpha.mean()),
            "alpha_median": float(np.median(alpha)),
            "n_days": int(traj_hd.shape[0]), "n_trials": int(arr.size)}


def render(all_results: dict) -> None:
    names, zs, scenarios = [], [], []
    for scenario, agents in all_results.items():
        for name, r in agents.items():
            names.append(name)
            zs.append(r["z"])
            scenarios.append(scenario)
    if not names:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#101216")
    ax.set_facecolor("#14171C")
    colors = ["#4C9AFF" if s == "neighborhood_74" else "#98C379" for s in scenarios]
    x = np.arange(len(names))
    ax.bar(x, zs, color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axhline(2, color="#E5C07B", ls="--", alpha=0.7)
    ax.axhline(-2, color="#E5C07B", ls="--", alpha=0.7, label="|z|=2 (p≈0.05)")
    ax.axhline(0, color="#6b7280", ls="-", alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", color="white", fontsize=9)
    ax.set_ylabel("z-score (real D_box vs AR(1) surrogates)",
                  color="#c0c4cc", fontsize=11)
    ax.set_title("AR(1) Surrogate Test: does identity survive beyond first-order conditioning?",
                 color="white", fontsize=13)
    ax.tick_params(colors="#c0c4cc", labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#3a3f46")
    ax.grid(axis="y", alpha=0.12, color="#6b7280")
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#4C9AFF", edgecolor="white", label="neighborhood_74"),
        Patch(facecolor="#98C379", edgecolor="white", label="parameter_golf"),
    ]
    leg = ax.legend(handles=handles, facecolor="#14171C", edgecolor="#3a3f46",
                    fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color("white")
    for i, z_val in enumerate(zs):
        if np.isfinite(z_val):
            ax.text(i, z_val + (0.15 if z_val >= 0 else -0.35),
                    f"{z_val:+.1f}", ha="center",
                    va="bottom" if z_val >= 0 else "top",
                    color="white", fontsize=8, fontweight="bold")

    FIG_OUT.mkdir(parents=True, exist_ok=True)
    out = FIG_OUT / "ar1_surrogate_comparison.png"
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out.relative_to(REPO_ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    args = ap.parse_args()

    df = load_embeddings("full_state")
    all_results: dict[str, dict] = {}

    for scenario in list_scenarios(df):
        print(f"\n=== {scenario} ===")
        scenario_df = df[df["scenario"] == scenario]
        agents = long_agents(scenario_df, MIN_ALIVE_DAYS)
        print(f"{'agent':<22} {'days':>4}  {'real_D':>6}  "
              f"{'ar1_mean':>8}  {'z':>7}  {'p':>6}  {'α_med':>6}  verdict")
        print("-" * 90)
        scenario_results = {}
        for name, _ in agents:
            traj = agent_trajectory(scenario_df, name, alive_only=True)
            if traj is None:
                continue
            r = bootstrap_ar1_one(traj, n_trials=args.n_trials, seed=args.seed)
            scenario_results[name] = r
            verdict = ("MORE CONSTRAINED" if r["z"] < -2
                       else "INSIDE AR(1)" if abs(r["z"]) <= 2 else "WANDERS MORE")
            print(f"{name:<22} {r['n_days']:>4}  {r['real_D']:>6.3f}  "
                  f"{r['ar1_mean']:>8.3f}  {r['z']:>+7.2f}  {r['p']:>6.3f}  "
                  f"{r['alpha_median']:>6.3f}  {verdict}")
        all_results[scenario] = scenario_results

    out = DATA_DIR / "ar1_surrogate_results.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\n-> {out.relative_to(REPO_ROOT)}")
    render(all_results)

    zs = [r["z"] for ags in all_results.values() for r in ags.values()
          if np.isfinite(r["z"])]
    more = sum(1 for z in zs if z < -2)
    inside = sum(1 for z in zs if abs(z) <= 2)
    wander = sum(1 for z in zs if z > 2)
    print(f"\nSUMMARY: {len(zs)} agents tested")
    print(f"  MORE CONSTRAINED than AR(1) (z<-2): {more}/{len(zs)}")
    print(f"  INSIDE AR(1) (|z|<=2):              {inside}/{len(zs)}")
    print(f"  WANDERS MORE than AR(1) (z>+2):     {wander}/{len(zs)}")


if __name__ == "__main__":
    main()
