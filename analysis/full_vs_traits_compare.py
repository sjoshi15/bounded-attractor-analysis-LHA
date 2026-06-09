"""Reproduces the 12/17 vs 17/17 result: side-by-side bootstrap of
traits-only vector vs full-personality-state vector. Default seed: 11.

Outputs:
    data/full_vs_traits_bootstrap.json
    figures/output/full_vs_traits_compare.png
    figures/output/trait_vs_full_drift_scatter.png
    figures/output/full_portrait_<scenario>.png
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
from analysis.correlation_dimension import correlation_dim_2d
from analysis.nulls import gaussian_null, random_walk_null, day_shuffle_null
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
N_BOOTSTRAP = 150


def bootstrap(traj_hd: np.ndarray, seed: int, n_trials: int = N_BOOTSTRAP) -> dict:
    coords_2d, mean_hd, basis = pca_2d(traj_hd)
    dense = interpolate_segments(coords_2d, pts_per_segment=25)
    real_D = float(box_count_scaling(dense)["global_D"])
    real_D2 = float(correlation_dim_2d(coords_2d)["D2"])

    g_traj = gaussian_null(traj_hd, n_trials=n_trials, seed=seed)
    w_traj = random_walk_null(traj_hd, n_trials=n_trials, seed=seed + 1)
    s_traj = day_shuffle_null(traj_hd, n_trials=n_trials, seed=seed + 2)

    g_D, w_D, s_D = [], [], []
    for g_hd, w_hd, s_hd in zip(g_traj, w_traj, s_traj):
        g_2d = (g_hd - mean_hd) @ basis.T
        w_2d = (w_hd - mean_hd) @ basis.T
        s_2d = (s_hd - mean_hd) @ basis.T
        g_D.append(box_count_scaling(interpolate_segments(g_2d))["global_D"])
        w_D.append(box_count_scaling(interpolate_segments(w_2d))["global_D"])
        s_D.append(box_count_scaling(interpolate_segments(s_2d))["global_D"])

    def summ(real, arr):
        a = np.array([x for x in arr if np.isfinite(x)])
        if a.size < 5:
            return {"mean": float("nan"), "z": float("nan"), "p": float("nan")}
        mean = float(a.mean())
        sd = float(a.std(ddof=1))
        z = (real - mean) / (sd + 1e-12) if np.isfinite(real) else float("nan")
        p = 2 * min(float((a >= real).mean()), float((a <= real).mean())) \
            if np.isfinite(real) else float("nan")
        return {"mean": mean, "z": float(z), "p": float(p)}

    mean_step = float(np.linalg.norm(np.diff(traj_hd, axis=0), axis=1).mean()) \
        if traj_hd.shape[0] > 1 else 0.0
    return {
        "n_days": int(traj_hd.shape[0]),
        "mean_step": mean_step,
        "real_D_box": real_D,
        "real_D2_2d": real_D2 if np.isfinite(real_D2) else float("nan"),
        "D_box_vs_gaussian": summ(real_D, g_D),
        "D_box_vs_random_walk": summ(real_D, w_D),
        "D_box_vs_day_shuffle": summ(real_D, s_D),
    }


def compare_scenario(scenario: str, traits_df, full_df, seed: int, n_trials: int) -> dict:
    print(f"\n=== {scenario} ===")
    scenario_traits = traits_df[traits_df["scenario"] == scenario]
    scenario_full = full_df[full_df["scenario"] == scenario]
    agents = long_agents(scenario_traits, MIN_ALIVE_DAYS)
    if not agents:
        return {}
    print(f"{'agent':<22} {'n':>3}  "
          f"{'trait D':>7}  {'trait step':>10}  {'z_shuf(trait)':>14}  "
          f"{'full D':>7}  {'full step':>9}  {'z_shuf(full)':>13}")
    print("-" * 110)
    results: dict[str, dict] = {}
    for name, _ in agents:
        t = agent_trajectory(scenario_traits, name, alive_only=True)
        f = agent_trajectory(scenario_full, name, alive_only=True)
        if t is None or f is None:
            continue
        r_t = bootstrap(t, seed=seed, n_trials=n_trials)
        r_f = bootstrap(f, seed=seed, n_trials=n_trials)
        results[name] = {"traits": r_t, "full": r_f}
        print(f"{name:<22} {r_t['n_days']:>3}  "
              f"{r_t['real_D_box']:>7.3f}  {r_t['mean_step']:>10.3f}  "
              f"{r_t['D_box_vs_day_shuffle']['z']:>+14.2f}  "
              f"{r_f['real_D_box']:>7.3f}  {r_f['mean_step']:>9.3f}  "
              f"{r_f['D_box_vs_day_shuffle']['z']:>+13.2f}")
    return results


def render_compare(all_results: dict) -> None:
    items = []
    for scenario, agents in all_results.items():
        for name, r in agents.items():
            items.append((scenario, name, r))
    items.sort(key=lambda x: -x[2]["full"]["D_box_vs_day_shuffle"]["z"]
               if np.isfinite(x[2]["full"]["D_box_vs_day_shuffle"]["z"]) else -99)
    names = [f"{n} ({s[:4]})" for s, n, _ in items]
    z_traits = [r["traits"]["D_box_vs_day_shuffle"]["z"] for _, _, r in items]
    z_full = [r["full"]["D_box_vs_day_shuffle"]["z"] for _, _, r in items]

    fig, ax = plt.subplots(figsize=(12, max(5, 0.4 * len(names) + 1.5)))
    fig.patch.set_facecolor("#101216")
    ax.set_facecolor("#14171C")
    y = np.arange(len(names))
    ax.barh(y - 0.2, z_traits, height=0.38, color="#4C9AFF", label="traits-only vector")
    ax.barh(y + 0.2, z_full, height=0.38, color="#98C379", label="full personality vector")
    ax.set_yticks(y)
    ax.set_yticklabels(names, color="white", fontsize=9)
    ax.axvline(0, color="#6b7280", lw=1)
    ax.axvline(2, color="#ffcc66", lw=0.8, ls="--", alpha=0.8, label="|z|=2 (p≈0.05)")
    ax.axvline(-2, color="#ffcc66", lw=0.8, ls="--", alpha=0.8)
    ax.set_xlabel("z-score vs day-shuffle null  (bigger = stronger time signal)",
                  color="#c0c4cc")
    ax.set_title("traits-only vs full-state vector — which reveals more temporal structure?",
                 color="white", fontsize=12)
    ax.tick_params(colors="#c0c4cc", labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#3a3f46")
    ax.grid(alpha=0.12, color="#6b7280", axis="x")
    leg = ax.legend(facecolor="#14171C", edgecolor="#3a3f46", fontsize=9, loc="lower right")
    for t in leg.get_texts():
        t.set_color("white")
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    out = FIG_OUT / "full_vs_traits_compare.png"
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {out.relative_to(REPO_ROOT)}")

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#101216")
    ax.set_facecolor("#14171C")
    for scenario, agents in all_results.items():
        color = "#4C9AFF" if scenario == "neighborhood_74" else "#98C379"
        for name, r in agents.items():
            ax.scatter(r["traits"]["mean_step"], r["full"]["mean_step"],
                       s=90, c=color, edgecolors="white", linewidths=0.8, alpha=0.85)
            ax.annotate(name, (r["traits"]["mean_step"], r["full"]["mean_step"]),
                        color="#c0c4cc", fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color="#6b7280", ls=":", alpha=0.6)
    ax.set_xlabel("mean day-to-day step in trait-only space", color="#c0c4cc")
    ax.set_ylabel("mean day-to-day step in full-personality space", color="#c0c4cc")
    ax.set_title("Day-to-day drift magnitude: trait tags vs full state",
                 color="white", fontsize=12)
    ax.tick_params(colors="#c0c4cc", labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#3a3f46")
    ax.grid(alpha=0.12, color="#6b7280")
    out = FIG_OUT / "trait_vs_full_drift_scatter.png"
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out.relative_to(REPO_ROOT)}")


def render_full_portraits(full_df) -> None:
    for scenario in list_scenarios(full_df):
        scenario_df = full_df[full_df["scenario"] == scenario]
        agents = long_agents(scenario_df, MIN_ALIVE_DAYS)
        if not agents:
            continue
        stack = []
        for name, _ in agents:
            t = agent_trajectory(scenario_df, name, alive_only=True)
            if t is not None:
                stack.append(t)
        if not stack:
            continue
        X = np.vstack(stack)
        mean = X.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
        basis = Vt[:2]

        fig, ax = plt.subplots(figsize=(12, 9))
        fig.patch.set_facecolor("#101216")
        ax.set_facecolor("#14171C")
        palette = plt.get_cmap("tab10")
        for i, (name, _) in enumerate(agents):
            t = agent_trajectory(scenario_df, name, alive_only=True)
            if t is None or t.shape[0] < 2:
                continue
            c = (t - mean) @ basis.T
            col = palette(i % 10)
            ax.plot(c[:, 0], c[:, 1], color=col, lw=1.3, alpha=0.8, label=name)
            ax.scatter(c[0, 0], c[0, 1], s=38, facecolor="white",
                       edgecolors=col, linewidths=1.4, zorder=3)
            ax.scatter(c[-1, 0], c[-1, 1], s=70, facecolor=col,
                       edgecolors="white", linewidths=1.2, zorder=4)
        ax.set_title(f"{scenario} — FULL personality-state trajectories",
                     color="white", fontsize=12)
        ax.tick_params(colors="#c0c4cc", labelsize=9)
        for sp in ax.spines.values():
            sp.set_color("#3a3f46")
        ax.grid(alpha=0.12, color="#6b7280")
        leg = ax.legend(facecolor="#14171C", edgecolor="#3a3f46", fontsize=9,
                        loc="center left", bbox_to_anchor=(1.01, 0.5))
        for t in leg.get_texts():
            t.set_color("white")
        out = FIG_OUT / f"full_portrait_{scenario}.png"
        fig.savefig(out, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        print(f"-> {out.relative_to(REPO_ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n-trials", type=int, default=N_BOOTSTRAP)
    args = ap.parse_args()

    traits_df = load_embeddings("traits")
    full_df = load_embeddings("full_state")

    all_results: dict[str, dict] = {}
    for scenario in list_scenarios(full_df):
        all_results[scenario] = compare_scenario(scenario, traits_df, full_df,
                                                  seed=args.seed,
                                                  n_trials=args.n_trials)
    out = DATA_DIR / "full_vs_traits_bootstrap.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\n-> {out.relative_to(REPO_ROOT)}")

    render_compare(all_results)
    render_full_portraits(full_df)


if __name__ == "__main__":
    main()
