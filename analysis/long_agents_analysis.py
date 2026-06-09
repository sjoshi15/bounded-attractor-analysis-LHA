"""Reproduces Section 5.3 day-shuffle z-score figure across all 17
long-trajectory agents using the trait-tag embeddings.

Default seed: 42.

Outputs:
    data/long_agents_bootstrap.json
    figures/output/long_portrait_<scenario>.png
    figures/output/long_portrait_<scenario>_global.png
    figures/output/long_bootstrap_<scenario>.png
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


def bootstrap_agent(traj_hd: np.ndarray, seed: int, n_trials: int = N_BOOTSTRAP) -> dict:
    if traj_hd.shape[0] < 10:
        return {"skipped": True}
    coords_2d, mean_hd, basis = pca_2d(traj_hd)
    dense = interpolate_segments(coords_2d, pts_per_segment=25)
    real_D = float(box_count_scaling(dense)["global_D"])
    real_D2 = float(correlation_dim_2d(coords_2d)["D2"])

    g_traj = gaussian_null(traj_hd, n_trials=n_trials, seed=seed)
    w_traj = random_walk_null(traj_hd, n_trials=n_trials, seed=seed + 1)
    s_traj = day_shuffle_null(traj_hd, n_trials=n_trials, seed=seed + 2)

    g_D, w_D, s_D = [], [], []
    g_D2, w_D2 = [], []
    for g_hd, w_hd, s_hd in zip(g_traj, w_traj, s_traj):
        g_2d = (g_hd - mean_hd) @ basis.T
        w_2d = (w_hd - mean_hd) @ basis.T
        s_2d = (s_hd - mean_hd) @ basis.T
        g_D.append(box_count_scaling(interpolate_segments(g_2d))["global_D"])
        w_D.append(box_count_scaling(interpolate_segments(w_2d))["global_D"])
        s_D.append(box_count_scaling(interpolate_segments(s_2d))["global_D"])
        g_D2.append(correlation_dim_2d(g_2d)["D2"])
        w_D2.append(correlation_dim_2d(w_2d)["D2"])

    def summary(real_val, arr):
        a = np.array([x for x in arr if np.isfinite(x)])
        if a.size < 5:
            return {"mean": float("nan"), "z": float("nan"), "p": float("nan")}
        mean = float(a.mean())
        sd = float(a.std(ddof=1))
        z = (real_val - mean) / (sd + 1e-12) if np.isfinite(real_val) else float("nan")
        p = 2 * min(float((a >= real_val).mean()), float((a <= real_val).mean())) \
            if np.isfinite(real_val) else float("nan")
        return {"mean": mean, "z": float(z), "p": float(p),
                "ci": [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]}

    return {
        "n_days": int(traj_hd.shape[0]),
        "real_D_box": real_D,
        "real_D2_2d": real_D2 if np.isfinite(real_D2) else float("nan"),
        "D_box_vs_gaussian": summary(real_D, g_D),
        "D_box_vs_random_walk": summary(real_D, w_D),
        "D_box_vs_day_shuffle": summary(real_D, s_D),
        "D2_vs_gaussian": summary(real_D2, g_D2),
        "D2_vs_random_walk": summary(real_D2, w_D2),
        "null_arrays": {
            "D_box_gaussian": g_D,
            "D_box_random_walk": w_D,
            "D_box_day_shuffle": s_D,
        },
    }


def render_portrait(scenario: str, agents: list[tuple[str, int]],
                    scenario_df, label: str) -> None:
    n = len(agents)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.3))
    fig.patch.set_facecolor("#101216")
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    cmap = plt.get_cmap("plasma")
    for ax, (name, alive_cnt) in zip(axes_flat, agents):
        ax.set_facecolor("#14171C")
        traj = agent_trajectory(scenario_df, name, alive_only=True)
        if traj is None or traj.shape[0] < 2:
            ax.set_visible(False)
            continue
        c, _, _ = pca_2d(traj)
        n_pts = c.shape[0]
        colors = cmap(np.linspace(0.15, 0.95, n_pts - 1))
        for i in range(n_pts - 1):
            ax.plot(c[i:i + 2, 0], c[i:i + 2, 1], color=colors[i], lw=1.2, alpha=0.85)
        ax.scatter(c[0, 0], c[0, 1], s=35, c="white", alpha=0.6,
                   edgecolors="#4C9AFF", linewidths=1.0, zorder=3)
        ax.scatter(c[-1, 0], c[-1, 1], s=55, c="#ffcc66", alpha=1.0,
                   edgecolors="white", linewidths=1.0, zorder=4)
        ax.set_title(f"{name}  ({alive_cnt} alive days)", color="white", fontsize=10)
        ax.tick_params(colors="#6b7280", labelsize=7)
        for s in ax.spines.values():
            s.set_color("#3a3f46")
        ax.grid(alpha=0.12, color="#6b7280")
    for ax in axes_flat[len(agents):]:
        ax.set_visible(False)
    fig.suptitle(f"{scenario} — long-trajectory agents ({label})",
                 color="white", fontsize=13)
    out = FIG_OUT / f"long_portrait_{scenario}.png"
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.relative_to(REPO_ROOT)}")


def render_global_pca(scenario: str, agents: list[tuple[str, int]],
                      scenario_df) -> None:
    trajs = {}
    stack = []
    for name, _ in agents:
        t = agent_trajectory(scenario_df, name, alive_only=True)
        if t is None:
            continue
        trajs[name] = t
        stack.append(t)
    if not stack:
        return
    X = np.vstack(stack)
    mean = X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
    basis = Vt[:2]

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#101216")
    ax.set_facecolor("#14171C")
    palette = plt.get_cmap("tab10")
    for i, (name, _) in enumerate(agents):
        t = trajs.get(name)
        if t is None or t.shape[0] < 2:
            continue
        c = (t - mean) @ basis.T
        col = palette(i % 10)
        ax.plot(c[:, 0], c[:, 1], color=col, lw=1.2, alpha=0.75, label=name)
        ax.scatter(c[0, 0], c[0, 1], s=36, facecolor="white",
                   edgecolors=col, linewidths=1.5, zorder=3)
        ax.scatter(c[-1, 0], c[-1, 1], s=70, facecolor=col,
                   edgecolors="white", linewidths=1.2, zorder=4)
    ax.set_title(f"{scenario} — all long-trajectory agents in ONE global PCA",
                 color="white", fontsize=12)
    ax.tick_params(colors="#c0c4cc", labelsize=9)
    for s in ax.spines.values():
        s.set_color("#3a3f46")
    ax.grid(alpha=0.12, color="#6b7280")
    leg = ax.legend(facecolor="#14171C", edgecolor="#3a3f46", fontsize=9,
                    loc="center left", bbox_to_anchor=(1.01, 0.5))
    for t in leg.get_texts():
        t.set_color("white")
    out = FIG_OUT / f"long_portrait_{scenario}_global.png"
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.relative_to(REPO_ROOT)}")


def render_bootstrap_summary(scenario: str, results: dict, rng: np.random.Generator) -> None:
    names = [n for n in results if not results[n].get("skipped")]
    if not names:
        return
    real_D = [results[n]["real_D_box"] for n in names]
    z_shuf = [results[n]["D_box_vs_day_shuffle"]["z"] for n in names]
    z_gauss = [results[n]["D_box_vs_gaussian"]["z"] for n in names]
    z_walk = [results[n]["D_box_vs_random_walk"]["z"] for n in names]
    order = np.argsort(-np.array(z_shuf))
    names = [names[i] for i in order]
    real_D = [real_D[i] for i in order]
    z_shuf = [z_shuf[i] for i in order]
    z_gauss = [z_gauss[i] for i in order]
    z_walk = [z_walk[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.35 * len(names) + 2)))
    fig.patch.set_facecolor("#101216")
    ax = axes[0]
    ax.set_facecolor("#14171C")
    y = np.arange(len(names))
    ax.barh(y - 0.25, z_shuf, height=0.24, color="#C678DD", label="vs day-shuffle")
    ax.barh(y, z_gauss, height=0.24, color="#E06C75", label="vs gaussian")
    ax.barh(y + 0.25, z_walk, height=0.24, color="#E5C07B", label="vs random walk")
    ax.set_yticks(y)
    ax.set_yticklabels(names, color="white", fontsize=9)
    ax.axvline(0, color="#6b7280", lw=1)
    ax.axvline(2, color="#4C9AFF", lw=0.8, ls="--", alpha=0.7)
    ax.axvline(-2, color="#4C9AFF", lw=0.8, ls="--", alpha=0.7)
    ax.set_xlabel("z-score (real − null mean) / null SD", color="#c0c4cc")
    ax.set_title(f"{scenario} — D_box: real trajectory vs nulls",
                 color="white", fontsize=11)
    ax.tick_params(colors="#c0c4cc", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#3a3f46")
    ax.grid(alpha=0.12, color="#6b7280", axis="x")
    leg = ax.legend(facecolor="#14171C", edgecolor="#3a3f46", fontsize=8, loc="lower right")
    for t in leg.get_texts():
        t.set_color("white")

    ax = axes[1]
    ax.set_facecolor("#14171C")
    for i, n in enumerate(names):
        shuf = np.array(results[n]["null_arrays"]["D_box_day_shuffle"])
        shuf = shuf[np.isfinite(shuf)]
        if shuf.size == 0:
            continue
        ax.scatter(shuf, np.full_like(shuf, i) + rng.normal(0, 0.05, size=shuf.shape),
                   s=6, c="#C678DD", alpha=0.35, edgecolors="none")
        ax.scatter(results[n]["real_D_box"], i, marker="D", s=70, c="#4C9AFF",
                   edgecolors="white", linewidths=1.0, zorder=4)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, color="white", fontsize=9)
    ax.set_xlabel("D_box  (diamonds = real, purple = day-shuffle null)",
                  color="#c0c4cc")
    ax.set_title(f"{scenario} — real vs day-shuffle null distributions",
                 color="white", fontsize=11)
    ax.tick_params(colors="#c0c4cc", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#3a3f46")
    ax.grid(alpha=0.12, color="#6b7280", axis="x")

    fig.suptitle(f"{scenario} — bootstrap signal of temporal structure",
                 color="white", fontsize=13)
    out = FIG_OUT / f"long_bootstrap_{scenario}.png"
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.relative_to(REPO_ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-trials", type=int, default=N_BOOTSTRAP)
    args = ap.parse_args()

    FIG_OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    df = load_embeddings("traits")

    all_results: dict[str, dict] = {}
    for scenario in list_scenarios(df):
        print(f"\n=== {scenario} ===")
        scenario_df = df[df["scenario"] == scenario]
        agents = long_agents(scenario_df, MIN_ALIVE_DAYS)
        if not agents:
            continue
        for n, cnt in agents:
            print(f"    {n:<22}  {cnt} alive days")
        render_portrait(scenario, agents, scenario_df, label="trait-tag PCA, color=time")
        render_global_pca(scenario, agents, scenario_df)

        print(f"  bootstrapping {args.n_trials} trials per agent...")
        results: dict[str, dict] = {}
        for name, _ in agents:
            traj = agent_trajectory(scenario_df, name, alive_only=True)
            if traj is None:
                continue
            results[name] = bootstrap_agent(traj, seed=args.seed, n_trials=args.n_trials)
            r = results[name]
            if r.get("skipped"):
                continue
            print(f"    {name:<22}  n={r['n_days']:>2}  "
                  f"D_box={r['real_D_box']:.2f}  "
                  f"z(shuf)={r['D_box_vs_day_shuffle']['z']:+.2f}  "
                  f"p(shuf)={r['D_box_vs_day_shuffle']['p']:.3f}")
        render_bootstrap_summary(scenario, results, rng)
        all_results[scenario] = {n: {k: v for k, v in r.items() if k != "null_arrays"}
                                 for n, r in results.items()}

    out = DATA_DIR / "long_agents_bootstrap.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\n-> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
