"""Worked example: run the four-null bootstrap on your own agent's
daily natural-language state strings.

Replace the synthetic `daily_states` list below with the actual daily
state strings from your persistent-state agent system, then run:

    PYTHONPATH=. OPENAI_API_KEY=sk-... python examples/custom_agent.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from analysis.bootstrap import run_bootstrap
from analysis.embed import embed_strings


def embed_daily_states(daily_states: list[str]) -> np.ndarray:
    """Embed a list of daily state strings -> (N, 1536) trajectory."""
    cache = Path(".embed_cache")
    vecs_by_str = embed_strings(daily_states, cache)
    return np.stack([vecs_by_str[s] for s in daily_states])


def main() -> None:
    daily_states = [
        "Day 1: I am cautious and quiet. I spent today reading and avoided the marketplace.",
        "Day 2: I am cautious and quiet. I read more and overheard an argument outside.",
        "Day 3: I am cautious but curious. I asked the neighbor about the argument.",
    ]
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("set OPENAI_API_KEY to run this example")
    if len(daily_states) < 10:
        print("NOTE: the bootstrap needs >= ~30 days for stable statistics.")
        print("      The 3-day example here only demonstrates the API shape.")

    traj = embed_daily_states(daily_states)
    print(f"trajectory shape: {traj.shape}")

    for null in ("day_shuffle", "gaussian", "random_walk", "ar1"):
        r = run_bootstrap(traj, null=null, n_trials=50, seed=0)
        print(f"  {null:<12}  real_D={r['real']:.3f}  null_mean={r['null_mean']:.3f}  "
              f"z={r['z']:+.2f}  p={r['p']:.3f}")


if __name__ == "__main__":
    main()
