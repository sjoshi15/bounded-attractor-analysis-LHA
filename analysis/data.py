"""Load the released parquet embeddings + per-agent metadata.

This module is the single place that knows where on disk the data lives.
All reproduction scripts go through `load_embeddings()` so swapping out
the on-disk layout (e.g. moving to a remote URL) only touches one file.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
EMB_DIR = DATA_DIR / "embeddings"
AGENTS_DIR = DATA_DIR / "agents"


def agent_slug(name: str) -> str:
    """Map an agent display name to its on-disk slug."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def list_scenarios(df: pd.DataFrame) -> list[str]:
    """Return the sorted list of scenarios present in the embeddings."""
    return sorted(df["scenario"].unique().tolist())


def load_embeddings(kind: str = "full_state") -> pd.DataFrame:
    """Load the released embeddings parquet.

    kind : "full_state" or "traits".
    Returns a DataFrame with columns [agent, day, scenario, alive, vector].
    `vector` is a numpy array of shape (1536,) per row.
    """
    if kind not in {"full_state", "traits"}:
        raise ValueError(f"kind must be 'full_state' or 'traits', got {kind!r}")
    path = EMB_DIR / f"{kind}.parquet"
    df = pd.read_parquet(path)
    df["vector"] = df["vector"].apply(lambda v: np.asarray(v, dtype=np.float32))
    return df


def agent_trajectory(df: pd.DataFrame, agent: str, alive_only: bool = True) -> np.ndarray | None:
    """Return the day-ordered (N, 1536) trajectory for one agent.

    `agent` may be the display name or the slug. If alive_only, restrict
    to rows with alive==True (matches the paper's protocol).
    """
    sub = df[df["agent"] == agent]
    if sub.empty:
        sub = df[df["agent"].apply(agent_slug) == agent_slug(agent)]
    if sub.empty:
        return None
    if alive_only and "alive" in sub.columns:
        sub = sub[sub["alive"].astype(bool)]
    if len(sub) < 10:
        return None
    sub = sub.sort_values("day")
    return np.stack(sub["vector"].to_numpy())


def long_agents(df: pd.DataFrame, min_alive: int = 40) -> list[tuple[str, int]]:
    """Agents with at least `min_alive` alive days, sorted by count desc."""
    if "alive" in df.columns:
        live = df[df["alive"].astype(bool)]
    else:
        live = df
    counts = live.groupby("agent").size().sort_values(ascending=False)
    return [(name, int(cnt)) for name, cnt in counts.items() if cnt >= min_alive]
