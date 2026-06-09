"""OpenAI text-embedding-3-small wrapper.

Reads a directory layout

    <input>/<agent_slug>/<subdir>/day_<NN>.txt

and produces a parquet file with columns [agent, day, scenario, alive, vector].
Cached per-unique-string in a local .embed_cache/ directory so repeated runs
don't re-bill the OpenAI API.

The OpenAI key must be in the OPENAI_API_KEY environment variable. There is
no .env parsing — by design. Set it explicitly before running:

    OPENAI_API_KEY=sk-... python analysis/embed.py --kind traits

Within this repo, only `--kind traits` is reproducible end-to-end: the bare
trait lists are shipped under data/agents/<slug>/traits/. The released
data/embeddings/full_state.parquet was generated from the full daily-state
text, which is NOT shipped (see paper Section 5); to re-embed your own
agents' daily state, point this script at your own directory and use any
custom subdir name.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
BATCH_SIZE = 96


def _client():
    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit("openai package not installed. pip install openai") from e
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set in environment")
    return OpenAI(api_key=key)


def _cache_path(cache_dir: Path, text: str) -> Path:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return cache_dir / f"{h}.npy"


def _embed_batch(client, batch: list[str]) -> list[np.ndarray]:
    for attempt in range(5):
        try:
            r = client.embeddings.create(model=EMBED_MODEL, input=batch)
            return [np.asarray(item.embedding, dtype=np.float32) for item in r.data]
        except Exception as e:
            wait = 2 ** attempt
            print(f"  embed retry {attempt + 1}/5 after {wait}s ({e})")
            time.sleep(wait)
    raise RuntimeError("embedding API failed after 5 attempts")


def embed_strings(strings: list[str], cache_dir: Path) -> dict[str, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    uniq = sorted({s for s in strings if s})
    pending = []
    out: dict[str, np.ndarray] = {}
    for s in uniq:
        cp = _cache_path(cache_dir, s)
        if cp.exists():
            out[s] = np.load(cp)
        else:
            pending.append(s)
    if pending:
        client = _client()
        print(f"  embedding {len(pending)} uncached strings...")
        for i in range(0, len(pending), BATCH_SIZE):
            batch = pending[i:i + BATCH_SIZE]
            vecs = _embed_batch(client, batch)
            for s, v in zip(batch, vecs):
                np.save(_cache_path(cache_dir, s), v)
                out[s] = v
            print(f"    batch {i // BATCH_SIZE + 1}/{(len(pending) + BATCH_SIZE - 1) // BATCH_SIZE}")
    return out


def collect_agent_days(input_dir: Path, subdir: str) -> list[dict]:
    """Scan <input_dir>/<slug>/<subdir>/day_NN.txt and emit one row per file.

    The display name is derived from the slug by title-casing underscore-
    separated tokens (e.g. `kira_nakamura` -> `Kira Nakamura`). Scenario
    is left empty when not encoded on disk; downstream callers can join
    against the released parquet to recover it if needed.
    """
    rows = []
    for slug_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        agent_dir = slug_dir / subdir
        if not agent_dir.is_dir():
            continue
        name = " ".join(t.capitalize() for t in slug_dir.name.split("_"))
        for f in sorted(agent_dir.glob("day_*.txt")):
            day = int(f.stem.split("_")[1])
            text = f.read_text().strip()
            rows.append({"agent": name, "scenario": "",
                         "day": day, "alive": True, "text": text})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/agents", type=Path,
                    help="root with <slug>/<subdir>/day_NN.txt")
    ap.add_argument("--kind", choices=["traits", "custom"], required=True,
                    help="'traits' reads <input>/<slug>/traits/day_NN.txt; "
                         "'custom' takes --subdir for your own layout")
    ap.add_argument("--subdir", default=None,
                    help="when --kind custom: subdir name under each slug")
    ap.add_argument("--output", type=Path, default=None,
                    help="output parquet (default: data/embeddings/<kind>.parquet)")
    ap.add_argument("--cache", type=Path, default=Path(".embed_cache"))
    args = ap.parse_args()

    if args.kind == "traits":
        subdir = "traits"
    else:
        if not args.subdir:
            raise SystemExit("--kind custom requires --subdir")
        subdir = args.subdir
    output = args.output or Path("data/embeddings") / f"{args.kind}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = collect_agent_days(args.input, subdir)
    if not rows:
        raise SystemExit(f"no day files found under {args.input}/<slug>/{subdir}/")
    print(f"collected {len(rows)} (agent, day) records")

    if args.kind == "traits":
        rows = [{**r, "text": f"A person who is: {r['text']}." if r["text"] else ""}
                for r in rows]

    cache = embed_strings([r["text"] for r in rows], args.cache)
    vectors = []
    for r in rows:
        v = cache.get(r["text"])
        if v is None:
            v = np.zeros(EMBED_DIM, dtype=np.float32)
        vectors.append(v.tolist())

    df = pd.DataFrame({
        "agent": [r["agent"] for r in rows],
        "scenario": [r["scenario"] for r in rows],
        "day": [r["day"] for r in rows],
        "alive": [r["alive"] for r in rows],
        "vector": vectors,
    })
    df.to_parquet(output, index=False)
    print(f"-> {output}  ({len(df)} rows, {EMBED_DIM}-D)")


if __name__ == "__main__":
    main()
