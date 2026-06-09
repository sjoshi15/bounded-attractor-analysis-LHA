# bounded-attractor-analysis

Reproducibility repository for *Bounded Attractor Dynamics in LLM Agent Personality Trajectories: An Honest Test of the Fractal-Mind Hypothesis* (Joshi, 2026).

## What's here

The full analysis pipeline behind the paper, plus the 17-agent embedding dataset (pre-computed 1536-D vectors + per-day trait tags) needed to reproduce every statistic, figure, and bootstrap result.

## What's not here

The OpenSim simulation framework itself is not released. The paper deliberately withholds the engine, prompts, scenario internals, and the full daily-state text for the reasons described in its Section 5. This repo contains only the post-hoc analysis pipeline, the per-day trait-tag lists, and the pre-computed embedding vectors of the full personality state.

## Quick start

```bash
pip install -r requirements.txt
PYTHONPATH=. python -m analysis.audit_fractal_test --agent kira_nakamura
PYTHONPATH=. python -m analysis.ar1_surrogate_test
PYTHONPATH=. python -m analysis.full_vs_traits_compare
PYTHONPATH=. python figures/make_figures.py
```

Every script has a fixed seed; the reproducibility regression in `tests/test_reproducibility.py` asserts every cached statistic for Kira Nakamura matches `tests/expected_kira.json` within float tolerance.

## Reproducing the paper's headline results

| Paper result | Script | Seed |
|---|---|---|
| 17/17 agents at p < 0.001 (day-shuffle, full state) | `analysis/full_vs_traits_compare.py` | 11 |
| 17/17 agents below AR(1) surrogate | `analysis/ar1_surrogate_test.py` | 42 |
| Kira Nakamura fractal audit | `analysis/audit_fractal_test.py` | 7 |
| 12/17 agents at trait-tag day-shuffle | `analysis/long_agents_analysis.py` | 42 |

## Using the pipeline on your own agents

The four-null bootstrap is generic. Feed `analysis/bootstrap.py` a sequence of daily natural-language state strings from any persistent-state agent system and it will produce the four-null comparison:

```python
from analysis.bootstrap import run_bootstrap
r = run_bootstrap(your_traj_hd, null="day_shuffle", n_trials=150, seed=0)
print(r["real"], r["null_mean"], r["z"], r["p"])
```

See `examples/custom_agent.py` for a worked end-to-end example that also embeds your raw daily-state strings via OpenAI `text-embedding-3-small`.

## Data

`data/agents/` contains 17 agents across two scenarios (neighborhood, 7 agents at 40-64 alive days; parameter golf, 10 agents at 58 days). The embeddings were generated with OpenAI `text-embedding-3-small` on 2026-04-15.

- `data/agents/<slug>/traits/day_NN.txt` — comma-joined trait list for that agent on that day.
- `data/embeddings/full_state.parquet` — 1536-D embedding of the full daily personality state (columns: `agent`, `scenario`, `day`, `alive`, `vector`). The underlying text is not released; only the vectors.
- `data/embeddings/traits.parquet` — 1536-D embedding of the trait sentence (columns same as above).

Re-embedding the released `traits/day_NN.txt` files reproduces `traits.parquet` exactly; `full_state.parquet` cannot be re-derived from this repo because the source text is withheld.

## Citing

If you use this code or data, please cite:

> Joshi, S. (2026). Bounded Attractor Dynamics in LLM Agent Personality Trajectories: An Honest Test of the Fractal-Mind Hypothesis.

## License

- Code: MIT (see `LICENSE`).
- Data: CC-BY-4.0 (see `data/LICENSE`). Use freely with attribution.

## Contact

Issues and pull requests welcome. For replication on other agent frameworks, open an issue and tag with `replication`.
