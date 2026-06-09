"""Regenerate Figures 1-5 from the paper.

Single-entry: runs the three reproduction scripts in the right order with
their fixed seeds, which in turn write all figures to figures/output/.
Output filenames:
    figures/output/long_portrait_<scenario>.png            (Fig 1: portraits)
    figures/output/long_portrait_<scenario>_global.png     (Fig 2: global PCA)
    figures/output/long_bootstrap_<scenario>.png           (Fig 3: z-scores)
    figures/output/ar1_surrogate_comparison.png            (Fig 4: AR(1))
    figures/output/trait_vs_full_drift_scatter.png         (Fig 5: drift)
    figures/output/full_vs_traits_compare.png              (companion)
    figures/output/full_portrait_<scenario>.png            (companion)
    figures/output/audit_bootstrap_kira_nakamura.png       (companion, audit)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(args: list[str]) -> None:
    print(f"\n>>> {' '.join(args)}\n")
    subprocess.run([PY, "-m", *args], cwd=REPO_ROOT, check=True)


def main() -> None:
    (REPO_ROOT / "figures" / "output").mkdir(parents=True, exist_ok=True)
    run(["analysis.long_agents_analysis", "--seed", "42"])
    run(["analysis.ar1_surrogate_test", "--seed", "42"])
    run(["analysis.full_vs_traits_compare", "--seed", "11"])
    run(["analysis.audit_fractal_test", "--agent", "kira_nakamura", "--seed", "7"])
    print("\nAll figures written to figures/output/")


if __name__ == "__main__":
    main()
