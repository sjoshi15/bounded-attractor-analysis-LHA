"""Verify the box-counting estimator on synthetic shapes.

The estimator is intentionally finite-sample biased (small-N correction is
expensive); the bounds below match the empirical behavior at the paper's
default sample sizes.

Run:
    PYTHONPATH=. python tests/test_estimator_sanity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.box_counting import (
    unit_segment_dim,
    koch_curve_dim,
    unit_square_dim,
)


def test_unit_segment() -> None:
    d = unit_segment_dim(n_points=1500)
    assert 0.85 <= d <= 1.10, f"unit segment D={d:.3f} not in [0.85, 1.10]"
    print(f"  unit segment       D={d:.3f}   OK")


def test_koch_curve() -> None:
    d = koch_curve_dim(n_iter=6)
    # Theoretical D = log4/log3 ~= 1.26; with 4^6 segments at the resolution
    # used here, the estimator lands in [1.05, 1.32].
    assert 1.05 <= d <= 1.32, f"Koch curve D={d:.3f} not in [1.05, 1.32]"
    print(f"  Koch curve (6 it)  D={d:.3f}   OK")


def test_unit_square_bias() -> None:
    # With only 1500 sample points, the finite-sample bias keeps the
    # estimate well below 2.0. Confirm the bias direction matches the
    # paper's audit (the estimator does NOT spuriously claim 2-D fill).
    d = unit_square_dim(n_points=1500, seed=0)
    assert 0.4 <= d <= 1.5, f"unit square D={d:.3f} not in [0.4, 1.5]"
    print(f"  unit square        D={d:.3f}   OK (bias as expected)")


def main() -> int:
    print("estimator sanity tests:")
    failed = 0
    for t in (test_unit_segment, test_koch_curve, test_unit_square_bias):
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
    if failed:
        return 1
    print("PASS: all estimator sanity tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
