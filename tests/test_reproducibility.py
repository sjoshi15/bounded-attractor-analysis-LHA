"""Reproducibility regression test.

Runs `analysis.audit_fractal_test` against Kira with seed=7 and compares
every numeric leaf in the resulting JSON to the committed
`tests/expected_kira.json` baseline within float tolerance. If this fails
on a fresh `git clone`, the pipeline has drifted (most likely from a
non-deterministic numpy update or a refactor that changed RNG order) and
the released numbers in the paper no longer match the code.

Tolerance: rtol=1e-6, atol=1e-9 — strict bit-exact-up-to-float-noise.

Run:
    PYTHONPATH=. python tests/test_reproducibility.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RTOL = 1e-6
ATOL = 1e-9


def _close(a, b) -> bool:
    if a is None or b is None:
        return a == b
    if isinstance(a, float) or isinstance(b, float):
        if a != a and b != b:
            return True
        return abs(a - b) <= ATOL + RTOL * abs(b)
    return a == b


def _walk(path: str, a, b, diffs: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a) | set(b)
        for k in sorted(keys):
            _walk(f"{path}.{k}", a.get(k), b.get(k), diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(f"{path}[{i}]", x, y, diffs)
    else:
        if not _close(a, b):
            diffs.append(f"{path}: {a!r} vs {b!r}")


def main() -> int:
    print("reproducibility test:")
    print("  running audit_fractal_test for kira_nakamura with seed=7...")
    res = subprocess.run(
        [sys.executable, "-m", "analysis.audit_fractal_test",
         "--agent", "kira_nakamura", "--seed", "7", "--n-trials", "200",
         "--skip-synthetic"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        return 1

    actual = json.loads((ROOT / "data" / "audit_bootstrap_kira_nakamura.json").read_text())
    expected = json.loads((ROOT / "tests" / "expected_kira.json").read_text())

    diffs: list[str] = []
    _walk("", actual, expected, diffs)
    if diffs:
        print(f"  FAIL: {len(diffs)} numeric differences:")
        for d in diffs[:20]:
            print(f"    {d}")
        if len(diffs) > 20:
            print(f"    ... and {len(diffs) - 20} more")
        return 1
    print("PASS: every numeric value matches tests/expected_kira.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
