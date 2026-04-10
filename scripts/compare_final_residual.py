#!/usr/bin/env python3
import csv
import math
import sys
from pathlib import Path


def load_final_value(path: Path, column: str) -> float:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"no rows in {path}")
    return float(rows[-1][column])


def main() -> int:
    if len(sys.argv) != 5:
        print("usage: compare_final_residual.py <reference.csv> <test.csv> <column> <rel_tol>")
        return 1

    ref = load_final_value(Path(sys.argv[1]), sys.argv[3])
    test = load_final_value(Path(sys.argv[2]), sys.argv[3])
    rel_tol = float(sys.argv[4])

    scale = max(abs(ref), 1.0e-14)
    rel_err = abs(test - ref) / scale

    print(f"[compare] reference={ref:.12e} test={test:.12e} rel_err={rel_err:.6e}")
    return 0 if rel_err <= rel_tol or math.isclose(test, ref, rel_tol=rel_tol, abs_tol=1.0e-14) else 2


if __name__ == "__main__":
    raise SystemExit(main())
