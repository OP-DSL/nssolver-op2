#!/usr/bin/env python3
import csv
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) not in (4, 5):
        print("usage: check_residual_csv.py <csv> <column> <max_abs> [min_abs]")
        return 1

    path = Path(sys.argv[1])
    column = sys.argv[2]
    max_abs = float(sys.argv[3])
    min_abs = float(sys.argv[4]) if len(sys.argv) == 5 else None

    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"error: no rows in {path}")
        return 1

    value = float(rows[-1][column])
    print(f"[check] {path.name} {column}={value:.12e}")

    ok = value <= max_abs
    if min_abs is not None:
        ok = ok and value >= min_abs

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
