#!/usr/bin/env python3
import csv
import math
import sys
from pathlib import Path


def load_xy_csv(path: Path, x_col: str, y_cols: list[str]):
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((float(row[x_col]), [float(row[c]) for c in y_cols]))
    rows.sort(key=lambda item: item[0])
    return rows


def interp(rows, x):
    if x <= rows[0][0]:
        return rows[0][1]
    if x >= rows[-1][0]:
        return rows[-1][1]
    for i in range(1, len(rows)):
        if rows[i][0] >= x:
            x0, y0 = rows[i - 1]
            x1, y1 = rows[i]
            t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
            return [a * (1.0 - t) + b * t for a, b in zip(y0, y1)]
    return rows[-1][1]


def compare_series(ref_rows, test_rows):
    err2 = [0.0 for _ in ref_rows[0][1]]
    max_abs = [0.0 for _ in ref_rows[0][1]]
    for x, ref_vals in ref_rows:
        test_vals = interp(test_rows, x)
        for i, (r, t) in enumerate(zip(ref_vals, test_vals)):
            diff = t - r
            err2[i] += diff * diff
            max_abs[i] = max(max_abs[i], abs(diff))
    n = max(len(ref_rows), 1)
    rmse = [math.sqrt(v / n) for v in err2]
    return rmse, max_abs


def main():
    if len(sys.argv) != 3:
        print("usage: compare_flatplate_benchmark.py <reference_prefix> <test_prefix>")
        return 1

    ref_prefix = Path(sys.argv[1])
    test_prefix = Path(sys.argv[2])

    series = [
        ("wall", "x", ["cf", "cp"]),
        ("profile_20", "y", ["u_over_uinf"]),
        ("profile_50", "y", ["u_over_uinf"]),
        ("profile_80", "y", ["u_over_uinf"]),
    ]

    overall_ok = True
    for stem, x_col, cols in series:
        ref_path = ref_prefix.parent / f"{ref_prefix.name}_{stem}.csv"
        test_path = test_prefix.parent / f"{test_prefix.name}_{stem}.csv"
        ref_rows = load_xy_csv(ref_path, x_col, cols)
        test_rows = load_xy_csv(test_path, x_col, cols)
        rmse, max_abs = compare_series(ref_rows, test_rows)
        print(f"[compare] {stem}")
        for col, r, m in zip(cols, rmse, max_abs):
            print(f"  {col}: rmse={r:.6e} max_abs={m:.6e}")
            if (col == "u_over_uinf" and m > 5.0e-2) or (col != "u_over_uinf" and m > 1.0e-2):
                overall_ok = False

    return 0 if overall_ok else 2


if __name__ == "__main__":
    sys.exit(main())
