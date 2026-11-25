"""Surface pressure of a unit-chord airfoil run against an experiment's:
Cp from the wall's state (nssolver_op2_benchmark_postprocess_helper wall,
whose node order follows the section around),
the experiment's from a Tecplot-style file of zones "... a = ALPHA" with
x/c, Cp pairs (as the buffet workshop's OAT15A Cp.dat). Prints the RMS
difference on each side and the shock position (the steepest pressure rise on
the upper surface); with matplotlib also a plot.

    python3 cp.py WALL.csv MACH RE T ALPHA EXPERIMENT.dat [PLOT.png]
"""
import csv
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from freestream import freestream  # noqa: E402


def interp(xs, ys, x):
    for i in range(1, len(xs)):
        if xs[i] >= x:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def shock(xs, cps):
    best, at = 0.0, None
    for i in range(1, len(xs)):
        if 0.2 < xs[i] < 0.9 and xs[i] > xs[i - 1]:
            slope = (cps[i] - cps[i - 1]) / (xs[i] - xs[i - 1])
            if slope > best:
                best, at = slope, 0.5 * (xs[i] + xs[i - 1])
    return at


def wall_sides(wall, p_inf, q):
    """(upper, lower), each [(x, Cp)] by x, from a wall file. Its nodes follow
    the section around (the grid's first line): split at the leading edge, a
    blunt trailing edge's base (x of 1) left out; the half higher up is the
    upper surface."""
    rows = [r for r in csv.DictReader(open(wall)) if abs(float(r["z"])) < 1e-9 and float(r["x"]) < 0.999]
    le = min(range(len(rows)), key=lambda i: float(rows[i]["x"]))
    halves = [rows[:le + 1], rows[le:]]
    halves.sort(key=lambda h: -sum(float(r["y"]) for r in h) / len(h))
    return [sorted((float(r["x"]), (float(r["p"]) - p_inf) / q) for r in h) for h in halves]


def zone(path, alpha):
    """The (x/c, value) pairs of a Tecplot-style file's zone "... a = ALPHA"."""
    points, at = [], None
    for line in open(path):
        m = re.search(r'ZONE.*a\s*=\s*([-0-9.]+)', line)
        if m:
            at = float(m.group(1))
            continue
        if at is not None and abs(at - alpha) < 1e-6 and line.strip() and line.strip()[0] in "0123456789-.":
            points.append(tuple(map(float, line.split()[:2])))
    if not points:
        sys.exit(f"no zone a = {alpha} in {path}")
    return points


def experiment_sides(points):
    """(upper, lower) of an experiment's Cp points, which run around the
    section: split where x turns; the side of lower mean pressure is the
    upper surface."""
    turn = max(range(len(points)), key=lambda i: points[i][0])
    a, b = points[:turn + 1], points[turn:]
    if sum(c for _, c in a) / len(a) > sum(c for _, c in b) / len(b):
        a, b = b, a
    return sorted(a), sorted(b)


def main():
    wall, mach, re_, t, alpha, exp_path = sys.argv[1], *map(float, sys.argv[2:6]), sys.argv[6]
    fs = freestream(mach, re_, t, alpha)
    upper, lower = wall_sides(wall, fs["p"], 0.5 * fs["rho"] * fs["speed"] ** 2)
    exp = zone(exp_path, alpha)
    exp_up, exp_lo = experiment_sides(exp)

    def rms(sim, ref):
        xs, cs = [x for x, _ in sim], [c for _, c in sim]
        d = [interp(xs, cs, x) - c for x, c in ref if 0.0 < x < 1.0]
        return math.sqrt(sum(v * v for v in d) / len(d))

    xs_u, cp_u = [x for x, _ in upper], [c for _, c in upper]
    print(f"Cp RMS difference: upper {rms(upper, exp_up):.4f}, lower {rms(lower, exp_lo):.4f}; "
          f"shock x/c {shock(xs_u, cp_u):.3f} (experiment {shock(*zip(*exp_up)):.3f})")
    if len(sys.argv) > 7:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        plt.figure(figsize=(7, 5))
        plt.plot(*zip(*upper), "b-", lw=1, label="nssolver upper")
        plt.plot(*zip(*lower), "g-", lw=1, label="nssolver lower")
        plt.plot(*zip(*exp), "ko", ms=3, label="experiment")
        plt.gca().invert_yaxis()
        plt.xlabel("x/c")
        plt.ylabel("Cp")
        plt.title(f"M {mach}, alpha {alpha}")
        plt.legend()
        plt.savefig(sys.argv[7], dpi=110)


if __name__ == "__main__":
    main()
