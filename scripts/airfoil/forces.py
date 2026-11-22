"""Lift and drag of a unit-chord airfoil from a dual-time run's history
(output_file's .history.csv): their mean and amplitude over the last
PERIODS periods of the lift, and the period's Strouhal number f c / U.

    python3 forces.py HISTORY.csv MACH RE T ALPHA_DEG SPAN [PERIODS]
"""
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from freestream import freestream  # noqa: E402


def main():
    path = sys.argv[1]
    mach, re, t, alpha_deg, span = map(float, sys.argv[2:7])
    periods = int(sys.argv[7]) if len(sys.argv) > 7 else 4
    fs = freestream(mach, re, t, alpha_deg)
    q = 0.5 * fs["rho"] * fs["speed"] ** 2 * span
    a = math.radians(alpha_deg)
    rows = list(csv.DictReader(open(path)))
    time = [float(r["time"]) for r in rows]
    fx = [float(r["fx"]) for r in rows]
    fy = [float(r["fy"]) for r in rows]
    cl = [(-x * math.sin(a) + y * math.cos(a)) / q for x, y in zip(fx, fy)]
    cd = [(x * math.cos(a) + y * math.sin(a)) / q for x, y in zip(fx, fy)]
    unconverged = sum(1 for r in rows if float(r["inner_reduction"]) > 1.5e-3)
    # Upward crossings of the lift's running mean (over the last half).
    half = len(cl) // 2
    mean = sum(cl[half:]) / max(len(cl) - half, 1)
    ups = [time[i - 1] + (time[i] - time[i - 1]) * (mean - cl[i - 1]) / (cl[i] - cl[i - 1])
           for i in range(1, len(cl)) if cl[i - 1] < mean <= cl[i]]
    head = f"{os.path.basename(path)}: {len(rows)} steps to t = {time[-1]:.4g}, {unconverged} short of 1e-3 inside"
    if len(ups) < periods + 1:
        print(f"{head}; last Cl {cl[-1]:.4f} Cd {cd[-1]:.5f}, {len(ups)} lift crossings (no steady oscillation yet)")
        return
    t0, t1 = ups[-periods - 1], ups[-1]
    w = [i for i in range(len(time)) if t0 <= time[i] <= t1]
    period = (t1 - t0) / periods
    st = 1.0 / (period * fs["speed"])
    print(f"{head}\n  St = f c/U {st:.4f} (period {period:.4g} s), Cl {sum(cl[i] for i in w) / len(w):.4f} "
          f"+- {0.5 * (max(cl[i] for i in w) - min(cl[i] for i in w)):.4f}, "
          f"Cd {sum(cd[i] for i in w) / len(w):.5f} +- {0.5 * (max(cd[i] for i in w) - min(cd[i] for i in w)):.5f}")


if __name__ == "__main__":
    main()
